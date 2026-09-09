# Miles × Bedrock AgentCore on SageMaker Training Jobs — from scratch

Train a policy with [Miles](https://github.com/radixark/miles) inside a **multi-instance
SageMaker Training Job**, while the agent that drives each trajectory runs on **Bedrock
AgentCore Runtime** and calls the policy **directly over the VPC** — no proxy, no NLB, no
public endpoint.

```
SageMaker Training Job (VPC)                              Bedrock AgentCore Runtime (VPC mode)
┌───────────────────────────────────────────┐              ┌───────────────────────────────┐
│ algo-1 (Ray head)                          │  tcp 30000+  │ agent microVM                  │
│  ├ trainer (FSDP)     ├ SGLang engine      │◀─────────────│  OpenAI client → base_url      │
│  └ session servers :30000-30031  ◀─────────┼──────────────│  = http://<algo-1>:3000x/…     │
│ algo-2 … algo-N (Ray workers)              │              └───────────────────────────────┘
│  └ SGLang engines                          │                          ▲
└───────────────────────────────────────────┘                          │ InvokeAgentRuntime
                    ▲ trainer calls the agent once per trajectory ─────┘
```

Everything here was verified in `us-west-2` (see [`sagemaker/AGENTCORE_VPC_SETUP.md`](sagemaker/AGENTCORE_VPC_SETUP.md)
for the networking facts and the exact resource IDs of the reference setup).

## What you need

* An AWS account with quotas for: 1 more VPC, 1 more Elastic IP, and the SageMaker training
  instance type you want (`ml.g6e.*`, `ml.g5.12xlarge`, `ml.p5.48xlarge`, …). Check first:
  ```bash
  aws service-quotas get-service-quota --service-code vpc --quota-code L-F678F1CE   # VPCs per region
  aws service-quotas get-service-quota --service-code ec2 --quota-code L-0263D0A3   # Elastic IPs
  aws service-quotas list-service-quotas --service-code sagemaker \
    --query "Quotas[?contains(QuotaName,'for training job usage') && Value>\`0\`].[QuotaName,Value]" --output text
  ```
* A SageMaker execution role (trusting `sagemaker.amazonaws.com`, with S3 + ECR + CloudWatch
  access) and an S3 bucket.
* A build machine with Docker (buildx with arm64 emulation for the agent image), the AWS CLI,
  Python ≥ 3.9 and `boto3`. ~60 GB of disk for the Miles image.
* The Miles repository checked out, since this directory lives at
  `examples/experimental/agentcore/` inside it and the training image copies it in.

Export once:

```bash
export AWS_REGION=us-west-2
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export ECR=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
export MILES_SM_ROLE_ARN=arn:aws:iam::$ACCOUNT:role/service-role/<your-sagemaker-execution-role>
export MILES_SM_BUCKET=<your-bucket>
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR
cd examples/experimental/agentcore
```

## 1. Network: VPC, NAT, security groups, AgentCore role

```bash
python sagemaker/infra.py            # idempotent; writes sagemaker/.infra.json
```

Creates `10.20.0.0/16` with one public subnet (NAT gateway) and one private subnet per AZ, an
S3 gateway endpoint, two security groups, and the AgentCore execution role
(`MilesAgentCoreExecRole`) with the ENI permissions VPC mode needs. The only exposure it
opens is `miles-agentcore-train ← miles-agentcore-agentcore : tcp 30000-30031`.

Why every AZ: AgentCore VPC mode only lands in certain AZ *IDs* (`usw2-az1/az2/az3` in
us-west-2); SageMaker picks any AZ you offer. Both are given all the private subnets and each
takes what it supports.

## 2. Agent image → AgentCore runtime (VPC mode)

```bash
aws ecr create-repository --repository-name miles-agentcore-math >/dev/null 2>&1 || true
docker buildx build --platform linux/arm64 -t $ECR/miles-agentcore-math:latest --push agent/

python sagemaker/agentcore_runtime.py create --image-uri $ECR/miles-agentcore-math:latest
# -> READY, writes sagemaker/.agentcore_runtime.json
```

`arm64` is mandatory for the default microVM compute type (an x86 image fails at invoke
time, not at create time). If your region is not us-west-2 the first attempt may end
`CREATE_FAILED` naming unsupported zones; the script deletes and recreates without them.

## 3. Prove the network path (no GPU, ~5 min, cents)

```bash
aws ecr create-repository --repository-name miles-sagemaker-smoke >/dev/null 2>&1 || true
docker build -f sagemaker/smoke/Dockerfile -t $ECR/miles-sagemaker-smoke:latest . && docker push $ECR/miles-sagemaker-smoke:latest

JOB=$(python sagemaker/launch_smoke.py start)          # 2x ml.m6i.large in the VPC
python sagemaker/launch_smoke.py watch $JOB            # prints HOST_REPORT / HEAD_READY ip=<head-ip>
python sagemaker/agentcore_runtime.py invoke --head-ip <head-ip>
python sagemaker/launch_smoke.py logs $JOB | grep "POST /sessions"   # request from a 10.20.x.x AgentCore ENI
python sagemaker/launch_smoke.py stop $JOB
```

Pass = the invoke returns `{"submitted_answer": ..., "exit_status": "submitted", ...}` **and**
`algo-1`'s log shows the `POST /sessions/<sid>/v1/chat/completions` from a VPC address.

## 4. Training image

```bash
aws ecr create-repository --repository-name miles >/dev/null 2>&1 || true
docker pull radixark/miles:latest                       # ~43 GB
docker build -f sagemaker/Dockerfile.train -t $ECR/miles:sagemaker-agentcore . && docker push $ECR/miles:sagemaker-agentcore
```

`Dockerfile.train` layers only this directory on top of the public Miles image (which
already has Miles installed editable at `/root/miles`) and sets the entrypoint to
`sagemaker/entrypoint.py`. Pass `--build-arg BASE=<other image>` to pin a different base.

## 5. Model and data → S3 channels

```bash
hf download Qwen/Qwen3-0.6B --local-dir /tmp/Qwen3-0.6B
python prepare_data.py --dataset gsm-hard --output-dir /tmp/data       # needs `datasets`
aws s3 sync /tmp/Qwen3-0.6B s3://$MILES_SM_BUCKET/miles/models/Qwen3-0.6B/
aws s3 cp /tmp/data/gsm-hard_train.jsonl s3://$MILES_SM_BUCKET/miles/data/
aws s3 cp /tmp/data/gsm-hard_eval.jsonl  s3://$MILES_SM_BUCKET/miles/data/
```

## 6. Permissions the training job needs

The trainer invokes the agent, so the SageMaker execution role needs:

```bash
RUNTIME_ARN=$(python -c "import json;print(json.load(open('sagemaker/.agentcore_runtime.json'))['agentRuntimeArn'])")
aws iam put-role-policy --role-name <your-sagemaker-execution-role> --policy-name miles-agentcore-invoke \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"bedrock-agentcore:InvokeAgentRuntime\",\"bedrock-agentcore:StopRuntimeSession\"],\"Resource\":\"$RUNTIME_ARN\"}]}"
```

## 7. Launch the training job

```bash
JOB=$(python sagemaker/launch_train.py start --mode smoke --instance-type ml.g6e.2xlarge --count 2)
python sagemaker/launch_train.py watch $JOB
```

`--mode smoke` runs 2 rollouts (2 prompts × 4 samples); `--mode normal` is the full recipe.
`--count N` is the number of instances; `entrypoint.py` detects the GPUs per host itself.
Watch prints the marker lines: `HOST_REPORT`, `HEAD_READY`, `RAY_NODES k/N joined`,
`Session servers launched`, `trial done`, `rollout/raw_reward`, and any traceback.
Checkpoints stream to `s3://$MILES_SM_BUCKET/miles/checkpoints/<job>/`.

Capacity note: in our account `ml.g6e.2xlarge ×2` took 6.5 h to be scheduled and
`ml.g6e.12xlarge ×2` never was within 48 h. Submit, walk away, `watch` later.

## How the pieces fit

| File | Role |
| --- | --- |
| `sagemaker/infra.py` | VPC / NAT / subnets / SGs / AgentCore role, idempotent |
| `sagemaker/agentcore_runtime.py` | create the VPC-mode runtime; `invoke` for the direct-path check |
| `sagemaker/smoke/` + `launch_smoke.py` | 2-host CPU job serving `fake_session_server.py` — proves reachability |
| `sagemaker/Dockerfile.train` + `entrypoint.py` | Miles image + this dir; Ray head/worker from `resourceconfig.json`; runs the launcher with `MILES_SCRIPT_EXTERNAL_RAY=1` |
| `sagemaker/launch_train.py` | `CreateTrainingJob` with VpcConfig, channels, checkpoints; `watch`/`logs`/`stop` |
| `agentcore_agent_function.py` | Miles-side agent function: `InvokeAgentRuntime` per trajectory; direct URL when `MILES_PROXY_BASE` is unset |
| `run_qwen3_agentcore_math.py` | the Miles launcher (GRPO, FSDP, session server, TITO) |
| `agent/` | the AgentCore agent (`/invocations`, `/ping`, two tools) |
| `proxy.py`, `train.sh`, `run_in_docker.sh` | the earlier EC2 + PUBLIC-mode path, kept for reference |

Security model: the session server authenticates nothing and has a catch-all route to the
SGLang control plane. In this setup the **only** thing that can reach it is an ENI in the
`miles-agentcore-agentcore` security group. Do not attach that SG to anything else.
