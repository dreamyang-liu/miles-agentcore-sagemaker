# Build the containers

Run these commands from the **root of this standalone repository**. No AWS
resource is created by `docker build`; the ECR commands below create repositories
and upload the resulting images.

## Images and their roles

| Image | Dockerfile | Platform | Purpose |
| --- | --- | --- | --- |
| Native math agent | `agent/Dockerfile` | `linux/arm64` for the default AgentCore microVM runtime used here | `/invocations` and `/ping` on port 8080 |
| Miles training | `sagemaker/Dockerfile.train` | Match the training host; `linux/amd64` for the H100/x86 setup | Miles, this integration, SageMaker entrypoint |
| Network smoke | `sagemaker/smoke/Dockerfile` | `linux/amd64` for the reference CPU training job | Fake session server and optional RFT front door |
| Your RFT SDK agent | From your agent's source repository | Match the selected AgentCore compute type | The existing `sagemaker.train.rft` agent |

The native example is **not** the RFT/Strands agent used by `--agent-mode rft`.
That RFT image is supplied externally. To reproduce its image, use its own
Dockerfile and pinned dependencies; this repository supplies the Miles adapter
and documents its runtime contract.

## Registry variables

```bash
export AWS_REGION=us-west-2
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export ECR="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
export BUILD_TAG=agentcore-20260911
export NATIVE_AGENT_IMAGE="${ECR}/miles-agentcore-math:${BUILD_TAG}"
export MILES_TRAIN_IMAGE="${ECR}/miles:sagemaker-${BUILD_TAG}"
export MILES_SMOKE_IMAGE="${ECR}/miles-sagemaker-smoke:latest"

aws ecr get-login-password --region "$AWS_REGION" |
  docker login --username AWS --password-stdin "$ECR"
```

For each repository below, describe it first and create it only if it is absent.
An authorization error is not evidence that a repository is absent.

```bash
aws ecr describe-repositories --repository-names miles-agentcore-math
aws ecr describe-repositories --repository-names miles
aws ecr describe-repositories --repository-names miles-sagemaker-smoke

# Run only for a missing repository:
aws ecr create-repository --repository-name miles-agentcore-math
aws ecr create-repository --repository-name miles
aws ecr create-repository --repository-name miles-sagemaker-smoke
```

## Native AgentCore image

Use an ARM64 builder, or a buildx builder with ARM64 emulation already configured.
`docker buildx inspect` shows the available platforms.

```bash
docker buildx build --platform linux/arm64 \
  -f agent/Dockerfile -t "$NATIVE_AGENT_IMAGE" --push agent/
```

`agent/` is the build context: its Dockerfile copies `requirements.txt` and
`agent.py`. The container listens on `0.0.0.0:8080`.
The [AWS HTTP runtime contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html)
documents the platform, listener and required routes.

For a CPU contract check on an x86 development host, build an AMD64 test image.
This checks the application but does not substitute for the ARM64 runtime image:

```bash
docker build --platform linux/amd64 -t miles-native-agent:local agent/
docker run --rm -p 127.0.0.1:18080:8080 miles-native-agent:local
# In another terminal:
curl --fail http://127.0.0.1:18080/ping
```

## Miles training image

The base must already contain Miles at `/root/miles`, its model plugins and its
GPU dependencies. This Dockerfile layers the example on top; it does not compile
Miles, CUDA, SGLang or Megatron from source.

Set `MILES_BASE_IMAGE` to a compatible image **by digest**. The
[validation record](VALIDATION.md) lists the tested stack and base digest.
A different base, including a moving `radixark/miles:latest` tag, needs its own
compatibility check.

```bash
export MILES_BASE_IMAGE='<registry>/miles@sha256:<compatible-base-digest>'
docker pull "$MILES_BASE_IMAGE"
docker build --platform linux/amd64 \
  --build-arg BASE="$MILES_BASE_IMAGE" \
  -f sagemaker/Dockerfile.train -t "$MILES_TRAIN_IMAGE" .
docker push "$MILES_TRAIN_IMAGE"
```

For the default public Miles image, first pull it and record its resolved digest:

```bash
docker pull radixark/miles:latest
docker image inspect radixark/miles:latest --format '{{json .RepoDigests}}'
```

Use the resolved reference as `MILES_BASE_IMAGE`; this records what you built
against, but does not establish that it matches the previously validated stack.

SageMaker passes `train` to the container; the image entrypoint reads
`/opt/ml/input/config/resourceconfig.json` and starts the head/worker processes.
For a local CLI check, override that entrypoint:

```bash
docker run --rm --entrypoint python3 "$MILES_TRAIN_IMAGE" \
  /root/miles/examples/experimental/agentcore/run_qwen3_agentcore_math.py --help
```

The Docker context excludes local credentials, resource state, handover notes
and experiment reports. Model weights and training data are mounted or supplied
as SageMaker channels, not copied into this image.

## CPU network-smoke image

```bash
docker build --platform linux/amd64 \
  -f sagemaker/smoke/Dockerfile -t "$MILES_SMOKE_IMAGE" .
docker push "$MILES_SMOKE_IMAGE"
```

The smoke launcher currently uses
`$ECR/miles-sagemaker-smoke:latest`. Its RFT mode starts the front door on 30100
and a fake model that returns the RFT `finish` tool. It does not load a GPU model.

## Reuse an existing RFT agent image

Record the immutable image reference from your agent build. Its serving process
must implement AgentCore's runtime HTTP contract. The RFT agent used with this
integration additionally:

- Reads the JSON record in `payload.prompt` and sampling settings in `inferenceParams`.
- Reads `metadata.trajectory_id`, `metadata.endpoint`, `metadata.region` and `metadata.job_arn`.
- Sends its model requests with the RFT trajectory header.
- Returns `status` and `agent_answer`; it can also report SDK feedback.
- Configures its model client through `RFT_RUNTIME_ENDPOINT`/`RFT_RUNTIME_URL`.

Create or update a VPC-mode runtime using that image as described in the
[VPC guide](../sagemaker/AGENTCORE_VPC_SETUP.md). The registration script can update
a runtime with an existing name, so use a dedicated name when preparing a separate
training endpoint.
