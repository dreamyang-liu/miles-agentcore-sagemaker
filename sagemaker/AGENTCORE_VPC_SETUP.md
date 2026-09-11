# AgentCore VPC connection

The training caller invokes the **AWS AgentCore API**. The running agent then
makes a separate **HTTP callback into the training VPC**. These require different
permissions and network paths.

This guide covers both an RFT SDK agent and the native example in this repository.
See [validation](../docs/VALIDATION.md) for the distinction between historical
private-network tests and the recent local-host HTTPS GPU run.

## Traffic and ports

| From | To | Port | Purpose |
| --- | --- | --- | --- |
| Training caller | AWS AgentCore API | HTTPS 443 | `InvokeAgentRuntime`, then `StopRuntimeSession` when enabled |
| RFT agent's VPC ENI | Training head | TCP 30100 | Model requests and RFT SDK feedback |
| Native agent's VPC ENI | Training head | TCP 30000–30031 | Per-trajectory session endpoints |
| RFT front door | Session servers on the head | Local TCP 30000–30031 | Routing and token recording |
| Training hosts | Other hosts of the same job | Ray/NCCL/rendezvous ports | Distributed training and inference |

The port range assumes base port 30000 and 32 session-server workers. Adapt it if
those settings change. The front door is required for an unmodified RFT agent's
fixed-endpoint contract; `proxy.py` is not used.

The native callback is:

```text
http://<head-private-ip>:<session-port>/sessions/<session-id>/v1
```

The RFT agent's fixed endpoint is:

```text
http://<head-private-ip>:30100
# or http://miles-head.miles.internal:30100
```

The agent appends `/v1/chat/completions`. The front door maps its trajectory header
to the correct session URL. Do not expose the SGLang router/control API to the agent.

## VPC and security groups

Use the same VPC, or networks with explicit routes between the agent ENIs and the
training head. Put AgentCore in private subnets supported by that service in your
region. A NAT gateway is the reference setup's outbound path for agent dependencies
and AWS/external API calls; an endpoint-only design must provide the endpoints its
actual workload uses. The inference callback itself stays on the private route.

Use separate security groups:

- **Training SG:** self-ingress for communication among training hosts using that SG; TCP 30100
  from the AgentCore SG for RFT. Add TCP 30000–30031 from that SG when using native mode.
- **AgentCore SG:** outbound access to the training head and the agent's other required
  destinations. No inbound listener is needed for this callback pattern.

`infra.py` creates rules for **both** agent modes, plus NAT, an S3 gateway endpoint,
AgentCore execution role and a private hosted zone. It is a reference builder for
`us-west-2`: subnet AZ names/CIDRs are constants near its top. Adapt those constants
before using another region or an existing network.

The runtime helper has an **observed** `us-west-2` AZ-ID list, learned from an earlier
create failure. Treat that as a helper default, not a permanent AWS guarantee.
`MILES_AGENTCORE_AZ_IDS` overrides it; the helper can also retry after a service
failure identifies unsupported subnets. AZ names and AZ IDs are different fields.

## Create the reference infrastructure

These commands create AWS resources. Run from the repository root with credentials
for the target account, after building the images in the [container guide](../docs/CONTAINERS.md).

```bash
export AWS_REGION=us-west-2
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export ECR="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
export MILES_SM_ROLE_ARN='arn:aws:iam::<account>:role/<sagemaker-execution-role>'
export MILES_SM_ROLE_NAME="${MILES_SM_ROLE_ARN##*/}"
export MILES_SM_BUCKET='<model-data-checkpoint-bucket>'

python -m pip install boto3 httpx
python sagemaker/infra.py
```

The role named by `MILES_SM_ROLE_ARN` must already trust `sagemaker.amazonaws.com`
and have the S3/ECR/logging permissions needed by your training job. Setting
`MILES_SM_ROLE_NAME` lets the infrastructure helper grant that role permission to
update the private head DNS record.

State is saved in **account-specific** files:

```text
sagemaker/.infra.<account>.json
sagemaker/.agentcore_runtime.<account>.json
```

The runtime file is written by `agentcore_runtime.py create`. These files contain
resource IDs for your deployment and are git-ignored. Helpers use the active AWS
credentials to select the account. Choose the agent branch below; creating another
runtime overwrites the account's default runtime state file, so pass an explicit
ARN to RFT launch/invoke commands when selecting an existing runtime.

## RFT SDK runtime

Use the image built from **your RFT agent's repository**, not `agent/Dockerfile`.
The chosen image must support the fixed model endpoint and feedback contract
shown in the [main README](../README.md#rft-compatibility-boundary).

```bash
export RFT_AGENT_IMAGE='<registry>/<rft-agent-repository>:<version>'
export RFT_FRONT_DOOR_URL='http://miles-head.miles.internal:30100'

python sagemaker/agentcore_runtime.py create \
  --name miles_rft_agent_vpc \
  --image-uri "$RFT_AGENT_IMAGE" \
  --env "RFT_RUNTIME_ENDPOINT=$RFT_FRONT_DOOR_URL" \
  --env "RFT_RUNTIME_URL=$RFT_FRONT_DOOR_URL"
```

Wait for `READY`. `create` updates a runtime if that name already exists; its
network configuration comes from the current account's infrastructure file.
It is not an arbitrary-runtime environment-only patcher.

For a runtime created elsewhere, inspect its network, image and environment first:

```bash
aws bedrock-agentcore-control get-agent-runtime \
  --agent-runtime-id '<runtime-id>' \
  --query '{Status:status,Network:networkConfiguration,Image:agentRuntimeArtifact,Environment:environmentVariables}'
```

The RFT model client reads `RFT_RUNTIME_ENDPOINT` (or the legacy `RFT_RUNTIME_URL`).
The invocation's `metadata.endpoint` is the SDK feedback address. Updating only
that payload field does not redirect the model client.

## Native runtime

Build the native ARM64 image from this repo, then:

```bash
python sagemaker/agentcore_runtime.py create \
  --name miles_math_agent_vpc --image-uri "$NATIVE_AGENT_IMAGE"
```

This agent accepts the session URL in each invocation. No fixed front door or DNS
record is needed for its model callback. Leave `MILES_PROXY_BASE` unset for VPC direct mode.

## IAM boundaries

The AgentCore execution role trusts `bedrock-agentcore.amazonaws.com`. The reference
helper creates a role with ECR/logging access and applies the VPC ENI policy.
An existing role must already have the appropriate trust and base permissions.
The ENI actions include:

```text
ec2:CreateNetworkInterface
ec2:CreateNetworkInterfacePermission
ec2:DeleteNetworkInterface
ec2:DescribeNetworkInterfaces
ec2:DescribeSubnets
ec2:DescribeSecurityGroups
ec2:DescribeVpcs
ec2:DescribeDhcpOptions
ec2:DescribeRouteTables
```

Add the permissions needed by your agent's own tools and SDK. The network helper
cannot infer those from an arbitrary externally supplied RFT image.

The **training execution role** calls AgentCore. An example inline statement is:

```json
{
  "Effect": "Allow",
  "Action": ["bedrock-agentcore:InvokeAgentRuntime", "bedrock-agentcore:StopRuntimeSession"],
  "Resource": "arn:aws:bedrock-agentcore:<region>:<account>:runtime/<runtime-id>"
}
```

The training role also needs access to its model/data/checkpoint S3 prefixes and,
for the automated RFT DNS path, Route 53 record updates in the selected hosted zone.
A SageMaker job must allow the outbound networking used by its agent calls;
do not enable network isolation for this architecture.

## Private IP or private DNS

**A hosted zone is not a networking requirement.** A fixed EC2 training head can
serve the RFT endpoint directly on its private IP:

```bash
# On the training head, with the VPC security groups configured:
python -m pip install fastapi uvicorn httpx
python rft_front_door.py --host 0.0.0.0 --port 30100
# The external RFT agent's model endpoint is http://<head-private-ip>:30100.
# The local Miles caller uses:
export MILES_RFT_FRONT_DOOR_LOCAL=http://127.0.0.1:30100
```

A SageMaker job gets its head IP after the containers start. The supplied
`entrypoint.py` reads `resourceconfig.json`, takes `hosts[0]` as the Ray head, finds
its address through `network_interface_name`, and updates a Route 53 A record with
TTL 30. The AgentCore runtime can keep the same fixed model endpoint across jobs.
The entrypoint logs the update status; it does not wait for Route 53 `INSYNC`.
Verify the name and callback path before treating the deployment as ready.

The supplied **SageMaker RFT entrypoint currently requires** `MILES_HEAD_DNS` and
`MILES_ROUTE53_ZONE_ID`. To use only a private IP with a changing SageMaker head,
an orchestration step must discover that IP, update the agent's **model** endpoint,
wait for the runtime to be ready and verify a new invocation before training.
That automatic IP-registration path is not implemented here.

One fixed runtime endpoint/head DNS record belongs to one active training head.
Concurrent jobs need independent endpoints/names; otherwise an update can redirect
one job's agents to another job's session registry.

## Verify the private callback before GPU training

### RFT

The smoke job starts a fake session server using `finish`, starts the front door,
updates the head DNS record and pre-registers trajectory `smoke-traj`. This lets
an API caller outside the VPC invoke the runtime without accessing local control endpoints.

```bash
export RFT_RUNTIME_ARN='arn:aws:bedrock-agentcore:<region>:<account>:runtime/<runtime-id>'
JOB=$(python sagemaker/launch_smoke.py start --rft --duration 1200)
python sagemaker/launch_smoke.py watch "$JOB"
python sagemaker/agentcore_runtime.py invoke-rft \
  --runtime-arn "$RFT_RUNTIME_ARN" \
  --front-door-url "$RFT_FRONT_DOOR_URL" --trajectory-id smoke-traj
python sagemaker/launch_smoke.py logs "$JOB"
python sagemaker/launch_smoke.py stop "$JOB"
```

Pass requires the correct answer **and** logs showing successful inference from
an AgentCore private source address, plus `/complete-rollout` and `/update-reward`
with HTTP 200. `HEAD_READY` or `/health` alone is not that proof. Create a fresh
smoke session/job when repeating a full trajectory test.

### Native

```bash
JOB=$(python sagemaker/launch_smoke.py start --duration 1200)
python sagemaker/launch_smoke.py watch "$JOB"
# Use the private address printed in HEAD_READY:
python sagemaker/agentcore_runtime.py invoke --head-ip '<head-private-ip>'
python sagemaker/launch_smoke.py logs "$JOB"
python sagemaker/launch_smoke.py stop "$JOB"
```

After connectivity passes, use the [training walkthrough](../docs/TRAINING.md).
A network mock does not validate model tokenization, gradients or checkpoint output.

## AWS references

- [Runtime VPC configuration and supported Availability Zones](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-vpc.html)
- [Runtime HTTP container contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html)
- [Runtime IAM permissions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html)
