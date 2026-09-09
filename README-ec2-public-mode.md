# Miles × Amazon Bedrock AgentCore

Train an agent hosted on [Bedrock AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)
against a policy served from inside the Miles training cluster.

This follows the same shape as the [Harbor recipe](../../swe-agent-harbor-docker): Miles
hands each trajectory a private, session-scoped OpenAI-compatible URL, and the external
agent calls back into the cluster. The session server records exact token ids and logprobs
on the way through, so TITO holds end to end and nothing has to be retokenized.

The one thing Harbor gets for free and AgentCore does not is the network path. Harbor runs
on a machine you own behind Tailscale; AgentCore is AWS-managed. **Miles' session server has
no authentication and a catch-all route straight to the SGLang control plane, so it must
never face AgentCore directly.** That is what `proxy.py` is for.

```
Miles agent function ──── InvokeAgentRuntime (async) ────▶ AgentCore Runtime
       │                                                        │
       │  ③ collect_samples (in-cluster, never via proxy)        │ ② every model call
       ▼                                                        ▼
  session server :30000-30031 ◀──── proxy (verify · whitelist · forward) ◀┘
```

## Files

| File | Purpose |
| --- | --- |
| `proxy.py` | The only component facing AgentCore: HMAC token verification, path whitelist, deterministic forwarding. Stateless. |
| `agent/agent.py` | AgentCore Runtime agent — `/invocations` + `/ping`, multi-turn tool-calling loop. |
| `agent/Dockerfile` | ARM64 image, required by the default microVM compute type. |
| `fake_session_server.py` | Stand-in for the session server so the path can be wired with no GPUs. Deliberately reproduces the real server's unsafe routes. |
| `smoke_test.py` | End-to-end check of the proxy leg. No AWS account needed. |

## Why the proxy holds no state

The Miles-side agent function mints a short-lived HMAC token that *carries the upstream
address*:

```python
token = sign({"sid": sid, "ip": "10.0.3.17", "port": 30007, "exp": now + trial_timeout}, SECRET)
```

The proxy verifies the signature, reads `ip`/`port` out of the token, and forwards. So:

- no Redis, no DynamoDB, no session registry — replicate the proxy freely
- the internal IP and port never appear in the URL the agent sees
- a Ray pod can be rescheduled mid-run; the next trajectory signs the new address
- a token is bound to one `sid`, so one trial's credential cannot touch another's session

Revocation is by expiry only. Set `exp` to the trial timeout. Add a denylist if you need
active revocation, but that puts state back.

## 1. Verify the proxy leg locally

No AWS account, no GPUs, ~10 seconds:

```bash
pip install fastapi uvicorn httpx openai
python smoke_test.py
```

This starts the fake session server and the proxy, drives the real agent loop from
`agent/agent.py` through the proxy, and checks 16 things — multi-turn tool calling, SSE
passthrough, five token-rejection cases, and that the whitelist blocks `POST /sessions`,
the samples op, and the catch-all routes that would otherwise reach SGLang.

`fake_session_server.py` logs `BREACH?` at ERROR if a dangerous route is ever served. The
smoke test fails if any such line appears.

## 2. Run the proxy on the EC2 side

```bash
export MILES_PROXY_SECRET=$(openssl rand -hex 32)     # share with the Miles side
python proxy.py serve --host 0.0.0.0 --port 8080

# separately, until the real session server is in play:
python fake_session_server.py --port 30007
```

`--upstream-timeout` must stay above the session server's `--miles-router-timeout`
(600s default), or a long completion gets cut off at the proxy instead of upstream.

Mint a token by hand to test from anywhere:

```bash
TOKEN=$(python proxy.py sign --sid $(openssl rand -hex 16) --ip 127.0.0.1 --port 30007 --ttl 7200)
```

## 3. Push the agent image (ARM64)

```bash
export AWS_REGION=us-west-2
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
REPO=miles-agentcore-agent

aws ecr create-repository --repository-name $REPO --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ECR

cd agent
docker buildx build --platform linux/arm64 -t $ECR/$REPO:latest --push .
```

ARM64 is not optional on the default microVM compute type. An x86 image fails at
invocation time, not at create time.

## 4. Create the runtime in PUBLIC mode

PUBLIC mode is the shortest path for a first connection: no subnets, no security groups,
no VPC endpoints, no AZ constraints. Read the security note below before leaving it here.

```bash
cat > trust.json <<'EOF'
{"Version": "2012-10-17",
 "Statement": [{"Effect": "Allow",
                "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                "Action": "sts:AssumeRole"}]}
EOF

aws iam create-role --role-name MilesAgentCoreExecRole \
  --assume-role-policy-document file://trust.json
aws iam attach-role-policy --role-name MilesAgentCoreExecRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam attach-role-policy --role-name MilesAgentCoreExecRole \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

aws bedrock-agentcore-control create-agent-runtime \
  --region $AWS_REGION \
  --agent-runtime-name miles_rollout_agent \
  --role-arn arn:aws:iam::$ACCOUNT:role/MilesAgentCoreExecRole \
  --network-configuration '{"networkMode": "PUBLIC"}' \
  --agent-runtime-artifact "{\"containerConfiguration\": {\"containerUri\": \"$ECR/$REPO:latest\"}}"
```

## 5. Invoke it end to end

```bash
SID=$(openssl rand -hex 16)
TOKEN=$(python proxy.py sign --sid $SID --ip <session-server-private-ip> --port 30007 --ttl 7200)
BASE_URL=https://<your-proxy-host>/s/$SID/v1

aws bedrock-agentcore invoke-agent-runtime \
  --region $AWS_REGION \
  --agent-runtime-arn <arn-from-step-4> \
  --runtime-session-id "miles-$(openssl rand -hex 16)" \
  --payload "$(jq -nc --arg u "$BASE_URL" --arg t "$TOKEN" \
      '{base_url: $u, token: $t,
        prompt: [{role: "user", content: "list the workspace then summarise it"}],
        sampling_params: {max_tokens: 512}, instance_id: "smoke-1"}')" \
  out.json && cat out.json
```

`--runtime-session-id` must be **at least 33 characters**. Miles session ids are
`uuid4().hex`, which is 32 — hence the `miles-` prefix everywhere in this example.

Success looks like `{"reward": 1.0, "exit_status": "completed", "agent_metrics": {...}}`,
with three request lines in the proxy log and no `BREACH?` line in the session-server log.

## Security note: PUBLIC mode is for bring-up only

AgentCore's PUBLIC-mode egress IPs are not published in `ip-ranges.json` and there is no
documented allowlist mechanism, so reaching the proxy this way means opening its port to
`0.0.0.0/0`. Two consequences:

- **Terminate TLS.** Over plain HTTP the token and the whole conversation cross the public
  internet in the clear. Put the proxy behind a TLS terminator (Caddy or nginx + certbot)
  before it carries anything real.
- **Move to VPC mode for anything beyond a spike.** Set `networkMode: VPC` with your
  subnets and security groups, put the proxy behind an internal NLB, and allow
  security-group to security-group. Then no port faces the internet at all, and a publicly
  trusted certificate is no longer required because the hop is ordinary VPC traffic.

VPC mode has three gotchas worth knowing before you switch: it only works in specific
Availability Zone **IDs** (`us-east-1`: `use1-az1` / `use1-az2` / `use1-az4` — check with
`describe-subnets --query 'Subnets[].AvailabilityZoneId'`, since AZ *names* map differently
per account); it needs interface endpoints for `ecr.dkr`, `ecr.api` and `logs` plus an S3
gateway endpoint, or the container cannot be pulled; and public subnets do not help, since
the ENI only receives a private IP.

## 6. Connect the real Miles side

Still to do — replace `fake_session_server.py` with the real thing and write the Miles-side
agent function, modelled on
[`swe_agent_function.py`](../../swe-agent-harbor-docker/swe_agent_function.py):

1. Sign a token from `metadata["session_server_id"]` (the `ip:port` that owns this session)
   plus `tracer.session_id`.
2. Rewrite `base_url` to the proxy, appending `/v1`.
3. Replace Harbor's `POST /run` with `InvokeAgentRuntime`. Because a synchronous invocation
   is capped at **15 minutes** while agentic trials routinely run past an hour, use the
   async job path (8h ceiling) and poll.
4. Return the reply dict — `{reward, exit_status, eval_report, agent_metrics}` — unchanged;
   `agentic_tool_call.generate` merges it into every sample's metadata, and
   `--custom-rm-path` turns `metadata["reward"]` into `Sample.reward`.
5. Implement `abort(args)` to call `StopRuntimeSession`, the analogue of Harbor's `/flush`,
   so oversampling abort releases in-flight trials instead of letting them run to timeout.

Launch flags follow the Harbor recipe:

```
--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
--custom-agent-function-path    agentcore_agent_function.run
--use-session-server
--tito-model                    <your model family>
--session-server-port           30000
--session-server-workers        32
```
