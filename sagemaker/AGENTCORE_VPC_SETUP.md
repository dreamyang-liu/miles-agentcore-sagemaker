# Prompt: connect a Bedrock AgentCore Runtime directly to a Miles session server inside a VPC

You are setting up networking so that an **Amazon Bedrock AgentCore Runtime** (the external
agent) can open TCP connections **directly** to the Miles session servers running inside a
**SageMaker Training Job** — no proxy, no NLB, no public IP. Miles' session server has no
authentication, so the security group is the entire boundary. Everything below was verified
end to end in account `241580540779`, region `us-west-2` (2026-09-04); IDs are given as the
concrete reference, adapt names if you rebuild elsewhere.

## Facts you must design around

1. **AgentCore VPC mode only lands in specific Availability Zone IDs.** In `us-west-2` those
   are `usw2-az1`, `usw2-az2`, `usw2-az3` (**not** `usw2-az4`). The list is not in the docs;
   `create-agent-runtime` fails with `CREATE_FAILED` and a `failureReason` naming the
   unsupported subnet. Check with
   `aws ec2 describe-subnets --query 'Subnets[].[SubnetId,AvailabilityZoneId]'` — AZ *names*
   map differently per account, AZ *IDs* do not.
2. **The AgentCore ENI only gets a private IP**, so it needs a NAT gateway (or interface
   endpoints for `ecr.api`, `ecr.dkr`, `logs`, plus an S3 gateway endpoint) to pull the
   agent image and write logs. A NAT is the simplest.
3. **A SageMaker training container has exactly one interface, `eth0`, with the VPC ENI's
   private IP**, and any port it listens on is reachable from other ENIs in the VPC when the
   security group allows it. The container does *not* need to be in an AgentCore-supported
   AZ — cross-AZ traffic inside the VPC is fine (verified az3 → az4).
4. Miles spawns **all** session servers on the Ray **head** (`hosts[0]` / `algo-1`), binding
   consecutive ports from `--session-server-port` (recipe: 30000, up to 32 workers →
   30000–30031). Only the head's IP:ports need to be reachable.
5. Quotas that bit: `VPCs per Region` (5) and `EC2-VPC Elastic IPs` (5, one per NAT). Check
   both before creating anything.

## Network to build

| Piece | Reference value | Notes |
| --- | --- | --- |
| VPC | `vpc-09bdc6b42c4fe9e90`, `10.20.0.0/16` | DNS support + DNS hostnames enabled |
| Public subnet | `subnet-0bb20d330d70ed09c`, `10.20.0.0/24`, us-west-2a | holds the NAT; route `0.0.0.0/0 → IGW` |
| Private subnets | `subnet-000c9cd0883e4e612` `10.20.1.0/24` (usw2-az2) · `subnet-05c3ab0fc851b93fd` `10.20.2.0/24` (usw2-az1) · `subnet-0df042dc329c771f2` `10.20.3.0/24` (usw2-az3) · `subnet-05f87bf0d5d99e76f` `10.20.4.0/24` (usw2-az4) | route `0.0.0.0/0 → NAT`; one per AZ so both services have room |
| NAT gateway | `nat-00a2636f2335074a2` | in the public subnet, one EIP |
| S3 gateway endpoint | on both route tables | SageMaker channels / checkpoints |
| SG `miles-agentcore-train` | `sg-011aa4180a8c9c7db` | attached to the **SageMaker job** (`VpcConfig.SecurityGroupIds`). Ingress: **all traffic from itself** (Ray 6379/8265/10001+, NCCL, torch rendezvous between hosts) and **tcp 30000–30031 from `sg-056dfc0ebe7743324`**. Egress: all. |
| SG `miles-agentcore-agentcore` | `sg-056dfc0ebe7743324` | attached to the **AgentCore runtime**. No ingress. Egress: all. |

The single deliberate exposure is the rule `train ← agentcore : tcp 30000-30031`. Nothing else
in the VPC (and nothing outside it) can reach the session servers.

## AgentCore runtime (VPC mode)

* Execution role needs, besides ECR read + CloudWatch logs, an inline policy allowing
  `ec2:CreateNetworkInterface`, `ec2:CreateNetworkInterfacePermission`,
  `ec2:DeleteNetworkInterface`, `ec2:DescribeNetworkInterfaces`, `ec2:DescribeSubnets`,
  `ec2:DescribeSecurityGroups`, `ec2:DescribeVpcs`, `ec2:DescribeDhcpOptions`,
  `ec2:DescribeRouteTables` (reference role: `MilesAgentCoreExecRole`).
* Create with

  ```json
  "networkConfiguration": {
    "networkMode": "VPC",
    "networkModeConfig": {
      "subnets": ["subnet-000c9cd0883e4e612", "subnet-05c3ab0fc851b93fd", "subnet-0df042dc329c771f2"],
      "securityGroups": ["sg-056dfc0ebe7743324"]
    }
  }
  ```

  — only the three supported-AZ subnets; including the az4 subnet fails the whole create.
* Reference runtime: `miles_math_agent_vpc-pZqJtz42PV`
  (`arn:aws:bedrock-agentcore:us-west-2:241580540779:runtime/miles_math_agent_vpc-pZqJtz42PV`),
  image `241580540779.dkr.ecr.us-west-2.amazonaws.com/miles-agentcore-math:latest`
  (linux/arm64 — required by the default microVM compute type).
* The invocation payload carries the session URL directly:
  `"base_url": "http://<head-vpc-ip>:<port>/sessions/<sid>/v1"`. The `token` field is a
  placeholder (the OpenAI client needs a non-empty api_key; the session server ignores it).

## SageMaker training job side

* `VpcConfig = {Subnets: <the four private subnets>, SecurityGroupIds: ["sg-011aa4180a8c9c7db"]}`.
* The execution role needs `bedrock-agentcore:InvokeAgentRuntime` and
  `bedrock-agentcore:StopRuntimeSession` on the runtime ARN (the trainer calls the agent).
* Multi-instance: read `/opt/ml/input/config/resourceconfig.json` (`current_host`, `hosts`,
  `network_interface_name`). `hosts[0]` starts `ray start --head --node-ip-address <its eth0 IP>`,
  the others `ray start --address=<hosts[0]>:6379`; `algo-N` hostnames resolve via DNS. Set
  `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` to `network_interface_name`. Miles' own IP
  detection (`get_host_info`) already picks the `eth0` VPC IP, so `--session-server-ip` needs
  no override.
* The Miles-side agent function must hand AgentCore the session URL unchanged (direct mode:
  leave `MILES_PROXY_BASE` unset).

## How to verify (cheap, no GPU)

1. Run a 2-host CPU training job (e.g. `ml.m6i.large ×2`) in the VPC whose `algo-1` serves any
   HTTP server on `0.0.0.0:30000` and logs client addresses; have `algo-2` probe
   `http://algo-1:30000/health`.
2. Invoke the VPC-mode runtime with `base_url = http://<algo-1 eth0 IP>:30000/sessions/<sid>/v1`.
3. Pass = the agent returns a reply **and** `algo-1`'s log shows the request from a `10.20.x.x`
   source (the AgentCore ENI). Verified result: `10.20.3.245 → 10.20.4.84:30000`, 200 OK, 0.9 s.

Reference implementation: `examples/experimental/agentcore/sagemaker/` in the Miles repo
(`infra.py` builds the network idempotently, `agentcore_runtime.py` creates the runtime and
invokes it, `smoke/` + `launch_smoke.py` is the verification above, `entrypoint.py` +
`launch_train.py` run the real recipe).
