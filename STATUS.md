# Status

What in this repo is verified on real hardware, what is not, and what it cost. Last updated
2026-09-10. Account identifiers are written as `<ACCOUNT>`; substitute your own.

## Verified end to end

| Capability | Evidence |
| --- | --- |
| AgentCore Runtime (VPC mode) reaches a SageMaker training container **directly** over the VPC — no proxy, no NLB, no public endpoint | 2-host CPU smoke: agent returned a reply and the head's access log showed `POST /sessions/<sid>/v1/chat/completions 200` from the runtime's ENI, cross-AZ (az3 → az4), 0.9 s |
| Multi-instance Ray bring-up inside one training job | 2 hosts, `RAY_NODES 2/2 joined` in 13 s; workers resolve `algo-1` by name and by VPC IP; SGLang engines placed on both hosts |
| Full training loop, single host, our own agent | Qwen3-0.6B on 1×p5.48xlarge, `Completed`: 8 session servers on :30000–30007, weight sync 2.6 s, 16 AgentCore-driven trajectories over 2 GRPO rollouts, `rollout/raw_reward` logged, callers seen as VPC ENIs of the runtime |
| Megatron-Bridge + LoRA on a 27B dense hybrid | Qwen3.6-27B, rank 32 / alpha 64, 1×p5.48xlarge (TP4 trainer + 2 TP4 engines colocated): bridge loaded the HF checkpoint, first weight sync 2.6 s, trainer ~22 GB/GPU after sync, 32 rollouts trained, adapter checkpoints (389 MB) written to S3 every 10 steps |
| An **unmodified** SageMaker-RFT agent (Strands + `sagemaker.train.rft`) driving a Miles policy | CPU smoke: the agent resolved the private head name from its own VPC ENI, streamed 3 turns through `rft_front_door.py` into one session (200 each), then posted `/complete-rollout` and `/update-reward`; `status=success` in 1.6 s |

## Not verified yet

- **A full training run with an RFT-contract agent.** The network path is proven and the two bugs
  that blocked it are fixed (below), but no GPU run has completed with `--agent-mode rft`.
- **A completed 50-step 27B run.** The run above was stopped at step 31 by the
  `reasoning_content` bug; it has not been repeated since the fix.
- **Multi-host training end to end.** Ray bring-up and engine placement across 2 hosts are proven;
  every completed training run so far used one host.

## Bugs found the expensive way, and their fixes

1. **A thinking model needs `reasoning_content` echoed back.** The session server compares the
   history an agent replays against what it stored, over `role`, `content`, `reasoning_content`
   and `tool_calls`. Qwen3.6's template preserves the chain of thought, so an agent that rebuilds
   assistant messages from `content` + `tool_calls` alone mismatches on *every* turn: the session
   rolls back to the empty checkpoint each time, history grows without bound and each turn's
   recorded tokens are discarded. Measured: 5312 rollbacks, **100 % to `checkpoint -1`**, sessions
   at 1309 messages, 80k-token prefills, trials 5 s → 311 s, run stalled — against 17 benign
   rollbacks and 7 messages max on the same recipe with a non-thinking model.
   *Fix:* `agent/agent.py` carries the field. For a third-party agent that cannot (Strands drops
   it too), `--session-message-matcher session_message_matcher.matches` is `loose_tool_call`
   minus `reasoning_content` — still compares content and the whole tool-call structure.
2. **Strands sends `stream_options` the engine rejects.** Strands hardcodes
   `stream_options={"include_usage": true}` while the session server pops `stream` to drive the
   engine itself; SGLang then rejects the orphaned option and every trajectory dies with a 503.
   *Fix:* the front door strips it. Symptom to look for: `forward tid=… 503` with zero successful
   trials.
3. **`AccessDenied` from `InvokeAgentRuntime` used to live-lock the job.** It was treated as a lost
   trial, so Miles resampled forever — two hours of p5 time over one policy ARN that did not cover
   the runtime. *Fix:* IAM/config errors now raise and kill the job at the first rollout.
4. **SageMaker's container runtime denies `pidfd_getfd`**, which torch needs to share
   expandable-segment CUDA allocations with colocated engines during weight sync. Run the trainer
   with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`; the entrypoint does.
5. **The `model` channel must be linked on every host**, because Ray places SGLang engines on
   workers too.

## Environment facts you cannot query

- **AgentCore VPC mode only lands in certain AZ *IDs*** — in `us-west-2`: `usw2-az1`, `usw2-az2`,
  `usw2-az3` (not `usw2-az4`). Not in the docs; `create-agent-runtime` fails and names the bad
  subnet. Offer it every private subnet and let it choose (`agentcore_runtime.py` does, and
  retries without the rejected ones).
- **A SageMaker training container has one interface, `eth0`**, carrying the VPC ENI's private IP,
  and ports it listens on are reachable from other ENIs in the VPC when the SG allows it.
  Cross-AZ within the VPC is fine, so the training hosts need not be in an AgentCore-supported AZ.
- **The AgentCore ENI is private-only**, so the runtime needs a NAT gateway (or interface
  endpoints for `ecr.api`, `ecr.dkr`, `logs`, plus an S3 gateway endpoint) to pull its image.
- Deleting a VPC-mode runtime leaves its `agentic_ai` ENIs attached for a while; they are
  AWS-managed (`delete-network-interface` is refused) and block subnet deletion until released.

## Measured cost and timing (us-west-2, on-demand)

| Thing | Number |
| --- | --- |
| Qwen3.6-27B, batch 32 (4 prompts × 8 samples), 8×H100 | **90–140 s per step** (median 134 s early, 100 s later as answers got shorter) |
| First step of any job | + ~14 min: image pull, 56 GB model channel, bridge load, 8 engines |
| A clean 50-step 27B run | ≈ 1 h 45 m, ≈ $180–200 |
| 2-rollout smoke, 27B | 1143 s billed |
| 2-rollout smoke, 0.6B | 495 s billed |
| CPU network smoke (2× m6i.large) | cents |
| p5.48xlarge ×1 queue wait, observed | 9 min / 20 min / 27 min / 64 min+ — highly variable |
| g6e.2xlarge ×2 queue wait | 6.5 h. `g6e.12xlarge ×2`: never scheduled in 48 h |

Rewards on gsm-hard for the 27B moved 0.5 → ~1.0 over 31 steps with mean response length
dropping 780 → 110 tokens, so the dataset is not saturated for a 27B at rank 32 — but pick
something harder for a real run.

## Layout

| Path | Role |
| --- | --- |
| `sagemaker/infra.py` | VPC, NAT, subnets, security groups, AgentCore role, Route 53 private zone — idempotent, one state file per account |
| `sagemaker/agentcore_runtime.py` | create a VPC-mode runtime (`--name`, `--env`); `invoke` / `invoke-rft` to exercise the path before spending GPU |
| `sagemaker/smoke/` + `launch_smoke.py` | 2-host CPU job serving a fake session server (and optionally the front door) — proves reachability for cents |
| `sagemaker/Dockerfile.train`, `entrypoint.py` | Miles image + this directory; Ray head/worker from `resourceconfig.json`, head DNS upsert, front door, launcher submit |
| `sagemaker/launch_train.py` | `CreateTrainingJob` with VpcConfig, channels, checkpoints; `watch` / `logs` / `stop` |
| `sagemaker/convert_rft_parquet.py` | SageMaker-RFT prompts parquet → the jsonl the launcher reads |
| `rft_front_door.py` | in-VPC shim that makes Miles look like the RFT runtime to an RFT agent |
| `rft_agent_function.py`, `agentcore_agent_function.py` | Miles-side agent functions: RFT contract, and our own agent |
| `session_message_matcher.py` | reasoning-tolerant session matcher for agents that cannot echo `reasoning_content` |
| `agent/` | our AgentCore agent (`/invocations`, `/ping`, calculator + submit_answer) |
| `proxy.py`, `README-ec2-public-mode.md`, `train.sh`, `run_in_docker.sh` | the earlier EC2 + PUBLIC-mode path, kept for reference |

## Security model

The session server authenticates nothing and has a catch-all route to the SGLang control plane.
In this setup the only thing that can reach it is an ENI in the `miles-agentcore-agentcore`
security group; that one rule is the whole boundary. Do not attach that group to anything else.
The front door forwards `/v1/chat/completions` only and 404s everything else.
