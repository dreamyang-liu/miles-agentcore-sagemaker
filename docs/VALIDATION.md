# Validation record

Updated 2026-09-11. This file is the public validation summary. Local `STATUS.md`
handover notes, resource state, credentials, datasets and raw experiment dumps are
not published. The [earlier committed record](https://github.com/dreamyang-liu/miles-agentcore-sagemaker/blob/b0bcbabb9d97bf5535c49445443d4d2aa0e7205b/STATUS.md)
remains available as historical evidence.

## Verified behavior and its scope

| Scope | Result | What it does not establish |
| --- | --- | --- |
| Native AgentCore → SageMaker over private VPC | Historical two-host CPU network smoke and single-host native GPU training completed | Current RFT/27B private-VPC training |
| RFT SDK agent → private front door | Historical CPU callback test reached the head and reported SDK feedback | Model prompt fidelity or GPU optimizer correctness |
| RFT SDK agent → local training host over HTTPS | Qwen3-0.6B: 16/16 faithful questions, two optimizer updates, checkpoint, exit 0 | A private callback path |
| Qwen3.6-27B LoRA with RFT agent → local HTTPS head | Three updates, 768/768 faithful samples, finite token/log-prob fields, checkpoint, exit 0 | A 50-step run or exact optimizer restoration |
| BSHD 20,000-token budget | All samples preserved; maximum observed padded microbatch 19,968; independent per-microbatch padding | Arbitrary sequences up to the original 65,536 context limit |
| Saved-data replay of a prior long sample | Untrimmed 11,529-token trajectory completed training inside the budget | Live agent behavior on all long trajectories |

The current recent 27B test uses one x86 host with eight H100 80GB GPUs, a TP4/DP2
trainer, two TP4 SGLang engines and LoRA rank32/alpha64. Each update uses 32 prompts
with eight samples each, for global batch256. Forty-eight of 64 layers use block
activation recomputation. Reward remains Miles' existing answer-based reward;
SDK reward callbacks are diagnostic acknowledgements.

The earlier 50-step RFT attempt stopped after 17 updates with an OOM caused by
long-sequence padding under its then-fixed microbatch rule. The token-budget
extension subsequently passed saved-data replay and short live runs. It does not
establish that the upstream repeated-session behavior or long-run stability is fixed.

## Tested software reference

The local GPU validation used a prebuilt Miles image with this content digest:

```text
sha256:de63c560eb93e9b69e89a8c24eab68de12932799a8df4ce8efa7b311aee7abe2
```

That digest was available in the experiment's ECR registry; it is not a promise
that the same digest exists under `radixark/miles`. Supply a registry reference you
can access. The image carried:

| Component | Version |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.13.0+cu130 |
| Ray | 2.58.0 |
| SGLang | 0.5.19.dev49+g4e230c3 |

The BSHD extension is tied to this Miles API shape and validates its scope at
startup. Pin the base image; a newer package version is a new validation target.
The external RFT agent was inspected with `sagemaker-train==1.7.1` and
`sagemaker-core==2.7.1`. The native example's Dockerfile has its own requirements.

## CPU checks

For SDK contract checks and command construction, use a separate Python environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install fastapi uvicorn httpx boto3 requests \
  sagemaker-core==2.7.1 sagemaker-train==1.7.1
export AWS_EC2_METADATA_DISABLED=true
python -m unittest -v test_rft_compat test_sagemaker_launch
python -m unittest -v test_bshd_token_batching.BudgetTests
```

`test_rft_compat` uses the real SDK decorator/header/feedback HTTP behavior against
local servers. Token generation and the AgentCore invocation are substituted;
these tests do not contact AWS. They cover registration failure, callbacks,
streaming, errors, model-name mapping, content normalization and local control.

`test_sagemaker_launch` checks explicit-runtime selection without a default state
file and preserves multiline/JSON argument boundaries. The scheduler tests check
padded-budget enforcement, sample coverage, DP alignment and oversized rows.

The full runtime integration tests require the compatible Miles image. NVIDIA
driver visibility is needed by some imported libraries; these tests run their
training-data tensors on CPU and do not launch training or load model weights:

```bash
docker run --rm --gpus all \
  -e AWS_EC2_METADATA_DISABLED=true -e PYTHONPATH=/example:/root/miles \
  -v "$PWD:/example:ro" -w /example \
  --entrypoint python3 "$MILES_BASE_IMAGE" \
  -m unittest -v test_bshd_token_batching test_rft_rollout
```

The retry tests exercise the installed Miles sampler, including whole-group
failure after retry exhaustion. No replacement prompt is silently substituted.

## Model-token verification

A model response with HTTP200 can still be wrong training data. In an earlier
attempt, text content blocks rendered as an empty question. The front door now
joins text-only blocks verbatim before sending them to the session server.

`fake_session_server.py` can check a source question against the **actual** model
chat template before a GPU smoke:

```bash
python fake_session_server.py --rft --port 30000 \
  --verify-model-path /path/to/model \
  --verify-chat-template /path/to/the-training-chat-template.jinja \
  --verify-expected-text 'What is 17 * 23 + 4?' \
  --verify-evidence-dir /tmp/prompt-evidence
```

For a completed **FSDP** smoke, the included verifier checks the recorded questions,
token/log-prob/mask lengths and saved optimizer steps:

```bash
python verify_rft_smoke.py --run-dir /path/to/fsdp-run \
  --rollouts 2 --samples-per-rollout 8 --output /tmp/fsdp-verification.json
```

That verifier understands the FSDP checkpoint format. It is not a Megatron LoRA
resume checker. For the LoRA run, inspect saved rollout/train data, the executed
`BSHD_TOKEN_BATCH` schedule and the adapter's per-rank training-state files.
Confirm coverage and original ordering, finite log-probs, completed optimizer
updates and the expected adapter configuration. Saving the adapter alone does not
prove that all Adam state is recoverable.

## Current limits

- The recent RFT GPU runs used HTTPS to a local head, not the entirely private
  SageMaker configuration documented as the next deployment procedure.
- A completed 50-step RFT run, full multi-host RFT training and exact optimizer
  restoration have not been verified.
- The reference SageMaker RFT entrypoint requires the private hosted zone. Dynamic
  per-job private-IP registration without DNS is not implemented.
- Private DNS publication does not yet wait for Route53 `INSYNC`.
- Assistant-text mismatch diagnostics can remain nonzero. Short successful runs
  do not prove the earlier session-history problem is resolved.
- The external RFT agent's source/build context is not distributed here.

## Container-publication checks

The publication checks passed on 2026-09-11:

- 32 tests: 14 RFT SDK contracts, 9 budget/runtime-data checks, 6 retry/sampler
  checks and 3 SageMaker argument/runtime-selection checks.
- AMD64 builds of the native agent, network-smoke and training images.
- Native image `/ping` and a complete two-turn invocation against the fake model,
  returning `395`; the RFT smoke emitted `calculator → finish` and accepted both
  feedback callbacks. DNS publication was stubbed for this local container check.
- The documented 27B args file passed the real recipe CLI, producing global
  batch256, token budget20000, LoRA32/64, TP4, recompute48 and concurrency96.
  Training execution was intercepted for this command check.
- Local documentation links and shell-block syntax checks.

ARM64 Dockerfile validation passed, but an ARM64 image was not executed by this
publication check: its local builder advertises only AMD64 variants. Use an
ARM64 builder or configured emulator for the actual AgentCore image build and
runtime smoke. No AWS deployment or new GPU training was performed for publication.
