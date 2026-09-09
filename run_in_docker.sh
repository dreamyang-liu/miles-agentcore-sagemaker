#!/usr/bin/env bash
# Launch the GSM8K AgentCore recipe inside the Miles container.
#
# --cap-add SYS_PTRACE --security-opt seccomp=unconfined are REQUIRED: weight sync between
# the FSDP trainer and the SGLang engines uses pidfd_getfd, which Docker's default seccomp
# profile blocks. Without them every engine dies with
#   RuntimeError: pidfd_getfd: Operation not permitted
# on the first update_weights, before rollout ever starts. The installation guide's ROCm tab
# carries these flags; the NVIDIA tab omits them.
#
# Usage:
#   ./run_in_docker.sh local                 # in-cluster agent, no AWS needed
#   ./run_in_docker.sh agentcore             # via Bedrock AgentCore + proxy
#   MODE=normal ./run_in_docker.sh local     # full run instead of the 2-rollout smoke
#   EXTRA_ARGS="--model-name Qwen3-0.6B --num-rollout 8" ./run_in_docker.sh agentcore
set -euo pipefail

AGENT_MODE="${1:-local}"
MODE="${MODE:-smoke}"
NAME="${NAME:-miles-train}"
HOST_REPO="${HOST_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# Passed through to the launcher verbatim, so recipe knobs do not need a wrapper flag each.
EXTRA_ARGS="${EXTRA_ARGS:-}"

extra_env=()
# Passed by env so the key stays out of the container's command line and out of `ps`.
[[ -n "${WANDB_API_KEY:-}" ]] && extra_env+=(-e "WANDB_API_KEY=$WANDB_API_KEY")
[[ -n "${WANDB_TEAM:-}" ]] && extra_env+=(-e "WANDB_TEAM=$WANDB_TEAM")

if [[ "$AGENT_MODE" == "agentcore" ]]; then
  : "${AGENTCORE_RUNTIME_ARN:?must be set for agentcore mode}"
  : "${MILES_PROXY_BASE:?must be set for agentcore mode}"
  : "${MILES_PROXY_SECRET:?must be set for agentcore mode}"
  extra_env+=(-e "AGENTCORE_RUNTIME_ARN=$AGENTCORE_RUNTIME_ARN")
  extra_env+=(-e "MILES_PROXY_BASE=$MILES_PROXY_BASE")
  extra_env+=(-e "MILES_PROXY_SECRET=$MILES_PROXY_SECRET")
  extra_env+=(-e "AWS_REGION=${AWS_REGION:-us-west-2}")
  # Decouples the GRPO batch shape from AgentCore's maxVms / session-creation-rate limits.
  extra_env+=(-e "AGENTCORE_MAX_CONCURRENT=${AGENTCORE_MAX_CONCURRENT:-16}")
  # The trainer calls InvokeAgentRuntime, so it needs credentials.
  [[ -n "${AWS_ACCESS_KEY_ID:-}" ]] && extra_env+=(-e "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID")
  [[ -n "${AWS_SECRET_ACCESS_KEY:-}" ]] && extra_env+=(-e "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY")
  [[ -n "${AWS_SESSION_TOKEN:-}" ]] && extra_env+=(-e "AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN")
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --network host --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$HOST_REPO:/root/miles" \
  -v $HOME/models:/root/models \
  -v $HOME/data:/root/data \
  -v $HOME/shared_data:/root/shared_data \
  -v $HOME/.aws:/root/.aws:ro \
  "${extra_env[@]}" \
  -w /root/miles/examples/experimental/agentcore \
  --entrypoint bash radixark/miles:latest -c \
  "PYTHONPATH=/root/miles python3 run_qwen3_agentcore_math.py --mode $MODE --agent-mode $AGENT_MODE --skip-prepare $EXTRA_ARGS 2>&1"

echo "started $NAME (mode=$MODE agent-mode=$AGENT_MODE ${EXTRA_ARGS:+extra: $EXTRA_ARGS})"
echo "follow with: docker logs -f $NAME"
