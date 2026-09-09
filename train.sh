#!/usr/bin/env bash
# One-shot setup + launch for the AgentCore math recipe.
#
# This is environment orchestration, not the launcher: the launcher is
# run_qwen3_agentcore_math.py, which this script ends up calling through run_in_docker.sh.
# Every step is idempotent -- re-running skips whatever is already in place.
#
#   ./train.sh local            # in-cluster agent. Needs GPUs + docker only. Start here.
#   ./train.sh agentcore        # full path through Bedrock AgentCore. Needs AWS + inbound 443.
#
# Knobs (all optional):
#   ROLLOUTS=50 MODEL=Qwen3-0.6B DATASET=gsm-hard BATCH=16 GROUP=8 ./train.sh local
#   WANDB_API_KEY=...           # enables wandb; without it Miles just skips it
#   AGENTCORE_RUNTIME_ARN=...   # required for agentcore mode
#   PROXY_PORT=18476 PUBLIC_PORT=443
set -euo pipefail

AGENT_MODE="${1:-local}"
ROLLOUTS="${ROLLOUTS:-50}"
MODEL="${MODEL:-Qwen3-0.6B}"
DATASET="${DATASET:-gsm-hard}"
BATCH="${BATCH:-16}"
GROUP="${GROUP:-8}"
PROXY_PORT="${PROXY_PORT:-18476}"
PUBLIC_PORT="${PUBLIC_PORT:-443}"
AGENTCORE_MAX_CONCURRENT="${AGENTCORE_MAX_CONCURRENT:-16}"

IMAGE="${IMAGE:-radixark/miles:latest}"
HOST_ROOT="${HOST_ROOT:-$HOME}"
MODEL_DIR="$HOST_ROOT/models"
DATA_DIR="$HOST_ROOT/data"
OUT_DIR="$HOST_ROOT/shared_data"
SECRET_FILE="$HOST_ROOT/.miles_proxy_secret"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mok\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m!!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31mFAILED:\033[0m %s\n\n' "$*" >&2; exit 1; }

case "$AGENT_MODE" in
  local|agentcore) ;;
  *) die "mode must be 'local' or 'agentcore', got '$AGENT_MODE'" ;;
esac

# ---------------------------------------------------------------- 1. prerequisites
step "1/7  prerequisites"
command -v docker >/dev/null || die "docker not found"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (permissions?)"
ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null)"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found -- this recipe needs GPUs"
GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[[ "$GPUS" -ge 1 ]] || die "no GPUs visible"
ok "$GPUS GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# The recipe assumes 8 visible GPUs (--actor-num-gpus-per-node 8 under --colocate).
[[ "$GPUS" -eq 8 ]] || warn "recipe is tuned for 8 GPUs; you have $GPUS -- pass --num-gpus-per-node via EXTRA"

docker run --rm --gpus all "$IMAGE" true >/dev/null 2>&1 \
  || die "docker cannot see the GPUs -- is the nvidia container toolkit installed?"
ok "nvidia container runtime works"

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
[[ "$USED" -lt 2000 ]] || warn "GPUs already hold ${USED}MiB -- a stale run may still be up (docker ps)"

# ---------------------------------------------------------------- 2. image
step "2/7  miles image"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  ok "$IMAGE present ($(docker image inspect "$IMAGE" --format '{{.Size}}' | awk '{printf "%.1fGB", $1/1e9}'))"
else
  warn "pulling $IMAGE -- this is ~43GB, expect a long wait"
  docker pull "$IMAGE"
fi

# ---------------------------------------------------------------- 3. model
step "3/7  model weights"
mkdir -p "$MODEL_DIR" "$DATA_DIR" "$OUT_DIR"
if [[ -f "$MODEL_DIR/$MODEL/config.json" ]]; then
  ok "$MODEL already at $MODEL_DIR/$MODEL"
else
  command -v hf >/dev/null || pip install -q "huggingface_hub" || die "cannot install huggingface_hub"
  hf download "Qwen/$MODEL" --local-dir "$MODEL_DIR/$MODEL" >/dev/null \
    || die "download of Qwen/$MODEL failed"
  ok "downloaded $MODEL"
fi

# ---------------------------------------------------------------- 4. data
step "4/7  dataset ($DATASET)"
if [[ -s "$DATA_DIR/${DATASET}_train.jsonl" ]]; then
  ok "$(wc -l < "$DATA_DIR/${DATASET}_train.jsonl") train / $(wc -l < "$DATA_DIR/${DATASET}_eval.jsonl" 2>/dev/null || echo 0) eval rows"
else
  # Run inside the image so the `datasets` dependency does not have to exist on the host.
  docker run --rm --network host \
    -v "$REPO_ROOT:/root/miles:ro" -v "$DATA_DIR:/root/data" \
    --entrypoint python3 "$IMAGE" \
    /root/miles/examples/experimental/agentcore/prepare_data.py --dataset "$DATASET" --output-dir /root/data \
    || die "dataset preparation failed"
  ok "prepared $DATASET"
fi

# ---------------------------------------------------------------- 5. proxy (agentcore only)
step "5/7  proxy"
if [[ "$AGENT_MODE" == "local" ]]; then
  ok "skipped -- local mode talks to the session server directly"
else
  # If a proxy container is already up, adopt ITS secret rather than minting a new one. Getting
  # this wrong is silent and expensive: the agent function signs tokens with one secret while
  # the proxy verifies with another, so every request 403s, every trajectory fails, and the run
  # dies several minutes later inside the eval metrics with an unrelated-looking TypeError.
  RUNNING_SECRET=""
  if docker inspect miles-proxy >/dev/null 2>&1; then
    RUNNING_SECRET=$(docker inspect miles-proxy --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
      | sed -n 's/^MILES_PROXY_SECRET=//p' | head -1)
  fi

  if [[ -n "$RUNNING_SECRET" ]] && curl -sf --max-time 3 "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1; then
    MILES_PROXY_SECRET="$RUNNING_SECRET"
    export MILES_PROXY_SECRET
    # Keep the file in step so a later restart reuses the same value.
    if [[ "$(cat "$SECRET_FILE" 2>/dev/null)" != "$MILES_PROXY_SECRET" ]]; then
      umask 077
      printf '%s' "$MILES_PROXY_SECRET" > "$SECRET_FILE"
      warn "adopted the running proxy's secret and refreshed $SECRET_FILE"
    fi
    ok "proxy already healthy on $PROXY_PORT (secret matched)"
  else
    [[ -s "$SECRET_FILE" ]] || { umask 077; openssl rand -hex 32 > "$SECRET_FILE"; }
    chmod 600 "$SECRET_FILE"
    MILES_PROXY_SECRET="$(cat "$SECRET_FILE")"
    export MILES_PROXY_SECRET
    docker rm -f miles-proxy >/dev/null 2>&1 || true
    # Containerised with --restart so it outlives this shell. Two earlier attempts ran it as a
    # background process and both were killed when the launching shell exited.
    docker run -d --name miles-proxy --restart unless-stopped --network host \
      -v "$SCRIPT_DIR:/app:ro" -e "MILES_PROXY_SECRET=$MILES_PROXY_SECRET" \
      -w /app --entrypoint python3 "$IMAGE" proxy.py serve --port "$PROXY_PORT" >/dev/null \
      || die "could not start the proxy container"
    for _ in $(seq 1 20); do
      sleep 2
      curl -sf --max-time 3 "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1 && break
    done
    curl -sf --max-time 3 "http://127.0.0.1:$PROXY_PORT/health" >/dev/null \
      || die "proxy did not come up: docker logs miles-proxy"
    ok "proxy started on $PROXY_PORT"
  fi

  # AgentCore can only reach an allowed inbound port, which is rarely the proxy's. Redirect,
  # scoped to the public interface: an unscoped PREROUTING rule also hijacks outbound HTTPS
  # from docker bridge containers, which breaks image pulls in confusing ways.
  IFACE=$(ip route get 8.8.8.8 2>/dev/null | grep -oE 'dev [a-z0-9]+' | awk '{print $2}')
  if sudo -n iptables -t nat -C PREROUTING -i "$IFACE" -p tcp --dport "$PUBLIC_PORT" \
       -j REDIRECT --to-port "$PROXY_PORT" 2>/dev/null; then
    ok "iptables $PUBLIC_PORT -> $PROXY_PORT already in place (on $IFACE)"
  elif sudo -n true 2>/dev/null; then
    sudo iptables -t nat -A PREROUTING -i "$IFACE" -p tcp --dport "$PUBLIC_PORT" \
      -j REDIRECT --to-port "$PROXY_PORT"
    ok "iptables $PUBLIC_PORT -> $PROXY_PORT added (on $IFACE)"
  else
    warn "no passwordless sudo -- add this yourself, or the agent cannot reach the proxy:"
    warn "  sudo iptables -t nat -A PREROUTING -i $IFACE -p tcp --dport $PUBLIC_PORT -j REDIRECT --to-port $PROXY_PORT"
  fi
fi

# ---------------------------------------------------------------- 6. AWS (agentcore only)
step "6/7  AgentCore runtime"
if [[ "$AGENT_MODE" == "local" ]]; then
  ok "skipped -- local mode needs no AWS"
else
  command -v aws >/dev/null || die "aws cli not found"
  aws sts get-caller-identity >/dev/null 2>&1 || die "AWS credentials not usable"
  ok "AWS account $(aws sts get-caller-identity --query Account --output text)"

  [[ -n "${AGENTCORE_RUNTIME_ARN:-}" ]] \
    || die "AGENTCORE_RUNTIME_ARN must be set (see README steps 3-4 to build the image and create the runtime)"
  RUNTIME_ID="${AGENTCORE_RUNTIME_ARN##*/}"
  STATUS=$(aws bedrock-agentcore-control get-agent-runtime \
    --region "${AWS_REGION:-us-west-2}" --agent-runtime-id "$RUNTIME_ID" \
    --query status --output text 2>/dev/null || echo MISSING)
  [[ "$STATUS" == "READY" ]] || die "runtime $RUNTIME_ID is '$STATUS', expected READY"
  ok "runtime $RUNTIME_ID READY"

  # The agent reaches us at the public address; discover it rather than hardcoding.
  if [[ -z "${MILES_PROXY_BASE:-}" ]]; then
    TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
      -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
    PUBIP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)
    [[ -n "$PUBIP" ]] || die "could not discover a public IP; set MILES_PROXY_BASE yourself"
    export MILES_PROXY_BASE="http://$PUBIP:$PUBLIC_PORT"
  fi
  ok "agent will call back at $MILES_PROXY_BASE"
  warn "that path is plain HTTP: the token and the whole conversation cross the internet in the clear"
fi

# ---------------------------------------------------------------- 7. launch
step "7/7  launch"
docker rm -f miles-train >/dev/null 2>&1 || true

export AGENTCORE_MAX_CONCURRENT
export MODE=normal
export EXTRA_ARGS="--model-name $MODEL --dataset $DATASET --rollout-batch-size $BATCH --n-samples-per-prompt $GROUP --num-rollout $ROLLOUTS ${EXTRA:-}"
export HOST_REPO="$REPO_ROOT"

"$SCRIPT_DIR/run_in_docker.sh" "$AGENT_MODE"

cat <<EOF

  model    $MODEL
  dataset  $DATASET
  shape    $BATCH prompts x $GROUP samples = $((BATCH * GROUP)) trajectories/rollout, $ROLLOUTS rollouts
  agent    $AGENT_MODE
  wandb    ${WANDB_API_KEY:+enabled}${WANDB_API_KEY:-disabled (set WANDB_API_KEY)}

  follow      docker logs -f miles-train
  reward      docker logs miles-train 2>&1 | grep -oE "'rollout/raw_reward': [-0-9.]+"
  behaviour   docker logs miles-train 2>&1 | grep -oE "agent metrics for rollout [0-9]+: \{[^}]*\}" | tail -3
  checkpoints $OUT_DIR/<run-id>/checkpoints
  trajectories $OUT_DIR/<run-id>/dump_details/trajectory/

EOF
