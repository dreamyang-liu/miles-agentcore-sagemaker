# Run training through AgentCore

Build the [containers](CONTAINERS.md) and complete the
[private-network smoke](../sagemaker/AGENTCORE_VPC_SETUP.md) first. This guide uses
an existing RFT SDK runtime. Its model endpoint must reach the training head's
front door; the training role must be able to invoke that runtime.

## Model and input records

SageMaker expects:

```text
s3://<bucket>/miles/models/Qwen3.6-27B/config.json
s3://<bucket>/miles/models/Qwen3.6-27B/<all model/tokenizer files>
s3://<bucket>/miles/data/<dataset>_train.jsonl
```

Model files are fully replicated to every training host. The entrypoint links
the model channel to `/root/models/<model-name>` on both the head and workers.

For the RFT parquet format, the `prompt` column contains a JSON string with the
original question, messages and `reward_spec.ground_truth`:

```bash
python -m pip install pyarrow
python sagemaker/convert_rft_parquet.py /path/to/prompts.parquet \
  --name rft-gsm8k --output-dir /tmp/miles-data --holdout 0
```

`--holdout 0` keeps all accepted records in training. The converter's default
holdout is 128; it skips records missing a ground truth or messages, so verify its
reported row counts. It does not deduplicate data.

One output record has this shape:

```json
{
  "prompt": [{"role": "user", "content": "What is 17 * 23 + 4?"}],
  "metadata": {
    "answer": "395",
    "instance_id": "example-0",
    "rft_record": {
      "instance_id": "example-0",
      "data_source": "math",
      "instance": "What is 17 * 23 + 4?",
      "prompt": [{"role": "user", "content": "What is 17 * 23 + 4?"}],
      "reward_spec": {"ground_truth": "395"},
      "extra_info": {}
    }
  }
}
```

The RFT agent receives `rft_record`, including the ground truth required by its SDK
contract. Miles uses `metadata.answer` for its own training reward. Native mode
has a different invocation contract.

```bash
aws s3 sync /path/to/Qwen3.6-27B/ \
  "s3://$MILES_SM_BUCKET/miles/models/Qwen3.6-27B/"
aws s3 cp /tmp/miles-data/rft-gsm8k_train.jsonl \
  "s3://$MILES_SM_BUCKET/miles/data/rft-gsm8k_train.jsonl"
```

## Qwen3.6-27B LoRA: start with three steps

[`configs/qwen36-rft-3step.args`](../configs/qwen36-rft-3step.args) contains the
current recipe's model/training settings:

| Setting | Value |
| --- | --- |
| Hardware | One host, eight GPUs; validated on H100 80GB |
| Training / inference parallelism | TP4/DP2 trainer; two TP4 SGLang engines |
| LoRA | Rank 32, alpha 64 |
| Global batch | 32 prompts × 8 samples = **256 trajectories** |
| BSHD microbatch budget | **20,000 tokens including padding** |
| Recompute | Block checkpointing, 48 of 64 layers |
| Sampling | Temperature/top-p 1/1; 4,096 output tokens per model call |
| Context limit | 65,536 tokens |
| Loss / clipping | PPO policy loss, group-based GRPO advantages, ratio range 0.8–1.2 |
| Optimizer | LR 4e-5; Adam 0.9/0.95/1e-8; weight decay 0; grad clip 1 |
| Validation run | Three rollouts/updates; save at three; evaluation disabled |
| Full-trajectory retries | Three retries after the first attempt; fail on exhaustion |

Use **normal mode** for this configuration: the launcher's `smoke` mode deliberately
overrides the batch and step count to its tiny smoke settings.

```bash
export RFT_RUNTIME_ARN='arn:aws:bedrock-agentcore:<region>:<account>:runtime/<runtime-id>'
QWEN_ARGS="$(cat configs/qwen36-rft-3step.args)"

JOB=$(python sagemaker/launch_train.py start \
  --mode normal --agent-mode rft \
  --runtime-arn "$RFT_RUNTIME_ARN" \
  --instance-type ml.p5.48xlarge --count 1 \
  --model-name Qwen3.6-27B --dataset rft-gsm8k \
  --agentcore-max-concurrent 96 --max-runtime 7200 \
  --extra-args "$QWEN_ARGS")
python sagemaker/launch_train.py watch "$JOB"
```

`MILES_TRAIN_IMAGE`, `MILES_SM_ROLE_ARN`, `MILES_SM_BUCKET`, `AWS_REGION` and the
account's infrastructure state must be configured as in the previous guides.
An explicit `--runtime-arn` avoids needing a default runtime state file.

The recent three-step GPU evidence was collected on a local training host with an
HTTPS callback. Applying this recipe to SageMaker/private-VPC is a deployment to
validate, not a claim that this exact combination already completed there.

For a later 50-step run, copy the args file and change `--num-rollout 3` to 50 and
`--save-interval 3` to 10. Budget wall time for startup and the measured step time;
the launcher's default three-hour timeout can be too short. The full 50-step RFT
run has not been accepted as stable.

## What the token budget means

The BSHD extension packs shorter trajectories into larger microbatches and pads
each microbatch independently. It bounds:

```text
number of trajectories × rounded longest sequence length <= 20,000
```

With TP4 and the validated padding multiple, a 5,000-token trajectory rounds up
to 5,120. Four such trajectories would cost 20,480, so only three can share that
microbatch. A single padded sequence exceeding the budget fails visibly; no
trajectory is truncated or dropped to make it fit.

The extension aligns gradient-accumulation counts between DP shards and restores
outputs to the original sample order. It uses
`--custom-megatron-init-path bshd_token_batching.install`; keep
`--no-use-dynamic-batch-size` in the launcher arguments so the hook can install
the BSHD implementation before enabling result-order restoration. The nominal
`--micro-batch-size 4` does not fix the executed batch sizes in this mode.

## Local EC2 training head in the VPC

For a fixed private-IP head, run the RFT front door on that host and point the
RFT agent's model endpoint at `http://<head-private-ip>:30100`. A hosted zone is
optional for this topology. Configure the VPC routes/security groups first.

The following runs the recipe inside the training image while overriding its
SageMaker-only entrypoint. It assumes Docker host networking and AWS credentials
available through the mounted profile directory. Set the paths to your existing
model/data/output locations:

```bash
export HEAD_PRIVATE_IP='<training-head-private-ip>'
export MODEL_PARENT='/path/containing/Qwen3.6-27B'
export DATA_DIR='/path/to/miles-data'
export OUTPUT_DIR='/path/to/training-output'
export MILES_RFT_FRONT_DOOR_URL="http://${HEAD_PRIVATE_IP}:30100"
export MILES_RFT_FRONT_DOOR_LOCAL=http://127.0.0.1:30100

docker run --rm --init --gpus all --network host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_PARENT:/root/models:ro" \
  -v "$DATA_DIR:/root/data:ro" \
  -v "$OUTPUT_DIR:/root/shared_data" \
  -v "$HOME/.aws:/root/.aws:ro" \
  -e AWS_REGION \
  -e "AGENTCORE_RUNTIME_ARN=$RFT_RUNTIME_ARN" \
  -e MILES_RFT_FRONT_DOOR_URL -e MILES_RFT_FRONT_DOOR_LOCAL \
  -e "MILES_HOST_IP=$HEAD_PRIVATE_IP" \
  --entrypoint python3 "$MILES_TRAIN_IMAGE" \
  /root/miles/examples/experimental/agentcore/scripts/run_with_args.py \
  --args-file /root/miles/examples/experimental/agentcore/configs/qwen36-rft-3step.args -- \
  --mode normal --agent-mode rft --skip-prepare \
  --model-name Qwen3.6-27B --dataset rft-gsm8k \
  --num-gpus-per-node 8 --agentcore-max-concurrent 96 \
  --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:False"}'
```

`run_with_args.py` parses the args file with `shlex` and executes an argument list.
It does not evaluate the file as shell code. Arguments after `--` follow the file's
arguments; use `--print-command` before `--` to inspect the command without running.
The SageMaker entrypoint likewise parses `MILES_SM_EXTRA_ARGS` into an argument list,
so multiline settings and JSON values retain their intended boundaries.

## Check outputs and stop

```bash
python sagemaker/launch_train.py logs "$JOB"
aws s3 ls "s3://$MILES_SM_BUCKET/miles/checkpoints/$JOB/" --recursive
# Stop only the job you launched:
python sagemaker/launch_train.py stop "$JOB"
```

SageMaker uses `/opt/ml/checkpoints` with `CheckpointConfig` for checkpoint syncing.
The adapter is under a run-specific `checkpoints/iter_*/adapter` directory;
`iter_0000002` denotes the third update in the tested path. An adapter save does not
establish exact optimizer resume. A final `/opt/ml/model` export is not automated.

Confirm all of the following from actual artifacts: successful AgentCore model and
feedback requests, source-question fidelity, 256 trajectories per intended batch,
finite token/log-prob/mask data, completed optimizer updates and checkpoint files.
The model-side callback's HTTP 200 alone does not prove a correct training step.
