"""GRPO on grade-school math with a tool-calling agent: Qwen3 on FSDP, or Qwen3.x on Megatron + LoRA.

The policy solves grade-school math by calling two tools -- ``calculator`` and
``submit_answer`` -- and the reward is whether the submitted answer matches the ground truth.
RLVR, with the verifier inside the cluster.

Two agent modes, and the difference is only where the agent loop runs:

* ``--agent-mode local`` runs it in the trainer process against the session server directly.
  Start here: it exercises FSDP, TITO and the reward with no AWS involved.
* ``--agent-mode agentcore`` runs it in a Bedrock AgentCore microVM. Requires
  AGENTCORE_RUNTIME_ARN. With MILES_PROXY_BASE (+ MILES_PROXY_SECRET) the agent reaches the
  session server back through the proxy; without them it connects directly, which needs the
  runtime in the same VPC as the training hosts (see ``sagemaker/``).

FSDP is chosen deliberately: it loads the HF directory as-is, so a 4B run needs no
``torch_dist`` conversion step at all.

Examples:
    # smoke: 2 rollouts, in-cluster agent
    python run_qwen3_agentcore_math.py --mode smoke --agent-mode local

    # gsm-hard through AgentCore
    python run_qwen3_agentcore_math.py --agent-mode agentcore \
        --model-name Qwen3-0.6B --dataset gsm-hard

    # Qwen3.6-27B, Megatron-Bridge + LoRA r32/a64, 8 GPUs colocated (TP4 trainer, 2 TP4 engines)
    python run_qwen3_agentcore_math.py --agent-mode agentcore --model-name Qwen3.6-27B \
        --train-backend megatron --megatron-model-type qwen3.6-27B --tensor-model-parallel-size 4 \
        --lora-rank 32 --lora-alpha 64 --qkv-format-bshd --no-use-dynamic-batch-size \
        --max-tokens-per-gpu 4096 --tito-model qwen36 --rollout-num-gpus-per-engine 4 \
        --sglang-mem-fraction-static 0.5 --lr 4e-5 --adam-beta2 0.95 --weight-decay 0.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

SCRIPT_DIR = str(Path(__file__).resolve().parent)


# SGLang parser pair per TITO family. These are REQUIRED, not redundant with --tito-model: the
# training path never propagates the family's parsers to the engine, and without them the
# model's <tool_call> block comes back as raw text, the agent loop sees no tool call and every
# reward is 0. Values match the corresponding TITOTokenizer subclass.
_PARSERS = {
    "qwen3": ("qwen25", "qwen3"),
    "qwen35": ("qwen3_coder", "qwen3"),
    "qwen36": ("qwen3_coder", "qwen3"),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "smoke"] = "normal"
    # local: agent loop in-process. agentcore: our agent via AgentCore (base_url in the payload).
    # rft: an unmodified SageMaker-RFT-contract agent via AgentCore, through rft_front_door.py.
    agent_mode: Literal["local", "agentcore", "rft"] = "local"
    run_id: str = U.create_run_id()
    skip_prepare: bool = False

    model_name: str = "Qwen3-4B"
    # gsm-hard is the same word problems with ~3.3e6-magnitude operands, so the calculator
    # stops being decorative. On plain gsm8k a 4B policy scores 1.0 from the first rollout
    # (zero within-group variance, so GRPO has no gradient) and a 0.6B one learns to answer
    # from mental arithmetic while calling the tool for show.
    # Any name works as long as <data_dir>/<dataset>_train.jsonl (and _eval.jsonl for eval) exist,
    # e.g. the RFT parquet converted by sagemaker/convert_rft_parquet.py.
    dataset: str = "gsm-hard"
    model_dir: str = "/root/models"
    data_dir: str = "/root/data"
    # fsdp: full fine-tune, HF checkpoint loaded directly (no megatron_model_type; execute_train
    # asserts on exactly that pairing). megatron: Megatron-Bridge path, required for LoRA
    # (FSDP has no LoRA in Miles); needs megatron_model_type = a scripts/models/*.py name.
    train_backend: Literal["fsdp", "megatron"] = "fsdp"
    megatron_model_type: str | None = None
    tensor_model_parallel_size: int = 1
    # LoRA (megatron only). rank 0 = full fine-tune. Default targets are the dense hybrid
    # (Qwen3.5/3.6-27B) set: attention + MLP + the GDN projections, which `all-linear` misses.
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: str = ",".join(
        f"language_model.decoder.layers.*.{m}"
        for m in (
            "self_attention.linear_qkv",
            "self_attention.linear_proj",
            "mlp.linear_fc1",
            "mlp.linear_fc2",
            "self_attention.in_proj",
            "self_attention.out_proj",
        )
    )
    lora_base_cpu_backup: bool = True
    # Megatron-core's GatedDeltaNet (bridge builds the model with it) rejects packed thd
    # sequences, so GDN-family models train unpacked: bshd, micro-batch 1, no dynamic batching.
    qkv_format_bshd: bool = False
    use_dynamic_batch_size: bool = True
    # TITO family; also selects SGLang's tool-call / reasoning parser pair (see _PARSERS).
    tito_model: Literal["qwen3", "qwen35", "qwen36"] = "qwen3"
    rollout_num_gpus_per_engine: int = 1
    sglang_mem_fraction_static: float = 0.75
    # Optimizer. 1e-6 is the full-fine-tune value; LoRA recipes run 1e-5..4e-5.
    lr: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    clip_grad: float = 1.0
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    enable_eval: bool = True
    num_gpus_per_node: int | None = 8

    # One trajectory per prompt-sample, each holding a session for its whole episode, so the
    # worker count has to cover the full concurrent fan-out.
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    num_rollout: int = 400
    # max_seq_len bounds the whole trajectory (trainer packing / --max-seq-len); the per-episode
    # generation cap defaults to it and can be set lower for long-context runs.
    max_seq_len: int = 4096
    max_response_len: int | None = None
    # Under --colocate the trainer only gets what SGLang's mem-fraction-static leaves behind,
    # so this is not a function of model size. It also has to survive the run getting *harder*:
    # as submit_rate rises the policy stops bailing out early and completes full multi-turn
    # episodes, so mean episode length grew 404 -> 1182 tokens over 29 rollouts and 32768
    # OOMed at rollout 28. 9216 is the value Miles' own 0.6B recipe uses.
    max_tokens_per_gpu: int = 9216
    # Frequent enough that an OOM or a preemption late in the run does not throw the run away.
    save_interval: int = 10
    eval_interval: int = 10
    # JSON env for the trainer processes. expandable_segments:True is the memory-friendly default,
    # but sharing such allocations with the colocated SGLang engines (CUDA IPC weight sync) goes
    # through pidfd_getfd, which SageMaker's container runtime forbids -- pass
    # '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:False"}' there (sagemaker/entrypoint.py does).
    train_env_vars: str = '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'

    agentcore_runtime_arn: str = os.environ.get("AGENTCORE_RUNTIME_ARN", "")
    proxy_base: str = os.environ.get("MILES_PROXY_BASE", "")
    proxy_secret: str = os.environ.get("MILES_PROXY_SECRET", "")
    aws_region: str = os.environ.get("AWS_REGION", "us-west-2")

    # Read from the environment so the key never appears in a command line or in this file.
    wandb_key: str = os.environ.get("WANDB_API_KEY", "")
    wandb_project: str = "miles-agentcore-gsm8k"
    wandb_team: str = os.environ.get("WANDB_TEAM", "")

    def __post_init__(self):
        if self.mode == "smoke":
            self.rollout_batch_size = 2
            self.n_samples_per_prompt = 4
            self.num_rollout = 2
        if (self.train_backend == "megatron") != (self.megatron_model_type is not None):
            raise ValueError("--train-backend megatron requires --megatron-model-type (and fsdp forbids it)")
        if self.lora_rank and self.train_backend != "megatron":
            raise ValueError("LoRA needs --train-backend megatron; FSDP has no LoRA in Miles")
        if self.agent_mode in ("agentcore", "rft"):
            if not self.agentcore_runtime_arn:
                raise ValueError(f"--agent-mode {self.agent_mode} requires AGENTCORE_RUNTIME_ARN")
        if self.agent_mode == "rft" and not os.environ.get("MILES_RFT_FRONT_DOOR_URL"):
            raise ValueError("--agent-mode rft requires MILES_RFT_FRONT_DOOR_URL (the address the agent resolves)")
            # Proxy mode is opt-in by MILES_PROXY_BASE; then the secret must come with it.
            if self.proxy_base and not self.proxy_secret:
                raise ValueError("MILES_PROXY_BASE is set, so MILES_PROXY_SECRET is required too")

    @property
    def global_batch_size(self) -> int:
        return self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def session_server_workers(self) -> int:
        # A session server is async and holds many concurrent sessions, so this does not need
        # to be 1:1 with trajectories. Capped because each worker is a fresh interpreter that
        # re-imports transformers at spawn: 128+ of them add minutes of startup for no gain.
        return min(32, max(8, self.global_batch_size))

    @property
    def agent_function_path(self) -> str:
        return {
            "local": "local_agent_function.run",
            "agentcore": "agentcore_agent_function.run",
            "rft": "rft_agent_function.run",
        }[self.agent_mode]


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.exec_command_cpu(f"python {SCRIPT_DIR}/prepare_data.py --dataset {args.dataset} --output-dir {args.data_dir}")


def execute(args: ScriptArgs):
    is_smoke = args.mode == "smoke"
    hf_checkpoint = f"{args.model_dir}/{args.model_name}"

    ckpt_args = (
        f"--hf-checkpoint {hf_checkpoint} "
        f"--save {args.output_dir}/{args.run_id}/checkpoints "
        f"--save-interval {2 if is_smoke else args.save_interval} "
    )
    if args.train_backend == "fsdp":
        # FSDP reads the same HF directory for the KL reference model. Bridge/LoRA has no
        # separate reference: the frozen base with adapters disabled is the reference.
        ckpt_args += f"--ref-load {hf_checkpoint} "

    # No --apply-chat-template and no --rm-type: Sample.prompt must stay a messages list for
    # the session server to render, and grading goes through --custom-rm-path.
    rollout_args = (
        f"--prompt-data {args.data_dir}/{args.dataset}_train.jsonl "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.global_batch_size} "
        f"--rollout-max-response-len {args.max_response_len or args.max_seq_len} "
        f"--rollout-temperature {args.rollout_temperature} "
        f"--rollout-top-p {args.rollout_top_p} "
        "--balance-data "
    )

    # Held-out eval, on the same agent path as training. Without it a flattening train curve is
    # ambiguous: gsm-hard only has 1119 training rows and a 50-rollout run at batch 16 consumes
    # ~700 of them, so "stopped learning" and "started memorising" look identical.
    # One sample per prompt keeps the AgentCore bill down -- this measures the greedy policy,
    # not its spread.
    eval_args = ""
    if not is_smoke and args.enable_eval:
        eval_args = (
            f"--eval-prompt-data {args.dataset} {args.data_dir}/{args.dataset}_eval.jsonl "
            "--n-samples-per-eval-prompt 1 "
            f"--eval-interval {args.eval_interval} "
            f"--eval-max-response-len {args.max_response_len or args.max_seq_len} "
        )

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        f"--custom-agent-function-path {args.agent_function_path} "
        "--custom-rm-path math_reward.reward_func "
        "--rollout-function-path math_reward.RolloutFn "
        # A trajectory that never produced a model call is useless for training; drop the
        # whole GRPO group rather than feed it a zero-variance batch.
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--use-session-server "
        f"--tito-model {args.tito_model} "
        "--session-server-port 30000 "
        f"--session-server-workers {args.session_server_workers} "
        f"--max-seq-len {args.max_seq_len} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    if args.train_backend == "fsdp":
        grpo_args += "--use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl "

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        f"--weight-decay {args.weight_decay} "
        f"--adam-beta1 {args.adam_beta1} "
        f"--adam-beta2 {args.adam_beta2} "
        f"--clip-grad {args.clip_grad} "
    )

    common_backend_args = (
        f"--update-weight-buffer-size {512 * 1024 * 1024} "
        f"--train-env-vars '{args.train_env_vars}' "
    )
    if args.train_backend == "fsdp":
        train_backend_args = (
            "--train-backend fsdp "
            "--attn-implementation flash_attention_2 "
            "--gradient-checkpointing "
        ) + common_backend_args
    else:
        train_backend_args = (
            "--train-backend megatron "
            # Bridge builds the Megatron model from the HF checkpoint; LoRA only exists on this path.
            "--megatron-to-hf-mode bridge "
            f"--tensor-model-parallel-size {args.tensor_model_parallel_size} --sequence-parallel "
            "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
            "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
            "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
            "--attention-dropout 0.0 --hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
            f"--adam-eps {args.adam_eps} "
        ) + common_backend_args
        if args.qkv_format_bshd:
            train_backend_args += "--qkv-format bshd --micro-batch-size 1 "
        if args.lora_rank:
            train_backend_args += (
                f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout {args.lora_dropout} "
                f'--target-modules "{args.target_modules}" '
                "--no-gradient-accumulation-fusion "
            )
            if args.lora_base_cpu_backup:
                train_backend_args += "--lora-base-cpu-backup "

    tool_call_parser, reasoning_parser = _PARSERS[args.tito_model]
    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-chunked-prefill-size 4096 "
        f"--sglang-tool-call-parser {tool_call_parser} "
        f"--sglang-reasoning-parser {reasoning_parser} "
    )
    if args.lora_rank:
        # The engines serve the live adapter under the fixed name the session server attaches.
        sglang_args += f"--sglang-max-lora-rank {args.lora_rank} --sglang-lora-backend triton "

    perf_args = f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    if args.use_dynamic_batch_size:
        perf_args += "--use-dynamic-batch-size "

    wandb_args = ""
    if args.wandb_key:
        wandb_args = (
            "--use-wandb "
            f"--wandb-project {args.wandb_project} "
            f"--wandb-group {args.model_name}-{args.dataset}-{args.agent_mode} "
            f"--wandb-run-id {args.run_id} "
            f"--wandb-key {args.wandb_key} "
        )
        if args.wandb_team:
            wandb_args += f"--wandb-team {args.wandb_team} "

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        # Trainer and engines share the 8 GPUs; sleep/wake_up hands the memory back and forth
        # while the external agent is thinking.
        "--colocate "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )

    train_args = (
        f"{ckpt_args}{rollout_args}{eval_args}{agent_args}{grpo_args}"
        f"{optimizer_args}{train_backend_args}{sglang_args}{perf_args}{wandb_args}{misc_args}"
    )

    extra_env_vars = {
        # SCRIPT_DIR so --custom-agent-function-path / --custom-rm-path resolve, matching the
        # Harbor recipe's convention.
        "PYTHONPATH": f"{SCRIPT_DIR}:{U.repo_base_dir}",
    }
    if args.agent_mode in ("agentcore", "rft"):
        extra_env_vars |= {
            "AGENTCORE_RUNTIME_ARN": args.agentcore_runtime_arn,
            "AWS_REGION": args.aws_region,
        }
        if args.proxy_base:
            extra_env_vars |= {"MILES_PROXY_BASE": args.proxy_base, "MILES_PROXY_SECRET": args.proxy_secret}
    if args.agent_mode == "rft":
        extra_env_vars |= {
            k: os.environ[k]
            for k in ("MILES_RFT_FRONT_DOOR_URL", "MILES_RFT_FRONT_DOOR_LOCAL", "TRAINING_JOB_ARN")
            if k in os.environ
        }

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
