"""Token-budgeted BSHD batches for the pinned Miles Megatron recipe.

Enable through --custom-megatron-init-path bshd_token_batching.install. The
launcher leaves upstream dynamic batching disabled during argument validation;
this trainer-local extension installs the BSHD implementation before enabling
Miles' dynamic-batch result reordering. No upstream files are modified.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def padded_length(length: int, multiple: int) -> int:
    return ((length + multiple - 1) // multiple) * multiple


def make_batches(lengths: Sequence[int], budget: int, multiple: int) -> list[list[int]]:
    """First-fit decreasing with a hard budget on B * padded(S), not sum(S)."""
    if budget <= 0 or multiple <= 0:
        raise ValueError("Token budget and padding multiple must be positive")
    if any(length <= 0 for length in lengths):
        raise ValueError("Training sequences must contain at least one token")
    batches: list[list[int]] = []
    widths: list[int] = []
    for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        width = padded_length(lengths[index], multiple)
        if width > budget:
            raise ValueError(
                f"Sequence {index} has {lengths[index]} tokens ({width} after padding), "
                f"exceeding the {budget}-token microbatch budget. No sample was truncated or dropped."
            )
        for batch, batch_width in zip(batches, widths, strict=True):
            if (len(batch) + 1) * batch_width <= budget:
                batch.append(index)
                break
        else:
            batches.append([index])
            widths.append(width)
    return batches


def split_to_count(batches: Sequence[Sequence[int]], count: int) -> list[list[int]]:
    """Align DP accumulation counts by splitting batches, never duplicating rows."""
    result = [list(batch) for batch in batches]
    if not len(result) <= count <= sum(map(len, result)):
        raise ValueError(f"Cannot split {len(result)} batches into {count} nonempty batches")
    while len(result) < count:
        index = max(range(len(result)), key=lambda i: len(result[i]))
        batch = result[index]
        middle = len(batch) // 2
        result[index:index + 1] = [batch[:middle], batch[middle:]]
    return result


def batch_widths(lengths: Sequence[int], batches: Sequence[Sequence[int]], multiple: int) -> list[int]:
    """Return each row's own microbatch padding length in original row order."""
    widths = [0] * len(lengths)
    for batch in batches:
        width = padded_length(max(lengths[i] for i in batch), multiple)
        for index in batch:
            if widths[index]:
                raise ValueError(f"Duplicate sample index {index} in microbatch schedule")
            widths[index] = width
    if any(width == 0 for width in widths):
        raise ValueError("Microbatch schedule omitted samples")
    return widths


def _check_scope(args, parallel) -> None:
    if args.qkv_format != "bshd" or args.train_backend != "megatron":
        raise ValueError("BSHD token batching requires the Megatron BSHD training path")
    if parallel.cp.size != 1 or parallel.pp.size != 1 or parallel.vpp_size != 1:
        raise ValueError("BSHD token batching currently supports CP1/PP1 without virtual pipelines")
    if getattr(args, "use_dynamic_global_batch_size", False):
        raise ValueError("BSHD token batching preserves a fixed global batch size")
    if getattr(args, "compress_ratios", None):
        raise ValueError("BSHD token batching does not support compressed-attention padding")
    if args.max_tokens_per_gpu is None or args.max_tokens_per_gpu <= 0:
        raise ValueError("BSHD token batching requires a positive --max-tokens-per-gpu")


def get_data_iterator(args, model, rollout_data):
    """Build one schedule shared by actor log-probs and gradient accumulation."""
    import torch
    import torch.distributed as dist

    from miles.backends.training_utils import data
    from miles.utils.ft_utils.process_group_utils import GeneralPGUtil

    parallel = data.get_parallel_state()
    _check_scope(args, parallel)
    if not args.use_dynamic_batch_size:
        raise RuntimeError("BSHD token batching was not installed on this trainer")
    if "adapter_slots" in rollout_data:
        raise ValueError("BSHD token batching requires single-adapter rollout data")
    if any(value is not None for value in rollout_data.get("multimodal_train_inputs", []) or []):
        raise ValueError("BSHD token batching currently supports text trajectories only")

    lengths = rollout_data["total_lengths"]
    if len(lengths) != len(rollout_data["tokens"]):
        raise ValueError("Token and length counts differ")
    if any(len(tokens) != length for tokens, length in zip(rollout_data["tokens"], lengths, strict=True)):
        raise ValueError("Reported sequence length differs from actual tokens")
    dp_size = parallel.effective_dp.size
    if args.global_batch_size % dp_size:
        raise ValueError("Global batch size must be divisible by the DP size")
    local_batch_size = args.global_batch_size // dp_size
    if not lengths or len(lengths) % local_batch_size:
        raise ValueError("Every optimizer step must contain a complete global batch")
    num_steps = len(lengths) // local_batch_size
    if rollout_data.get("num_rollouts", [args.global_batch_size] * num_steps) != [args.global_batch_size] * num_steps:
        raise ValueError("Upstream optimizer-step boundaries do not match the fixed global batch")
    multiple = parallel.tp.size * args.data_pad_size_multiplier
    budget = args.max_tokens_per_gpu

    # All DP ranks participate even when one shard contains an oversized row.
    # Otherwise an error on only that rank would strand its peers in the collective.
    invalid = any(length <= 0 or padded_length(length, multiple) > budget for length in lengths)
    invalid_tensor = torch.tensor([int(invalid)], dtype=torch.int, device=torch.cuda.current_device())
    group = parallel.effective_dp.group
    pg = GeneralPGUtil.create(group)
    pg.all_reduce(invalid_tensor, group, op=dist.ReduceOp.MAX)
    if invalid_tensor.item():
        raise ValueError(
            f"A DP shard contains a sequence outside the {budget}-token padded microbatch budget. "
            "No sample was truncated or dropped."
        )

    steps = [
        make_batches(lengths[start:start + local_batch_size], budget, multiple)
        for start in range(0, len(lengths), local_batch_size)
    ]
    counts = torch.tensor([len(step) for step in steps], dtype=torch.int, device=torch.cuda.current_device())
    pg.all_reduce(counts, group, op=dist.ReduceOp.MAX)
    counts = counts.tolist()
    schedule = []
    for step_index, (step, count) in enumerate(zip(steps, counts, strict=True)):
        start = step_index * local_batch_size
        schedule.extend([[start + i for i in batch] for batch in split_to_count(step, count)])

    widths = batch_widths(lengths, schedule, multiple)
    costs = [len(batch) * widths[batch[0]] for batch in schedule]
    if max(costs) > budget:
        raise RuntimeError("BSHD scheduler exceeded its padded token budget")
    rollout_data["max_seq_lens"] = widths
    # The rollout manager can precompute fixed-size batches before trainer init.
    # Replace that schedule as well so saved debug data reflects actual execution.
    rollout_data["micro_batch_indices"] = schedule
    rollout_data["num_microbatches"] = counts
    logger.info(
        "BSHD_TOKEN_BATCH %s",
        json.dumps({
            "dp_rank": parallel.effective_dp.rank,
            "samples": len(lengths),
            "global_batch_size": args.global_batch_size,
            "budget": budget,
            "microbatches_per_step": counts,
            "max_padded_tokens": max(costs),
            "total_padded_tokens": sum(costs),
            "real_tokens": sum(lengths),
            "max_samples_per_microbatch": max(map(len, schedule)),
            "max_sequence_tokens": max(lengths),
        }),
    )
    return [data.DataIterator(rollout_data, micro_batch_indices=schedule)], counts


def install(args) -> None:
    """Miles' supported trainer-init hook installs only this run's batching mode."""
    from miles.backends.megatron_utils import actor
    from miles.backends.training_utils import data

    _check_scope(args, data.get_parallel_state())
    if tuple(inspect.signature(data.get_data_iterator).parameters) != ("args", "model", "rollout_data"):
        raise RuntimeError("Miles get_data_iterator interface changed; re-review BSHD token batching")
    data.get_data_iterator = get_data_iterator
    actor.get_data_iterator = get_data_iterator
    # aggregate_forward_results uses this flag to restore log-probs to original
    # sample order after the length-based schedule. The BSHD implementation above
    # replaces the upstream iterator that otherwise assumes packed THD here.
    args.use_dynamic_batch_size = True
    logger.info("Installed BSHD token batching: budget=%d, global_batch=%d", args.max_tokens_per_gpu, args.global_batch_size)
