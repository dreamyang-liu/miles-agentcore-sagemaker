"""Verify trusted Miles smoke artifacts on CPU inside the training image.

Checks the Qwen/ChatML user prompt, sampled-token fields, saved optimizer steps and
checkpoint metadata. Runtime/network callback checks remain separate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoTokenizer


def source_question(sample: dict) -> str:
    record = (sample.get("metadata") or {}).get("rft_record") or {}
    if isinstance(record.get("instance"), str):
        return record["instance"]
    for message in reversed(sample.get("prompt") or []):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def verify(run_dir: Path, rollouts: int, samples_per_rollout: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "dump_details/tokenizer", local_files_only=True)
    errors = []
    rows = []
    rollout_ids = []
    for path in sorted((run_dir / "dump_details/rollout_data").glob("*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        rid = data["rollout_id"]
        rollout_ids.append(rid)
        if len(data["samples"]) != samples_per_rollout:
            errors.append(f"rollout {rid}: unexpected sample count")
        for sample in data["samples"]:
            label = f"rollout {rid}, sample {sample['index']}"
            length = sample["response_length"]
            question = source_question(sample)
            prefix = tokenizer.decode(sample["tokens"][:-length], skip_special_tokens=False) if length else ""
            user_turns = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prefix, re.DOTALL)
            present = bool(question) and any(question in text for text in user_turns)
            mask = sample["loss_mask"]
            logprobs = sample["rollout_log_probs"]
            aligned = len(sample["tokens"]) > length > 0 and length == len(mask) == len(logprobs)
            finite = all(math.isfinite(float(value)) for value in logprobs)
            if not present:
                errors.append(f"{label}: original question missing from decoded model user turn")
            if not aligned or not finite or sum(mask) <= 0:
                errors.append(f"{label}: invalid token/mask/logprob fields")
            rows.append({
                "rollout": rid,
                "sample_index": sample["index"],
                "question_present": present,
                "source_question": question,
                "decoded_user_turns": user_turns,
                "prompt_tokens": len(sample["tokens"]) - length,
                "response_tokens": length,
                "loss_tokens": sum(mask),
                "aligned_token_fields": aligned,
                "finite_logprobs": finite,
                "reward": sample["reward"],
                "exit_status": (sample.get("metadata") or {}).get("exit_status"),
            })
    if sorted(rollout_ids) != list(range(rollouts)):
        errors.append(f"unexpected rollout ids: {rollout_ids}")

    checkpoint = run_dir / "checkpoints" / f"iter_{rollouts:07d}"
    meta = json.loads((checkpoint / "meta.json").read_text())
    if meta.get("iteration") != rollouts or meta.get("next_rollout_id") != rollouts:
        errors.append("checkpoint iteration/next_rollout_id mismatch")
    model_shards = list((checkpoint / "model").glob("*.distcp"))
    if not model_shards or not (checkpoint / "model/.metadata").is_file():
        errors.append("model checkpoint shards/metadata missing")

    reader = dcp.FileSystemReader(checkpoint / "optimizer")
    metadata = reader.read_metadata()
    steps = {
        name: torch.empty(item.size, dtype=item.properties.dtype)
        for name, item in metadata.state_dict_metadata.items()
        if name.endswith(".step") and isinstance(item, dcp.metadata.TensorStorageMetadata)
    }
    step_values = []
    if steps:
        dcp.load(steps, storage_reader=reader, planner=dcp.DefaultLoadPlanner(flatten_state_dict=False))
        step_values = sorted({float(value.item()) for value in steps.values()})
    if step_values != [float(rollouts)]:
        errors.append(f"unexpected saved optimizer steps: {step_values}")

    return {
        "passed": not errors,
        "errors": errors,
        "run_dir": str(run_dir),
        "rollout_ids": sorted(rollout_ids),
        "sample_count": len(rows),
        "question_fidelity_pass_count": sum(row["question_present"] for row in rows),
        "loss_tokens_total": sum(row["loss_tokens"] for row in rows),
        "submitted_count": sum(row["exit_status"] == "submitted" for row in rows),
        "positive_reward_count": sum(row["reward"] > 0 for row in rows),
        "optimizer_step_fields": len(steps),
        "optimizer_step_values": step_values,
        "checkpoint_meta": meta,
        "model_checkpoint_shards": len(model_shards),
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--samples-per-rollout", type=int, default=8)
    args = parser.parse_args()
    report = verify(args.run_dir, args.rollouts, args.samples_per_rollout)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
