"""Convert a math dataset into the Miles JSONL the AgentCore recipe expects.

Each row becomes a ``messages``-shaped prompt plus the ground truth in metadata:

    {"prompt": [{"role": "user", "content": "Natalia sold clips to ..."}],
     "metadata": {"answer": "72", "instance_id": "gsm8k-train-0"}}

``prompt`` stays a message list on purpose. The TITO session server renders the first turn
itself and appends later turns incrementally, so the data must not be pre-templated -- do
not pass ``--apply-chat-template`` for this recipe.

The answer never leaves the cluster: the agent running in AgentCore only ever sees the
question, and grading happens in Miles via ``--custom-rm-path``.

Two datasets, same shape:

* ``gsm8k`` -- grade-school word problems. Small numbers, so a 4B policy solves them by
  mental arithmetic and the calculator tool is decorative; a 0.6B policy reaches ~0.79.
* ``gsm-hard`` -- the same problems with the operands replaced by large numbers (Gao et al.,
  PAL). Median operand ~3.3e6, so mental arithmetic stops being viable and the calculator
  becomes load-bearing. This is the one to use when the goal is genuine tool use.

Usage:
    python prepare_data.py --dataset gsm-hard --output-dir /root/data
    python prepare_data.py --dataset gsm8k    --output-dir /root/data --limit 2000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from datasets import load_dataset

# gsm-hard ships a single split, so a tail slice becomes the eval set.
_DEFAULT_HOLDOUT = 200


def _gsm8k_answer(row: dict) -> str | None:
    """GSM8K solutions end with '#### 72'."""
    raw = row["answer"]
    if "####" not in raw:
        return None
    return raw.rsplit("####", 1)[1].strip().replace(",", "")


def _gsm_hard_answer(row: dict) -> str | None:
    """gsm-hard targets are floats; render whole numbers without a trailing .0.

    Keep full precision otherwise: 22.9% of targets are non-integer and some carry double
    representation error (3244047.0999999996). The reward function compares with
    math.isclose, so the exact text does not have to be pretty -- but it must not be rounded,
    or a correct answer would grade as wrong.
    """
    value = float(row["target"])
    return str(int(value)) if value.is_integer() else repr(value)


@dataclass(frozen=True)
class _Spec:
    hf_path: str
    config: str | None
    question_key: str
    answer_of: Callable[[dict], str | None]
    train_split: str
    eval_split: str | None


DATASETS = {
    "gsm8k": _Spec("openai/gsm8k", "main", "question", _gsm8k_answer, "train", "test"),
    "gsm-hard": _Spec("reasoning-machines/gsm-hard", None, "input", _gsm_hard_answer, "train", None),
}


def _load(spec: _Spec, split: str):
    return (
        load_dataset(spec.hf_path, spec.config, split=split)
        if spec.config
        else load_dataset(spec.hf_path, split=split)
    )


def _write(rows, spec: _Spec, name: str, out: Path, limit: int | None) -> int:
    written = skipped = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for index, row in rows:
            if limit is not None and written >= limit:
                break
            answer = spec.answer_of(row)
            if answer is None:
                skipped += 1
                continue
            record = {
                "prompt": [{"role": "user", "content": str(row[spec.question_key]).strip()}],
                "metadata": {"answer": answer, "instance_id": f"{name}-{index}"},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} rows to {out}" + (f" ({skipped} skipped)" if skipped else ""))
    return written


def convert(dataset: str, output_dir: Path, limit: int | None, holdout: int) -> None:
    spec = DATASETS[dataset]
    train = list(enumerate(_load(spec, spec.train_split)))

    if spec.eval_split:
        evaluation = list(enumerate(_load(spec, spec.eval_split)))
    else:
        # No native eval split: hold out the tail so train and eval never overlap.
        train, evaluation = train[:-holdout], train[-holdout:]

    _write(train, spec, f"{dataset}-train", output_dir / f"{dataset}_train.jsonl", limit)
    _write(evaluation, spec, f"{dataset}-eval", output_dir / f"{dataset}_eval.jsonl", holdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="gsm-hard", choices=sorted(DATASETS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Cap the training rows.")
    parser.add_argument("--holdout", type=int, default=_DEFAULT_HOLDOUT)
    args = parser.parse_args()
    convert(args.dataset, args.output_dir, args.limit, args.holdout)


if __name__ == "__main__":
    main()
