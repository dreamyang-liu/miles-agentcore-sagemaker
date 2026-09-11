"""Convert a SageMaker-RFT prompts parquet into the jsonl this recipe trains on.

The RFT input format is one string column ``prompt`` holding a JSON record::

    {"instance_id": "gsm8k_train_0", "data_source": "gsm8k",
     "instance": "<problem text>", "prompt": [{"role": "user", "content": "..."}],
     "reward_spec": {"ground_truth": "72"}, "extra_info": {...}}

Miles' launcher reads ``<data_dir>/<name>_train.jsonl`` (and ``<name>_eval.jsonl`` when eval is
on) with ``--input-key prompt --metadata-key metadata``. Each output line is::

    {"prompt": <the record's messages>,
     "metadata": {"answer": <reward_spec.ground_truth>, "instance_id": <instance_id>,
                  "rft_record": <the whole original record>}}

``answer``/``instance_id`` keep ``math_reward.py`` and the existing agent function working
unchanged; ``rft_record`` lets an RFT-contract agent function hand the agent exactly the
JSON string it expects. The RFT path forwards the original record, including
ground truth, to the external agent; Miles still computes its own training reward.

Usage:
    python convert_rft_parquet.py training_prompts.parquet --name rft-gsm8k --output-dir /tmp/data [--holdout 128]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def convert(parquet: Path, name: str, output_dir: Path, holdout: int) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for raw in pq.read_table(parquet).column("prompt").to_pylist():
        record = json.loads(raw)
        ground_truth = (record.get("reward_spec") or {}).get("ground_truth")
        messages = record.get("prompt")
        if ground_truth is None or not messages:
            continue
        records.append(
            {
                "prompt": messages,
                "metadata": {
                    "answer": str(ground_truth),
                    "instance_id": record.get("instance_id"),
                    "rft_record": record,
                },
            }
        )
    eval_rows, train_rows = records[:holdout], records[holdout:]
    for split, rows in (("train", train_rows), ("eval", eval_rows)):
        with (output_dir / f"{name}_{split}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(train_rows), len(eval_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--name", default="rft-gsm8k", help="dataset name: files are <name>_train.jsonl / <name>_eval.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=128, help="rows taken from the front for the eval split")
    args = parser.parse_args()
    train, eval_ = convert(args.parquet, args.name, args.output_dir, args.holdout)
    print(f"{args.name}: {train} train rows, {eval_} eval rows -> {args.output_dir}")


if __name__ == "__main__":
    main()
