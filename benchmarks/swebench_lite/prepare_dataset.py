#!/usr/bin/env python3
"""Materialize SWE-bench Lite instances locally in JSONL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.utils.dataset import get_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export SWE-bench Lite split to local JSONL (same schema as HF rows)."
        )
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "dev"],
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=300,
        help="Number of instances to export (use 100 or 300).",
    )
    parser.add_argument(
        "--output",
        default="data/swebench_lite/swebench_lite_test_300.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    df = get_dataset(
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split=args.split,
        eval_limit=args.n_limit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "rows": int(len(df)),
                "dataset": "princeton-nlp/SWE-bench_Lite",
                "split": args.split,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
