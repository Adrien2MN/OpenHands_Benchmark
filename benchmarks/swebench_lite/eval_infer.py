#!/usr/bin/env python3
"""Evaluate SWE-Bench Lite outputs with SWE-Bench evaluator defaults."""

from __future__ import annotations

import sys

from benchmarks.swebench.eval_infer import main as swebench_eval_main


def _ensure_default_dataset_args(argv: list[str]) -> list[str]:
    if "--dataset" not in argv:
        argv.extend(["--dataset", "princeton-nlp/SWE-bench_Lite"])
    if "--split" not in argv:
        argv.extend(["--split", "test"])
    return argv


def main() -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = _ensure_default_dataset_args(sys.argv[:])
        swebench_eval_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
