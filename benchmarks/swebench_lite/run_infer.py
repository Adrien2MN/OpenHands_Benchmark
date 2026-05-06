#!/usr/bin/env python3
"""Run SWE-Bench inference with SWE-bench Lite defaults."""

from __future__ import annotations

import sys

from benchmarks.swebench.run_infer import main as swebench_main


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
        swebench_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
