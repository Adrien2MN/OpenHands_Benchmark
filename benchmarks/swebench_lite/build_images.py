#!/usr/bin/env python3
"""Build SWE-Bench Lite images using the SWE-Bench image pipeline."""

from __future__ import annotations

import sys

from benchmarks.swebench.build_images import main as swebench_build_images_main


def _ensure_default_dataset_args(argv: list[str]) -> list[str]:
    out = argv[:]
    if "--dataset" not in out:
        out.extend(["--dataset", "princeton-nlp/SWE-bench_Lite"])
    if "--split" not in out:
        out.extend(["--split", "test"])
    return out


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    return swebench_build_images_main(_ensure_default_dataset_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
