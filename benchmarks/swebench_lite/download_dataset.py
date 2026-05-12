#!/usr/bin/env python3
"""
Download SWE-Bench-Lite dataset and save locally.

This is optional - the Lite dataset can be streamed from Hugging Face.
Downloading locally is useful if you want to:
1. Inspect instances before running
2. Work offline
3. Create subsets for testing

Usage:
    uv run benchmarks/swebench_lite/download_dataset.py [--output PATH] [--limit N]

Example:
    # Download all 300 Lite instances
    uv run benchmarks/swebench_lite/download_dataset.py --output data/swebench_lite.jsonl

    # Download just first 20 for testing
    uv run benchmarks/swebench_lite/download_dataset.py --output data/swebench_lite_20.jsonl --limit 20
"""

import argparse
import json
import sys
from collections.abc import Sized
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from openhands.sdk import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SWE-Bench-Lite dataset and save locally."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/swebench_lite_instances.jsonl",
        help="Output path for JSONL file (default: data/swebench_lite_instances.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N instances (default: all 300)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading SWE-Bench-Lite dataset...")
    logger.info(f"Output: {output_path}")

    try:
        # Load Lite dataset from Hugging Face
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        if not isinstance(dataset, Sized):
            raise TypeError(
                "Loaded dataset does not support len(); disable streaming or convert to a list first."
            )
        total_instances = len(dataset)
        logger.info(f"✓ Loaded {total_instances} instances from Hugging Face")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return 1

    # Apply limit if specified
    if args.limit:
        limit = min(args.limit, total_instances)
        instances = list(islice(dataset, limit))
        logger.info(f"Limited to first {len(instances)} instances")
    else:
        instances = list(dataset)

    # Save as JSONL
    try:
        with open(output_path, "w") as f:
            for i, instance in enumerate(instances):
                f.write(json.dumps(instance) + "\n")

                if (i + 1) % 50 == 0:
                    logger.info(f"  Saved {i + 1}/{len(instances)} instances...")

        logger.info(f"\n✓ Saved {len(instances)} instances to {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        return 0

    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
