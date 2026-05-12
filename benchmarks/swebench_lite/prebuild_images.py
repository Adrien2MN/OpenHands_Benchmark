#!/usr/bin/env python3
"""
Pre-build all unique base Docker images for SWE-Bench-Lite before inference.

This script significantly speeds up inference by building all ~35-40 unique base images
upfront in parallel, rather than having each instance trigger individual builds.

Expected speedup: 3-5x faster inference after pre-building.

Usage:
    uv run benchmarks/swebench_lite/prebuild_images.py [--dataset DATASET] [--split SPLIT]

Example:
    # Build all Lite images
    uv run benchmarks/swebench_lite/prebuild_images.py

    # Build just 50 instances for testing
    uv run benchmarks/swebench_lite/prebuild_images.py --eval-limit 50
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

from benchmarks.swebench.build_images import (
    collect_unique_base_images,
    extract_custom_tag,
)
from benchmarks.swebench.constants import TargetType
from benchmarks.utils.build_utils import build_image
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from openhands.sdk import get_logger


logger = get_logger(__name__)


def _check_docker_ready() -> str | None:
    """Return an actionable error when Docker cannot be reached."""
    try:
        import docker  # type: ignore
    except Exception as exc:
        return (
            "Docker Python SDK is unavailable in this environment. "
            f"Install or repair the OpenHands bench dependencies and retry. Details: {exc}"
        )

    try:
        client = docker.from_env()
        client.ping()
        return None
    except Exception as exc:
        return (
            "Docker daemon is not reachable. Start Docker Desktop (or your Docker engine), "
            "confirm the socket is available, then rerun the Lite prebuild. "
            f"Details: {exc}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-build all unique base images for SWE-Bench-Lite."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="Dataset name (default: princeton-nlp/SWE-bench_Lite)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=300,
        help="Number of instances to scan for unique images (default: 300 for Lite)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel builds (default: 4 - don't go too high to avoid Docker daemon contention)",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["binary", "binary-minimal", "source", "source-minimal"],
        default="source-minimal",
        help="Build target type (default: source-minimal)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if images already exist",
    )
    return parser.parse_args()


def build_single_image(
    base_image: str,
    custom_tag: str,
    target: TargetType = "source-minimal",
    force: bool = False,
) -> tuple[str, bool, str]:
    """
    Build a single image.

    Args:
        base_image: Full base image tag (e.g., docker.io/swebench/sweb.eval.x86_64.django_1776_django-11333:v1)
        custom_tag: Custom tag extracted from base image
        target: Build target (binary, source, source-minimal)
        force: Force rebuild

    Returns:
        tuple: (base_image, success, error_or_tag)
    """
    try:
        logger.info(f"Building image from {base_image}...")

        output = build_image(
            base_image=base_image,
            target_image=EVAL_AGENT_SERVER_IMAGE,
            custom_tag=custom_tag,
            target=target,
            push=False,
            force_build=force,
            full_tag_prefix=None,
        )

        if output.error:
            logger.error(f"✗ Failed to build {custom_tag}: {output.error}")
            return base_image, False, output.error
        else:
            logger.info(f"✓ Built {custom_tag}: {output.tags}")
            return base_image, True, output.tags[0] if output.tags else "unknown"

    except Exception as e:
        logger.error(f"✗ Exception building {custom_tag}: {e}")
        return base_image, False, str(e)


def main() -> int:
    args = parse_args()

    docker_error = _check_docker_ready()
    if docker_error:
        logger.error(docker_error)
        return 1

    print(
        """
╔════════════════════════════════════════════════════════════════════╗
║     SWE-Bench-Lite: Pre-build Base Images                         ║
║     This will build ~35-40 unique base images in parallel          ║
║     Expected time: 30-60 minutes (one-time, saves ~10+ hours)     ║
╚════════════════════════════════════════════════════════════════════╝
"""
    )

    logger.info(f"Dataset: {args.dataset} (split: {args.split})")
    logger.info(f"Scanning up to {args.eval_limit} instances for unique base images...")

    # Collect all unique base images
    try:
        unique_images = collect_unique_base_images(
            dataset=args.dataset,
            split=args.split,
            n_limit=args.eval_limit,
        )
    except Exception as e:
        logger.error(f"Failed to collect unique images: {e}")
        return 1

    logger.info(f"\n✓ Found {len(unique_images)} unique base images\n")

    if not unique_images:
        logger.warning("No base images found. Exiting.")
        return 1

    # Build in parallel
    print(f"Starting parallel builds with {args.workers} workers...\n")
    start_time = time.time()

    results: dict[str, tuple[bool, str]] = {}
    failed_count = 0
    success_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for base_image in unique_images:
            custom_tag = extract_custom_tag(base_image)
            future = executor.submit(
                build_single_image,
                base_image=base_image,
                custom_tag=custom_tag,
                target=cast(TargetType, args.target),
                force=args.force,
            )
            futures[future] = base_image

        # Process results as they complete
        for future in as_completed(futures):
            base_image, success, output = future.result()
            results[base_image] = (success, output)

            if success:
                success_count += 1
                logger.info(f"[{success_count}/{len(unique_images)}] ✓ {output}")
            else:
                failed_count += 1
                logger.warning(
                    f"[{success_count + failed_count}/{len(unique_images)}] ✗ {output}"
                )

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'=' * 70}")
    print("Build Summary:")
    print(f"{'=' * 70}")
    print(f"Total images: {len(unique_images)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Time elapsed: {elapsed / 60:.1f} minutes")
    if success_count > 0:
        print(f"Average time per image: {elapsed / success_count:.1f} seconds")

    if failed_count > 0:
        print(f"\n⚠️  {failed_count} images failed to build:")
        for base_image, (success, output) in results.items():
            if not success:
                print(f"  - {base_image}: {output}")

    print(f"\n{'=' * 70}")

    if success_count == len(unique_images):
        print(
            f"✓ All {len(unique_images)} images pre-built successfully!\n"
            f"You can now run inference ~3-5x faster:\n"
            f"  uv run benchmarks/swebench/run_infer.py --config-name swebench_lite"
        )
        return 0
    elif success_count > 0:
        print(
            f"⚠️  {success_count}/{len(unique_images)} images built successfully.\n"
            f"Inference will proceed but will rebuild missing images (slower)."
        )
        return 0
    else:
        print("✗ No images built successfully. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
