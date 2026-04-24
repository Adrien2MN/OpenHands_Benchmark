#!/usr/bin/env python3
"""
Unified script to run SWE-bench infer and eval pipeline.

Usage:
    python run_eval_pipeline.py --model gpt-4.1-mini [--instances 1] [--max-iterations 100]
    python run_eval_pipeline.py --model mistral-small-2503 --instances 5
    python run_eval_pipeline.py --model o1 --instances 1 --max-iterations 200
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


AVAILABLE_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4.1",
    "gpt-5.4",
    "mistral-small-2503",
    "mistral-large-3",
    "deepseek-v3.1",
    "deepseek-v3-1",
    "kimi-k2.6",
    "claude-opus-4-7",
    "o1",
]

DEFAULT_LLM_CONFIG = ".llm_config/litellm.json"
DATASET = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"
SCRIPT_DIR = Path(__file__).resolve().parent


def _slug(value: str) -> str:
    """Create a filesystem-friendly name without over-normalizing the model id."""
    return value.replace("/", "__").replace(" ", "_").replace("\t", "_").strip("._-")


def get_llm_config_path(model: str) -> str:
    """Get LLM config for the model."""
    config_dir = Path(".llm_config").resolve()  # Use absolute path

    # Try different naming patterns for existing configs
    config_patterns = [
        config_dir / f"litellm-{model}.json",
        config_dir
        / f"litellm-{model.replace('.', '-')}.json",  # Convert dots to hyphens
        config_dir / f"{model}.json",
    ]

    for config_path in config_patterns:
        if config_path.exists():
            print(f"✓ Using config: {config_path}")
            return str(config_path)

    # If no existing config found, list available configs
    if config_dir.exists():
        available = list(config_dir.glob("*.json"))
        if available:
            print(f"Error: Config not found for model '{model}'")
            print(f"Available configs in {config_dir}:")
            for cfg in sorted(available):
                print(f"  - {cfg.name}")
            sys.exit(1)

    print(f"Error: No LLM configs found in {config_dir}")
    print(f"Please create config files in {config_dir}/ first")
    sys.exit(1)


def run_infer(
    llm_config: str,
    instances: int,
    max_iterations: int,
    model: str,
    force_build: bool,
) -> str:
    """Run swebench-infer and return output directory."""
    kwargs_note = f"{model}_n{instances}_iter{max_iterations}"

    cmd = [
        "uv",
        "run",
        "swebench-infer",
        llm_config,
        "--dataset",
        DATASET,
        "--split",
        SPLIT,
        "--max-iterations",
        str(max_iterations),
        "--workspace",
        "docker",
        "--n-limit",
        str(instances),
        "--note",
        kwargs_note,
    ]

    print(f"\n{'=' * 70}")
    print(f"Running infer with {model}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")

    run_started = time.time()
    env = os.environ.copy()
    if force_build:
        # Build per-instance workspace images locally to avoid missing prebuilt tags.
        env["FORCE_BUILD"] = "1"

    result = subprocess.run(cmd, cwd=SCRIPT_DIR, env=env)
    if result.returncode != 0:
        print(f"Error: Infer failed with exit code {result.returncode}")
        sys.exit(1)

    # Find the output directory created by this run
    output_dir = SCRIPT_DIR / "outputs"
    if not output_dir.exists():
        print("Error: No output directory found")
        sys.exit(1)

    # Look for the newest output file produced during this run.
    matching_files = sorted(
        [
            p
            for p in output_dir.rglob("output.jsonl")
            if p.stat().st_mtime >= run_started - 5
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not matching_files:
        print("Error: Could not find output.jsonl for the completed run")
        sys.exit(1)

    output_jsonl = matching_files[0]

    print("\n✓ Infer completed successfully")
    print(f"  Output file: {output_jsonl}")
    return str(output_jsonl)


def get_stable_run_dir(output_jsonl: str, model: str) -> Path:
    """Build the stable output directory path for a completed run."""
    output_path = Path(output_jsonl).resolve()
    output_dir = output_path.parent

    instance_ids: list[str] = []
    with open(output_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            instance_id = data.get("instance_id")
            if instance_id:
                instance_ids.append(str(instance_id))

    if instance_ids:
        instance_part = "__".join(
            _slug(instance_id) for instance_id in sorted(set(instance_ids))
        )
    else:
        instance_part = "no_instances"

    stable_name = f"{_slug(model)}__{instance_part}"
    return output_dir.parent / stable_name


def _collect_instance_ids(run_dir: Path, output_jsonl: str) -> list[str]:
    """Collect instance ids for a run from output JSONL with fallbacks."""
    instance_ids: set[str] = set()

    output_path = Path(output_jsonl)
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instance_id = data.get("instance_id")
                if isinstance(instance_id, str) and instance_id.strip():
                    instance_ids.add(instance_id.strip())

    conversations_dir = run_dir / "conversations"
    if conversations_dir.exists():
        for archive in conversations_dir.glob("*.tar.gz"):
            name = archive.name.removesuffix(".tar.gz")
            if name:
                instance_ids.add(name)

    logs_dir = run_dir / "logs"
    if logs_dir.exists():
        for log_file in logs_dir.glob("instance_*.log"):
            match = re.fullmatch(r"instance_(.+?)(?:\.output)?\.log", log_file.name)
            if match:
                instance_ids.add(match.group(1))

    return sorted(instance_ids)


def _organize_multi_instance_outputs(model_dir: Path, instance_ids: list[str]) -> None:
    """Move per-instance artifacts into model/<instance_id>/... folders."""
    logs_dir = model_dir / "logs"
    conv_dir = model_dir / "conversations"
    report_dir = model_dir / "report"

    for instance_id in instance_ids:
        instance_dir = model_dir / _slug(instance_id)
        (instance_dir / "logs").mkdir(parents=True, exist_ok=True)
        (instance_dir / "conversations").mkdir(parents=True, exist_ok=True)
        (instance_dir / "report").mkdir(parents=True, exist_ok=True)

        if logs_dir.exists():
            for suffix in (".log", ".output.log"):
                src = logs_dir / f"instance_{instance_id}{suffix}"
                if src.exists():
                    shutil.move(str(src), str(instance_dir / "logs" / src.name))

        if conv_dir.exists():
            src = conv_dir / f"{instance_id}.tar.gz"
            if src.exists():
                shutil.move(str(src), str(instance_dir / "conversations" / src.name))

        if report_dir.exists():
            src = report_dir / f"report_{instance_id}.json"
            if src.exists():
                shutil.move(str(src), str(instance_dir / "report" / src.name))

        if logs_dir.exists():
            run_eval_root = logs_dir / "run_evaluation"
            if run_eval_root.exists():
                for match in run_eval_root.glob(f"**/{instance_id}"):
                    if match.is_dir():
                        dest = instance_dir / "logs" / "run_evaluation" / match.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.move(str(match), str(dest))

    # Remove empty containers after moves; keep non-empty as shared artifacts.
    for folder in (conv_dir, report_dir, logs_dir):
        if folder.exists() and not any(folder.iterdir()):
            folder.rmdir()


def finalize_run_directory(output_jsonl: str, model: str) -> Path:
    """Finalize output layout.

    - Single instance: keep legacy model__instance folder naming.
    - Multiple instances: create outputs/<dataset>/<model>/ with per-instance
      folders and shared run-level artifacts in the model root.
    """
    current_dir = Path(output_jsonl).resolve().parent
    instance_ids = _collect_instance_ids(current_dir, output_jsonl)

    if len(instance_ids) <= 1:
        stable_dir = get_stable_run_dir(output_jsonl, model)

        if current_dir == stable_dir:
            return stable_dir

        if stable_dir.exists():
            shutil.rmtree(stable_dir)

        current_dir.rename(stable_dir)
        return stable_dir

    model_dir = current_dir.parent / _slug(model)

    if current_dir != model_dir:
        if model_dir.exists():
            shutil.rmtree(model_dir)
        current_dir.rename(model_dir)

    _organize_multi_instance_outputs(model_dir, instance_ids)
    return model_dir


def run_eval(output_jsonl: str, model: str) -> None:
    """Run swebench-eval on the output file."""
    # Extract run ID from output directory for better reporting
    output_path = Path(output_jsonl)
    run_id = output_path.parent.name

    cmd = [
        "uv",
        "run",
        "swebench-eval",
        output_jsonl,
        "--dataset",
        DATASET,
        "--split",
        SPLIT,
        "--run-id",
        run_id,
        "--no-modal",
        "--timeout",
        "1800",
    ]

    print(f"\n{'=' * 70}")
    print("Running eval on infer output")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"Error: Eval failed with exit code {result.returncode}")
        sys.exit(1)

    print("\n✓ Eval completed successfully")

    # Show report location
    report_path = output_path.parent / "report" / "report.json"
    if report_path.exists():
        print(f"  Report: {report_path}")
        with open(report_path) as f:
            report = json.load(f)
            if "total_instances" in report:
                print(f"  Total instances: {report['total_instances']}")
            if "resolved_instances" in report:
                print(f"  Resolved: {report['resolved_instances']}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench infer and eval pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python run_eval_pipeline.py --model gpt-4.1-mini --instances 1
  python run_eval_pipeline.py --model mistral-small-2503 --instances 5 --max-iterations 100
  python run_eval_pipeline.py --model o1 --instances 1 --max-iterations 200

Available models: {", ".join(AVAILABLE_MODELS)}
        """,
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=AVAILABLE_MODELS,
        help="Model to use for inference",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=1,
        help="Number of instances to evaluate (default: 1)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Max iterations per instance (default: 100)",
    )
    parser.add_argument(
        "--infer-only",
        action="store_true",
        help="Only run infer, skip eval",
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        metavar="OUTPUT_JSONL",
        help="Only run eval on an existing output.jsonl file",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        default=False,
        help=(
            "Force local docker image builds for SWE-bench instances to avoid missing "
            "prebuilt image tags"
        ),
    )
    parser.add_argument(
        "--no-force-build",
        action="store_false",
        dest="force_build",
        help="Disable local image force-build and use existing/prebuilt images",
    )

    args = parser.parse_args()

    # Handle eval-only mode
    if args.eval_only:
        if not Path(args.eval_only).exists():
            print(f"Error: File not found: {args.eval_only}")
            sys.exit(1)
        run_eval(args.eval_only, args.model)
        return

    # Standard pipeline: infer then eval
    llm_config = get_llm_config_path(args.model)
    output_jsonl = run_infer(
        llm_config,
        args.instances,
        args.max_iterations,
        args.model,
        args.force_build,
    )

    if not args.infer_only:
        run_eval(output_jsonl, args.model)

    stable_dir = finalize_run_directory(output_jsonl, args.model)
    stable_output_jsonl = stable_dir / "output.jsonl"

    if stable_output_jsonl.exists():
        print(f"\n✓ Stable output folder: {stable_dir}")
        print(f"  Output file: {stable_output_jsonl}")
    else:
        print(f"\n✓ Infer complete. Output: {output_jsonl}")
        print(f"  To run eval: python run_eval_pipeline.py --eval-only {output_jsonl}")


if __name__ == "__main__":
    main()
