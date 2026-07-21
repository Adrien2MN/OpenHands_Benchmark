#!/usr/bin/env python3
"""
Parse SWE-bench output.jsonl files and summarise patch results.

Usage:
    python3 parse_results.py <results_dir>
    python3 parse_results.py results_20260713_172628

Looks for *.jsonl files under <results_dir>/swebench_outputs/ and prints
a per-instance table plus a summary.
"""

import json
import sys
from pathlib import Path


def load_instances(path: Path) -> list[dict]:
    instances = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "instance_id" in obj:
                    instances.append(obj)
            except json.JSONDecodeError:
                continue
    return instances


def summarise(instances: list[dict]) -> None:
    if not instances:
        print("  (no instance records found)")
        return

    resolved = [i for i in instances if i.get("test_result", {}).get("resolved")]
    unresolved = [i for i in instances if not i.get("test_result", {}).get("resolved")]
    patched = [
        i for i in instances if i.get("test_result", {}).get("git_patch", "").strip()
    ]

    print(f"  Total:      {len(instances)}")
    print(f"  Resolved:   {len(resolved)}  ({100 * len(resolved) // len(instances)}%)")
    print(f"  Unresolved: {len(unresolved)}")
    print(f"  Has patch:  {len(patched)}")
    print()

    col = 40
    header = f"  {'instance_id':<{col}} {'resolved':<10} {'patch':<8} {'iters'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for inst in sorted(instances, key=lambda x: x["instance_id"]):
        tr = inst.get("test_result", {})
        res = tr.get("resolved", None)
        patch = bool(tr.get("git_patch", "").strip())
        iters = tr.get("iteration_count", "?")
        res_str = "YES" if res else ("NO" if res is False else "?")
        print(
            f"  {inst['instance_id']:<{col}} {res_str:<10} {'yes' if patch else 'no':<8} {iters}"
        )


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_results.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    swebench_dir = results_dir / "swebench_outputs"

    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    jsonl_files = sorted(swebench_dir.rglob("*.jsonl")) if swebench_dir.exists() else []

    if not jsonl_files:
        # fallback: look for output.jsonl directly in results_dir
        jsonl_files = sorted(results_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No .jsonl files found under {results_dir}")
        sys.exit(1)

    all_instances = []
    for f in jsonl_files:
        instances = load_instances(f)
        if not instances:
            continue
        print(f"\n=== {f.name} ({len(instances)} instances) ===")
        summarise(instances)
        all_instances.extend(instances)

    if len(jsonl_files) > 1 and all_instances:
        print("\n=== TOTAL ACROSS ALL FILES ===")
        summarise(all_instances)


if __name__ == "__main__":
    main()
