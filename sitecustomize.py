"""
Top-level sitecustomize to ensure our Modal logging patch is always applied.

Python will auto-import ``sitecustomize`` if it is importable on ``sys.path``.
During evaluation ``/workspace/benchmarks`` is on ``PYTHONPATH``, so placing
this file at the repo root guarantees the patch runs before swebench is used.
"""

import os
import sys
from pathlib import Path


def _should_apply_modal_patches() -> bool:
    """Apply heavy benchmark patches only for evaluation-related invocations."""
    if os.getenv("BENCHMARKS_FORCE_SITECUSTOMIZE") == "1":
        return True

    entrypoint = Path(sys.argv[0]).name.lower() if sys.argv else ""
    triggers = (
        "swebench-infer",
        "swebench-eval",
        "run_eval_pipeline.py",
        "run_infer.py",
    )
    return any(trigger == entrypoint for trigger in triggers)


if os.getenv("BENCHMARKS_SITECUSTOMIZE_VERBOSE") == "1":
    print("benchmarks sitecustomize imported", file=sys.stderr, flush=True)

if _should_apply_modal_patches():
    try:
        # Reuse the actual patch logic that lives alongside the benchmarks package.
        from benchmarks.utils.sitecustomize import _apply_modal_logging_patch

        _apply_modal_logging_patch()
    except Exception:
        # Avoid breaking startup for non-swebench runs; logging is best-effort.
        pass
