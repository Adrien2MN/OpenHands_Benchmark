"""
Top-level sitecustomize to ensure our Modal logging patch is always applied.

Python will auto-import ``sitecustomize`` if it is importable on ``sys.path``.
During evaluation ``/workspace/benchmarks`` is on ``PYTHONPATH``, so placing
this file at the repo root guarantees the patch runs before swebench is used.
"""

import os
import sys


def _is_swebench_context() -> bool:
    """Only activate benchmark startup patches for SWE-bench invocations."""
    cmdline = " ".join(sys.argv).lower()
    swebench_markers = (
        "swebench",
        "run_evaluation",
        "run_infer",
        "benchmarks.swebench",
    )
    return any(marker in cmdline for marker in swebench_markers)


if os.getenv("BENCHMARKS_SITECUSTOMIZE_DEBUG") == "1":
    print("benchmarks sitecustomize imported", file=sys.stderr, flush=True)

if _is_swebench_context():
    try:
        # Reuse the actual patch logic that lives alongside the benchmarks package.
        from benchmarks.utils.sitecustomize import _apply_modal_logging_patch

        _apply_modal_logging_patch()
    except Exception:
        # Avoid breaking startup for non-swebench runs; logging is best-effort.
        pass
