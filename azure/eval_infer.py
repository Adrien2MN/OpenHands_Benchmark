#!/usr/bin/env python3

import logging
import subprocess
import sys
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_workspace():
    repo_root = Path(__file__).resolve().parent.parent

    logger.info("Repository root: %s", repo_root)
    logger.info("Python: %s", sys.version)

    logger.info("=== REPO TREE ===")

    for root, dirs, files in os.walk(repo_root / "vendor"):
        logger.info(root)
        if root.count("/") > 15:  # optional depth limit
            continue

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        "pip",
        "uv",
    ])

    logger.info("=== Checking OpenHands workspace dirs ===")

    for d in [
        "vendor/software-agent-sdk",
        "vendor/software-agent-sdk/openhands-sdk",
        "vendor/software-agent-sdk/openhands-tools",
        "vendor/software-agent-sdk/openhands-agent-server",
        "vendor/software-agent-sdk/openhands-workspace",
    ]:
        path = repo_root / d
        logger.info("%s: %s", d, path.exists())

    logger.info("=== vendor contents ===")
    vendor = repo_root / "vendor"
    if vendor.exists():
        for item in vendor.iterdir():
            logger.info("%s", item)

    subprocess.check_call(
        ["uv", "sync"],
        cwd=repo_root,
    )

    logger.info("Workspace installed")


def verify_imports():
    repo_root = Path(__file__).resolve().parent.parent

    cmd = [
        "uv",
        "run",
        "python",
        "-c",
        """
import openhands
import fastmcp
import frontmatter

print("SUCCESS")
print("openhands:", openhands.__file__)
"""
    ]

    subprocess.check_call(cmd, cwd=repo_root)


def main():
    setup_workspace()
    verify_imports()


if __name__ == "__main__":
    main()