#!/usr/bin/env python3

import logging
import shutil
import subprocess
import sys
from pathlib import Path
import os
import socket
import subprocess
import time
import requests as http_requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

repo_root = Path(__file__).resolve().parent.parent
venv_python = repo_root / ".venv/bin/python"
venv_bin = repo_root / ".venv/bin"


def write_proxy_env(env_path=".env"):
    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "API_KEY": os.getenv("API_KEY"),
        "CURL_CA_BUNDLE": os.getenv("CURL_CA_BUNDLE"),
        "LITELLM_MASTER_KEY": os.getenv("LITELLM_MASTER_KEY"),
        "GPT41_MINI_API_URL": os.getenv("GPT41_MINI_API_URL"),
        "MODEL_NAME": "mistral-7b",
        "VLLM_BASE_URL": "http://127.0.0.1:8000/v1",
    }

    with open(env_path, "w") as f:
        for key, value in env_vars.items():
            if value is not None:
                f.write(f"{key}={value}\n")

    logger.info("Proxy environment variables written to %s", env_path)


def _make_env():
    """
    Return an env dict with the venv bin dir and the CUDA shim prepended to PATH.
    This ensures that both 'ninja' (installed via pip into the venv) and
    other venv binaries are visible to any subprocess we launch.
    """
    env = os.environ.copy()

    cuda_bin = (
        repo_root
        / ".venv/lib/python3.12/site-packages/nvidia/cu13/bin"
    )

    env["PATH"] = f"{venv_bin}:{cuda_bin}:{env.get('PATH', '')}"
    env["CUDA_HOME"] = str(cuda_bin.parent)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return env


def setup_workspace():
    logger.info("Repository root: %s", repo_root)
    logger.info("Python: %s", sys.version)
    logger.info("sys.executable=%s", sys.executable)

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "uv"])
    subprocess.check_call(["uv", "--version"])
    subprocess.check_call(["uv", "sync", "-v"])
    subprocess.check_call(["uv", "pip", "list", "--python", str(venv_python)])

    for pkg in ["vllm", "litellm[proxy]", "regex", "ninja"]:
        subprocess.check_call([
            "uv", "pip", "install", "--python", str(venv_python), pkg,
        ])

    # Verify ninja binary is importable AND on PATH via the venv
    env = _make_env()
    result = subprocess.run(
        ["ninja", "--version"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ninja binary not found on PATH after install. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    logger.info("ninja version: %s", result.stdout.strip())

    subprocess.check_call([
        str(venv_python),
        "-c",
        'import importlib.metadata; print(importlib.metadata.version("regex"))',
    ])

    logger.info("Workspace installed")


def download_model(model_path: Path):
    subprocess.check_call([
        str(venv_python),
        "-c",
        f"""
from huggingface_hub import login, snapshot_download
import os

token = os.getenv("HF_TOKEN")
if token:
    login(token=token)

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir="{model_path}",
)
print("Download complete")
""",
    ])


def start_vllm(model_path: Path):
    logger.info("Starting vLLM server...")

    env = _make_env()

    return subprocess.Popen(
        [
            str(venv_python),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model", str(model_path),
            "--host", "0.0.0.0",
            "--port", "8000",
            "--max-model-len", "8192",
            "--enforce-eager",
        ],
        env=env,
    )


def wait_for_vllm(
    url="http://127.0.0.1:8000/v1/models",
    timeout=600,
    poll_interval=5,
):
    """Wait until vLLM is serving requests."""
    logger.info("Waiting for vLLM...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            r = http_requests.get(url, timeout=10)
            if r.status_code == 200:
                logger.info("vLLM is ready")
                logger.info("Models: %s", r.json())
                return
        except Exception as e:
            logger.info("vLLM not ready yet: %s", e)
        time.sleep(poll_interval)

    raise TimeoutError(f"vLLM did not become ready within {timeout} seconds")


def start_litellm_proxy():
    logger.info("Starting LiteLLM proxy...")
    env = _make_env()

    subprocess.Popen(
        [
            "uv", "run", "litellm",
            "--config", str(repo_root / "configs/litellm_openhands_proxy.yaml"),
            "--port", "4000",
        ],
        env=env,
        cwd=repo_root,
    )


def wait_for_litellm(
    url="http://127.0.0.1:4000/v1/models",
    timeout=300,
    poll_interval=3,
):
    """Wait until LiteLLM proxy is serving requests."""
    logger.info("Waiting for LiteLLM proxy...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            r = http_requests.get(url, timeout=10)
            if r.status_code == 200:
                logger.info("LiteLLM proxy is ready")
                logger.info("Models: %s", r.json())
                return
        except Exception as e:
            logger.info("LiteLLM not ready yet: %s", e)
        time.sleep(poll_interval)

    raise TimeoutError(f"LiteLLM did not become ready within {timeout} seconds")


def test_litellm_proxy():
    logger.info("Testing LiteLLM → vLLM → Mistral path")

    payload = {
        "model": "mistral-7b",
        "messages": [{"role": "user", "content": "Reply with the word READY"}],
        "temperature": 0,
        "max_tokens": 10,
    }

    r = http_requests.post(
        "http://127.0.0.1:4000/v1/chat/completions",
        headers={"Authorization": "Bearer dummy"},
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    logger.info("LiteLLM test response: %s", r.json())
    return r.json()


def verify_imports():
    import textwrap

    code = textwrap.dedent("""
    from datasets import load_dataset
    ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
    print(len(ds))
    print(ds[0]['instance_id'])
    """)
    subprocess.check_call([str(venv_python), "-c", code])

    # FIX: uv is a standalone binary, not a Python module — call it directly
    subprocess.check_call(
        [
            "uv", "run", "python", "-c",
            "import openhands; import fastmcp; import frontmatter; print('SUCCESS')",
        ],
        cwd=repo_root,
        env=_make_env(),
    )


def main():
    write_proxy_env()
    setup_workspace()

    model_path = Path("/tmp/mistral-7b")
    download_model(model_path)

    vllm_proc = start_vllm(model_path)
    wait_for_vllm()

    start_litellm_proxy()
    wait_for_litellm()
    test_litellm_proxy()

    verify_imports()

    try:
        # run OpenHands / SWE-bench evaluation here
        pass
    finally:
        vllm_proc.terminate()


if __name__ == "__main__":
    main()