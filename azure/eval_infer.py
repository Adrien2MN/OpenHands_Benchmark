#!/usr/bin/env python3

import logging
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


def setup_workspace():
    repo_root = Path(__file__).resolve().parent.parent

    logger.info("Repository root: %s", repo_root)
    logger.info("Python: %s", sys.version)

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

    subprocess.check_call(
        ["uv", "sync"],
        cwd=repo_root,
    )

    subprocess.check_call([
    "uv",
    "pip",
    "install",
    "vllm"
])
    subprocess.check_call([
    "uv",
    "pip",
    "install",
    "litellm[proxy]"
    ])

    logger.info("Workspace installed")


def download_model(model_path: Path):
    subprocess.check_call([
        "uv",
        "run",
        "python",
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
"""
    ])


def wait_for_port(port: int, timeout: int = 300):
    start = time.time()

    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(2)

    raise TimeoutError(f"Port {port} did not open")


def start_vllm(model_path: Path):
    logger.info("Starting vLLM server...")

    return subprocess.Popen([
        "uv",
        "run",
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ])


def verify_imports():
    repo_root = Path(__file__).resolve().parent.parent
    import textwrap

    code = textwrap.dedent("""
    from datasets import load_dataset

    ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
    print(len(ds))
    print(ds[0]['instance_id'])
    """)

    subprocess.check_call([
        "uv",
        "run",
        "python",
        "-c",
        code,
    ])
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
            """
        ]

    subprocess.check_call(cmd, cwd=repo_root)


def write_proxy_env(env_path=".env"):
    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "API_KEY": os.getenv("API_KEY"),
        "CURL_CA_BUNDLE": os.getenv("CURL_CA_BUNDLE"),
        "LITELLM_MASTER_KEY": os.getenv("LITELLM_MASTER_KEY"),
        "GPT41_MINI_API_URL": os.getenv("GPT41_MINI_API_URL"),
        "MODEL_NAME": "mistral-7b",
        "VLLM_BASE_URL": "http://127.0.0.1:4000/v1",
    }

    with open(env_path, "w") as f:
        for key, value in env_vars.items():
            if value is not None:
                f.write(f"{key}={value}\n")


def wait_for_vllm(
    url="http://127.0.0.1:8000/v1/models",
    timeout=600,
    poll_interval=5,
):
    """
    Wait until vLLM is serving requests.
    """

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

    raise TimeoutError(
        f"vLLM did not become ready within {timeout} seconds"
    )


def wait_for_litellm(
    url="http://127.0.0.1:4000/v1/models",
    timeout=300,
    poll_interval=3,
):
    """
    Wait until LiteLLM proxy is serving requests.
    """

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

    raise TimeoutError(
        f"LiteLLM did not become ready within {timeout} seconds"
    )


def test_litellm_proxy():
    logger.info("Testing LiteLLM → vLLM → Mistral path")

    payload = {
        "model": "mistral-7b",
        "messages": [
            {
                "role": "user",
                "content": "Reply with the word READY",
            }
        ],
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


def main():
    write_proxy_env()
    setup_workspace()

    model_path = Path("/tmp/mistral-7b")
    download_model(model_path)
    
    vllm_proc = start_vllm(model_path)

    wait_for_vllm()

    subprocess.Popen(
    [
        "make",
        "run-litellm-proxy",
        "PROXY_ENV_FILE=.env",
    ]
    )

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