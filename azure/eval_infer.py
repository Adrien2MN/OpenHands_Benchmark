#!/usr/bin/env python3

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
import os
import subprocess
import time
import requests as http_requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

repo_root = Path(__file__).resolve().parent.parent
venv_python = repo_root / ".venv/bin/python"
venv_bin = repo_root / ".venv/bin"

# --- Config ---
MODEL_ALIAS = "mistral-7b"
MODEL_HF_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_PATH = Path("/tmp/mistral-7b")
SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Lite"
SWEBENCH_SPLIT = "test"
N_INSTANCES = 3          # number of SWE-bench instances to run
MAX_ITERATIONS = 30      # agent steps per instance (keep low for a first run)
NUM_WORKERS = 1          # parallel workers (1 = sequential, safe for single GPU)
OUTPUT_DIR = Path("/tmp/swebench_output")


def write_proxy_env(env_path=".env"):
    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "API_KEY": os.getenv("API_KEY"),
        "CURL_CA_BUNDLE": os.getenv("CURL_CA_BUNDLE"),
        "LITELLM_MASTER_KEY": os.getenv("LITELLM_MASTER_KEY"),
        "GPT41_MINI_API_URL": os.getenv("GPT41_MINI_API_URL"),
        "MODEL_NAME": MODEL_ALIAS,
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
    VLLM_USE_FLASHINFER_SAMPLER=0 avoids FlashInfer JIT compilation failure
    caused by nvcc/CCCL version mismatch in the nvidia-cu13 pip package.
    """
    env = os.environ.copy()
    cuda_bin = repo_root / ".venv/lib/python3.12/site-packages/nvidia/cu13/bin"
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

    # cwd=repo_root so uv finds pyproject.toml and the workspace members
    subprocess.check_call(["uv", "sync", "-v"], cwd=repo_root)
    subprocess.check_call(["uv", "pip", "list", "--python", str(venv_python)])

    # Install GPU-specific packages not in pyproject.toml
    for pkg in ["vllm", "litellm[proxy]", "regex"]:
        subprocess.check_call([
            "uv", "pip", "install", "--python", str(venv_python), pkg,
        ])

    # Reinstall all local SDK packages as editable AFTER vllm/litellm installs.
    # Editable installs (-e) make Python read directly from vendor/ source dirs,
    # which correctly merges the openhands.sdk namespace across packages
    # (openhands-sdk and openhands-workspace both contribute to openhands/sdk/).
    # openhands-workspace must come first so its openhands/sdk/workspace/ subtree
    # is registered before openhands-sdk imports from it.
    sdk_root = repo_root / "vendor/software-agent-sdk"
    for pkg_dir in [
        sdk_root / "openhands-workspace",
        sdk_root / "openhands-sdk",
        sdk_root / "openhands-tools",
        sdk_root / "openhands-agent-server",
    ]:
        logger.info("Installing local package: %s", pkg_dir)
        subprocess.check_call([
            "uv", "pip", "install", "--python", str(venv_python),
            "--no-deps",
            "-e", str(pkg_dir),
        ])

    subprocess.check_call([
        str(venv_python), "-c",
        'import importlib.metadata; print(importlib.metadata.version("regex"))',
    ])

    # Confirm workspace package resolves correctly
    subprocess.check_call([
        str(venv_python), "-c",
        "from openhands.sdk.workspace.base import BaseWorkspace; print('openhands.sdk.workspace OK')",
    ])

    logger.info("Workspace installed")

def download_model(MODEL_PATH: Path):
    if MODEL_PATH.exists() and any(MODEL_PATH.iterdir()):
        logger.info("Model already present at %s, skipping download", MODEL_PATH)
        return

    subprocess.check_call([
        str(venv_python), "-c",
        f"""
from huggingface_hub import login, snapshot_download
import os

token = os.getenv("HF_TOKEN")
if token:
    login(token=token)

snapshot_download(
    repo_id="{MODEL_HF_ID}",
    local_dir="{MODEL_PATH}",
)
print("Download complete")
""",
    ])


def start_vllm(MODEL_PATH: Path):
    logger.info("Starting vLLM server...")
    env = _make_env()
    return subprocess.Popen(
        [
            str(venv_python), "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(MODEL_PATH),
            "--served-model-name", MODEL_ALIAS,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--max-model-len", "8192",
            "--enforce-eager",
        ],
        env=env,
    )


def wait_for_vllm(url="http://127.0.0.1:8000/v1/models", timeout=600, poll_interval=5):
    logger.info("Waiting for vLLM...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = http_requests.get(url, timeout=10)
            if r.status_code == 200:
                logger.info("vLLM is ready — models: %s", r.json())
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


def wait_for_litellm(url="http://127.0.0.1:4000/v1/models", timeout=300, poll_interval=3):
    logger.info("Waiting for LiteLLM proxy...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = http_requests.get(url, timeout=10)
            if r.status_code == 200:
                logger.info("LiteLLM proxy is ready")
                return
        except Exception as e:
            logger.info("LiteLLM not ready yet: %s", e)
        time.sleep(poll_interval)
    raise TimeoutError(f"LiteLLM did not become ready within {timeout} seconds")


def test_litellm_proxy():
    logger.info("Testing LiteLLM → vLLM → Mistral path")
    payload = {
        "model": MODEL_ALIAS,
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


def write_llm_config(config_path: Path):
    """
    Write the LLM config JSON that swebench-infer expects.
    Uses the litellm_proxy/ prefix so OpenHands routes through LiteLLM.
    """
    config = {
        "model": f"litellm_proxy/{MODEL_ALIAS}",
        "base_url": "http://127.0.0.1:4000/v1",
        "api_key": "dummy",
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    logger.info("LLM config written to %s: %s", config_path, config)


def run_swebench_infer(llm_config_path: Path, output_dir: Path):
    """
    Run swebench-infer for the first N_INSTANCES of SWE-bench Lite.
    Results land in output_dir/output.jsonl.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Running SWE-bench inference: dataset=%s split=%s n=%d max_iter=%d workers=%d",
        SWEBENCH_DATASET, SWEBENCH_SPLIT, N_INSTANCES, MAX_ITERATIONS, NUM_WORKERS,
    )

    cmd = [
        "uv", "run", "swebench-infer",
        str(llm_config_path),
        "--dataset", SWEBENCH_DATASET,
        "--split", SWEBENCH_SPLIT,
        "--workspace", "docker",
        "--n-limit", str(N_INSTANCES),
        "--max-iterations", str(MAX_ITERATIONS),
        "--num-workers", str(NUM_WORKERS),
        "--output-dir", str(output_dir),
    ]

    logger.info("swebench-infer command: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=repo_root, env=_make_env())
    logger.info("swebench-infer complete — results in %s", output_dir)


def print_results_summary(output_dir: Path):
    """
    Parse output.jsonl and print a simple per-instance summary so you can
    see what happened without running the full eval harness.
    """
    output_file = output_dir / "output.jsonl"
    if not output_file.exists():
        logger.warning("No output.jsonl found in %s", output_dir)
        return

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    instances = []
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))

    for inst in instances:
        instance_id = inst.get("instance_id", "?")
        patch = inst.get("model_patch", "")
        status = "PATCH" if patch and patch.strip() else "NO_PATCH"
        error = inst.get("error", None)
        if error:
            status = f"ERROR: {str(error)[:80]}"
        logger.info("  [%s] %s", status, instance_id)

    patched = sum(1 for i in instances if i.get("model_patch", "").strip())
    logger.info("=" * 60)
    logger.info("Total: %d instances | %d produced a patch | %d did not",
                len(instances), patched, len(instances) - patched)
    logger.info("Full output: %s", output_file)

    # Also copy output to AzureML outputs folder if running in AzureML
    azureml_output = Path(os.getenv("AZUREML_DATAREFERENCE_eval_results", ""))
    if azureml_output and azureml_output.parent.exists():
        dest = azureml_output / "output.jsonl"
        shutil.copy(output_file, dest)
        logger.info("Copied results to AzureML output: %s", dest)


def verify_imports():
    subprocess.check_call(
        [
            "uv", "run", "python", "-c",
            """
import openhands.sdk.workspace
import importlib.metadata, pathlib
# Show where openhands-workspace is installed from
try:
    dist = importlib.metadata.distribution('openhands-workspace')
    print('openhands-workspace location:', dist.locate_file('.'))
except Exception as e:
    print('openhands-workspace not found:', e)

import openhands; import fastmcp; import frontmatter; print('imports OK')
""",
        ],
        cwd=repo_root,
        env=_make_env(),
    )


def main():
    write_proxy_env()
    setup_workspace()

    download_model(MODEL_PATH)

    vllm_proc = start_vllm(MODEL_PATH)
    wait_for_vllm()

    start_litellm_proxy()
    wait_for_litellm()
    test_litellm_proxy()

    verify_imports()

    llm_config_path = repo_root / ".llm_config/mistral-7b.json"
    write_llm_config(llm_config_path)

    try:
        run_swebench_infer(llm_config_path, OUTPUT_DIR)
        print_results_summary(OUTPUT_DIR)
    finally:
        vllm_proc.terminate()
        logger.info("vLLM process terminated")


if __name__ == "__main__":
    main()