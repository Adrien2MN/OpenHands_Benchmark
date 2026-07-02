#!/bin/bash
set -euo pipefail

# ============================================================
# Inner script: runs INSIDE the container on the GPU VM.
#
# Two modes controlled by MODEL_SOURCE env var:
#
#   MODEL_SOURCE=local  (default)
#     Downloads MODEL_ID from HuggingFace, serves via vLLM,
#     routes through LiteLLM proxy. For open models (Mistral,
#     Qwen, Llama...).
#     Example:
#       -e MODEL_SOURCE=local
#       -e MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
#       -e LLM_CONFIG=litellm-mistral7b.json
#
#   MODEL_SOURCE=proxy
#     Skips download and vLLM. LiteLLM routes to an external
#     API already configured in the proxy yaml. For proprietary
#     models (GPT-4.1, Claude, Mistral-Large...).
#     Example:
#       -e MODEL_SOURCE=proxy
#       -e LLM_CONFIG=litellm-gpt-4-1-mini.json
#
# Other env vars:
#   N_LIMIT          number of SWE-bench instances (default: 1)
#   MAX_ITERATIONS   agent iterations per instance (default: 30)
#   HF_TOKEN         HuggingFace token (needed for gated models)
#   RESULTS_DIR      where to write outputs (default: /results)
# ============================================================

MODEL_SOURCE="${MODEL_SOURCE:-local}"
MODEL_ID="${MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.3}"
LLM_CONFIG="${LLM_CONFIG:-litellm-mistral7b.json}"
N_LIMIT="${N_LIMIT:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
RESULTS_DIR="${RESULTS_DIR:-/results}"
REPO_ROOT="/app"
MODEL_CACHE="/root/.cache/huggingface/bench-model"
PINNED_SDK_COMMIT="3e0a3a0915b369c7e2057c77722e98585855d30a"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "MODEL_SOURCE:   $MODEL_SOURCE"
echo "MODEL_ID:       $MODEL_ID"
echo "LLM_CONFIG:     $LLM_CONFIG"
echo "N instances:    $N_LIMIT"
echo "Max iterations: $MAX_ITERATIONS"
echo "Results dir:    $RESULTS_DIR"
echo "============================================"

# --- 0. Verify Docker socket ---
echo ">>> Checking Docker socket..."
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker socket not accessible."
    exit 1
fi
echo "Docker OK: $(docker info --format '{{.ServerVersion}}')"

# --- 1. Install Python deps ---
echo ">>> Installing Python dependencies..."
cd "$REPO_ROOT"

if ! command -v uv &>/dev/null; then
    pip install -U uv -q
fi

# Clone SDK fresh
SDK_ROOT="$REPO_ROOT/vendor/software-agent-sdk"
if [ -d "$SDK_ROOT" ]; then
    rm -rf "$SDK_ROOT"
fi
git clone https://github.com/OpenHands/software-agent-sdk.git "$SDK_ROOT"
git -C "$SDK_ROOT" checkout "$PINNED_SDK_COMMIT"

for pkg in openhands-sdk openhands-workspace openhands-tools openhands-agent-server; do
    count=$(find "$SDK_ROOT/$pkg/openhands" -name "*.py" 2>/dev/null | wc -l)
    echo "  $pkg: $count .py files"
    [ "$count" -gt 0 ] || { echo "ERROR: $pkg missing source"; exit 1; }
done

uv sync -v

# Install packages not in pyproject.toml AFTER the final uv sync,
# so uv sync doesn't wipe them out.
for pkg in "litellm[proxy]" regex ninja; do
    uv pip install --python .venv/bin/python "$pkg"
done

# vllm only needed for local models — installed last so uv sync can't remove it
if [ "$MODEL_SOURCE" = "local" ]; then
    uv pip install --python .venv/bin/python "vllm==0.8.5" "transformers==4.51.1"
fi
echo ">>> Dependencies installed"

# --- 2. Start GPU energy monitoring ---
echo ">>> Starting GPU energy monitoring..."
ENERGY_LOG="$RESULTS_DIR/energy_log.csv"
echo "timestamp,power_draw_W,gpu_util_pct,mem_used_MiB,temperature_C" > "$ENERGY_LOG"
(
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,power.draw,utilization.gpu,memory.used,temperature.gpu \
            --format=csv,noheader,nounits >> "$ENERGY_LOG" 2>/dev/null
        sleep 1
    done
) &
ENERGY_PID=$!

VLLM_PID=""

if [ "$MODEL_SOURCE" = "local" ]; then
    # --- 3a. Download model weights ---
    echo ">>> Downloading model: $MODEL_ID ..."
    if [ -d "$MODEL_CACHE" ] && [ "$(ls -A "$MODEL_CACHE" 2>/dev/null)" ]; then
        echo "Model already cached, skipping download"
    else
        mkdir -p "$MODEL_CACHE"
        .venv/bin/python -c "
from huggingface_hub import login, snapshot_download
import os
token = os.getenv('HF_TOKEN')
if token:
    login(token=token)
snapshot_download(repo_id='${MODEL_ID}', local_dir='${MODEL_CACHE}')
print('Download complete')
"
    fi

    # --- 3b. Start vLLM ---
    echo ">>> Starting vLLM server for $MODEL_ID ..."
    export PATH=".venv/bin:$PATH"
    export VLLM_USE_FLASHINFER_SAMPLER=0

    .venv/bin/python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_CACHE" \
        --served-model-name "mistral-7b" \
        --host 0.0.0.0 \
        --port 8000 \
        --gpu-memory-utilization 0.95 \
        --max-model-len 1024 \
        --dtype half \
        >> "$RESULTS_DIR/vllm.log" 2>&1 &
    VLLM_PID=$!

    echo ">>> Waiting for vLLM (up to 600s)..."
    for i in $(seq 1 120); do
        if curl -s http://127.0.0.1:8000/v1/models | grep -q "mistral-7b" 2>/dev/null; then
            echo "vLLM ready after $((i * 5))s"
            break
        fi
        if [ "$i" -eq 120 ]; then
            echo "ERROR: vLLM failed to start"
            tail -30 "$RESULTS_DIR/vllm.log"
            kill $ENERGY_PID 2>/dev/null || true
            exit 1
        fi
        sleep 5
    done
else
    echo ">>> MODEL_SOURCE=proxy — skipping download and vLLM"
fi

# --- 4. Start LiteLLM proxy ---
echo ">>> Starting LiteLLM proxy..."
export PATH=".venv/bin:$PATH"

unset SSL_CERT_FILE
export SSL_CERT_FILE=/app/azure/vm/axa_combined_ca.pem
uv run litellm \
    --config "$REPO_ROOT/configs/litellm_openhands_proxy.yaml" \
    --port 4000 \
    >> "$RESULTS_DIR/litellm.log" 2>&1 &
LITELLM_PID=$!

echo ">>> Waiting for LiteLLM (up to 120s)..."
for i in $(seq 1 40); do
    if curl -s http://127.0.0.1:4000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('data') else 1)" 2>/dev/null; then
        echo "LiteLLM ready after $((i * 3))s"
        break
    fi
    if [ "$i" -eq 40 ]; then
        echo "ERROR: LiteLLM failed to start"
        tail -30 "$RESULTS_DIR/litellm.log"
        kill $ENERGY_PID ${VLLM_PID:-} 2>/dev/null || true
        exit 1
    fi
    sleep 3
done

# Smoke test — use proxy alias (strip provider prefix like openai/ from model name)
MODEL_NAME=$(python3 -c "import json; m=json.load(open('$REPO_ROOT/.llm_config/$LLM_CONFIG'))['model']; print(m.split('/')[-1] if '/' in m else m)")
echo ">>> Smoke test: calling $MODEL_NAME..."
RESPONSE=$(curl -s --max-time 30 -X POST http://127.0.0.1:4000/v1/chat/completions \
    -H "Authorization: Bearer dummy" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with READY\"}],\"max_tokens\":5}" || echo '{"error":"curl failed"}')
echo "Raw response: $RESPONSE"
echo "$RESPONSE" | python3 -c "import sys,json; r=json.load(sys.stdin); print('Smoke test OK:', r.get('choices',[{}])[0].get('message',{}).get('content','NO CHOICES - full response: '+str(r)))" || true

# --- 5. Run SWE-bench ---
echo ">>> Running SWE-bench Lite (n=$N_LIMIT, max_iter=$MAX_ITERATIONS)..."
EXPERIMENT_START=$(date +%s)

uv run swebench-lite-infer \
    "$REPO_ROOT/.llm_config/$LLM_CONFIG" \
    --n-limit "$N_LIMIT" \
    --num-workers 1 \
    --workspace docker \
    2>&1 | tee "$RESULTS_DIR/benchmark.log"

EXPERIMENT_END=$(date +%s)
DURATION=$((EXPERIMENT_END - EXPERIMENT_START))

# --- 6. Cleanup ---
kill $ENERGY_PID ${VLLM_PID:-} ${LITELLM_PID:-} 2>/dev/null || true

[ -d "$REPO_ROOT/outputs" ] && cp -r "$REPO_ROOT/outputs" "$RESULTS_DIR/swebench_outputs" || true

# --- 7. Energy summary ---
python3 << 'PYEOF'
import csv, json, os
from pathlib import Path
results_dir = os.environ.get("RESULTS_DIR", "/results")
log = Path(results_dir) / "energy_log.csv"
rows = list(csv.DictReader(log.open()))
if rows:
    powers = [float(r["power_draw_W"]) for r in rows if r.get("power_draw_W","").strip()]
    avg = sum(powers)/len(powers) if powers else 0
    summary = {
        "duration_seconds": len(powers),
        "avg_gpu_power_watts": round(avg, 2),
        "max_gpu_power_watts": round(max(powers), 2) if powers else 0,
        "total_gpu_energy_wh": round(avg * len(powers) / 3600, 4),
        "samples": len(powers),
    }
else:
    summary = {"error": "No power data"}
Path(results_dir + "/energy_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PYEOF

echo "============================================"
echo "COMPLETE — Duration: ${DURATION}s"
ls -lh "$RESULTS_DIR/"
echo "============================================"