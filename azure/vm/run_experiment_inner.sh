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

mkdir -p "$RESULTS_DIR"

# Force the correct in-container CA bundle regardless of what may have
# been set upstream (env-file, .env baked in some other way, etc.).
export SSL_CERT_FILE=/app/azure/vm/axa_combined_ca.pem
export REQUESTS_CA_BUNDLE=/app/azure/vm/axa_combined_ca.pem

MODEL_SOURCE="${MODEL_SOURCE:-local}"
MODEL_ID="${MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.3}"
LLM_CONFIG="${LLM_CONFIG:-litellm-mistral7b.json}"
N_LIMIT="${N_LIMIT:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
CONVERSATION_TIMEOUT="${CONVERSATION_TIMEOUT:-3600}"
RESULTS_DIR="${RESULTS_DIR:-/results}"
REPO_ROOT="/app"
MODEL_CACHE="/root/.cache/huggingface/bench-model"
PINNED_SDK_COMMIT="3e0a3a0915b369c7e2057c77722e98585855d30a"

export CONVERSATION_TIMEOUT

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "MODEL_SOURCE:   $MODEL_SOURCE"
echo "MODEL_ID:       $MODEL_ID"
echo "LLM_CONFIG:     $LLM_CONFIG"
echo "N instances:    $N_LIMIT"
echo "Max iterations: $MAX_ITERATIONS"
echo "Conv timeout:   $CONVERSATION_TIMEOUT"
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
for pkg in "litellm[proxy]" regex ninja "opentelemetry-semantic-conventions>=0.50b0"; do
    uv pip install --python .venv/bin/python "$pkg"
done

# vllm only needed for local models — installed last so uv sync can't remove it
if [ "$MODEL_SOURCE" = "local" ]; then
    uv pip install --python .venv/bin/python "vllm==0.8.5" "transformers==4.51.1"
    # vllm pins opentelemetry-api<1.27.0 but lmnr requires 1.39.1.
    # Use --no-deps to force-override without re-solving vllm's constraints.
    uv pip install --python .venv/bin/python --no-deps \
        "opentelemetry-api==1.39.1" \
        "opentelemetry-sdk==1.39.1" \
        "opentelemetry-proto==1.39.1" \
        "opentelemetry-exporter-otlp-proto-common==1.39.1" \
        "opentelemetry-exporter-otlp-proto-grpc==1.39.1" \
        "opentelemetry-exporter-otlp-proto-http==1.39.1" \
        "opentelemetry-instrumentation==0.60b1" \
        "opentelemetry-semantic-conventions==0.60b1"
fi
echo ">>> Dependencies installed"

# Ensure the benchmark writes only fresh outputs for this run.
rm -rf "$REPO_ROOT/outputs"

# --- 2. Record system info ---
echo ">>> Recording system info..."
nvidia-smi --query-gpu=name,memory.total,driver_version,power.max_limit --format=csv > "$RESULTS_DIR/gpu_info.csv"
echo "kernel: $(uname -r)" > "$RESULTS_DIR/system_info.txt"
echo "arch: $(uname -m)" >> "$RESULTS_DIR/system_info.txt"
echo "vllm: $(pip show vllm 2>/dev/null | grep Version || echo unknown)" >> "$RESULTS_DIR/system_info.txt"
echo "model_id: $MODEL_ID" >> "$RESULTS_DIR/system_info.txt"
echo "llm_config: $LLM_CONFIG" >> "$RESULTS_DIR/system_info.txt"
echo "n_limit: $N_LIMIT" >> "$RESULTS_DIR/system_info.txt"
echo "max_iterations: $MAX_ITERATIONS" >> "$RESULTS_DIR/system_info.txt"
echo "start_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RESULTS_DIR/system_info.txt"

# --- 3. Start GPU energy monitoring ---
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

# --- 3b. Start periodic results sync (crash protection) ---
(
    while true; do
        sleep 120
        # Sync any outputs written so far
        if [ -d "$REPO_ROOT/outputs" ]; then
            cp -r "$REPO_ROOT/outputs" "$RESULTS_DIR/swebench_outputs" 2>/dev/null || true
        fi
        # Extract vLLM throughput stats
        grep -E "Added request|throughput|Avg prompt" "$RESULTS_DIR/vllm.log" > "$RESULTS_DIR/vllm_stats.log" 2>/dev/null || true
    done
) &
SYNC_PID=$!

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

    SERVED_NAME=$(python3 -c "import json; m=json.load(open('$REPO_ROOT/.llm_config/$LLM_CONFIG'))['model']; print(m.split('/')[-1] if '/' in m else m)")

    # Model-specific vLLM flags
    VLLM_EXTRA_ARGS=""
    if echo "$MODEL_ID" | grep -qi "mistral"; then
        VLLM_EXTRA_ARGS=""
    elif echo "$MODEL_ID" | grep -qi "qwen"; then
        VLLM_EXTRA_ARGS=""
    fi

    .venv/bin/python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_CACHE" \
        --served-model-name "$SERVED_NAME" \
        --host 0.0.0.0 \
        --port 8000 \
        --gpu-memory-utilization 0.96 \
        --max-model-len 32768 \
        --dtype half \
        $VLLM_EXTRA_ARGS \
        >> "$RESULTS_DIR/vllm.log" 2>&1 &
    VLLM_PID=$!

    echo ">>> Waiting for vLLM (up to 600s)..."
    for i in $(seq 1 120); do
        if curl -s http://127.0.0.1:8000/v1/models | grep -q "$SERVED_NAME" 2>/dev/null; then
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

# --- 4. Start LiteLLM proxy (proxy models only) ---
# For local models, the agent-server calls vLLM directly via 172.17.0.1:8000.
# LiteLLM is only needed for proxy (cloud API) models.
LITELLM_PID=""
if [ "$MODEL_SOURCE" = "proxy" ]; then
    echo ">>> Starting LiteLLM proxy..."
    export PATH=".venv/bin:$PATH"
    unset SSL_CERT_FILE
    export SSL_CERT_FILE=/app/azure/vm/axa_combined_ca.pem
    .venv/bin/litellm \
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
else
    echo ">>> MODEL_SOURCE=local — skipping LiteLLM, agent-server calls vLLM directly"
fi

# Smoke test — verify end-to-end routing
MODEL_NAME=$(python3 -c "import json; m=json.load(open('$REPO_ROOT/.llm_config/$LLM_CONFIG'))['model']; print(m.split('/')[-1] if '/' in m else m)")
if [ "$MODEL_SOURCE" = "local" ]; then
    SMOKE_URL="http://127.0.0.1:8000/v1/chat/completions"
else
    SMOKE_URL="http://127.0.0.1:4000/v1/chat/completions"
fi
echo ">>> Smoke test: calling $MODEL_NAME at $SMOKE_URL ..."
RESPONSE=$(curl -s --max-time 60 -X POST "$SMOKE_URL" \
    -H "Authorization: Bearer dummy" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word READY\"}],\"max_tokens\":10}" || echo '{"error":"curl failed"}')
echo "Raw response: $RESPONSE"
SMOKE_OK=$(echo "$RESPONSE" | python3 -c "
import sys, json
r = json.load(sys.stdin)
content = r.get('choices', [{}])[0].get('message', {}).get('content', '')
if content:
    print('Smoke test OK:', content)
    sys.exit(0)
else:
    print('Smoke test FAILED — no choices. Full response:', r)
    sys.exit(1)
" 2>&1)
echo "$SMOKE_OK"
if echo "$SMOKE_OK" | grep -q "FAILED"; then
    echo "ERROR: Smoke test failed — aborting before benchmark run"
    [ -f "$RESULTS_DIR/litellm.log" ] && tail -50 "$RESULTS_DIR/litellm.log" || true
    [ -f "$RESULTS_DIR/vllm.log" ] && tail -50 "$RESULTS_DIR/vllm.log" || true
    kill $ENERGY_PID ${VLLM_PID:-} ${LITELLM_PID:-} 2>/dev/null || true
    exit 1
fi

# --- 5. Pre-build SWE-bench agent-server images ---
# Prune all unused Docker images/containers/cache first to free disk space.
echo ">>> Pruning unused Docker resources..."
docker system prune -af --volumes 2>&1 | tail -3

# Must run before infer so ensure_local_image finds them and doesn't try
# to rebuild them one-by-one inside Docker-in-Docker during inference.
echo ">>> Pre-building SWE-bench agent-server images (n=$N_LIMIT)..."
.venv/bin/swebench-lite-build-images \
    --n-limit "$N_LIMIT" \
    2>&1 | tee "$RESULTS_DIR/build_images.log"
echo ">>> Image build complete"

# Record phase transition timestamps for energy splitting
echo "$(date +%s)" > "$RESULTS_DIR/ts_inference_start.txt"

# --- 6. Run SWE-bench ---
echo ">>> Running SWE-bench Lite (n=$N_LIMIT, max_iter=$MAX_ITERATIONS)..."
EXPERIMENT_START=$(date +%s)

.venv/bin/swebench-lite-infer \
    "$REPO_ROOT/.llm_config/$LLM_CONFIG" \
    --n-limit "$N_LIMIT" \
    --num-workers 1 \
    --workspace docker \
    --max-iterations "$MAX_ITERATIONS" \
    2>&1 | tee "$RESULTS_DIR/benchmark.log"

EXPERIMENT_END=$(date +%s)
DURATION=$((EXPERIMENT_END - EXPERIMENT_START))
echo "end_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RESULTS_DIR/system_info.txt"
echo "duration_seconds: $DURATION" >> "$RESULTS_DIR/system_info.txt"

# --- 7. Cleanup ---
kill $ENERGY_PID ${VLLM_PID:-} ${LITELLM_PID:-} ${SYNC_PID:-} 2>/dev/null || true

[ -d "$REPO_ROOT/outputs" ] && cp -r "$REPO_ROOT/outputs" "$RESULTS_DIR/swebench_outputs" || true

# Extract vLLM throughput and token stats
grep -E "Added request|Avg prompt throughput" "$RESULTS_DIR/vllm.log" > "$RESULTS_DIR/vllm_stats.log" 2>/dev/null || true

# Extract per-request token counts from vLLM log
python3 << 'TOKEOF'
import re, json, os
from pathlib import Path
results_dir = Path(os.environ.get("RESULTS_DIR", "/results"))
vllm_log = results_dir / "vllm.log"
if not vllm_log.exists():
    exit()
token_stats = []
for line in vllm_log.open(errors="ignore"):
    m = re.search(r"(\d{2}:\d{2}:\d{2}).*Received request (chatcmpl-\w+):.*prompt_token_ids: (?:None|\[.+?\])", line)
    if not m:
        continue
# Simpler: count requests and extract throughput lines
throughput = []
for line in vllm_log.open(errors="ignore"):
    m = re.search(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s.*GPU KV cache usage: ([\d.]+)%", line)
    if m:
        throughput.append({
            "prompt_tps": float(m.group(1)),
            "gen_tps": float(m.group(2)),
            "kv_cache_pct": float(m.group(3)),
        })
total_requests = sum(1 for l in vllm_log.open(errors="ignore") if "Added request" in l)
summary = {
    "total_requests": total_requests,
    "throughput_samples": len(throughput),
    "avg_prompt_tps": round(sum(t["prompt_tps"] for t in throughput) / max(len(throughput), 1), 1),
    "avg_gen_tps": round(sum(t["gen_tps"] for t in throughput) / max(len(throughput), 1), 1),
    "avg_kv_cache_pct": round(sum(t["kv_cache_pct"] for t in throughput) / max(len(throughput), 1), 1),
}
(results_dir / "vllm_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
TOKEOF

# --- 8. Energy analysis (per-phase + per-instance) ---
python3 << 'PYEOF'
import csv, json, os, re
from pathlib import Path
from datetime import datetime

results_dir = Path(os.environ.get("RESULTS_DIR", "/results"))
log = results_dir / "energy_log.csv"

# Load energy samples
rows = list(csv.DictReader(log.open()))
if not rows:
    (results_dir / "energy_summary.json").write_text(json.dumps({"error": "No power data"}))
    exit()

# Parse energy data with timestamps
energy_data = []
for r in rows:
    ts_str = r.get("timestamp", "").strip()
    pw = r.get("power_draw_W", "").strip()
    if ts_str and pw:
        try:
            ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")
        except ValueError:
            try:
                ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
            except ValueError:
                continue
        energy_data.append({"ts": ts, "power": float(pw)})

powers = [e["power"] for e in energy_data]
total_avg = sum(powers) / len(powers)

# Phase split: read inference start timestamp
ts_file = results_dir / "ts_inference_start.txt"
inference_start_epoch = None
if ts_file.exists():
    inference_start_epoch = int(ts_file.read_text().strip())

setup_energy = []
inference_energy = []
if inference_start_epoch and energy_data:
    inference_start = datetime.fromtimestamp(inference_start_epoch)
    for e in energy_data:
        if e["ts"] < inference_start:
            setup_energy.append(e["power"])
        else:
            inference_energy.append(e["power"])

# Per-instance timing from benchmark.log
benchmark_log = results_dir / "benchmark.log"
instance_times = []
if benchmark_log.exists():
    content = benchmark_log.read_text()
    starts = re.findall(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - === Evaluation Started \(instance (.+?)\) ===",
        content
    )
    for i, (ts_str, iid) in enumerate(starts):
        start = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        # End is start of next instance, or end of energy data
        if i + 1 < len(starts):
            end = datetime.strptime(starts[i+1][0], "%Y-%m-%d %H:%M:%S")
        else:
            end = energy_data[-1]["ts"] if energy_data else start
        # Sum energy samples in this window
        inst_powers = [e["power"] for e in energy_data if start <= e["ts"] < end]
        inst_energy_wh = sum(inst_powers) / 3600 if inst_powers else 0
        instance_times.append({
            "instance_id": iid,
            "start": ts_str,
            "duration_seconds": len(inst_powers),
            "avg_power_watts": round(sum(inst_powers) / max(len(inst_powers), 1), 2),
            "energy_wh": round(inst_energy_wh, 4),
        })

# Build summary
summary = {
    "total": {
        "duration_seconds": len(powers),
        "avg_gpu_power_watts": round(total_avg, 2),
        "max_gpu_power_watts": round(max(powers), 2),
        "total_gpu_energy_wh": round(total_avg * len(powers) / 3600, 4),
        "samples": len(powers),
    },
    "phases": {
        "setup": {
            "duration_seconds": len(setup_energy),
            "avg_gpu_power_watts": round(sum(setup_energy) / max(len(setup_energy), 1), 2),
            "energy_wh": round(sum(setup_energy) / 3600, 4),
        } if setup_energy else None,
        "inference": {
            "duration_seconds": len(inference_energy),
            "avg_gpu_power_watts": round(sum(inference_energy) / max(len(inference_energy), 1), 2),
            "energy_wh": round(sum(inference_energy) / 3600, 4),
        } if inference_energy else None,
    },
    "per_instance": instance_times,
}

(results_dir / "energy_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["total"], indent=2))
print(f"\nSetup:     {summary['phases']['setup']['energy_wh']:.1f} Wh ({summary['phases']['setup']['duration_seconds']}s)" if summary['phases']['setup'] else "")
print(f"Inference: {summary['phases']['inference']['energy_wh']:.1f} Wh ({summary['phases']['inference']['duration_seconds']}s)" if summary['phases']['inference'] else "")
print(f"Per-instance: {len(instance_times)} entries")
PYEOF

echo "============================================"
echo "COMPLETE — Duration: ${DURATION}s"
ls -lh "$RESULTS_DIR/"
echo "============================================"