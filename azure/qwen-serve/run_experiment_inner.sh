#!/bin/bash
set -euo pipefail

# ============================================================
# Inner script: runs INSIDE the container on the GPU VM.
# Starts vLLM, measures energy, runs benchmark, saves results.
# ============================================================

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
RESULTS_DIR="/results"
ENERGY_LOG="${RESULTS_DIR}/energy_log.csv"
BENCHMARK_OUTPUT="${RESULTS_DIR}/benchmark_output"
mkdir -p "$RESULTS_DIR" "$BENCHMARK_OUTPUT"

echo "============================================"
echo "Starting end-to-end experiment"
echo "Model: $MODEL_ID"
echo "Results: $RESULTS_DIR"
echo "============================================"

# --- 0. Download model weights (if not already cached) ---
echo ">>> Downloading model weights..."
if [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN" 2>/dev/null || true
fi
python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"
echo ">>> Model downloaded."

# --- 1. Start GPU power monitoring in background ---
echo ">>> Starting GPU energy monitoring..."
(
  echo "timestamp,power_draw_W,gpu_util_pct,mem_used_MiB,temperature_C" > "$ENERGY_LOG"
  while true; do
    nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used,temperature.gpu \
      --format=csv,noheader,nounits >> "$ENERGY_LOG" 2>/dev/null
    sleep 1
  done
) &
ENERGY_PID=$!

# --- 2. Start vLLM server ---
echo ">>> Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 \
  --model "$MODEL_ID" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --trust-remote-code &
VLLM_PID=$!

# Wait for vLLM to be ready
echo ">>> Waiting for vLLM to load model..."
for i in $(seq 1 120); do
  if curl -s http://localhost:8000/v1/models | grep -q "$MODEL_ID" 2>/dev/null; then
    echo "vLLM ready after ${i}s"
    break
  fi
  if [ $i -eq 120 ]; then
    echo "ERROR: vLLM failed to start within 120s"
    kill $ENERGY_PID 2>/dev/null
    exit 1
  fi
  sleep 1
done

# --- 3. Record baseline (idle) power for 10s ---
echo ">>> Recording idle baseline (10s)..."
sleep 10

# --- 4. Run the benchmark ---
echo ">>> Running SWE-bench benchmark..."
EXPERIMENT_START=$(date +%s)

# Point benchmark at local vLLM
cat > /tmp/llm_config.json << EOF
{
  "model": "openai/${MODEL_ID}",
  "base_url": "http://localhost:8000/v1",
  "api_key": "dummy",
  "max_output_tokens": 8192,
  "native_tool_calling": true
}
EOF

# Run inference — uses host Docker socket mounted into this container
swebench-infer /tmp/llm_config.json \
  --workspace docker \
  --n-limit "${N_LIMIT:-5}" \
  --max-iterations "${MAX_ITERATIONS:-30}" \
  --output-dir "$BENCHMARK_OUTPUT" \
  2>&1 | tee "${RESULTS_DIR}/benchmark.log"

EXPERIMENT_END=$(date +%s)
DURATION=$((EXPERIMENT_END - EXPERIMENT_START))

# --- 5. Stop monitoring ---
kill $ENERGY_PID 2>/dev/null || true
kill $VLLM_PID 2>/dev/null || true

# --- 6. Compute energy summary ---
echo ">>> Computing energy summary..."
python3 << 'PYEOF'
import csv
import json
from pathlib import Path

log = Path("/results/energy_log.csv")
rows = list(csv.DictReader(log.open()))

if rows:
    powers = [float(r["power_draw_W"]) for r in rows if r["power_draw_W"].strip()]
    duration_s = len(powers)  # 1 sample/sec
    avg_power_w = sum(powers) / len(powers) if powers else 0
    total_energy_wh = (avg_power_w * duration_s) / 3600
    max_power_w = max(powers) if powers else 0

    summary = {
        "duration_seconds": duration_s,
        "avg_gpu_power_watts": round(avg_power_w, 2),
        "max_gpu_power_watts": round(max_power_w, 2),
        "total_gpu_energy_wh": round(total_energy_wh, 4),
        "total_gpu_energy_kwh": round(total_energy_wh / 1000, 6),
        "samples": len(powers),
    }
else:
    summary = {"error": "No power data collected"}

Path("/results/energy_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PYEOF

echo ""
echo "============================================"
echo "EXPERIMENT COMPLETE"
echo "Duration: ${DURATION}s"
echo "Results in: $RESULTS_DIR"
echo "  - energy_log.csv      (1Hz GPU power readings)"
echo "  - energy_summary.json (aggregated stats)"
echo "  - benchmark_output/   (SWE-bench results)"
echo "  - benchmark.log       (full output log)"
echo "============================================"
