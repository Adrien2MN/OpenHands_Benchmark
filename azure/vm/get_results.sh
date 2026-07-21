#!/bin/bash
set -euo pipefail

# Download results from VM to local machine

RESOURCE_GROUP="token-energy-cliff"
VM_NAME="openhands-bench-gpu"
LOCAL_RESULTS="./results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOCAL_RESULTS"

echo ">>> Downloading results..."

# Get energy summary
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts 'cat /home/benchuser/results/energy_summary.json' \
  --query "value[0].message" --output tsv > "$LOCAL_RESULTS/energy_summary.json"

# Get energy log
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts 'cat /home/benchuser/results/energy_log.csv' \
  --query "value[0].message" --output tsv > "$LOCAL_RESULTS/energy_log.csv"

# Get benchmark log
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts 'cat /home/benchuser/results/benchmark.log' \
  --query "value[0].message" --output tsv > "$LOCAL_RESULTS/benchmark.log"

# Get build images log
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts 'cat /home/benchuser/results/build_images.log 2>/dev/null || echo "(build_images.log not found)"' \
  --query "value[0].message" --output tsv > "$LOCAL_RESULTS/build_images.log"

# Get vllm log
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts 'cat /home/benchuser/results/vllm.log 2>/dev/null || echo "(vllm.log not found)"' \
  --query "value[0].message" --output tsv > "$LOCAL_RESULTS/vllm.log"

# Get litellm log
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts 'cat /home/benchuser/results/litellm.log 2>/dev/null || echo "(litellm.log not found)"' \
  --query "value[0].message" --output tsv > "$LOCAL_RESULTS/litellm.log"

# Get the current run's output path from the benchmark log and transfer just that file.
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts 'python3 - <<"PY"
import json
from pathlib import Path

log_path = Path("/home/benchuser/results/benchmark.log")
if not log_path.exists():
    raise SystemExit(1)

for line in reversed(log_path.read_text(errors="ignore").splitlines()):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        continue
    output_json = payload.get("output_json")
    if output_json:
        print(output_json)
        break
PY' \
  --query "value[0].message" --output tsv > "$LOCAL_RESULTS/output_paths.txt"

mkdir -p "$LOCAL_RESULTS/swebench_outputs"
grep '^/' "$LOCAL_RESULTS/output_paths.txt" | while IFS= read -r remote_path; do
  [ -z "$remote_path" ] && continue
  fetch_path="$remote_path"
  if [[ "$fetch_path" == /app/outputs/* ]]; then
    fetch_path="/home/benchuser/results/swebench_outputs/${fetch_path#/app/outputs/}"
  fi
  # Derive a safe local filename from the path
  local_name=$(echo "$fetch_path" | sed 's|/home/benchuser/results/swebench_outputs/||' | tr '/' '__')
  echo "  Fetching $fetch_path ..."
  az vm run-command invoke \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --command-id RunShellScript \
    --scripts "base64 -w0 '$fetch_path'" \
    --query "value[0].message" --output tsv \
  | grep -v '^\[' | grep -v '^$' \
  | base64 -d > "$LOCAL_RESULTS/swebench_outputs/$local_name"
done

echo ""
echo "Results saved to: $LOCAL_RESULTS/"
ls -la "$LOCAL_RESULTS/"
