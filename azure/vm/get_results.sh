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

# Get swebench output jsonl files (skip old deepseek outputs from previous runs)
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts 'find /home/benchuser/results/swebench_outputs -name "output.jsonl" 2>/dev/null | while read f; do echo "=== $f ==="; cat "$f"; done || echo "(no swebench_outputs found)"' \
  --query "value[0].message" --output tsv > "$LOCAL_RESULTS/output.jsonl"

# Also copy the full swebench_outputs directory listing
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts 'find /home/benchuser/results/swebench_outputs -name "output.jsonl" 2>/dev/null' \
  --query "value[0].message" --output tsv > "$LOCAL_RESULTS/output_paths.txt"

echo ""
echo "Results saved to: $LOCAL_RESULTS/"
ls -la "$LOCAL_RESULTS/"
