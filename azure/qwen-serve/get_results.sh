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

echo ""
echo "Results saved to: $LOCAL_RESULTS/"
ls -la "$LOCAL_RESULTS/"
