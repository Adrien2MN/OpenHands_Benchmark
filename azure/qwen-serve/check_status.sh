#!/bin/bash
set -euo pipefail

# Check experiment status and get results from the VM

RESOURCE_GROUP="token-energy-cliff"
VM_NAME="openhands-bench-gpu"

echo ">>> Checking GPU status..."
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts '
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=name,power.draw,utilization.gpu,memory.used,memory.total --format=csv
    echo ""
    echo "=== Running containers ==="
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
    echo ""
    echo "=== Results files ==="
    ls -la /home/benchuser/results/ 2>/dev/null || echo "No results yet"
    echo ""
    echo "=== Energy summary ==="
    cat /home/benchuser/results/energy_summary.json 2>/dev/null || echo "Not yet available"
  '
