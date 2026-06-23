#!/bin/bash
set -euo pipefail

# ============================================================
# Kick off the full experiment on the GPU VM (no SSH needed).
# Uses az vm run-command to orchestrate everything remotely.
# ============================================================

RESOURCE_GROUP="token-energy-cliff"
VM_NAME="openhands-bench-gpu"
REGISTRY_NAME="diffusionregistry"
IMAGE_NAME="openhands-bench-full"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
N_LIMIT="${N_LIMIT:-5}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30}"

echo "============================================"
echo "Running experiment on VM: $VM_NAME"
echo "Model: $MODEL_ID"
echo "Instances: $N_LIMIT"
echo "Max iterations: $MAX_ITERATIONS"
echo "============================================"

# Run the experiment container on the VM
# - Mount Docker socket (for SWE-bench Docker workspace)
# - Mount results volume
# - Use host network (so container can use Docker)
echo ">>> Starting experiment container..."
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts "
    set -e
    mkdir -p /home/benchuser/results

    docker run --rm \
      --gpus all \
      --network host \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v /home/benchuser/results:/results \
      -v /home/benchuser/hf_cache:/root/.cache/huggingface \
      -e MODEL_ID='${MODEL_ID}' \
      -e HF_TOKEN='${HF_TOKEN:-}' \
      -e N_LIMIT=${N_LIMIT} \
      -e MAX_ITERATIONS=${MAX_ITERATIONS} \
      ${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:latest

    echo 'Experiment finished. Results in /home/benchuser/results/'
  "

echo ""
echo "============================================"
echo "Experiment launched!"
echo "Check status:  ./check_status.sh"
echo "Get results:   ./get_results.sh"
echo "============================================"
