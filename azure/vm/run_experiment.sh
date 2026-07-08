#!/bin/bash
set -euo pipefail

# ============================================================
# Kick off experiment on the GPU VM.
#
# For local open-source models (Mistral, Qwen, Llama...):
#   MODEL_SOURCE=local MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3 \
#     LLM_CONFIG=litellm-mistral7b.json ./run_experiment.sh
#
# For proprietary models via LiteLLM proxy (GPT, Claude...):
#   MODEL_SOURCE=proxy LLM_CONFIG=litellm-gpt-4-1-mini.json \
#     ./run_experiment.sh
# ============================================================

RESOURCE_GROUP="${RESOURCE_GROUP:-token-energy-cliff}"
VM_NAME="${VM_NAME:-openhands-bench-gpu}"
REGISTRY_NAME="${REGISTRY_NAME:-diffusionregistry}"
IMAGE_NAME="${IMAGE_NAME:-openhands-bench-full}"

MODEL_SOURCE="${MODEL_SOURCE:-local}"
MODEL_ID="${MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.3}"
LLM_CONFIG="${LLM_CONFIG:-litellm-mistral7b.json}"
N_LIMIT="${N_LIMIT:-20}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
HF_TOKEN="${HF_TOKEN:-}"

echo "============================================"
echo "VM:             $VM_NAME"
echo "MODEL_SOURCE:   $MODEL_SOURCE"
echo "MODEL_ID:       $MODEL_ID"
echo "LLM_CONFIG:     $LLM_CONFIG"
echo "N instances:    $N_LIMIT"
echo "Max iterations: $MAX_ITERATIONS"
echo "============================================"

az vm run-command invoke \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --command-id RunShellScript \
    --scripts "
        set -e
        rm -rf /home/benchuser/results
        mkdir -p /home/benchuser/results /home/benchuser/hf_cache

        docker rm -f openhands-bench 2>/dev/null || true
        docker pull ${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:latest

        docker run -d --name openhands-bench \
            --gpus all \
            --network host \
            -v /var/run/docker.sock:/var/run/docker.sock \
            -v /home/benchuser/results:/results \
            -v /home/benchuser/hf_cache:/root/.cache/huggingface/bench-model \
            -e MODEL_SOURCE=${MODEL_SOURCE} \
            -e MODEL_ID=${MODEL_ID} \
            -e LLM_CONFIG=${LLM_CONFIG} \
            -e HF_TOKEN=${HF_TOKEN} \
            -e N_LIMIT=${N_LIMIT} \
            -e MAX_ITERATIONS=${MAX_ITERATIONS} \
            -e RESULTS_DIR=/results \
            ${REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:latest

        docker ps --filter name=openhands-bench
    "

echo ""
echo "============================================"
echo "Launched! Monitor with:"
echo "  ssh benchuser@<VM_IP> 'docker logs --follow openhands-bench'"
echo "  ./check_status.sh"
echo "============================================"