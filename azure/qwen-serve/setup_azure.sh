#!/bin/bash
set -euo pipefail

# ============================================================
# Azure setup for serving Qwen via vLLM on ACI (GPU)
# Same pattern as diff-vs-llm: ACR build + container deploy
# No VM, no SSH — just a running container with a public API.
# ============================================================

# --- Configuration (adjust as needed) ---
RESOURCE_GROUP="openhands-bench"
LOCATION="swedencentral"          # Has GPU ACI availability (V100/T4)
REGISTRY_NAME="openhandsbenchacr"
IMAGE_NAME="qwen-vllm-serve"
CONTAINER_NAME="qwen-serve"

# Model configuration
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"  # Change for larger models
GPU_SKU="V100"                    # V100 (16GB) or T4; check az availability
GPU_COUNT=1                       # 1 for 7B, 2+ for larger models
CPU=4
MEMORY=16                         # GB RAM

# HuggingFace token (for gated models only; Qwen2.5 is open)
HF_TOKEN="${HF_TOKEN:-}"

echo "============================================"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location:       $LOCATION"
echo "Registry:       $REGISTRY_NAME"
echo "Model:          $MODEL_ID"
echo "GPU:            ${GPU_COUNT}x ${GPU_SKU}"
echo "============================================"

# 1. Login to Azure
echo ">>> Logging into Azure..."
az login --use-device-code

# 2. Create Resource Group
echo ">>> Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

# 3. Create ACR with admin enabled (belt-and-suspenders for auth)
echo ">>> Creating container registry..."
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$REGISTRY_NAME" \
  --sku Basic \
  --admin-enabled true

# 4. Build image in ACR (builds in the cloud, no local Docker needed)
echo ">>> Building image in ACR (this downloads model weights — may take 10-20 min)..."
az acr build \
  --registry "$REGISTRY_NAME" \
  --image "${IMAGE_NAME}:latest" \
  --build-arg "MODEL_ID=${MODEL_ID}" \
  --build-arg "HF_TOKEN=${HF_TOKEN}" \
  --timeout 3600 \
  .

# 5. Get ACR credentials (same approach as diff-vs-llm)
ACR_SERVER="${REGISTRY_NAME}.azurecr.io"
echo ">>> Retrieving ACR admin credentials (Access Keys)..."
ACR_PASSWORD=$(az acr credential show --name "$REGISTRY_NAME" --query "passwords[0].value" --output tsv)

if [ -z "$ACR_PASSWORD" ]; then
  echo "ERROR: Could not retrieve ACR password. Ensure admin is enabled:"
  echo "  az acr update -n $REGISTRY_NAME --admin-enabled true"
  exit 1
fi

# 5b. Verify we can authenticate to the registry
echo ">>> Verifying registry login..."
echo "$ACR_PASSWORD" | docker login "$ACR_SERVER" --username "$REGISTRY_NAME" --password-stdin 2>/dev/null \
  && echo "Docker login OK" \
  || echo "WARNING: Local docker login failed (non-fatal if only deploying to ACI)"

# 6. Deploy to ACI with GPU
# Auth: pass admin credentials explicitly so ACI can pull the private image.
# This mirrors the Access-Keys approach from diff-vs-llm/setup_azure.sh.
echo ">>> Deploying container instance with GPU..."
az container create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$CONTAINER_NAME" \
  --image "${ACR_SERVER}/${IMAGE_NAME}:latest" \
  --registry-login-server "$ACR_SERVER" \
  --registry-username "$REGISTRY_NAME" \
  --registry-password "$ACR_PASSWORD" \
  --cpu "$CPU" \
  --memory "$MEMORY" \
  --gpu-count "$GPU_COUNT" \
  --gpu-sku "$GPU_SKU" \
  --ports 8000 \
  --ip-address Public \
  --os-type Linux \
  --environment-variables MODEL_ID="$MODEL_ID" \
  --command-line "python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model $MODEL_ID --gpu-memory-utilization 0.90 --max-model-len 16384 --trust-remote-code"

# 7. Wait for container to be running
echo ">>> Waiting for container to start..."
az container show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$CONTAINER_NAME" \
  --query "instanceView.state" \
  --output tsv

# 8. Get the public IP
PUBLIC_IP=$(az container show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$CONTAINER_NAME" \
  --query "ipAddress.ip" \
  --output tsv)

echo ""
echo "============================================"
echo "DEPLOYMENT COMPLETE"
echo "============================================"
echo "Endpoint: http://${PUBLIC_IP}:8000/v1"
echo ""
echo "Test with:"
echo "  curl http://${PUBLIC_IP}:8000/v1/models"
echo ""
echo "Create your LLM config:"
echo "  cat > .llm_config/qwen-azure.json << EOF"
echo "  {"
echo "    \"model\": \"openai/${MODEL_ID}\","
echo "    \"base_url\": \"http://${PUBLIC_IP}:8000/v1\","
echo "    \"api_key\": \"dummy\","
echo "    \"max_output_tokens\": 8192,"
echo "    \"native_tool_calling\": true"
echo "  }"
echo "  EOF"
echo ""
echo "Run benchmark:"
echo "  uv run swebench-infer .llm_config/qwen-azure.json \\"
echo "    --workspace docker --n-limit 5 --max-iterations 30"
echo ""
echo "To stop and save costs:"
echo "  az container stop -g $RESOURCE_GROUP -n $CONTAINER_NAME"
echo "To restart:"
echo "  az container start -g $RESOURCE_GROUP -n $CONTAINER_NAME"
echo "To delete everything:"
echo "  az group delete -g $RESOURCE_GROUP --yes"
echo "============================================"
