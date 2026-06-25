#!/bin/bash
set -euo pipefail

# Tear down the Qwen ACI deployment (stop billing)
RESOURCE_GROUP="openhands-bench"
CONTAINER_NAME="qwen-serve"

echo "Stopping container..."
az container stop --resource-group "$RESOURCE_GROUP" --name "$CONTAINER_NAME" 2>/dev/null || true

echo "Deleting container..."
az container delete --resource-group "$RESOURCE_GROUP" --name "$CONTAINER_NAME" --yes 2>/dev/null || true

echo "Done. Registry and resource group preserved."
echo "To delete everything: az group delete -g $RESOURCE_GROUP --yes"
