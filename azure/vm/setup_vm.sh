#!/bin/bash
set -euo pipefail

# ============================================================
# Deploy a GPU VM that runs the FULL experiment end-to-end:
#   vLLM (Qwen) + OpenHands Benchmark + Docker + Energy measurement
#
# No SSH needed — all interaction via az vm run-command.
# Same pattern as diff-vs-llm but self-contained.
# ============================================================

RESOURCE_GROUP="token-energy-cliff"
LOCATION="westeurope"
VM_NAME="openhands-bench-gpu"
VM_SIZE="Standard_NC24ads_A100_v4"  # A100 80GB, 24 vCPUs, 220GB RAM
IMAGE="Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"
REGISTRY_NAME="diffusionregistry"

echo "============================================"
echo "VM:       $VM_NAME ($VM_SIZE)"
echo "Location: $LOCATION"
echo "Group:    $RESOURCE_GROUP"
echo "============================================"

# 1. Create the VM if it does not already exist.
if az vm show --resource-group "$RESOURCE_GROUP" --name "$VM_NAME" >/dev/null 2>&1; then
  echo ">>> VM already exists; skipping create"
else
  echo ">>> Creating GPU VM..."
  az vm create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --location "$LOCATION" \
    --size "$VM_SIZE" \
    --image "$IMAGE" \
    --admin-username benchuser \
    --generate-ssh-keys \
    --nsg-rule NONE \
    --public-ip-sku Standard \
    --security-type Standard \
    --os-disk-size-gb 256
fi

# 2. Open no ports (we don't need SSH or inbound access)
# The VM only needs outbound to pull from ACR and HuggingFace

# 3. Install NVIDIA GPU drivers + Docker + NVIDIA Container Toolkit on the VM
echo ">>> Installing GPU drivers, Docker, and NVIDIA runtime..."
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts '
    set -e
    export DEBIAN_FRONTEND=noninteractive

    # NVIDIA GPU drivers
    apt-get update
    apt-get install -y linux-headers-$(uname -r)
    apt-get install -y nvidia-driver-550

    # Docker
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker benchuser

    # NVIDIA Container Toolkit
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    echo "Packages installed. Rebooting to load NVIDIA kernel module..."
  '

# 3b. Reboot so the NVIDIA kernel module loads
echo ">>> Rebooting VM to load NVIDIA kernel module..."
az vm restart \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME"

# 3c. Post-reboot: start Docker and verify GPU
echo ">>> Verifying GPU and Docker after reboot..."
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts '
    set -e
    systemctl start docker
    nvidia-smi
    docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
    echo "Docker + GPU OK"
  '

# 4. Login to ACR from VM
echo ">>> Logging VM into ACR..."
ACR_PASSWORD=$(az acr credential show --name "$REGISTRY_NAME" --query "passwords[0].value" --output tsv)
az vm run-command invoke \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts "docker login ${REGISTRY_NAME}.azurecr.io --username ${REGISTRY_NAME} --password '${ACR_PASSWORD}'"

echo ""
echo "============================================"
echo "VM READY: $VM_NAME"
echo "============================================"
echo ""
echo "Next: build the experiment image, then run:"
echo "  ./run_experiment.sh"
echo ""
echo "To check status:"
echo "  ./check_status.sh"
echo ""
echo "To tear down:"
echo "  az vm delete -g $RESOURCE_GROUP -n $VM_NAME --yes"
echo "  az disk delete -g $RESOURCE_GROUP -n ${VM_NAME}OsDisk --yes"
echo "============================================"
