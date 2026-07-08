#!/bin/bash
set -e
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey > /tmp/nvidia.gpg
gpg --dearmor < /tmp/nvidia.gpg > /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list > /tmp/nvidia-ctk.list
sed -i "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" /tmp/nvidia-ctk.list
cp /tmp/nvidia-ctk.list /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
