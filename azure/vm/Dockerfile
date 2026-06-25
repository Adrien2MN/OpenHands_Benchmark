# Full experiment image for GPU VM.
# Bakes in the repo code + inner script.
# Model weights are downloaded at container startup (cached via volume mount).
# Docker socket is mounted from the host VM at runtime.
#
# Build (from repo root):
#   az acr build \
#     --registry diffusionregistry \
#     --image openhands-bench-full:latest \
#     --file azure/vm/Dockerfile.full \
#     --timeout 3600 \
#     .
#
# Run (via run_experiment.sh):
#   docker run --rm \
#     --gpus all \
#     --network host \
#     -v /var/run/docker.sock:/var/run/docker.sock \
#     -v /home/benchuser/results:/results \
#     -v /home/benchuser/hf_cache:/root/.cache/huggingface \
#     -e HF_TOKEN=... \
#     -e N_LIMIT=1 \
#     -e MAX_ITERATIONS=30 \
#     diffusionregistry.azurecr.io/openhands-bench-full:latest

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps: Python, git, curl, Docker CLI
RUN apt-get update -qq && \
    apt-get install -y -q --no-install-recommends \
        python3.12 python3.12-dev python3-pip \
        git curl ca-certificates gnupg lsb-release \
        build-essential ninja-build && \
    # Docker CLI (to use host socket mounted at runtime)
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list && \
    apt-get update -qq && \
    apt-get install -y -q --no-install-recommends \
        docker-ce-cli docker-buildx-plugin && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy repo into image
WORKDIR /app
COPY . /app

# Make inner script executable
RUN chmod +x azure/vm/run_experiment_inner.sh

# Default entrypoint
CMD ["bash", "azure/vm/run_experiment_inner.sh"]