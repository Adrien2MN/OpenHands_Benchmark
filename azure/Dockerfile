# Base: AML curated GPU image (Ubuntu 22.04, CUDA, OpenMPI, Python 3.10)
# We add Docker Engine on top so that --workspace docker works inside the job.
#
# Build and push once locally (replace with your ACR name):
#   az acr login --name <your_acr_name>
#   docker build -t <your_acr_name>.azurecr.io/openhands-swebench:v1 -f azure/Dockerfile .
#   docker push <your_acr_name>.azurecr.io/openhands-swebench:v1
#
# Then update azure/job_eval.yml:
#   environment:
#     image: <your_acr_name>.azurecr.io/openhands-swebench:v1
#
# The AML compute node runs your job container with /var/run/docker.sock
# bind-mounted from the host — so Docker Engine installed here + the host
# daemon socket = full docker build/pull/run capability inside the job,
# without needing privileged execution or Docker-in-Docker.
# (If the socket is NOT mounted, eval_infer.py's probe_and_setup_docker()
# will log a clear warning rather than failing silently.)

FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest

# Install Docker Engine (CLI + daemon, but daemon won't be used —
# we rely on the host socket mounted at /var/run/docker.sock)
RUN apt-get update -qq && \
    apt-get install -y -q --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        lsb-release && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list && \
    apt-get update -qq && \
    apt-get install -y -q --no-install-recommends \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv (used by eval_infer.py for dependency management)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Verify docker CLI is on PATH
RUN docker --version