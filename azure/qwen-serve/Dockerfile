# vLLM serving container for Qwen on Azure GPU (ACI)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git wget ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM (includes torch with CUDA)
RUN pip install --no-cache-dir "vllm>=0.8.0" "huggingface-hub[hf_transfer]"

# Pre-download model weights at build time (baked into the image)
# Change MODEL_ID for different Qwen sizes
ARG MODEL_ID=Qwen/Qwen2.5-7B-Instruct
ENV MODEL_ID=${MODEL_ID}
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Download model; requires HF_TOKEN at build time for gated models
ARG HF_TOKEN=""
RUN if [ -n "$HF_TOKEN" ]; then \
      hf auth login --token "$HF_TOKEN"; \
    fi && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"

EXPOSE 8000

# Serve with vLLM OpenAI-compatible API
# Adjust gpu-memory-utilization and max-model-len as needed
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--host", "0.0.0.0", "--port", "8000", \
     "--model", "${MODEL_ID}", \
     "--gpu-memory-utilization", "0.90", \
     "--max-model-len", "16384", \
     "--trust-remote-code"]
