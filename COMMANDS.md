PREBUILD :
uv run swebench-lite-build-images \
  --n-limit 10 \
  --max-workers 5 \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --force-build


INFER : 
uv run swebench-lite-infer .llm_config/litellm-kimi-k2-6.json \
  --n-limit 50 \
  --num-workers 3 \
  --workspace docker

uv run swebench-lite-infer .llm_config/litellm-gpt-4-1-mini.json\
  --n-limit 50\ 
  --num-workers 1\ 
  --workspace docker\ 



EVAL : 
uv run swebench-lite-eval outputs/princeton-nlp__SWE-bench_Lite-test/gpt-4.1-mini_sdk_3e0a3a0_maxiter_500-07-11-42/output.jsonl \
  --run-id gpt-4.1-mini_sdk_3e0a3a0_maxiter_500-07-11-42 \
  --no-modal




mkdir -p "$HOME/.docker/certs.d/ghcr.io"
cp "/Users/z338mn/Library/CloudStorage/OneDrive-AXA/Cursor/AXA-Proxy-ROOT-CA.pem" "$HOME/.docker/certs.d/ghcr.io/ca.crt"

docker buildx rm openhands-builder
docker buildx create --name openhands-builder --driver docker-container --use
docker buildx inspect --bootstrap