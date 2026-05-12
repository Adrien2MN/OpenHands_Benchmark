PREBUILD :
uv run swebench-lite-build-images \
  --n-limit 10 \
  --max-workers 5 \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --force-build


INFER : 
uv run swebench-lite-infer .llm_config/litellm-gpt-4-1-mini.json \
  --n-limit 50 \
  --num-workers 3 \
  --workspace docker



EVAL : 
uv run swebench-lite-eval outputs/princeton-nlp__SWE-bench_Lite-test/claude-opus-4-6_sdk_3e0a3a0_maxiter_500-12-11-15/output.jsonl \
  --run-id claude-opus-4-6_sdk_3e0a3a0_maxiter_500-12-11-15 \
  --no-modal



