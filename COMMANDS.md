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



AZURE LOGIN :

set -a
source .env
set +a
az login --tenant d65b03ed-6a7d-41ca-a17d-4798d70d1d3f

CREATE CLUSTER :

az ml compute create --name gpu-cluster-amn --type amlcompute --size Standard_NC24ads_A100_v4 --min-instances 0 --max-instances 1 --idle-time-before-scale-down 120 --workspace-name ai-research --resource-group token-energy-cliff

SEND JOB TO CLUSTER

cd /Users/z338mn/Library/CloudStorage/OneDrive-AXA/Cursor/OpenHands_bench/benchmarks && set -a && source .env && set +a && cd download_models && az ml job create -f job_eval.yml --set name=swebench-lite-infer-33 display_name=swebench-lite-infer-33 --workspace-name ai-research --resource-group token-energy-cliff --web 2>&1 | tail -60

az ml job create \
  -f azure/job_eval.yml \
  --workspace-name ai-research \
  --resource-group token-energy-cliff



DELETE CLUSTER

az ml compute delete --name gpu-cluster-amn --workspace-name ai-research --resource-group token-energy-cliff --yes