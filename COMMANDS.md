LITELLM PROXY

make run-litellm-proxy PROXY_ENV_FILE=.env 

PREBUILD

uv run swebench-lite-build-images \
  --n-limit 10 \
  --max-workers 5 \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --force-build


INFER

uv run swebench-lite-infer .llm_config/litellm-gpt-4-1-mini.json \
  --n-limit 50 \
  --num-workers 3 \
  --workspace docker



EVAL 

uv run swebench-lite-eval outputs/princeton-nlp__SWE-bench_Lite-test/claude-opus-4-6_sdk_3e0a3a0_maxiter_500-12-11-15/output.jsonl \
  --run-id claude-opus-4-6_sdk_3e0a3a0_maxiter_500-12-11-15 \
  --no-modal



AZURE LOGIN 

set -a
source .env
set +a
az login --tenant d65b03ed-6a7d-41ca-a17d-4798d70d1d3f

CREATE CLUSTER

az ml compute create --name gpu-cluster-amn --type amlcompute --size Standard_NC24ads_A100_v4 --min-instances 0 --max-instances 1 --idle-time-before-scale-down 120 --workspace-name ai-research --resource-group token-energy-cliff

SEND JOB TO CLUSTER

set -a
source .env
set +a
az ml job create \
  -f azure/job_eval.yml \
  --workspace-name ai-research \
  --resource-group token-energy-cliff \
  --set environment_variables.HF_TOKEN="$HF_TOKEN" \
  --set environment_variables.API_KEY="$API_KEY" \
  --set environment_variables.GPT41_MINI_API_URL="$GPT41_MINI_API_URL"

DELETE CLUSTER

az ml compute delete --name gpu-cluster-amn --workspace-name ai-research --resource-group token-energy-cliff --yes


SETUP THE VM and OPEN PORT 22 FOR LOGS

./azure/vm/setup_vm.sh

az vm open-port \
  --resource-group token-energy-cliff \
  --name openhands-bench-gpu \
  --port 22


BUILD THE ACR IMAGE (1 time)

az acr build \
  --registry diffusionregistry \
  --image openhands-bench-full:latest \
  --file azure/vm/Dockerfile.full \
  --timeout 3600 \
  .

RUN THE VM WITH MODELS 

set -a && source .env && set +a
MODEL_SOURCE=local \
MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3 \
LLM_CONFIG=litellm-mistral7b.json \
  ./azure/vm/run_experiment.sh


DEBUG LOGS

az vm run-command invoke \
  --resource-group token-energy-cliff \
  --name openhands-bench-gpu \
  --command-id RunShellScript \
  --scripts '
    echo "=== Last container status ==="
    docker ps -a --filter name=openhands-bench

    echo ""
    echo "=== Container logs (last 100 lines) ==="
    docker logs openhands-bench 2>&1 | tail -100
  '