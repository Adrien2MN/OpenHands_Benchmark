#!/bin/bash
# Quick wrapper for running eval pipeline
# Usage: ./run-pipeline.sh gpt-4.1-mini 1    # model, instances
# Usage: ./run-pipeline.sh mistral-small-2503 5 200  # model, instances, max-iterations
# Usage: ./run-pipeline.sh gpt-4.1-mini 20 100 --num-workers 20 --n-critic-runs 1 --max-retries 1

cd "$(dirname "$0")"

if [ $# -lt 2 ]; then
    echo "Usage: ./run-pipeline.sh <model> <instances> [max-iterations]"
    echo ""
    echo "Examples:"
    echo "  ./run-pipeline.sh gpt-4.1-mini 1"
    echo "  ./run-pipeline.sh mistral-small-2503 5"
    echo "  ./run-pipeline.sh o1 1 200"
    echo ""
    echo "Available models:"
    echo "  - gpt-4.1-mini"
    echo "  - gpt-5.4"
    echo "  - gpt-4.1"
    echo "  - mistral-small-2503"
    echo "  - deepseek-v3.1"
    echo "  - deepseek-v3-1"
    echo "  - kimi-k2.6"
    echo "  - o1"
    echo "  - llama-3.3-70b-instruct"
    exit 1
fi

MODEL="$1"
INSTANCES="$2"

shift 2
MAX_ITER="100"

if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    MAX_ITER="$1"
    shift 1
fi

python3 run_eval_pipeline.py \
    --model "$MODEL" \
    --instances "$INSTANCES" \
    --max-iterations "$MAX_ITER" \
    "$@"
