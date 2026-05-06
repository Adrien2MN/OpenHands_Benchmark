# SWE-Bench Lite

SWE-Bench Lite is the lower-cost subset of SWE-Bench.

## What It Is

- Official dataset: `princeton-nlp/SWE-bench_Lite`
- Curated subset of SWE-Bench intended for faster/cheaper evaluation
- Publicly documented size: 300 Lite benchmark instances (test)

## In This Folder

- `run_infer.py`: wrapper around `benchmarks.swebench.run_infer` with default dataset `princeton-nlp/SWE-bench_Lite`
- `build_images.py`: wrapper around SWE-Bench image prebuild with Lite defaults
- `eval_infer.py`: wrapper around SWE-Bench evaluator with Lite defaults
- `prepare_dataset.py`: exports Lite rows to local JSONL (for 100/300-instance workflows)

## Usage

Prebuild Lite images (first 100):

`uv run python -m benchmarks.swebench_lite.build_images --n-limit 100 --target source-minimal --image ghcr.io/openhands/eval-agent-server`

Run Lite inference (first 100):

`uv run swebench-lite-infer .llm_config/litellm-gpt-5-4.json --workspace docker --n-limit 100 --num-workers 5`

Run Lite inference (all 300 test instances):

`uv run swebench-lite-infer .llm_config/litellm-gpt-5-4.json --workspace docker --n-limit 300 --num-workers 8`

Convert an OpenHands output file to SWE-Bench predictions and evaluate:

`uv run swebench-lite-eval outputs/<dataset>/<run>/output.jsonl --run-id my_lite_run`

Export Lite split locally as JSONL:

`uv run python -m benchmarks.swebench_lite.prepare_dataset --split test --n-limit 300 --output data/swebench_lite/swebench_lite_test_300.jsonl`
