# SWE-Bench Eval Pipeline

Quick tools to run infer and eval without typing long commands.

## Quick Start

### Option 1: Shell Wrapper (Fastest)
```bash
cd benchmarks
./run-pipeline.sh gpt-4.1-mini 1
./run-pipeline.sh mistral-small-2503 5
./run-pipeline.sh o1 1 200
```

### Option 2: Python Script (Full Control)
```bash
cd benchmarks
python3 run_eval_pipeline.py --model gpt-4.1-mini --instances 1
python3 run_eval_pipeline.py --model mistral-small-2503 --instances 5 --max-iterations 100
python3 run_eval_pipeline.py --model o1 --instances 1 --max-iterations 200
```

## Features

- **Automatic config generation**: Creates model-specific LLM configs on first use
- **Auto-infer → auto-eval**: Runs evaluation immediately after inference completes
- **Smart output detection**: Finds the right output.jsonl file automatically
- **Stable output folder**: Renames completed runs to `model__instance_id`
- **Overwrite on rerun**: Repeated runs on the same model + instance replace the previous folder
- **Progress tracking**: Shows which files are being created
- **Report summary**: Displays resolution stats after eval completes

## Available Models

- `gpt-4.1-mini` (default, fast/cheap)
- `gpt-4.1` (more capable)
- `mistral-small-2503` (fast alternative)
- `o1` (reasoning model, slower)
- `llama-3.3-70b-instruct` (open source)

## Common Usage Patterns

### Test a new model (1 instance, standard iterations)
```bash
./run-pipeline.sh mistral-small-2503 1
```

### Run a small batch (5 instances)
```bash
./run-pipeline.sh gpt-4.1 5
```

### Reasoning model with more iterations
```bash
./run-pipeline.sh o1 1 200
```

### Infer only (skip eval)
```bash
python3 run_eval_pipeline.py --model mistral-small-2503 --instances 5 --infer-only
```

### Eval only (on existing output)
```bash
python3 run_eval_pipeline.py --eval-only outputs/path/to/output.jsonl
```

## Output

Results are saved to `benchmarks/outputs/` with structure:
```
outputs/
├── {dataset}__{split}/
│   └── {model}__{instance_id}/
│       ├── output.jsonl           # Full infer results
│       ├── output.report.json     # Per-instance report (tokens, energy)
│       └── report/
│           └── report.json        # Summary (resolved count, totals)
```

If you run the same model on the same instance again, the pipeline removes the
existing folder first and writes a fresh one in the same place.

## Example Output

```
======================================================================
Running infer with mistral-small-2503
======================================================================
Command: uv run swebench-infer .llm_config/litellm-mistral-small-2503.json ...

✓ Infer completed successfully
  Output file: benchmarks/outputs/.../output.jsonl

✓ Stable output folder: benchmarks/outputs/princeton-nlp__SWE-bench_Verified-test/mistral-small-2503__django__django-13439
  Output file: benchmarks/outputs/princeton-nlp__SWE-bench_Verified-test/mistral-small-2503__django__django-13439/output.jsonl

======================================================================
Running eval on infer output
======================================================================
✓ Eval completed successfully
  Report: benchmarks/outputs/.../report/report.json
  Total instances: 5
  Resolved: 2
```

## Troubleshooting

**Error: LLM config file not found**
- Make sure `.llm_config/litellm.json` exists with valid credentials
- Proxy must be running: `docker-compose up -d` in project root

**Eval shows "resolved: 0"**
- Check the instance report for patch syntax errors
- Try a different model with better generation

**Proxy connection refused**
- Ensure proxy is running: `docker ps | grep litellm`
- Check it's listening: `curl http://localhost:4000/v1/models`

## Model-Specific Tips

| Model | Best For | Speed | Cost | Notes |
|-------|----------|-------|------|-------|
| gpt-4.1-mini | Testing, iterations | Fast | Low | Good baseline |
| gpt-4.1 | Production | Medium | Medium | More capable |
| mistral-small-2503 | Fast turnaround | Very Fast | Low | Alternative to mini |
| o1 | Hard problems | Slow | High | Reasoning model |
| llama-3.3-70b | Open source | Medium | Medium | Privacy-friendly |
