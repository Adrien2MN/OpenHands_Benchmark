# Qwen on Azure — GPU VM Experiment

Runs a Qwen model (default: `Qwen2.5-7B-Instruct`) via vLLM on an Azure GPU VM, executes the OpenHands SWE-bench benchmark against it, and measures GPU energy throughout.

## Prerequisites

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) (`az`) installed and signed in
- Azure subscription with GPU quota in `westeurope`
- A HuggingFace token (`HF_TOKEN`) only for gated models — Qwen2.5 is open

## Architecture

```
Local machine                  Azure GPU VM (NC8as_T4_v3)
─────────────                  ────────────────────────────
setup_vm.sh        ──────────► Creates VM, installs drivers/Docker
run_experiment.sh  ──────────► Pulls image, runs container:
                                  ┌─ run_experiment_inner.sh ─┐
                                  │  1. Download weights       │
                                  │  2. nvidia-smi poll (1 Hz) │
                                  │  3. vLLM server (port 8000)│
                                  │  4. swebench-infer         │
                                  │  5. energy_summary.json    │
                                  └───────────────────────────┘
check_status.sh    ──────────► GPU stats, running containers, results
get_results.sh     ──────────► Downloads results to local ./results_*/
```

All remote commands use `az vm run-command` — no SSH or open inbound ports required.

## Steps

### 1. Provision the VM

```bash
cd azure/qwen-serve
./setup_vm.sh
```

Creates a `Standard_NC8as_T4_v3` VM (T4 16 GB, 8 vCPUs, 56 GB RAM) and installs NVIDIA driver 550, Docker, and the NVIDIA Container Toolkit. Takes ~10–15 minutes.

### 2. Build the experiment image

From the repo root, build `Dockerfile.full` in ACR. It includes vLLM, the benchmark code, and energy tools. Model weights are downloaded at container startup, keeping the build fast.

```bash
az acr build \
  --registry diffusionregistry \
  --image openhands-bench-full:latest \
  --build-arg "MODEL_ID=Qwen/Qwen2.5-7B-Instruct" \
  --timeout 3600 \
  .
```

### 3. Run the experiment

```bash
./run_experiment.sh
```

Optional overrides:

```bash
N_LIMIT=20 MAX_ITERATIONS=50 MODEL_ID=Qwen/Qwen2.5-14B-Instruct ./run_experiment.sh
```

The container runs `run_experiment_inner.sh`, which:
1. Downloads the model weights (cached in `/home/benchuser/hf_cache` across runs)
2. Starts `nvidia-smi` polling at 1 Hz into `energy_log.csv`
3. Launches the vLLM OpenAI-compatible server on port 8000
4. Waits up to 120 s for the model to load, then records 10 s of idle baseline
5. Runs `swebench-infer` and streams output to `benchmark.log`
6. Computes `energy_summary.json` from the power readings

### 4. Check status

```bash
./check_status.sh
```

Prints GPU utilisation, running containers, and the energy summary as it becomes available.

### 5. Download results

```bash
./get_results.sh
```

Creates a timestamped local directory `./results_YYYYMMDD_HHMMSS/` containing:

| File | Contents |
|------|----------|
| `energy_log.csv` | 1 Hz GPU power, utilisation, memory, temperature |
| `energy_summary.json` | Aggregated stats: avg/max watts, total Wh/kWh, duration |
| `benchmark.log` | Full stdout of the benchmark run |

`energy_summary.json` example:

```json
{
  "duration_seconds": 1847,
  "avg_gpu_power_watts": 112.4,
  "max_gpu_power_watts": 148.7,
  "total_gpu_energy_wh": 57.63,
  "total_gpu_energy_kwh": 0.057630,
  "samples": 1847
}
```

### 6. Tear down

```bash
# Delete the VM (keep registry and resource group)
az vm delete -g token-energy-cliff -n openhands-bench-gpu --yes
az disk delete -g token-energy-cliff -n openhands-bench-gpuOsDisk --yes

# Delete everything
az group delete -g token-energy-cliff --yes
```
