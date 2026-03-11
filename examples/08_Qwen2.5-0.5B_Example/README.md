# Qwen2.5-0.5B-Instruct Benchmark Example

This example benchmarks `Qwen/Qwen2.5-0.5B-Instruct` against either a vLLM or
SGLang server. It is intended as a small-GPU example that works on typical
8-16 GB cards.

## Requirements

- Python 3.12+
- Docker with NVIDIA GPU support
- NVIDIA GPU with at least 8 GB VRAM

## Fastest Path

From the repo root:

```bash
# Offline benchmark with vLLM
bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh vllm offline

# Offline benchmark with SGLang
bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh sglang offline

# Online concurrency sweep
bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh vllm online
```

The script prepares the dataset, starts or reuses a container, waits for the
server, and runs the benchmark.

## Manual Flow

If you do not want to use `run_benchmark.sh`, the minimum manual flow is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

python examples/08_Qwen2.5-0.5B_Example/prepare_dataset.py
```

Start one server:

```bash
# vLLM
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -p 8000:8000 \
  --ipc=host \
  --name vllm-qwen \
  -d \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --gpu-memory-utilization 0.85

# SGLang
docker run --runtime nvidia --gpus all --net host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  --name sglang-qwen \
  -d \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.9 \
  --attention-backend flashinfer
```

Run one benchmark:

```bash
# vLLM offline
inference-endpoint benchmark from-config \
  -c examples/08_Qwen2.5-0.5B_Example/offline_qwen_benchmark.yaml

# SGLang offline
inference-endpoint benchmark from-config \
  -c examples/08_Qwen2.5-0.5B_Example/sglang_offline_qwen_benchmark.yaml

# vLLM online sweep
python scripts/concurrency_sweep/run.py \
  --config examples/08_Qwen2.5-0.5B_Example/online_qwen_benchmark.yaml

# SGLang online sweep
python scripts/concurrency_sweep/run.py \
  --config examples/08_Qwen2.5-0.5B_Example/sglang_online_qwen_benchmark.yaml
```

## Files

- `offline_qwen_benchmark.yaml`: vLLM offline benchmark
- `online_qwen_benchmark.yaml`: vLLM online concurrency sweep
- `sglang_offline_qwen_benchmark.yaml`: SGLang offline benchmark
- `sglang_online_qwen_benchmark.yaml`: SGLang online concurrency sweep
- `prepare_dataset.py`: converts `tests/datasets/dummy_1k.pkl` into the example dataset
- `run_benchmark.sh`: wrapper that automates dataset prep, container startup, and benchmark execution

## Results

- vLLM offline: `results/qwen_offline_benchmark/`
- vLLM online: `results/qwen_online_benchmark/concurrency_sweep/`
- SGLang offline: `results/qwen_sglang_offline_benchmark/`
- SGLang online: `results/qwen_sglang_online_benchmark/concurrency_sweep/`

To summarize an online sweep:

```bash
python scripts/concurrency_sweep/summarize.py \
  results/qwen_online_benchmark/concurrency_sweep/
```

## Notes

- The online sweep defaults to `1 2 4 8 16 32 64 128 256 512 1024`.
- Use `scripts/concurrency_sweep/run.py --concurrency ... --duration-ms ...` to shorten or customize the sweep.
- If vLLM runs out of memory at higher concurrency, lower `--gpu-memory-utilization`.
