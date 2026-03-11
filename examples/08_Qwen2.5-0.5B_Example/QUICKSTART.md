# Quick Start - Qwen2.5-0.5B

Use the wrapper script from the repo root:

```bash
# vLLM offline
bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh vllm offline

# SGLang offline
bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh sglang offline

# Online concurrency sweep
bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh vllm online
```

Outputs:

- vLLM offline: `results/qwen_offline_benchmark/`
- vLLM online: `results/qwen_online_benchmark/concurrency_sweep/`
- SGLang offline: `results/qwen_sglang_offline_benchmark/`
- SGLang online: `results/qwen_sglang_online_benchmark/concurrency_sweep/`

Summarize an online sweep:

```bash
python scripts/concurrency_sweep/summarize.py \
  results/qwen_online_benchmark/concurrency_sweep/
```

For manual setup, server commands, and config details, see [README.md](README.md).
