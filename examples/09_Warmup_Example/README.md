# Warmup Example

This directory shows how to add a warmup phase to offline and online benchmark
configs for [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

Warmup sends randomly generated requests before the timed run to prime the
endpoint. Those samples complete before `TEST_STARTED`, so they are excluded
from reported throughput and latency metrics.

## Files

- `warmup_offline.yaml`: offline max-throughput example
- `warmup_online.yaml`: online Poisson-QPS example

Both configs use the same `warmup` block:

```yaml
warmup:
  num_samples: 64
  input_seq_length: 256
  output_seq_length: 64
  range_ratio: 0.9
  random_seed: 42
```

Warmup data is generated at runtime from random token IDs using the model
tokenizer, so no separate warmup dataset is needed.

## Run Locally

With the built-in echo server:

```bash
python -m inference_endpoint.testing.echo_server --port 8000
inference-endpoint benchmark from-config -c examples/09_Warmup_Example/warmup_offline.yaml
inference-endpoint benchmark from-config -c examples/09_Warmup_Example/warmup_online.yaml
```

Against a real endpoint, point `endpoint_config.endpoints` in the YAML at that
server and run the same commands.

## Tuning

- `num_samples`: use enough requests to reach a steady state
- `input_seq_length`: match the typical prompt length of the workload
- `output_seq_length`: match the expected response length
- `range_ratio`: use values below `1.0` to add light ISL variation
