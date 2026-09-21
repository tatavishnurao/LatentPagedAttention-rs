# Reference Benchmark Report

## Purpose

This benchmark validates reference-level attention mechanics and KV memory accounting for four synthetic NumPy workloads. It measures local NumPy reference runtime and does not represent a GPU implementation.

## Variants

- GQA
- Paged GQA
- Latent KV
- Paged Latent KV

## Metrics

- KV bytes/token/layer
- total KV MiB
- compression ratio against full GQA KV cache
- NumPy reference runtime
- output shape

## How to run

```bash
UV_PROJECT_ENVIRONMENT=attention99 uv run plkv-bench-ref --config small --iters 20
UV_PROJECT_ENVIRONMENT=attention99 uv run plkv-bench-ref --config medium --iters 10
bash scripts/run_reference_benchmarks.sh
```

## Interpretation

Paged GQA does not reduce theoretical KV bytes/token. Latent KV reduces stored cache bytes/token. Paged Latent KV combines allocation layout with latent cache storage. Reference runtime includes NumPy reconstruction/projection costs.

## Limitations

These numbers do not measure GPU performance, cuTile kernels, CUDA kernels, or real model inference. They do not prove quality preservation and should not be compared against vLLM, llama.cpp, TensorRT-LLM, or FlashAttention.
