# LatentPagedAttention-rs v0.1 - Paged Latent-Cache Attention on an RTX 4060

## Summary

LatentPagedAttention-rs is a Rust and Python research prototype that combines
paged physical cache addressing with direct latent-space grouped-query attention.
It validates FP16 latent-cache storage, FP32 attention arithmetic, runtime
sequence masking, partial blocks, GPU cache mutation, and a model-shaped profile
on an RTX 4060.

This release is correctness-first. It is not a production serving runtime and it
does not claim speedups over vLLM, FlashAttention, TensorRT-LLM, or any
production inference engine.

## What Was Built

- Physical paged latent-cache storage.
- Runtime non-identity block tables.
- Runtime active sequence lengths and partial-final-block masking.
- Direct latent-space GQA without persistent reconstructed K/V tensors.
- GPU paged latent-cache writes followed by attention using the updated device cache.
- FP16 latent storage with FP32 score, softmax, and context arithmetic.
- A synthetic model-shaped GPU profile.
- An FP16 paged full-KV baseline with FP32 accumulation.
- Python, Rust CPU, and cuTile GPU parity validation.

## Key Result

For the committed `model_small` profile, the prototype stores `16x` fewer
persistent FP16 cache bytes than the FP16 full-KV baseline:

- FP16 latent cache: `65,536` bytes.
- FP16 full-KV cache: `1,048,576` bytes.
- Persistent cache-byte ratio: `16x`.

The current latent read path is also slower:

- FP16 full-KV read mean: `1391.022 ms`.
- FP16 latent read mean: `1844.891 ms`.
- Latent read overhead: approximately `32.6%`.

This is a measured memory-versus-compute trade-off, not an unconditional speedup.

## Correctness Methodology

The validation chain is:

```text
Python oracle
-> deterministic fixture
-> Rust CPU reference
-> cuTile GPU execution
-> readback and parity checks
```

The release checks finite outputs, probability row sums, inactive probabilities,
non-identity block-table controls, changed-element counts, no host cache round
trip in the write-to-attention path, and bit-exact FP16 cache storage.

## Supported Profiles

| profile | q_heads | kv_heads | group_size | head_dim | latent_dim | block_size | max_seq_len | storage |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tiny | 4 | 2 | 2 | 8 | 8 | 2 | 8 | f32/f16 |
| model_small | 16 | 4 | 4 | 64 | 32 | 16 | 1024 | f16 |

`model_small` is a synthetic model-shaped profile. It is not a production model
match.

## Benchmark Methodology

The committed benchmark uses `SYNCHRONIZED_HOST_END_TO_END_TIME` with three
measured processes. Compilation, cuTile JIT, and warmup are excluded. The results
are not kernel-only latency.

## Benchmark Results

| operation | process_count | min_ms | mean_ms | max_ms |
|---|---:|---:|---:|---:|
| full_kv_paged_attention_read | 3 | 1366.969 | 1391.022 | 1405.751 |
| latent_paged_attention_read | 3 | 1705.150 | 1844.891 | 2017.385 |
| latent_write_to_attention | 3 | 1367.174 | 1487.776 | 1586.213 |

## Persistent Cache Accounting

The `16x` ratio applies only to persistent cache bytes for the synthetic
`model_small` profile. It does not describe total runtime GPU memory.

## Reproduction Commands

```bash
UV_PROJECT_ENVIRONMENT=attention99 uv sync
UV_PROJECT_ENVIRONMENT=attention99 uv run pytest -q
UV_PROJECT_ENVIRONMENT=attention99 uv run ruff check .
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
source scripts/cutile_env.sh
bash scripts/run_gpu_runtime_sequence_validation.sh
bash scripts/run_gpu_model_profile_validation.sh
bash scripts/run_gpu_fp16_full_kv_baseline.sh
bash scripts/run_final_benchmark.sh
```

## Explicit Limitations

This release is not production-ready, not faster than vLLM, not faster than
FlashAttention, not faster than TensorRT-LLM, not complete DeepSeek MLA, not a
real-model checkpoint integration, not proof of Tensor Core usage, not proof of
total GPU-memory reduction, not proof of model-quality preservation, not a
production PagedAttention runtime, not a continuous-batching system, and not a
dynamic cache allocator.

## Documentation Links

- [README](README.md)
- [Final report](docs/FINAL_REPORT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Limitations](docs/LIMITATIONS.md)
- [Benchmark summary](reports/final_benchmark/summary.md)
- [Citation metadata](CITATION.cff)

## Citation

Please cite the release using `CITATION.cff`.

## Hardware and Software Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU.
- Compute capability: 8.9.
- VRAM: 8188 MiB.
- CUDA toolkit path: `/opt/cuda`.
- cuTile: `0.2.0`.
- Rust workspace version: `0.1.0`.
