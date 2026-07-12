# LatentPagedAttention-rs

LatentPagedAttention-rs is a uv-centric Rust + Python research-engineering scaffold for studying paged KV-cache allocation, GQA decode attention, MLA-style latent KV compression, and later kernel work on memory-constrained GPUs.

It is not a full inference engine, not a vLLM competitor, and not a claim of a new attention mechanism. The first milestone is reproducible reference code and clean repo structure.

## Why this repo exists

The target hardware is an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM. That is a useful constraint because decode experiments quickly become memory-bound, especially as context length and concurrency rise.

This repo starts with:

- Python and NumPy reference implementations for correctness
- Rust workspace scaffolding for cache and block-table logic
- A simple memory model for reasoning about KV footprint
- Docs that keep claims narrow and reproducible

## Why uv is central

The Python side is intentionally `uv`-first:

- dependency resolution via `UV_PROJECT_ENVIRONMENT=attention99 uv sync`
- commands via `UV_PROJECT_ENVIRONMENT=attention99 uv run`
- tests via `UV_PROJECT_ENVIRONMENT=attention99 uv run pytest -q`
- linting via `UV_PROJECT_ENVIRONMENT=attention99 uv run ruff check .`

That keeps the reference layer lightweight and reproducible without introducing a larger environment stack.

## Why Rust and cuTile matter

Rust is the systems layer for block tables, cache layout, validation harnesses, and GPU integration. cuTile `0.2.0` is integrated behind the optional `gpu-cutile` feature.

## What this project is not

- Not a production inference runtime
- Not a claim of beating vLLM, FlashAttention, or llama.cpp
- Not a promise that MLA-style latent KV is a drop-in runtime swap for existing Llama or Qwen checkpoints
- Not a production cuTile kernel stack

## First milestone checklist

- [x] uv-centric Python scaffold
- [x] importable Python reference package under `python_ref/`
- [x] NumPy memory model and small decode-attention references
- [x] paged block-table model in Python and Rust
- [x] repo docs for scope, formulas, and reporting discipline
- [x] validation scripts
- [x] cuTile vector-add smoke test on RTX 4060
- [x] non-identity GPU paged lookup
- [x] Python/Rust/GPU parity fixtures
- [x] single-token paged KV write completion
- [x] contiguous f32 GQA CPU reference
- [x] contiguous f32 GQA cuTile validation
- [x] direct paged f32 GQA cuTile validation
- [x] standalone f32 latent-KV reconstruction
- [ ] end-to-end model integration

## Current milestone

This repo currently validates:

- GQA decode attention
- paged KV lookup
- latent KV reconstruction
- paged latent KV lookup
- KV memory estimation
- cuTile vector-add execution on the RTX 4060
- non-identity GPU paged lookup
- Python/Rust/GPU paged lookup parity
- single-token paged KV-cache write validation
- GPU write-to-lookup round trip
- contiguous f32 GQA CPU reference
- contiguous f32 GQA cuTile validation
- direct paged f32 GQA cuTile validation
- standalone f32 latent-KV reconstruction
- direct contiguous latent-KV GQA
- direct paged latent-KV GQA
- paged latent-cache write and write-to-attention round trip

The current milestone is GPU paged latent-cache write and write-to-attention round
trip. Real model inference remains unimplemented.

## Setup

```bash
UV_PROJECT_ENVIRONMENT=attention99 uv sync
UV_PROJECT_ENVIRONMENT=attention99 uv run pytest -q
UV_PROJECT_ENVIRONMENT=attention99 uv run ruff check .
cargo test --workspace
bash scripts/env_check.sh
```

## Example memory-model command

```bash
UV_PROJECT_ENVIRONMENT=attention99 uv run plkv-memory \
  --layers 28 \
  --seq-len 4096 \
  --batch-size 1 \
  --kv-heads 8 \
  --head-dim 128 \
  --latent-dim 512 \
  --dtype-bytes 2
```

## Reference benchmark

This repo includes a NumPy-only reference benchmark for GQA, Paged GQA, Latent KV, and Paged Latent KV.

```bash
UV_PROJECT_ENVIRONMENT=attention99 uv run plkv-bench-ref --config small --iters 20
UV_PROJECT_ENVIRONMENT=attention99 uv run plkv-bench-ref --config medium --iters 10
bash scripts/run_reference_benchmarks.sh
```

CSV outputs are written to `reports/reference_benchmarks/`. These are not GPU benchmarks and should not be interpreted as cuTile or CUDA performance.

## Golden fixtures

The repo includes Python-generated JSON fixtures under `fixtures/reference/`. These validate that Rust memory and block-table logic matches the Python reference implementation.

Generate fixtures:

```bash
UV_PROJECT_ENVIRONMENT=attention99 bash scripts/generate_golden_fixtures.sh
```

Validate:

```bash
UV_PROJECT_ENVIRONMENT=attention99 uv run pytest -q
cargo test --workspace
```

## RTX 4060 baseline

The project targets an RTX 4060 Laptop GPU with 8 GB VRAM. Before writing GPU kernels, capture the local hardware and toolchain state:

```bash
UV_PROJECT_ENVIRONMENT=attention99 bash scripts/rtx4060_env_snapshot.sh
UV_PROJECT_ENVIRONMENT=attention99 bash scripts/rtx4060_memory_sanity.sh
```

These scripts do not benchmark cuTile or CUDA kernels. They capture environment and memory-model context only.

## GPU status

cuTile `0.2.0` is pinned behind the optional `gpu-cutile` feature. The first GPU
pass has completed the following:

- cuTile `0.2.0` integration
- RTX 4060 vector-add smoke test
- GPU non-identity paged lookup
- GPU single-token paged K/V write
- GPU write-to-lookup round trip
- contiguous f32 GQA CPU reference
- contiguous f32 GQA cuTile validation
- direct paged f32 GQA cuTile validation
- GPU latent-KV reconstruction
- GPU direct contiguous latent-KV GQA
- GPU direct paged latent-KV GQA
- GPU paged latent-cache write and write-to-attention round trip
- FP16 latent storage with FP32 accumulation
- Python/Rust/GPU parity

Current milestone:

- BF16 latent storage with f32 accumulation

Not implemented:

- BF16
- real model inference
- performance benchmarking

```bash
source scripts/cutile_env.sh
bash scripts/run_cutile_smoke.sh
bash scripts/run_gpu_paged_lookup.sh
bash scripts/run_gpu_paged_kv_write.sh
bash scripts/run_gpu_gqa_decode.sh
bash scripts/run_gpu_paged_gqa_decode.sh
bash scripts/run_gpu_latent_kv_reconstruction.sh
bash scripts/run_gpu_direct_latent_gqa.sh
bash scripts/run_gpu_direct_paged_latent_gqa.sh
bash scripts/run_gpu_paged_latent_write_attention.sh
bash scripts/run_gpu_fp16_paged_latent_attention.sh
```

These commands validate compilation, JIT execution, synchronization, host
readback, and correctness. They do not benchmark attention, claim speedups, or
represent production inference.

## Repo layout

```text
.
├── pyproject.toml
├── .python-version
├── python_ref/
├── tests/
├── scripts/
├── docs/
├── crates/
└── Cargo.toml
```

## Research roadmap

1. Python reference correctness — done
2. NumPy reference benchmark — done
3. Rust parity and golden fixtures — done
4. RTX 4060 baseline — done
5. cuTile smoke test — done
6. GPU paged lookup — done
7. GPU single-token paged KV write — done
8. GPU contiguous f32 GQA decode — done
9. GPU direct Paged f32 GQA decode — done
10. GPU latent-KV reconstruction — done
11. GPU direct contiguous latent-KV GQA — done
12. GPU direct Paged Latent KV GQA — done
13. GPU paged latent-cache write and attention round trip — done
14. FP16 latent storage with FP32 accumulation — done
15. BF16 latent storage with FP32 accumulation — next
16. variable-shape and partial-block support
17. RTX 4060 performance study
