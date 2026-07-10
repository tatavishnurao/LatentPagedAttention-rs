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

## Why Rust and cuTile are planned

Rust is the long-term systems layer for block tables, cache layout, benchmark harnesses, and future GPU integration. cuTile-related work is planned later, after the reference models and measurement harness are stable enough to justify kernel work.

## What this project is not

- Not a production inference runtime
- Not a claim of beating vLLM, FlashAttention, or llama.cpp
- Not a promise that MLA-style latent KV is a drop-in runtime swap for existing Llama or Qwen checkpoints
- Not a validated cuTile kernel stack yet

## First milestone checklist

- [x] uv-centric Python scaffold
- [x] importable Python reference package under `python_ref/`
- [x] NumPy memory model and small decode-attention references
- [x] paged block-table model in Python and Rust
- [x] repo docs for scope, formulas, and reporting discipline
- [x] validation scripts
- [ ] real GPU kernels
- [ ] end-to-end model integration

## Current milestone

This repo currently validates the reference-level mechanics of:

- GQA decode attention
- paged KV lookup
- latent KV reconstruction
- paged latent KV lookup
- KV memory estimation

It does not yet contain CUDA/cuTile kernels or real model inference.

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
3. Rust parity + golden fixtures — this pass
4. RTX 4060 baseline — this pass
5. cuTile smoke test
6. GPU paged lookup / KV write primitive
7. GPU GQA decode
8. GPU paged GQA
9. GPU latent KV reconstruction
10. GPU paged latent KV
