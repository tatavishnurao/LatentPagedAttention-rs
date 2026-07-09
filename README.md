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

- dependency resolution via `uv sync`
- commands via `uv run`
- tests via `uv run pytest -q`
- linting via `uv run ruff check .`

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

## Setup

```bash
uv sync
uv run pytest -q
uv run ruff check .
cargo test --workspace
bash scripts/env_check.sh
```

## Example memory-model command

```bash
uv run plkv-memory \
  --layers 28 \
  --seq-len 4096 \
  --batch-size 1 \
  --kv-heads 8 \
  --head-dim 128 \
  --latent-dim 512 \
  --dtype-bytes 2
```

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

1. Python reference correctness
2. Rust block-table and cache model
3. cuTile kernel validation
4. GQA decode kernel
5. Paged GQA
6. Latent KV
7. Paged Latent KV
8. Quantized cache
9. RTX 4060 benchmark report
