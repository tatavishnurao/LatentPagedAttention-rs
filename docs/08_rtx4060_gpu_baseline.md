# RTX 4060 GPU Baseline

## Purpose

This document captures the hardware target for the project.

## Why RTX 4060 matters

- 8 GB consumer laptop GPU constraint
- KV cache and memory pressure are central
- the project targets memory behavior before production inference speed

## What the baseline scripts capture

- NVIDIA driver and CUDA runtime visibility
- `nvcc` toolchain status
- total, used, and free VRAM
- Rust, uv, Python, and git commit
- theoretical KV-cache estimates for representative configurations

## How to run

```bash
UV_PROJECT_ENVIRONMENT=attention99 bash scripts/rtx4060_env_snapshot.sh
UV_PROJECT_ENVIRONMENT=attention99 bash scripts/rtx4060_memory_sanity.sh
```

## What this proves

- the local machine has visible NVIDIA tooling
- basic hardware and toolchain state is reproducible
- memory-model estimates can be interpreted against real VRAM

## What this does not prove

- cuTile compatibility
- CUDA kernel correctness
- GPU attention performance
- model quality
- production inference throughput

## Next GPU milestone

The first GPU kernel should be a tiny paged lookup or KV-write primitive validated against Python and Rust fixtures, not a full attention kernel.
