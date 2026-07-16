# Reproducibility

This is the exact command reference for the frozen `v0.1.x` release. Most
users should start with:

```bash
bash scripts/validate_release.sh
```

That runs CPU and repository checks. On the validated RTX 4060 environment, the
full GPU validation suite is available with:

```bash
bash scripts/validate_release.sh --gpu
```

Maintainers can use the individual scripts below to isolate a regression. The
unified entry point calls existing scripts and does not replace their reports.

## Environment

- NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9
- CUDA toolkit path: `/opt/cuda`
- cuTile `0.2.0`
- Rust workspace version `0.1.0`
- Python environment selected with `UV_PROJECT_ENVIRONMENT=attention99`

## Core environment and smoke

```bash
UV_PROJECT_ENVIRONMENT=attention99 uv sync
UV_PROJECT_ENVIRONMENT=attention99 uv run pytest -q
UV_PROJECT_ENVIRONMENT=attention99 uv run ruff check .
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
source scripts/cutile_env.sh
cargo check -p plkv-kernels --features gpu-cutile --examples
bash scripts/run_cutile_smoke.sh
```

Supporting environment and reference commands:

```bash
bash scripts/env_check.sh
bash scripts/rtx4060_env_snapshot.sh
bash scripts/rtx4060_memory_sanity.sh
bash scripts/run_memory_model.sh
bash scripts/run_reference_benchmarks.sh
bash scripts/generate_golden_fixtures.sh
```

## Paging primitives

```bash
bash scripts/run_gpu_paged_lookup.sh
bash scripts/run_gpu_paged_kv_write.sh
```

## Attention correctness

```bash
bash scripts/run_gpu_gqa_decode.sh
bash scripts/run_gpu_paged_gqa_decode.sh
```

## Latent-cache paths

```bash
bash scripts/run_gpu_latent_kv_reconstruction.sh
bash scripts/run_gpu_direct_latent_gqa.sh
bash scripts/run_gpu_direct_paged_latent_gqa.sh
bash scripts/run_gpu_paged_latent_write_attention.sh
```

## Precision validation

```bash
bash scripts/run_gpu_fp16_paged_latent_attention.sh
bash scripts/run_gpu_fp16_full_kv_baseline.sh
```

## Runtime and model-shaped profiles

```bash
bash scripts/run_gpu_runtime_sequence_validation.sh
bash scripts/run_gpu_model_profile_validation.sh
```

## Final benchmark

```bash
bash scripts/run_final_benchmark.sh
```

The canonical benchmark artifacts are committed in
`reports/final_benchmark/summary.csv` and `summary.md`. Timing is synchronized
host end-to-end timing, not kernel-only latency. The benchmark script writes
fresh report files, so it is intentionally separate from release validation.
GPU validation is manual because standard CI has no GPU runner.
