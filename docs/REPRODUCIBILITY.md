# Reproducibility

Validated local environment:

- NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9
- CUDA toolkit path: `/opt/cuda`
- cuTile `0.2.0`
- Rust workspace version `0.1.0`
- Python environment selected with `UV_PROJECT_ENVIRONMENT=attention99`

Core commands:

```bash
UV_PROJECT_ENVIRONMENT=attention99 uv sync
UV_PROJECT_ENVIRONMENT=attention99 uv run pytest -q
UV_PROJECT_ENVIRONMENT=attention99 uv run ruff check .
cargo fmt --all --check
cargo test --workspace
source scripts/cutile_env.sh
cargo check -p plkv-kernels --features gpu-cutile --examples
bash scripts/run_gpu_runtime_sequence_validation.sh
bash scripts/run_gpu_model_profile_validation.sh
bash scripts/run_gpu_fp16_full_kv_baseline.sh
bash scripts/run_final_benchmark.sh
```

GPU validation is manual. Standard CI is CPU-only.
