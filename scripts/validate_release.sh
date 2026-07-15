#!/usr/bin/env bash
set -euo pipefail

run_cpu_validation() {
  UV_PROJECT_ENVIRONMENT=attention99 uv sync
  UV_PROJECT_ENVIRONMENT=attention99 uv run pytest -q
  UV_PROJECT_ENVIRONMENT=attention99 uv run ruff check .
  cargo fmt --all --check
  cargo test --workspace
  cargo clippy --workspace --all-targets -- -D warnings
  git diff --check
  echo "CPU_RELEASE_VALIDATION_OK=1"
}

run_gpu_validation() {
  source scripts/cutile_env.sh
  cargo check -p plkv-kernels --features gpu-cutile --examples
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
  bash scripts/run_gpu_runtime_sequence_validation.sh
  bash scripts/run_gpu_model_profile_validation.sh
  bash scripts/run_gpu_fp16_full_kv_baseline.sh
  echo "GPU_RELEASE_VALIDATION_OK=1"
}

run_cpu_validation

if [[ "${1:-}" == "--gpu" ]]; then
  run_gpu_validation
elif [[ $# -ne 0 ]]; then
  printf 'usage: %s [--gpu]\n' "$0" >&2
  exit 2
fi

echo "RELEASE_VALIDATION_OK=1"
