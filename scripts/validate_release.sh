#!/usr/bin/env bash
set -euo pipefail

run_step() {
  if [[ "${VALIDATE_RELEASE_TEST_MODE:-0}" == "1" ]]; then
    printf 'VALIDATION_TEST_COMMAND='
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_cpu_validation() {
  run_step env UV_PROJECT_ENVIRONMENT=attention99 uv sync
  run_step env UV_PROJECT_ENVIRONMENT=attention99 uv run pytest -q
  run_step env UV_PROJECT_ENVIRONMENT=attention99 uv run ruff check .
  run_step cargo fmt --all --check
  run_step cargo test --workspace
  run_step cargo clippy --workspace --all-targets -- -D warnings
  run_step git diff --check
  echo "CPU_RELEASE_VALIDATION_OK=1"
}

run_gpu_validation() {
  if [[ "${VALIDATE_RELEASE_TEST_MODE:-0}" == "1" ]]; then
    echo "GPU_VALIDATION_START=1"
  else
    source scripts/cutile_env.sh
  fi
  run_step cargo check -p plkv-kernels --features gpu-cutile --examples
  run_step bash scripts/run_cutile_smoke.sh
  run_step bash scripts/run_gpu_paged_lookup.sh
  run_step bash scripts/run_gpu_paged_kv_write.sh
  run_step bash scripts/run_gpu_gqa_decode.sh
  run_step bash scripts/run_gpu_paged_gqa_decode.sh
  run_step bash scripts/run_gpu_latent_kv_reconstruction.sh
  run_step bash scripts/run_gpu_direct_latent_gqa.sh
  run_step bash scripts/run_gpu_direct_paged_latent_gqa.sh
  run_step bash scripts/run_gpu_paged_latent_write_attention.sh
  run_step bash scripts/run_gpu_fp16_paged_latent_attention.sh
  run_step bash scripts/run_gpu_runtime_sequence_validation.sh
  run_step bash scripts/run_gpu_model_profile_validation.sh
  run_step bash scripts/run_gpu_fp16_full_kv_baseline.sh
  echo "GPU_RELEASE_VALIDATION_OK=1"
}

case "${1:-}" in
  "")
    run_cpu_validation
    ;;
  --gpu)
    run_cpu_validation
    run_gpu_validation
    ;;
  *)
    printf 'usage: %s [--gpu]\n' "$0" >&2
    exit 2
    ;;
esac

echo "RELEASE_VALIDATION_OK=1"
