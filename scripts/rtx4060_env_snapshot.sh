#!/usr/bin/env bash
set -euo pipefail

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-attention99}"

report_dir="reports/rtx4060_baseline"
timestamp="$(date +%Y%m%d_%H%M%S)"
report_path="${report_dir}/env_snapshot_${timestamp}.txt"
mkdir -p "${report_dir}"

{
  echo "RTX 4060 environment snapshot"
  echo "============================="
  echo
  echo "Timestamp:"
  date || true
  echo
  echo "Git commit:"
  git rev-parse HEAD || true
  echo
  echo "OS:"
  uname -a || true
  cat /etc/os-release || true
  echo
  echo "Rust:"
  rustc --version || true
  cargo --version || true
  echo
  echo "uv and Python:"
  uv --version || true
  uv run python --version || true
  echo
  echo "NVIDIA driver, GPU table, and CUDA version:"
  nvidia-smi || true
  echo
  echo "GPU query:"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,power.limit,temperature.gpu --format=csv || true
  echo
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo
  echo "nvcc:"
  nvcc --version || true
} 2>&1 | tee "${report_path}"

echo "Saved environment snapshot to ${report_path}"
