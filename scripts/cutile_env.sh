#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CUDA_TOOLKIT_PATH:-}" ]]; then
  nvcc_path="$(command -v nvcc)"
  if [[ -z "$nvcc_path" ]]; then
    echo "error: nvcc was not found on PATH" >&2
    exit 1
  fi
  nvcc_path="$(readlink -f "$nvcc_path")"
  CUDA_TOOLKIT_PATH="${nvcc_path%/bin/nvcc}"
fi

if [[ ! -x "$CUDA_TOOLKIT_PATH/bin/nvcc" ]]; then
  echo "error: CUDA_TOOLKIT_PATH=$CUDA_TOOLKIT_PATH does not contain bin/nvcc" >&2
  exit 1
fi

export CUDA_TOOLKIT_PATH
echo "CUDA_TOOLKIT_PATH=$CUDA_TOOLKIT_PATH"
