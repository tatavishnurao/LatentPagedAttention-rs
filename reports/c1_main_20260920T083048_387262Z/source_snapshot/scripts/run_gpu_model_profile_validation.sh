#!/usr/bin/env bash
set -euo pipefail

source scripts/cutile_env.sh

mkdir -p reports/rtx4060_gpu_smoke

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="reports/rtx4060_gpu_smoke/model_profile_${stamp}.txt"

{
  echo "LatentPagedAttention-rs RTX 4060 model-shaped profile validation"
  echo "generated_utc=$(date -u --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD)"

  nvidia-smi \
    --query-gpu=name,compute_cap,driver_version,memory.total,memory.used,memory.free \
    --format=csv,noheader

  echo

  cargo run \
    --release \
    -p plkv-kernels \
    --features gpu-cutile \
    --example model_profile_gpu
} 2>&1 | tee "$out"

grep -q "PROFILE=model_small" "$out"
grep -q "MODEL_SHAPED_PROFILE_GPU_OK=1" "$out"
grep -q "MODEL_SHAPED_PARTIAL_BLOCK_OK=1" "$out"
grep -q "MODEL_SHAPED_MAX_SEQUENCE_OK=1" "$out"
grep -q "MODEL_SHAPED_NON_IDENTITY_MAPPING_OK=1" "$out"

echo "GPU_REPORT=$out"
