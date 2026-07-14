#!/usr/bin/env bash
set -euo pipefail

source scripts/cutile_env.sh

mkdir -p reports/rtx4060_gpu_smoke

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="reports/rtx4060_gpu_smoke/fp16_full_kv_baseline_${stamp}.txt"

{
  echo "LatentPagedAttention-rs RTX 4060 FP16 full-KV baseline validation"
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
    --example fp16_full_kv_baseline_gpu
} 2>&1 | tee "$out"

grep -q "FP16_FULL_KV_BASELINE_GPU_OK=1" "$out"
grep -q "FP16_LATENT_PATH_GPU_OK=1" "$out"
grep -q "BASELINE_AND_LATENT_USE_SAME_STORAGE_WIDTH=1" "$out"
grep -q "CACHE_BYTE_ACCOUNTING_CONFIRMED=1" "$out"

echo "GPU_REPORT=$out"
