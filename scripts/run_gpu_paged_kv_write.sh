#!/usr/bin/env bash
set -euo pipefail

source scripts/cutile_env.sh

mkdir -p reports/rtx4060_gpu_smoke
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="reports/rtx4060_gpu_smoke/paged_kv_write_${stamp}.txt"

{
  echo "LatentPagedAttention-rs RTX 4060 paged KV-write validation"
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
    --example paged_kv_write_gpu
} 2>&1 | tee "$out"

grep -q "PAGED_KV_WRITE_GPU_OK=1" "$out"
echo "GPU_REPORT=$out"
