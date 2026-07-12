#!/usr/bin/env bash
set -euo pipefail

source scripts/cutile_env.sh

mkdir -p reports/rtx4060_gpu_smoke

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="reports/rtx4060_gpu_smoke/fp16_paged_latent_attention_${stamp}.txt"

{
  echo "LatentPagedAttention-rs RTX 4060 FP16 paged latent-storage validation"
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
    --example paged_latent_write_attention_fp16_gpu
} 2>&1 | tee "$out"

grep -q "FP16_PAGED_LATENT_ATTENTION_GPU_OK=1" "$out"
grep -q "FP16_CACHE_BIT_MISMATCH_COUNT=0" "$out"
grep -q "GPU_F32_TO_FP16_WRITE_CONVERSION_CONFIRMED=1" "$out"
grep -q "GPU_FP16_STORAGE_CPU_MATCH=1" "$out"
grep -q "FP16_UNCHANGED_REGION_BITS_OK=1" "$out"
grep -q "GPU_WRITE_TO_ATTENTION_NO_HOST_ROUNDTRIP_CONFIRMED=1" "$out"

echo "GPU_REPORT=$out"
