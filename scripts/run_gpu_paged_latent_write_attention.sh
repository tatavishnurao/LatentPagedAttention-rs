#!/usr/bin/env bash
set -euo pipefail

source scripts/cutile_env.sh

mkdir -p reports/rtx4060_gpu_smoke

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="reports/rtx4060_gpu_smoke/paged_latent_write_attention_${stamp}.txt"

{
  echo "LatentPagedAttention-rs RTX 4060 paged latent write-to-attention validation"
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
    --example paged_latent_write_attention_gpu
} 2>&1 | tee "$out"

grep -q "PAGED_LATENT_WRITE_ATTENTION_GPU_OK=1" "$out"
grep -q "GPU_WRITE_CPU_MATCH=1" "$out"
grep -q "GPU_POST_WRITE_ATTENTION_CPU_MATCH=1" "$out"
grep -q "WRITTEN_TOKEN_SCORE_COLUMN_CHANGED=1" "$out"
grep -q "LATENT_UNCHANGED_REGION_OK=1" "$out"
grep -q "WRITE_NON_IDENTITY_MAPPING_EFFECT_CONFIRMED=1" "$out"
grep -q "GPU_WRITE_TO_ATTENTION_NO_HOST_ROUNDTRIP_CONFIRMED=1" "$out"

echo "GPU_REPORT=$out"
