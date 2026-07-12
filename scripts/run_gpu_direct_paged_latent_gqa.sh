#!/usr/bin/env bash
set -euo pipefail
source scripts/cutile_env.sh
mkdir -p reports/rtx4060_gpu_smoke
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="reports/rtx4060_gpu_smoke/direct_paged_latent_gqa_${stamp}.txt"
{
  echo "LatentPagedAttention-rs RTX 4060 direct Paged Latent GQA validation"
  echo "generated_utc=$(date -u --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.used,memory.free --format=csv,noheader
  cargo run --release -p plkv-kernels --features gpu-cutile --example direct_paged_latent_gqa_gpu
} 2>&1 | tee "$out"
grep -q "DIRECT_PAGED_LATENT_GQA_GPU_OK=1" "$out"
grep -q "GPU_CPU_DIRECT_PAGED_SCORES_MATCH=1" "$out"
grep -q "GPU_CPU_DIRECT_PAGED_CONTEXT_MATCH=1" "$out"
grep -q "NON_IDENTITY_MAPPING_EFFECT_CONFIRMED=1" "$out"
grep -q "NO_LOGICAL_LATENT_MATERIALIZATION_CONFIRMED=1" "$out"
grep -q "NO_FULL_KV_MATERIALIZATION_CONFIRMED=1" "$out"
echo "GPU_REPORT=$out"
