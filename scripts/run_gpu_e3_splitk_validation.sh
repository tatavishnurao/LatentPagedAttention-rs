#!/usr/bin/env bash
set -euo pipefail

source scripts/cutile_env.sh

mkdir -p reports/rtx4060_gpu_smoke

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="reports/rtx4060_gpu_smoke/e3_splitk_${stamp}.txt"

{
  echo "LatentPagedAttention-rs E3 split-K correctness validation (no timing)"
  echo "generated_utc=$(date -u --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD)"

  nvidia-smi \
    --query-gpu=name,compute_cap,driver_version,temperature.gpu,memory.total,memory.used \
    --format=csv,noheader

  echo

  cargo run \
    --release \
    -p plkv-kernels \
    --features gpu-cutile \
    --example e3_splitk_gpu
} 2>&1 | tee "$out"

for marker in \
  E3_ORACLE_CHAIN_OK E3_TINY_C3_OK E3_TINY_A3_OK E3_MODEL_C3_OK E3_MODEL_A3_OK \
  E3_PARTIAL_BLOCK_OK E3_SPLIT_INVARIANCE_OK E3_NON_IDENTITY_TABLE_OK E3_SPLITK_GPU_OK; do
  grep -q "^${marker}=1$" "$out"
done

echo "GPU_REPORT=$out"
