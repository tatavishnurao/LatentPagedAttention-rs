#!/usr/bin/env bash
set -euo pipefail

source scripts/cutile_env.sh

mkdir -p reports/rtx4060_gpu_smoke

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="reports/rtx4060_gpu_smoke/runtime_sequence_${stamp}.txt"

{
  echo "LatentPagedAttention-rs RTX 4060 runtime sequence validation"
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
    --example runtime_sequence_gpu
} 2>&1 | tee "$out"

grep -q "RUNTIME_SEQUENCE_GPU_OK=1" "$out"
grep -q "RUNTIME_ACTIVE_SEQUENCE_LENGTH_CONFIRMED=1" "$out"
grep -q "PARTIAL_FINAL_BLOCK_MASKING_CONFIRMED=1" "$out"
grep -q "INACTIVE_PROBABILITIES_ZERO=1" "$out"
grep -q "ACTIVE_PROBABILITY_ROWS_SUM_TO_ONE=1" "$out"
grep -q "RUNTIME_BLOCK_TABLE_MAPPING_CONFIRMED=1" "$out"
grep -q "NO_INACTIVE_TOKEN_CONTEXT_CONTRIBUTION=1" "$out"

echo "GPU_REPORT=$out"
