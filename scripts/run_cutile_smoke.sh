#!/usr/bin/env bash
set -euo pipefail

source scripts/cutile_env.sh

mkdir -p reports/rtx4060_gpu_smoke
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="reports/rtx4060_gpu_smoke/cutile_smoke_${stamp}.txt"

{
  echo "LatentPagedAttention-rs cuTile smoke test"
  echo "generated_utc=$(date -u --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.used,memory.free --format=csv,noheader
  echo
  cargo run --release -p plkv-kernels --features gpu-cutile --example cutile_smoke
} 2>&1 | tee "$out"

grep -q "CUTILE_SMOKE_OK=1" "$out"
echo "GPU_REPORT=$out"
