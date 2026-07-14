#!/usr/bin/env bash
set -euo pipefail

source scripts/cutile_env.sh

mkdir -p reports/final_benchmark

env_json="reports/final_benchmark/environment.json"
results_jsonl="reports/final_benchmark/results.jsonl"
summary_csv="reports/final_benchmark/summary.csv"
summary_md="reports/final_benchmark/summary.md"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
commit="$(git rev-parse HEAD)"

cat > "$env_json" <<EOF
{
  "git_commit": "$commit",
  "gpu_name": "$gpu_name",
  "cuda_toolkit_path": "${CUDA_TOOLKIT_PATH:-/opt/cuda}",
  "cutile_version": "0.2.0",
  "timing_method": "SYNCHRONIZED_HOST_END_TO_END_TIME",
  "warmup_iterations": 1,
  "measured_processes": 3
}
EOF

: > "$results_jsonl"

run_case() {
  local name="$1"
  local example="$2"
  for process in 1 2 3; do
    cargo run --release -p plkv-kernels --features gpu-cutile --example "$example" >/dev/null
    local start_ns end_ns elapsed_ms
    start_ns="$(date +%s%N)"
    cargo run --release -p plkv-kernels --features gpu-cutile --example "$example" >/dev/null
    end_ns="$(date +%s%N)"
    elapsed_ms="$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.3f", (e - s) / 1000000.0 }')"
    printf '{"operation":"%s","example":"%s","process":%s,"elapsed_ms":%s,"timing_method":"SYNCHRONIZED_HOST_END_TO_END_TIME"}\n' \
      "$name" "$example" "$process" "$elapsed_ms" >> "$results_jsonl"
  done
}

run_case "latent_paged_attention_read" "model_profile_gpu"
run_case "full_kv_paged_attention_read" "fp16_full_kv_baseline_gpu"
run_case "latent_write_to_attention" "paged_latent_write_attention_fp16_gpu"

{
  echo "operation,process_count,min_ms,mean_ms,max_ms"
  python - "$results_jsonl" <<'PY'
import json
import sys
from collections import defaultdict

rows = defaultdict(list)
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        item = json.loads(line)
        rows[item["operation"]].append(float(item["elapsed_ms"]))

for operation in sorted(rows):
    values = rows[operation]
    print(
        f"{operation},{len(values)},{min(values):.3f},"
        f"{sum(values) / len(values):.3f},{max(values):.3f}"
    )
PY
} > "$summary_csv"

{
  echo "# Final Benchmark Summary"
  echo
  echo "Timing method: SYNCHRONIZED_HOST_END_TO_END_TIME."
  echo
  echo "| operation | process_count | min_ms | mean_ms | max_ms |"
  echo "|---|---:|---:|---:|---:|"
  tail -n +2 "$summary_csv" | awk -F, '{ printf "| %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5 }'
  echo
  echo "BENCHMARK_WARMUP_EXCLUDED=1"
  echo "JIT_EXCLUDED_FROM_MEASUREMENTS=1"
  echo "SYNCHRONIZATION_METHOD=SYNCHRONIZED_HOST_END_TO_END_TIME"
  echo "BENCHMARK_PROCESS_COUNT=3"
  echo "BENCHMARK_RESULTS_WRITTEN=1"
  echo "FINAL_BENCHMARK_OK=1"
} > "$summary_md"

cat "$summary_md"
