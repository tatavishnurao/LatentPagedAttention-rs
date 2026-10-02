#!/usr/bin/env bash
# E1: focused single-launch ncu on current A1 (full) and B1/C1 (latent) context kernels.
# Cool-start only (temp < 80 C, wait <= 300 s, else skip); abort if >= 91 C.
set -uo pipefail
ROOT=<repo>
OUT=<home>/plkv_runs/e1
mkdir -p "$OUT"
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3 LIBCLANG_PATH=/usr/lib/llvm-18/lib \
  CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.3/bin/tileiras OPENBLAS_NUM_THREADS=1
NCU=/usr/local/cuda-13.3/bin/ncu
METRICS=gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,launch__grid_size,launch__block_size,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,dram__bytes_read.sum,lts__t_bytes.sum,lts__t_sector_hit_rate.pct,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio

temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d ' '; }
cool() {
  local s=$SECONDS
  while (( $(temp) >= 80 )); do
    (( SECONDS - s >= 300 )) && return 1
    sleep 5
  done
  return 0
}

for seq in 1024 32768; do
  for kname in "regex:full_kv_context.*_${seq}_entry\$" "regex:^model_small_context_fp16_storage_rtable_${seq}_entry\$"; do
    tag=$(echo "$kname" | sed 's/regex://; s/[^a-z0-9_]//g')
    if ! cool; then echo "SKIP $seq $tag: not cool-start (temp $(temp) C)"; continue; fi
    echo "RUN $seq $tag start_temp=$(temp)"
    nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active \
      --format=csv,noheader,nounits -lms 100 > "$OUT/tele_${seq}_${tag}.csv" &
    LOG=$!
    timeout 300 "$NCU" --kernel-name "$kname" -c 1 --metrics "$METRICS" --csv \
      "$ROOT/target/release/examples/c1_rtable" --seq "$seq" --warmup 0 --iterations 1 \
      --output-dir "$OUT/run_${seq}_${tag}" > "$OUT/ncu_${seq}_${tag}.csv" 2> "$OUT/ncu_${seq}_${tag}.err" &
    NPID=$!
    while kill -0 $NPID 2>/dev/null; do
      t=$(temp); if (( t >= 91 )); then echo "ABORT 91C"; kill $NPID; fi; sleep 0.5
    done
    wait $NPID; echo "rc=$? max_temp=$(cut -d, -f2 "$OUT/tele_${seq}_${tag}.csv" | sort -n | tail -1)"
    kill $LOG
  done
done
