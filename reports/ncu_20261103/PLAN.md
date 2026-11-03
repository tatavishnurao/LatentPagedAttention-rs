# Nsight Compute profile at Step B8·N8192: declared before any profile was run

Written and committed on 2026-11-03 **before** `ncu` was run on any kernel, so the choice of what
to report cannot follow the numbers. Host: laptop RTX 4060, WSL2, not a cloud result.

## What is profiled

- Binary: `target/release/examples/e3_splitk_gpu` (SHA-256 recorded in `README.md` after the run;
  kernels identical to the E4 campaign build: `crates/plkv-kernels/src` is unchanged since
  `c3c900c`, and `2ffd88e` changed only the Single/E3 harness).
- Command: `--suite e4-timing --batch 8 --seq 8192` with a short timing loop
  (`--iterations 1 --warmup 1`): the profile needs one launch per kernel, not timing.
- Kernels (Step runs GQA-pad and Latent only): `a3_split_pad`, `a3_reduce_pad` (GQA-pad);
  `c3_split` in accurate mode, `c3_reduce` (Latent). One launch of each: `--launch-skip 1
  --launch-count 1` per kernel name, so the very first launch of each kernel is not the one profiled.
- Two clock settings, both recorded: `--clock-control base` and the default (`boost`, per the
  Nsight Compute CLI documentation, cited as [12] in the paper).
- Cache control left at the `ncu` default (caches flushed before each profiled pass), which matches
  Step's cold-L2 design.

## Metrics to report (all of them, whatever they show)

Per kernel and clock setting:

1. `gpu__time_duration.sum`: kernel duration under the profiler.
2. `dram__bytes_read.sum` and `dram__bytes_write.sum`: measured DRAM traffic.
3. `dram__throughput.avg.pct_of_peak_sustained_elapsed`: DRAM throughput as % of peak.
4. `sm__warps_active.avg.pct_of_peak_sustained_active`: achieved occupancy.
5. `sm__maximum_warps_per_active_cycle_pct`: theoretical occupancy (cross-check against SASS).
6. `launch__grid_size`, `launch__block_size`, `launch__registers_per_thread`.
7. `sm__throughput.avg.pct_of_peak_sustained_elapsed`: SM throughput as % of peak.
8. `sm__cycles_elapsed.avg.per_second`: the SM clock during profiling (checks `--clock-control`).
9. The three largest warp-stall reasons from the WarpStateStats section
   (`smsp__average_warps_issue_stalled_*_per_issue_active.ratio`), ranked by value.

## Rules fixed now

- No time ratio between kernels is computed from profiler durations: the profiler serializes and
  replays kernels, so its durations are not comparable to the campaign's timing.
- Measured DRAM bytes are compared with the analytic bytes only as a check of the analytic count.
- Any explanation of what limits a kernel stays labelled HYPOTHESIS; the profile is one launch of
  each kernel in one configuration.
- If `ncu` cannot access the performance counters (for example `ERR_NVGPUCTRPERM`, as on
  2026-10-02), this is recorded in one line and the profile is skipped. No setting is changed to
  work around it.
