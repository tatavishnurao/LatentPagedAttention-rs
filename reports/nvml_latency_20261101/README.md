# NVML host-call latency (laptop RTX 4060, WSL2, not a cloud result)

**A separate micro-measurement, not campaign data.** It re-measures, into a committed file, the latency of
`nvmlDeviceGetCurrentClocksEventReasons` that was first measured interactively during the campaign (output not
saved then; see `../local_4060_20261101/instrument_checks/README.md`). No kernel is launched and no CUDA context
is created: each NVML query is called 50 times back to back from the CPU and each call is timed with
`time.perf_counter_ns`.

- Command: `python3 scripts/paper/nvml_latency.py --out reports/nvml_latency_20261101/nvml_latency.json`
- Date (UTC), host (`uname`), GPU, driver, GPU temperature and SM clock before and after, git commit, and every
  per-call time are in `nvml_latency.json`.
- Run on 2026-11-01 with the GPU idle (no compute processes, 62 C, SM at 210 MHz before and after).

Result (from the file): the reasons query has a median of 20.8 ms (max 29.0 ms); the SM/memory clock and
temperature reads have medians of 0.03-0.15 ms. This agrees with the interactive measurement quoted in
`docs/LOCAL_4060_RESULTS.md` (median 20.6 ms, max 30 ms).

Not established: whether the latency depends on GPU load or clock state (measured idle only), or how it behaves
on native Linux (no native host was available).
