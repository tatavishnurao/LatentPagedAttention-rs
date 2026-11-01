# Instrument checks (laptop RTX 4060, WSL2, not a cloud result)

Single harness processes run during the campaign (2026-10-31 to 2026-11-01) **to check the timing instrument, not to measure the kernels.** They are copied here unchanged from the session scratch area and `~/runs/` so that the paper's measurement-validity claims cite committed files. They were not re-run. **None of them is evidence for any C3/A3 ratio**, and no ratio from them is used anywhere.

Each directory has the harness's `timing_meta.json`, `correctness.json` (where written) and `samples.jsonl.xz` (xz-compressed, otherwise byte-identical).

| Directory | Harness state | What it shows |
|---|---|---|
| `e4_B1_N1024_dryrun_VOID` | before any fix (commit 83ee786; per-sample NVML reasons query) | E4 layers mostly at 210 MHz; the dry run is VOID |
| `e4_B1_N1024_10it_fix1` | fix 1 (reasons once per step, re-warm with a sync per burst; committed as 507dea2), **10 iterations** | in band at 1890 MHz, i.e. the short check passed |
| `e3_N1024_50it_prefix`, `e3_N8192_50it_prefix` | E3 still with the per-sample reasons query (between 507dea2 and 92e1411), 50 iterations | N=1024 drops off the reference clock after about 15 iterations; N=8192 does not |
| `e3_N1024_50it_fix1`, `e3_N8192_50it_fix1` | fix 1 applied to E3 (92e1411), 50 iterations | N=1024 mostly in band, with an under-load drop near the end |
| `e4_B1_N1024_50it_fix2`, `e4_B8_N4096_50it_fix2` | fix 2 (saturating re-warm, efece3c; harness `e3e4-rewarm-saturating-20ms-2026-11-01`), 50 iterations | the full-length check that cleared the primary E4 blocks |

The full-length failure of fix 1 is the campaign run `runs/local4060_b2r_e4n1024_20261101_051528` (instrument failure).

Not here: the ~20 ms latency of `nvmlDeviceGetCurrentClocksEventReasons` was measured interactively (a ctypes loop, 50 calls) and its output was not saved to a file.
