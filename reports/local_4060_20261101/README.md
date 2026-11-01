# Local E3 / E4 timing campaign, 2026-11-01

**Host: laptop RTX 4060 (8 GB, sm_89), WSL2, driver 617.42, CUDA 13.3. Not a cloud result, not an L4 result.**

Write-up: [`docs/LOCAL_4060_RESULTS.md`](../../docs/LOCAL_4060_RESULTS.md). Which runs count, fixed before the data: [`docs/LOCAL_4060_PLAN.md`](../../docs/LOCAL_4060_PLAN.md). Analysis rules: `docs/E0B_CLOUD_RUNBOOK.md` section 4.

## Contents

- `raw/<run>.tar.xz`: each complete results directory (all samples, per-process telemetry, logs), without oracle dumps.
- `runs/<run>/`: the small files people actually read: `environment.txt`, `manifest.json` (per-process start temperature, cooldown, host check), `variants_analysis*.json`, `e3e4_campaign.log`, `correctness_summary.txt`, and status labels (`ABORTED.txt`, `INSTRUMENT_FAILURE.txt`, `ANALYSIS_REGENERATED.txt`).
- `primary_merged_analysis.json`: the primary per-configuration results copied from the primary runs into one file for the figures. No processes are pooled across runs.
- `figures/`: rendered with `scripts/paper/` (matplotlib from a throwaway `uv run --no-project --with matplotlib`; the project lockfile is unchanged).

## Runs

| Run | What | Status | Commit (manifest) | Harness version |
|---|---|---|---|---|
| `local4060_b1_e3_20261101_043331` | E3 N=1024, 8192 | preliminary (Brave open, 0 % GPU); reported beside the re-run, never pooled | 92e1411 | pre-2026-11-01 |
| `local4060_b1r_e3_20261101_051132` | E3 N=1024, 8192 | **primary E3** | 86c4d2e (binary built at 92e1411) | pre-2026-11-01 |
| `local4060_b2_e4n1024_20261101_050107` | E4 N=1024 | **aborted** after 7 processes (Brave closed mid-run); not analysed | 92e1411 | pre-2026-11-01 |
| `local4060_b2r_e4n1024_20261101_051528` | E4 N=1024 | **instrument failure** (clock collapse under the old re-warm); ratios not used | 86c4d2e | pre-2026-11-01 |
| `local4060_e4n4096_20261101_054159` | E4 N=4096, B=1,8,32 | **primary E4** | c3c900c | e3e4-rewarm-saturating-20ms-2026-11-01 |
| `local4060_e4n8192_20261101_054853` | E4 N=8192, B=1,8,32 | **primary E4** | c3c900c | same |
| `local4060_e4n1024v2_20261101_055641` | E4 N=1024, B=1,8,32 | **primary E4** | c3c900c | same |
| `local4060_e3a3sens_20261101_063414` | E3 N=8192 + unpadded A3 | **secondary** sensitivity run (plan addendum 2) | 2ffd88e | ...+a3unpad |

Dry runs (`~/runs/dry_*`, not copied) are never evidence; `dry_20261031_194558` is VOID (pre-fix).

## Command (primary E4 blocks; E3 and the sensitivity run differ only in the stage flags)

```
bash scripts/e0b/run_e0b_portable.sh --out-dir ~/runs/local4060_e4n<N>_<stamp> --allow-wsl --skip-e0b \
  --e3-lengths "" --e4-batches 1,8,32 --e4-seqs <N> --max-footprint-gib 6 --min-cool 5 --max-attempts 8
# E3: --e3-lengths 1024,8192 --e4-batches ""   sensitivity: --e3-lengths 8192 --e4-batches "" --e3-a3-unpadded
```

Defaults kept: cool-start 80 C, cooldown up to 300 s, abort 91 C, 50 timed iterations, 10 warm-up.

## Host conditions

AC power and the Windows "High performance" plan (checked through `powershell.exe`); EA Desktop closed. Brave: open during Block 1 (0 % GPU engine use in a 5 s check), reopened itself during the B32 tail of the Block 2 re-run (05:30:49 UTC), and was checked closed before **every** process of the primary E4 blocks and the sensitivity run (`host_check` in each manifest). Surface, cooling pad and room temperature were reported by the user and not verified.

## Figure captions

- `fig3_regime_map.png`: C3 accurate / padded A3, primary dataset, vs padded-A3 bytes per layer. Filled = claim-eligible (>= 5 clean processes); hollow = not. E3 points (squares) are from the old harness version; E4 points from the new one. **The L2 line is a reference only: E4 flushes 64 MiB before every layer and each KV block is read once per layer, so every E4 layer starts cold whatever its size.** Configurations with 0 clean processes have no point.
- `fig2_roofline_e4.png`: clean median step time vs analytic bytes per step for E4. The 256 GB/s floor is the nominal spec of the RTX 4060 Laptop's memory interface, **not measured** on this host.
- `fig1_clock_trace_e4_B32_N8192_clean.png`: SM clock and per-layer latency for a clean process (in band; note the y-axis spans only 2580-2595 MHz). The first ~30 samples are ~1.9 ms for both variants against ~1.3 ms afterwards (start-of-timing transient).
- `fig1_clock_trace_e4_B1_N1024_instrument_failure.png`: the same plot for a Block 2 re-run process under the old re-warm, where the clock oscillated between ~300 and ~1550 MHz.
