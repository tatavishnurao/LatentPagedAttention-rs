# Local RTX 4060 campaign: which runs count (fixed 2026-11-01, before the re-runs)

Host: laptop RTX 4060, WSL2, not a cloud result. Analysis rules: `docs/E0B_CLOUD_RUNBOOK.md` section 4, unchanged.

## Why there are re-runs

The thermal/host checklist asks for all other GPU-capable apps to be closed. Two runs started before that was fully met:

- **Block 1** (`~/runs/local4060_b1_e3_20261101_043331`, E3 N=1024/8192): EA Desktop was closed, but Brave was open (0 % GPU engine use in a 5 s check before the run).
- **Block 2** (`~/runs/local4060_b2_e4n1024_*`, E4 N=1024): Brave was open at the start and closed during the run. It was stopped deliberately after 7 processes and is marked `ABORTED.txt`. Its timing data was not analyzed before this file was written.

## Rule

1. **Primary dataset:** the blocks re-run with the full checklist met (AC power, High performance plan, EA Desktop and Brave closed). Every configuration is reported from these runs, with its clean / THROTTLED / hot-start counts as they come out.
2. **Block 1 is kept and reported next to the re-run as a preliminary run.** It is never pooled with the re-run. If the two disagree (for example one is claim-eligible and the other is not, or the CIs do not overlap), the disagreement is reported, not resolved by choosing one.
3. **Aborted Block 2 is descriptive only** and is not pooled.
4. **Each block is run once under the primary conditions.** A configuration with too few clean processes is reported as "not enough clean processes"; it is not re-run until it passes.
5. This decision was made after Block 1's results were seen. Rule 2 is the safeguard: both runs are published.

## Addendum 2026-11-01 (after the Block 2 re-run, before any new data)

Fixed before the harness change and before any data from it.

1. **The E4 harness fix is an instrument change.** In the Block 2 re-run (`~/runs/local4060_b2r_e4n1024_20261101_051528`) the SM clock fell to about 400-1200 MHz and the re-warm failed in 91-100 % of steps. That run is reported as an **instrument failure**. It is not re-analysed and its ratios are not used. E3 shares the timing helpers, so E3 data taken before the change (Block 1 and its re-run) is labelled with the old harness version.
2. **Harness version.** Every process writes `harness_version` into `timing_meta.json`; the new value is `e3e4-rewarm-saturating-20ms-2026-11-01` (older output has none and is labelled `pre-2026-11-01`). The analyzer refuses to pool processes with different versions in one configuration.
3. **Re-warm change (option A):** about 20 ms of back-to-back enqueued work with no synchronization in between, then one sync and one clock read, repeated until the SM clock is within 95 % of the warm-up reference or 500 ms pass.
4. **Instrument check, no ratios:** one full 50-iteration process each of E4 B8_N4096 and B1_N1024. Only the in-band fraction and the clock trace are reported. If either is not near 1.0 (below 0.95), stop and report; no block runs.
5. **Block order under the new version:** E4 N=4096, then N=8192, then N=1024 last. A report after each block. Each block runs once; a configuration with fewer than 5 clean processes is reported as "not enough clean processes".
6. **The E4 re-run under the new harness version is primary for E4.**
7. **New exclusion, absolute reference guard:** a process is THROTTLED if its warm-up reference clock is more than 5 % below the highest warm-up reference among the processes of the same configuration in the same campaign (`reference < 0.95 x max reference`). This closes the gap where two processes warmed up at 915 and 1185 MHz and passed the reference rule against their own low reference. It applies to E3 and E4 and only adds an exclusion. It is also applied to the E3 data already reported; any change that causes is reported.
8. **Host check before every process, not every block:** before each process the campaign checks (through `powershell.exe Get-Process`) that Brave, EA Desktop and EALocalHostSvc are not running. If one is, the campaign stops with `HOST_CHECK_FAILED` before starting that process. No process starts under a failed check.
9. **Optional sensitivity run afterwards:** unpadded A3 as a fourth E3 variant at N=8192. It is a sensitivity analysis against a stronger baseline (padded A3 spends about 4x the multiply-adds on zero rows). It does not replace padded A3 as the pre-declared baseline and is reported whatever it shows.

## Addendum 2: unpadded-A3 sensitivity run (2026-11-01, before any data from it)

Written after the primary E3 and E4 results were seen (E3 N=8192: C3 accurate / padded A3 = 0.893 [0.860, 0.936]).

1. **Secondary analysis, never primary.** It does not replace padded A3 as the pre-declared baseline and is reported separately from the primary runs, whatever it shows.
2. **What runs:** E3 timing at N=8192 only, with a fourth variant, unpadded A3 (`a3`: the `a3_split` / `a3_reduce` kernels the correctness suite already checks, no zero query rows). Same campaign rules: >= 5 cool-start processes, `--max-attempts 8`, all exclusion rules unchanged, host check before every process, one run.
3. **Own harness version:** `e3e4-rewarm-saturating-20ms-2026-11-01+a3unpad`. It is never pooled with the primary E3 data (the analyzer refuses mixed versions).
4. **What is reported:** C3 accurate / unpadded A3 and C3 fast / unpadded A3 (geomean, 95 % hierarchical-bootstrap CI, claim eligibility), plus C3 / padded A3 within the same run for comparison.
5. **Interpretation fixed now:** if C3 beats padded A3 but its CI against unpadded A3 includes or exceeds 1.0, the E3 "about 11 % faster" result depends on the baseline, and the write-up says so in the verdict paragraph. If C3 also beats unpadded A3, the write-up says the result holds against both baselines on this laptop at N=8192 only.
6. **Bytes:** unpadded A3 reads the same KV bytes as padded A3 (padding is in the query rows only); no separate byte count is reported for it.
