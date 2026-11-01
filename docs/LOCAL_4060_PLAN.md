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
