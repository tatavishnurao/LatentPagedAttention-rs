# Main budget freeze

Pilot: `reports/c1_pilot_20260920T082449_625698Z` completed successfully,
including Rust GPU/CPU checks and independent FP64 reconstructed-K/V oracle.
One process, 1K, one warmup, three samples per phase; excluded from main results.
Cooldown reached 78 C without changing thresholds. Main budget remains exactly
as predeclared: 3 independent processes per length, 3 warmups, 12 samples/phase,
lengths 1K/2K/4K/8K. No sample rejection based on latency or winner.

After pilot, an untimed changed-query regression was added to detect stale
projection reuse; the timed pipeline and kernels are unchanged. Main snapshots
identify this source difference. CPU tests: 65 passed; Rust workspace and shell
release-wrapper checks completed (details in c1_build_20260920 logs).
