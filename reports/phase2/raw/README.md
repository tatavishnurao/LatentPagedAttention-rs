# Phase-2 raw evidence

`c1_exploratory/` contains CUDA-event samples, GPU/CPU correctness JSON, query-refresh checks, oracle inputs/outputs, independent FP64 oracle results, stdout, and pre/post `nvidia-smi` snapshots.

Aggregation policy in `scripts/analyze_phase2.py`:

- N <= 2K: complete p1/p2 runs, 50 samples/component/process.
- 4K: complete p0/p3/p4 runs, 10 samples/component/process.
- 8K, 16K, 32K: complete p0/p3 runs, 10 samples/component/process.
- Partial files are never included.
- A row is marked nonstationary if any selected pipeline variant's last-quartile median differs by more than 20% from its first-quartile median.
- Process is the replication unit for primary medians and ratios; pooled samples provide descriptive p10/p90/p95/MAD only.

All timings are exploratory because the active Windows power scheme was `Balanced`. See `FAILURES.md` for overwritten-failure disclosure and incomplete attempts.

`phase2_working_tree.patch`, `source_snapshot/`, environment text, and the executable SHA256 establish source identity. The base commit is `7a2a4501c123433e6cfbaaaf29c0c84733b787c4`.
