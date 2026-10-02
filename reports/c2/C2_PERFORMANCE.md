> **VOID (2026-10-02): the latent/full ratios and latencies in this file were measured with the GPU at idle clocks (~210 MHz) for N<=16K and at uncontrolled clocks at 32K; harness wrote JSON to /mnt/c between samples. Do not cite. See docs/DIAGNOSIS_2026-10-02.md (E0). Removing these first two lines (banner + blank line) restores the original SHA256.**

# C2 performance disposition

**C2 is not distinct from C1 in this repository.** Current C1 already hoists Q projection, scores directly against latent rows, accumulates values in latent space, and applies V projection once. Therefore `C2 vs C1 = 1` by construction; no duplicate timing is reported.

The Phase-2 C1 exploratory sweep found the following paired process-median ratios against A1 full KV:

| N | C1/full | evidence status |
|---:|---:|---|
| 128 | 1.128 | exploratory, Balanced power |
| 256 | 0.950 | exploratory, Balanced power |
| 512 | 0.826 | exploratory, Balanced power |
| 1,024 | 0.755 | exploratory, Balanced power |
| 2,048 | 0.718 | exploratory, Balanced power |
| 4,096 | 0.695 | nonstationary exploratory |
| 8,192 | 0.630 | nonstationary exploratory |
| 16,384 | 0.462 | nonstationary exploratory |
| 32,768 | 0.731 | nonstationary exploratory |

Only 128--2,048 passed the implemented first/last-quartile drift screen; none is canonical because Windows remained on `Balanced` and the predeclared C1 main policy was not satisfied. The 256 result is approximately a 5% marginal advantage, not robust near-parity or speedup evidence. Cross-length absolute latency above 2K is visibly contaminated by DVFS/thermal behavior and must not be used to claim scaling or cache crossover.

Within the stationary exploratory subset, C1 crosses from slower at 128 to slightly faster at 256 for the canonical shape. This is a useful hypothesis for a future accepted campaign, not a peer-review-ready speedup claim.

At 1,024, median component timings were approximately:

- projection: 18.0 us (6.3% of pipeline);
- score: 39.9 us;
- softmax: 20.5 us;
- latent context plus one V projection: 218.1 us;
- pipeline: 286.7 us.

Thus query projection is not the dominant C1 cost beyond short sequences. Context serialization is the next target.
