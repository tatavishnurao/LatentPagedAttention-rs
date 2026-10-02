> **VOID (2026-10-02): the latent/full ratios and latencies in this file were measured with the GPU at idle clocks (~210 MHz) for N<=16K and at uncontrolled clocks at 32K; harness wrote JSON to /mnt/c between samples. Do not cite. See docs/DIAGNOSIS_2026-10-02.md (E0). Removing these first two lines (banner + blank line) restores the original SHA256.**

# Phase 2 summary

## Evidence state

Base commit: `7a2a4501c123433e6cfbaaaf29c0c84733b787c4` plus the preserved working-tree patch in `raw/phase2_working_tree.patch`.

This phase changed only generated length coverage, the C1 example dispatch, generator tests, and analysis tooling. It did not change full/latent arithmetic, precision, paging, launch topology, or the comparison semantics.

## C1 blocker classification and resolution

### BEFORE

The historical main campaign built successfully and then stopped before timing because Windows reported the `Balanced` power scheme rather than the protocol's required High performance scheme. Classification: **OTHER — environmental power/thermal acceptance policy**, not BUILD, VRAM, numerical, indexing, or algorithm failure.

An initial Phase-2 direct run also failed because `CUTILE_TILEIRAS_PATH` was absent at runtime. Classification: **CUTILE environment configuration**. The retry accidentally reused those stdout paths; this is disclosed as a procedural defect in `raw/FAILURES.md`.

### CHANGE

- Generated R-TABLE and C1 kernels were extended to 128, 256, 512, 16,384, and 32,768 in addition to 1K--8K.
- The C1 example dispatch was extended to all nine lengths.
- Runtime used the documented CUDA 13.3/cuTile environment.
- The original strict canonical guard was not weakened or bypassed. Direct runs are explicitly labeled exploratory.

### AFTER

All nine lengths built and executed, passed Rust/GPU reference checks, changed-Q refresh checks, and an independent FP64 explicit reconstructed-K/V oracle. Windows remained on `Balanced`; therefore no result is upgraded to the historical protocol's canonical status.

Long campaigns showed severe time drift at 4K and above. The partial 8K p4 campaign is retained, but initial CUTILE failures and a partial 32K attempt were accidentally overwritten by retries; this procedural defect is disclosed in `raw/FAILURES.md` and prevents canonical use. The aggregate CSV marks nonstationary cases rather than deleting them.

## Central algorithmic finding

The user-proposed C2 transformation was already present in C1:

```text
q' = q Wk^T
scores = q' Z^T
z_context = softmax(scores) Z
output = z_context Wv
```

The context kernel accumulates 32 latent values across sequence blocks and applies the `[32,64]` V projection once. It never reconstructs full V per token. A duplicate C2 was therefore not implemented. See `docs/C2_DESIGN.md`.

## Exploratory sequence sweep

Primary rows are medians of paired process medians. `B1` is the redundant per-block projection ablation; `C1` is hoisted projection plus latent-space value accumulation.

| N | full A1 us | B1 us | C1 us | B1/full | C1/full | status |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 88.1 | 115.2 | 99.3 | 1.308 | 1.128 | exploratory |
| 256 | 138.0 | 167.9 | 131.1 | 1.217 | 0.950 | exploratory |
| 512 | 212.0 | 257.0 | 175.1 | 1.213 | 0.826 | exploratory |
| 1,024 | 379.9 | 456.7 | 286.7 | 1.202 | 0.755 | exploratory |
| 2,048 | 714.8 | 861.7 | 513.0 | 1.206 | 0.718 | exploratory |
| 4,096 | 1,388.0 | 1,680.4 | 964.6 | 1.211 | 0.695 | nonstationary |
| 8,192 | 2,991.9 | 3,053.3 | 1,871.9 | 1.020 | 0.630 | nonstationary |
| 16,384 | 5,513.2 | 5,217.9 | 2,512.9 | 0.946 | 0.462 | nonstationary |
| 32,768 | 1,352.7 | 1,634.2 | 988.4 | 1.208 | 0.731 | nonstationary |

The absolute 32K latency being below 16K is direct evidence of DVFS/nonstationarity, not superlinear acceleration. Cross-length scaling and cache crossover cannot be inferred from these rows.

The stationary exploratory subset suggests a launch-amortization crossover between 128 and 256 for the canonical shape. This must be replicated under accepted power/clocks before becoming a claim.

## Query projection decomposition

C1 projection share fell with N:

| N | projection / pipeline |
|---:|---:|
| 128 | 17.5% |
| 256 | 13.3% |
| 512 | 10.2% |
| 1,024 | 6.3% |
| 2,048 | 3.4% |

At 1K, C1 medians were approximately 18.0 us projection, 39.9 us score, 20.5 us softmax, 218.1 us context, and 286.7 us pipeline. Isolated medians are diagnostics and are not summed. Projection is important at short N but not the large-N bottleneck; the one-program-per-Q-head serial context loop dominates.

## Memory and traffic findings

Canonical persistent compression is 16x, but source-level FP16 row payload is only approximately 2x smaller because each of 16 Q-head programs requests latent rows in score and context. Theoretical source payload amplification relative to persistent state is about 4x full versus 32x latent.

Historical Nsight data (not C1) show:

- score DRAM reads: 563 KB full vs 148 KB latent;
- score L2 bytes: 35.8 MB full vs 40.7 MB latent;
- score instructions: 0.88M full vs 3.18M latent;
- context DRAM reads: 604 KB full vs 195 KB latent;
- context L2 bytes: 34.2 MB full vs 33.4 MB latent.

The score instruction pathology is associated with repeated projection in B1. Current-C1 profiler evidence is still absent.

## Cache residency

With 32 MiB L2, canonical full persistent state is 1 MiB at 1K, 8 MiB at 8K, 16 MiB at 16K, and 32 MiB at 32K. Latent state is 64 KiB, 512 KiB, 1 MiB, and 2 MiB respectively. The desired `full nonresident / latent resident` regime is most plausible around 32K--64K, but the Phase-2 long runs are nonstationary and no cache flush was used. Cache-residency benefit remains unresolved.

## Capacity calculation

Ignoring all non-cache allocations and using GiB:

| cache budget | full tokens | latent tokens |
|---:|---:|---:|
| 1 GiB | 1,048,576 | 16,777,216 |
| 4 GiB | 4,194,304 | 67,108,864 |
| 8 GiB | 8,388,608 | 134,217,728 |
| 24 GiB | 25,165,824 | 402,653,184 |
| 80 GiB | 83,886,080 | 1,342,177,280 |

These are analytical token-layer capacity counts for this synthetic shape, not usable serving capacities.

## What was not completed

- Latent/head/GQA/block-size parameter sweeps.
- Current-C1 Nsight profiling.
- Cache flushing.
- Stronger full baseline redesign.
- Contiguous/paged 2x2 benchmark.
- Batched/multi-sequence decode.
- Split-sequence context kernel.
- Second GPU.

The static cuTile specialization and unstable laptop power/thermal state make a broad campaign inappropriate before fixing the measurement environment and context topology.

## Files

- `crossover_results.csv`: canonical-shape C1 crossover data.
- `ablation_results.csv`: raw-distribution summaries by variant/component.
- `cache_residency_results.csv`: working-set/L2 classification.
- `profiler_summary.csv`: historical P05 counters, explicitly not C1.
- `plots/`: SVG plots generated without external plotting dependencies.
- `raw/`: samples, failures, partial campaigns, source snapshots, patch, environment, and binary checksum.
