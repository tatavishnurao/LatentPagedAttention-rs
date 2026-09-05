# LPA P1.5A Kernel Scaling Diagnosis

## 1. Executive Result

The diagnostic question was:

> Which kernel prevents the current matched A/B control implementation from reaching the long-context memory-hierarchy regime?

Answer: **MULTIPLE_KERNELS — the score kernels and the context kernels, not softmax.**

Per-kernel CUDA-event timing of the unchanged P0/P1 kernels across 1K–16K shows:

- **Softmax is NEGLIGIBLE.** Despite its compile-time full-width tile growing from 1,024 to 16,384 elements, isolated softmax median stays flat at 0.006–0.069 ms across all lengths (0.0102 ms A / 0.0112 ms B at 16K = 0.01% of total). P1's hypothesis that fused full-width softmax drives the runtime explosion is **falsified** by measurement.
- **Context kernels are MAJOR.** Dominant share (~63–65%) at 2K–4K, scaling with an observed exponent ≈ 2.08–2.09 (vs 1.0 expected for O(N)): 0.08 ms at 1K → 26.4 ms at 16K. They still launch only 16 blocks and serially iterate N/16 logical blocks, and each iteration reloads the whole block table.
- **Score kernels are DOMINANT at the longest safe length.** 70–71% of total at 16K, with the steepest observed exponent ≈ 2.52–2.53: 0.056 ms at 1K → 62.9 ms at 16K (≈1,121× for 16× sequence). This is inconsistent with the O(N) behavior expected from their N-block parallel grid.
- Source inspection identifies a concrete shared N-dependent cost: `physical_block()` loads the **entire** block table (N/16 entries) as a tile on every call, then extracts one entry. It is called once per score tile (N tiles ⇒ O(N²) aggregate table traffic) and once per serial context iteration (16 heads × N/16 iterations ⇒ O(N²) on the serial critical path). Softmax never touches the table, consistent with its measured flatness. This mechanism is source-derived and timing-consistent; it is not yet counter-verified.
- 32K was **THERMAL_ABORT** (optional length) with no per-kernel data; 16K completed all timed phases before the 86 °C policy kill and is retained with that caveat.

Recommended smallest symmetric repair (spec only, **not implemented**): **R-TABLE** — replace the per-call full-table tile load with a single-entry lookup in all four A/B score/context kernels (expected to restore near-linear scaling), followed by **R-CTX-PAR** (Candidate B, parallel two-stage context reduction) if/when the still-serial 16-block context becomes the residual barrier beyond ~32K. Hierarchical softmax (Candidate A) is **explicitly not recommended**: measurement shows no softmax barrier.

## 2. Repository State

- Repository: `tatavishnurao/LatentPagedAttention-rs`
- WSL path: `/mnt/c/Users/VishnuRao/OneDrive/Desktop/Vishnu/projects/LatentPagedAttention-rs`
- Branch: `main`
- Commit: `5b542088fffa113cfd77e545ddb59e9cb10cb91f`
- At start, `git status --short` showed 144 tracked files modified with 53,661 insertions and 53,661 deletions; `git diff --ignore-cr-at-eol` is **empty**, i.e. the entire pre-existing diff is a CRLF/LF line-ending artifact of the Windows/WSL checkout, not content change. No file was rewritten, reset, stashed, rebased, amended, committed or pushed.
- This experiment adds new untracked files only:
  - `crates/plkv-kernels/examples/p15_kernel_scaling.rs` (diagnostic harness; no kernel changes)
  - `scripts/run_p15_kernel_scaling.py` (thermal-guarded driver)
  - `scripts/analyze_p15_kernel_scaling.py` (aggregation)
  - `reports/p15_kernel_scaling/**` (artifacts)
  - `.gitignore`: two appended exact entries `notes/LPA_P15_KERNEL_SCALING_DIAGNOSIS.md` and `reports/p15_kernel_scaling/` (no wildcard note/report ignoring)
- All P0, P0.5 and P1 work (kernels, generated sources, harnesses, notes, reports) is preserved untouched.

Initial safety commands:

```text
pwd
git status --short
git branch --show-current
git rev-parse HEAD
git diff
git diff --ignore-cr-at-eol | wc -l   # => 0 (line-ending-only diff)
```

## 3. Prior Evidence Verification

Read before any modification:

- `notes/LPA_P0_GPU_BASELINE_EXPERIMENT.md` — confirms STORY A: the old ~32.6% latent slowdown was benchmark methodology artifact; final interleaved 300-sample medians A 0.137216 ms, B 0.139184 ms (B/A 1.01434×) at 1K.
- `notes/LPA_P05_HARDWARE_ATTRIBUTION.md` — at 1K: full score `CACHE/LATENCY_BOUND` (L2 80.67%, DRAM 4.34%, L2 hit 98.49%); latent score `MIXED` with 28.84M FP32 instructions = 11.58× full score; both context kernels `OCCUPANCY/RESOURCE_BOUND` (16 blocks, 0.17 waves/SM, 8.33% occupancy, serial 64-block accumulation); both working sets overwhelmingly cache-resident (32 MiB L2).
- `notes/LPA_P1_SEQUENCE_CROSSOVER.md` — reconnaissance medians 1K ~0.14 ms, 2K ~0.4–0.5 ms, 4K ~1.8 ms, 8K ~15.7 ms, 16K ~122 ms, 32K probe 644.4 ms (A) / 725.0 ms (B); no accepted process at any length; 32K full run stopped at 87 °C; P1 named "full-width compile-time-specialized softmax + serial 16-block context topology" as suspected limitations.

Machine-readable artifacts verified present under `reports/p0_gpu_baseline/`, `reports/p05_hardware_attribution/`, `reports/p05_clean_display_replications/`, `reports/p1_sequence_crossover/` (summaries, manifests, NCU outputs, reconnaissance summaries).

Cross-validation obtained in this phase: the P1.5A 1K A-pipeline median (0.135168 ms) exactly equals P0.5 clean-display attempt 27's A median (0.135168 ms), confirming the instrumentation reproduces the established baseline.

## 4. Source-Level Scaling Analysis

Sources inspected: `crates/plkv-kernels/src/cutile/full_kv_baseline.rs`, `crates/plkv-kernels/src/cutile/model_profile.rs`, `crates/plkv-kernels/src/cutile/p1_sequence_kernels.rs`, `scripts/generate_p1_sequence_sources.py`, `crates/plkv-kernels/examples/p1_seq_1024.rs`.

How sequence length N affects the generated kernels (blocks = N/16):

- **Score grid**: output tensor `[16, N]` partitioned by `[1, 16]` → grid `[16, N/16]` = **N blocks**. Per-block work constant (full: 16×64 K-tile dot; latent: 32×64 projection reduce + 16×32 dot).
- **Softmax grid**: output `[16, N]` partitioned by `[1, N]` → **16 blocks**; the kernel loads the entire row as one compile-time tile `Tile<f32, {[1, N]}>` and performs mask, `reduce_max`, subtract, `exp`, `reduce_sum`, divide across it. Tile extent grows 1024→32768 with N.
- **Context grid**: output `[16, 64]` partitioned by `[1, 64]` → **16 blocks**; serial `for logical_block in 0..blocks` loop (64 → 2048 iterations). Latent adds one O(1) 32×64 output projection.
- **Block table**: `physical_block()` loads the whole table tile `Tile<i32, {[blocks]}>` and extracts one entry; called per score tile and per context iteration.
- **Temporary tensors**: scores `[16, N]` f32, probabilities `[16, N]` f32 → O(N) bytes; context accumulators O(1).
- **Kernel launches**: 3 per decode for A and B at every length (unchanged).

Complexity table (source-derived; written before running anything):

| Kernel | Grid scaling | Per-block work scaling | Temporary storage scaling | Expected complexity |
| --- | --- | --- | --- | --- |
| Full score | 16×(N/16) = N blocks | O(1) per tile (16 dot-products of dim 64 + mask) — but includes one full-table tile load of N/16 entries | scores [16, N] f32 → O(N) | O(N) if per-tile cost constant; O(N²) aggregate table traffic from `physical_block` |
| Latent score | N blocks | O(1) per tile but ~11.58× full-score FP32 instructions (W_k@q in every tile) + same full-table load | scores [16, N] f32 → O(N) | same as full score, larger constant |
| Softmax | 16 blocks (constant) | O(N): one [1, N] tile — mask, reduce_max, subtract, exp, reduce_sum, divide | N-wide f32 tiles live in one block | O(N) work in 16 blocks; compile-time tile extent grows with N |
| Full context | 16 blocks (constant) | serial N/16 iterations × (1×16 probs load + 16×64 V load + weighted accumulate + full-table load) | O(1) accumulator | O(N) serial if per-iteration cost constant; O(N²) aggregate table traffic on serial chain |
| Latent context | 16 blocks (constant) | serial N/16 iterations × (1×16 probs + 16×32 latent + accumulate + full-table load) + one 32×64 projection | O(1) accumulator | same structure as full context, half cache elements |

No performance conclusion was drawn from this table alone; it directed the measurement.

## 5. Diagnostic Timing Methodology

New diagnostic harness: `crates/plkv-kernels/examples/p15_kernel_scaling.rs` (auto-discovered example, `gpu-cutile` feature). It is instrumentation only:

- **No kernel source was changed.** The harness dispatches to the existing generated per-length modules `full_kv_baseline_kernel_{L}` / `model_profile_kernel_{L}` exactly as the P1 examples do.
- **Timing boundary**: CUDA events (`cuEventRecord` start/stop on one explicit stream, `cuEventElapsedTime_v2`) around a **single kernel launch** only. Pre-measurement stream sync before each sample. Excluded from timing: process startup, cuTile JIT, allocations, input generation, host→device upload, correctness readback, teardown.
- **Dependency handling** (Section 7 protocol):
  1. Untimed: run full A pipeline (score→softmax→context) and full B pipeline once; sync; read back outputs for correctness (validated against the established Rust CPU references `paged_full_kv_gqa_decode_fp16_storage_runtime_f32_accum` / `direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum`, unchanged tolerances).
  2. Untimed: refresh — run both pipelines once more so resident scores/probabilities are fresh and identical for both paths.
  3. Timed isolated phases: `A_score` and `B_score` repeatedly launch only the score kernels (resident q/K/latent/projections/table inputs; scores buffer rewritten each iteration). `A_softmax`/`B_softmax` repeatedly launch only softmax reading the resident valid scores produced in steps 1–2. `A_context`/`B_context` repeatedly launch only context reading the resident valid probabilities. Context therefore always consumes correct probabilities; softmax always consumes valid scores.
  4. `A_pipeline`/`B_pipeline` phases measure the full 3-kernel chain in one event span as a cross-check total.
- **Phase order** (fixed, documented): A_score, B_score, A_softmax, B_softmax, A_context, B_context, A_pipeline, B_pipeline. Per phase: per-phase warmup (untimed) + sync, then measured samples.
- **Iteration budget** (diagnostic, not publication): 1K 10/50, 2K 10/30, 4K 5/20, 8K 3/10, 16K 1/5, 32K 1/1 (warmup/measured).
- Medians reported where ≥3 samples; 16K has 5 samples (min/max spread 62.67–62.96 ms for A_score, i.e. stable), 32K none. 1–5-sample long-context values are diagnostic timings, not statistical estimates.
- Driver: `scripts/run_p15_kernel_scaling.py` (thermal gate + monitoring + cooldown); aggregation: `scripts/analyze_p15_kernel_scaling.py`.
- Environment: `CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3 LIBCLANG_PATH=/usr/lib/llvm-18/lib CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.3/bin/tileiras`, release build. No NCU, no nsys, no clock/power changes.

Known methodological caveats: (a) fixed phase order means early phases can run at lower boost clocks than late phases during fast clock ramps (visible at 4K, see §14); (b) isolated-kernel timing adds per-launch event/sync overhead and removes inter-kernel pipelining, so isolated sums may exceed pipeline medians; (c) 16K phases were executed while the GPU climbed to 86 °C, so 16K absolute values may include mild thermal clock depression (tight within-phase min/max argues the effect is small).

## 6. Correctness

Verified once per executed length against the unchanged CPU references and unchanged tolerances (max abs ≤ 5e-3, probability row-sum ≤ 1e-4). All lengths PASS. Values match P1's `correctness_by_sequence.json` exactly, proving the instrumentation did not alter algorithm semantics.

| Seq | A max abs | A RMSE | A row-sum err | B max abs | B RMSE | B row-sum err | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1,024 | 1.431e-5 | 1.092e-6 | 8.345e-7 | 7.629e-6 | 8.310e-7 | 1.132e-6 | PASS |
| 2,048 | 2.575e-5 | 1.158e-6 | 1.192e-6 | 1.287e-5 | 9.260e-7 | 1.192e-6 | PASS |
| 4,096 | 3.624e-5 | 1.175e-6 | 2.265e-6 | 2.384e-5 | 1.065e-6 | 2.027e-6 | PASS |
| 8,192 | 6.151e-5 | 1.269e-6 | 4.470e-6 | 4.244e-5 | 1.047e-6 | 4.351e-6 | PASS |
| 16,384 | 1.016e-4 | 1.352e-6 | 6.735e-6 | 7.677e-5 | 1.190e-6 | 7.272e-6 | PASS |

## 7. Thermal Safety

Policy enforced by the driver: pre-run gate ≥84 °C → cooldown/skip; during-run ≥86 °C → kill + `THERMAL_ABORT`; cooldown between lengths to ≤78 °C; clocks/power limits never modified. RTX 4060 Laptop GPU, 8188 MiB.

| Seq | Status | Pre-run temp | Peak during | P-states observed | SM clocks observed (MHz) |
| ---: | --- | ---: | ---: | --- | --- |
| 1,024 | OK | 76 °C | 76 °C | P0, P5 | 1125, 1890 |
| 2,048 | OK | 75 °C | 78 °C | P0 | 1890, 2685 |
| 4,096 | OK | 77 °C | 79 °C | P0 | 1890, 2685 |
| 8,192 | OK | 76 °C | 80 °C | P0, P3 | 1890, 2670 |
| 16,384 | THERMAL_ABORT (data complete) | 77 °C | 86 °C | P0 | 1890, 2670 |
| 32,768 | THERMAL_ABORT (no data) | 78 °C | 86 °C | P0, P3 | 555, 1020, 1890, 2670 |

Details:

- **16K**: all 8 timed phases completed and the summary JSON was written; the 86 °C policy kill landed at the very end of the run (5th ~1.5 s monitoring sample, ≈7.5 s after launch). Data retained; the run is still marked THERMAL_ABORT per policy.
- **32K**: aborted during untimed preparation (correctness/JIT pipeline); zero stdout produced. No retry was attempted: the environment's idle floor sits at ~74–79 °C (desktop/compositor activity, consistent with P0.5 observations), and a 32K attempt needs more sustained full-load seconds than 16K, so a retry from this floor would very likely re-abort. This is consistent with P1's 32K experience (87 °C stop). The protocol marks 32K optional; it is reported as THERMAL_ABORT. The only 32K observations remain P1's single-iteration probe medians: A 644.398 ms, B 724.988 ms.
- The 84 °C/87 °C boundary from P1 was never intentionally approached; two brief 86 °C contacts occurred as specified by the abort policy.
- Pre-run snapshots (utilization, temperature, P-state, SM clock, compute processes) are recorded per length in `thermal_manifest.json`; no competing compute processes were present (desktop compositor only).

## 8. Score Scaling

Median isolated score latency (ms):

| Seq | A score | B score | B/A |
| ---: | ---: | ---: | ---: |
| 1,024 | 0.0561 | 0.0602 | 1.073 |
| 2,048 | 0.1955 | 0.2005 | 1.026 |
| 4,096 | 0.7828 | 0.8074 | 1.031 |
| 8,192 | 8.8402 | 7.4035 | 0.837 |
| 16,384 | 62.8900 | 65.1528 | 1.036 |

### Full KV

Score latency does **not** scale approximately with `query_heads × logical_blocks` as the N-block parallel grid predicts. A grid-parallel kernel with constant per-tile cost should double per doubling of N (~2×). Observed doubling factors are 3.48 / 4.00 / 11.29 / 7.11 (1K→16K growth 1,120.8×, observed exponent ≈ 2.53). The observed scaling is **inconsistent with O(N)** and consistent with a per-tile cost that itself grows with N. The source provides exactly one such term: `physical_block()` loads the full N/16-entry table tile per score tile ⇒ O(N²) aggregate table traffic (16.7M i32 elements ≈ 67 MB of dependent loads at 16K), plus growing instruction work per tile. All other per-tile work (q row, K tile, dots, mask, store) is constant-size.

### Latent

B score shows the same superlinear shape (growth 1,083×, exponent ≈ 2.52), so the repeated `W_k @ q` projection does **not** make latent score scaling disproportionately worse than full with block count: B/A score stays 0.84–1.07 across all lengths (the 8K inversion is within-run clock variability). The P0.5 fact that latent score executes 11.58× full-score FP32 instructions manifests only as a small constant-factor difference (≤ +7%), because the N-dependent table-access term dominates at long context. Not rerun: no counter experiment here, per scope.

## 9. Softmax Scaling

The generated softmax kernel's compile-time tile extent grows with sequence length exactly as

```text
1024 → 2048 → 4096 → 8192 → 16384 → 32768
```

(one `Tile<f32, {[1, N]}>` row per 16-block grid entry), and across that tile it performs: active-mask construction, `reduce_max`, subtract-broadcast, `select`/`exp`, `reduce_sum`, divide-broadcast, store. Structurally this looked like P1's prime suspect.

Measured isolated medians (ms):

| Seq | A softmax | B softmax |
| ---: | ---: | ---: |
| 1,024 | 0.0082 | 0.0086 |
| 2,048 | 0.0065 | 0.0061 |
| 4,096 | 0.0691 | 0.0092 |
| 8,192 | 0.0532 | 0.0617 |
| 16,384 | 0.0102 | 0.0112 |

Softmax is **flat within noise across a 16× sequence range** (the 4K/8K wiggles are microsecond-scale outliers; even the worst value, 0.069 ms, is 2.8% of A's 4K total). At 16K it is 0.0102/0.0112 ms = **0.01%** of the total. The 16K tile (64 KB f32 row plus mask/temporaries) is evidently lowered by the tile compiler into an efficient streamed form rather than register catastrophe, and the kernel issues no block-table access.

**Answer: No — full-width softmax is not one of the major causes of P1's nonlinear runtime explosion.** Classification: **NEGLIGIBLE** (≤0.01% share at the longest safe length; no scaling trend; falsifies the P1 softmax hypothesis).

## 10. Context Scaling

Topology unchanged: 16 blocks (one per query head), serial loop over `blocks = N/16` logical iterations (64 → 1024 at 16K), each iteration loading a 1×16 probability tile, a 16×64 (A) or 16×32 (B) value tile, and — via `physical_block()` — the **entire** N/16-entry block table, with the extract dependent on that load.

Measured isolated medians (ms):

| Seq | A context | B context |
| ---: | ---: | ---: |
| 1,024 | 0.0819 | 0.0795 |
| 2,048 | 0.3827 | 0.3410 |
| 4,096 | 1.4746 | 1.4106 |
| 8,192 | 7.0773 | 6.8771 |
| 16,384 | 26.4192 | 26.2984 |

Doubling factors: A 4.67 / 3.85 / 4.80 / 3.73; B 4.29 / 4.14 / 4.88 / 3.82. 1K→16K growth: A 322.5×, B 330.6×; observed exponent ≈ 2.08–2.09. A pure serial O(N) loop with constant per-iteration cost would give ~2× per doubling (exponent 1.0). The observed near-quadratic growth means **per-iteration cost also grows with N**; the source's per-iteration full-table tile load supplies exactly that N-dependent term ((N/16)² aggregate table elements per head on the serial critical path). B context is consistently a few percent faster than A (half-size value tiles), matching P0.5's byte accounting; neither escapes the superlinear shape.

**Answer: serial context accumulation is a major long-context scalability failure** (dominant share at 2K–4K; ~N² observed), **but it is not the single dominant barrier at the longest safe length** — score overtakes it by 8K–16K. Classification: **MAJOR** (would be DOMINANT if score's even-steeper superlinearity were absent).

## 11. Component Fractions

Required timing table (medians, ms; "total" = measured 3-kernel pipeline median):

| Seq | A score | A softmax | A context | A total | B score | B softmax | B context | B total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.0561 | 0.0082 | 0.0819 | 0.1352 | 0.0602 | 0.0086 | 0.0795 | 0.1386 |
| 2,048 | 0.1955 | 0.0065 | 0.3827 | 0.5775 | 0.2005 | 0.0061 | 0.3410 | 0.5356 |
| 4,096 | 0.7828 | 0.0691 | 1.4746 | 2.4387 | 0.8074 | 0.0092 | 1.4106 | 1.7270 |
| 8,192 | 8.8402 | 0.0532 | 7.0773 | 13.8280 | 7.4035 | 0.0617 | 6.8771 | 13.3358 |
| 16,384 | 62.8900 | 0.0102 | 26.4192 | 92.9679 | 65.1528 | 0.0112 | 26.2984 | 89.7616 |

Component fractions of each variant's isolated-kernel median sum:

| Seq | A score % | A softmax % | A context % | B score % | B softmax % | B context % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 38.37 | 5.60 | 56.02 | 40.56 | 5.82 | 53.62 |
| 2,048 | 33.43 | 1.12 | 65.45 | 36.62 | 1.11 | 62.27 |
| 4,096 | 33.65 | 2.97 | 63.38 | 36.25 | 0.41 | 63.33 |
| 8,192 | 55.35 | 0.33 | 44.31 | 51.62 | 0.43 | 47.95 |
| 16,384 | 70.41 | 0.01 | 29.58 | 71.23 | 0.01 | 28.75 |

Dominant kernel: at 1K context (56.0% A / 53.6% B); at the longest safe length (16K) score (70.4% A / 71.2% B).

## 12. Doubling Factors

T(2N)/T(N) on medians (diagnostic; linear O(N) ⇒ ~2.0, quadratic ⇒ ~4.0):

| Kernel | 1K→2K | 2K→4K | 4K→8K | 8K→16K | 1K→16K growth | Observed exponent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A score | 3.48 | 4.00 | 11.29 | 7.11 | 1,120.8× | ≈2.53 |
| A softmax | 0.80 | 10.59 | 0.77 | 0.19 | 1.2× (flat) | no trend |
| A context | 4.67 | 3.85 | 4.80 | 3.73 | 322.5× | ≈2.08 |
| B score | 3.33 | 4.03 | 9.17 | 8.80 | 1,083.0× | ≈2.52 |
| B softmax | 0.71 | 1.51 | 6.69 | 0.18 | 1.3× (flat) | no trend |
| B context | 4.29 | 4.14 | 4.88 | 3.82 | 330.6× | ≈2.09 |
| A pipeline | 4.27 | 4.22 | 5.67 | 6.72 | 687.8× | ≈2.36 |
| B pipeline | 3.86 | 3.22 | 7.72 | 6.73 | 647.7× | ≈2.33 |

Absolute growth vs 1K (T(N)/T(1K)) — where the explosion occurs:

| Kernel | 2K | 4K | 8K | 16K |
| --- | ---: | ---: | ---: | ---: |
| A score | 3.5× | 14.0× | 157.5× | 1,120.8× |
| A softmax | 0.8× | 8.4× | 6.5× | 1.2× |
| A context | 4.7× | 18.0× | 86.4× | 322.5× |
| B score | 3.3× | 13.4× | 123.1× | 1,083.0× |
| B softmax | 0.7× | 1.1× | 7.1× | 1.3× |
| B context | 4.3× | 17.7× | 86.5× | 330.6× |
| A pipeline | 4.3× | 18.0× | 102.3× | 687.8× |
| B pipeline | 3.9× | 12.5× | 96.2× | 647.7× |

Interpretation: no kernel shows clean ~2× doubling. Score's observed scaling is inconsistent with the O(N) expectation of its N-block grid and consistent with an additional N-dependent per-tile cost (aggregate ≈ between quadratic and cubic; the 4K→8K jump of 11.29× also overlaps the onset of sustained-load clock behavior, so individual factors are not micro-attributed). Context is consistent with quadratic aggregate behavior arising from an O(N) serial loop whose per-iteration cost grows with N. Softmax is flat.

## 13. Primary Scalability Barrier

Measured verdict (from the categories requested): **MULTIPLE_KERNELS**.

- Score kernels (both A and B) are the dominant latency at the longest safely measured length (70–71% at 16K) and have the steepest superlinear scaling (exponent ≈2.5). They are the reason the pipeline total reaches ~90 ms at 16K instead of the ~2.2 ms a truly O(N) implementation would suggest.
- Context kernels (both A and B) are the second barrier: dominant share at 2K–4K, exponent ≈2.1, 16-block serial topology with per-iteration full-table loads.
- Softmax is eliminated as a barrier (NEGLIGIBLE).
- Thermal behavior is a consequence of these runtimes, not an independent cause: with near-linear kernels, 16K decodes would be millisecond-scale and would not drive the laptop GPU to 86 °C in seconds. Classification `THERMAL_ONLY` is rejected.

Shared candidate mechanism (source-derived, timing-consistent, not yet counter-verified): the per-call full-block-table tile load in `physical_block()` — O(N²) aggregate table traffic for score, and O(N²) work injected into context's serial critical path — combined with context's 16-block serial topology. This explains why exactly the two table-using kernels explode while the table-free softmax stays flat.

## 14. Full-KV vs Latent Differences

- **Parity persists in shape.** B/A pipeline ratios: 1.025 (1K), 0.927 (2K), 0.708 (4K), 0.964 (8K), 0.966 (16K). The 4K outlier is a phase-ordering/DVFS artifact: A_softmax recorded a one-off 0.069 ms median and A_pipeline ran earlier in the clock ramp; isolated-sum totals give A 2.32 ms vs B 2.23 ms (B/A 0.96), in line with the other lengths. No A/B crossover conclusion is drawn from rejected-quality timings.
- **Latent's 16× smaller persistent representation produces no long-context timing advantage yet** — because the implementation is dominated by shared superlinear terms (table traffic, serial topology), not by K/V vs latent cache traffic. This is exactly why the cache-residency crossover could not be observed: the implementation barrier precedes the memory-hierarchy barrier.
- Consistent with P0.5: B context is slightly faster than A context at every length (half-size value tiles: 341 µs vs 383 µs at 2K; 26.30 vs 26.42 ms at 16K); B score carries the projection constant (B/A score 1.03–1.07 except the 8K clock-noise point).
- The 1K A pipeline median (0.135168 ms) reproduces P0.5 clean-display attempt 27 exactly; 1K B/A = 1.025 here vs 1.0066 in P0.5's fully stabilized environment — the difference is attributed to the shorter warmup and clock ramp of the diagnostic budget, not to semantic change (correctness identical).

## 15. Proposed Symmetric Control Repair

Proposed only; **not implemented**. Repairs must preserve mathematical semantics and keep A0/B0 runnable.

| Rank | Repair | Kernel(s) | Evidence | Expected research value | Engineering risk |
| ---: | --- | --- | --- | --- | --- |
| 1 | **R-TABLE: single-entry block-table lookup** — replace `physical_block()`'s full-table `load_tile` + `extract` with a 1-entry (scalar/1-wide tile) lookup of `table[logical]`, symmetrically in A and B score + context kernels | A score, A context, B score, B context | Only N-dependent per-work-unit cost in the superlinear kernels; softmax (no table access) measured flat; removes O(N²) aggregate table traffic from score and the serial chain | HIGH — restores ~O(N) scaling for both dominant kernels; 16K expected to drop from ~90 ms toward low single-digit ms, making the cache-residency crossover region (≥32K full K+V) safely measurable | LOW — one helper per module; grids, tiles, arithmetic, launch counts unchanged; outputs expected bitwise-identical |
| 2 | **R-CTX-PAR: parallel context reduction (Candidate B)** — stage 1 grid (query_head, sequence_chunk) produces partial weighted sums; stage 2 reduces partials; B reduces latent partials then applies the unchanged output projection once | A context, B context | 16-block serial topology (8.33% occupancy at 1K, P0.5) remains the residual O(N) serial chain after R-TABLE; needed for lengths beyond where 16-block serial O(N) is acceptable (~≥32K) | MEDIUM-HIGH — unlocks 32K–128K crossover study | MEDIUM — extra transient partial buffers + one extra launch per path; FP32 accumulation order changes within normal tolerance |
| 3 | **NO-SOFTMAX-REPAIR (Candidate A rejected)** | — | Softmax NEGLIGIBLE at all lengths (0.01% at 16K) | NONE — would add complexity without removing a measured barrier | Rejected |

Symmetry: A0/B0 (original kernels) remain untouched and runnable; repaired variants are A1/B1. Both repairs apply identically to full-KV and latent paths.

## 16. Implementation Scope for P1.5B

See the contract in §26/P1.5B Implementation Contract below. In short: implement R-TABLE as A1/B1 (additive, generated per length exactly like P1), validate bitwise-or-tolerance equality against A0/B0 and CPU references, remeasure the same per-kernel grid; add R-CTX-PAR only if residual serial context measurably blocks the target lengths.

## 17. Remaining Limitations

1. **16K data caveat**: the 16K run touched 86 °C at its very end (policy kill after all phases completed); mild late-run thermal clock depression cannot be fully excluded, though within-phase min/max spreads are ≤0.5%.
2. **No 32K per-kernel data**: THERMAL_ABORT during preparation; no retry because the environment's idle floor (~74–79 °C with desktop activity) leaves insufficient headroom for a longer sustained load. P1's single-iteration probe (A 644.4 ms / B 725.0 ms) remains the only 32K observation.
3. **Diagnostic sample sizes**: 3–50 samples per kernel per length (1–5 at 16K). Medians are stable where sampled, but these are not publication statistics and no confidence intervals are claimed.
4. **Fixed phase order**: early phases can run at lower boost clocks during ramp-up (visible in the 4K A_softmax/A_pipeline values); per-phase medians remain valid for decomposition, but cross-phase absolute fairness at short lengths has ±tens-of-percent uncertainty. The 1K run sampled P5/1125 MHz states at its monitoring points, so 1K baselines may be mildly conservative (which would mildly exaggerate growth exponents, not hide them).
5. **Mechanism not counter-verified**: the full-table-load attribution is source-derived and timing-consistent; NCU (explicitly deferred) or an A/B kernel micro-experiment in P1.5B is needed to confirm it is the dominant term rather than e.g. loop-unroll/register effects in the serialized context body.
6. **Environment**: laptop RTX 4060 shared with the Windows desktop (compositor memory floor, P-state transitions), WSL2/WDDM; clock CV was not controlled for this diagnostic sweep, unlike P0.5 accepted runs. These timings are deliberately diagnostic.
7. Only one canonical shape (16/4 heads, 64/32 dims, block 16, FP16/FP32) on one GPU.

---

## P1.5B Implementation Contract

### Preserve

- All P0/P0.5/P1 code, artifacts, and notes. No rewrite of existing kernels.
- Exact configuration: `q_heads=16, kv_heads=4, group_size=4, head_dim=64, latent_dim=32, block_size=16`, FP16 storage, FP32 arithmetic, paging enabled, active=max seq length, deterministic inputs (`deterministic_values` formulas, seed-free), unchanged block-table formula `(logical*17+11) % blocks`.
- **A0 and B0 remain runnable and unmodified** (original `full_kv_baseline` and `model_profile` kernels and all `p1_sequence_kernels` modules). Never replace the controls.
- Unchanged correctness tolerances: max absolute error ≤ 5e-3 vs CPU reference; probability row-sum error ≤ 1e-4.
- The P1.5A harness (`p15_kernel_scaling.rs`) as the measurement instrument, with its thermal driver and iteration budget.
- No clocks/power changes; same thermal policy (≥84 °C pre-gate, ≥86 °C abort).

### Implement

- **Repair R-TABLE only (rank 1)**, as new additive A1/B1 kernel modules (e.g. `p15_repaired_kernels.rs` generated per length by extending the existing generator pattern): in every `physical_block()`-equivalent, load exactly one table entry for the requested logical index (1-element tile load/gather + scalar use) instead of the full N/16-entry table tile.
- Apply the change identically and symmetrically to four kernels: full score, full context, latent score, latent context. Everything else — tile shapes, grids (score `[16, blocks]`, softmax `[16]`, context `[16]`), loop bounds, masking constants, scaling, projection arithmetic — stays textually identical to A0/B0.
- R-CTX-PAR (parallel context reduction) is deferred unless R-TABLE validation shows serial context still blocks the intended lengths; do not implement it preemptively.

### Full-KV path

- A1 score = A0 score with single-entry table lookup; A1 context = A0 context with single-entry lookup inside the serial loop. Same launch geometry. Softmax unchanged (shared kernel, already NEGLIGIBLE).
- Expected: per-tile cost becomes N-independent; A1 score ≈ O(N) wall time; A1 context ≈ O(N) with constant per-iteration cost (verify empirically).

### Latent path

- B1 score = B0 score with single-entry lookup (projection-per-tile deliberately retained — P0 established it is not a significant bottleneck and removing it would break control symmetry).
- B1 context = B0 context with single-entry lookup; output projection unchanged.

### Correctness tests

1. A1/B1 vs established CPU references at every measured length (unchanged tolerances) — scores, probabilities, context, row-sum.
2. A1 vs A0 and B1 vs B0 GPU-output comparison: expect bitwise-identical or ≤1e-6 max absolute difference (same arithmetic, same data); any larger divergence is a semantics regression and a stop condition.
3. Reuse the P1.5A dependency protocol: prepare valid resident scores before softmax timing and valid resident probabilities before context timing; never time a kernel on invalid inputs.

### Benchmark comparison

- Same grid: 1K/2K/4K/8K/16K (+32K only if thermally safe), same warmup/measured budgets, same CUDA-event isolated phases, same thermal driver.
- Report: **A0 vs A1** and **B0 vs B1** per-kernel medians, doubling factors and observed exponents (target: score/context exponents ≈1.0 after repair); **A1 vs B1** ratios across lengths (the first clean look at long-context A/B behavior); component fractions; residual dominant kernel at the longest safe length.
- Then, and only then, assess whether the cache-residency crossover region (analytically ~32K full K+V vs 32 MiB L2) is safely reachable, and whether R-CTX-PAR is required next.

### Explicitly out of scope

- Any softmax change (Candidate A is rejected by measurement).
- Kernel fusion, tiling changes, new projection strategies, preprojected-C-style restructuring, block-size/dimension sweeps, batch sweeps.
- NCU/NSIGHT profiling in P1.5B's first step (allowed later only for mechanism confirmation of the table-load attribution).
- Lengths above 32K, 1000-iteration publication benchmarks, clock/power manipulation.
- Replacing or removing A0/B0 controls.
