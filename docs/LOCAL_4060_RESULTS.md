# Local RTX 4060 results: latent paged decode vs padded full-KV (E3 / E4)

**Host: laptop RTX 4060 (8 GB, sm_89), WSL2. Not a cloud result, not an L4 result.** Data: `reports/local_4060_20261101/`. Run selection fixed in advance: `docs/LOCAL_4060_PLAN.md`. Rules: `docs/E0B_CLOUD_RUNBOOK.md` section 4. Every number below is MEASURED from the files named, unless marked SOURCE (read in code) or HYPOTHESIS.

## Verdict

On this laptop, latent paged decode (C3 accurate) is **not uniformly faster** than equal-treatment padded full-KV (A3). Of 11 configurations, 4 are claim-eligible, and they do not line up with working-set size. C3 wins in single-sequence E3 at N=8192 (0.893 [0.860, 0.936]) and in batched E4 at B8_N8192 (0.865 [0.842, 0.916]). It **loses** at E4 B32_N4096 (1.030 [1.009, 1.051]), and at B32_N8192 neither a win nor parity is shown (0.972 [0.915, 1.029]). The other 7 configurations have fewer than 5 clean processes and are descriptive only. The E3 N=8192 win also holds against the stronger unpadded-A3 baseline in a secondary run, with a smaller margin (0.924 [0.888, 0.971] instead of about 0.89). C3 moves 3.5-13x fewer analytic bytes than padded A3 in every configuration, so on this hardware **compression is not the same as speedup**; the ratio depends on batch and context length. The hypothesis that latent wins once the working set outgrows L2 is **not tested by E4 at all**: E4 flushes L2 before every layer and reads each KV block once per layer, so every E4 layer starts cold (see "L2 is not an axis for E4").

## Primary results

C3 accurate / padded A3, geometric mean of per-process ratio-of-medians over clean processes, 95 % hierarchical-bootstrap CI. Clean / THROTTLED / hot-start counts per configuration. No process in any primary run was a hot start.

| Config | Runs | Clean / THR / hot | Median per step, A3pad · C3acc (µs) | C3acc / A3pad [95 % CI] | Claim | A3pad bytes per layer (analytic) | Byte ratio A3pad / C3 | Achieved GB/s (analytic), A3pad · C3 | Modal SM clock, clean | Peak °C |
|---|---|---|---|---|---|---|---|---|---|---|
| E3 N=1024 | b1r | 0 / 5 / 0 | — | — | not enough clean processes | 1.05 MiB | 7.7 | — | — | 76 |
| **E3 N=8192** | b1r | 5 / 0 / 0 | 140.4 · 124.0 | **0.893 [0.860, 0.936]** | **yes** | 8.10 MiB | 13.2 | 60.5 · 5.2 | 2580-2595 MHz | 80 |
| E4 B1_N1024 | e4n1024v2 | 0 / 5 / 0 | — | — | not enough clean processes | 1.20 MiB | 4.9 | — | — | 80 |
| E4 B8_N1024 | e4n1024v2 | 0 / 5 / 0 | — | — | not enough clean processes | 5.98 MiB | 3.5 | — | — | 80 |
| E4 B32_N1024 | e4n1024v2 | 0 / 5 / 0 | — | — | not enough clean processes | 23.2 MiB | 3.5 | — | — | 78 |
| E4 B1_N4096 | e4n4096 | 0 / 5 / 0 | — | — | not enough clean processes | 4.20 MiB | 9.7 | — | — | 75 |
| E4 B8_N4096 | e4n4096 | 2 / 3 / 0 | 1071 · 1088 (2 processes) | 1.003 [0.977, 1.037] | not enough clean processes | 18.7 MiB | 7.4 | 73.2 · 9.7 | 2595 MHz | 77 |
| **E4 B32_N4096** | e4n4096 | 5 / 0 / 0 | 3962 · 4104 | **1.030 [1.009, 1.051]** | **yes (C3 slower)** | 70.5 MiB | 7.3 | 74.6 · 9.9 | 2595 MHz | 77 |
| E4 B1_N8192 | e4n8192 | 0 / 5 / 0 | — | — | not enough clean processes | 8.20 MiB | 12.0 | — | — | 77 |
| **E4 B8_N8192** | e4n8192 | 5 / 0 / 0 | 1537 · 1305 | **0.865 [0.842, 0.916]** | **yes** | 35.6 MiB | 9.9 | 97.1 · 11.5 | 2595 MHz | 79 |
| **E4 B32_N8192** | e4n8192 | 5 / 0 / 0 | 5849 · 5566 | 0.972 [0.915, 1.029] | yes; **not shown** | 133.5 MiB | 9.7 | 95.7 · 10.3 | 2580-2595 MHz | 80 |

- Run directories: b1r = `local4060_b1r_e3_20261101_051132`; e4n4096 / e4n8192 / e4n1024v2 = `local4060_e4n*_20261101_*` (harness `e3e4-rewarm-saturating-20ms-2026-11-01`).
- E4 "per step" = all layers of one decode step (B launch pairs per layer, L2 flushed before each layer); E3 "per step" = one split + reduce for one sequence, L2 not flushed.
- "Not shown" at B32_N8192: the 95 % CI includes 1.0, and the per-process ratios spread from 0.88 to 1.03. Parity (TOST, 90 % CI inside [0.95, 1.05]) was not computed and the 95 % CI already extends below 0.95.
- GB/s is analytic bytes divided by clean median time, not a measured DRAM bandwidth. C3's 5-12 GB/s means C3 is not limited by memory bandwidth here (HYPOTHESIS: compute or launch overhead dominates).

**Where the ratio crosses 1.0:** not along working-set size. Among claim-eligible E4 configurations, it crosses between B8_N8192 (0.865) and B32_N4096 (1.030), and B32_N8192 sits at 0.97 with a CI over 1.0. With 3 claim-eligible E4 points no trend in B or N can be established.

**Counts:** 4 claim-eligible, 7 descriptive only (one of them, B8_N4096, with 2 clean processes; six with 0).

## Secondary: unpadded-A3 sensitivity (E3 N=8192)

Declared in `docs/LOCAL_4060_PLAN.md` addendum 2 before the run; own harness version (`...+a3unpad`), run `local4060_e3a3sens_20261101_063414`, 5 / 0 / 0, all at 2595 MHz, peak 72 °C. Reported separately; padded A3 remains the pre-declared baseline.

| Same run, E3 N=8192 | Ratio [95 % CI] |
|---|---|
| C3 accurate / unpadded A3 | **0.924 [0.888, 0.971]** |
| C3 fast / unpadded A3 | 0.909 [0.884, 0.943] |
| C3 accurate / padded A3 | 0.891 [0.856, 0.925] |
| unpadded A3 / padded A3 | 0.964 [0.928, 0.989] |

Per the pre-declared interpretation: the E3 N=8192 win holds against both baselines, on this laptop, at this one size; the margin against unpadded A3 is about 7.6 % instead of about 11 %. Unpadded A3 is only about 3.6 % faster than padded A3 although padded A3 does about 4x the multiply-adds (HYPOTHESIS: at this size A3 time is not dominated by multiply-adds).

## Preliminary Block 1 (reported, never pooled)

`local4060_b1_e3_20261101_043331`, same harness, Brave open (0 % GPU). E3 N=8192: 5 / 0 / 0, 0.888 [0.869, 0.914], consistent with the primary 0.893 [0.860, 0.936]. E3 N=1024: 2 / 3 / 0, descriptive only (primary re-run: 0 clean). No disagreement in claim status at N=8192.

## L2 is not an axis for E4

SOURCE: each sequence owns a disjoint slice of the per-layer pool (`crates/plkv-kernels/src/e4_config.rs:230-236`), and the launches in `e3_splitk_gpu.rs` (`launch_async`) give each padded-A3 CTA one (split, KV head) tile and each C3 CTA one split over all heads, so each KV or latent block is read once per layer; only queries, projections, tables and partials (< 1 MiB) are reused. With the 64 MiB memset before every layer, every E4 layer reads its KV from DRAM regardless of size. The padded-A3 analytic bandwidth is similar on both sides of 32 MiB (73-97 GB/s). E3 is not flushed: its working set (8.5 MB at N=8192) stays L2-resident across iterations, so E3 vs E4 is a warm-cache vs cold-cache contrast, not a size contrast. The block reports during the campaign used "above L2" labels for E4; those labels were wrong and are not used here.

## Instrument findings (methodology)

1. **The NVML clock-event-reasons query takes about 20 ms under WSL2** (median 20.6 ms, max 30 ms; other NVML reads 0.04-0.16 ms). Called after every timed sample, it idled the GPU between ~0.02 ms layers; in the first dry run 80 % of E4 layers ran at 210 MHz, about 16x slower, and the modal-clock THROTTLED rule passed the process because the mode itself was 210 MHz. That dry run is VOID.
2. **Fix 1** (`507dea2`, `92e1411`): reasons once per step / iteration, plus an untimed re-warm. It passed a 10-iteration check but **failed at full length**: in the 50-iteration E4 N=1024 run (`local4060_b2r_e4n1024_20261101_051528`) the clock fell to about 400-1200 MHz and the re-warm failed in 91-100 % of steps (only the idle bit set, 73-78 °C). That run is an **instrument failure**; its ratios are not used. HYPOTHESIS: the re-warm synchronized after every ~0.05 ms burst, so the GPU duty cycle stayed low.
3. **Fix 2** (`efece3c`): the re-warm enqueues ~20 ms of work back to back before one sync. Checked at full length with no ratios looked at: E4 B8_N4096 in-band 1.0, B1_N1024 0.986.
4. **New rules, each fixed before the data they apply to:** THROTTLED if the modal clock is > 5 % below the process's own warm-up reference (`190345a`); THROTTLED if the warm-up reference is < 0.95 x the best reference in the configuration (`7553210`); host check before every process (`c3c900c`). Applied to the E3 data already taken, the new rules change nothing.

## Limitations: what this laptop cannot establish

- **Clock stability.** Clocks cannot be locked under WSL2. All exclusions are by measured clocks, and 7 of 11 configurations lost too many processes.
- **B1 is THROTTLED at every N.** Each B1 process warms up at about 1890 MHz, then runs mostly at 2595 MHz with some reads near 1890, and the modal-clock rule excludes it. Changing that rule after seeing the data would loosen it; a B1 result needs a new rule declared before a fresh run under a new harness version.
- **N=1024 clocks wander** within processes (about 1500-2200 MHz) on a warm GPU (process starts at 71-79 °C, all but the first at 76-79 °C).
- **Start-of-timing transient:** in the clean B32_N8192 p0 trace the first ~30 timed samples are about 1.9 ms for both variants against about 1.3 ms afterwards (`figures/fig1_clock_trace_e4_B32_N8192_clean.png`). Ratio-of-medians absorbs most of it, but it was not excluded.
- **No 32K.** Both earlier 32K laptop runs hit software thermal slowdown; 32K was not attempted.
- **No Nsight profile, no measured DRAM bandwidth, no second GPU.** The roofline floor (256 GB/s) is the nominal spec, not measured.
- **E3 and E4 come from different harness versions** (E3 before the saturating re-warm). They are never pooled.
- **E0b (A1 vs C1) was not run locally:** its harness (`c1_rtable`) still calls the slow NVML query per sample (in-band 0.75 in a dry run). The paper can cite the single clean 1K A1/C1 run from 2026-10-02, labelled as one run.
- **The windows host:** Brave reopened itself during the Block 2 re-run (the instrument-failure run); from then on a check before every process guaranteed it was closed. Surface, cooling and room temperature were not verified.

## What this supports in a paper

A characterization paper with a laptop-labelled result: latent compression (3.5-13x fewer bytes) does not translate into a uniform decode speedup on this GPU; the measured ratio ranges from 0.865 to 1.030 across claim-eligible configurations and depends on batch and context. HYPOTHESIS, not tested here: grid size matters as much as bytes, because padded A3 launches a CTA per (split, KV head) while C3 launches one per split, so occupancy at different batch sizes can swing the ratio. The instrument findings above (a telemetry call that is cheap on native Linux stalls the GPU under WSL2, and a short instrument check missed a full-length clock collapse) are secondary methodology evidence.
