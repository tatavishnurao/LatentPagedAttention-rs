---
title: "Measuring Paged Latent-Cache Decode on Consumer GPUs: Clock Artifacts, Numerics, and Fair Baselines"
author: "[TODO: authors and affiliations]"
date: "Draft, 2026-11-01"
abstract: |
  We compare a paged latent-KV decode kernel with a paged full-KV GQA kernel built with the same tools, on one laptop RTX 4060 under WSL2 (not a cloud result). Of 11 configurations, 4 had enough clean processes to support a claim. In those 4 the latent kernel moves 7.3 to 13.2 times fewer analytic bytes, yet the time ratio ranges only from 0.865 to 1.030: faster in two, slower in one, not shown to differ in one. Compression is not the same as speedup on this GPU. Getting even this mixed answer required three fixes to the measurement. In the current harness, a 20.8 ms telemetry call (measured idle, one driver) idled the GPU between short kernels, so in one process 73 % of layer samples ran at 210 MHz while a modal-clock check passed it; a 10-iteration instrument check of the first fix then missed a collapse that appeared at full length. Separately, slow output writes had caused idle clocks in an older harness. On one heavy-tailed dataset, the large latent-kernel error comes from FP16 rounding of the projected query, not accumulation; and a fair full-KV baseline under cuTile needs query-head padding to reach tensor cores.
---

<!--
Labels used in this draft (remove before submission): [M:key] MEASURED, evidence file in Appendix B under that key;
[S:key] SOURCE, read in code or a design document; [H] HYPOTHESIS, not tested. Every result in this paper is from
one laptop RTX 4060 under WSL2 and is not a cloud result.
Tables are included from paper/generated/ by scripts/paper/assemble_paper.py at the include markers.
-->

# Introduction

Decode attention reads the whole KV cache of every sequence once per generated token, so formats that shrink the cache look like direct speedups. Multi-head latent attention (MLA) is one such format: it caches a low-dimensional latent vector per token and projects queries into the latent space at decode time [2]. Paging [1] is orthogonal: it changes how the cache is allocated, not how many bytes it holds.

We asked a narrow question: on one consumer GPU, with both kernels written in the same tile language and checked against the same references, does a paged latent decode kernel beat an equal-treatment paged full-KV kernel? We expected the answer to follow the byte ratio. It does not.

**Thesis.** Across the 4 claim-eligible configurations, a latent KV cache moves 7.3 to 13.2 times fewer analytic bytes than our padded full-KV baseline, yet on this GPU the decode time ratio ranges only from 0.865 to 1.030 [M:timing]; compression is not the same as speedup, and getting even that answer required fixing three measurement traps.

**Contributions.**

1. *A clock-validity failure mode, and an exclusion protocol for hosts where clocks cannot be locked* (Section 3): under WSL2 a 20.8 ms NVML call (measured idle, one driver) left 73 % of one process's layer samples at 210 MHz, about 16 times slower, a modal-clock rule passed it, and a 10-iteration check of the first fix passed where the 50-iteration run collapsed; idle clocks distorting short kernels is known [10, 11], these specifics we have not found reported (Section 7).
2. *Numerics* (Section 4): on one heavy-tailed dataset the large latent-kernel error comes from FP16 rounding of the projected query, not accumulation, and a hi/lo split fixes it at double the score MMAs and, at the one size where both modes were timed, 1.7 to 5.2 % more time (CIs include 1.0).
3. *A fair baseline under cuTile's tensor-core rules* (Section 5): in our probes (4 and 16 rows, 256 and 512 outputs) only 16-row tiles with at least 512 outputs got HMMA, so GQA needs query-head padding, and at one configuration (E3 N=8192, warm L2) unpadded A3 is only about 3.6 % faster and the latent win holds against both.
4. *A mixed, bounded result* (Section 5): of 11 configurations the latent kernel is faster in two, slower in one, not shown to differ in one, and seven are descriptive only; L2 size is not an axis of E4.

# Setup

**Hardware and software** [M:timing]. One laptop RTX 4060 (8 GB, sm_89) under WSL2; Table 1. Clocks cannot be locked under WSL2, so every validity decision is made from measured clocks (Section 3).

<!-- include: generated/tab_setup.md -->
*Table 1: Setup. The shape is synthetic and model-shaped (`model_small`), not a production model.*

**Kernels** [S:kern]. All are written in cuTile (Rust bindings, version 0.2), with FP16 storage and FP32 accumulation, split-K over blocks gathered through a block table, a log-sum-exp reduce, at most 128 two-block steps per CTA, and host-side table and alignment validation before every launch.

- *A3 (full KV, paged GQA).* One CTA per (split, KV head), covering the 4 query heads of that KV head. *Padded A3*, the pre-declared baseline, zero-pads them to 16 rows so both GEMMs reach tensor cores (Section 5).
- *C3 (latent, paged).* One CTA per split, covering all 16 query heads, which share one latent tile. It projects the queries once per CTA in FP32 (q' = q P^K); the reduce applies the value projection once per head. *C3 fast* uses an FP16 q'; *C3 accurate* uses q' = hi + lo (Section 4).
- *A1 and C1*, earlier unsplit full-KV and latent pipelines, appear only in Trap 1; no A1/C1 ratio is claimed.

**Benchmarks** [S:e4]. *E3* times one decode step (split + reduce) for one sequence, with no L2 flush, so its working set stays L2-resident across iterations. *E4* times all layers of one decode step for a batch of B sequences of length N with deterministic, unequal active lengths; each layer has its own pool, and the harness writes a 64 MiB buffer before every layer to flush L2. Each sequence owns a disjoint slice of the pool, so every KV or latent block is read once per layer. Bytes are *analytic* (counted from the access pattern), not measured DRAM traffic.

**Timing protocol** [S:rules]. Each process checks correctness against a CPU reference, runs 10 warm-up iterations and a clock warm-up (its second-half modal SM clock is the process's reference clock), then 50 timed iterations with variants interleaved in rotating order. Each configuration gets at least 5 cool-start processes (below 80 C, at most 8 attempts, abort at 91 C). The estimate is the geometric mean over clean processes of per-process ratios of medians, with a 95 % hierarchical-bootstrap CI. A claim needs at least 5 clean processes; parity would need TOST (90 % CI inside [0.95, 1.05]), which we did not compute. Which runs count was fixed in writing before the re-runs.

# Measurement validity: three traps

An E4 layer at B=1, N=1024 takes about 24 µs at the reference clock [M:void]. Any host-side pause between such kernels lets the GPU drop to an idle P-state. We hit this three times in different forms (Table 2; none of these processes is used for any ratio).

<!-- include: generated/tab_clock_validity.md -->
*Table 2: Clock validity of single instrument-check processes. In-band fraction: share of in-process SM clock reads within 95 % of the process's reference clock. Last row: the five processes of the campaign run that failed at full length [M:checks].*

## Trap 1: slow output writes (older A1/C1 harness only)

An earlier A1/C1 harness wrote oracle and correctness files to a slow OneDrive-backed Windows path during timing, so the GPU idled at low clocks [M:t1]. That voided the A1/C1 ratios of 2026-10-02. The E3/E4 harness used for every ratio in this paper never wrote output during timing [S:harness].

## Trap 2: a 20 ms telemetry call

The E3/E4 harness read NVML clocks, temperature and clock-event-reason bits after every timed sample. Under WSL2 the reasons query is slow. We re-measured it as a separate micro-measurement, not campaign data: 50 back-to-back CPU calls with the GPU idle, once, at 62 C on one driver (617.42), gave a median of 20.8 ms and a maximum of 29.0 ms, against medians of 0.03 to 0.15 ms for the clock and temperature reads [M:nvml]. Called after every sample, it left the GPU idle for several hundred times longer than the kernel it had just timed.

In the dry run before any fix, 73 % of E4 layer samples were at exactly 210 MHz and the in-band fraction was 0.09; the median padded-A3 layer took 378 µs at idle clock against 24 µs at the reference clock, about 16 times slower in this one process [M:void]. Our original THROTTLED rule compared clock reads with the process's *modal* clock, which was itself 210 MHz, so it passed the process. The dry run is VOID. The trap depends on kernel length: with the per-sample query still in place, an E3 process at N=1024 had an in-band fraction of 0.29, while one at N=8192, whose kernels are longer, stayed at 1.00 (Table 2).

## Trap 3: a short instrument check that passed

Fix 1 read the reasons bits once per step and added an untimed re-warm of short bursts, each followed by a synchronization. A 10-iteration check of B1_N1024 was fully in band (1.00). The 50-iteration campaign run at N=1024 then collapsed: across its five processes the in-band fraction was 0.002 to 0.096 [M:checks]. It is reported as an instrument failure; none of its ratios is used. [H] In that run the re-warm synchronized after every short burst, so the GPU duty cycle stayed low and the clock never ramped. Fix 2 enqueues about 20 ms of work back to back before one synchronization and repeats until the clock is within 95 % of the reference or 500 ms pass. Checked at full length before any ratio was looked at: in-band fractions of 0.99 (B1_N1024) and 1.00 (B8_N4096) [M:checks]. All primary E4 data use fix 2; the primary E3 data use fix 1 (Section 5).

## The protocol that came out of it

Beyond the timing protocol in Section 2 [S:rules], a process is **THROTTLED** if any SM clock read is more than 5 % below its modal clock or any thermal or hardware-slowdown bit is set; if its modal clock is more than 5 % below its own warm-up reference (this closes the Trap 2 gap); or if that reference is below 0.95 times the best reference in the configuration. The **in-band fraction** is reported for every process as an instrument check, not a gate. Output from a broken harness state is **VOID** (or an **instrument failure**), and every harness change gets a full-length check first. Each block runs **once**: a configuration with too few clean processes is reported, not re-run until it passes.

# Numerics

Speed comparisons are only fair between kernels with comparable error. Besides FP32 oracles, we check each kernel against an *exact* reference (FP32 arithmetic on the FP16-stored cache) and a *matched-rounding* reference that applies the kernel's own FP16 roundings and step structure but accumulates in FP64 [S:kern]. Total error is kernel vs exact; kernel-attributable error is kernel vs matched-rounding. Figure 1 shows the GPU correctness results (all values in Table C2).

**FP16 q' is the cause, not P accumulation.** On heavy-tailed inputs, C3 fast reaches 4.182e-3 at N=8192 [M:scale], above our 3e-3 gate. A CPU emulation of the same dataset at the same length (one dataset, one N) reproduces it and isolates the source (Figure 1b): with all-FP32 arithmetic the error is 1.907e-05; rounding only P to FP16 gives 1.426e-03; rounding only q' to FP16 gives 4.163e-03, the same as rounding both [M:emu]. P was already accumulated in FP32, so a higher-precision accumulator would not have helped. [H] Heavy-tailed inputs produce large scores, so the 2^-11 relative rounding of q' shifts the probabilities of the dominant tokens.

**A hi/lo split fixes it.** C3 accurate represents q' as an FP16 high part plus an FP16 low part and issues two score MMAs into one FP32 accumulator. Its heavy-tailed error at N=8192 falls to 1.088e-3 [M:scale], comparable to padded A3 (1.336e-3), which has the same FP16-P term. In the compiled code, accurate mode doubles the score MMAs (8 to 16 per CTA per step, against 8 PV MMAs), so total MMAs per step rise by 50 %; registers stay at 95 and loads are unchanged [M:sass]. Its run-time cost was measured at one size only, E3 N=8192 with a warm L2: C3 accurate / C3 fast is 1.052 [0.999, 1.089] in the primary run and 1.017 [0.974, 1.057] in the sensitivity run, 5 clean processes each [M:fastacc]. Both CIs include 1.0, so at that size the cost (point estimates 5.2 % and 1.7 %) is not shown to differ from zero. This is a secondary analysis with C3 fast as the baseline, computed after the runs.

**Kernel error is small and separate.** Kernel-attributable error is at most 4.072e-4 for every variant and input [M:scale]; the rounding choices, not the kernels' arithmetic, set the total. Bounding the plan at 128 two-block steps per CTA keeps uniform-input error flat from N=4096 to 32768 (Table C2).

![Numerics. (a) Max error vs N, total (solid) and kernel-attributable (dotted), heavy-tailed (circles) and uniform (faded squares) inputs. (b) CPU emulation at N=8192 on heavy-tailed inputs: which FP16 rounding produces the error. GPU correctness only; laptop RTX 4060, WSL2, not a cloud result.](../reports/local_4060_20261101/figures/fig4_numerics.png)

# Results

## Main result

Table 3 gives every configuration of the primary dataset, including the descriptive ones; bytes, bandwidth, clocks and temperatures are in Appendix Table C1. Figure 2 plots the ratios against the padded-A3 bytes per layer.

<!-- include: generated/tab_timing.md -->
*Table 3: Primary results, C3 accurate vs padded A3 [M:timing]. Clean / THROTTLED / hot-start process counts; median µs per step over clean processes (E4: all layers of one decode step; E3: one split and reduce for one sequence). Ratio: geometric mean of per-process ratios of medians, 95 % hierarchical-bootstrap CI. Claim: *shown*, at least 5 clean processes and the 95 % CI excludes 1.0; *not shown*, at least 5 clean processes but the CI includes 1.0 (no difference shown; parity not tested); *descriptive*, fewer than 5 clean processes, no claim. E3 and E4 rows come from different harness versions and are never pooled.*

Four configurations are claim-eligible. The latent kernel is faster at E3 N=8192, 0.893 [0.860, 0.936]. The E3 data come from the fix-1 harness (Section 3); every E3 N=8192 process had an in-band fraction of 1.0, a preliminary run taken with one extra application open agrees (0.888 [0.869, 0.914]) [M:prelim], and E3 and E4 are never pooled. It is also faster at E4 B8_N8192, 0.865 [0.842, 0.916]. It is slower at E4 B32_N4096, 1.030 [1.009, 1.051]. At E4 B32_N8192 it is 0.972 [0.915, 1.029]: no difference shown, and no parity claimed. Over all 11 configurations, descriptive ones included, the byte ratio is 3.5 to 13.2 (Table C1). The other seven (all B=1 and E4 N=1024 configurations, E3 N=1024, and B8_N4096) had fewer than 5 clean processes and are descriptive only.

![Regime map: C3 accurate / padded A3 with 95 % CI vs padded-A3 working set per layer. Filled: claim-eligible; hollow: fewer than 5 clean processes; configurations with 0 clean processes have no point. Dotted: the ratio if both kernels ran at the same achieved bandwidth. The L2 line is a reference only: E4 flushes L2 before each layer. Laptop RTX 4060, WSL2, not a cloud result.](../reports/local_4060_20261101/figures/fig3_regime_map.png)

If both kernels were bandwidth-bound, the ratio would sit near the reciprocal of the byte ratio (dotted line, Figure 2); every point is far above it. [H] At these sizes C3 is limited by something other than bytes, such as its smaller grid or launch overhead: padded A3 launches one CTA per (split, KV head) and C3 one per split, so the two kernels fill the GPU differently as B and N change. We did not test this.

## The baseline: why padded A3, and how much padding costs

cuTile decides whether an `mma` lowers to HMMA. In our compile-only SASS probes (sm_89, 4 warps per CTA) only 16-row tiles with at least 512 outputs got HMMA [M:sass]. The unpadded A3 has 4 query rows per KV head and compiled to 0 HMMA; widening its step to 512 score outputs (8 blocks) still gave 0 HMMA and added 616 bytes of stack spills; a 16 x 16 tile (256 outputs) also fell back. Zero-padding to 16 rows gave 40 static HMMA with no spills, at 4 times the multiply-adds (Table 4). We tested only 4 and 16 rows and 256 and 512 outputs, so the threshold between them is not established.

<!-- include: generated/tab_kernels.md -->
*Table 4: Compile-only SASS for the kernels at the `model_small` shape, sm_89 [M:sass].*

Is padded A3 a strawman? A secondary run declared before its data [S:rules] added unpadded A3 at one configuration, E3 N=8192 with a warm L2, in one run (Table 5). Unpadded A3 is only about 3.6 % faster than padded A3, 0.964 [0.928, 0.989], despite doing a quarter of the multiply-adds, and C3 accurate is still faster against it, 0.924 [0.888, 0.971] [M:sens]. The E3 N=8192 win holds against both baselines (margin about 7.6 % instead of about 11 %). E4 was not repeated with unpadded A3.

<!-- include: generated/tab_sensitivity.md -->
*Table 5: Secondary sensitivity run, E3 N=8192, 5 clean processes, own harness version [M:sens].*

## What E4 can and cannot show

Because E4 flushes L2 before every layer (Section 2) [S:e4], *L2 size is not an axis of E4*, and the hypothesis that the latent cache wins once the working set outgrows L2 is not tested. E3 vs E4 contrasts warm and cold caches, not sizes. With three claim-eligible E4 points, no trend in B or N can be established.

# Limitations and what we do not claim

- **One laptop GPU under WSL2.** No cloud GPU, no second GPU, no native Linux host. Clocks cannot be locked; all exclusions come from measured clocks, and 7 of 11 configurations lost too many processes. No B=1 result exists: B=1 processes ran mostly at 2595 MHz with some reads near 1890 MHz, and the modal-clock rule excluded them. Changing the rule after seeing the data would loosen it, so we did not.
- **No profiler data.** No Nsight profile and no measured DRAM bandwidth; bytes are analytic, and the only DRAM bandwidth figure we have, 256 GB/s, is a nominal specification, not a measurement.
- **Narrow grid.** Context lengths up to 8192 only (earlier 32K laptop runs hit thermal slowdown), one synthetic shape, FP16 storage, no RoPE, no real model checkpoint, and no external baseline such as FlashInfer [7] on identical inputs (planned, not run).
- **Harness versions.** E3 primary data predate fix 2; E3 and E4 are never pooled. A start-of-timing transient is visible in at least one clean E4 trace and was not excluded; ratios of medians absorb most of it.
- **The NVML latency** was measured idle on one driver; we do not know how it behaves under load or on native Linux.
- **Not claimed:** that latent caches are faster in general, that the result transfers to data-centre GPUs, that L2 capacity explains anything in E4, that grid size explains the ratios (hypothesis), or parity at any configuration.

# Related work

PagedAttention and vLLM [1] page the KV cache to cut memory waste; paging does not reduce bytes per token. Grouped-query attention [3] shares each key/value head among several query heads, which is why our baseline needs padding. DeepSeek-V2 [2] introduced multi-head latent attention, reporting a 93.3 % KV-cache reduction for that model. MHA2MLA [5] and TransMLA [6] convert multi-head models to MLA. KIVI [4] shrinks the cache by 2-bit quantization, which we did not evaluate. FlashAttention-2 [8] and FlashInfer [7] are optimized attention kernels; our split-K decode follows FlashDecoding [TODO cite: Flash-Decoding, verify source]. Our kernels use cuTile [TODO cite: cuTile / cutile-rs, verify source], and the clock-event-reasons call is part of NVML [TODO cite: NVML API reference, verify source].

**Measurement methodology.** Innocuous-looking details of an experimental setup can bias systems measurements enough to change a conclusion [9]; our three traps are GPU instances of that hazard. That GPU clock state changes performance substantially is well known: the DVFS literature models core and memory frequency as a variable, for example for energy [10], and frequency switching itself takes measurable time [11], which is why a GPU left idle between short kernels does not return to its reference clock at once. The standard answer is a fixed clock; Nsight Compute, for instance, locks GPC and memory clocks during profiling by default [12]. Under WSL2 we could not lock clocks, so we exclude processes by measured clock instead (Section 3). That a short warm-up can look settled while a longer run never reaches a steady state is documented for language virtual machines [13]; Trap 3 has the same shape on a GPU. In a bounded, non-exhaustive search we did not find reports of three specifics: an NVML query costing about 20 ms under WSL2 and idling the GPU between timed samples (we did not measure it on native Linux); a modal-clock validity rule that passes a process whose mode is the idle clock; and a 10-iteration instrument check passing where the 50-iteration run collapsed. We present these as observations we have not seen reported, not as established novelty.

# References

[1] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. arXiv:2309.06180 (arXiv comment, unverified: SOSP 2023).

[2] DeepSeek-AI, Aixin Liu, Bei Feng, et al. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv:2405.04434.

[3] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. arXiv:2305.13245 (arXiv comment, unverified: accepted at EMNLP 2023).

[4] Zirui Liu, Jiayi Yuan, Hongye Jin, et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. arXiv:2402.02750 (arXiv comment, unverified: ICML 2024).

[5] Tao Ji, Bin Guo, Yuanbin Wu, et al. Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs. arXiv:2502.14837 (arXiv comment, unverified: accepted to ACL 2025).

[6] Fanxu Meng, Pingzhi Tang, Xiaojuan Tang, Zengwei Yao, Xing Sun, Muhan Zhang. TransMLA: Multi-Head Latent Attention Is All You Need. arXiv:2502.07864.

[7] Zihao Ye, Lequn Chen, Ruihang Lai, et al. FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving. arXiv:2501.01005 (arXiv comment, unverified: accepted by MLSys 2025).

[8] Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691.

[9] Todd Mytkowicz, Amer Diwan, Matthias Hauswirth, Peter F. Sweeney. Producing Wrong Data Without Doing Anything Obviously Wrong! Proceedings of the 14th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '09), 2009, pp. 265-276. DOI 10.1145/1508244.1508275.

[10] Xinxin Mei, Qiang Wang, Xiaowen Chu. A Survey and Measurement Study of GPU DVFS on Energy Conservation. arXiv:1610.01784.

[11] Daniel Velicka, Ondrej Vysocky, Lubomir Riha. Methodology for GPU Frequency Switching Latency Measurement. arXiv:2502.20075.

[12] NVIDIA. Nsight Compute CLI documentation, option `--clock-control`, version v2026.3.1. https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html (accessed 2026-11-01).

[13] Edd Barrett, Carl Friedrich Bolz-Tereick, Rebecca Killick, Sarah Mount, Laurence Tratt. Virtual Machine Warmup Blows Hot and Cold. arXiv:1602.00602; DOI 10.1145/3133876 (arXiv journal-ref, unverified: OOPSLA, October 2017, pp. 52:1-52:27).

<!-- Each entry above was read at its source on 2026-11-01: [1]-[8], [10], [11], [13] on their arXiv abstract pages; [9] via its Crossref record (proceedings DOI; abstract read from the SIGPLAN Notices record of the same paper, ACM page returned 403); [12] the literal option row of the NVIDIA page (description and default value `boost`).
Venues are given only as the authors' arXiv comment states them and are marked unverified; confirm against the proceedings before submission. -->

# Appendix A: Corrections ledger

Claims made earlier in this repository or in the paper brief that the evidence does not support, and what replaces them. Nothing was deleted from history; documents carry dated notes.

| # | Earlier claim (where) | Replacement | Evidence |
|---|---|---|---|
| A1 | "80 % of E4 layers ran at 210 MHz" in the VOID dry run (`docs/LOCAL_4060_RESULTS.md`, commit `507dea2` message, PR #10 description) | 73 % of E4 layer samples at exactly 210 MHz; in-band fraction 0.09. Dated correction note added to the results document; commit message and PR text not rewritten. | [M:void] |
| A2 | "~16x slower" as a general idle-clock factor (brief) | "up to about 16 times in one process" (E4 B1_N1024 dry run, 378 vs 24 µs); the 2026-10-02 slow-write case gave about 7.5 times (321.5 vs 43.0 µs). The two are kept separate. | [M:void], [M:t1] |
| A3 | Accurate mode costs "+50 % score MMAs" (brief) | Accurate mode doubles the score MMAs (8 to 16 per CTA per step); total MMAs per step +50 %. | [M:sass] |
| A4 | cuTile gives HMMA "only with >= 16 rows and >= 512 outputs" (brief) | "Only 16-row tiles with >= 512 outputs got HMMA in our probes"; only 4 and 16 rows and 256 and 512 outputs were tested. | [M:sass] |
| A5 | Slow output writes as a cause of 210 MHz samples in the current campaign (implied by the brief's contribution 1) | The slow-write trap belongs to the older A1/C1 harness (2026-10-02) only; E3/E4 timing never wrote output during timing. | [M:t1], [S:harness] |
| A6 | The ~20 ms NVML latency, cited from an interactive measurement whose output was not saved | Re-measured into a committed file as a separate micro-measurement (median 20.8 ms, max 29.0 ms, idle, driver 617.42). | [M:nvml] |
| A7 | Literature review names works without references (`docs/01_literature_review.md`) | Only works verified on their arXiv pages are cited; everything else is marked `[TODO cite]`. | this paper |
| A8 | "Above L2" labels for E4 configurations in the campaign's block reports | E4 flushes L2 before each layer; L2 size is not an axis of E4 (already corrected in `docs/LOCAL_4060_RESULTS.md`). | [S:e4] |
| A9 | Accurate mode's run-time cost "is not measured" (`docs/E3_RESULTS.md`, 2026-10-03, written before any timing) | Measured at one size, E3 N=8192, warm L2: C3 accurate / C3 fast 1.052 [0.999, 1.089] (primary) and 1.017 [0.974, 1.057] (sensitivity run); CIs include 1.0. | [M:fastacc] |
| A10 | Abstract and contribution 1 of the first draft (merged in PR #11) listed slow output writes and the telemetry call together as causes of the idle-clock samples, so the 73 % could be read as caused by output writes | The 73 % at 210 MHz comes from the telemetry call in the E3/E4 harness only; slow output writes are named separately as an older A1/C1-harness defect, as in Section 3. | [M:void], [M:t1] |
| A11 | Limitations cited "our roofline figure (not shown)" | Reference to a figure the paper does not contain removed; the 256 GB/s value is stated as a nominal specification, not a measurement. | this paper |
| A12 | Contribution 1 claimed "the protocol that catches" the clock failure mode, and Related work said clock/DVFS literature was unsurveyed | GPU clock effects and fixed-clock protocols are prior art [9]-[13]; contribution 1 is narrowed to an exclusion protocol for hosts where clocks cannot be locked, plus three specifics we have not found reported in a bounded search. | Section 7 |

# Appendix B: Evidence and reproducibility

**Evidence keys.** All paths are relative to the repository root.

| Key | Files |
|---|---|
| M:timing | `reports/local_4060_20261101/primary_merged_analysis.json` (from runs `local4060_b1r_e3_20261101_051132`, `local4060_e4n4096_20261101_054159`, `local4060_e4n8192_20261101_054853`, `local4060_e4n1024v2_20261101_055641`) |
| M:sens | `reports/local_4060_20261101/runs/local4060_e3a3sens_20261101_063414/variants_analysis.json` |
| M:fastacc | `reports/local_4060_20261101/runs/local4060_b1r_e3_20261101_051132/variants_analysis_vs_c3_fast.json` and `.../local4060_e3a3sens_20261101_063414/variants_analysis_vs_c3_fast.json` (secondary analysis: `scripts/e0b/analyze_variants.py <run>/e3e4 --baseline c3_fast` on the raw samples in `raw/<run>.tar.xz`; the same command without `--baseline` reproduces the committed primary analysis exactly) |
| M:prelim | `reports/local_4060_20261101/runs/local4060_b1_e3_20261101_043331/` |
| M:void | `reports/local_4060_20261101/instrument_checks/e4_B1_N1024_dryrun_VOID/` |
| M:checks | `reports/local_4060_20261101/instrument_checks/`; `reports/local_4060_20261101/runs/local4060_b2r_e4n1024_20261101_051528/` |
| M:nvml | `reports/nvml_latency_20261101/nvml_latency.json` (separate micro-measurement, not campaign data) |
| M:t1 | `reports/diagnosis_20261002/e0/` (`seq1024_run1`, `seq1024_cool2`); `docs/DIAGNOSIS_2026-10-02.md` (Trap 1: one 1K run took 122 s of wall time against 5 s with output on ext4, modal SM clock 210 MHz, 79 % of samples low-clock; A1 context kernel 321.5 µs at idle clock against 43.0 µs at boost) |
| M:sass | `reports/e3_splitk_20261002/sass/`, `sass_a3_equal_treatment/`, `sass_c3_modes/`; `docs/E3_RESULTS.md`; `docs/E3_SPLITK_DESIGN.md` (b) for the 256-output tile |
| M:scale | `reports/e3_splitk_20261002/scale_c3_modes/scale_suite.log` |
| M:emu | `reports/e3_splitk_20261002/scale/emulate_c3_numerics.out` |
| S:rules | `docs/E0B_CLOUD_RUNBOOK.md` section 4; `docs/LOCAL_4060_PLAN.md` and its addenda |
| S:kern | `crates/plkv-kernels/src/cutile/e3_splitk.rs`; `crates/plkv-kernels/src/splitk_plan.rs`; `docs/E3_SPLITK_DESIGN.md` |
| S:e4 | `crates/plkv-kernels/src/e4_config.rs`; `crates/plkv-kernels/examples/e3_splitk_gpu.rs` (`launch_async`, `L2_FLUSH_WORDS`) |
| S:harness | `crates/plkv-kernels/examples/e3_splitk_gpu.rs`, `crates/plkv-kernels/examples/timing_common/mod.rs` |

**Regenerating tables, figures and checks (CPU only).**

```
python3 scripts/paper/make_tables.py                 # paper/generated/*.md, numbers.json, summary.json
python3 scripts/paper/check_numbers.py               # every number in paper/*.md traceable
python3 scripts/paper/assemble_paper.py              # paper/build/paper.md (tags) + paper_submission.md (no tags); page estimate
uv run --no-project --with matplotlib python scripts/paper/fig4_numerics.py \
  --out reports/local_4060_20261101/figures/fig4_numerics.png
uv run --no-project --with matplotlib python scripts/paper/fig3_regime_map.py \
  reports/local_4060_20261101/primary_merged_analysis.json --l2-mib 32 \
  --title "Laptop RTX 4060 (WSL2), not a cloud result" \
  --l2-label $'L2 32 MiB: reference only,\nE4 flushes L2 before each layer' \
  --out reports/local_4060_20261101/figures/fig3_regime_map.png
python3 scripts/paper/nvml_latency.py --out <dir>/nvml_latency.json   # NVML calls only, no kernels
```

**GPU campaign (not re-run for this paper).** The command and per-run commits are in `reports/local_4060_20261101/README.md`; primary E4 runs used harness version `e3e4-rewarm-saturating-20ms-2026-11-01` at commit `c3c900c`, primary E3 used `pre-2026-11-01` (binary built at `92e1411`), and the sensitivity run `...+a3unpad` at `2ffd88e`.

**Hashes (SHA-256).**

```
d3ef6ca3984183f72c5b2d8457b5593ae27885ce8a714781848b39ce463e5633  reports/local_4060_20261101/primary_merged_analysis.json
8bb01bd5ea170bd6dffcca823f3329ca7d105d537c9f327fe4f4100e481ef182  reports/nvml_latency_20261101/nvml_latency.json
cc132ffefd418620368d35e27168e2f1830be98d610d1d5b1de7354dca8125d1  reports/e3_splitk_20261002/scale_c3_modes/scale_suite.log
6b777f94ef0c17e9f828af7d3aaead108c10292d9fb2b423befe6108d6e56e00  reports/e3_splitk_20261002/scale/emulate_c3_numerics.out
```

# Appendix C: Additional tables

<!-- include: generated/tab_timing_appendix.md -->
*Table C1: Per-configuration analytic bytes and bandwidth, clocks and temperature for the primary dataset [M:timing]. GB/s is analytic bytes divided by the clean median time, not measured DRAM bandwidth. Modal SM clock over clean processes; maximum in-process temperature over all processes of the configuration.*

<!-- include: generated/tab_numerics_scale.md -->
*Table C2: Max |error| for each variant, total / kernel-attributable, uniform and heavy-tailed inputs, bounded split plans; GPU correctness only [M:scale].*
