# E3 design: split-K paged decode with an amortized in-CTA projection

Status: **implemented, correctness only** (2026-10-02, branch `diagnose/e3-splitk`); results and SASS in [`E3_RESULTS.md`](E3_RESULTS.md). No timing. Written 2026-10-02 on `diagnose/feasibility-2026-10-02`.

It replaces the rejected "fuse the projection into the C1 score kernel" step: projection-into-score fusion is out of scope. The projection is instead amortized across a CTA that covers many blocks.

Evidence labels:
- **MEASURED**: from a named artifact.
- **SOURCE**: from code.
- **DERIVED**: arithmetic from the stated shape.
- **HYPOTHESIS**: not verified.

## 0. Constraints from probes P1–P3 (MEASURED, compile-only, sm_89; `docs/MMA_PROBE_PLAN.md`)

These were measured after the design was first written. They override anything below that disagrees.

**(a) The alignment hint changes nothing at runtime.**
- cuTile's runtime JIT derives divisibility from the real device pointers, shapes and strides (`cutile-0.2.0/src/tensor.rs:582`, `compute_spec`).
- The runtime-JIT cubins of today's kernels already match the `max_divisibility(16)` builds: same load widths, ±3 % instruction count. Examples: latent context REG 128, 3152 instructions, 50×LDG.E + 4×LDG.E.128 + 25×LDG.E.64; A1 context REG 181.
- So forcing the hint is **not** an optimization, and E3 claims no speedup from it. It matters only for compile-only probes, and as a correctness hazard: a forced hint on misaligned storage is wrong. Hence the host-side check in (e).

**(b) The score GEMM must cover at least 2 blocks (32 tokens) per `mma` step. This is mandatory.**
- With 128 threads (4 warps), cuTile lowers `mma` to `HMMA.16816.F32` only when the output tile has at least 4 × (16×8) = 512 elements.
- A per-block score tile (16 heads × 16 tokens = 256 outputs) falls back to CUDA cores (FMUL+FADD) in both operand orientations.
- Two gathered blocks concatenated to 32 tokens (16×32 = 512 outputs) give full HMMA, including the register-transpose form.
- The PV GEMM (16×32 output) meets the rule at one block per step.
- **Gather width:** with 2-block steps the gathered tiles load as **`LDG.E.64`** (9 per warp at R=4: 8 tiles + q′). One-block steps load as `LDGSTS.E.64` into shared memory. **No 128-bit gather was observed** in the gather probes; a 1 KiB tile over 128 threads is exactly 8 B per thread. The 128-bit table-indirected loads in P2 come from the existing one-head-per-CTA kernels, not from an `mma` path. Whether a different tile/warp arrangement yields `LDG.E.128` is open.

**(c) Pipeline depth cannot be controlled directly (SOURCE).**
- cuTile 0.2 exposes no explicit pipeline-depth or stage-count control.
- Entry-level hints: `occupancy`, `num_cta_in_cga`, `max_divisibility` (`cutile-compiler-0.2.0/src/hints.rs:13-24`).
- Per-load hints: `latency: Option<i32>` and `allow_tma` on partition loads (`load_view_tko`, `cutile-0.2.0/src/_core.rs:2335`).
- `tileiras --help-hidden` lists no stage, unroll or max-register option.
- Software pipelining appears automatically when divisibility is known: today's context kernels gain a prefetch prologue and a reduction-draining epilogue (+1832 / +2384 instructions). The only levers on its depth and register cost are the `occupancy` hint, the per-load `latency` hint and loop structure. Their effect is **untested** (HYPOTHESIS).

**(d) Register target: ≤ 128 registers per thread for C3 and A3 split kernels.**
- Today: latent context 126–128, A1 context 168–181, gather probe 106 (R=4) and 162 (R=16, fully unrolled).
- ≤ 128 registers keeps ≥ 4 CTAs/SM (33 % theoretical occupancy).
- Means: keep the block loop a runtime loop (not a compile-time-unrolled R), keep the score GEMM on HMMA (the CUDA-core fallback was the main register cost in the probes), and try the `occupancy` hint if needed.
- Every build reports `cuobjdump --dump-resource-usage`. Exceeding 128 registers is a review flag, not a correctness failure.

**(e) Validate block tables on the host before every launch.**
- Every logical block that covers `active_seq_len` must map to a physical block in `[0, num_physical_blocks)`.
- The table must have at least `ceil(active_seq_len / 16)` entries.
- The latent/K/V base pointers and strides must pass `plkv_kernels::alignment::check_layouts` (16-byte base, unit inner stride, 16-byte outer strides).
- Duplicate physical blocks are allowed (read-only aliasing), but are reported.
- An invalid table fails before launch. Kernels never read out-of-range physical blocks.

## 1. Problem being solved

Today's pipelines have three structural costs.

| cost | evidence |
|---|---|
| Context reduction runs on 16 CTAs (one per Q head) as a serial loop over N/16 blocks. Only 16 of 24 SMs get a CTA, and achieved occupancy is 8.33 %. | SOURCE `p15b_model_profile.rs:215-239`; MEASURED P05 ncu (`docs/DIAGNOSIS_2026-10-02.md`) |
| Each score CTA serves one Q head, so a latent tile is fetched by 16 CTAs and a full-KV tile by 4. | SOURCE |
| C1 issues 4 launches and A1 issues 3. Launch overhead is ~15–18 % of the 1K pipeline at boost, and C1 is 1.13× slower than A1 even though its context kernel is 0.74× A1's. | MEASURED `reports/diagnosis_20261002/e0/seq1024_cool2` |

Both variants get the same treatment, so the latent-vs-full comparison stays fair:
- **C3** is the latent split-K kernel.
- **A3** is the full-KV split-K kernel with GQA grouping.

## 2. Canonical shape and notation

| symbol | meaning | value |
|---|---|---|
| Hq / Hkv / g | Q heads / KV heads / group size (Hq/Hkv) | 16 / 4 / 4 |
| d | head dim | 64 |
| dl | latent dim | 32 |
| blk | tokens per block | 16 |
| N | tokens | — |
| NB | blocks, N/16 | — |
| B | batch | — |
| S | number of splits | — |
| R | blocks per split, NB/S | — |

All arithmetic is FP32 accumulate. Storage is FP16.

## 3. C3 — latent split-K kernel

**Kernel 1, `c3_split`.** Grid `(S, G, B)`. G is the number of head groups per latent tile; the default is G = 1, so one CTA handles all 16 heads.

1. **Projection, amortized.** Load q `[16/G, 64]` and the K-projections of the KV heads in this head group, then compute `q' = q·P^K` as `[16/G, 32]`. That costs 2·(16/G)·32·64 FLOP once per CTA, not once per block (as B1 does) and not in a separate launch (as C1 does).
2. **Table entries.** Load the R entries for this CTA's contiguous logical range with one `[R]` tile load. This replaces R dependent single-entry loads (the per-iteration dependency A1/C1 pay today).
3. **For each pair of blocks among the R (two blocks per step, required by §0(b)):**
   - gather two latent tiles and concatenate them: `Z [32 tok, 32]` (FP16, 2 KiB);
   - compute scores `s = q'·Zᵀ / 8` as `[16/G, 32]` (512 outputs at G = 1), masked by the active length;
   - update online softmax: `m_new = max(m, rowmax s)`, `α = exp(m − m_new)`, `p = exp(s − m_new)`, `l = α·l + rowsum p`, `acc = α·acc + p·Z` with `acc` of shape `[16/G, 32]`.
   - At G = 1 the products are 16×32×32 (scores) and 16×32×32 (PV) tiles, both ≥ 512 outputs, which fit `cutile::core::mma` (tensor-core MMA, SOURCE `cutile-0.2.0/src/_core.rs:1264`) with FP16 operands and FP32 accumulate. Compile-only SASS shows `HMMA.16816.F32` at these shapes on sm_89 (`docs/MMA_PROBE_PLAN.md`). Runtime tensor-pipe utilization needs ncu `sm__pipe_tensor_cycles_active` once counters are available.
4. **Write the partial** `(m, l, acc)`: (16/G)·(1+1+32) FP32 values per CTA, which is 2,176 B at G = 1.

**Kernel 2, `c3_reduce`.** Grid `(Hq, B)`.
- Merge the S partials with log-sum-exp: `M = max mₛ`, `L = Σ e^{mₛ−M} lₛ`, `ctx = Σ e^{mₛ−M} accₛ / L`.
- Apply the V projection once per head: `[32]·P^V [32, 64]`, which is 4,096 FLOP per head.
- Write the `[64]` context.
- Optional one-launch variant: the last CTA to finish does the reduction, detected with an atomic counter (`atomic_add`, SOURCE `_core.rs:495`). Memory-ordering support in cuTile needs checking first.

**Choosing S.** Redundant projection FLOPs as a fraction of score+PV FLOPs is

    2·Hq·dl·d·S / (4·Hq·dl·N) = d·S/(2N) = 2/R   (with N = 16·R·S)

So R ≥ 16 blocks per CTA keeps projection overhead ≤ 12.5 %, and R ≥ 32 keeps it ≤ 6.25 %. The default is `R = max(16, ceil(NB/48))`, giving S = NB/R. G trades latent re-fetches (G× more L2 requests for Z, DRAM traffic unchanged) for more CTAs at small B·N.

## 4. A3 — full-KV split-K counterpart (GQA-grouped)

**Kernel 1.** Grid `(S, Hkv, B)`. Each CTA handles the g = 4 Q heads of one KV head over R blocks:
- load the R table entries once;
- per step of two blocks (same treatment as C3): gather K and V tiles `[32, 64]` (4 KiB each, FP16), compute scores `[4, 32]` with q `[4, 64]` via `mma`, update online softmax, and accumulate `acc [4, 64]` via `mma`;
- **tensor-core caveat (HYPOTHESIS, to be measured in E3 SASS):** with only g = 4 query rows, the score output (4×32 = 128) and PV output (4×64 = 256) are below the 512-output rule. A3 may therefore fall back to CUDA cores unless the 4 head rows are padded to 16, which costs 4× MACs. That asymmetry is inherent to GQA vs a shared latent, where all 16 heads share one tile. E3 reports A3's SASS as measured and does not claim tensor-core use it does not have;
- write the partial: 4·(2 + 64) FP32 = 1,056 B.

**Kernel 2.** LSE merge per head, write `[64]`. There is no projection.

Each K/V tile is fetched once per KV head instead of once per Q head, which removes today's 4× request amplification.

## 5. Bytes, FLOPs, launches per decode step (B = 1, one layer; DERIVED)

C3 uses G = 1. Bytes are unique bytes the step must read, with partials counted twice (written and read). "Floor" is unique bytes at 256 GB/s (spec value).

| N | variant | R, S | CTAs (kernel 1) | cache bytes | other bytes | total | DRAM floor | FLOPs | launches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1K | A1 (today) | — | 1024 + 16 + 16 | 1 MiB | probs 128 KiB | ~1.13 MiB | 4.6 µs | 4.2 M | 3 |
| 1K | C1 (today) | — | 16 + 1024 + 16 + 16 | 64 KiB | proj 64 KiB, probs 128 KiB | ~256 KiB | 1.0 µs | 2.2 M | 4 |
| 1K | A3 | R=16, S=4 | 16 | 1 MiB | partials 33 KiB | 1.03 MiB | 4.2 µs | 4.2 M | 2 |
| 1K | C3 | R=16, S=4 | 4 | 64 KiB | proj 64 KiB, partials 17 KiB | 145 KiB | 0.6 µs | 2.4 M | 2 |
| 8K | A1 (today) | — | 8192 + 16 + 16 | 8 MiB | probs 1 MiB | 9 MiB | 36.9 µs | 33.6 M | 3 |
| 8K | C1 (today) | — | 16 + 8192 + 16 + 16 | 512 KiB | proj 64 KiB, probs 1 MiB | 1.56 MiB | 6.4 µs | 16.9 M | 4 |
| 8K | A3 | R=16, S=32 | 128 | 8 MiB | partials 264 KiB | 8.26 MiB | 33.8 µs | 33.6 M | 2 |
| 8K | C3 | R=16, S=32 | 32 | 512 KiB | proj 64 KiB, partials 136 KiB | 712 KiB | 2.8 µs | 10.6 M | 2 |
| 32K | A3 | R=43, S=48 | 192 | 32 MiB | partials 396 KiB | 32.4 MiB | 133 µs | 134 M | 2 |
| 32K | C3 | R=43, S=48 | 48 | 2 MiB | proj 64 KiB, partials 204 KiB | 2.3 MiB | 9.4 µs | 70.3 M | 2 |

Notes:
- The A1/C1 rows include the FP32 score/probability round trip (Hq·N·4 B written and read twice) that split-K removes.
- C3 FLOPs = 2048·N (score + PV) + 65,536·S (amortized projection) + 65,536 (V projection).
- A3 FLOPs = 4096·N.
- At B = 1, C3 has 4× fewer CTAs than A3 for the same S. With G = 4 the CTA counts are equal (S·4). G is a sweep axis, not a fixed choice.

**Launch floor.** The planned measured value comes from E0b's empty-event and smallest-kernel spans (`reports/diagnosis_20261002/e0b/`; the smoke run gave about 1.4 µs and 5.1 µs respectively). That puts A3/C3 at about 2 × 5 µs ≈ 10 µs, compared with about 15 µs (A1, 15 % of its 1K pipeline) and 20 µs (C1, 18 %) today.

## 6. Correctness requirements

These are unchanged from the C1 protocol:
- **Correctness bound:** absolute 5e-3 against the Rust FP32 reference and the independent FP64 reconstructed-K/V oracle; probability row sums within 1e-4. C3/A3 do not materialize probabilities; check `l` and an exported diagnostic P for one configuration.
- **Paging:** non-identity block table; partial-final-block masking at active lengths 17, 129, 513 and N−3.
- **Projection freshness:** a changed-query refresh test confirms the projection is recomputed.
- **Split invariance:** for S ∈ {1, 2, 4, S_default}, contexts agree within 1e-5 relative. Only FP32 reassociation differs.
- **Optional bitwise check:** with S = 1, G = 1, C3 matches a serial reference except for summation order.

## 7. Measurement protocol

Use the E0b harness: output on ext4 with all I/O after timing, NVML clock per sample, keep-alive, A1/C1/A3/C3 interleaved with rotation, ≥5 cool-start processes per configuration. Apply the THROTTLED rules, and keep THROTTLED runs out of all ratios. Report hierarchical-bootstrap 95 % CIs on per-process ratios of medians.

Configurations:
- N ∈ {1K, 8K, 32K}; S at the default plus {½, 2}× the default; G ∈ {1, 4}; B = 1.
- B ∈ {8, 32} only after E4 adds batched inputs.

## 8. Success and failure criteria (decided before running)

1. **Topology fix, both variants:** A3/A1 and C3/C1 pipeline upper 95 % CI bound ≤ 0.33 at 8K (≥3× faster). Failure for either means the split-K implementation, not the latent idea, is the bottleneck. Stop and profile (E1).
2. **Baseline credibility:** A3 within 2× of FlashInfer paged decode at the same shape (E2, which needs install approval). If not, an A3 comparison still cannot support a claim against "paged GQA decode".
3. **Latent claim, this regime:** C3/A3 upper CI bound < 1.0 at 8K and 32K, B = 1. Even then the claim is limited to "single-layer, L2-resident, B = 1". The DRAM-bound claim needs E4.
4. **Projection amortization:** at R ≥ 16, C3 time with the projection disabled (a test-only flag) differs from normal C3 by ≤ 10 %. Failure means the amortization arithmetic in §3 is wrong for this hardware.
5. **Stop rule:** if criterion 1 passes but C3/A3's CI includes 1.0 at both 8K and 32K, do no further kernel work on this shape. Go to E4, where the byte ratio, not per-iteration work, decides the outcome.

## 9. Risks and unknowns

- **`mma` lowering: resolved for compile (MEASURED, CPU-only probe, `docs/MMA_PROBE_PLAN.md`).** FP16 `mma` at 16×T×32 (T = 64…512) lowers to `HMMA.16816.F32` on sm_89, with the whole GEMM on the tensor cores. Probe P1: the scalar `LDG.E.U16` loads seen in hint-less compile-only builds become `LDG.E.128` with `max_divisibility(16)` or runtime-style `spec_args`. That is a probe artifact with no runtime effect (§0(a)). **Remaining risk:** paged gathers (row offset computed from a loaded table entry) are untested; E3 must check SASS on the paged kernel. Runtime tensor-pipe utilization is still unmeasured.
- **Static specialization:** kernels are specialized per N today (SOURCE `scripts/generate_c1.py`), so R and S must be compile-time constants per configuration. The generator has to emit (N, S, G) variants.
- **Atomics:** the one-launch reduction depends on cuTile atomics memory-ordering semantics. Two launches are the safe default.
- **Thermal:** split-K raises SM utilization, so it will heat the GPU faster than today's kernels. Expect more THROTTLED tags at 32K; the policy is unchanged.
