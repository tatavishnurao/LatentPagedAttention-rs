# E3 split-K: correctness and SASS results (2026-10-02, updated 2026-10-03)

**Scope.** These are correctness and code-generation results only. **No timing was taken and no performance claim is made** (the RTX 4060 Laptop cannot hold stable clocks; see `docs/DIAGNOSIS_2026-10-02.md`). The design is in [`E3_SPLITK_DESIGN.md`](E3_SPLITK_DESIGN.md).

Implementation:
- `crates/plkv-kernels/src/cutile/e3_splitk.rs`: kernels `c3_split`, `c3_reduce`, `a3_split`, `a3_reduce`.
- `crates/plkv-kernels/examples/e3_splitk_gpu.rs`: correctness harness, wrapped by `scripts/run_gpu_e3_splitk_validation.sh`.
- `crates/plkv-kernels/examples/e3_sass_probe.rs`: compile-only SASS report.
- `plkv_kernels::block_table`: host-side table validation.

Evidence: `reports/e3_splitk_20261002/`. The GPU log was produced from a working tree whose harness and kernels are identical to commits `188c051` + `64cf63e`. The log records `git_commit=188c051`, because the harness was still uncommitted at run time.

Labels: **MEASURED** (from the artifacts) or **SOURCE**.

## What was built

| | C3 (latent) | A3 (full KV) |
|---|---|---|
| split grid | (S) — one CTA covers all HQ = 16 heads of the shared latent tile | (S, Hkv) — one CTA covers the G = 4 heads of one KV head |
| projection | q′ = q·Pᴷ once per CTA, FP32, chunked over D | none |
| step | two gathered blocks (32 tokens), so the score GEMM has 16×32 = 512 outputs | two gathered blocks per K and V |
| GEMMs | `mma` FP16 operands (q′, P), FP32 accumulate | `mma` FP16 operands (q, P), FP32 accumulate |
| softmax | online, with a finite −1e30 mask sentinel | same |
| reduce | log-sum-exp merge over S, normalize, V projection once per head | log-sum-exp merge, normalize |
| launches per step | 2 | 2 |

Both block loops have runtime bounds (`steps_per_split`, `num_splits`), so they are not unrolled per configuration. Every launch is preceded by host-side validation: block-table bounds (`validate_block_table`) and 16-byte layouts (`alignment::check_layouts` on the real device pointers and strides).

## Correctness (MEASURED; `reports/e3_splitk_20261002/gpu_correctness.log`)

**Chain:**
1. Python oracle (`fixtures/reference`, FP32).
2. Rust CPU FP32 reference. The harness asserts it is within 1e-5 of the oracle on both tiny fixtures.
3. Rust CPU FP16-storage runtime reference.
4. GPU E3, required to be within atol 5e-3 of step 3.

| profile | tables | active lengths | splits | worst C3 error | worst A3 error | split spread |
|---|---|---|---|---:|---:|---:|
| tiny (Hq 4, Hkv 2, d 8, dl 8, block 2, N 8), fixture "balanced" | [2, 0, 3, 1] | 1, 3, 4, 7, 8 | 1, 2 | 3.45e-5 | 1.21e-4 | 5.3e-5 |
| tiny, fixture "stable_softmax" | [2, 0, 3, 1] | 1, 3, 4, 7, 8 | 1, 2 | 1.47e-5 | 0 | 6.2e-6 |
| model_small (Hq 16, Hkv 4, d 64, dl 32, block 16, N 1024) | (17l+11) mod 64 | 17, 129, 513, 1021, 1024 | 1, 4, 8 | 1.08e-3 | 1.56e-3 | 1.25e-3 |

- **Against the Python oracle directly (tiny, full length):** C3 is within 3.6e-5 and A3 within 8.6e-4. These figures include FP16 storage rounding, which the FP32 oracle does not have.
- **Partial final blocks:** tiny at 1, 3 and 7 tokens; model at 17, 129, 513 and 1021.
- **Negative control:** the same physical data read through an **identity** table at active length 129 differs from the non-identity reference by **1.79** (C3 and A3), against a tolerance of 5e-3. That proves the kernels gather through the table. At full length an identity table legitimately matches, because attention is invariant to token order.
- **Precision note:** model_small errors (~1e-3) are about 100× larger than C1's all-FP32 path (~1e-5). The cause is FP16 `mma` operands (q′ or q, and P), and the errors stay within the repository's 5e-3 bound. This is a numerics change relative to C1; it is not hidden.

## SASS (MEASURED, compile-only, sm_89, runtime-equivalent specialization; `reports/e3_splitk_20261002/sass/`)

| kernel (model_small) | HMMA (static / in main loop per warp) | global loads in main loop, per warp per step | REG | spills | SHARED B | theor. occ |
|---|---|---|---:|---:|---:|---:|
| **c3_split** | 20 / **4** (= 16 per CTA per step: 8 score + 8 PV, full tensor-core GEMMs) | 2×LDG.E (table) + **2×LDG.E.64** (two 1 KiB latent tiles) | **95** | 0 | 10240 | 41.7 % |
| c3_reduce | 0 (FP32 FMA merge + V projection) | — | 80 | 0 | 9216 | 50 % |
| **a3_split** | **0** / 0 | 2×LDG.E (table) + **4×LDG.E.128** (2 KiB K and V tiles) | **219** | 0 | 11264 | 16.7 % |
| a3_reduce | 0 | — | 90 | 0 | 5136 | 41.7 % |

Notes on the table:
- c3_split's static HMMA outside the main loop (16) is software-pipelining prologue and epilogue code, the same structure as P3.
- c3_split's first loop is the FP32 projection (CUDA cores, 4 iterations).
- On the tiny profile every kernel has 0 HMMA, as expected under the 512-output rule.

Findings:
1. **C3 meets the design constraints.** Both GEMMs run on tensor cores in the steady-state loop, there are no spills, and registers are 95, within the ≤ 128 target. Gathered tiles load as `LDG.E.64`: 1 KiB per 128 threads = 8 B per thread, the same rule as P3.
2. **A3 does not use tensor cores and exceeds the register target.** Its outputs are 4 heads × 32 tokens = 128 (scores) and 4 × 64 = 256 (PV), both below the 512-output rule, so both GEMMs fall back to CUDA cores (280 FMUL/FADD per loop iteration per warp), with **219 registers** (16.7 % theoretical occupancy).
   - Its loads are 128-bit, because 2 KiB K/V tiles are 16 B per thread.
   - This is the GQA asymmetry predicted in the design: with g = 4 query rows per KV head, a fair full-KV `mma` path needs head padding (4× MACs) or a different CTA mapping.
   - **The A3 vs C3 comparison is therefore not yet "same treatment" at the instruction level.** Any future timing comparison must state this.
3. **Not done here:** timing; a stable-clock host; batch > 1 / multi-layer (E4); and pipeline-depth control (no cuTile knob, see design §0(c)). The A3 tensor-core and register findings in item 2 are superseded by the padded A3 in the update below.

## Update (2026-10-03): equal treatment, numerics, bounded splits

Still **correctness and code generation only. No timing; no performance claim.** Evidence:
- `reports/e3_splitk_20261002/sass_a3_equal_treatment/` (A3 variants);
- `sass_c3_modes/` (C3 projection modes);
- `scale/` (the run that hit the stop condition, plus the CPU emulation);
- `scale_c3_modes/` (the passing re-run).

### A3 equal treatment (MEASURED, compile-only SASS, model_small)

| A3 variant | HMMA (static / main loop per warp per step) | REG | spills | K/V loads |
|---|---|---:|---:|---|
| `a3_split` (2-block steps, 4 query rows) | 0 / 0 | 219 | 0 | 128-bit |
| `a3_split8` (8-block steps: 4 × 128 = 512 score outputs, as requested) | 0 / 0 | 255 | **616 B stack (287 LDL / 160 STL)** | 128-bit |
| **`a3_split_pad`** (4 query rows zero-padded to 16) + `occupancy(4)` hint | **40 / 8** (= 32 per CTA: 16 score + 16 PV) | **128** | **0** | 128-bit |

- **The 512-output rule is necessary but not sufficient.** With M = 4 query rows (below 16), cuTile does not lower `mma` to HMMA at all, so the 8-block variant only adds register pressure.
- **Padding gives A3 C3's tensor-core shapes.** Scores become 16×32 and PV 16×64. The cost is 4× multiply-adds on zero rows.
- **The `occupancy(4)` entry hint caps registers at 128.** Without it the padded kernel uses 148 registers. The same hint on the unpadded A3 forces 320 B of spills, so it only works together with padding.
- The padded A3 is the A3 used for comparison from here on. It matches the Rust reference everywhere (worst 1.55e-3 on model_small).

### C3 projection modes (MEASURED SASS, model_small)

| mode | score operand | HMMA (static / main loop per warp per step) | REG | spills |
|---|---|---|---:|---:|
| fast | FP16 q′ | 20 / 4 (8 score + 8 PV per CTA) | 95 | 0 |
| accurate | q′ = hi + lo, two FP16 MMAs into one FP32 accumulator | 24 / 6 (16 score + 8 PV per CTA) | 95 | 0 |

The mode is a const generic chosen at launch. Accurate mode costs one extra score `mma` per step (+50 % tensor-core instructions per step) and the same loads. Its runtime cost is **not measured**.

### The 4.18e-3 heavy-tailed finding and its cause

- **What happened.** The first scale run (`scale/scale_suite.log`) used FP16 q′ and unbounded splits, and stopped on the 3e-3 gate. C3 on heavy-tailed inputs reached **4.18e-3 at N = 8192 at every split count**, and 3.27e-3 at 32K.
- **Cause, from a CPU numpy emulation of the same data** (`scale/emulate_c3_numerics.py`):

  | rounding applied | max error |
  |---|---:|
  | none (online softmax only) | 1.9e-5 |
  | FP16 P only | 1.43e-3 |
  | **FP16 q′ only** | **4.16e-3** |
  | both | 4.16e-3 |

  The emulation reproduces the GPU's 4.18e-3. The dominant cause is **FP16 rounding of the projected query q′ in the score GEMM**: scores reach |s| = 26, and a relative error of 2⁻¹¹ in q′ shifts the probabilities of the dominant outlier tokens.
- **What it was not.** It was not P accumulation, which was already FP32. So "FP32 accumulation of P" would not have fixed it.
- **A second, separate effect: error growth with N in single-split runs.** On uniform inputs, C3 and padded A3 grew to 2.9e-3 / 3.1e-3 at 32K when one CTA accumulated ~1000 tensor-core steps. The CUDA-core A3 did not grow (7.6e-4). That is consistent with `mma` accumulation over long sequential chains; the hardware rounding involved is a hypothesis, not emulated.

### Fixes

1. **Accurate projection mode** (above), for the q′ rounding.
2. **Bounded split plan** (`plkv_kernels::splitk_plan::plan_bounded`, asserted at every launch): at most **128 two-block steps (4096 tokens) per CTA**, i.e. `splits ≥ ceil(steps / 128)`. Split counts at block size 16:

   | N | steps | splits | steps per split |
   |---:|---:|---:|---:|
   | 1024 | 32 | 1 | 32 |
   | 4096 | 128 | 1 | 128 |
   | 8192 | 256 | 2 | 128 |
   | 32768 | 1024 | 8 | 128 |

### Validation chain (extended)

1. Python oracle (FP32).
2. Rust FP32 reference (≤ 1e-5 on the tiny fixtures).
3. Two references on identical inputs:
   - the **exact** reference: FP32 arithmetic on FP16 storage, as before;
   - a **matched-rounding** reference: the kernels' own FP16 roundings of q′ (hi, or hi + lo), q and P, the same split and two-block step structure, the same masking and log-sum-exp merge, with **FP64 accumulation**.

**Total error** = kernel vs exact. **Kernel-attributable error** = kernel vs matched-rounding, which isolates accumulation order and tensor-core arithmetic from the rounding choices.

### Scale results (MEASURED, GPU correctness only; `scale_c3_modes/scale_suite.log`)

Max |error| over 1024 context values. Bounded plans; uniform and heavy-tailed inputs on identical data for all three variants.

| inputs | N (splits) | C3 fast: total / kernel | C3 accurate: total / kernel | padded A3: total / kernel |
|---|---|---|---|---|
| uniform | 4096 (1) | 6.28e-4 / 3.52e-4 | 5.66e-4 / 4.07e-4 | 7.22e-4 / 3.82e-4 |
| uniform | 8192 (2) | 6.33e-4 / 3.59e-4 | 4.84e-4 / 3.93e-4 | 7.29e-4 / 3.94e-4 |
| uniform | 32768 (8) | 6.29e-4 / 3.66e-4 | 5.01e-4 / 3.88e-4 | 7.23e-4 / 3.97e-4 |
| heavy-tailed | 4096 (1) | 2.73e-3 / 8.6e-5 | **1.25e-3** / 8.6e-5 | 1.25e-3 / 2.65e-4 |
| heavy-tailed | 8192 (2) | **4.18e-3** / 7.8e-5 | 1.09e-3 / 7.9e-5 | 1.34e-3 / 1.66e-4 |
| heavy-tailed | 32768 (8) | 3.19e-3 / 6.9e-5 | 7.15e-4 / 6.6e-5 | 1.00e-3 / 1.40e-4 |

Mean errors are 1e-5 to 5e-4 (log). **Gate (C3 accurate total ≤ 3e-3): pass**, worst 1.25e-3.

### Numeric trade-offs

- **Total error is dominated by the rounding choices, not by kernel arithmetic.** Kernel-attributable error is ≤ 4.1e-4 for every variant and case. Rounding-only error (matched vs exact) tracks the total.
- **C3 fast is input-dependent.** It is fine on uniform data (≤ 6.3e-4) but exceeds 3e-3 on heavy-tailed data because of FP16 q′. **C3 accurate** removes that term (heavy-tailed ≤ 1.25e-3). Its remaining error comes mainly from FP16 P, the same source that bounds padded A3 (FP16 q and P).
- **Fairness of the comparison.** Accurate C3 and padded A3 now have comparable worst-case totals (1.25e-3 vs 1.34e-3) on identical inputs. Any future speed comparison should use **C3 accurate vs padded A3**, or else report C3 fast together with its input-dependent error.
- **Bounded splits remove the growth with N** (uniform errors are flat from 4K to 32K at 128 steps per CTA). The price is more partials (8 splits at 32K) and a slightly larger reduce, an effect that is analytical and small.
- **Not addressed:** an FP32 P path. It isn't needed for the gate, and padded A3 has the same FP16-P term. Each variant's remaining error is close to its own FP16 rounding floor.

## Reproduce

```bash
export PATH=$HOME/.cargo/bin:$PATH CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3 \
  LIBCLANG_PATH=/usr/lib/llvm-18/lib CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.3/bin/tileiras
bash scripts/run_gpu_e3_splitk_validation.sh            # GPU, correctness only
cargo run --release -p plkv-kernels --features gpu-cutile --example e3_splitk_gpu -- --suite scale  # GPU, correctness only
cargo run --release -p plkv-kernels --features gpu-cutile --example e3_sass_probe -- \
  --output-dir /tmp/plkv_e3_sass                         # CPU only
cargo test -p plkv-kernels block_table alignment         # CPU host-validation tests
```

Thermal policy during the GPU runs: 68–72 °C, below the 91 °C abort. No THROTTLED tagging applies, because no timing was taken.
