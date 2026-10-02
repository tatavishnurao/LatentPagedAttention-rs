# E3 split-K: correctness and SASS results (2026-10-02)

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
3. **Not done here:** timing; a stable-clock host; batch > 1 / multi-layer (E4); pipeline-depth control (no cuTile knob, see design §0(c)); and a register-reducing variant of A3.

## Reproduce

```bash
export PATH=$HOME/.cargo/bin:$PATH CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3 \
  LIBCLANG_PATH=/usr/lib/llvm-18/lib CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.3/bin/tileiras
bash scripts/run_gpu_e3_splitk_validation.sh            # GPU, correctness only
cargo run --release -p plkv-kernels --features gpu-cutile --example e3_sass_probe -- \
  --output-dir /tmp/plkv_e3_sass                         # CPU only
cargo test -p plkv-kernels block_table alignment         # CPU host-validation tests
```

Thermal policy during the GPU runs: 68–72 °C, below the 91 °C abort. No THROTTLED tagging applies, because no timing was taken.
