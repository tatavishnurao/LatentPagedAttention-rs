# cuTile 0.2 tensor-core mapping: latent score GEMM probe

Question: does cuTile 0.2 lower the latent score GEMM `q' [16×32] × Zᵀ [32×T]` (T = 64…512) to tensor-core MMA on sm_89?

Scope: this probe only answers the question. No kernel work beyond it (in particular no E3 implementation).

## Result (MEASURED, compile-only, 2026-10-02)

**Yes.** With FP16 operands and an FP32 accumulator, `cutile::core::mma` lowers to **`HMMA.16816.F32`** on sm_89 for every T tested, in both operand orientations.

| kernel | T=64 | T=128 | T=256 | T=512 |
|---|---:|---:|---:|---:|
| `scores_qz_f16`: q′[16,32] × Zᵀ[32,T] → [16,T] | 4 HMMA / 384 instr | 8 / 584 | 16 / 968 | 32 / 1352 |
| `scores_zq_f16`: Z[T,32] × q′ᵀ[32,16] → [T,16] | 4 / 320 | 8 / 456 | 16 / 584 | 32 / 1072 |
| `scores_reduce_f32` (current kernel style, control) | 0 HMMA; FMUL + FADD | 0 | 0 | 0 |

Counts are per-warp SASS instruction counts from `cuobjdump -sass`.

**Sanity check.** A 16×T×32 GEMM needs (16/16)·(T/8)·(32/16) = **T/4** `HMMA.16816` (m16n8k16) instructions per CTA. The cubin declares `EIATTR_REQNTID = 128` threads (4 warps), so the expected count is T/16 per warp. The measured count is exactly **T/16** (4 at T=64, 32 at T=512). **The whole GEMM is issued on the tensor cores**, with no residual FFMA.

**Existing kernels (control).** MEASURED: across all 48 distinct cuTile kernel functions in this repository's cached JIT cubins (`/tmp/*.cubin`, sm_89), **0 contain HMMA or IMMA**. Today's score and context kernels run entirely on CUDA cores (FP32 FMUL/FADD), which matches the 0 % tensor-pipe utilization in the P05 ncu profiles. Survey: `reports/diagnosis_20261002/mma_probe/existing_kernels_sass_survey.txt`.

**New issue (MEASURED for the probe's compile configuration only).** The probe kernels load FP16 data with scalar 16-bit global loads: `LDG.E.U16`, 132 per warp at T=512, staged through `LDS.128`. The repository's latent context kernel, compiled by the normal JIT path, uses `LDG.E`, `LDG.E.64` and `LDG.E.128`.
- HYPOTHESIS: the compile-only path received no divisibility or alignment hints (`KernelCompiler::spec_args` / `scalar_hints` / `max_divisibility` were not set), so cuTile could not prove alignment and fell back to scalar loads.
- E3 must check vector-load emission as well as HMMA. A tensor-core kernel fed by 16-bit scalar loads would be load-issue-bound, the same `lg_throttle` failure seen in P05.

## Method

1. `crates/plkv-kernels/examples/mma_probe.rs` defines the three kernels. Each is generic over a const `T`.
2. For each (kernel, T), `cutile::compile_api::KernelCompiler::new(...).generics([T]).strides(...).target("sm_89").compile()` produces Tile IR and bytecode **on the CPU, without a GPU or CUDA context**. The upstream reference is `cutile-0.2.0/tests/flash_attention_compile.rs`.
3. `tileiras --gpu-name sm_89 --opt-level 3` compiles the bytecode to a cubin, with the same flags the runtime JIT uses (`cutile-compiler-0.2.0/src/cuda_tile_runtime_utils.rs:76-84`).
4. `cuobjdump -sass` provides the per-function opcode histogram, and the IR is checked for the `mmaf` op.
5. Artifacts are in `reports/diagnosis_20261002/mma_probe/` (`.mlir`, `.cubin`, `mma_probe_summary.json`).

Reproduce (CPU only):

```bash
export PATH=$HOME/.cargo/bin:$PATH CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3 \
  LIBCLANG_PATH=/usr/lib/llvm-18/lib CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.3/bin/tileiras
cargo build --release -p plkv-kernels --features gpu-cutile --example mma_probe
target/release/examples/mma_probe --target sm_89 --output-dir /tmp/plkv_mma_probe
```

The GPU was idle throughout the probe (210 MHz before and after, no compute process).

## Follow-up plan (not run; needs approval; no kernel work beyond the probe)

| step | what | measurement | decision |
|---|---|---|---|
| P1 | Recompile with divisibility hints (`spec_args` / `max_divisibility = 16`) | SASS: `LDG.E.128` vs `LDG.E.U16` count | If vector loads appear, record the required hint set for E3. If not, it is an E3 risk to raise with cuTile. |
| P2 | Same probe with FP32 operands into `mma` | SASS: HMMA (TF32) vs FFMA | Decides whether E3 can keep FP32 q′ or must cast it to FP16 |
| P3 | Run the probe in `run_e0b_portable.sh` on an L4 | same SASS checks (sm_89) | Confirms the result is toolchain-only, not machine-specific |
| P4 | Timed microbenchmark: one launch per T, CUDA events, interleaved MMA vs reduce variants, ≥5 processes, clock-logged harness, on a stable-clock host | latency, plus `sm__pipe_tensor_cycles_active` via focused ncu once counters are available | Only after P1. On this laptop it needs your approval and a stable clock, which is currently not achievable. |

Tile-shape caveat: T is a compile-time constant. Production E3 tiles would use T = 16·R (R blocks per CTA) with paging, so this probe uses contiguous Z and no block table. It answers the lowering question only, not paged-load efficiency.
