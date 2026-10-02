# cuTile 0.2 tensor-core mapping: latent score GEMM probe

Question: does cuTile 0.2 lower the latent score GEMM `q' [16×32] × Zᵀ [32×T]` (T = 64…512) to tensor-core MMA on sm_89?

Scope: this probe only answers the question. No kernel work beyond it (in particular no E3 implementation).

## Result (MEASURED, compile-only, 2026-10-02)

**Yes.** With FP16 operands and an FP32 accumulator, `cutile::core::mma` lowers to **`HMMA.16816.F32`** on sm_89 for every T tested, in both operand orientations. (Caveat from P3: this needs an `mma` output tile of at least 512 elements with 4 warps; a 16×16 score tile falls back to CUDA cores.)

| kernel | T=64 | T=128 | T=256 | T=512 |
|---|---:|---:|---:|---:|
| `scores_qz_f16`: q′[16,32] × Zᵀ[32,T] → [16,T] | 4 HMMA / 384 instr | 8 / 584 | 16 / 968 | 32 / 1352 |
| `scores_zq_f16`: Z[T,32] × q′ᵀ[32,16] → [T,16] | 4 / 320 | 8 / 456 | 16 / 584 | 32 / 1072 |
| `scores_reduce_f32` (current kernel style, control) | 0 HMMA; FMUL + FADD | 0 | 0 | 0 |

Counts are per-warp SASS instruction counts from `cuobjdump -sass`.

**Sanity check.** A 16×T×32 GEMM needs (16/16)·(T/8)·(32/16) = **T/4** `HMMA.16816` (m16n8k16) instructions per CTA. The cubin declares `EIATTR_REQNTID = 128` threads (4 warps), so the expected count is T/16 per warp. The measured count is exactly **T/16** (4 at T=64, 32 at T=512). **The whole GEMM is issued on the tensor cores**, with no residual FFMA.

**Existing kernels (control).** MEASURED: across all 48 distinct cuTile kernel functions in this repository's cached JIT cubins (`/tmp/*.cubin`, sm_89), **0 contain HMMA or IMMA**. Today's score and context kernels run entirely on CUDA cores (FP32 FMUL/FADD), which matches the 0 % tensor-pipe utilization in the P05 ncu profiles. Survey: `reports/diagnosis_20261002/mma_probe/existing_kernels_sass_survey.txt`.

**P1: vectorized loads (MEASURED, compile-only, CPU; `reports/diagnosis_20261002/mma_probe_p1/`).**

Without hints, the compile-only probe loads FP16 data with scalar 16-bit global loads (`LDG.E.U16`). The runtime JIT never sees that state: at launch it derives divisibility from the real pointers, shapes and strides (`cutile-compiler-0.2.0/src/specialization.rs`, `compute_spec`), and `KernelCompiler` without hints assumes divisor 1. Either of the two hints below turns all FP16 global loads into 128-bit (or 64-bit) vector loads. HMMA is unchanged.

| kernel (T=512) | hint mode | global loads per warp | HMMA | SASS instr |
|---|---|---|---:|---:|
| `scores_zq_f16` | none | 132 × `LDG.E.U16` | 32 | 1072 |
| `scores_zq_f16` | `CompileOptions::new().max_divisibility(16)` | **16 × `LDG.E.128` + 1 × `LDG.E.64`** | 32 | 784 |
| `scores_zq_f16` | `spec_args` from `compute_spec(aligned_ptr, shape, strides, bytes)` | 16 × `LDG.E.128` + 1 × `LDG.E.64` | 32 | 784 |
| `scores_zq_f16` | both | 16 × `LDG.E.128` + 1 × `LDG.E.64` | 32 | 784 |
| `scores_qz_f16` | none → any hint | 132 × `LDG.E.U16` → 16 × `LDG.E.128` + 1 × `LDG.E.64` | 32 | 1352 → 824 |
| `scores_reduce_f32` (control) | none → any hint | 129 × `LDG.E` (32-bit) → 32 × `LDG.E.128` + 1 × `LDG.E` | 0 | 976 → 1800 |

- **Byte check:** 16 × `LDG.E.128` per warp × 32 lanes × 16 B × 4 warps = 32 KiB, exactly the 512×32 FP16 Z tile. 1 × `LDG.E.64` × 32 × 8 B × 4 = 1 KiB, exactly the 16×32 FP16 q′. The same holds for T = 64, 128 and 256 (2, 4 and 8 × `LDG.E.128`). Every probe kernel and T is in `mma_probe_summary.json`.
- **Which change gets there:** `KernelCompiler::options(CompileOptions::new().max_divisibility(16))` alone is sufficient. So is `KernelCompiler::spec_args(...)` with `SpecializationBits` computed as the runtime does for 256-byte-aligned allocations, where `DivHint::from_ptr` clamps to 16. The SASS for `maxdiv16` and `spec` is identical, ignoring addresses. In kernel source, the equivalent is the entry attribute `optimization_hints = (sm_89 = (max_divisibility = 16,),)`, the form used in upstream `cutile-0.2.0/tests/flash_attention_compile.rs`. That form was **not tested here**.
- **What it means for E3:** the scalar-load risk was an artifact of compiling without the runtime's alignment information. It applies to contiguous, aligned tiles. **Not covered:** paged gathers, where the row offset is `physical_block × 16 × 32 × 2 B` computed from a loaded table entry. Divisibility then depends on how cuTile propagates the table value. E3 must re-check SASS on the paged kernel itself.

**P2: real R-TABLE paged-gather score/context kernels (MEASURED, compile-only, CPU; `reports/diagnosis_20261002/paged_gather_p2/`).**

A compile-only probe of the 1024-token latent R-TABLE kernels shows the alignment hint also vectorizes the table-indirected FP16 latent loads. This is not an E3 implementation: it compiles the existing serial score/context kernels and counts SASS only.

| kernel | hint mode | table-indirected FP16 latent loads | all global loads per warp | HMMA | SASS instr | vs no hint |
|---|---|---|---|---:|---:|---:|
| `model_small_scores_fp16_storage_rtable_1024` | none | 4 × `LDG.E.U16` | 19 × `LDG.E` + 4 × `LDG.E.U16` | 0 | 544 | baseline |
| same | builder `CompileOptions::max_divisibility(16)` | 4 × `LDG.E.128` | 3 × `LDG.E` + 1 × `LDG.E.64` + 4 × `LDG.E.128` | 0 | 688 | +144 |
| same | source `#[cutile::entry(optimization_hints = (sm_89 = (max_divisibility = 16,),))]` | 4 × `LDG.E.128` | 3 × `LDG.E` + 1 × `LDG.E.64` + 4 × `LDG.E.128` | 0 | 688 | +144 |
| `model_small_context_fp16_storage_rtable_1024` | none | 4 × `LDG.E.U16` | 18 × `LDG.E` + 4 × `LDG.E.U16` | 0 | 1256 | baseline |
| same | builder `CompileOptions::max_divisibility(16)` | 4 × `LDG.E.128` | 50 × `LDG.E` + 25 × `LDG.E.64` + 4 × `LDG.E.128` | 0 | 3088 | +1832 |
| same | source `#[cutile::entry(optimization_hints = (sm_89 = (max_divisibility = 16,),))]` | 4 × `LDG.E.128` | 50 × `LDG.E` + 25 × `LDG.E.64` + 4 × `LDG.E.128` | 0 | 3088 | +1832 |

- The builder form and in-source form produce the same load-width counts and instruction counts.
- HMMA remains 0 because these are the current reduce-style kernels, not an `mma` rewrite.
- The hint removes scalar 16-bit global loads for the table-indirected latent tile. The instruction count increases in the current reduce-style kernels, so this result should be used only as a load-width proof, not as a performance prediction.

**Runtime assert design for safe alignment hints.** Before launching any hinted paged-gather kernel, assert that the base pointer and row pitch make every table-indirected row start at a 16-byte boundary:

```text
require base_ptr(latent_fp16) % 16 == 0
require (latent_dim * sizeof(f16)) % 16 == 0        # 32 * 2 = 64 today
require block_size * latent_dim * sizeof(f16) % 16 == 0  # 16 * 32 * 2 = 1024 today
require all physical_block values are in bounds
```

For FP32 score/probability/projection tensors that are also compiled with the same max-divisibility ceiling, apply the analogous base-pointer and stride-byte checks. If any check fails, dispatch an unhinted fallback or fail before launch; do not use `max_divisibility(16)` on unaligned storage.

**P3: resources, the +1832 growth, a gathered-tile MMA kernel, and the alignment assert (MEASURED, compile-only, CPU; `reports/diagnosis_20261002/p3_probe/`).**

Probe: `crates/plkv-kernels/examples/p3_probe.rs`, with shared helpers in `examples/common/mod.rs`. It compiles with `KernelCompiler` for sm_89, runs `tileiras`, and reads `cuobjdump -sass`, `--dump-resource-usage` and `-elf` (`EIATTR_REQNTID`). No GPU or CUDA context is used.

How the numbers are computed:
- **Theoretical occupancy** uses the sm_89 limits: 48 warps/SM; 64 K registers/SM, allocated per warp in 256-register units with registers per thread rounded up to 8; 24 CTAs/SM; 100 KiB shared memory/SM with 1 KiB reserved per CTA. This reproduces P05 ncu (113 regs × 128 threads → 33.3 %).
- **"Static load B/CTA"** is Σ(per-warp load instructions × width × 32 lanes) × warps, assuming all lanes are active. It is split into loads outside loops and loads per iteration of the innermost loop, where a loop is a backward branch.

### 1. Production kernels, no hint vs `max_divisibility(16)`

N = 1024. N = 8192 is identical except for softmax.

| kernel | hint | SASS instr | REG | LOCAL (spill) | SHARED B | theor. occ (limiter) | global loads / warp | static load B/CTA: outside loop / per iteration |
|---|---|---:|---:|---:|---:|---|---|---|
| A1 score | none | 152 | 30 | 0 | 256 | 100 % | 3×LDG.E, 8×LDG.E.U16 | 3584 / – |
| A1 score | maxdiv16 | 136 | 24 | 0 | 256 | 100 % | 3×LDG.E, 1×LDG.E.128 | 3584 / – |
| B1 score | none → maxdiv16 | 544 → 688 | 48 → 48 | 0 | 2560 → 2304 | 83.3 % (regs) | 19×LDG.E + 4×U16 → 3×LDG.E + 4×.128 + 1×.64 | 10752 / – (both) |
| C1 projection | none → maxdiv16 | 160 → 208 | 40 → 29 | 0 | 512 → 256 | 100 % | 17×LDG.E → 1×LDG.E + 4×.128 | 8704 / – (both) |
| C1 score | none → maxdiv16 | 120 → 120 | 24 → 20 | 0 | 256 → 0 | 100 % | 3×LDG.E + 4×U16 → 3×LDG.E + 1×.64 | 2560 / – (both) |
| softmax N=8192 | none → maxdiv16 | 3040 → 2448 | 116 → 99 | 0 | 16 | 33.3 % (regs) | 65×LDG.E → 1×LDG.E + 16×.128 | 33280 / – (both) |
| **A1 context** | none | 232 | 38 | 0 | 0 | 100 % | 2×LDG.E, 8×LDG.E.U16 | 0 / 3072 |
| **A1 context** | maxdiv16 | **2616 (+2384)** | **168** | 0 | 3072 | **25 % (regs)** | 40×LDG.E, 20×LDG.E.128 | 58368 / 3072 |
| **latent context** | none | 1256 | 80 | 0 | 16896 | 41.7 % (smem) | 18×LDG.E, 4×LDG.E.U16 | 8192 / 2048 |
| **latent context** | maxdiv16 | **3088 (+1832)** | **126** | 0 | 10752 | **33.3 % (regs)** | 50×LDG.E, 4×LDG.E.128, 25×LDG.E.64 | 57344 / 2048 |

**Byte reconciliation:**
- **Loop-free kernels:** static bytes per CTA are identical before and after the hint. The hint changes load width, not bytes.
- **Context kernels:** bytes per loop iteration are unchanged (3072 for A1; 2048 for latent).
- **What the extra outside-loop bytes are:** A1 has 58 368 = 19 × 3072, and latent has 57 344 = 8192 (V projection, also loaded once without the hint) + 24 × 2048. These are whole iterations' worth of loads hoisted into a prologue.
- **Dynamic totals:** if the steady-state loop runs correspondingly fewer trips, dynamic bytes are unchanged. That is inferred from static code (HYPOTHESIS until ncu `dram__bytes`/`lts__t_bytes` can be read).

**Why latent context grows by +1832 instructions:** SASS regions, maxdiv16, N=1024, with the loop at `0x2250–0x2ba0`.

| region | instr | LDG | BAR | SHFL | FADD | FMUL | LDS / STS | BRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prologue (before loop) | 549 | 63 | 31 | 0 | 0 | 0 | 15 / 16 | 32 (guarded forward branches) |
| loop body | 150 (unhinted body ≈ 91) | 3 | 4 | 10 | 14 | 4 | 6 / 6 | 3 |
| epilogue (after loop) | 2389 | 13 | 82 | 432 | 720 | 352 | 195 / 149 | 16 |

- **Not spills.** `LOCAL` = 0 and `STACK` = 0, with no LDL/STL, in every kernel.
- **Not classic unrolling.** The steady-state loop body stays small (150 instructions) and its per-iteration bytes don't change.
- **Address math is a minor part.** Versus the unhinted build: LEA +66, SHF +93, ISETP +82, IADD3 −41.
- **The growth is software pipelining.** A guarded prologue issues loads for about 20+ future iterations (63 LDG behind 32 predicated branches, staged through shared memory with BAR). The epilogue drains the remaining iterations' work: the per-iteration cross-thread reductions (SHFL/FADD/LDS/STS/BAR) plus the final V projection.
- **Cost of the in-flight stages:** registers 80 → 126 (latent) and 38 → 168 (A1), i.e. theoretical occupancy 41.7 → 33.3 % and 100 → 25 %. At B = 1 the context grid is 16 CTAs on 24 SMs, so occupancy is not the binding limit today.
- A1 context shows the same structure (+2384).

**This is the code that actually runs.** The runtime-JIT cubins that earlier GPU runs left in `/tmp` already match the hinted builds, because the runtime derives divisibility from the real pointers:

| runtime-JIT kernel | REG | SHARED | instr | loads / warp |
|---|---:|---:|---:|---|
| latent context, 1K and 8K | 128 | 10752 | 3152 | 50×LDG.E, 4×LDG.E.128, 25×LDG.E.64 |
| A1 context | 181 | 3072 | 2656 | 40×LDG.E, 20×LDG.E.128 |

So E0, E0b and p15b measured the vectorized, software-pipelined variants. The "none" compile is a probe artifact, not what executed.

### 2. Gathered FP16 latent tile → `mma` (minimal kernel; not E3)

`latent_gather_mma<R>`: each CTA (128 threads) walks R logical blocks. Per block it does the following:
- reads the table entry;
- gathers the physical 16×32 FP16 tile (`latent.load_tile([16,32], [physical, 0])`);
- computes S = q′·Zᵀ with `mma`;
- computes P = FP16(exp S);
- accumulates `acc[16×32] += P·Z` with `mma` into an FP32 accumulator.

Split variants isolate each GEMM. N = 8192, results with `maxdiv16`:

| kernel | R | instr | REG | LOCAL | SHARED B | theor. occ | HMMA (static) | FMUL+FADD | gather / global loads per warp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full: scores + PV | 4 | 1128 | 106 | 0 | 10752 | 33.3 % | 4 | 536 | 4×LDGSTS.E.64 (tiles), 4×LDG.E (table), 1×LDG.E.64 (q′) |
| full: scores + PV | 16 | 3960 | 162 | 0 | 35328 | 16.7 % | 16 | 2144 | 16×LDGSTS.E.64, 16×LDG.E, 1×LDG.E.64 |
| PV only (acc += P·Z, output 16×32) | 4 | 232 | 32 | 0 | 9728 | 75 % | 4 | 0 | 4×LDGSTS.E.64 + 4×LDGSTS.E (P), 4×LDG.E |
| scores only, q′·`z.transpose()` (output 16×16) | 4 | 896 | 102 | 0 | 10240 | 33.3 % | **0** | 520 | 4×LDGSTS.E.64, 4×LDG.E, 1×LDG.E.64 |
| scores only, Z·q′ᵀ, no transpose (output 16×16) | 4 | 1024 | 79 | 0 | 7168 | 50 % | **0** | 520 | same |
| scores, **two gathered blocks** cat → 32 tokens, Z·q′ᵀ (output 32×16) | 4 | 424 | 56 | 0 | 11264 | 66.7 % | **8** | 0 | 9×LDG.E.64, 8×LDG.E |
| same, q′·`z.transpose()` (output 16×32) | 4 | 376 | 47 | 0 | 11264 | 66.7 % | **8** | 0 | 9×LDG.E.64, 8×LDG.E |

Findings:
- **Gather load width.** With the hint, each block's 1 KiB tile is fetched by one `LDGSTS.E.64` per warp: an asynchronous global→shared copy, 8 B × 32 lanes × 4 warps = 1 KiB, the whole tile. The table-indirected address keeps its alignment.
  - Why it's 64-bit, not 128-bit: a 1 KiB tile over 128 threads is exactly 8 B per thread (HYPOTHESIS: wider per-thread work, e.g. fewer warps or larger tiles, would allow 128-bit).
  - The two-block variant uses `LDG.E.64` directly (register path).
  - Without the hint, every variant falls back to scalar `LDG.E.U16` (5 per block per warp).
- **Staging.** The gathered tiles go through shared memory (`LDGSTS` → `LDSM.16.M88.4` / `LDSM.16.MT88.2`, i.e. ldmatrix, including the transposed form), with `BAR` synchronization. There is **no SHFL** in any gather variant. A small scalar path: 8 × `STS.U16` in the full kernel (FP16 P staging after `F2FP.F16.F32.PACK_AB`).
- **HMMA rule (new, MEASURED).** cuTile/tileiras lowers `mma` to HMMA only when the output tile can give every warp at least one m16n8 fragment. With 4 warps that means at least 512 output elements:
  - The PV GEMM (16×32 = 512) is fully on tensor cores: 4 HMMA per CTA per block = (16/16)·(32/8)·(16/16), as required.
  - The per-block score GEMM (16 heads × 16 tokens = 256) falls back to CUDA cores (FMUL+FADD, exactly 16·16·32/128 = 64 + 64 per thread per block), in **both** orientations. The register transpose is **not** the cause.
  - Gathering **two** blocks per step (32 tokens, 512 outputs) puts the score GEMM fully on HMMA in both orientations, including the register transpose: 8 HMMA per warp at R=4 = 2 per warp per block pair = the full (32/16)·(16/8)·(32/16) = 8 per CTA.
  - This revises P1's "both orientations lower to HMMA". That holds when the output tile is ≥ 512 elements; every P1 shape had ≥ 1024.
- **Registers and spills.** No spills anywhere (LOCAL = 0). Registers grow with R because R is a compile-time constant and the block loop is unrolled: 106 at R=4, 162 at R=16. The CUDA-core score fallback is the main register cost; the PV-only kernel needs 32 registers.

**Implications for E3** (design inputs only; E3 is not implemented):
1. Process at least 2 blocks (32 tokens) per score `mma` step with 16 heads and 4 warps, or reduce warps per CTA.
2. Expect cuTile to software-pipeline the block loop when divisibility is known, so budget registers (REG 126–181 in today's kernels).
3. Paged gathers keep vector width through the block table and arrive via `cp.async` (LDGSTS) into shared memory.

### 3. Host-side alignment assert

- **Where:** `plkv_kernels::alignment` (CPU, not behind the GPU feature).
- **What it checks:** `check_layout` / `check_layouts` require that the base pointer is 16-byte aligned, the innermost stride is 1, and every outer stride × element size is a multiple of 16 bytes. For the latent cache this implies every table-indirected block row (`physical × 16 × 32 × 2 B`) is 16-byte aligned. It reports every violation, not just the first.
- **Tests:** 5 CPU unit tests. They accept an aligned [N, 32] FP16 layout, **reject a real heap allocation viewed at +2 bytes**, and reject a 72-byte row stride, a non-unit inner stride, and multiple violations reported together.
- **In the probes:** `assert_hint_layouts` runs before every `max_divisibility(16)` compile. `p3_probe` starts with a self-test that a base+2 layout is rejected (`ALIGNMENT_ASSERT_SELF_TEST_OK`).
- **Launch-time use:** pass `Tensor::device_pointer().cu_deviceptr()` and `Tensor::strides()` (both public in cuTile 0.2). Not covered: table-value bounds (`physical < num_blocks`), which need a separate check.

Reproduce (CPU only):

```bash
cargo build --release -p plkv-kernels --features gpu-cutile --example p3_probe
target/release/examples/p3_probe --target sm_89 --output-dir /tmp/plkv_p3_probe
cargo test -p plkv-kernels alignment
```

The GPU was not used: no compute process ran, and the only GPU activity during the probe came from the desktop session.

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
| P1 | Recompile with divisibility hints | SASS: `LDG.E.128` vs `LDG.E.U16` | **DONE.** `max_divisibility(16)` or runtime-style `spec_args` gives 128-bit loads (see P1 above). Paged gathers remain to be checked in E3. |
| P2 | Compile real R-TABLE paged-gather score/context kernels with builder and in-source alignment hints | SASS load widths for table-indirected latent loads, HMMA count, instruction count | **DONE.** Hints change the table-indirected latent loads from `LDG.E.U16` to `LDG.E.128`; builder and source forms match. Current reduce-style kernels still have 0 HMMA. |
| P3 | Resources and spills for production kernels (none vs maxdiv16); gathered-tile `mma` kernel; host-side alignment assert | `--dump-resource-usage`, SASS regions, HMMA/LDGSTS counts, CPU tests | **DONE.** +1832 is software pipelining, not spills; gathers stay vectorized (`LDGSTS.E.64`); score `mma` needs ≥512 outputs (2 blocks); assert in `plkv_kernels::alignment` (see P3 above). |
| P4 | Run the probe in `run_e0b_portable.sh` on an L4 | same SASS checks (sm_89) | Confirms the result is toolchain-only, not machine-specific |
| P5 | Timed microbenchmark: one launch per T, CUDA events, interleaved MMA vs reduce variants, ≥5 processes, clock-logged harness, on a stable-clock host | latency, plus `sm__pipe_tensor_cycles_active` via focused ncu once counters are available | Only after P1. On this laptop it needs your approval and a stable clock, which is currently not achievable. |

Tile-shape caveat: T is a compile-time constant. Production E3 tiles would use T = 16·R (R blocks per CTA) with paging, so this probe uses contiguous Z and no block table. It answers the lowering question only, not paged-load efficiency.
