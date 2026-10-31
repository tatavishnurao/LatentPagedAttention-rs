# External full-KV decode baseline: FlashInfer (proposal only)

Status: **proposal**. Nothing has been installed or run, and no number in this file is measured. Labels: SOURCE (read in this repo's code), HYPOTHESIS (not checked; must be confirmed on the VM before use).

## Why

Padded A3 is this repo's own full-KV GQA kernel. Reviewers will ask whether it is a reasonable full-KV baseline or a strawman. One external, widely used paged decode kernel on the same shapes, inputs and timing protocol answers that question. It does **not** replace padded A3 as the equal-treatment baseline for the C3 ratio (see "What the comparison can and cannot say").

## Shapes and inputs (identical to E4)

| Parameter | Value | Source |
|---|---|---|
| Hq / Hkv / head dim | 16 / 4 / 64 (GQA group 4) | SOURCE `e4_config.rs` |
| Cache dtype | FP16 K and V; FP32 accumulation in the reference | SOURCE |
| Page size | 16 tokens (= repo block size) | SOURCE |
| Batch B | 1, 8, 32 | SOURCE campaign defaults |
| Context N | 1024, 8192, 32768 (same grid as E4 timing) | SOURCE `--e4-seqs` |
| Active lengths | b = 0: N; b > 0: `1 + ((b·373) mod 1024)·N/1024` | SOURCE `e4_active_len` |
| Page tables | the non-identity pool table `(lb·1021 + 11) mod (B·N/16)`, sliced per sequence | SOURCE `e4_layout` |
| Layers | `max(4, ceil(65536 / (B·N)))` separate pools; a 64 MiB L2 flush before each layer | SOURCE |
| Softmax scale | 1/√64 | SOURCE `Dims::scale` |

**Same bytes in, not just the same shapes.** Python does not regenerate the inputs. A new GPU-free Rust example (`e4_dump`, proposed) writes, for each layer and (B, N):
- q per sequence;
- the FP16 K and V pools in the harness's head-major physical layout `[page][kv_head][16][64]`;
- the page tables and active lengths;
- the CPU FP32-accumulation reference output (`paged_full_kv_gqa_decode_fp16_storage_runtime_f32_accum`, the reference padded A3 is checked against).

All of it goes to one `.npz` or raw binary with a JSON header. The Python script only loads it. At B = 32, N = 32K, one layer is 1 GiB of FP16 K+V; dumps stay in the results directory and never enter the tarball (same rule as `oracle_inputs_outputs.json`).

## FlashInfer mapping (HYPOTHESIS: confirm against the installed version)

- API: `flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="HND")`. `plan(indptr, indices, last_page_len, num_qo_heads=16, num_kv_heads=4, head_dim=64, page_size=16, pos_encoding_mode="NONE", q_data_type=float16, kv_data_type=float16)` runs once per (B, N), outside timing. `run(q, (k_cache, v_cache))` then runs per layer. Older releases name these `begin_forward` / `forward`. Record which one is used.
- Layout: the harness physical layout `[page][kv_head][token][d]` is FlashInfer's `HND` `[num_pages, num_kv_heads, page_size, head_dim]`. No transpose is needed; if one were, it would sit outside timing.
- `indptr` / `indices`: per sequence, the first `ceil(active/16)` entries of its table slice. `last_page_len = active - 16·(ceil(active/16) - 1)`.
- sm_89 (L4) and head dim 64 with group size 4 are believed to be supported. If the installed build rejects the shape or the GPU, **stop this phase and report**. Do not substitute another library without asking.

## Correctness first (per process, before any timing)

For every (layer, sequence): max |FlashInfer − CPU FP32 reference| ≤ 5e-3, the same `ATOL` as E4. A failure stops the process with a non-zero exit; tolerances are not loosened. Record the worst error in `correctness.json`, in the same schema as the harness.

## Timing protocol (same as `e4-timing`)

- One `run()` per layer covering all B sequences (FlashInfer batches them in one launch; the repo harness launches per sequence). Each launch interval is bracketed by CUDA events on one stream; the step total is the sum of the per-layer intervals for that iteration.
- The 64 MiB L2 flush (`torch.empty(16 Mi, int32).zero_()`) runs before each layer, outside the interval.
- 10 warm-up iterations, 2 s clock warm-up, 50 timed iterations. Defaults match `timing_args`.
- Per-sample NVML SM/mem clock and clock-event reasons, read before and after each interval (`nvidia-ml-py`). External `nvidia-smi -lms 100` telemetry comes from `run_logged`.
- ≥ 5 cool-start processes per (B, N), 91 °C abort, and the same THROTTLED rule: SM clock > 5 % below modal, or thermal/HW-slowdown bits 0x8/0x20/0x40/0x80. `sw_power_cap` (0x4) alone is not THROTTLED.
- Output uses the harness schema (`samples.jsonl` with `variant: "flashinfer"`, `component: "layer" | "step_total"`, `bytes_analytical`, `achieved_gbps`; `timing_meta.json`; `correctness.json`). The campaign then runs it as an extra kind (`ext_flashinfer`, config `B{b}_N{n}`).
- Analytic bytes for FlashInfer: full KV over the active tokens only, plus q, output and tables (no split padding). It is reported as its own column.

## What the comparison can and cannot say

- FlashInfer and padded A3 run in **different processes**, so there is no within-process paired ratio. Report per-config clean medians with process-level bootstrap CIs side by side. A cross-process ratio is labelled descriptive and is not a claim.
- FlashInfer reads only active tokens and batches all sequences in one launch. Padded A3 pads to the split boundary, spends 4× multiply-adds on zero rows and launches per sequence. If FlashInfer is much faster than padded A3, the C3/A3 ratio overstates the latent benefit against a production full-KV kernel, and the paper must say so.
- The primary result stays C3 accurate vs padded A3 (equal treatment). FlashInfer only checks whether that baseline is credible.

## Packages to install (ask first)

- `torch` with a CUDA build matching the VM driver: a multi-GB download. Record `torch.__version__` and `torch.version.cuda`.
- `flashinfer-python` (and, if needed, the JIT cache / cubin wheel for that torch/CUDA pair). Record the exact version and wheel URL.
- `nvidia-ml-py` for per-sample NVML.
- `numpy` is already in the project environment.

Install into a separate venv under the results directory or `~/venvs/flashinfer`, not into `attention99`. Write `pip freeze` into the results.

## Deliverables (after approval)

1. `crates/plkv-kernels/examples/e4_dump.rs` (GPU-free) and a CPU test that the dump matches `e4_layout`.
2. `scripts/baselines/flashinfer_gqa_decode.py`: load, correctness, timing, harness-schema output.
3. `e3e4_campaign.py --with-flashinfer`: off by default, so the default protocol is unchanged.
4. The FlashInfer columns in `docs/L4_RESULTS.md`, with versions.

Estimated extra GPU time: (HYPOTHESIS) 9 configs × 5 processes. That is similar to the E4 grid itself, plus install time.
