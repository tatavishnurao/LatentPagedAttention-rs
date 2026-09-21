# GPU Paged GQA Decode

## Purpose

This pass validates direct paged `f32` GQA decode on the RTX 4060 using
physical K/V cache blocks and a runtime non-identity block table. The goal is
correctness of attention over paged physical storage, not production serving
behavior.

## Fixed dimensions

- batch = 1
- q_heads = 4
- kv_heads = 2
- group_size = 2
- seq_len = 8
- head_dim = 8
- block_size = 2
- num_logical_blocks = 4
- num_physical_blocks = 4
- dtype = `f32`

## Physical K/V layout

Conceptual physical cache layout:

- `K`: `[num_physical_blocks, kv_heads, block_size, head_dim]`
- `V`: `[num_physical_blocks, kv_heads, block_size, head_dim]`

For this fixed case:

- `[4, 2, 2, 8]`

Flattened GPU layout:

- `[num_physical_blocks * kv_heads * block_size, head_dim]`
- `[16, 8]`

Each `[2, 8]` tile corresponds to one `(physical_block, kv_head)` pair holding
both token offsets for that block and KV head.

Physical tile index:

`physical_tile = physical_block * kv_heads + kv_head`

For this configuration:

`physical_tile = physical_block * 2 + kv_head`

## Logical-to-physical mapping

Committed runtime block table:

- `[2, 0, 3, 1]`

Logical token resolution:

- `logical_block = token_position / block_size`
- `block_offset = token_position % block_size`
- `physical_block = block_table[logical_block]`

The GPU validation includes an identity-table control to prove the runtime
block table is actually used by the attention kernels.

## Q-head to KV-head mapping

- q_head 0 -> kv_head 0
- q_head 1 -> kv_head 0
- q_head 2 -> kv_head 1
- q_head 3 -> kv_head 1

The mapping is resolved on the GPU as:

`kv_head = q_head / group_size`

## Three GPU pipeline stages

This implementation is intentionally not fused.

Stage 1:

- direct paged score computation from physical K tiles

Stage 2:

- stable row-wise softmax on GPU

Stage 3:

- direct paged probability-weighted V reduction from physical V tiles

Scores and probabilities remain on the GPU between stages. Host readback occurs
only after the complete pipeline for validation.

## Why no contiguous K/V materialization occurs

The GPU path does not reconstruct a full logical K/V tensor before attention.
Instead:

- score computation resolves `logical_block -> physical_block` at runtime and
  reads physical K tiles directly
- context computation resolves the same mapping at runtime and reads physical V
  tiles directly

The Python and Rust references may materialize logical K/V because they are
correctness oracles, not the implementation path under validation.

## Fixture generation

The fixture starts from the existing contiguous GQA cases:

- `balanced`
- `stable_softmax`

Logical token-major K/V are split into four logical blocks of two tokens, then
placed into physical storage according to `[2, 0, 3, 1]`. The fixture stores:

- physical token-major K/V
- physical GPU head-major K/V
- expected scores
- expected probabilities
- expected context

Expected outputs are generated from the Python paged GQA intermediate reference.

## Validation chain

Python paged reference
-> committed JSON fixture
-> Rust CPU paged reference
-> cuTile paged score kernel
-> cuTile stable softmax kernel
-> cuTile paged context kernel
-> full host-side comparison

## Negative identity-table control

The GPU example reruns the paged score stage with the identity table
`[0, 1, 2, 3]`. That control must change the output relative to the committed
non-identity mapping. The validation prints:

- `NON_IDENTITY_MAPPING_EFFECT_CONFIRMED=1`

This guards against accidentally ignoring the runtime block table while still
matching the fixture.

## Numerical errors

Observed on the validated RTX 4060 run:

- balanced score error: `1.1920929e-7`
- balanced probability error: `1.4901161e-8`
- balanced context error: `5.9604645e-8`
- balanced max probability row sum error: `5.9604645e-8`
- stable-softmax score error: `1.5258789e-5`
- stable-softmax probability error: `2.6999175e-21`
- stable-softmax context error: `0`
- stable-softmax max probability row sum error: `0`

These values passed the starting tolerances. No tolerance expansion was needed.

## cuTile limitations encountered

No unsafe code was required.

The main adaptation was keeping the implementation as a three-stage pipeline
instead of trying to fuse paged score, softmax, and paged context into one
kernel. That keeps runtime block-table resolution explicit while staying within
the patterns already validated in this repository.

Stage synchronization was performed between kernels through cuTile `sync()`
calls because the current API returns device tensors from each launch. The
intermediate tensors remained on the GPU and were not copied to the host
between stages.

## Commands

```bash
source scripts/cutile_env.sh
bash scripts/run_cutile_smoke.sh
bash scripts/run_gpu_paged_lookup.sh
bash scripts/run_gpu_paged_kv_write.sh
bash scripts/run_gpu_gqa_decode.sh
bash scripts/run_gpu_paged_gqa_decode.sh
```

## What this proves

- runtime non-identity block-table use in attention
- direct physical K reads for score computation
- GPU stable softmax
- direct physical V reads for context computation
- GQA head mapping
- Python/Rust/GPU correctness

## What this does not prove

- vLLM-style PagedAttention serving
- dynamic block allocation
- block eviction
- prefix sharing
- continuous batching
- variable sequence lengths
- long-context efficiency
- kernel fusion
- production memory efficiency
- FP16/BF16 correctness
- performance superiority
- model quality
- end-to-end inference
