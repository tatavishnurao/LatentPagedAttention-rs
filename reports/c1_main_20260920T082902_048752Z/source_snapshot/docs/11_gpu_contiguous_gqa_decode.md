# GPU Contiguous GQA Decode

## Purpose

This pass validates contiguous `f32` GQA decode attention on the RTX 4060
without paging. The goal is to isolate the attention mathematics from block
table addressing before starting Paged GQA.

## Fixed dimensions

- batch = 1
- q_heads = 4
- kv_heads = 2
- group_size = 2
- seq_len = 8
- head_dim = 8
- dtype = `f32`

## Q-head to KV-head mapping

- q_head 0 -> kv_head 0
- q_head 1 -> kv_head 0
- q_head 2 -> kv_head 1
- q_head 3 -> kv_head 1

The kernel computes `kv_head = q_head / group_size` on the GPU.

## Input and output layouts

Reference Python layout:

- `Q`: `[batch, q_heads, head_dim]`
- `K`: `[batch, seq_len, kv_heads, head_dim]`
- `V`: `[batch, seq_len, kv_heads, head_dim]`

Fixed GPU validation layout:

- `Q`: `[4, 8]`
- `K`: `[2, 8, 8]`, reshaped on device as `[16, 8]`
- `V`: `[2, 8, 8]`, reshaped on device as `[16, 8]`
- `scores`: `[4, 8]`
- `probabilities`: `[4, 8]`
- `context`: `[4, 8]`

The head-major K/V fixture is generated in Python and committed as JSON so the
Rust CPU reference and the GPU example validate against exactly the same values.

## Operation

For each query head tile:

1. load `Q[q_head, :]`
2. resolve `kv_head = q_head / group_size`
3. load the matching `[seq_len, head_dim]` K and V tiles
4. compute `scores = QK^T / sqrt(head_dim)`
5. compute stable softmax:
   `shifted = scores - max(scores)`
   `probabilities = exp(shifted) / sum(exp(shifted))`
6. compute `context = probabilities V`

## Validation chain

Python reference
-> committed JSON fixture
-> Rust CPU contiguous GQA reference
-> cuTile GPU scores
-> cuTile GPU probabilities
-> cuTile GPU context
-> full host-side comparison

## Observed numerical errors

On the validated RTX 4060 run:

- balanced scores max abs error: `1.1920929e-7`
- balanced probabilities max abs error: `1.4901161e-8`
- balanced context max abs error: `5.9604645e-8`
- stable-softmax scores max abs error: `1.5258789e-5`
- stable-softmax probabilities max abs error:
  `1.1646703e-20`
- stable-softmax context max abs error: `0`
- maximum probability row sum error: `0`

These values passed the initial tolerances, so no tolerance expansion was
required.

## cuTile issue encountered

The fused scaffold required two small corrections:

- the score scale had to be broadcast into a tile instead of multiplying a
  tile by a raw Rust scalar
- `load_tile` for K/V uses tile coordinates, not element-row offsets, so the
  KV-head selector had to use tile index `kv_head` instead of `kv_head * 8`

The implementation remained fused after those fixes.

## Commands

```bash
source scripts/cutile_env.sh
bash scripts/run_cutile_smoke.sh
bash scripts/run_gpu_paged_lookup.sh
bash scripts/run_gpu_paged_kv_write.sh
bash scripts/run_gpu_gqa_decode.sh
```

## What this proves

- fused cuTile contiguous GQA decode compiles and executes on the RTX 4060
- GPU head mapping matches the Python and Rust references
- GPU score computation matches scaled dot-product attention
- GPU stable softmax is correct for the committed balanced and stress cases
- GPU context aggregation matches Python and Rust

## What this does not prove

- Paged GQA
- PagedAttention
- long-context behavior
- FP16/BF16 correctness
- production performance
- model quality
- end-to-end inference
