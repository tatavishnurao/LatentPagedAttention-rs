# GPU Paged KV Write

## Purpose

This milestone writes one decode token's K and V vectors into paged physical
cache storage using cuTile Rust. It is a correctness validation, not a
performance benchmark.

## Layout

K and V use the layout `[num_physical_blocks, block_size, width]` with
`num_physical_blocks=4`, `block_size=2`, and `width=8`.

## Address resolution

```text
logical_block = token_position / block_size
offset = token_position % block_size
physical_block = block_table[logical_block]
```

The test uses the non-identity block table `[2, 0, 1]` and the padded GPU table
`[2, 0, 1, 3]`.

## Safe ownership design

Each cuTile output partition owns one physical `[2, 8]` block. Every tile loads
and stores only its exclusively owned K and V block. Only the tile whose
physical block ID matches the runtime block-table result replaces one row;
other tiles store their original values unchanged.

## Correctness chain

Python reference -> JSON fixture -> Rust CPU reference -> cuTile GPU write ->
complete-cache comparison.

The cases cover position 3 with offset 1 and position 4 with offset 0. The
example also performs a GPU write-to-GPU paged-lookup round trip.

## Commands

```bash
source scripts/cutile_env.sh
bash scripts/run_gpu_paged_kv_write.sh
bash scripts/run_gpu_paged_kv_write.sh
```

## What this proves

- Runtime token-position resolution
- Runtime non-identity block-table resolution
- Correct K and V writes
- No adjacent cache corruption
- Python/Rust/GPU parity
- GPU write-to-GPU lookup round trip

## What this does not prove

- GQA attention
- PagedAttention performance
- FP16 behavior
- Production concurrency
- Cache allocator behavior
- Model correctness
- Throughput
