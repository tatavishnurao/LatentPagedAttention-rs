# GPU Latent-KV Reconstruction

## Purpose

This milestone validates standalone MLA-style synthetic latent-KV reconstruction
on the RTX 4060 using cuTile. It checks linear reconstruction correctness only:

```text
K_reconstructed = latent_cache x K_projection
V_reconstructed = latent_cache x V_projection
```

This is not a complete DeepSeek MLA implementation.

## Fixed Dimensions

- batch: 1
- sequence length: 8
- latent dimension: 8
- KV heads: 2
- head dimension: 8
- projection width: 16
- dtype: f32

## Layouts

Input latent cache:

```text
[batch, seq_len, latent_dim] = [1, 8, 8]
```

Projection matrices:

```text
K projection: [latent_dim, kv_heads * head_dim] = [8, 16]
V projection: [latent_dim, kv_heads * head_dim] = [8, 16]
```

GPU output layout:

```text
K: [seq_len, projection_width] = [8, 16]
V: [seq_len, projection_width] = [8, 16]
```

Validation also checks the future attention-friendly head-major view:

```text
[kv_heads, seq_len, head_dim] = [2, 8, 8]
```

## Implementation

The cuTile implementation is fused for this fixed problem size. One GPU tile
owns one token row. Each tile loads one latent row and both projection matrices,
then writes one reconstructed K row and one reconstructed V row.

No CPU projection is used in the GPU validation path.

## Correctness Chain

```text
Python reference
-> deterministic JSON fixture
-> Rust CPU reference
-> cuTile GPU reconstruction
-> full K/V comparison
```

The fixture contains two cases:

- `balanced_projection`
- `signed_accumulation`

## Negative Controls

The GPU validation confirms:

- K and V projection matrices produce distinct results
- at least two token rows differ
- the head-major validation layout is orientation-sensitive

## Commands

```bash
source scripts/cutile_env.sh
bash scripts/run_gpu_latent_kv_reconstruction.sh
```

## Observed Errors

Validated on the RTX 4060 Laptop GPU with cuTile `0.2.0`:

```text
balanced_projection K max abs error: 0.000000029802322
balanced_projection V max abs error: 0.000000059604645
signed_accumulation K max abs error: 0.000000014901161
signed_accumulation V max abs error: 0.000000029802322
```

No tolerance increase was required.

## Theoretical Cache Ratio

The latent cache stores 8 values per token while full K and V store 32 values
per token for this synthetic configuration, giving a theoretical 4x
stored-cache ratio.

This standalone validation materializes reconstructed K and V tensors, so it
does not yet demonstrate a 4x reduction in runtime peak memory.

## What This Proves

- linear latent-KV reconstruction executes on the GPU
- K and V projections are independently applied
- signed accumulation matches Python and Rust references
- token-major and head-major validation layouts agree
- Python/Rust/GPU parity holds for the fixed f32 configuration

## What This Does Not Prove

- DeepSeek MLA parity
- Paged Latent KV
- on-demand reconstruction inside attention
- runtime peak-memory reduction
- FP16/BF16 correctness
- model quality preservation
- end-to-end inference
- latency, bandwidth, throughput, or speedups
