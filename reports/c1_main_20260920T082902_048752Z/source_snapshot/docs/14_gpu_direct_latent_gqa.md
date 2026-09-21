# GPU Direct Latent GQA

## Purpose
Validate direct contiguous latent-space f32 GQA decode on the RTX 4060 without materializing full reconstructed K/V tensors.

## Fixed dimensions
- batch 1
- sequence length 8
- 4 query heads
- 2 KV heads
- group size 2
- head dimension 8
- latent dimension 8
- projection width 16
- dtype f32

## Canonical and GPU projection layouts
Canonical projection layout:

- `[latent_dim, kv_heads * head_dim]`

GPU head-major projection layout:

- `[kv_heads, latent_dim, head_dim]`

The GPU kernel loads one projection head tile at a time.

## GQA mapping
`q_head -> q_head / group_size`

## Direct latent-space algebra
The direct path uses:

`scores = latent_cache × (K_projection_head × query) / sqrt(head_dim)`

`probabilities = stable_softmax(scores)`

`context = (probabilities × latent_cache) × V_projection_head`

This is the synthetic linear reassociation used in the current fixture set.

## Materialized oracle
The materialized oracle still exists for validation:

`latent -> reconstructed K/V -> contiguous GQA -> context`

The direct path is validated against that oracle, but does not allocate reconstructed K/V on the GPU.

## Implementation shape
- fused cuTile kernel
- one tile per query head
- no K output tensor
- no V output tensor
- no reconstruction kernel invocation
- no contiguous GQA invocation after reconstruction

## Cache accounting
The synthetic latent cache stores 8 values per token.
Full K and V store 32 values per token for this configuration.
That is a theoretical 4x stored-cache ratio.

This pass does not prove a 4x reduction in total runtime peak memory.

## Negative controls
- swapping KV projection heads changes scores
- replacing V projection with K projection changes context
- direct output matches the materialized oracle

## Validation chain
Python direct reference
-> Python materialized oracle
-> JSON fixture
-> Rust direct CPU reference
-> Rust materialized control
-> cuTile GPU direct latent GQA
-> host comparison

## Numerical notes
The stable-softmax case is used to check that the implementation remains finite under a larger score range.

## Commands
```bash
source scripts/cutile_env.sh
bash scripts/run_gpu_direct_latent_gqa.sh
```

## What this proves
- direct latent-space GQA algebra is correct for the synthetic linear setup
- the GPU implementation can compute scores, stable softmax, and context without full K/V materialization
- Python, Rust, and GPU outputs agree for the committed fixtures

## What this does not prove
- DeepSeek MLA completeness
- Paged Latent KV
- production memory savings
- attention throughput
- FP16/BF16 correctness
- end-to-end inference
- model quality
