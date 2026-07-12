# GPU direct paged latent-space GQA

This milestone validates direct paged latent-space attention for the fixed synthetic
`f32` shape: batch 1, sequence 8, four query heads, two KV heads, head dimension 8,
latent dimension 8, and block size 2. Physical latent storage is `[4, 2, 8]`, flattened
to `[8, 8]`, and the runtime table is `[2, 0, 3, 1]`. Query heads map to KV heads as
`[0, 0, 1, 1]`.

The GPU pipeline has three stages: direct paged latent scores, GPU stable softmax,
and direct paged latent-context aggregation followed by the selected V projection.
The score and context stages resolve the physical block table and KV-head mapping on
the GPU. Physical latent blocks are read directly; no logical latent device tensor or
reconstructed K/V device tensors are persisted.

The materialized paged oracle and contiguous direct latent implementation are
independent correctness controls. Cache accounting is 64 stored latent values versus
256 hypothetical full K/V values, a 4x ratio for stored cache values only. This does
not claim a 4x reduction in total GPU memory.

Run with `bash scripts/run_gpu_direct_paged_latent_gqa.sh`. This is direct paged
latent-space attention for a fixed synthetic linear configuration. It is not a
complete DeepSeek MLA or production PagedAttention implementation.
