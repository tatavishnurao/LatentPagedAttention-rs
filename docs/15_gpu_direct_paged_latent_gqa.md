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

The cuTile tile coordinates are tile indices, not element-row offsets. Therefore a
`[2, 8]` physical block is loaded from `[physical_block, 0]` in the flattened
`[8, 8]` storage. Probability chunks use `[0]`, `[1]`, `[2]`, and `[3]` for the four
`[1, 2]` tiles. This addressing is exercised by the non-identity table control.

Run with:

```bash
bash scripts/run_gpu_direct_paged_latent_gqa.sh
```

The validated run executed `direct_paged_latent_scores`, the existing
`stable_softmax_8`, and `direct_paged_latent_context` as separate GPU stages. Scores
and probabilities remained on the device between stages; only the final scores,
probabilities, and context were read back. The K/V distinction, swapped projection
head, identity-table, materialized-oracle, and no-device-materialization controls all
passed.

Observed maximum absolute errors were:

| case | scores | probabilities | context | probability row sum |
| --- | ---: | ---: | ---: | ---: |
| balanced_attention | 2.9802322e-8 | 2.2351742e-8 | 5.9604645e-8 | 5.9604645e-8 |
| stable_softmax_attention | 1.7881393e-6 | 1.4901161e-7 | 8.940697e-8 | 1.1920929e-7 |

Reports from the two required runs were:

```text
reports/rtx4060_gpu_smoke/direct_paged_latent_gqa_20260712T193639Z.txt
reports/rtx4060_gpu_smoke/direct_paged_latent_gqa_20260712T193641Z.txt
```

This is direct paged latent-space attention for a fixed synthetic linear
configuration. It is not a complete DeepSeek MLA or production PagedAttention
implementation. Paged latent-cache writing remains a separate milestone.
