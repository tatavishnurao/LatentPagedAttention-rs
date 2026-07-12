# GPU paged latent-cache write and attention round trip

This milestone validates mutation of one token position in an already allocated
physical latent cache, followed immediately by direct paged latent-space GQA. It
uses the fixed `f32` shape: batch 1, sequence 8, four query heads, two KV heads,
head dimension 8, latent dimension 8, block size 2, and four physical blocks.

The physical cache is `[4, 2, 8]`, flattened on GPU as `[8, 8]`. The runtime
block table is `[2, 0, 3, 1]`. The two cases cover token position 3 at physical
block 0, offset 1, and token position 4 at physical block 3, offset 0.

Each cuTile write tile owns one complete physical `[2, 8]` block. The kernel
resolves token position to logical block, block offset, and physical block on
the GPU, then masks one row and stores the owned tile. Writes are disjoint and
do not require unsafe code.

The four GPU stages are:

```text
paged latent-cache write
direct paged latent scores
GPU stable softmax
direct paged latent context and V projection
```

The writer's device output is passed directly into the attention stages. The
updated latent cache is not read to the host or uploaded again before attention.
Only after the complete post-write attention pipeline is synchronized are the
attention outputs and updated cache read back for validation.

The CPU and Python controls verify the resolved location, exact eight-element
mutation, unchanged cache region, pre-write/post-write score behavior, context
change, and probability normalization. The GPU identity-table control performs
an actual alternate GPU write and confirms that the non-identity table changes
the physical target.

Run the validation with:

```bash
bash scripts/run_gpu_paged_latent_write_attention.sh
```

Final validation reports:

```text
reports/rtx4060_gpu_smoke/paged_latent_write_attention_20260712T195012Z.txt
reports/rtx4060_gpu_smoke/paged_latent_write_attention_20260712T195014Z.txt
```

The validated RTX 4060 runs produced post-write maximum errors of approximately
`4.47e-8` scores and `2.98e-8` context for the balanced case, and `1.79e-6`
scores and `1.19e-7` context for the stable case. The largest probability row
sum error was `1.19e-7`.

This pass mutates an already allocated token position. It does not allocate new
blocks or increase sequence length. It does not implement dynamic allocation,
append, masking, eviction, batching, or a production cache manager.

The cache contains 64 latent values versus 256 hypothetical full K/V values for
this synthetic shape. The 4x figure applies to stored cache values only, not
total runtime GPU memory.
