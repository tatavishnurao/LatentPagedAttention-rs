# GPU FP16 paged latent storage with FP32 attention

This milestone validates FP16 physical latent-cache storage while retaining
FP32 for the incoming write vector, Q, projection weights, scores, softmax,
latent-context reductions, and final context.

The fixed shape is batch 1, sequence 8, four query heads, two KV heads, head
dimension 8, latent dimension 8, block size 2, and four physical blocks. The
runtime block table remains `[2, 0, 3, 1]`, with write cases at token positions
3 and 4.

The four GPU stages are:

```text
FP32 latent write vector -> FP16 cache write
FP16 cache load -> FP32 direct paged scores
FP32 stable softmax
FP16 cache load -> FP32 latent context and V projection
```

The updated FP16 cache stays on the GPU between the write and attention stages.
It is read back only after context computation for bit-exact storage validation.
No logical latent cache or reconstructed K/V device buffers are allocated.

The released cuTile `0.2.0` API provides `cutile::half::f16` and the device-side
`convert_tile` operation. FP32 write tiles are converted to FP16 on GPU, and
FP16 latent tiles are converted to FP32 before multiplication or reduction.
The existing FP32 stable-softmax kernel is reused unchanged.

Run:

```bash
bash scripts/run_gpu_fp16_paged_latent_attention.sh
```

Observed GPU-vs-FP16-storage-oracle maximum errors were:

```text
balanced: scores 5.96e-8, probabilities 1.49e-8, context 2.98e-8
stable:   scores 1.91e-6, probabilities 2.38e-7, context 1.04e-7
```

FP16-storage-vs-FP32-baseline post-write maximum errors were:

```text
balanced: scores 9.64e-5, probabilities 1.29e-5, context 1.34e-5
stable:   scores 1.56e-4, probabilities 3.11e-5, context 1.24e-5
```

The physical latent cache contains 64 values: 128 bytes in FP16 versus 256
bytes in FP32. A hypothetical full FP16 K/V cache for this fixed shape is 512
bytes. The 2x and 4x figures describe these storage-value comparisons only,
not total runtime GPU memory.

This pass validates FP16 latent-cache storage with FP32 attention arithmetic. It
does not establish Tensor Core use or a performance advantage. It also does not
implement BF16, dynamic allocation, sequence growth, variable shapes, masking,
or production cache management.
