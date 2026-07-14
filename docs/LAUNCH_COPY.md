# Launch Copy

## LinkedIn Launch Post

I’ve released LatentPagedAttention-rs v0.1, a correctness-first GPU research
prototype exploring what happens when paged cache addressing is combined with
latent KV compression.

The project is built in Rust, Python, and cuTile, and validated on an NVIDIA RTX
4060 Laptop GPU. It implements physical paged latent-cache storage, runtime
non-identity block tables, runtime sequence masking, partial-final-block masking,
FP16 cache writes, GPU write-to-attention handoff, and FP32 attention arithmetic.

For a synthetic model-shaped profile, the latent path uses 16x fewer persistent
FP16 cache bytes than an FP16 full-KV paged baseline: 65,536 bytes versus
1,048,576 bytes. The trade-off is real: the current latent read path is
approximately 32.6% slower under synchronized host end-to-end timing.

That is the point of the release. This is not a production serving runtime or a
speedup claim. It is a reproducible, correctness-first prototype for studying a
memory-versus-compute trade-off in decode attention.

Repo/release: <link>

I’d welcome technical review from people working on attention kernels, KV-cache
systems, Rust GPU tooling, or low-VRAM inference.

## X / Twitter Post

Released LatentPagedAttention-rs v0.1: Rust+Python+cuTile paged latent-cache decode attention on RTX 4060. Synthetic profile: 16x fewer persistent FP16 cache bytes vs full-KV, with ~32.6% slower latent read time. Memory-compute trade-off, not a speedup claim.

## Hacker News

Title:

```text
Show HN: LatentPagedAttention-rs - paged latent-cache attention on an RTX 4060
```

First comment:

```text
Hi HN. I built LatentPagedAttention-rs as a correctness-first research prototype
for combining paged cache addressing with latent KV compression.

The repo implements a direct paged latent-space GQA decode path in Rust, Python,
and cuTile, validated on an RTX 4060. The GPU path supports runtime non-identity
block tables, runtime active sequence lengths, partial-final-block masking, FP16
latent storage, FP32 attention arithmetic, and GPU cache writes that feed directly
into attention without a host cache round trip.

For the synthetic model-shaped profile, persistent FP16 cache storage is 65,536
bytes versus 1,048,576 bytes for an FP16 full-KV paged baseline: a 16x persistent
cache-byte reduction. The current latent read path is not faster; it is about
32.6% slower under synchronized host end-to-end timing. Compilation and cuTile JIT
are excluded, but these are not kernel-only timings.

Limitations: no real checkpoint, no production allocator, no continuous batching,
no Tensor Core claim, no total-GPU-memory reduction claim, and no comparison
against vLLM/FlashAttention/TensorRT-LLM performance.

I’d especially appreciate feedback on the kernel design, validation methodology,
and what profiling or kernel restructuring would be most useful next.
```

## r/rust Variant

I released LatentPagedAttention-rs v0.1, a Rust + Python + cuTile research
prototype for paged latent-cache decode attention on an RTX 4060.

The Rust side is used for CPU references, validation harnesses, cache/block-table
logic, and GPU integration. The project validates Python -> Rust CPU -> cuTile GPU
parity, including runtime sequence masking, partial final blocks, FP16 cache
storage, and bit-exact FP16 write checks.

The result is not a production inference engine. For a synthetic model-shaped
profile, the latent cache uses 16x fewer persistent FP16 cache bytes than an
FP16 full-KV baseline, but the current latent read path is about 32.6% slower
under synchronized host end-to-end timing.

I’d be interested in feedback from Rust systems engineers on the API shape,
validation structure, and how to keep GPU research code reproducible without
overclaiming.

## r/LocalLLaMA Variant

I released a small research prototype studying a memory trade-off in decode
attention: what if a paged KV cache stored a latent representation instead of
full K/V rows?

LatentPagedAttention-rs validates this on an RTX 4060 using Rust, Python, and
cuTile. It supports runtime block tables, partial-final-block masking, FP16
latent storage, GPU cache writes, and FP32 attention arithmetic. For a synthetic
model-shaped profile, the latent cache uses 16x fewer persistent FP16 cache bytes
than an FP16 full-KV paged baseline.

The current read path is slower by about 32.6%, so this is not a “faster
inference” post. It is a correctness-first memory-versus-compute experiment. It
also does not load a real model checkpoint or preserve model quality; those are
explicitly out of scope for v0.1.

## CUDA / GPU Programming Variant

I released LatentPagedAttention-rs v0.1, a cuTile-based RTX 4060 prototype for
direct paged latent-cache decode attention.

The GPU path reads physical FP16 latent blocks through runtime block tables,
converts loaded tiles to FP32, computes latent-space scores, applies masked
stable softmax, accumulates latent context in FP32, and applies an FP32 output
projection. It also validates GPU FP32-to-FP16 latent writes followed immediately
by attention using the updated device cache.

The interesting result is mixed: 16x fewer persistent FP16 cache bytes versus an
FP16 full-KV paged baseline for a synthetic model-shaped profile, but about
32.6% slower latent-read time under synchronized host end-to-end timing.

I’d welcome feedback on the kernel structure, masking approach, block-table
addressing, and what profiling should come before any optimization work.
