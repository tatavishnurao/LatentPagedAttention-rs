# Career Positioning

## Project summary

Built and released a correctness-first paged latent-cache attention prototype
on an RTX 4060, demonstrating a 16x persistent-cache-byte reduction with a
measured 32.6% latent-read overhead against an FP16 full-KV baseline.

## Resume entry

**LatentPagedAttention-rs - Rust, Python, cuTile, CUDA**

- Built a correctness-first paged latent-cache decode-attention prototype with runtime block tables, partial-final-block masking, FP16 storage, and FP32 arithmetic.
- Chained Python oracle, Rust CPU reference, deterministic fixtures, and cuTile GPU execution for parity, negative controls, and bit-exact FP16 validation.
- Measured the 16x persistent-cache-byte reduction and 32.6% synchronized host end-to-end latent-read overhead for the synthetic model-shaped profile.

## One-line descriptions

Portfolio:

> Correctness-first paged latent-cache attention experiment in Rust, Python, and cuTile on an RTX 4060.

Pinned repository:

> Paged latent-cache attention experiment with Python/Rust/GPU parity on an RTX 4060.

## 30-second explanation

I built LatentPagedAttention-rs to answer one systems question: can a paged
latent cache be mutated and consumed directly on GPU without persistent full K/V
tensors? The project uses a synthetic linear formulation, validates the path
from Python to Rust CPU to cuTile GPU, and compares it with an FP16 full-KV paged
baseline. The result is 16x fewer persistent cache bytes for the model-shaped
profile, with a measured 32.6% latent-read overhead under synchronized host
end-to-end timing. It demonstrates a memory-compute trade-off, not production
readiness or a serving-system speedup.

## Technical explanation

The direct path stores physical FP16 latent rows, resolves runtime non-identity
block tables, optionally writes a new latent row on GPU, converts loads to FP32,
computes latent-space scores, applies masked stable softmax, aggregates latent
context, and projects the output. The baseline stores projected K and V in FP16
with the same paging controls. Validation covers partial final blocks, active
sequence lengths, GPU write-to-attention handoff, unchanged regions, and
bit-exact FP16 storage. The algebra is synthetic and linear; there is no real
checkpoint or claim of complete MLA support.

## Claims to retain

- The cache-byte result counts persistent cache storage only, not total GPU memory.
- Timing is synchronized host end-to-end timing for this implementation and profile.
- The project does not claim production readiness or speedup against vLLM, FlashAttention, TensorRT-LLM, or another serving system.
- No real model quality, dynamic allocation, continuous batching, or additional precision format is evaluated.
