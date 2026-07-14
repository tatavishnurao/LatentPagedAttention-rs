# Career Positioning

## Resume Entry

**LatentPagedAttention-rs - Rust, Python, cuTile, CUDA**

- Implemented a correctness-first paged latent-cache decode-attention prototype with runtime block tables, partial-final-block masking, FP16 cache writes, and FP32 grouped-query attention arithmetic.
- Built Python and Rust reference oracles plus cuTile GPU validation, covering deterministic fixtures, bit-exact FP16 cache checks, runtime sequence lengths up to 1,024 tokens, and RTX 4060 readback parity.
- Measured a 16x persistent-cache-byte reduction against an FP16 full-KV paged baseline for a synthetic model-shaped profile, with approximately 32.6% higher synchronized host end-to-end read time in the current prototype.

## One-Line Portfolio Description

Correctness-first Rust/cuTile prototype for paged latent-cache LLM decode attention on an RTX 4060.

## GitHub Pinned-Repository Description

Paged latent-cache attention prototype in Rust + cuTile, validated on RTX 4060.

## LinkedIn Project Description

LatentPagedAttention-rs is a Rust and Python research prototype for paged latent-cache decode attention on an RTX 4060. It validates runtime block tables, partial-final-block masking, FP16 cache writes, GPU write-to-attention handoff, FP16 latent storage, and FP32 attention arithmetic through Python, Rust CPU, and cuTile GPU parity. The synthetic model-shaped profile shows a 16x persistent-cache-byte reduction versus an FP16 full-KV baseline, with approximately 32.6% higher synchronized host end-to-end latent-read time in the current implementation.

## 30-Second Interview Version

I built LatentPagedAttention-rs to study whether a paged latent cache can be mutated and consumed directly on GPU for decode attention. It is a Rust, Python, and cuTile prototype validated on an RTX 4060. The model-shaped synthetic profile stores 16x fewer persistent FP16 cache bytes than a full-KV baseline, but the current latent read path is about 32.6% slower under synchronized host end-to-end timing. The point was to validate the memory-versus-compute trade-off honestly, not to claim a production speedup.

## 90-Second Interview Version

LatentPagedAttention-rs investigates a narrow inference-systems question: can we combine paged cache addressing with latent KV compression without materializing persistent full K/V tensors? I implemented Python oracles, Rust CPU references, deterministic fixtures, and cuTile GPU kernels for an RTX 4060. The GPU path supports runtime non-identity block tables, active sequence lengths, partial final blocks, FP16 latent storage, FP32 score/softmax/context arithmetic, and a GPU write-to-attention handoff where the updated cache stays on device. I also implemented an FP16 full-KV paged baseline using the same storage width. The final synthetic `model_small` profile showed 16x fewer persistent cache bytes for the latent path, while the current latent read path was about 32.6% slower than the full-KV read path. I framed it as a memory-versus-compute trade-off and documented what it does not prove: no real model, no production allocator, no vLLM or FlashAttention comparison, and no total GPU-memory reduction claim.

## Deep Technical Version

The project starts from the observation that paged attention solves allocation and fragmentation, but still stores full K/V rows. I tested a synthetic linear latent-cache formulation where `K_t = L_t W_k` and `V_t = L_t W_v`. Scores can be reassociated as `q (L_t W_k)^T = L_t (W_k q)`, so the kernel projects the query into latent space and dots it with paged latent rows. Context can be reassociated as `sum p_t (L_t W_v) = (sum p_t L_t) W_v`, so the kernel aggregates latent context first and applies the V projection after.

The implementation uses physical latent blocks, runtime block tables, FP16 persistent cache storage, and FP32 arithmetic. I validated runtime sequence masking so inactive tokens have zero probability and do not contribute to context. I also validated GPU cache mutation: an FP32 latent row is converted to FP16 on GPU, stored into one physical block row, and consumed by attention without a host cache round trip. Correctness is chained from Python references to Rust CPU references to cuTile GPU readback. Negative controls cover identity block tables, projection changes, changed-element counts, unchanged regions, and bit-exact FP16 storage.

The final `model_small` profile has 16 query heads, 4 KV heads, head dimension 64, latent dimension 32, block size 16, and max sequence length 1024. Compared with an FP16 full-KV paged baseline, the latent cache stores 65,536 persistent bytes instead of 1,048,576, a 16x persistent cache-byte reduction. The current latent read path is slower by about 32.6% under synchronized host end-to-end timing, so the result is an honest memory-versus-compute trade-off rather than a speedup claim.
