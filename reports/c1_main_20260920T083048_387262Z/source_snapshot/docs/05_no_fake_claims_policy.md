# No Fake Claims Policy

This repository will follow these rules:

- No claims of beating vLLM, FlashAttention, or llama.cpp without direct, reproducible benchmarks.
- No claims of NVFP4 acceleration on RTX 4060 hardware.
- No claims that MLA is a drop-in runtime replacement for Qwen or Llama families without explicit adaptation evidence.
- Always distinguish synthetic kernel benchmarks from real model inference.
- Always report commands, hardware, driver, CUDA, model, dtype, batch size, context length, and commit hash.

If a result is preliminary, synthetic, CPU-only, or shape-only, it must be labeled that way.
