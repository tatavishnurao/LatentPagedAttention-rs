# Limitations

- Synthetic linear algebra only; this is not a complete DeepSeek MLA implementation.
- No real model checkpoint integration.
- No dynamic block allocation, eviction, prefix sharing, continuous batching, or production scheduling.
- No BF16, FP8, FP4, RoPE, masks, batching, CUDA graphs, or automatic tuning in v0.1.x.
- Timing reports are synchronized host end-to-end measurements unless explicitly stated otherwise.
- Cache-byte ratios count persistent cache storage only, not total GPU memory.
- The `model_small` profile is synthetic and model-shaped; it is not claimed to match a production model.
