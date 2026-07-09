# Benchmark Methodology

Future benchmark reports should track at least:

- KV bytes per token
- Peak VRAM
- Max context length
- Max batch size or concurrency
- TTFT
- TPOT
- Tokens per second
- Kernel latency
- Fragmentation waste
- Numerical or quality drift

Reporting rules:

- Separate synthetic microbenchmarks from end-to-end model inference.
- State whether the result is CPU reference, GPU kernel microbenchmark, or real decode path.
- Keep one hardware target fixed for comparability: RTX 4060 Laptop GPU with 8 GB VRAM.
- Record the exact software stack for every benchmark run.

Interpretation guidance:

- Paging metrics matter for allocator efficiency and fragmentation.
- Latent KV metrics matter for bytes per token and decode overhead.
- Quantization metrics matter for memory reduction versus reconstruction error and throughput cost.
