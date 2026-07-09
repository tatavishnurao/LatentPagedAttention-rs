# Research Question

Core question:

Can paged allocation plus latent KV compression plus optional KV quantization extend context length and concurrency for LLM decode workloads on 8 GB consumer GPUs?

This repository studies that question as a reproducible engineering and kernel-lab problem, not as a claim of a new production inference engine.

Working assumptions:

- The first bottleneck on RTX 4060 laptop GPUs is often KV-cache memory, not raw arithmetic throughput.
- Paged allocation can improve utilization and fragmentation behavior for dynamic sequence growth.
- Latent KV compression can reduce bytes per token, but it introduces reconstruction or projection work during decode.
- Quantization may reduce bytes further, but it adds numerical error and implementation complexity.

Immediate milestone:

- Build small, correctness-tested Python and Rust references.
- Use those references to constrain later GPU kernel work.
- Avoid performance claims until the benchmark harness and reporting discipline are in place.
