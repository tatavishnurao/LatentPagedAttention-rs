# Architecture diagrams

## What the system does
LatentPagedAttention-rs is a correctness-first Rust, Python, and cuTile experiment for paged latent-cache decode attention, compared with an FP16 full-KV paged baseline in a controlled synthetic linear formulation.

## Problem addressed
It studies whether persistent full K/V rows can be replaced by smaller physical latent rows while doing score and value algebra in latent space at decode time.

## Overview
[System overview](system-overview.svg) follows runtime block-table addressing, optional device-side write, FP16 storage, FP32 arithmetic, masking, aggregation, and output projection. The baseline is a comparison lane.

## Critical path
[Critical path](critical-path.svg) shows the persistent-state versus decode-compute trade-off and the Python oracle → Rust CPU reference → cuTile GPU parity lane.

## Evidence used
README.md; docs/ARCHITECTURE.md; docs/02_memory_model.md; docs/04_benchmark_methodology.md; docs/06_reference_benchmark_report.md; reports/final_benchmark/summary.csv and summary.md; tests/test_memory_model.py, test_cache_ref.py, test_block_table.py, test_attention_shapes.py; scripts/run_gpu_fp16_paged_latent_storage.sh, run_gpu_paged_latent_write_attention.sh, run_final_benchmark.sh.

## Limitations and uncertainty
The 16× figure is persistent cache-byte ratio for the stated synthetic profile. The approximately 32.6% slower latent read path is synchronized host end-to-end timing for this implementation/profile/hardware. No claim is made about total GPU memory, production serving, DeepSeek MLA reproduction, model quality, or general performance ranking.

