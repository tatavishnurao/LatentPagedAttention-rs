# Final Report

## Abstract

LatentPagedAttention-rs v0.1.0 validates a paged latent-cache decode-attention
prototype on an RTX 4060. It covers direct paged latent attention, FP16 latent
storage with FP32 arithmetic, runtime active sequence lengths, partial-final-block
masking, a synthetic model-shaped profile, and an FP16 full-KV paged baseline.

## Research Question

Can a paged latent cache be read and mutated directly on GPU while preserving
correct decode-attention behavior against Python and Rust references?

## Scope

The release is a fixed-profile research prototype. It is not a serving runtime,
not full DeepSeek MLA, and not a real-checkpoint integration.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md). The direct path stores physical latent
blocks, resolves a runtime block table, computes scores in latent space, applies
masked stable softmax, aggregates latent context, and applies the V projection.

## Correctness Methodology

Correctness is checked through Python references, deterministic fixtures, Rust CPU
references, cuTile GPU execution, readback, finite-output checks, probability
row-sum checks, inactive-probability checks, and non-identity block-table controls.

## Paging Design

Logical blocks map to physical blocks through explicit runtime block tables.
The tiny profile uses `[2, 0, 3, 1]`. The `model_small` profile uses a deterministic
64-block permutation with checksum `11337306458621306720`.

## Latent-Space Algebra

The algebra is synthetic and linear. Queries are projected into latent space
through K projection heads, scores are computed against latent cache rows, and
latent context is projected through selected V projection heads.

## Precision Design

FP16 latent storage uses FP32 incoming writes converted to FP16 on GPU. Latent
loads are converted back to FP32 before score, softmax, context, and output
projection arithmetic. The FP16 full-KV baseline stores K and V in FP16 and also
uses FP32 arithmetic.

## Runtime Sequence Masking

Runtime active lengths are validated for tiny lengths `1, 3, 4, 7, 8` and
model-shaped lengths `17, 129, 513, 1024`. Inactive probabilities are zero and
active rows sum to one.

## Baseline

The FP16 full-KV paged baseline reconstructs logical K/V from the same synthetic
latent source, quantizes projected K/V to FP16 storage, applies the same block
table, and computes FP32 attention.

## Benchmark Methodology

`scripts/run_final_benchmark.sh` records synchronized host end-to-end timing for
the validation binaries after warmup. These timings include explicit process-level
synchronization and are not kernel-only latency.

## Results

The canonical final benchmark summary is committed in
`reports/final_benchmark/summary.csv` and `reports/final_benchmark/summary.md`.
The timing method is synchronized host end-to-end timing with three measured
processes; it is not kernel-only latency.

| operation | process_count | min_ms | mean_ms | max_ms |
|---|---:|---:|---:|---:|
| full_kv_paged_attention_read | 3 | 1366.969 | 1391.022 | 1405.751 |
| latent_paged_attention_read | 3 | 1705.150 | 1844.891 | 2017.385 |
| latent_write_to_attention | 3 | 1367.174 | 1487.776 | 1586.213 |

The latent read path is approximately `32.6%` slower than the FP16 full-KV read
baseline by mean synchronized host end-to-end time. The result is therefore a
memory-versus-compute trade-off, not an unconditional speedup.

## Numerical Error

Observed GPU-vs-Rust errors are within the tolerances printed by each validation
script. Model-shaped latent errors were at most `7.6293945e-6` for scores and
context in the validated run. Full-KV baseline errors were at most `1.4305115e-5`
for context in the validated run.

## Cache-Memory Accounting

For `model_small`, FP16 latent cache bytes are `65,536`. FP16 full-KV cache bytes
are `1,048,576`. The persistent cache-byte ratio is `16x` for this synthetic
profile. This ratio does not describe total GPU memory.

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md).

## Reproduction

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Future Work

Deferred work includes BF16, real checkpoints, dynamic allocation, prefix sharing,
eviction, production scheduling, CUDA graphs, and distributed inference.

## Conclusion

v0.1.0 provides a finite, evidence-backed paged latent-cache attention prototype
with narrow claims and reproducible validation commands.
