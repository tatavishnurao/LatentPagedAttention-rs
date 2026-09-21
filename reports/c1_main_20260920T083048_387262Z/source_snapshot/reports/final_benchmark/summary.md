# Final Benchmark Summary

Timing method: SYNCHRONIZED_HOST_END_TO_END_TIME.

| operation | process_count | min_ms | mean_ms | max_ms |
|---|---:|---:|---:|---:|
| full_kv_paged_attention_read | 3 | 1366.969 | 1391.022 | 1405.751 |
| latent_paged_attention_read | 3 | 1705.150 | 1844.891 | 2017.385 |
| latent_write_to_attention | 3 | 1367.174 | 1487.776 | 1586.213 |

BENCHMARK_WARMUP_EXCLUDED=1
JIT_EXCLUDED_FROM_MEASUREMENTS=1
SYNCHRONIZATION_METHOD=SYNCHRONIZED_HOST_END_TO_END_TIME
BENCHMARK_PROCESS_COUNT=3
BENCHMARK_RESULTS_WRITTEN=1
FINAL_BENCHMARK_OK=1
