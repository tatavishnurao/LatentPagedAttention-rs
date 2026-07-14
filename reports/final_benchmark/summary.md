# Final Benchmark Summary

Timing method: SYNCHRONIZED_HOST_END_TO_END_TIME.

| operation | process_count | min_ms | mean_ms | max_ms |
|---|---:|---:|---:|---:|
| full_kv_paged_attention_read | 3 | 1417.207 | 1458.249 | 1493.987 |
| latent_paged_attention_read | 3 | 1720.900 | 1809.968 | 1864.154 |
| latent_write_to_attention | 3 | 1394.896 | 1464.487 | 1546.910 |

BENCHMARK_WARMUP_EXCLUDED=1
JIT_EXCLUDED_FROM_MEASUREMENTS=1
SYNCHRONIZATION_METHOD=SYNCHRONIZED_HOST_END_TO_END_TIME
BENCHMARK_PROCESS_COUNT=3
BENCHMARK_RESULTS_WRITTEN=1
FINAL_BENCHMARK_OK=1
