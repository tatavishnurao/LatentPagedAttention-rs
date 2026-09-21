# C1 claim-to-evidence ledger

All results are descriptive. A process is an independent replication unit; CUDA-event
samples within a process are not independent experiments.

| claim | raw artifact | source / variant | boundary; samples/processes | acceptance / regeneration | status |
|---|---|---|---|---|---|
| redundant per-block query projection exists | `crates/plkv-kernels/src/cutile/p15b_rtable_kernels.rs` | `213d45a`; B1/R-TABLE source | static source | `rg -n 'q_row|k_projection|reduce_sum' .../p15b_rtable_kernels.rs` | direct confirmed |
| old C uses full-table lookup | `crates/plkv-kernels/src/cutile/model_profile_preprojected.rs` | `213d45a`; C | static source | inspect `physical_block`; generator command below | direct confirmed |
| C1 algebra and paging implementation | `crates/plkv-kernels/src/cutile/c1.rs`, `examples/c1_rtable.rs` | working-tree commit not yet made; C1 | projection in pipeline, FP32, R-TABLE single-entry | `python scripts/generate_c1.py`; cargo build command below | implemented/builds |
| C1 correctness at 1K pilot | `reports/c1_pilot_20260920T082449_625698Z/` incl. `samples.jsonl`, `correctness.json`, `independent_oracle.json`, telemetry, source snapshot, SHA256SUMS | working tree; A1/B1/C1; nonidentity table | 1 process, 1 warmup, 3 samples/phase; correctness outside timing | `.venv/bin/python scripts/analyze_c1.py ... --allow-pilot` | direct pass; diagnostic only |
| C1 correctness 2K/4K/8K | raw evidence unavailable | intended main A1/B1/C1 | 3 processes x 12 samples planned | `OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_c1_milestone.py` | blocked by power policy |
| C1 pipeline/component latency | raw evidence unavailable for accepted main campaign; pilot raw only | intended A1/B1/C1 | pipeline primary; components diagnostic; CUDA events; resident buffers | `scripts/run_c1_milestone.py`, then `scripts/analyze_c1.py` | no main measurement |
| projected-query workspace | `docs/C1_PROTOCOL.md`; source allocation in `c1_rtable.rs` | C1 | `[16,32] FP32 = 2048 bytes` per C1 buffer; common workspaces listed | source inspection | direct analytical |
| historical 32.6% | `reports/final_benchmark/results.jsonl`, `summary.csv` | commit recorded in `environment.json`; old latent/full examples | whole-process synchronized host `cargo run`; 3 samples/process operation, no resident kernel boundary | preserved; do not reconstruct | direct raw, interpretation corrected |
| historical profiler evidence | `reports/p05_hardware_attribution/ncu_*.ncu-rep` (9 files) and extraction scripts | historical commits in package metadata; full/latent score/context | Nsight reports; raw report artifacts | `scripts/extract_p05_ncu.py` | direct artifacts; attribution scope must be checked per report |
| 95-file claim | `git ls-tree -r 14766ac` = 95; HEAD `213d45a` = 205; readiness manifest package separate | historical source count vs evidence package | count only, no timing | `git ls-tree -r --name-only <commit> | wc -l`; inspect package MANIFEST | corrected distinction |
| tests | `reports/c1_build_20260920/python_tests.log`, `rust_tests.log`, `shell_tests.log` | working tree | pytest 65; Rust 24; shell wrapper pass | commands recorded in progress/report | direct pass |

## Reproduction

```bash
python3 scripts/generate_c1.py
CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3 LIBCLANG_PATH=/usr/lib/llvm-18/lib \
  CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.3/bin/tileiras \
  cargo build --release -p plkv-kernels --features gpu-cutile --example c1_rtable
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3 LIBCLANG_PATH=/usr/lib/llvm-18/lib \
  CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.3/bin/tileiras cargo test --workspace
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_c1_milestone.py
python3 scripts/analyze_c1.py reports/<complete-main-dir> reports/c1_results
```

The main command fails closed unless Windows reports AC and High performance power
scheme, then requires idle <=73 C and run <=82 C with no active throttling. The
observed environment reported AC online but `Balanced`, so no main GPU samples were
collected. The procedural abort and guard-rejected attempt are retained under
`reports/c1_main_*`; no failed attempt was deleted or counted as a result.
