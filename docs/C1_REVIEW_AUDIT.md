# External-review audit (HEAD `213d45a`, 2026-09-20)

| reviewer hypothesis | source check | verdict | action |
|---|---|---|---|
| `p15b_rtable_kernels.rs` recomputes query projection per head/block | `src/cutile/p15b_rtable_kernels.rs`, e.g. `p15b_model_profile_kernel_1024::model_small_scores...`: `q_row`, `kp`, and `reduce_sum` occur in the `(q_head, logical_block)` entry | confirmed | C1 hoists to one `[16,32]` FP32 buffer per invocation |
| Existing C hoists projection but uses old lookup | `src/cutile/model_profile_preprojected.rs`: `table.load_tile([64],[0])` + `extract`; score entry does not perform table lookup | confirmed | C1 generator replaces it with `table.load_tile([1],[logical])` |
| No C + R-TABLE experiment exists | historical `p15b_rtable.rs` variants are A0/A1/B0/B1; preprojected module is separate | confirmed at HEAD | added `c1_rtable.rs` and generated `c1.rs` |
| Original context launches only 16 programs and serially loops blocks | `p1_sequence_kernels.rs`: context entries use head tile id and literal `for logical_block in ...`; launch wrappers in sequence examples dispatch context over heads | confirmed | unchanged control; no R-CTX-PAR |
| headline/profile raw evidence may be absent | current checkout contains `reports/final_benchmark/results.jsonl`, P15 raw JSONL, and 9 `.ncu-rep` files | contradicted for this checkout; provenance is not yet uniform | ledger distinguishes direct raw artifacts and historical process boundary |
| active docs call 32.6% a kernel result | README, release notes, technical/article/launch docs did; source `run_final_benchmark.sh` times whole `cargo run` | confirmed | active docs now label it process-level and withdraw kernel interpretation |
| correctness wrappers document finite/bitwise only at 1K | `run_paper_repair_correctness.py` accepts only `--seq 1024` profiles; native summary checks bitwise A1/A0 and B1/B0 | confirmed | C1 harness covers all four lengths; independent oracle added |
| “95-file bundle” confused with tracked count | `git ls-tree -r 14766ac` = 95; HEAD `213d45a` = 205; readiness canonical manifest is a package listing, not git count | partially confirmed | report exact counts and package/working-tree distinction |
| test counts differ across commits | historical readiness summary says 24 Rust/62 Python; current run says 24 Rust/65 Python after C1 tests | confirmed | ledger records exact command and scope |

Historical commit relationship: current HEAD is exactly `213d45a`; no later commit is
present in this worktree. The pre-existing untracked manuscript/audit material was not
discarded or rewritten.
