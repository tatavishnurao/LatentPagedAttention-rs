# C1 milestone progress

Base HEAD: `213d45a`; branch `research/c1-rtable`; no AGENTS.md found.
Pre-existing untracked manuscript/audit files were preserved.

## Completed

- Audited HEAD and historical relationship; verified reviewer hypotheses against source.
- Confirmed B1 repeats `q @ K_projection` in each `(head, logical_block)` score program.
- Confirmed existing C uses full-table load+extract paging helper.
- Implemented generated C1 kernels: one FP32 projected query per head/invocation,
  single-entry R-TABLE lookup, unchanged FP16 storage/FP32 arithmetic, softmax/context.
- Added C1 example, independent FP64 reconstructed-K/V oracle, changed-query refresh
  regression, source generator, analysis script, protocol, audit, ledger and report.
- Corrected active historical 32.6% language to whole-process timing; preserved raw data.
- Pilot passed at 1K and was sealed: `reports/c1_pilot_20260920T082449_625698Z/`.
- Current tests pass: pytest 65; Rust workspace 24; shell validator pass.
- Historical counts checked: commit `14766ac` has 95 tracked files; `213d45a` has 205;
  readiness package manifest is distinct. Nine local Nsight reports exist.

## Exact commands/results

```text
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q       # 65 passed
CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3 LIBCLANG_PATH=/usr/lib/llvm-18/lib \
 CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.3/bin/tileiras cargo test --workspace -q # 24 passed
bash tests/test_validate_release_script.sh                 # VALIDATE_RELEASE_SCRIPT_TEST_OK=1
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_c1_milestone.py --pilot
 # pilot C1_OK=1; independent oracle pass; sealed SHA256SUMS
```

## Measurements/blockers

No accepted main measurements. The frozen main budget is 3 processes × 12 samples,
3 warmups at 1K/2K/4K/8K with rotating A1/B1/C1 and CUDA-event pipeline/component
records. First attempt `reports/c1_main_20260920T082902_048752Z` was stopped during
build after discovering the newer strict policy. Guarded retry
`reports/c1_main_20260920T083048_387262Z` failed closed before execution because
Windows reported AC but `Balanced`, not High performance. Current telemetry: 78 C,
3% GPU, no compute process; no threshold was changed. Pilot is diagnostic only and
excluded from main claims. Full details: `docs/C1_THERMAL_CORRECTION.md`.

## Next action

On AC + the predeclared High performance Windows scheme, rerun:
`OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_c1_milestone.py`, then
`python3 scripts/analyze_c1.py reports/<complete-main-dir> reports/c1_results`.
Do not run 16K/32K, R-CTX-PAR, cache eviction, or manuscript performance claims yet.
