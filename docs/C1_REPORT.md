# C1 bounded milestone report

## Verdict

The source review confirms redundant per-block query projection and confirms that
existing C was confounded by the old full-table block-table helper. C1 is implemented
and its 1K pilot passed the Rust reference, independent FP64 reconstructed-K/V oracle,
finite/normalization checks, nonidentity paging and changed-query projection-refresh
regression. The requested matched 1K/2K/4K/8K main latency experiment was **not
completed**: the current strict campaign guard correctly rejected the machine because
Windows reported `Balanced`, not the predeclared High performance scheme. Therefore
there is no evidence-based answer yet to whether C1 improves the full pipeline or how
it compares with A1 across lengths. This is an inconclusive/blocked result, not a
favorable result.

## Algebra

For query head `h`, `g=floor(h/4)`, query `q_h[d]`, and key projection
`P^K_g[l,d]`, each token score is

`q_h · (z_t P^K_g) / sqrt(64) = (q_h P^{K,T}_g) · z_t / 8`.

The projected query has no token or logical-block index. C1 computes it once per
head per decode invocation in `model_small_project_query_once`, then each logical
block loads `table[logical_block]` with a single-entry lookup and scores its 16 FP16
latent rows after FP32 conversion. Softmax and latent context retain B1 semantics;
value projection remains after latent aggregation. A1 and B1 controls remain intact.
The projection is called inside every timed C1 pipeline invocation.

## Validation

Pilot artifact: `reports/c1_pilot_20260920T082449_625698Z/`.
The independent oracle reconstructs K/V from FP16-rounded latent rows without using
C1's factorization, and reports absolute/relative error and failure locations. The
main harness would apply `atol=5e-3`, probability row-sum `1e-4`, relative denominator
floor `1e-12`; bitwise C1/B1 is diagnostic only. The exact pilot values are in
`correctness.json`, `independent_oracle.json`, and `query_refresh.json`; samples and
telemetry are immutable and checksummed.

## Measurement status

The predeclared main budget was 3 processes × 12 samples per component per length,
3 warmups, balanced rotating variant order, resident buffers, explicit stream/CUDA
events, and projection inside C1 pipeline timing. Components were diagnostic; their
medians would not be summed. This repeated-buffer/warm-cache regime would not imply
cold DRAM or multilayer decoding. No 16K/32K or context-parallel work was attempted.

The pilot is not a matched result. The main attempt built successfully but was stopped
before GPU execution when the newer thermal/power policy was discovered; the guarded
retry stopped before measurement on the `Balanced` power scheme. See
`C1_THERMAL_CORRECTION.md` and the retained `reports/c1_main_*` manifests.

## Evidence and limitations

Historical 32.6% timing is retained with raw process samples but is now explicitly
identified as whole-process execution; it does not establish steady-state kernel
slowdown. The analytical persistent-state ratio remains 16x (1024 vs 64 bytes/token)
and excludes projections, metadata, intermediates and allocator overhead. Repository
file count and evidence-package count are distinct. Historical readiness reported
24 Rust/62 Python tests; this working tree executed 24 Rust/65 Python tests plus the
shell validator.

## Handoff

Before manuscript revision: satisfy the existing AC/High performance guard without
changing thresholds, run the frozen main campaign in a new immutable directory,
run `scripts/analyze_c1.py`, and update this report only from generated raw-artifact
tables. Then separately design R-CTX-PAR or cache-eviction experiments; do not mix
them with C1 attribution. Do not call the manuscript ready based on this blocked
milestone.
