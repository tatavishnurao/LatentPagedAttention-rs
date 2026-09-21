# C1 bounded protocol (written before pilot)

**Thermal policy superseded before main GPU execution:** see
[C1_THERMAL_CORRECTION.md](C1_THERMAL_CORRECTION.md). The older policy below
applies only to the retained pilot; main runs must pass the stricter current
73 C idle / 82 C run policy. Main sample budget is unchanged.

Scope: synthetic latent-KV GQA; Hq=16, Hkv=4, group=4, D=64,
L=32, block=16. Lengths 1024/2048/4096/8192 only. Mapping
(17*l+11) mod B. A1 and B1 use unchanged R-TABLE kernels. C1 uses
existing C's FP32 projection and score arithmetic with single-entry paging,
and unchanged B1 softmax/context. No context parallelization.

Pilot: one independent process at 1K, 1 warmup and 3 samples/phase.
If successful and feasible, freeze main budget to 3 independent processes
per length, 3 warmups and 12 samples per phase per process. Pilot excluded
from main analysis. No outcome-dependent adjustment. An infeasible pilot
must be recorded and a new protocol revision written before main execution.

Each process has resident, reusable inputs/outputs. CUDA events on an explicit
stream bracket either the complete pipeline or one component. Synchronize
before start and stop event after launches. C1 projection is INSIDE every
pipeline invocation, not amortized or precomputed. Components are diagnostic;
do not sum medians. Host dispatch gaps within the event span are included.
JIT, CPU/GPU correctness, readbacks, input/output dumps and warmup are excluded.
For each iteration and component, rotate A1/B1/C1 by (iteration+process)%3;
12 samples give four occurrences of each starting variant. Process order is
length-major, process indices 0,1,2. Pipeline precedes diagnostic components.
This is repeated-buffer/warm-cache measurement, not cold DRAM or multilayer
serving. No bandwidth or measured-DRAM claims follow from requested bytes.

Thermal policy unchanged from run_p15b_rtable.py: pre-run >=84 C waits up to
360 seconds for <=82 C, otherwise skip; during >=86 C kills process; between
processes wait up to 300 seconds for <=78 C. Never modify clocks/power/thresholds.
Poll telemetry every 0.5 seconds; driver telemetry has finite sampling resolution.
Unavailable management fields are retained as unavailable, not fabricated.
Abort whole campaign on technical/thermal failure, preserve all partial files.
A later retry is a new directory/attempt, never overwrite failed evidence.
All samples from successful, correct processes retained (no outlier trimming).
Process, not sample, is independent replication unit; report descriptive medians,
IQR, process ranges and paired process-median ratios, no equivalence inference.

Correctness: finite scores/probabilities/context; probabilities in [0,1] and
row sum absolute error <=1e-4; per-element absolute error <=5e-3 against Rust
FP32 scalar latent reference and independent FP64 reconstructed-K/V oracle,
and C1 vs B1. Relative errors reported with denominator floor 1e-12, not used
for near-zero rejection. Existing 5e-3 repair bound accommodates FP32 tree vs
serial reduction over sequence; observed maxima must also be reported.
C1/B1 bitwise agreement diagnostic only (projection launch can alter compiler
scheduling); full-KV/latent bitwise equality is not required. Paging checked by
nonidentity permutation against logical-order oracle. Projection FP32 storage
adds 16*32*4=2048 bytes per C1 buffer. Harness allocates equal projected workspace
for all variants for simplicity (6144 total); only C1 accesses it. Common score
and probability workspaces are 2*16*N*4 bytes, context 4096 bytes per variant.

Provenance: base commit plus dirty diff and complete tracked/source-file snapshots
with SHA256 (new files included), executable hash, tool versions and command.
Unique run directories; samples flushed each row, correctness before timing.
Checksums seal each completed/aborted campaign. Main analysis must validate
expected sample counts and successful status; preserve excluded attempts.
