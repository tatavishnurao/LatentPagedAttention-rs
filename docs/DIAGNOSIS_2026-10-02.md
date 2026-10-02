# Feasibility diagnosis — 2026-10-02

Scope: can paged latent decode credibly beat paged full-KV (GQA) decode on an RTX 4060 Laptop (8 GB, CC 8.9, 24 SMs, 32 MiB L2), and what is the shortest evidence-backed path to a defensible claim.

Evidence labels:
- **MEASURED**: read from a committed or untracked artifact, or from a command run in this session. The source is named.
- **SOURCE**: read in code.
- **HYPOTHESIS**: not verified.

GPU work ran under the user's revised thermal policy (2026-10-02):
- hard abort at 91 °C;
- 100 ms telemetry during every run;
- THROTTLED tag if any window sample is >5 % below the modal SM clock, or any thermal or HW-slowdown reason bit is set;
- cooldown to <80 °C (at most 5 min), otherwise the run is tagged hot-start.

E0 ran; E1 is blocked (see "GPU results" below). `validate_release.sh --gpu` was not run.

**4060 idle diagnostic** (MEASURED, `nvidia-smi`, 2026-10-02):
- 77–83 °C at P5/P8, 7–11 W, 210–495 MHz SM.
- Only process: `/Xwayland` (graphics, no compute apps).
- `fan.speed` = N/A (not exposed under WSL).
- Enforced power limit 80 W (max 140 W); `power.limit` = N/A.
- Max SM clock 3105 MHz.
- Idle reason bits 0x1 (GPU idle).

## Executive summary

1. C1 (hoisted K-projection) and R-TABLE are on `origin/main` (7a2a450). The 9-length phase-2 extension is an uncommitted diff, and `reports/phase2`, `reports/c2`, `reports/crossover` and `reports/ablations` are untracked. **CPU CI on `main` has failed since 7a2a450** because of `ruff` errors in the committed C1 scripts.
2. The full-KV baseline (A1) is not a credible paged GQA decode kernel. It uses one Q-head per CTA, loads each K/V tile 4×, uses FP32 CUDA cores only (tensor pipe 0 %), and reduces context in a serial loop over 16 CTAs.
3. Both variants are issue-latency/occupancy-bound, not bandwidth-bound. MEASURED (P05 ncu, 1K): the context grid is 16 CTAs on 24 SMs, with 8.33 % achieved occupancy, 0.8–3 % DRAM throughput and ≥98 % L2 hit rate. The dominant stall is LSU queue throttle (`lg_throttle`), not memory latency.
4. **Phase 2's latency numbers below 32K were measured at idle GPU clocks (MEASURED, E0).** With output on `/mnt/c`, the harness's JSON writes leave the GPU idle long enough to drop to 210 MHz, and its synchronized timing bursts never ramp it back up. On the identical binary at 1890 MHz, the clean 1K run gives **C1/A1 pipeline = 1.13: C1 is slower**. Phase 2's "C1 is 25 % faster at 1K" was a low-clock artifact. At 32K the high-clock samples give C1/A1 ≈ 0.77, but both 32K runs tripped the SW thermal slowdown bit, so that figure is THROTTLED and not a usable ratio.
5. Every benchmark is batch 1, a single layer, with warm buffers that stay L2-resident. Real decode streams all layers' KV plus the weights from DRAM on every step. The current comparison cannot show a bandwidth benefit.
6. The 16× compression comes from a synthetic shape (a 32-dim shared latent replacing 4 KV heads × 64 × K and V, random weights, no RoPE). Plausible MLA-style ratios for 7–8B GQA models are about 1.6–13×, and they depend on rank and on quality.
7. Verdict: a kernel-level win over a competent baseline in a DRAM-bound regime is plausible (~60 %). The full goal of real inference metrics plus real-model compression on a 4060 is unlikely within this repo's scope (~20 %).

## Phase 1 — repository state

### Validation run in this session
| check | result | evidence |
|---|---|---|
| `bash scripts/validate_release.sh` (CPU) | **FAIL** (exit 1) | log: `reports/diagnosis_20261002/validate_cpu.log` |
| pytest | 65 passed | same log |
| `ruff check .` | **380 errors**. Tracked offenders: `scripts/run_c1_milestone.py` (29), `scripts/analyze_c1.py` (16), `tests/test_c1.py` (13). Untracked offenders: `scripts/analyze_phase2.py`, `manuscript/tools/generate_figures.py`, `reports/**/source_snapshot/**` | `uv run ruff check . --output-format concise` |
| cargo fmt / test / clippy | **not reached** (`set -e` stopped at ruff) | — |
| GitHub CPU CI on main | **failure** on 7a2a450 (run 35617221826, 2026-09-21) | `gh run list --branch main` |
| `--gpu` | skipped (thermal gate) | `nvidia-smi` 83 °C |

The `reports/**/source_snapshot` copies are linted because `ruff` has no exclude for `reports/`. Even with an exclude, the tracked C1 scripts fail on their own.

A separate environment hazard (MEASURED, `type cargo`): in this shell, `cargo` resolves to `~/.npm-global/bin/cargo`, which is the `cargo-ai` CLI, not Rust's cargo. `cargo build` fails with "unknown command 'build'". Once ruff passes, the validator's cargo steps would hit the wrong binary unless `~/.cargo/bin` comes first on PATH. All builds in this session used `~/.cargo/bin/cargo`.

### State table
| feature | status | evidence |
|---|---|---|
| Paged latent cache, runtime block table, FP16 storage, masking | done | P15B/phase-2 `correctness.json` (MEASURED); pytest |
| R-TABLE single-entry lookup (A1/B1) | done, on origin/main | `p15b_full_kv_baseline.rs:5-8` (SOURCE); `reports/p15b_rtable/rtable_verdict.json` (MEASURED) |
| A0/B0 full-table controls | done | `reports/p15b_rtable/per_kernel_timings.json` |
| C1 hoisted K-projection | done for 1K–8K on origin/main; 128–32K extension uncommitted | `c1.rs`; `git diff --stat` (+1116 lines) |
| V-side absorption (accumulate latent, one V projection) | done, also in B1 | `p15b_model_profile.rs:215-239` (SOURCE) |
| Accepted C1 timing campaign | **missing** (power-scheme guard) | `docs/C1_REPORT.md`, `reports/c1_main_*` |
| Phase-2 sweep 128–32K | exploratory only: nonstationary ≥4K, no clock telemetry, overwritten failure logs | `reports/phase2/raw/FAILURES.md` |
| Split-K / parallel context | missing | `docs/C2_DESIGN.md` |
| Batch > 1, multi-layer, L2 flush | missing (no harness supports it) | SOURCE: all launches are single-sequence |
| External baseline (FlashInfer / vLLM / Triton) | missing; torch/triton/flashinfer are not installed | site-packages check |
| Nsight profile of current A1/C1 | missing; only historical P05 (full-table lookup, B-variant) | `reports/p05_hardware_attribution/` |
| Real model / quality evaluation | missing | `docs/LIMITATIONS.md` |

### Stale or unsupported claims
- **README** leads with the 16× ratio and the 32.6 % whole-process table. It labels them as process-level but reports no kernel-level numbers at all.
- **`docs/FINAL_REPORT.md`, Results section**, still says the latent path is "approximately 32.6% slower … therefore a memory-versus-compute trade-off". Only the header note withdraws this.
- **`docs/C1_EVIDENCE_LEDGER.md`** says "working-tree commit not yet made". C1 was committed in 7a2a450.
- **`docs/C1_REPORT.md`** predates phase 2. The phase-2 and C2 conclusions exist only in untracked files.
- **`reports/phase2/PHASE2_SUMMARY.md`** attributes the 32K-faster-than-16K result to DVFS. The per-iteration arithmetic below suggests instead that phase 2 ran at low clocks for ≤16K and only 32K ran at boost.
- **`SESSION_SAVE.md`** says C1 was pushed to `origin/main`. That is correct, but local `main` is one commit behind it.

### Branch reconciliation
- Hoisted projection (C1), R-TABLE (A1/B1) and the A0/B0 controls are all in `origin/main`.
- The phase-2 length extension (128/256/512/16K/32K kernels and dispatch) exists only in the working tree, now on `diagnose/feasibility-2026-10-02`.
- Open PRs are Dependabot only (#2, #6).

## Phase 2 — kernel bottlenecks

Canonical shape: Hq=16, Hkv=4, group=4, d=64, dl=32, block=16. All SOURCE unless marked.

| kernel | grid | per-program work | block-table lookup | notes |
|---|---|---|---|---|
| A1 score | (16, N/16) | one q row against a 16×64 FP16 K tile | single-entry `load_tile([1])` | each K tile is fetched by 4 Q-head CTAs |
| B1 score | (16, N/16) | reloads the 32×64 FP32 K-projection (8 KiB) and re-projects q **per block** | single-entry | adds 2·Hq·dl·d·N/16 = 4096·N FLOP and 8 KiB·N of L2 requests |
| C1 projection | (16) | q_h·P^K_g once per head per step (65 K FLOP total) | — | inside every timed pipeline |
| C1 score | (16, N/16) | dot of a 16×32 latent tile with q′ | single-entry | each latent tile is fetched by all 16 Q-head CTAs |
| softmax (shared) | (16) | entire [1, N] row in one tile | — | 128 KiB of f32 per CTA at 32K, likely spills (HYPOTHESIS) |
| A1 context | (16) | serial loop of N/16 iterations: table entry → dependent V-tile load → weighted reduce | per iteration | MEASURED P05: 113 regs, 33 % theoretical / 8.33 % achieved occupancy |
| B1/C1 context | (16) | same loop over 16×32 latent tiles, then one 32×64 V projection | per iteration | V absorption already present |

### Projection
In C1 the K projection is hoisted: it runs once per head per decode step. B1 repeats it per (head, block). That costs +4.2 MFLOP at 1K, which doubles score FLOPs, and adds 8 MiB of L2 requests. This is consistent with B1's score kernel being ~5.6× slower than C1's (223.7 vs 39.9 µs at 1K, MEASURED `reports/phase2/ablation_results.csv`).

### Context parallelism
The context grid is 16·B CTAs regardless of N. Registers limit occupancy to 4 CTAs/SM, so capacity is 4 × 24 = 96 CTAs:

| | CTAs | waves | serial chain (iterations) |
|---|---:|---:|---|
| B=1 | 16 | 0.17 | N/16: 64 at 1K, 512 at 8K, 2048 at 32K |
| B=8 | 128 | 1.3 | same |
| B=32 | 512 | 5.3 | same |

The score kernel has N CTAs and is not occupancy-limited.

### What the context kernel stalls on (MEASURED, P05 ncu, 1K, 1.89 GHz)
| kernel | duration | issue/scheduler | lg_throttle | long_scoreboard | short_scoreboard | wait | tensor pipe |
|---|---:|---:|---:|---:|---:|---:|---:|
| full context | 76.7 µs | 0.11 | 2.44 | 0.52 | 2.09 | 1.78 | 0 % |
| latent context | 90.6 µs | 0.09 | 4.31 | 1.71 | 0.98 | 1.77 | 0 % |
| full score | 50.1 µs | 0.10 | 14.25 | 6.02 | 3.01 | 1.95 | 0 % |
| latent score (B-variant) | 54.0 µs | 0.34 | 0.68 | 1.06 | 1.37 | 0.97 | 0 % |

Stall columns are cycles per issued instruction. These are pre-R-TABLE P05 kernels, so the profile of the current A1/C1 kernels is still missing.

The full context kernel costs ~1.2 µs (≈2,300 cycles) per block iteration. With only 4 warps per SM, each instruction waits ~9 cycles, mostly for the load/store queue (many narrow loads) and for shared-memory reductions, rather than for DRAM. The fix is more warps per SM (split-K), wider vectorized loads, and GQA grouping. This also means the plan's E1 hypothesis ("long-scoreboard dominant") is **refuted** for the P05 kernels.

### Roofline (one layer, B=1)
Bytes are unique cache bytes. The FP32 peak (~14.6 TFLOPS) and DRAM bandwidth (256 GB/s) are spec values, not measured.

| N | full KV bytes | full FLOP | DRAM floor | measured A1 pipeline | latent bytes | latent FLOP | DRAM floor |
|---|---|---|---|---|---|---|---|
| 1K | 1 MiB | 4.2 M | 4.1 µs | 86.5 µs (p15b) / 380 µs (phase 2) | 64 KiB | 2.1 M | 0.26 µs |
| 8K | 8 MiB | 33.6 M | 32.8 µs | 236 µs (p15b) / 2992 µs (phase 2) | 512 KiB | 16.8 M | 2.0 µs |
| 32K | 32 MiB | 134 M | 131 µs | 1353 µs (phase 2) | 2 MiB | 67 M | 8.2 µs |

- Measured pipelines are 7–21× above the DRAM floor even though the data is L2-resident.
- Arithmetic intensity on unique bytes is ~4 FLOP/B for full KV and ~32 FLOP/B for latent. Both are below the FP32 ridge of ~57 FLOP/B, so competent kernels would be memory-bound.
- L2 bandwidth is not measured.
- **Classification: issue-latency/occupancy-bound.** Launch overhead is a secondary term, about 15–18 % of the 1K pipeline (corrected 2026-10-02; see the launch-floor table below).

### Launch overhead (corrected 2026-10-02)
- **Correction:** this section originally said "3–4 launches give a ~45–70 µs floor, roughly 50–80 % of the 1K pipelines". **That was wrong.** Its 14–17 µs per-launch inputs (p15b A1 score at 2K, phase-2 softmax at N=128) were clock-dependent, and the phase-2 ones were taken at idle clocks.
- **Measured at boost** (`reports/diagnosis_20261002/e0b_smoke`, SM 2175–2490 MHz, descriptive):
  - empty event pair 1.36 µs;
  - smallest kernel (C1 projection) 5.12 µs;
  - so A1's 3 launches ≈ 15.4 µs (**15 %** of 104.3 µs) and C1's 4 launches ≈ 20.5 µs (**18 %** of 111.6 µs).
- The event span includes cuTile host dispatch under WSL2 (SOURCE: `c1_rtable.rs` records the event, then calls `async_on`).
- A fused online-softmax split-K kernel removes 2 launches. Whether cuTile-rs supports CUDA graph capture is still unchecked.

## Phase 3 — benchmark validity

- **No discriminating benchmark.** The existing benchmarks are `final_benchmark` (whole-process timing), `p15_kernel_scaling`, `p15b_rtable` (1K–8K), `c1_rtable` phase 2 (128–32K) and `p05` ncu. All are B=1, single layer, with warm resident buffers. Full KV stays ≤32 MiB through 32K, so it is L2-resident.
- **The "near parity" / "C1 faster" hypothesis is confirmed.** The latent advantage comes from less work per serial iteration, not from bytes.
- **Clock validity (new finding).** Per-iteration cost of the A1 context kernel:
  - 0.72 µs at 1K in p15b
  - 4.7–5.0 µs at 1K–8K in phase 2
  - 0.61 µs at 32K in phase 2
  - 1.2 µs at 1K under ncu at 1.89 GHz

  This is **now MEASURED (E0)**: phase 2 ran at idle SM clocks for ≤16K. See the GPU results section.

### 8 GB capacity (analytical)
KV budget is ≈2 GiB after INT4 weights (~4.5–5 GB for 7–8B) and runtime/display reserve. FP16 weights (14–16 GB) do not fit.

| shape | KV/token FP16 | tokens in 2 GiB (FP16 / FP8) |
|---|---|---|
| Llama-3-8B (32 layers, Hkv=8, d=128) | 128 KiB | 16,384 / 32,768 → B=1@16K, B=8@2K, B=32@512 |
| Qwen2.5-7B (28 layers, Hkv=4, d=128) | 56 KiB | 37,449 / 74,898 |
| MLA dl=512+64 rope, 32 layers | 36 KiB (3.6× vs Llama-3-8B, 1.6× vs Qwen2.5-7B) | 58,254 |
| MLA dl=256+64 | 20 KiB (6.4×) | 104,857 |
| MLA dl=128+32 | 10 KiB (12.8×) | 209,715 |

In real multi-layer decode, whole-model KV exceeds the 32 MiB L2 at N ≥ 256 for Llama-3-8B, and ~5 GB of weights also pass through L2 on every step. So KV is DRAM-resident in practice, and the single-layer "L2 crossover" framing is a microbenchmark artifact. To emulate real decode, rotate ≥64 MiB of per-layer buffers or flush L2.

### Statistics
Current state: 2–3 processes, within-process samples treated as descriptive, no CIs, no equivalence test.

Proposed protocol:
- Log SM and memory clocks during runs (`nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu -lms 100`).
- Pre-warm 20 s to P0 and reject samples whose clock is outside ±3 % of the mode. `nvidia-smi -lgc` is probably unavailable under WSL2; verify.
- Interleave variants as now.
- Use ≥5 processes.
- Report bootstrap 95 % CIs on paired process-median ratios.
- Use TOST with a ±5 % margin for any parity claim.

## Phase 4 — feasibility verdict

**(a) Decode latency per token vs a competent paged GQA kernel: MAYBE.** At this synthetic shape, latent decode moves 16× fewer bytes with ½ the FLOPs. So a competent latent kernel (split-K, all heads sharing a latent tile per CTA) should win wherever KV reads dominate: multi-layer, or large B·N. Against today's A1 the comparison means nothing.

**(b) Tokens/s at fixed batch, end to end: UNLIKELY in scope.** The gain is capped by KV's share of bytes per step. For an 8B INT4 model at B=1 that share is ~18 % at 8K and ~30 % at 16K, giving a best case of ~15–28 %. It also needs a real latent model and a decode stack.

**(c) Max batch/context in 8 GB: YES, analytically** (3.6–12.8× more tokens; see table). It is meaningful only with a model whose quality survives that rank.

### Quality caveat
For the compression ratio to survive on real models, three things are needed:
- a learned joint low-rank KV compression at realistic rank (DeepSeek-V2 uses dl=512 for 128 heads; the repo's dl=32 for 4×64 K+V is far more aggressive);
- decoupled RoPE, since RoPE breaks K absorption and the repo has no RoPE;
- absorbed W_UK and W_UV.

The minimal real validation is a GQA→MLA-converted small model (MHA2MLA/TransMLA-style; whether checkpoints exist is a HYPOTHESIS), compared with its source model on perplexity plus one long-context task. DeepSeek-V2-Lite (15.7B) does not fit in 8 GB at FP16.

### Comparators
None of these is installed, and installs need approval.
- FlashInfer paged batch decode: sm89 is supported; whether its MLA decode runs on sm89 is a HYPOTHESIS.
- vLLM decode path (WSL2): only small models fit.
- Triton or PyTorch SDPA paged decode.

### Top 5 blockers (impact × confidence, effort)
1. Uncontrolled clocks and measurement environment (high × high, ~1 day).
2. Non-credible full-KV baseline (high × high, 1–2 days with an external comparator).
3. Serial 16-CTA context reduction in both variants (high × high, ~1 week for split-K in cuTile).
4. Non-representative regime: B=1, single layer, L2-warm (high × high, 3–5 days of harness work).
5. Synthetic latent shape and no quality evidence (high × medium, weeks).

Housekeeping, cheap but needed: red CPU CI (ruff), the uncommitted phase-2 extension, and the stale docs listed above.

## GPU results (2026-10-02)

Artifacts are in `reports/diagnosis_20261002/`. They include `thermal_run.py` (the telemetry wrapper) and `attribute_clocks.py`, plus per-run `samples.jsonl`, `telemetry.csv`, `progress.json`, `run_summary.json`, `clock_attributed_summary.json` and `correctness.json`.

- Binary: `target/release/examples/c1_rtable`, SHA256 `bb92d5c6…14fa`, **identical to the phase-2 binary** (`reports/phase2/raw/c1_binary.sha256`). No source was changed.
- Each run is one process.
- Correctness passed in all 6 E0 runs (16/16 `"pass": true` entries).

### Clock attribution
Each sample row is mapped to the nearest 100 ms telemetry sample, using the 50 ms polling of the `samples.jsonl` row count. Buckets:
- HIGH: ≥95 % of the run's maximum window clock
- LOW: <50 % of it
- MID: in between

### E0 — did phase 2 run at low clocks? **Confirmed.**

| run | output FS | start | timing window | SM clock in window | reason bits | tag |
|---|---|---|---|---|---|---|
| seq1024_run1 | `/mnt/c` | cool (79 °C) | 77 s | mode 210 MHz; 79 % LOW / 21 % at 1890 | 0x1 | LOW-CLOCK (idle P-state; not caught by the THROTTLED rule because the mode itself is low) |
| seq1024_hot1 | ext4 | hot (82 °C) | 0.9 s | 1890; 4 % MID | 0x1 | THROTTLED (clock rule) |
| seq1024_cool2 | ext4 | cool (<80 °C after 162 s) | 0.6 s | 1890 throughout | 0x1 | **clean** |
| seq32768_run1 | `/mnt/c` | cool | 17 s (wall 1352 s) | median 1215; 53 % below mode | 0x1 | THROTTLED (clock rule) |
| seq32768_hot1 | ext4 | hot (83 °C) | 1.2 s | mode 330; no sample ≥95 % of 2505 | **0x21 (SW thermal)** | THROTTLED (thermal) |
| seq32768_cool2 | ext4 | cool (80 °C) | 0.7 s | 49 % at ≥2451, rest LOW; max 87 °C | **0x21 (SW thermal)** | THROTTLED (thermal) |

**Mechanism.** On `/mnt/c` (OneDrive-backed 9P), the harness's oracle and correctness JSON writes take 1–20 minutes: 122 s wall at 1K vs 5 s on ext4, and 1352 s vs 11 s at 32K. The GPU sits idle throughout and drops to 210 MHz. The timing loop synchronizes before every sample, so the duty cycle is low, and on Balanced power the clock usually does not ramp back up.

Evidence that this explains phase 2 (all MEASURED):
- LOW-bucket A1 context at 1K is **321.5 µs**, against phase 2's 322.6 µs.
- HIGH-bucket A1 context is **48.1 µs** in run1 and **43.0 µs** in the clean cool2 run, against p15b's 45.8 µs.
- Phase 2's 32K A1 pipeline (1353 µs) matches run1's HIGH bucket (1361 µs). So phase 2's 32K was the only length measured at boost, which explains why its 32K looked faster than its 16K.

This also explains why the phase-2 32K campaign timed out "between correctness/export and timing" (`reports/phase2/raw/FAILURES.md` item 2).

### Clean result (the only non-THROTTLED run): seq1024_cool2, 1890 MHz, 300 samples per variant/component

| component | A1 full | B1 latent (per-block projection) | C1 latent (hoisted) | C1/A1 |
|---|---:|---:|---:|---:|
| pipeline | 74.75 µs | 74.75 µs | 84.75 µs | **1.13** |
| context | 43.01 µs | 31.74 µs | 31.74 µs | 0.74 |

Per-sample medians for the other components are in `reports/diagnosis_20261002/e0/seq1024_cool2/clock_attributed_summary.json`.

Interpretation:
- At 1K and boost clock, the latent context kernel is about 26 % faster.
- C1's pipeline is still 13 % slower. C1 adds a fourth launch (projection, ~5 µs at boost), which explains only part of the ~10 µs gap. The rest is unattributed: inter-kernel gaps and the softmax/score differences visible in the per-component medians.
- So **hoisting the projection into a separate kernel is not a net win at 1K**. This is one process: descriptive only, no CI. Fusing projection into score is explicitly out of scope; E3 amortizes the projection inside a split-K CTA instead.

### Thermal-condition spread at 1K, same 1890 MHz clock (descriptive; hot1 is THROTTLED)

| | cool2 (clean) | hot1 (THROTTLED, HIGH bucket) | spread |
|---|---:|---:|---:|
| A1 pipeline | 74.75 | 101.22 | +35 % |
| C1 pipeline | 84.75 | 125.92 | +49 % |
| A1 context | 43.01 | 51.04 | +19 % |

The pipeline spreads by more than any single kernel does, at the same SM clock. HYPOTHESIS: the difference is host-side, from WSL2/CPU dispatch state inside the event span. This needs E1/nsys to confirm.

### 32K — THROTTLED only, reported separately and never used in ratios

| run | bucket | A1 pipeline | B1 pipeline | C1 pipeline | C1/A1 (descriptive) |
|---|---|---:|---:|---:|---:|
| seq32768_run1 | HIGH (1890) | 1361.4 (n=28) | 1678.9 | 1040.7 | 0.76 |
| seq32768_cool2 | HIGH (≥2451) | 1000.5 (n=29) | 1153.0 | 774.9 | 0.77 |

**32K is UNRESOLVED.** Both runs are THROTTLED (SW thermal bit, mixed clocks), and each HIGH bucket has ~30 samples from one process. The ~0.77 agreement is descriptive and supports no claim, not even a directional one. It needs either a sustained high-clock setting (Windows "High performance" plan plus NVIDIA "Prefer maximum performance", or locked clocks) or a duty cycle long enough to hold the clock, plus ≥5 processes.

### E1 — focused ncu on current A1/C1 context kernels: **BLOCKED**
- First attempt: no kernels matched, because the real kernel names end in `_entry`.
- Second attempt: `==ERROR== ERR_NVGPUCTRPERM`. The user lacks permission for GPU performance counters (`reports/diagnosis_20261002/e1_blocked/*.err`).
- All 4 attempts were cool-start (79 °C), max 81 °C, and collected no profile data.
- History: P05 hit the same error and was profiled after a WSL restart (`reports/p05_hardware_attribution/ncu_access_probe*.txt`).
- To unblock: enable "Allow access to the GPU performance counters to all users" in the Windows NVIDIA Control Panel (Developer settings), then `wsl --shutdown`. Both are outside the repo and need you to do them.

### Revised conclusions
- Every phase-2 C1/full ratio below 32K is void: those lengths ran at idle clocks.
- At a clean boost clock at 1K, C1 is **slower** than full-KV at the pipeline level (1.13×), despite a 26 % faster context kernel.
- The only clean data point is 1K: context 0.74×, pipeline 1.13×. **32K is unresolved** (THROTTLED only), and 8K has no clean data. No trend with N is established.
- The measurement environment is now the first fix: ext4 output, a sustained-clock setting, a telemetry wrapper, ≥5 processes. Kernel topology (split-K, E3) is the second. Launch overhead (~15–18 % at 1K) is third.

## Harness fix, P0 verdict and E0b (2026-10-02, second pass)

### Harness changes (`crates/plkv-kernels/examples/c1_rtable.rs`; uncommitted)
1. **All file output happens after the timing loop.** Samples are buffered in memory. `correctness.json`, `query_refresh.json`, `timing_meta.json`, `samples.jsonl` and the oracle export are written after timing. A correctness failure still aborts before timing and writes its JSON.
2. **The output directory is guarded.** Paths under `/mnt` are refused unless `--allow-slow-output` is given. The default is now `/tmp/plkv_c1_native`.
3. **No per-sample I/O.** The old per-sample `writeln!` + `flush()` to `/mnt/c` is gone. The `synchronize` before each event pair stays: it is needed to isolate samples, and its cost is now measured by the `empty` phase.
4. **Clock warm-up and keep-alive.**
   - Warm-up: `--clock-warm-ms` (default 2000) of back-to-back untimed pipelines. The reference clock is the mode of the second half of the warm-up NVML reads.
   - Keep-alive: before each sample, if the SM clock is below 95 % of the reference, run untimed pipeline bursts for up to `--keepalive-max-ms`. **Default 0 (off)**: E0b showed it cannot override temperature-driven DVFS and only adds heat.
5. **Per-sample NVML** (in-process `libnvidia-ml.so.1` via `libloading`, which was already in the lockfile, so nothing was downloaded): SM clock before and after the sample, memory clock, temperature and event-reason bits. If any reading is ≥91 °C, the run writes partial results and exits with code 3.
6. **Variant selection and interleaving.** `--variants` selects the variants (default A1,B1,C1; E0b used A1,C1). Order rotates by `(iteration + process + offset)`, and pairs are adjacent within each (iteration, component).
7. **New `empty` phase:** an event pair with no launch, used to measure the floor.

**Builds and lint:**
- `~/.cargo/bin/cargo build --offline --release -p plkv-kernels --features gpu-cutile --example c1_rtable` succeeds. `clippy` reports only the two pre-existing dead-code warnings in this file.
- `Cargo.lock` gains one line (`libloading` added to plkv-kernels).
- Binary SHA256: `389b4a8d…72a2`.

### Does P0's 0.137216 / 0.139184 ms have the idle-clock defect? **No** (MEASURED and SOURCE)
- **P0 harness (`examples/p0_gpu_baseline.rs`):**
  - samples collect in a `Vec` (lines 345–364), and the summary and oracle JSON are written at lines 478–500, after both timing phases;
  - there is no per-sample I/O, and correctness is checked in memory before warm-up;
  - it does synchronize the stream before each event pair (lines 259–261), the same as C1;
  - it logged no clocks.
- **The P05 clean-display cohort (paper value B/A = 1.008662129) did log clocks.** In every attempt's `gpu_telemetry.csv`, all samples between `gpu_timing_start_unix_ms` and `host_timing_end_unix_ms` are **P0 at 1890 MHz** (with 2385–2685 MHz in the last samples, during host timing). Accepted attempts:

  | attempt | B/A |
  |---|---:|
  | 01 | 1.0076 |
  | 02 | 1.0075 |
  | 04 | 1.0131 |
  | 08 | 1.0130 |
  | 09 | 1.0087 |

  Source: `reports/p05_clean_display_replications/attempt_*/`.
- **P0's medians are consistent with 1890 MHz, but P0's clock is inferred, not logged.** A = 137.2 µs, which matches p15b A0 at 138.2 µs and the cohort's A of 133–136 µs. Under the defect, A would be about 7× larger.
- **Verdict:**
  - **Status: CONSISTENT, CLOCKS INFERRED.** The P05 cohort (B/A = 1.0087) was measured at a logged, steady 1890 MHz P0 clock. P0's own 0.137216 / 0.139184 ms has no clock log; 1890 MHz is inferred because its medians match the cohort and p15b A0, and are about 7× too small to be idle-clock values.
  - The idle-clock defect is specific to the C1/phase-2 harness, caused by its per-sample flushes and pre-timing JSON writes on `/mnt/c`.
  - Remaining caveats on the paper's numbers: they are A0/B0 kernels (pre-R-TABLE), B = 1, single layer, and P0 had a single process with no clock log.

### E0b: **STOPPED, no clean data** (MEASURED; `reports/diagnosis_20261002/e0b/`)
Plan: 1K and 8K, 5 processes each, interleaved, A1 vs C1.

What happened:
- **No cool start was reachable.** Process `seq1024_p0` waited the full 300 s and started hot at 80 °C. The GPU's idle floor in this session is 77–83 °C at 7 W (idle diagnostic above), so the <80 °C gate is effectively out of reach.
- **The clocks could not be held:**
  - SM clock pre-reads were spread from 795 to 1890 MHz (mode 1890, 3157/4000 reads below 95 % of mode);
  - **memory clock moved between 810, 6001 and 8001 MHz**;
  - temperatures were 82–87 °C, with **SW thermal slowdown (0x20)** set in both NVML and external telemetry.
- **The keep-alive failed** to restore the 2355 MHz reference clock on 1,899 of 2,000 samples. It launched 385 k untimed pipelines and added heat. This is temperature-driven DVFS, which a keep-alive cannot override.
- I stopped the campaign after p0. The remaining nine processes would all have been hot-start and THROTTLED, about 70 minutes of extra heat for no usable ratio. `seq8192_p0` was still in cooldown and never launched.

**p0 (THROTTLED, hot-start), descriptive only. Not a ratio and not in any CI.**

| component | A1 | C1 | C1/A1 |
|---|---:|---:|---:|
| pipeline | 149.5 µs | 161.3 µs | 1.08 |
| score | 33.3 µs | 28.7 µs | 0.86 |
| softmax | 12.3 µs | 15.4 µs | 1.25 |
| context | 88.1 µs | 67.1 µs | 0.76 |

The clock-matched subset (SM 1890 MHz before and after, memory 8001 MHz) has only ~15 samples per cell, and its medians contradict the full-run medians (for example, C1 context 98.3 vs A1 78.6). Nothing is concluded from it.

**Launch-count floor** (descriptive, both runs non-clean):

| run | empty event | smallest kernel (C1 projection, 16 CTAs) | A1 floor (3 launches) | C1 floor (4 launches) |
|---|---:|---:|---:|---:|
| `e0b_smoke` (warm, SM 2175–2490 MHz) | 1.36 µs | 5.12 µs | 15.4 µs (15 % of 104.3) | 20.5 µs (18 % of 111.6) |
| `e0b/seq1024_p0` (hot, mixed clocks) | 2.05 µs | 17.41 µs | 52.2 µs (35 %) | 69.6 µs (43 %) |

The smoke run used the 500 ms keep-alive and was built before the memory-clock field was added.

Interpretation: the irreducible per-launch cost on this WSL2 + cuTile stack is about 5 µs at boost clock. The 14–17 µs "floors" seen earlier are clock-dependent, not pure host overhead. C1's extra launch costs about 5 µs at boost, about 25 % of the A1/C1 pipeline gap at 1K.

### What would unblock E0b (each needs your decision; all are outside the repo)
1. **Lock clocks at a sustainable level** from an elevated Windows prompt, e.g. `nvidia-smi -lgc 1500,1500` and `nvidia-smi -lmc 8001,8001`, then `nvidia-smi -rgc` / `-rmc` afterwards. A locked clock below the thermal knee makes hot and cool runs comparable. Under WSL, `-lgc` is expected to be refused; untested.
2. **Fix the thermal floor.** 77–83 °C idle at 7 W is abnormal. Check fans with the OEM tool, select the vendor "performance" fan profile, raise the laptop off the surface, and set the Windows High performance plan.
3. Alternatively, run E0b/E3 on a desktop or cloud GPU (E7).

### E3 design
See [`docs/E3_SPLITK_DESIGN.md`](E3_SPLITK_DESIGN.md) for the design: split-K with an in-CTA projection amortized over R ≥ 16 blocks (overhead 2/R of FLOPs), 2 launches per step, per-N bytes/FLOPs tables, and pre-declared success and stop criteria. Not implemented.

## Follow-ups (2026-10-02, third pass; no GPU campaigns on the laptop)

- **Tensor-core mapping (MEASURED, compile-only, CPU):** cuTile 0.2 lowers FP16 `mma` for the latent score GEMM [16×32]×[32×T], T = 64…512, to `HMMA.16816.F32` on sm_89. The whole GEMM is on tensor cores: T/16 HMMA per warp with 4 warps = T/4 per CTA, as required. All 48 existing kernel functions contain 0 HMMA. Without hints the probe's FP16 loads were scalar `LDG.E.U16`. **P1:** `CompileOptions::max_divisibility(16)` or runtime-style `spec_args` makes them `LDG.E.128`, with HMMA unchanged. Paged gathers are untested. See [`MMA_PROBE_PLAN.md`](MMA_PROBE_PLAN.md).
- **Portable E0b runner** for a native-Linux sm_89 host (e.g. a cloud L4): `scripts/e0b/run_e0b_portable.sh`, with requirements and a checklist in [`E0B_CLOUD_RUNBOOK.md`](E0B_CLOUD_RUNBOOK.md). Its refusal paths and individual checks were tested here; the full native run is untested.
- **Driver/toolkit fact (MEASURED):** this laptop's driver (595.97) reports CUDA 13.2 and runs the CUDA 13.3 toolkit correctly. tileiras emits sm_89 cubins ahead of time, so the driver does not need to report 13.3.

## Experiment plan (each needs approval)

| # | hypothesis | change | measurement | success / failure | cost |
|---|---|---|---|---|---|
| E0 | Phase 2 ran at low clocks for ≤16K | none (telemetry wrapper only) | per-sample clocks plus event times | **DONE: CONFIRMED.** LOW-bucket 1K A1 context 321.5 µs (phase 2: 322.6); 1890 MHz: 43.0–48.1 µs (p15b: 45.8) | done |
| E1 | Current A1/C1 context is issue-latency-bound (lg_throttle/short_scoreboard), as in P05 | none: focused `ncu --metrics`, `-c 1`, cool-start only | stalls, occupancy, L2/DRAM split | **BLOCKED: ERR_NVGPUCTRPERM.** Needs Windows counter permission plus `wsl --shutdown` | <10 min GPU once unblocked |
| E2 | A1 is far from state of the art | FlashInfer paged decode at the same shape, in a separate venv | CUDA-event kernel time | FlashInfer ≥5× faster than A1 → withdraw all A1-relative speedup language | install plus 30 min |
| E3 | Split-K plus GQA grouping fixes both variants | new kernels on a `diagnose/split-k` branch: S splits plus a reduce; all 4 Q heads per KV tile for full, all 16 heads per latent tile for latent | pipeline time, N = 1K–32K | ≥3× context speedup for both variants; only then is latent/full meaningful | ~1 week |
| E4 | Compression pays off only when DRAM-bound | harness with B ∈ {1, 8, 32} and rotating per-layer buffers ≥64 MiB, or an L2 flush | per-step time vs bytes | latent/full moves toward the byte ratio as B·N·L grows; no separation = failure | 3–5 days |
| E5 | Fusion/graphs remove the launch floor | one fused online-softmax split-K kernel; check cuTile graph capture | pipeline time at N ≤ 2K | floor drops from ~15–20 µs (3–4 launches at boost) to ≤10 µs (2 launches) | 3–5 days |
| E6 | Quantized latent compounds the win | FP8/INT8 latent storage | bytes and error | deferred until E3/E4 show a DRAM-bound regime | later |
| E7 | Results are 4060-specific | cloud L4/A100 rerun of E3/E4 | same | deferred | later |

**E0 is done.** Next, in order:
- **E0b:** fix the measurement environment. Write output to ext4 (or `--output-dir` under `/home`); set a sustained-clock mode, which is a Windows-side setting you would make; replicate 1K/8K/32K with ≥5 cool-start processes and bootstrap CIs over the non-THROTTLED runs. Under 10 min of GPU time.
- **E1** once counters are enabled.
- **E2:** the FlashInfer comparator, which needs install approval.
- **E5a:** fuse the C1 projection into the score kernel. It is the cheapest code change and targets the measured 1K regression.
- **E3/E4** after that.
