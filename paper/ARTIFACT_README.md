# Artifact README: corrections ledger, evidence and reproducibility

Companion to `paper/paper.md`. **All results: laptop RTX 4060, WSL2, not a cloud result.** Sections A and B were Appendices A and B of the paper until commit `3cb79e2`; they moved here on 2026-11-03 when the paper was redrafted for submission. The `[M:key]`/`[S:key]` tags in `paper/paper.md` refer to the evidence keys in section B (the submission copy built by `scripts/paper/assemble_paper.py` strips them).

**Names.** The paper uses new names; data files, code and older documents use the internal ones.

| Paper | Internal |
|---|---|
| GQA-pad, "GQA, padded (full KV)" | `a3pad`, padded A3 |
| GQA-unpad | `a3`, unpadded A3 |
| Latent, "Latent (hi/lo)" | `c3_accurate`, C3 accurate |
| Latent-fast, "Latent (FP16 q′)" | `c3_fast`, C3 fast |
| Single | E3 (`e3_timing`) |
| Step, e.g. Step B8·N8192 | E4 (`e4_timing`), e.g. `B8_N8192` |
| fix 1 / fix 2 | harness `pre-2026-11-01` / `e3e4-rewarm-saturating-20ms-2026-11-01` |

## A. Corrections ledger

Claims made earlier in this repository, in the paper brief or in a draft that the evidence does not support, and what replaces them. Nothing was deleted from history; documents carry dated notes.

| # | Earlier claim (where) | Replacement | Evidence |
|---|---|---|---|
| A1 | "80 % of E4 layers ran at 210 MHz" in the VOID dry run (`docs/LOCAL_4060_RESULTS.md`, commit `507dea2` message, PR #10 description) | 73 % of E4 layer samples at exactly 210 MHz; in-band fraction 0.09. Dated correction note added to the results document; commit message and PR text not rewritten. | [M:void] |
| A2 | "~16x slower" as a general idle-clock factor (brief) | "up to about 16 times in one process" (E4 B1_N1024 dry run, 378 vs 24 µs); the 2026-10-02 slow-write case gave about 7.5 times (321.5 vs 43.0 µs). The two are kept separate. | [M:void], [M:t1] |
| A3 | Accurate mode costs "+50 % score MMAs" (brief) | Accurate mode doubles the score MMAs (8 to 16 per CTA per step); total MMAs per step +50 %. | [M:sass] |
| A4 | cuTile gives HMMA "only with >= 16 rows and >= 512 outputs" (brief) | "Only 16-row tiles with >= 512 outputs got HMMA in our probes"; only 4 and 16 rows and 256 and 512 outputs were tested. | [M:sass] |
| A5 | Slow output writes as a cause of 210 MHz samples in the current campaign (implied by the brief's contribution 1) | The slow-write trap belongs to the older A1/C1 harness (2026-10-02) only; E3/E4 timing never wrote output during timing. | [M:t1], [S:harness] |
| A6 | The ~20 ms NVML latency, cited from an interactive measurement whose output was not saved | Re-measured into a committed file as a separate micro-measurement (median 20.8 ms, max 29.0 ms, idle, driver 617.42). | [M:nvml] |
| A7 | Literature review names works without references (`docs/01_literature_review.md`) | Only works read at their source are cited (arXiv pages, Crossref, and NVIDIA and PyTorch pages); as of 2026-11-02 no citation placeholder remains. | this paper |
| A8 | "Above L2" labels for E4 configurations in the campaign's block reports | E4 flushes L2 before each layer; L2 size is not an axis of E4 (already corrected in `docs/LOCAL_4060_RESULTS.md`). | [S:e4] |
| A9 | Accurate mode's run-time cost "is not measured" (`docs/E3_RESULTS.md`, 2026-10-03, written before any timing) | Measured at one size, E3 N=8192, warm L2: C3 accurate / C3 fast 1.052 [0.999, 1.089] (primary) and 1.017 [0.974, 1.057] (sensitivity run); CIs include 1.0. | [M:fastacc] |
| A10 | Abstract and contribution 1 of the first draft (merged in PR #11) listed slow output writes and the telemetry call together as causes of the idle-clock samples, so the 73 % could be read as caused by output writes | The 73 % at 210 MHz comes from the telemetry call in the E3/E4 harness only; slow output writes are named separately as an older A1/C1-harness defect, as in Section 3. | [M:void], [M:t1] |
| A11 | Limitations cited "our roofline figure (not shown)" | Reference to a figure the paper does not contain removed; the 256 GB/s value is stated as a nominal specification, not a measurement. | this paper |
| A12 | Contribution 1 claimed "the protocol that catches" the clock failure mode, and Related work said clock/DVFS literature was unsurveyed | GPU clock effects and fixed-clock protocols are prior art [9]-[13]; contribution 1 is narrowed to an exclusion protocol for hosts where clocks cannot be locked, plus three specifics we have not found reported in a bounded search. | Section 7 |
| A13 | The 2026-11-03 redraft (Table 1 and the Benchmarks paragraph) said Step times "all 4 layers" | Step uses `max(4, ceil(65536 / (B*N)))` layers (SOURCE `crates/plkv-kernels/src/e4_config.rs`; per configuration in `primary_merged_analysis.json`): 4 at every claim-eligible configuration, 8, 16 or 64 at the small-B·N descriptive ones. No ratio changes. | [M:timing], [S:e4] |
| A14 | The 2026-11-03 redraft wrote time ratios as speed ("13.5 % faster" for 0.865, "10.7 % faster" for 0.893, "about 3.6 % faster" for 0.964) | They are reductions in time per step: "13.5 % less time" etc. As throughput, 0.865 would be about 15.6 % higher; the paper uses the time wording throughout so the percentages match the tables. | [M:timing], [M:sens] |

## B. Evidence and reproducibility

**Evidence keys.** All paths are relative to the repository root.

| Key | Files |
|---|---|
| M:timing | `reports/local_4060_20261101/primary_merged_analysis.json` (from runs `local4060_b1r_e3_20261101_051132`, `local4060_e4n4096_20261101_054159`, `local4060_e4n8192_20261101_054853`, `local4060_e4n1024v2_20261101_055641`) |
| M:sens | `reports/local_4060_20261101/runs/local4060_e3a3sens_20261101_063414/variants_analysis.json` |
| M:fastacc | `reports/local_4060_20261101/runs/local4060_b1r_e3_20261101_051132/variants_analysis_vs_c3_fast.json` and `.../local4060_e3a3sens_20261101_063414/variants_analysis_vs_c3_fast.json` (secondary analysis: `scripts/e0b/analyze_variants.py <run>/e3e4 --baseline c3_fast` on the raw samples in `raw/<run>.tar.xz`; the same command without `--baseline` reproduces the committed primary analysis exactly) |
| M:prelim | `reports/local_4060_20261101/runs/local4060_b1_e3_20261101_043331/` |
| M:void | `reports/local_4060_20261101/instrument_checks/e4_B1_N1024_dryrun_VOID/` |
| M:checks | `reports/local_4060_20261101/instrument_checks/`; `reports/local_4060_20261101/runs/local4060_b2r_e4n1024_20261101_051528/` |
| M:ncu | `reports/ncu_20261103/` (`PLAN.md` declared before the attempt; `README.md`: `ERR_NVGPUCTRPERM`, no profile) |
| M:dev | `reports/device_query_20261103/device_query.json` (driver-reported attributes via `cuDeviceGetAttribute`; peak bandwidth computed from them, not measured) |
| M:nvml | `reports/nvml_latency_20261101/nvml_latency.json` (separate micro-measurement, not campaign data) |
| M:t1 | `reports/diagnosis_20261002/e0/` (`seq1024_run1`, `seq1024_cool2`); `docs/DIAGNOSIS_2026-10-02.md` (Trap 1: one 1K run took 122 s of wall time against 5 s with output on ext4, modal SM clock 210 MHz, 79 % of samples low-clock; A1 context kernel 321.5 µs at idle clock against 43.0 µs at boost) |
| M:sass | `reports/e3_splitk_20261002/sass/`, `sass_a3_equal_treatment/`, `sass_c3_modes/`; `docs/E3_RESULTS.md`; `docs/E3_SPLITK_DESIGN.md` (b) for the 256-output tile |
| M:scale | `reports/e3_splitk_20261002/scale_c3_modes/scale_suite.log` |
| M:emu | `reports/e3_splitk_20261002/scale/emulate_c3_numerics.out` |
| S:rules | `docs/E0B_CLOUD_RUNBOOK.md` section 4; `docs/LOCAL_4060_PLAN.md` and its addenda |
| S:kern | `crates/plkv-kernels/src/cutile/e3_splitk.rs`; `crates/plkv-kernels/src/splitk_plan.rs`; `docs/E3_SPLITK_DESIGN.md` |
| S:e4 | `crates/plkv-kernels/src/e4_config.rs`; `crates/plkv-kernels/examples/e3_splitk_gpu.rs` (`launch_async`, `L2_FLUSH_WORDS`) |
| S:harness | `crates/plkv-kernels/examples/e3_splitk_gpu.rs`, `crates/plkv-kernels/examples/timing_common/mod.rs` |

**Regenerating tables, figures and checks (CPU only).** Figures use throwaway environments (`uv run --no-project --with ...`); the project lockfile is untouched.

```
python3 scripts/paper/make_tables.py                 # paper/generated/*.md, numbers.json, summary.json
python3 scripts/paper/check_numbers.py               # every number in paper/*.md traceable
python3 scripts/paper/assemble_paper.py              # paper/build/paper.md (tags) + paper_submission.md (no tags); rough page estimate
F=reports/local_4060_20261101/figures
uv run --no-project --with playwright==1.48.0 python scripts/paper/render_dataflow.py --out $F/fig_dataflow.png
uv run --no-project --with matplotlib python scripts/paper/fig_clock_traces.py --out $F/fig_clock_traces.png
uv run --no-project --with matplotlib python scripts/paper/fig4_numerics.py --out $F/fig_numerics.png
uv run --no-project --with matplotlib python scripts/paper/fig3_regime_map.py \
  reports/local_4060_20261101/primary_merged_analysis.json --l2-mib 32 \
  --l2-label $'L2 32 MiB: reference only,\nStep flushes L2 before each layer' --out $F/fig_regime_map.png
uv run --no-project --with matplotlib python scripts/paper/fig2_roofline.py \
  reports/local_4060_20261101/primary_merged_analysis.json --peak-gbps 256.032 \\
  --title "Step, cold L2; floor = 256 GB/s peak, driver-reported, not measured" --out $F/fig_roofline.png
python3 scripts/paper/device_query.py --out <dir>/device_query.json      # driver attributes only, no context, no kernel
python3 scripts/paper/nvml_latency.py --out <dir>/nvml_latency.json   # NVML calls only, no kernels
```

**GPU campaign (not re-run for this paper).** The command and per-run commits are in `reports/local_4060_20261101/README.md`; primary E4 runs used harness version `e3e4-rewarm-saturating-20ms-2026-11-01` at commit `c3c900c`, primary E3 used `pre-2026-11-01` (binary built at `92e1411`), and the sensitivity run `...+a3unpad` at `2ffd88e`.

**Hashes (SHA-256).**

```
d3ef6ca3984183f72c5b2d8457b5593ae27885ce8a714781848b39ce463e5633  reports/local_4060_20261101/primary_merged_analysis.json
8bb01bd5ea170bd6dffcca823f3329ca7d105d537c9f327fe4f4100e481ef182  reports/nvml_latency_20261101/nvml_latency.json
cc132ffefd418620368d35e27168e2f1830be98d610d1d5b1de7354dca8125d1  reports/e3_splitk_20261002/scale_c3_modes/scale_suite.log
6b777f94ef0c17e9f828af7d3aaead108c10292d9fb2b423befe6108d6e56e00  reports/e3_splitk_20261002/scale/emulate_c3_numerics.out
```
