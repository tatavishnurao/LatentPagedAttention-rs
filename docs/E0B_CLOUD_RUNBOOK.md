# E0b cloud runbook (native Linux, sm_89, e.g. NVIDIA L4)

Purpose: run E0b (A1 full-KV vs C1 latent, N = 1K / 8K / 32K, ≥5 cool-start processes per length) on a machine that can hold its clocks. Every sample records its SM and memory clock. The RTX 4060 laptop could not do this: it idles at 77–83 °C, and E0b was stopped there (`docs/DIAGNOSIS_2026-10-02.md`).

Entry point: `scripts/e0b/run_e0b_portable.sh`.

## 1. Requirements

**Validated** means it was used to produce the repository's evidence. **Required** means the scripts check it and refuse to run without it.

| item | requirement | status |
|---|---|---|
| GPU | compute capability **8.9** (Ada: L4, RTX 40xx). Kernels and the MMA probe target `sm_89`. | required (preflight fails otherwise) |
| OS | native Linux x86_64. Not WSL; not a `/mnt` or network filesystem for the repo or the results. | required |
| NVIDIA driver | must load `sm_89` cubins from tileiras 13.3. Validated: **595.97**, which `nvidia-smi` reports as **CUDA 13.2**. | validated; the preflight warns below CUDA 13.2 |
| NVML | `libnvidia-ml.so.1`, shipped with the driver and loadable by the dynamic loader (checked with `ctypes.CDLL`), used for per-sample clocks | required |
| CUDA toolkit | **13.3** (validated `nvcc` V13.3.73), resolved from `CUDA_TOOLKIT_PATH`, else from `nvcc` on `PATH`, else `/usr/local/cuda-13.3`, else `/usr/local/cuda` | validated; other versions warn |
| tileiras | `$CUDA_TOOLKIT_PATH/bin/tileiras` (ships with CUDA 13.3). cuTile invokes it at runtime to compile kernels. Override with `CUTILE_TILEIRAS_PATH`. | required |
| cuobjdump | `$CUDA_TOOLKIT_PATH/bin/cuobjdump`, for the MMA probe | required |
| cuTile | crates `cutile`, `cutile-compiler`, `cuda-bindings`, `cuda-core`, `cuda-async` **= 0.2.0**, pinned in `Cargo.lock`, built with `--locked` | required |
| Rust | **1.96.0** via rustup (`rust-toolchain.toml` pins it, with `clippy` and `rustfmt`) | required |
| libclang | for bindgen in `cuda-bindings`. Validated LLVM **18**. Resolved from `LIBCLANG_PATH`, else `llvm-config-18 --libdir` / `llvm-config --libdir`, else `/usr/lib/llvm-18/lib`. | required |
| Python | 3.x standard library only (validated 3.12). No NumPy needed for E0b. | required |
| Network | crates.io access for the first `cargo build`, or a pre-populated `~/.cargo/registry` | first build only |
| Disk | about 3 GB (`target/` plus results). Oracle dumps are written only for process 0 (32K ≈ 43 MB). | — |
| Profiling (E1, optional) | `ncu` needs counter access: root, or the driver option `NVreg_RestrictProfilingToAdminUsers=0` | not used by E0b |

**PATH hazard seen on the laptop:** an npm package installed a `cargo` shim (the cargo-ai CLI). The runner puts `${CARGO_HOME:-$HOME/.cargo}/bin` first on `PATH` and checks that `cargo --version` prints `cargo …`. It honours `CARGO_TARGET_DIR`.

**L4 vs RTX 4060 Laptop.** These are vendor specs, not measured; confirm them from `environment.txt` after the run.

| | RTX 4060 Laptop | L4 |
|---|---|---|
| architecture | AD107, sm_89 | AD104, sm_89 |
| SMs | 24 | 58 |
| L2 | 32 MiB | 48 MB |
| DRAM bandwidth | ~256 GB/s | ~300 GB/s |
| TDP | 80 W enforced here | 72 W |

Consequences:
- L4 results are **not** a 4060 result. They test the measurement question (does C1/A1 hold under stable clocks?), not 4060-specific performance.
- The larger L2 moves the full-KV L2-residency boundary from 32K (4060) to about 48K, so 32K full-KV stays L2-resident on the L4.
- L4 is power-capped at 72 W. Expect `sw_power_cap` (0x4) reason bits. They are recorded but are **not** thermal bits, so they do not trigger THROTTLED; SM-clock drops of more than 5 % do.

## 2. Setup checklist

1. Provision an L4 VM (or other sm_89 host) with the NVIDIA driver installed. Confirm `nvidia-smi` shows the GPU with compute capability 8.9.
2. Install CUDA toolkit 13.3, then either put its `bin/` on `PATH` or export `CUDA_TOOLKIT_PATH`. Check `ls "$CUDA_TOOLKIT_PATH"/bin/{nvcc,tileiras,cuobjdump}`.
3. Install LLVM/libclang 18 (for example `apt install libclang-18-dev llvm-18`). If `llvm-config-18` is not on `PATH`, export `LIBCLANG_PATH`.
4. Install rustup. `rust-toolchain.toml` selects 1.96.0 automatically on first `cargo` use.
5. Clone the repository onto the VM's local disk, not a mounted share, and check out the commit to test (record it; the runner logs `git_head` and dirty state).
6. Stop other GPU work. `nvidia-smi --query-compute-apps=pid,name --format=csv` should list nothing.
7. Optional, as root: enable persistence mode (`nvidia-smi -pm 1`) so the driver doesn't reinitialize between processes.
8. Optional second condition: a locked SM clock (`--lock-sm-mhz <MHz>`, root only; the runner resets clocks on exit). Run the default unlocked campaign first.
9. Preflight:
   ```bash
   bash scripts/e0b/run_e0b_portable.sh --out-dir ~/e0b_l4_$(date -u +%Y%m%dT%H%M%SZ) --preflight-only
   ```
   It must print `PREFLIGHT_OK`. The preflight creates the output directory and writes `environment.txt` there, so use a new `--out-dir` (or delete that directory) for the full run.
10. Dry run, with a new `--out-dir`. It runs every stage (build, MMA probe, one 1K process, analysis, packaging), takes a few minutes, and fails fast on a broken environment:
    ```bash
    bash scripts/e0b/run_e0b_portable.sh --out-dir ~/e0b_l4_dry_<stamp> --dry-run
    ```
    It must end with `DRY_RUN_OK`. It checks correctness, per-sample NVML clocks, telemetry, HMMA in the FP16 probe kernels, and that a 1K analysis row exists. A dry run is never evidence.
11. Full run, with a new `--out-dir`:
    ```bash
    bash scripts/e0b/run_e0b_portable.sh --out-dir ~/e0b_l4_<stamp>
    ```
    It builds, runs the CPU-only MMA probe, collects ≥5 cool-start processes per length (at most 8 attempts each), analyzes, and writes `~/e0b_l4_<stamp>.tar.gz` without oracle dumps.
12. Bring back the tarball. In this repository it goes under `reports/e0b_l4_<stamp>/`, and the results get added to the diagnosis doc only after you approve.

**Expected duration.** HYPOTHESIS, not measured on an L4: the build takes 5–10 min the first time; each process takes about 10–40 s of GPU time, plus cooldown, which should be near zero on a datacenter GPU idling below 80 °C. With 15 E0b processes, expect about 10–20 min for E0b. The E3/E4 stages add 15 E3 timing processes and 15 E4 timing processes (≥ 5 each per configuration) plus the correctness suites. Expect roughly 30–60 min more on an L4 (HYPOTHESIS; dominated by 32K E3 and B = 32 E4 setup).

## 3. What the run records

- `environment.txt`: driver, CUDA, tileiras, rustc, power limits, max clocks, `nvidia-smi -q` performance/clock/power/temperature, and git state.
- `binaries.sha256`: hashes of the built `c1_rtable` and `mma_probe`.
- `mma_probe/`: Tile IR (`.mlir`), cubins and `mma_probe_summary.json` (SASS opcode counts, HMMA presence) for T ∈ {64, 128, 256, 512}.
- `campaign/manifest.json`: per process, the start condition, start temperature, cooldown wait, return code and thermal kill flag.
- `campaign/seq*_p*/`:
  - `samples.jsonl`: per sample, latency, SM clock before and after, memory clock, temperature, reason bits and keep-alive fields (keep-alive is off by default);
  - `timing_meta.json`;
  - `correctness.json` and `query_refresh.json`: correctness is checked before timing, and the run aborts on failure.
- `campaign/seq*_p*_telemetry.csv`: 100 ms external telemetry.
- `campaign/e0b_analysis.json`: per-length, per-component C1/A1 results:
  - per-process ratio of medians;
  - geometric mean;
  - hierarchical-bootstrap 95 % CI (10,000 replicates, seed 20261002), computed over **clean** runs only, i.e. cool-start and not THROTTLED;
  - launch floor: the empty event pair and the smallest kernel × launch count.

## 3b. E3 / E4 stages (same run, after E0b)

`run_e0b_portable.sh` also runs the E3 split-K and E4 batch/layer paths, unless `--skip-e3e4` is given:

1. **E3 correctness gate:** `e3_splitk_gpu --suite e3` (oracle chain, partial final blocks, split invariance, padded A3, both C3 modes) and `--suite scale` (N = 4K/8K/32K, uniform and heavy-tailed, matched-rounding reference). The scale suite is skipped in a dry run.
2. **E3 timing** (`scripts/e0b/e3e4_campaign.py`): padded A3 vs C3 fast vs C3 accurate at `--e3-lengths` (default 1024, 8192, 32768), with ≥ 5 cool-start processes each.
   - Each sample is one variant's split + reduce between CUDA events.
   - Variant order rotates by (iteration + process).
   - Plans come from `splitk_plan::plan_bounded`: at most 128 steps per CTA, giving 1 / 2 / 8 splits.
3. **E4 correctness:** `--suite e4 --seq N --batches <e4-batches>`, one process per N in `--e4-seqs` (default 1024), all before any timing. Every batch element is validated against per-sequence references over a shared block pool. If one fails, the campaign stops.
4. **E4 timing:** `--suite e4-timing --batch B --seq N` over the grid `--e4-batches` (default 1, 8, 32) × `--e4-seqs` (default 1024), with ≥ 5 cool-start processes each. Configs are labelled `B{b}_N{n}`; older campaigns used `B{b}`, which the analyzer reads as N = 1024.
   - L = max(4, ceil(65536 / (B·N))) resident per-layer pools. At N = 1024 this is the original max(4, ceil(64 / B)); for any N it keeps the full-KV pools at ≥ 64 MiB.
   - Active lengths: sequence 0 runs N tokens; sequence b runs 1 + ((b·373) mod 1024)·N / 1024. At N = 1024 this equals the original 1 + (b·373) mod 1024.
   - Footprint check before allocation: full KV + latent pools + the 64 MiB flush must fit `--max-footprint-gib` (default 20), and the host f32 copies must fit `--max-host-footprint-gib` (default 24). A refused configuration prints `E4_CONFIG_REFUSED` and exits 2 before opening a GPU stream.
   - Check the grid without a GPU first: `cargo run --release -p plkv-kernels --example e4_plan -- --seq 32768 --batches 1,8,32`, and `python3 scripts/e0b/e3e4_campaign.py --out-dir /tmp/unused --print-plan --e4-seqs 1024,8192,32768`.
   - Step rows carry `bytes_analytical` and `achieved_gbps`. Bytes are analytical (`e4_config::e4_bytes`), not measured.
   - A 64 MiB L2-flush memset runs before each layer, outside the timed interval.
   - The per-step total is the per-iteration sum of the layer intervals.
5. **Analysis:** `scripts/e0b/analyze_variants.py` writes `e3e4/variants_analysis.json`.
   - It reports per-process ratios of medians against padded A3, their geometric mean over clean processes, and a hierarchical-bootstrap 95 % CI.
   - THROTTLED and hot-start processes are reported separately.
   - `claim_eligible` requires ≥ 5 clean processes.

**Dry run:** E3 timing at 1K and E4 timing at B = 1, one process each, 5 iterations, plus the E3 correctness suite. The full E4 correctness suite is skipped, but its path is exercised by the E4-timing pre-check, which validates every batch element of layer 0 before timing. The dry run also checks that the samples carry clocks, the expected variants and a correctness record, and that the analysis has E3 and E4 rows.

**Status on the development laptop:** none of these timing paths was run there. They were exercised only with `--iterations 0` (setup, async-path correctness and output). The first real run is the cloud dry run.

**Labeling:** results from this package must be labeled with the host (for example "cloud L4, native Linux"). Laptop results, if any are ever taken, must be labeled "laptop, WSL2, not a cloud result".

## 4. Analysis rules (fixed before running)

- **THROTTLED**: any in-process SM clock read more than 5 % below the run's modal clock, or any thermal or HW-slowdown reason bit (0x8, 0x20, 0x40, 0x80) in NVML or external telemetry. THROTTLED and hot-start runs are reported separately and never enter ratios or CIs.
- **THROTTLED (reference clock), added 2026-10-31 before any local Phase 1 data:** a process is also THROTTLED if its modal SM clock is more than 5 % below its own warm-up reference clock (`reference_clock_mhz`, the mode of the second half of the warm-up reads). This applies to E0b, E3 and E4. It only adds an exclusion; no earlier rule is relaxed. Reason: on WSL2 a slow host-side NVML call idled the GPU between samples, the whole process ran at 210 MHz, and the modal-clock rule passed it because the mode was 210 MHz.
- **In-band fraction (reported, not a gate):** for every process the analyzers report the fraction of in-process SM clock reads within 95 % of that reference clock. It is an instrument check of the harness, not a pass mark for results.
- **Void output:** the WSL2 dry run `~/runs/dry_20261031_194558` (before the per-step reasons fix, commit `507dea2`) is VOID: its E4 layers ran mostly at idle clocks. It is not evidence of anything.
- **Primary estimate**: C1/A1 ratio of medians per process, geometric mean across clean processes, hierarchical-bootstrap 95 % CI.
- **A claim needs ≥5 clean processes per length.** With fewer, results are descriptive only.
- **Parity**: claim it only with TOST at a ±5 % margin, i.e. the 90 % CI inside [0.95, 1.05]. Otherwise say "not shown".

## 5. Known limitations

- The runner was syntax-checked, and its refusal paths and individual checks were exercised on the WSL laptop. The full native-Linux path has **not** been run end to end.
- Correctness tolerances and kernels are unchanged from C1. E0b measures existing kernels; it does not test E3.
