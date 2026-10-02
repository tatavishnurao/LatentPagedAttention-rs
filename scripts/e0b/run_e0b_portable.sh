#!/usr/bin/env bash
# Portable E0b runner for a native-Linux sm_89 GPU host (validated target: cloud L4).
# Preflight -> build -> MMA compile probe (CPU only) -> E0b campaign -> analysis -> tarball.
# See docs/E0B_CLOUD_RUNBOOK.md. Refuses /mnt paths and WSL.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/e0b/run_e0b_portable.sh --out-dir DIR [options]
  --out-dir DIR        results directory on a native Linux filesystem (must not exist)
  --lengths LIST       default 1024,8192,32768
  --min-cool N         cool-start processes required per length (default 5)
  --max-attempts N     attempts per length before giving up (default 8)
  --cool-c C           cool-start threshold (default 80)
  --cooldown-s S       max cooldown wait per process (default 300)
  --abort-c C          hard thermal abort (default 91)
  --lock-sm-mhz MHZ    optional: lock SM clocks (needs root; reset on exit)
  --skip-build         reuse an existing release build
  --preflight-only     check the environment and exit
  --dry-run            fail-fast smoke test: every stage, but 1K only, 1 process, 1 attempt;
                       then verifies the outputs (correctness, per-sample clocks, analysis)
EOF
}

OUT_DIR="" LENGTHS="1024,8192,32768" MIN_COOL=5 MAX_ATTEMPTS=8 COOL_C=80 COOLDOWN_S=300
ABORT_C=91 LOCK_SM="" SKIP_BUILD=0 PREFLIGHT_ONLY=0 DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --lengths) LENGTHS="$2"; shift 2 ;;
    --min-cool) MIN_COOL="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --cool-c) COOL_C="$2"; shift 2 ;;
    --cooldown-s) COOLDOWN_S="$2"; shift 2 ;;
    --abort-c) ABORT_C="$2"; shift 2 ;;
    --lock-sm-mhz) LOCK_SM="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$OUT_DIR" ]] || { usage >&2; exit 2; }
if [[ "$DRY_RUN" == 1 ]]; then
  LENGTHS=1024 MIN_COOL=1 MAX_ATTEMPTS=1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
OUT_DIR="$(realpath -m "$OUT_DIR")"
fail() { echo "PREFLIGHT_FAIL: $*" >&2; exit 1; }

# ---- preflight -------------------------------------------------------------------------------
[[ "$ROOT" != /mnt/* ]] || fail "repository is under /mnt ($ROOT); clone onto a native filesystem"
[[ "$OUT_DIR" != /mnt/* ]] || fail "--out-dir is under /mnt ($OUT_DIR)"
[[ ! -e "$OUT_DIR" ]] || fail "--out-dir already exists ($OUT_DIR); results directories are immutable"
grep -qi microsoft /proc/version 2>/dev/null && fail "WSL detected; this runner is for native Linux"

# Tool resolution: explicit environment first, then discovery, then documented defaults.
CARGO_BIN="${CARGO_HOME:-$HOME/.cargo}/bin"
[[ -d "$CARGO_BIN" ]] && export PATH="$CARGO_BIN:$PATH"  # rustup's cargo ahead of any shim
if [[ -z "${CUDA_TOOLKIT_PATH:-}" ]]; then
  if command -v nvcc >/dev/null; then
    CUDA_TOOLKIT_PATH="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  elif [[ -d /usr/local/cuda-13.3 ]]; then
    CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
  else
    CUDA_TOOLKIT_PATH=/usr/local/cuda
  fi
fi
export CUDA_TOOLKIT_PATH
export CUTILE_TILEIRAS_PATH="${CUTILE_TILEIRAS_PATH:-$CUDA_TOOLKIT_PATH/bin/tileiras}"
if [[ -z "${LIBCLANG_PATH:-}" ]]; then
  for cfg in llvm-config-18 llvm-config; do
    command -v "$cfg" >/dev/null && { LIBCLANG_PATH="$("$cfg" --libdir)"; break; }
  done
  LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-18/lib}"
fi
export LIBCLANG_PATH
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
export C1_RTABLE_BIN="$TARGET_DIR/release/examples/c1_rtable"
MMA_PROBE_BIN="$TARGET_DIR/release/examples/mma_probe"
export OPENBLAS_NUM_THREADS=1

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found (NVIDIA driver missing)"
[[ -x "$CUDA_TOOLKIT_PATH/bin/nvcc" ]] || fail "nvcc not found under CUDA_TOOLKIT_PATH=$CUDA_TOOLKIT_PATH"
[[ -x "$CUTILE_TILEIRAS_PATH" ]] || fail "tileiras not found at $CUTILE_TILEIRAS_PATH"
[[ -x "$CUDA_TOOLKIT_PATH/bin/cuobjdump" ]] || fail "cuobjdump not found"
[[ -e "$LIBCLANG_PATH/libclang.so" || -n "$(ls "$LIBCLANG_PATH"/libclang*.so* 2>/dev/null)" ]] \
  || fail "libclang not found under LIBCLANG_PATH=$LIBCLANG_PATH"
command -v cargo >/dev/null || fail "cargo not found (install rustup)"
cargo --version | grep -q '^cargo ' || fail "'cargo' on PATH is not Rust's cargo: $(command -v cargo)"
command -v python3 >/dev/null || fail "python3 not found"
python3 -c 'import ctypes; ctypes.CDLL("libnvidia-ml.so.1")' 2>/dev/null \
  || fail "libnvidia-ml.so.1 not loadable (per-sample clock logging requires NVML)"

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
[[ "$CC" == "8.9" ]] || fail "compute capability $CC; kernels and the probe target sm_89 (L4 / RTX 40xx)"
NVCC_VER="$("$CUDA_TOOLKIT_PATH/bin/nvcc" --version | grep -o 'release [0-9.]*' | cut -d' ' -f2)"
[[ "$NVCC_VER" == "13.3" ]] || echo "PREFLIGHT_WARN: nvcc $NVCC_VER; validated with 13.3" >&2
DRV_CUDA="$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')"
# Validated: driver 595.97 reporting CUDA 13.2 with toolkit 13.3 (tileiras emits sm_89 cubins
# ahead of time; the driver only loads them). Older drivers are unvalidated, not known-bad.
python3 - "$DRV_CUDA" <<'EOF' || echo "PREFLIGHT_WARN: driver reports CUDA $DRV_CUDA (< 13.2, unvalidated)" >&2
import sys
major, minor = (int(x) for x in sys.argv[1].split(".")[:2])
sys.exit(0 if (major, minor) >= (13, 2) else 1)
EOF

mkdir -p "$OUT_DIR"
{
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "mode: $([[ "$DRY_RUN" == 1 ]] && echo dry-run || echo full) lengths=$LENGTHS min_cool=$MIN_COOL max_attempts=$MAX_ATTEMPTS"
  echo "host: $(hostname)"; uname -a
  nvidia-smi --query-gpu=name,compute_cap,driver_version,vbios_version,memory.total,power.limit,power.max_limit,clocks.max.sm,clocks.max.mem,temperature.gpu --format=csv
  echo "driver_cuda: $DRV_CUDA"; echo "CUDA_TOOLKIT_PATH=$CUDA_TOOLKIT_PATH LIBCLANG_PATH=$LIBCLANG_PATH"
  echo "cargo: $(command -v cargo)"; "$CUDA_TOOLKIT_PATH/bin/nvcc" --version | tail -2
  "$CUTILE_TILEIRAS_PATH" --version 2>&1 | head -3 || true
  rustc --version; cargo --version; python3 --version
  echo "git_head: $(git -C "$ROOT" rev-parse HEAD)"; git -C "$ROOT" status --short --untracked-files=no
  nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
} > "$OUT_DIR/environment.txt" 2>&1
echo "PREFLIGHT_OK (environment in $OUT_DIR/environment.txt)"
[[ "$PREFLIGHT_ONLY" == 1 ]] && exit 0

# ---- optional clock lock ---------------------------------------------------------------------
if [[ -n "$LOCK_SM" ]]; then
  [[ "$(id -u)" == 0 ]] || fail "--lock-sm-mhz needs root"
  trap 'nvidia-smi -rgc >/dev/null 2>&1 || true' EXIT
  nvidia-smi -lgc "$LOCK_SM,$LOCK_SM" | tee -a "$OUT_DIR/environment.txt"
fi

# ---- build -----------------------------------------------------------------------------------
if [[ "$SKIP_BUILD" == 0 ]]; then
  (cd "$ROOT" && cargo build --locked --release -p plkv-kernels --features gpu-cutile \
     --example c1_rtable --example mma_probe) 2>&1 | tee "$OUT_DIR/build.log" | tail -3
fi
sha256sum "$C1_RTABLE_BIN" "$MMA_PROBE_BIN" \
  | tee "$OUT_DIR/binaries.sha256"

# ---- MMA compile probe (CPU only; no launches) -----------------------------------------------
"$MMA_PROBE_BIN" --target sm_89 --output-dir "$OUT_DIR/mma_probe" \
  | tee "$OUT_DIR/mma_probe.log"

# ---- E0b campaign ----------------------------------------------------------------------------
python3 "$ROOT/scripts/e0b/e0b_campaign.py" --out-dir "$OUT_DIR/campaign" --lengths "$LENGTHS" \
  --min-cool "$MIN_COOL" --max-attempts "$MAX_ATTEMPTS" --cool-c "$COOL_C" \
  --cooldown-s "$COOLDOWN_S" --abort-c "$ABORT_C" | tee "$OUT_DIR/campaign.log"

# ---- analysis + package ----------------------------------------------------------------------
python3 "$ROOT/scripts/e0b/analyze_e0b.py" "$OUT_DIR/campaign" > "$OUT_DIR/analysis_stdout.json"
tar --exclude='oracle_inputs_outputs.json' -czf "$OUT_DIR.tar.gz" -C "$(dirname "$OUT_DIR")" \
  "$(basename "$OUT_DIR")"
if [[ "$DRY_RUN" == 1 ]]; then
  # Fail fast on anything a full campaign would need: a finished process, passing correctness,
  # per-sample NVML clocks, telemetry, MMA lowering, and an analysis row.
  python3 - "$OUT_DIR" <<'EOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
problems = []
camp = out / "campaign"
runs = json.loads((camp / "manifest.json").read_text())["runs"]
if not runs or runs[0]["returncode"] != 0:
    problems.append(f"campaign process failed: {runs[:1]}")
else:
    run = camp / runs[0]["label"]
    corr = json.loads((run / "correctness.json").read_text())
    if not all(corr[k]["pass"] for k in ("A1_reference", "B1_reference", "C1_reference", "C1_B1")):
        problems.append("correctness failed")
    rows = [json.loads(line) for line in (run / "samples.jsonl").open()]
    if not rows:
        problems.append("no timing samples")
    elif any(r["sm_clock_pre_mhz"] is None or r["mem_clock_post_mhz"] is None for r in rows):
        problems.append("per-sample NVML clocks missing")
    tele = (camp / f"{runs[0]['label']}_telemetry.csv").read_text().splitlines()
    if len(tele) < 5:
        problems.append("external telemetry missing")
    print(f"dry-run process: {runs[0]['label']} start={runs[0]['start_condition']} "
          f"samples={len(rows)} telemetry_rows={len(tele)}")
probe = json.loads((out / "mma_probe" / "mma_probe_summary.json").read_text())
f16 = [x for x in probe if x["kernel"].endswith("_f16") and "error" not in x]
if not f16 or any(x["sass_mma_instructions"] == 0 for x in f16):
    problems.append("MMA probe: FP16 kernels missing HMMA")
analysis = json.loads((camp / "e0b_analysis.json").read_text())
if "1024" not in analysis["results"]:
    problems.append("analysis has no 1024 result")
if problems:
    print("DRY_RUN_FAIL: " + "; ".join(problems))
    sys.exit(1)
print("DRY_RUN_OK (a dry run is never evidence: 1 process, no CI)")
EOF
fi
echo "E0B_DONE results=$OUT_DIR archive=$OUT_DIR.tar.gz"
