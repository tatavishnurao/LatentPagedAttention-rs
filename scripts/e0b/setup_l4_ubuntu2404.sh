#!/usr/bin/env bash
# One-time setup for an NVIDIA L4 (sm_89) VM running Ubuntu 24.04 x86_64.
# Installs: uv, build tools, libclang/LLVM 18, CUDA toolkit 13.3 (tileiras), rustup, Claude Code.
# It does NOT install the GPU driver: pick a provider image that already has one
# (nvidia-smi must work before you run this). Run as a normal user with sudo.
# Based on docs/E0B_CLOUD_RUNBOOK.md section 2. Written without access to a VM: UNTESTED.
# Phase 3 only: --lock-sm-mhz needs root for the WHOLE runner. Build as your normal user first, then:
#   sudo -E env "PATH=$PATH" bash scripts/e0b/run_e0b_portable.sh --out-dir ... --skip-build --lock-sm-mhz <MHz>
set -euo pipefail

say() { printf '\n== %s\n' "$*"; }
die() { printf 'SETUP_FAIL: %s\n' "$*" >&2; exit 1; }

say "1. Host checks"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "need Linux x86_64"
grep -qi microsoft /proc/version && die "WSL detected; run this on the cloud VM"
# shellcheck source=/dev/null
. /etc/os-release
[[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 24.04 ]] \
  || die "script targets Ubuntu 24.04 (found ${PRETTY_NAME:-unknown}); adapt the repo URL and LLVM package names"
command -v nvidia-smi >/dev/null || die "nvidia-smi missing: use an image with the NVIDIA driver, or install the driver first"
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
[[ "$CC" == 8.9 ]] || die "compute capability $CC, expected 8.9 (L4 / RTX 40xx)"
DRV_CUDA="$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')"
python3 -c "import sys; a=tuple(map(int,'$DRV_CUDA'.split('.')[:2])); sys.exit(a<(13,2))" \
  || die "driver reports CUDA $DRV_CUDA; the runner is validated with >= 13.2 (driver 595.97). Pick a newer image."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[[ "$FREE_GB" -ge 30 ]] || die "only ${FREE_GB} GB free in \$HOME; want at least 30"
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
[[ "$RAM_GB" -ge 28 ]] || echo "WARNING: ${RAM_GB} GB RAM; E4 at B=32, N=32K is estimated to need ~11.7 GiB plus the build"

say "2. Packages"
sudo apt-get update
sudo apt-get install -y build-essential pkg-config git curl ca-certificates python3 python3-venv \
  llvm-18 libclang-18-dev

say "3. CUDA toolkit 13.3 (NVIDIA apt repo)"
if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
  TMP="$(mktemp -d)"
  curl -fsSL -o "$TMP/cuda-keyring.deb" \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i "$TMP/cuda-keyring.deb"
  rm -rf "$TMP"
  sudo apt-get update
fi
apt-cache policy cuda-toolkit-13-3 | grep -q 'Candidate: [0-9]' \
  || die "package cuda-toolkit-13-3 not found in the NVIDIA repo; check 'apt-cache search cuda-toolkit-13' and the runbook's required version"
sudo apt-get install -y cuda-toolkit-13-3 cuda-tileiras-13-3
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
for t in nvcc tileiras cuobjdump; do
  [[ -x "$CUDA_TOOLKIT_PATH/bin/$t" ]] || die "$CUDA_TOOLKIT_PATH/bin/$t missing after install"
done
"$CUDA_TOOLKIT_PATH/bin/nvcc" --version | tail -2

say "4. rustup (the 1.96.0 toolchain is installed in step 9, inside the repo)"
if ! command -v rustup >/dev/null && [[ ! -x "$HOME/.cargo/bin/rustup" ]]; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
fi

say "4b. uv (validate_release.sh uses uv sync / uv run)"
command -v uv >/dev/null || [[ -x "$HOME/.local/bin/uv" ]] || curl -LsSf https://astral.sh/uv/install.sh | sh

say "5. Environment for later shells"
MARK="# >>> latentpagedattention L4 env >>>"
if ! grep -qF "$MARK" "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<'RC'
# >>> latentpagedattention L4 env >>>
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.3/bin:$HOME/.local/bin:$PATH"
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
export CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.3/bin/tileiras
export LIBCLANG_PATH=/usr/lib/llvm-18/lib
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:${LD_LIBRARY_PATH:-}
# <<< latentpagedattention L4 env <<<
RC
fi

say "6. Optional: persistence mode (stops the driver reinitialising between processes)"
sudo nvidia-smi -pm 1 || echo "could not enable persistence mode (not fatal)"

say "7. Claude Code"
if ! command -v claude >/dev/null && [[ ! -x "$HOME/.local/bin/claude" ]]; then
  curl -fsSL https://claude.ai/install.sh | bash
fi

say "8. Repository (local disk, prep branch)"
if [[ ! -d "$HOME/LatentPagedAttention-rs/.git" ]]; then
  git clone https://github.com/tatavishnurao/LatentPagedAttention-rs.git "$HOME/LatentPagedAttention-rs"
fi
cd "$HOME/LatentPagedAttention-rs"
git fetch origin research/l4-e4-20261031
git checkout research/l4-e4-20261031
git log -1 --oneline

say "9. Rust toolchain pinned by rust-toolchain.toml"
"$HOME/.cargo/bin/rustup" toolchain install 1.96.0 --profile minimal -c clippy -c rustfmt

cat <<'DONE'

SETUP_OK
Next:
  1. source ~/.bashrc
  2. cd ~/LatentPagedAttention-rs && claude          # log in when asked
  3. paste the contents of l4-e4-campaign-prompt.md
If anything is already running on the GPU, stop it first:
  nvidia-smi --query-compute-apps=pid,name --format=csv   # should list nothing
DONE
