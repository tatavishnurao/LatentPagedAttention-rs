#!/usr/bin/env bash
set -euo pipefail

echo "OS:"
uname -a
echo

echo "Git commit:"
git rev-parse HEAD
echo

echo "rustc:"
rustc --version || true
echo

echo "cargo:"
cargo --version || true
echo

echo "uv:"
uv --version || true
echo

echo "Python via uv:"
uv run python --version || true
echo

echo "nvidia-smi:"
nvidia-smi || true
echo

echo "nvcc:"
nvcc --version || true
