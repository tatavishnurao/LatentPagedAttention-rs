#!/usr/bin/env bash
set -euo pipefail

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-attention99}"

mkdir -p reports/reference_benchmarks

uv run plkv-bench-ref --config small --iters 20 --csv reports/reference_benchmarks/small.csv
uv run plkv-bench-ref --config medium --iters 10 --csv reports/reference_benchmarks/medium.csv
