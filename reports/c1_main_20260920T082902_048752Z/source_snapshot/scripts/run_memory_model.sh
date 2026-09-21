#!/usr/bin/env bash
set -euo pipefail

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-attention99}"

uv run plkv-memory \
  --layers 28 \
  --seq-len 4096 \
  --batch-size 1 \
  --kv-heads 8 \
  --head-dim 128 \
  --latent-dim 512 \
  --dtype-bytes 2

uv run plkv-memory \
  --layers 32 \
  --seq-len 8192 \
  --batch-size 2 \
  --kv-heads 8 \
  --head-dim 128 \
  --latent-dim 768 \
  --dtype-bytes 2
