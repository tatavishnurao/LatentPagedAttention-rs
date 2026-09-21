#!/usr/bin/env bash
set -euo pipefail

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-attention99}"

echo "RTX 4060 memory sanity checks"
echo "=============================="
echo "Free VRAM reported by nvidia-smi:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits || true
echo
echo "These are theoretical KV-cache estimates only."
echo

echo "Small reference config:"
uv run plkv-memory \
  --layers 24 \
  --seq-len 128 \
  --batch-size 1 \
  --kv-heads 2 \
  --head-dim 32 \
  --latent-dim 32 \
  --dtype-bytes 2
echo

echo "Medium reference config:"
uv run plkv-memory \
  --layers 28 \
  --seq-len 1024 \
  --batch-size 1 \
  --kv-heads 4 \
  --head-dim 64 \
  --latent-dim 128 \
  --dtype-bytes 2
echo

echo "Larger hypothetical config:"
uv run plkv-memory \
  --layers 28 \
  --seq-len 4096 \
  --batch-size 1 \
  --kv-heads 8 \
  --head-dim 128 \
  --latent-dim 512 \
  --dtype-bytes 2
echo

echo "Estimates exclude model weights, activations, temporary buffers, CUDA context, allocator fragmentation, and runtime workspace."
