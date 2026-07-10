#!/usr/bin/env bash
set -euo pipefail

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-attention99}"

mkdir -p fixtures/reference
uv run python -m latent_paged_attention.fixtures --out-dir fixtures/reference
