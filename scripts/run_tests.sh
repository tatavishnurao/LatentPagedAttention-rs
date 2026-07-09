#!/usr/bin/env bash
set -euo pipefail

uv run pytest -q
uv run ruff check .
cargo test --workspace
