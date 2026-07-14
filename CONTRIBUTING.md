# Contributing

This repository is a research prototype. Keep contributions narrow, reproducible,
and explicit about what is measured.

Before opening a change:

1. Run `UV_PROJECT_ENVIRONMENT=attention99 uv run pytest -q`.
2. Run `UV_PROJECT_ENVIRONMENT=attention99 uv run ruff check .`.
3. Run `cargo fmt --all --check`.
4. Run `cargo test --workspace`.
5. Run GPU scripts manually only on a CUDA/cuTile-capable machine.

Do not add production-serving claims, real checkpoint claims, or total GPU-memory
reduction claims unless the repository contains direct evidence.
