#!/usr/bin/env python3
"""Fail closed on the A0/B0 versus A1/B1 R-TABLE source contract.

This is source-level validation only: it verifies generated source patterns, not
CUDA instructions, memory transactions, or runtime complexity.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "crates/plkv-kernels/src/cutile/p1_sequence_kernels.rs"
REPAIRED = ROOT / "crates/plkv-kernels/src/cutile/p15b_rtable_kernels.rs"


def modules(text: str, prefix: str) -> list[tuple[str, str]]:
    starts = list(re.finditer(rf"pub mod ({prefix}_\d+) \{{", text))
    result = []
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        result.append((match.group(1), text[match.start():end]))
    return result


def main() -> int:
    old = ORIGINAL.read_text(encoding="utf-8")
    new = REPAIRED.read_text(encoding="utf-8")
    errors: list[str] = []
    old_modules = modules(old, r"(?:full_kv_baseline_kernel|model_profile_kernel)")
    new_modules = modules(new, r"(?:p15b_full_kv_baseline_kernel|p15b_model_profile_kernel)")
    if not old_modules:
        errors.append("no original A0/B0 generated modules found")
    if not new_modules:
        errors.append("no repaired A1/B1 generated modules found")

    for name, body in old_modules:
        if "table.load_tile(const_shape![" not in body or "extract(table_tile" not in body:
            errors.append(f"{name}: A0/B0 original whole-table lookup pattern missing")
    for name, body in new_modules:
        if "table.load_tile(const_shape![1], [logical])" not in body:
            errors.append(f"{name}: A1/B1 single-entry lookup missing")
        if re.search(r"table\.load_tile\(const_shape!\[(?!1\])", body):
            errors.append(f"{name}: A1/B1 contains non-singleton block-table load")
        if "extract(table_tile" in body:
            errors.append(f"{name}: A1/B1 retains whole-table extraction")
        if "physical_block(table, logical_block)" not in body:
            errors.append(f"{name}: repaired path may bypass runtime table lookup")

    pairs = {"full": ("p15b_full_kv_baseline_kernel", "p15b_model_profile_kernel")}
    for label, (a, b) in pairs.items():
        if not any(name.startswith(a) for name, _ in new_modules) or not any(name.startswith(b) for name, _ in new_modules):
            errors.append(f"{label}: symmetric full-KV/latent repaired module family incomplete")

    if errors:
        print("R_TABLE_STRUCTURE_FAIL")
        print("\n".join(f"- {x}" for x in errors))
        return 1
    print(f"R_TABLE_STRUCTURE_OK original_modules={len(old_modules)} repaired_modules={len(new_modules)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
