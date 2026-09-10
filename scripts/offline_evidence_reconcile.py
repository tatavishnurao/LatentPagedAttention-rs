#!/usr/bin/env python3
"""Create a reproducible offline evidence reconciliation package.

This script reads supplied artifacts only. It never invokes Cargo, CUDA, nvidia-smi,
GPU benchmarks, authentication, or git-mutating commands.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "p15b_rtable"
OUT = ROOT / "reports" / "evidence_reconciliation_20260907T154752Z"


def load(name: str):
    with (SRC / name).open(encoding="utf-8") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "context").mkdir(exist_ok=True)
    correctness = load("correctness.json")
    scaling = load("scaling_exponents.json")
    verdict = load("rtable_verdict.json")
    feasibility = load("feasibility_32k.json")
    thermal = load("thermal_manifest.json")
    doubling = load("doubling_factors.json")

    checks = []
    for seq, row in correctness["by_seq"].items():
        checks.append({"check": f"correctness_pass_{seq}", "pass": row["pass"], "value": row["pass"]})
        checks.append({"check": f"a1_equals_a0_{seq}", "pass": row["a1_a0"]["max_absolute_error"] == 0.0, "value": row["a1_a0"]})
        checks.append({"check": f"b1_equals_b0_{seq}", "pass": row["b1_b0"]["max_absolute_error"] == 0.0, "value": row["b1_b0"]})
    checks.append({"check": "verdict_confirmed", "pass": verdict["score"] == "CONFIRMED_MAJOR_FIX" and verdict["context"] == "CONFIRMED_MAJOR_FIX", "value": verdict["decision"]})
    checks.append({"check": "32k_not_attempted", "pass": feasibility["status"] == "NOT_ATTEMPTED_THERMAL_GATE_FAILED", "value": feasibility["status"]})
    checks.append({"check": "16k_thermal_abort_recorded", "pass": any(r["seq"] == 16384 and r["status"] == "THERMAL_ABORT" for r in thermal["runs"]), "value": thermal["policy"]})

    status = git("status", "--short")
    head = git("rev-parse", "HEAD")
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "head": head,
        "branch": git("branch", "--show-current"),
        "working_tree_status_before_or_during_phase": status.splitlines(),
        "historical_mutation_performed": False,
        "gpu_execution_performed": False,
        "authentication_performed": False,
        "paper_package_zip_found": False,
        "source_artifacts": [],
    }
    for path in sorted(SRC.glob("*")):
        if path.is_file():
            provenance["source_artifacts"].append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    provenance["source_artifacts"].extend([
        {"path": "crates/plkv-kernels/src/cutile/p15b_rtable_kernels.rs", "sha256": sha256(ROOT / "crates/plkv-kernels/src/cutile/p15b_rtable_kernels.rs")},
        {"path": "scripts/analyze_p15b_rtable.py", "sha256": sha256(ROOT / "scripts/analyze_p15b_rtable.py")},
        {"path": "scripts/run_p15b_rtable.py", "sha256": sha256(ROOT / "scripts/run_p15b_rtable.py")},
    ])

    (OUT / "numerical_validation.json").write_text(json.dumps({"checks": checks, "all_pass": all(c["pass"] for c in checks), "scaling_exponents": scaling, "doubling_factors": doubling}, indent=2) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (OUT / "evidence_matrix.json").write_text(json.dumps({
        "measured": ["supplied GPU runs through 8K", "supplied CPU correctness comparisons through 8K", "thermal statuses in thermal_manifest.json"],
        "analytical": ["cache-byte ratios and memory formulas"],
        "source_level": ["R-TABLE single-entry lookup in p15b_rtable_kernels.rs", "A0/B0 controls retained"],
        "inferred": ["R-TABLE removes the observed superlinear table-load behavior through 8K"],
        "hypothesized": ["R-CTX-PAR may address remaining context topology cost"],
        "unsupported_or_unavailable": ["corrected 16K before/after timing", "32K timing", "real-model or serving-system conclusions", "arXiv package ZIP contents"],
    }, indent=2) + "\n", encoding="utf-8")

    summary = """# Offline Evidence Reconciliation\n\nGenerated without GPU execution.\n\n## Defensible result\nThe supplied evidence supports a source-level R-TABLE repair and a materially improved observed scaling profile through 8K, with A1/B1 matching A0/B0 in the supplied comparisons.\n\n## Explicit limits\nThe corrected 16K run thermally aborted at 86°C before writing a summary. The 32K run was not attempted. Therefore no corrected 16K or 32K timing conclusion is made. R-CTX-PAR remains future work, not an implemented result.\n\n## Integrity\nThe dirty checkout was preserved. No GPU workload, authentication, commit, push, reset, rebase, or deletion was performed. See `provenance.json`, `numerical_validation.json`, and `evidence_matrix.json`.\n"""
    (OUT / "README.md").write_text(summary, encoding="utf-8")
    (OUT / "context" / "PROJECT_CONTEXT.md").write_text((ROOT / "reports/evidence_reconciliation_20260907T154752Z/context/PROJECT_CONTEXT.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "head": head, "checks": len(checks), "all_pass": all(c["pass"] for c in checks)}, indent=2))


if __name__ == "__main__":
    main()
