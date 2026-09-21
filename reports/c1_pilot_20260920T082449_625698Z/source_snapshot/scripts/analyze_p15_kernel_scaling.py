"""Aggregate P1.5A per-kernel timing summaries into machine-readable artifacts.

Reads reports/p15_kernel_scaling/p15_summary_seq*.json and produces:
  * per_kernel_timings.json   (medians + raw samples, with growth vs 1K)
  * component_fractions.json
  * doubling_factors.json
  * correctness.json
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/p15_kernel_scaling"

KERNELS = ("A_score", "A_softmax", "A_context", "B_score", "B_softmax", "B_context")


def load_summaries() -> dict[int, dict]:
    summaries = {}
    for path in glob.glob(str(OUT / "p15_summary_seq*.json")):
        data = json.loads(Path(path).read_text())
        summaries[data["seq"]] = data
    return dict(sorted(summaries.items()))


def main() -> None:
    summaries = load_summaries()
    seqs = sorted(summaries)
    base = seqs[0]

    per_kernel = {"base_seq": base, "note": "median CUDA-event ms; raw samples preserved", "by_seq": {}}
    fractions = {"note": "fraction of isolated-kernel median sum per variant", "by_seq": {}}
    doubling = {"note": "T(2N)/T(N) on medians; diagnostic, not statistical inference", "by_kernel": {k: {} for k in KERNELS + ("A_pipeline", "B_pipeline")}}
    correctness = {"tolerances": {"max_absolute_error": 5.0e-3, "max_probability_row_sum_error": 1.0e-4}, "by_seq": {}}

    for seq in seqs:
        kernels = summaries[seq]["kernels"]
        medians = {k: kernels[k]["median_ms"] for k in kernels}
        per_kernel["by_seq"][str(seq)] = {
            "medians_ms": medians,
            "kernels": {
                k: {
                    "warmup": kernels[k]["warmup"],
                    "iterations": kernels[k]["iterations"],
                    "median_ms": kernels[k]["median_ms"],
                    "mean_ms": kernels[k]["mean_ms"],
                    "min_ms": kernels[k]["min_ms"],
                    "max_ms": kernels[k]["max_ms"],
                    "samples_ms": kernels[k]["samples_ms"],
                }
                for k in kernels
            },
        }
        if base in summaries and seq != base:
            base_medians = summaries[base]["kernels"]
            per_kernel["by_seq"][str(seq)]["growth_vs_base"] = {
                k: medians[k] / base_medians[k]["median_ms"] for k in medians if base_medians[k]["median_ms"] > 0
            }

        for variant, prefix in (("A", "A"), ("B", "B")):
            total = sum(medians[f"{prefix}_{part}"] for part in ("score", "softmax", "context"))
            fractions["by_seq"].setdefault(str(seq), {})[variant] = {
                f"{part}_fraction": medians[f"{prefix}_{part}"] / total if total > 0 else None
                for part in ("score", "softmax", "context")
            }
            fractions["by_seq"][str(seq)][variant]["isolated_sum_ms"] = total
            fractions["by_seq"][str(seq)][variant]["pipeline_median_ms"] = medians[f"{prefix}_pipeline"]

    for kernel in doubling["by_kernel"]:
        for earlier, later in zip(seqs, seqs[1:]):
            if later == earlier * 2:
                m0 = summaries[earlier]["kernels"][kernel]["median_ms"]
                m1 = summaries[later]["kernels"][kernel]["median_ms"]
                doubling["by_kernel"][kernel][f"{earlier}->{later}"] = m1 / m0 if m0 > 0 else None

    for seq in seqs:
        correctness["by_seq"][str(seq)] = summaries[seq]["correctness"]

    (OUT / "per_kernel_timings.json").write_text(json.dumps(per_kernel, indent=2))
    (OUT / "component_fractions.json").write_text(json.dumps(fractions, indent=2))
    (OUT / "doubling_factors.json").write_text(json.dumps(doubling, indent=2))
    (OUT / "correctness.json").write_text(json.dumps(correctness, indent=2))

    # Console tables.
    print("| Seq | A score | A softmax | A context | A total | B score | B softmax | B context | B total |")
    for seq in seqs:
        k = summaries[seq]["kernels"]
        print(
            f"| {seq} | {k['A_score']['median_ms']:.4f} | {k['A_softmax']['median_ms']:.4f} "
            f"| {k['A_context']['median_ms']:.4f} | {k['A_pipeline']['median_ms']:.4f} "
            f"| {k['B_score']['median_ms']:.4f} | {k['B_softmax']['median_ms']:.4f} "
            f"| {k['B_context']['median_ms']:.4f} | {k['B_pipeline']['median_ms']:.4f} |"
        )
    print()
    print("| Seq | A score % | A softmax % | A context % | B score % | B softmax % | B context % |")
    for seq in seqs:
        f = fractions["by_seq"][str(seq)]
        row = f"| {seq}"
        for variant in ("A", "B"):
            for part in ("score", "softmax", "context"):
                row += f" | {100.0 * f[variant][f'{part}_fraction']:.2f}"
        print(row + " |")
    print()
    print("| Kernel | " + " | ".join(f"{a}->{b}" for a, b in zip(seqs, seqs[1:]) if b == a * 2) + " |")
    for kernel, factors in doubling["by_kernel"].items():
        if not factors:
            continue
        print("| " + kernel + " | " + " | ".join(f"{v:.2f}" if v else "-" for v in factors.values()) + " |")


if __name__ == "__main__":
    main()
