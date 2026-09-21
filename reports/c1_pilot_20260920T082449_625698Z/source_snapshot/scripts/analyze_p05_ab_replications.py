"""Summarize accepted P0.5 A/B processes at the run level."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def _range(records: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        if value is not None:
            values.append(float(value))
    return [min(values), max(values)]


def _sign_test_two_sided(ratios: list[float]) -> float:
    positive = sum(ratio > 1.0 for ratio in ratios)
    negative = sum(ratio < 1.0 for ratio in ratios)
    n = positive + negative
    tail = min(positive, negative)
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=200_000)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    accepted = [record for record in manifest["records"] if record["status"] == "ACCEPTED"]
    rejected = [
        record for record in manifest["records"] if record["status"] != "ACCEPTED"
    ]
    ratios = [float(record["b_over_a"]) for record in accepted]
    differences = [float(record["difference_us"]) for record in accepted]
    ratio_array = np.asarray(ratios, dtype=np.float64)
    rng = np.random.default_rng(0)
    resampled = ratio_array[
        rng.integers(0, len(ratio_array), size=(args.bootstrap_replicates, len(ratio_array)))
    ]
    bootstrap_medians = np.median(resampled, axis=1)
    bootstrap_means = np.mean(resampled, axis=1)

    exact_median_coverage = 1.0 - 2.0 * (0.5 ** len(ratios))
    exact_median_interval = [min(ratios), max(ratios)]
    bands = {}
    for percent in (2.0, 5.0):
        fraction = percent / 100.0
        interval = [1.0 - fraction, 1.0 + fraction]
        bands[f"plus_minus_{int(percent)}_percent"] = {
            "band": interval,
            "all_accepted_runs_inside": all(
                interval[0] <= ratio <= interval[1] for ratio in ratios
            ),
            "exact_median_interval_inside": (
                interval[0] <= exact_median_interval[0]
                and exact_median_interval[1] <= interval[1]
            ),
        }

    sign_p = _sign_test_two_sided(ratios)
    winner = (
        "statistically_indistinguishable_by_exact_sign_test"
        if sign_p >= 0.05
        else "A_faster_by_exact_sign_test"
        if statistics.median(ratios) > 1.0
        else "B_faster_by_exact_sign_test"
    )
    supported_bands = [
        name for name, band in bands.items()
        if exact_median_coverage >= 0.95 and band["exact_median_interval_inside"]
    ]
    equivalence = (
        "exact_median_equivalence_supported_at_" + ",".join(supported_bands)
        if supported_bands
        else "practical_equivalence_not_established_at_95_percent_by_exact_interval"
    )
    result = {
        "statistical_unit": "independent benchmark process",
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_runs": [
            {
                "attempt": record["attempt"],
                "median_a_ms": record["median_a_ms"],
                "median_b_ms": record["median_b_ms"],
                "b_over_a": record["b_over_a"],
                "difference_us": record["difference_us"],
            }
            for record in accepted
        ],
        "b_over_a": {
            "median": statistics.median(ratios),
            "mean": statistics.mean(ratios),
            "sample_standard_deviation": statistics.stdev(ratios),
            "min": min(ratios),
            "max": max(ratios),
            "bootstrap_percentile_median_ci95": np.quantile(
                bootstrap_medians, [0.025, 0.975]
            ).tolist(),
            "bootstrap_percentile_mean_ci95": np.quantile(
                bootstrap_means, [0.025, 0.975]
            ).tolist(),
            "exact_distribution_free_median_interval": exact_median_interval,
            "exact_distribution_free_median_interval_coverage": exact_median_coverage,
            "two_sided_sign_test_p_value": _sign_test_two_sided(ratios),
        },
        "difference_us": {
            "median": statistics.median(differences),
            "mean": statistics.mean(differences),
            "min": min(differences),
            "max": max(differences),
        },
        "practical_equivalence": bands,
        "environmental_ranges_accepted": {
            "pre_gpu_utilization_median_percent": _range(
                accepted, ("environmental", "pre_gpu_utilization_median_percent")
            ),
            "pre_memory_utilization_median_percent": _range(
                accepted, ("environmental", "pre_memory_utilization_median_percent")
            ),
            "pre_temperature_max_c": _range(
                accepted, ("environmental", "pre_temperature_max_c")
            ),
            "measured_temperature_max_c": _range(
                accepted, ("environmental", "measured_temperature_max_c")
            ),
            "measured_active_pstate_fraction": _range(
                accepted, ("environmental", "measured_active_pstate_fraction")
            ),
            "measured_active_sm_clock_cv": _range(
                accepted, ("environmental", "measured_active_sm_clock_cv")
            ),
        },
        "conclusion": winner + "; " + equivalence,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
