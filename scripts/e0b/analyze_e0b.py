#!/usr/bin/env python3
"""Analyze an E0b campaign directory.

Run-level THROTTLED (user policy 2026-10-02): any timed sample whose in-process SM clock
(pre or post) is >5 % below the run's modal timed-sample clock, or any thermal/HW slowdown
reason bit in in-process NVML reads or in external 100 ms telemetry during the run.
THROTTLED runs are reported separately and never enter ratios or CIs.

Primary estimate per length/component: geometric mean over clean processes of the
within-process ratio median(C1)/median(A1). 95 % CI: hierarchical bootstrap (resample
processes, then iterations within each process with A1/C1 pairs kept together), 10,000
replicates, percentile interval. Also reported: median of per-iteration paired ratios.
Launch-count floor: median 'empty' event span + launches x median C1 projection span
(the smallest kernel: 16 CTAs, 64 KFLOP).
"""

import collections
import json
import math
import random
import statistics
import sys
from pathlib import Path

THERMAL_BITS = 0x8 | 0x20 | 0x40 | 0x80
COMPONENTS = ["pipeline", "score", "softmax", "context"]
LAUNCHES = {"A1": 3, "C1": 4}
B = 10000


def mode(xs):
    return collections.Counter(xs).most_common(1)[0][0]


def load_run(root, run):
    d = root / run["label"]
    rows = [json.loads(line) for line in (d / "samples.jsonl").open()]
    clocks = [
        c for r in rows for c in (r["sm_clock_pre_mhz"], r["sm_clock_post_mhz"]) if c is not None
    ]
    m = mode(clocks)
    low = sum(1 for c in clocks if c < 0.95 * m)
    reasons = 0
    for r in rows:
        reasons |= r["clock_event_reasons"] or 0
    ext_reasons, ext_temps = 0, []
    for line in (root / f"{run['label']}_telemetry.csv").read_text().splitlines():
        p = [x.strip() for x in line.split(",")]
        try:
            ext_reasons |= int(p[5], 16)
            ext_temps.append(int(p[1]))
        except (ValueError, IndexError):
            pass
    thermal = bool((reasons | ext_reasons) & THERMAL_BITS)
    tags = []
    if low:
        tags.append(f"THROTTLED(clock: {low}/{len(clocks)} reads <95% of mode {m})")
    if thermal:
        tags.append(f"THROTTLED(thermal bits in-proc {hex(reasons)} ext {hex(ext_reasons)})")
    if run["start_condition"] != "cool-start":
        tags.append(run["start_condition"])
    meta = json.loads((d / "timing_meta.json").read_text())
    corr = json.loads((d / "correctness.json").read_text())
    return {
        "rows": rows,
        "modal_clock": m,
        "low_reads": low,
        "reads": len(clocks),
        "tags": tags,
        "throttled": bool(low) or thermal,
        "mem_clock_mode": mode(
            [r["mem_clock_post_mhz"] for r in rows if r.get("mem_clock_post_mhz")]
        ),
        "temp_max_ext": max(ext_temps) if ext_temps else None,
        "temp_max_inproc": max((r["temperature_c"] or 0) for r in rows),
        "keepalive_launches": sum(r["keepalive_launches"] for r in rows),
        "keepalive_failed": sum(1 for r in rows if r.get("keepalive_restored") is False),
        "reference_clock": meta["reference_clock_mhz"],
        "correct": all(
            corr[k]["pass"] for k in ("A1_reference", "B1_reference", "C1_reference", "C1_B1")
        ),
    }


def by_iteration(rows, component):
    out = collections.defaultdict(dict)
    for r in rows:
        if r["component"] == component and r["variant"] in ("A1", "C1"):
            out[r["iteration"]][r["variant"]] = r["latency_ms"] * 1000
    return [(v["A1"], v["C1"]) for _, v in sorted(out.items()) if "A1" in v and "C1" in v]


def ratio_of_medians(pairs):
    return statistics.median(c for _, c in pairs) / statistics.median(a for a, _ in pairs)


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def bootstrap(proc_pairs, rng):
    stats = []
    for _ in range(B):
        sample = [rng.choice(proc_pairs) for _ in proc_pairs]
        ratios = [ratio_of_medians([rng.choice(pp) for _ in pp]) for pp in sample]
        stats.append(geo(ratios))
    stats.sort()
    return stats[int(0.025 * B)], stats[int(0.975 * B) - 1]


def main(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    rng = random.Random(20261002)
    runs = []
    for run in manifest["runs"]:
        if run["returncode"] != 0:
            runs.append({**run, "excluded": f"returncode {run['returncode']}"})
            continue
        runs.append({**run, **load_run(root, run)})
    report = {
        "campaign": str(root),
        "binary_sha256": manifest["binary_sha256"],
        "runs": [],
        "results": {},
    }
    for r in runs:
        report["runs"].append(
            {
                k: r.get(k)
                for k in (
                    "label",
                    "start_condition",
                    "start_temp_c",
                    "cooldown_wait_s",
                    "modal_clock",
                    "mem_clock_mode",
                    "low_reads",
                    "reads",
                    "temp_max_ext",
                    "temp_max_inproc",
                    "keepalive_launches",
                    "keepalive_failed",
                    "reference_clock",
                    "correct",
                    "tags",
                    "throttled",
                    "excluded",
                )
            }
        )
    for seq in sorted({r["seq"] for r in runs}):
        seq_runs = [r for r in runs if r["seq"] == seq and "rows" in r]
        res = {}
        for group, sel in (
            (
                "clean",
                [
                    r
                    for r in seq_runs
                    if not r["throttled"] and r["start_condition"] == "cool-start"
                ],
            ),
            (
                "THROTTLED_or_hot_descriptive",
                [r for r in seq_runs if r["throttled"] or r["start_condition"] != "cool-start"],
            ),
        ):
            g = {"processes": [r["label"] for r in sel]}
            for comp in COMPONENTS:
                proc_pairs = [by_iteration(r["rows"], comp) for r in sel]
                proc_pairs = [pp for pp in proc_pairs if pp]
                if not proc_pairs:
                    continue
                per_proc = [ratio_of_medians(pp) for pp in proc_pairs]
                entry = {
                    "A1_median_us_per_process": [
                        round(statistics.median(a for a, _ in pp), 2) for pp in proc_pairs
                    ],
                    "C1_median_us_per_process": [
                        round(statistics.median(c for _, c in pp), 2) for pp in proc_pairs
                    ],
                    "C1_over_A1_per_process": [round(x, 4) for x in per_proc],
                    "C1_over_A1_geomean": round(geo(per_proc), 4),
                    "paired_iteration_ratio_median": round(
                        statistics.median(c / a for pp in proc_pairs for a, c in pp), 4
                    ),
                }
                if group == "clean" and len(proc_pairs) >= 2:
                    lo, hi = bootstrap(proc_pairs, rng)
                    entry["C1_over_A1_95ci_hierarchical_bootstrap"] = [round(lo, 4), round(hi, 4)]
                g[comp] = entry
            if sel:
                empty = statistics.median(
                    x["latency_ms"] * 1000
                    for r in sel
                    for x in r["rows"]
                    if x["component"] == "empty"
                )
                proj = statistics.median(
                    x["latency_ms"] * 1000
                    for r in sel
                    for x in r["rows"]
                    if x["component"] == "projection"
                )
                g["launch_floor"] = {
                    "empty_event_us": round(empty, 2),
                    "smallest_kernel_projection_us": round(proj, 2),
                    "floor_us": {v: round(n * proj, 2) for v, n in LAUNCHES.items()},
                    "floor_fraction_of_pipeline": {
                        v: round(
                            n
                            * proj
                            / statistics.median(
                                x["latency_ms"] * 1000
                                for r in sel
                                for x in r["rows"]
                                if x["component"] == "pipeline" and x["variant"] == v
                            ),
                            3,
                        )
                        for v, n in LAUNCHES.items()
                    },
                }
            res[group] = g
        report["results"][str(seq)] = res
    (root / "e0b_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
