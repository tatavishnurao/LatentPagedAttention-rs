#!/usr/bin/env python3
"""Analyze an E3/E4 timing campaign (scripts/e0b/e3e4_campaign.py).

For every timing configuration (E3: N{n}; E4: B{b}_N{n}, legacy B{b} = N 1024) and every
variant, the statistic is the within-process ratio median(variant) / median(baseline) over
paired iterations, baseline = padded A3 ("a3pad"). E3 uses component "pipeline"; E4 uses
"step_total".

Policy (2026-10-02, same as analyze_e0b.py):
- THROTTLED: any in-process SM clock read > 5 % below the process's modal clock, or any
  thermal/HW slowdown reason bit (0x8, 0x20, 0x40, 0x80) in NVML or external telemetry.
- Only cool-start, non-THROTTLED, successful processes are "clean". THROTTLED and hot-start
  processes are reported separately and never enter ratios or CIs.
- Primary estimate: geometric mean of per-process ratios over clean processes; 95 % CI by
  hierarchical bootstrap (resample processes, then paired iterations within each), 10,000
  replicates, seed 20261002. A claim needs >= 5 clean processes per configuration.

Bytes are analytical (timing_meta.json, written by the harness), never measured. Per
configuration the report adds batch / seq, bytes per step, the working set (padded-A3 bytes per
layer for E4, per step for E3; the L2 is flushed before every E4 layer), the analytic byte
ratio a3pad / c3_accurate, and achieved GB/s = bytes per step / clean median step time.

Standard library only.
"""

import collections
import json
import math
import os
import random
import re
import statistics
import sys
from pathlib import Path

THERMAL_BITS = 0x8 | 0x20 | 0x40 | 0x80
BASELINE = "a3pad"
COMPONENT = {"e3_timing": "pipeline", "e4_timing": "step_total"}
MIN_CLEAN_FOR_CLAIM = 5
B = 10000


DEFAULT_E4_SEQ = 1024


def parse_config(kind, config):
    """{"batch", "seq"} for a config label; E3 has batch 1. Legacy E4 "B8" means N = 1024."""
    if kind == "e3_timing":
        m = re.fullmatch(r"N(\d+)", config)
        return {"batch": 1, "seq": int(m.group(1))} if m else {"batch": None, "seq": None}
    m = re.fullmatch(r"B(\d+)(?:_N(\d+))?", config)
    if not m:
        return {"batch": None, "seq": None}
    return {"batch": int(m.group(1)), "seq": int(m.group(2) or DEFAULT_E4_SEQ)}


def bytes_info(kind, metas):
    """Analytical bytes from the first timing_meta.json that has them (identical per config)."""
    for meta in metas:
        step = meta.get("bytes_per_step_analytical")
        if not step:
            continue
        per_layer = meta.get("bytes_per_layer_analytical") or step
        info = {
            "bytes_per_step_analytical": step,
            "working_set_bytes": per_layer.get(BASELINE),
            "working_set_bytes_c3_accurate": per_layer.get("c3_accurate"),
            "layers": meta.get("layers", 1),
        }
        if step.get(BASELINE) and step.get("c3_accurate"):
            info["byte_ratio_a3pad_over_c3"] = round(step[BASELINE] / step["c3_accurate"], 4)
        minimum = meta.get("cache_only_minimum_bytes_per_layer") or meta.get(
            "cache_only_minimum_bytes"
        )
        if minimum and minimum.get("c3_accurate"):
            info["cache_only_byte_ratio"] = round(minimum[BASELINE] / minimum["c3_accurate"], 4)
        if "footprint_bytes" in meta:
            info["footprint_bytes"] = meta["footprint_bytes"]
        return info
    return {}


def gbps(nbytes, ms):
    return round(nbytes / (ms * 1e-3) / 1e9, 3) if nbytes and ms else None


def mode(values):
    return collections.Counter(values).most_common(1)[0][0]


def tag(rows, external_reasons=0):
    """Returns (throttled, details) for one process from its sample rows."""
    clocks = [
        c
        for r in rows
        for c in (r.get("sm_clock_pre_mhz"), r.get("sm_clock_post_mhz"))
        if c is not None
    ]
    reasons = external_reasons
    for r in rows:
        reasons |= r.get("clock_event_reasons") or 0
    details = []
    low = 0
    if clocks:
        m = mode(clocks)
        low = sum(1 for c in clocks if c < 0.95 * m)
        if low:
            details.append(f"clock: {low}/{len(clocks)} reads <95% of mode {m}")
    else:
        details.append("no clock telemetry")
    thermal = bool(reasons & THERMAL_BITS)
    if thermal:
        details.append(f"thermal bits {hex(reasons)}")
    return (bool(low) or thermal or not clocks), details


def external_reasons(path):
    reasons = 0
    if not path.exists():
        return reasons
    for line in path.read_text().splitlines():
        parts = [x.strip() for x in line.split(",")]
        try:
            reasons |= int(parts[5], 16)
        except (ValueError, IndexError):
            pass
    return reasons


def paired(rows, component, variant):
    """[(baseline_ms, variant_ms)] for iterations that have both."""
    by_it = collections.defaultdict(dict)
    for r in rows:
        if r.get("component") == component and r.get("variant") in (BASELINE, variant):
            by_it[r["iteration"]][r["variant"]] = r["latency_ms"]
    return [
        (v[BASELINE], v[variant])
        for _, v in sorted(by_it.items())
        if BASELINE in v and variant in v
    ]


def ratio_of_medians(pairs):
    return statistics.median(x for _, x in pairs) / statistics.median(b for b, _ in pairs)


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def bootstrap_ci(per_process_pairs, rng, replicates=B):
    stats = []
    for _ in range(replicates):
        sample = [rng.choice(per_process_pairs) for _ in per_process_pairs]
        ratios = [ratio_of_medians([rng.choice(p) for _ in p]) for p in sample]
        stats.append(geomean(ratios))
    stats.sort()
    return stats[int(0.025 * replicates)], stats[int(0.975 * replicates) - 1]


def analyze(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    rng = random.Random(20261002)
    groups = collections.defaultdict(list)
    report = {"campaign": str(root), "runs": [], "results": {}}
    if os.environ.get("E0B_HOST_LABEL"):
        report["host_label"] = os.environ["E0B_HOST_LABEL"]
    for run in manifest["runs"]:
        kind = run.get("kind")
        if kind not in COMPONENT:
            report["runs"].append(
                {"label": run["label"], "kind": kind, "returncode": run["returncode"]}
            )
            continue
        entry = {
            "label": run["label"],
            "kind": kind,
            "config": run["config"],
            "start_condition": run["start_condition"],
            "returncode": run["returncode"],
        }
        samples = root / run["label"] / "samples.jsonl"
        if run["returncode"] != 0 or not samples.exists():
            entry["excluded"] = "failed or no samples"
            report["runs"].append(entry)
            continue
        rows = [json.loads(line) for line in samples.open()]
        throttled, details = tag(rows, external_reasons(root / f"{run['label']}_telemetry.csv"))
        entry.update({"throttled": throttled, "tags": details})
        clean = run["start_condition"] == "cool-start" and not throttled
        entry["clean"] = clean
        report["runs"].append(entry)
        meta_path = root / run["label"] / "timing_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        groups[(kind, run["config"])].append((clean, rows, run["label"], meta))
    for (kind, config), procs in sorted(groups.items()):
        component = COMPONENT[kind]
        variants = sorted(
            {
                r["variant"]
                for _, rows, _, _ in procs
                for r in rows
                if r.get("component") == component
            }
            - {BASELINE}
        )
        info = bytes_info(kind, [m for _, _, _, m in procs])
        res = {
            "component": component,
            "baseline": BASELINE,
            **parse_config(kind, config),
            **info,
            "clean_processes": [lbl for c, _, lbl, _ in procs if c],
            "excluded_processes": [lbl for c, _, lbl, _ in procs if not c],
        }
        step_bytes = info.get("bytes_per_step_analytical", {})
        for variant in variants:
            clean_pairs = [paired(rows, component, variant) for c, rows, _, _ in procs if c]
            clean_pairs = [p for p in clean_pairs if p]
            other_pairs = [paired(rows, component, variant) for c, rows, _, _ in procs if not c]
            other_pairs = [p for p in other_pairs if p]
            entry = {
                "per_process_ratio_clean": [round(ratio_of_medians(p), 4) for p in clean_pairs],
                "per_process_ratio_excluded_descriptive": [
                    round(ratio_of_medians(p), 4) for p in other_pairs
                ],
            }
            if clean_pairs:
                entry["geomean_ratio_clean"] = round(
                    geomean([ratio_of_medians(p) for p in clean_pairs]), 4
                )
                med = {
                    BASELINE: statistics.median(b for p in clean_pairs for b, _ in p),
                    variant: statistics.median(x for p in clean_pairs for _, x in p),
                }
                entry["median_ms_clean"] = {k: round(v, 4) for k, v in med.items()}
                entry["achieved_gbps_clean_median"] = {
                    k: gbps(step_bytes.get(k), v) for k, v in med.items()
                }
            if len(clean_pairs) >= 2:
                lo, hi = bootstrap_ci(clean_pairs, rng)
                entry["ci95_hierarchical_bootstrap"] = [round(lo, 4), round(hi, 4)]
            entry["claim_eligible"] = len(clean_pairs) >= MIN_CLEAN_FOR_CLAIM
            res[variant] = entry
        report["results"][f"{kind}/{config}"] = res
    (root / "variants_analysis.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(analyze(sys.argv[1]), indent=1))
