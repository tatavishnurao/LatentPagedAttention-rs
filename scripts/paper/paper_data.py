"""Data extraction for the paper figures. Reads only results JSON written by the campaign
(variants_analysis.json from scripts/e0b/analyze_variants.py, samples.jsonl from the harness).
Standard library only, so it is CPU-tested without matplotlib.

Every number the figures show comes from these files; nothing is computed from hardware specs
except the reference lines the caller passes explicitly (L2 size, DRAM peak).
"""

import json
import math
import random
from pathlib import Path

MIB = float(1 << 20)
BASELINE = "a3pad"
PRIMARY = "c3_accurate"

# Categorical slots 1-3 of the validated default palette (light surface #fcfcfb).
COLORS = {"a3pad": "#2a78d6", "c3_accurate": "#eb6834", "c3_fast": "#1baf7a"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb"}
NAMES = {
    "a3pad": "padded A3 (full KV)",
    "c3_accurate": "C3 accurate (latent)",
    "c3_fast": "C3 fast (latent)",
}


def load_analysis(path):
    path = Path(path)
    if path.is_dir():
        path = path / "variants_analysis.json"
    return json.loads(path.read_text())


def regime_points(analysis, variant=PRIMARY):
    """One point per configuration with a ratio: working set, ratio, CI, byte-ratio floor."""
    points = []
    for key, res in sorted(analysis["results"].items()):
        kind = key.split("/", 1)[0]
        entry = res.get(variant)
        ws = res.get("working_set_bytes")
        if not entry or "geomean_ratio_clean" not in entry or not ws:
            continue
        ci = entry.get("ci95_hierarchical_bootstrap")
        byte_ratio = res.get("byte_ratio_a3pad_over_c3")
        points.append(
            {
                "key": key,
                "kind": kind,
                "batch": res.get("batch"),
                "seq": res.get("seq"),
                "working_set_mib": ws / MIB,
                "ratio": entry["geomean_ratio_clean"],
                "ci": ci,
                "claim_eligible": bool(entry.get("claim_eligible")),
                "clean": len(res.get("clean_processes", [])),
                "excluded": len(res.get("excluded_processes", [])),
                # Time ratio if both variants ran at the same achieved bandwidth.
                "bandwidth_bound_ratio": (1.0 / byte_ratio) if byte_ratio else None,
            }
        )
    points.sort(key=lambda p: (p["kind"], p["working_set_mib"]))
    return points


def roofline_points(analysis, kind="e4_timing"):
    """(variant, bytes per step, median us) for clean medians that have analytic bytes."""
    points = []
    for key, res in sorted(analysis["results"].items()):
        if not key.startswith(kind + "/"):
            continue
        step = res.get("bytes_per_step_analytical") or {}
        for _variant, entry in res.items():
            if not isinstance(entry, dict) or "median_ms_clean" not in entry:
                continue
            for v, ms in entry["median_ms_clean"].items():
                if step.get(v):
                    points.append(
                        {
                            "key": key,
                            "variant": v,
                            "bytes": step[v],
                            "us": ms * 1e3,
                            "claim_eligible": bool(entry.get("claim_eligible")),
                        }
                    )
    # The baseline appears once per compared variant; keep one copy.
    unique = {(p["key"], p["variant"]): p for p in points}
    return sorted(unique.values(), key=lambda p: (p["variant"], p["bytes"]))


def clock_trace(samples_path, component=None):
    """Per-sample (index, variant, SM clock MHz, latency us) from one process's samples.jsonl."""
    rows = [json.loads(line) for line in Path(samples_path).read_text().splitlines() if line]
    out = []
    for r in rows:
        if r.get("sm_clock_pre_mhz") is None or "latency_ms" not in r:
            continue
        if component and r.get("component") != component:
            continue
        out.append(
            {
                "i": len(out),
                "variant": r.get("variant"),
                "sm_mhz": r["sm_clock_pre_mhz"],
                "us": r["latency_ms"] * 1e3,
            }
        )
    return out


# ------------------------------------------------------------------ synthetic self-test data --


def synthetic_analysis(seed=7):
    """A fake analysis with the real schema. Values are invented; never use them as results."""
    rng = random.Random(seed)
    results = {}
    for n in (1024, 8192, 32768):
        for b in (1, 8, 32):
            ws = b * n * 1024 * 1.1
            ratio = 1.2 * math.exp(-ws / (256 * MIB)) + 0.35
            results[f"e4_timing/B{b}_N{n}"] = {
                "batch": b,
                "seq": n,
                "working_set_bytes": ws,
                "byte_ratio_a3pad_over_c3": 9.9,
                "bytes_per_step_analytical": {
                    "a3pad": int(4 * ws),
                    "c3_accurate": int(4 * ws / 9.9),
                },
                "clean_processes": ["p"] * 5,
                "excluded_processes": [],
                "c3_accurate": {
                    "geomean_ratio_clean": ratio,
                    "ci95_hierarchical_bootstrap": [ratio * 0.96, ratio * 1.05],
                    "claim_eligible": rng.random() > 0.2,
                    "median_ms_clean": {
                        "a3pad": 4 * ws / 250e6,
                        "c3_accurate": 4 * ws / 250e6 * ratio,
                    },
                },
            }
    return {"campaign": "SYNTHETIC", "results": results}


def synthetic_samples(path, n=120, seed=3):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        clock = 2040 if i < 80 else 1800 + rng.randint(-30, 30)
        for v in ("a3pad", "c3_accurate"):
            rows.append(
                {
                    "iteration": i,
                    "variant": v,
                    "component": "pipeline",
                    "sm_clock_pre_mhz": clock,
                    "latency_ms": (0.05 if v == "a3pad" else 0.04) * 2040 / clock,
                }
            )
    Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path
