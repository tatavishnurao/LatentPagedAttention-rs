"""CPU tests for scripts/e0b/analyze_variants.py on synthetic E3/E4 timing campaigns."""

import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    sys.path.insert(0, str(ROOT / "scripts" / "e0b"))
    spec = importlib.util.spec_from_file_location(
        "analyze_variants", ROOT / "scripts" / "e0b" / "analyze_variants.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_process(root, label, kind, ratio, *, clock=1890, low_clock_every=0, reasons=0, seed=0):
    rng = random.Random(seed)
    component = "pipeline" if kind == "e3_timing" else "step_total"
    rows = []
    for it in range(40):
        base = 1.0 + rng.uniform(-0.02, 0.02)
        for variant, ms in (("a3pad", base), ("c3_accurate", base * ratio)):
            pre = clock - 400 if low_clock_every and it % low_clock_every == 0 else clock
            rows.append(
                {
                    "iteration": it,
                    "variant": variant,
                    "component": component,
                    "latency_ms": ms,
                    "sm_clock_pre_mhz": pre,
                    "sm_clock_post_mhz": clock,
                    "clock_event_reasons": reasons,
                }
            )
    (root / label).mkdir()
    (root / label / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def campaign(tmp_path, specs):
    runs = []
    for i, (kind, config, ratio, condition, extra) in enumerate(specs):
        label = f"{kind}_{config}_p{i}"
        write_process(tmp_path, label, kind, ratio, seed=i, **extra)
        runs.append(
            {
                "label": label,
                "kind": kind,
                "config": config,
                "start_condition": condition,
                "returncode": 0,
            }
        )
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": runs}))
    return tmp_path


def test_recovers_known_ratio_with_ci(tmp_path):
    mod = load()
    root = campaign(tmp_path, [("e3_timing", "N1024", 0.8, "cool-start", {})] * 5)
    res = mod.analyze(root)["results"]["e3_timing/N1024"]["c3_accurate"]
    assert abs(res["geomean_ratio_clean"] - 0.8) < 0.01
    lo, hi = res["ci95_hierarchical_bootstrap"]
    assert lo <= 0.8 <= hi
    assert res["claim_eligible"] is True


def test_throttled_and_hot_start_are_excluded(tmp_path):
    mod = load()
    specs = [("e4_timing", "B8", 0.5, "cool-start", {})] * 2 + [
        ("e4_timing", "B8", 2.0, "cool-start", {"low_clock_every": 5}),
        ("e4_timing", "B8", 2.0, "cool-start", {"reasons": 0x20}),
        ("e4_timing", "B8", 2.0, "hot-start", {}),
    ]
    report = mod.analyze(campaign(tmp_path, specs))
    res = report["results"]["e4_timing/B8"]
    assert len(res["clean_processes"]) == 2
    assert len(res["excluded_processes"]) == 3
    assert abs(res["c3_accurate"]["geomean_ratio_clean"] - 0.5) < 0.01
    assert all(
        abs(r - 2.0) < 0.05 for r in res["c3_accurate"]["per_process_ratio_excluded_descriptive"]
    )
    assert res["c3_accurate"]["claim_eligible"] is False
    throttled = [r for r in report["runs"] if r.get("throttled")]
    assert len(throttled) == 2


def test_tag_flags_missing_clocks(tmp_path):
    mod = load()
    throttled, details = mod.tag([{"latency_ms": 1.0}])
    assert throttled and "no clock telemetry" in details
