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


def write_meta(root, label, kind, batch, seq, c3_step, a3_step, layers=4):
    meta = {
        "suite": kind,
        "seq": seq,
        "bytes_per_step_analytical": {"c3_accurate": c3_step, "a3pad": a3_step},
    }
    if kind == "e4_timing":
        meta.update(
            {
                "batch": batch,
                "layers": layers,
                "bytes_per_layer_analytical": {
                    "c3_accurate": c3_step // layers,
                    "a3pad": a3_step // layers,
                },
                "cache_only_minimum_bytes_per_layer": {"c3_accurate": 100, "a3pad": 1600},
                "footprint_bytes": {"full_kv": 1, "latent": 1, "device_total": 2},
            }
        )
    (root / label / "timing_meta.json").write_text(json.dumps(meta))


def test_multi_n_campaign_parses_labels_bytes_and_gbps(tmp_path):
    """Synthetic B x N grid: per-config ratios, batch/seq parsed, bytes and GB/s attached."""
    mod = load()
    grid = {(1, 1024): 1.1, (8, 8192): 0.7, (32, 32768): 0.4}
    specs = []
    for (b, n), ratio in grid.items():
        specs += [("e4_timing", f"B{b}_N{n}", ratio, "cool-start", {})] * 5
    specs += [("e4_timing", "B8", 0.9, "cool-start", {})] * 2  # legacy label = N 1024
    specs += [("e3_timing", "N8192", 0.8, "cool-start", {})] * 2
    root = campaign(tmp_path, specs)
    runs = json.loads((root / "manifest.json").read_text())["runs"]
    for run in runs:
        cfg = mod.parse_config(run["kind"], run["config"])
        # 1 ms baseline step (synthetic) -> 1e9 bytes => 1000 GB/s for a3pad.
        write_meta(root, run["label"], run["kind"], cfg["batch"], cfg["seq"], 10**8, 10**9)
    res = mod.analyze(root)["results"]
    for (b, n), ratio in grid.items():
        r = res[f"e4_timing/B{b}_N{n}"]
        assert (r["batch"], r["seq"]) == (b, n)
        assert abs(r["c3_accurate"]["geomean_ratio_clean"] - ratio) < 0.01
        assert r["c3_accurate"]["claim_eligible"] is True
        lo, hi = r["c3_accurate"]["ci95_hierarchical_bootstrap"]
        assert lo <= ratio <= hi
        assert r["byte_ratio_a3pad_over_c3"] == 10.0
        assert r["cache_only_byte_ratio"] == 16.0
        assert r["working_set_bytes"] == 10**9 // 4
        gb = r["c3_accurate"]["achieved_gbps_clean_median"]
        assert abs(gb["a3pad"] - 1000.0) < 25.0
        assert abs(gb["c3_accurate"] - 100.0 / ratio) < 0.05 * 100.0 / ratio
    legacy = res["e4_timing/B8"]
    assert (legacy["batch"], legacy["seq"]) == (8, 1024)
    assert legacy["c3_accurate"]["claim_eligible"] is False
    e3 = res["e3_timing/N8192"]
    assert (e3["batch"], e3["seq"]) == (1, 8192)
    assert e3["working_set_bytes"] == 10**9


def test_parse_config_rejects_garbage():
    mod = load()
    assert mod.parse_config("e4_timing", "X1") == {"batch": None, "seq": None}
    assert mod.parse_config("e4_timing", "B32_N32768") == {"batch": 32, "seq": 32768}
    assert mod.parse_config("e3_timing", "N1024") == {"batch": 1, "seq": 1024}


def test_missing_meta_still_analyzes(tmp_path):
    mod = load()
    root = campaign(tmp_path, [("e4_timing", "B1_N1024", 1.0, "cool-start", {})] * 2)
    r = mod.analyze(root)["results"]["e4_timing/B1_N1024"]
    assert "bytes_per_step_analytical" not in r
    assert r["c3_accurate"]["achieved_gbps_clean_median"] == {"a3pad": None, "c3_accurate": None}


def test_idle_clock_process_is_throttled_against_reference(tmp_path):
    """A process whose every read sits at an idle clock passes the modal rule (the mode is the
    idle clock) but fails the warm-up reference rule (runbook section 4, 2026-10-31)."""
    mod = load()
    root = campaign(tmp_path, [("e4_timing", "B1_N1024", 0.9, "cool-start", {"clock": 210})])
    label = "e4_timing_B1_N1024_p0"
    (root / label / "timing_meta.json").write_text(json.dumps({"reference_clock_mhz": 1890}))
    run = mod.analyze(root)["runs"][0]
    assert run["throttled"] is True and run["clean"] is False
    assert run["in_band_fraction"] == 0.0
    assert any("warm-up reference 1890" in t for t in run["tags"])


def test_reference_clock_run_reports_in_band_fraction(tmp_path):
    mod = load()
    root = campaign(tmp_path, [("e3_timing", "N1024", 0.9, "cool-start", {"clock": 1890})])
    (root / "e3_timing_N1024_p0" / "timing_meta.json").write_text(
        json.dumps({"reference_clock_mhz": 1890})
    )
    run = mod.analyze(root)["runs"][0]
    assert run["clean"] is True
    assert run["in_band_fraction"] == 1.0
    assert run["sm_clock_modal_mhz"] == 1890


def _meta(root, label, **kw):
    (root / label / "timing_meta.json").write_text(json.dumps(kw))


def test_absolute_reference_guard_excludes_low_warmup(tmp_path):
    """A process that warmed up far below the configuration's best reference is THROTTLED even
    though its own clocks match its own (low) reference."""
    mod = load()
    specs = [("e4_timing", "B8_N1024", 0.9, "cool-start", {"clock": c}) for c in (1890, 915)]
    root = campaign(tmp_path, specs)
    _meta(root, "e4_timing_B8_N1024_p0", reference_clock_mhz=1890)
    _meta(root, "e4_timing_B8_N1024_p1", reference_clock_mhz=915)
    runs = {r["label"]: r for r in mod.analyze(root)["runs"]}
    assert runs["e4_timing_B8_N1024_p0"]["clean"] is True
    low = runs["e4_timing_B8_N1024_p1"]
    assert low["clean"] is False and low["in_band_fraction"] == 1.0
    assert any("best in config 1890" in t for t in low["tags"])


def test_mixed_harness_versions_are_refused(tmp_path):
    import pytest

    mod = load()
    specs = [("e3_timing", "N8192", 0.9, "cool-start", {})] * 2
    root = campaign(tmp_path, specs)
    _meta(root, "e3_timing_N8192_p0", reference_clock_mhz=1890, harness_version="v-new")
    _meta(root, "e3_timing_N8192_p1", reference_clock_mhz=1890)
    with pytest.raises(SystemExit, match="different harness versions"):
        mod.analyze(root)


def test_legacy_runs_are_labelled(tmp_path):
    mod = load()
    root = campaign(tmp_path, [("e3_timing", "N1024", 0.9, "cool-start", {})])
    report = mod.analyze(root)
    assert report["runs"][0]["harness_version"] == "pre-2026-11-01"
    assert report["results"]["e3_timing/N1024"]["harness_version"] == "pre-2026-11-01"


def test_secondary_baseline_writes_its_own_file(tmp_path):
    """--baseline a3: ratios against unpadded A3, written beside (not over) the primary file."""
    mod = load()
    root = campaign(tmp_path, [("e3_timing", "N8192", 0.8, "cool-start", {})] * 5)
    for i in range(5):
        path = root / f"e3_timing_N8192_p{i}" / "samples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        extra = [
            {**r, "variant": "a3", "latency_ms": r["latency_ms"] * 0.5}
            for r in rows
            if r["variant"] == "a3pad"
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows + extra) + "\n")
    primary = mod.analyze(root)["results"]["e3_timing/N8192"]
    secondary = mod.analyze(root, "a3")["results"]["e3_timing/N8192"]
    assert primary["baseline"] == "a3pad" and secondary["baseline"] == "a3"
    assert abs(primary["c3_accurate"]["geomean_ratio_clean"] - 0.8) < 0.01
    assert abs(secondary["c3_accurate"]["geomean_ratio_clean"] - 1.6) < 0.03
    assert (root / "variants_analysis.json").exists()
    assert (root / "variants_analysis_vs_a3.json").exists()
