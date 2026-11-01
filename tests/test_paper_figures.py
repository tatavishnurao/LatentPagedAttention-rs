"""CPU tests for scripts/paper: data extraction always runs; rendering runs only if matplotlib
is importable (the project environment does not ship it)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "scripts" / "paper"
sys.path.insert(0, str(PAPER))

import paper_data  # noqa: E402


def test_regime_points_from_synthetic_analysis():
    pts = paper_data.regime_points(paper_data.synthetic_analysis())
    assert len(pts) == 9
    assert all(p["kind"] == "e4_timing" for p in pts)
    assert pts == sorted(pts, key=lambda p: p["working_set_mib"])
    for p in pts:
        lo, hi = p["ci"]
        assert lo <= p["ratio"] <= hi
        assert abs(p["bandwidth_bound_ratio"] - 1 / 9.9) < 1e-9


def test_regime_points_skip_configs_without_bytes_or_ratio():
    analysis = {"results": {
        "e4_timing/B1_N1024": {"c3_accurate": {"geomean_ratio_clean": 1.1}},  # no bytes
        "e4_timing/B8_N1024": {"working_set_bytes": 1 << 20, "c3_accurate": {}},  # no ratio
        "e3_timing/N1024": {"working_set_bytes": 2 << 20, "byte_ratio_a3pad_over_c3": 8.0,
                            "c3_accurate": {"geomean_ratio_clean": 0.9}},
    }}
    pts = paper_data.regime_points(analysis)
    assert [p["key"] for p in pts] == ["e3_timing/N1024"]
    assert pts[0]["working_set_mib"] == 2.0
    assert pts[0]["ci"] is None and pts[0]["claim_eligible"] is False


def test_roofline_points_dedupe_baseline():
    pts = paper_data.roofline_points(paper_data.synthetic_analysis())
    assert len(pts) == 18  # 9 configs x (a3pad, c3_accurate), baseline once
    assert {p["variant"] for p in pts} == {"a3pad", "c3_accurate"}


def test_clock_trace(tmp_path):
    trace = paper_data.clock_trace(paper_data.synthetic_samples(tmp_path / "s.jsonl"))
    assert len(trace) == 240
    assert trace[0]["sm_mhz"] == 2040 and trace[0]["us"] == pytest.approx(50.0)
    (tmp_path / "m.jsonl").write_text(json.dumps({"latency_ms": 1, "sm_clock_pre_mhz": None}))
    assert paper_data.clock_trace(tmp_path / "m.jsonl") == []


@pytest.mark.parametrize("script", ["fig1_clock_trace.py", "fig2_roofline.py",
                                    "fig3_regime_map.py"])
def test_self_test_renders(tmp_path, script):
    pytest.importorskip("matplotlib")
    out = tmp_path / (script + ".png")
    subprocess.run([sys.executable, str(PAPER / script), "--self-test", "--out", str(out)],
                   check=True)
    assert out.stat().st_size > 10_000


@pytest.mark.parametrize("script", ["fig2_roofline.py", "fig3_regime_map.py"])
def test_cli_requires_reference_line_inputs(tmp_path, script):
    (tmp_path / "variants_analysis.json").write_text(json.dumps({"results": {}}))
    r = subprocess.run([sys.executable, str(PAPER / script), str(tmp_path), "--out", "x.png"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "required" in r.stderr
