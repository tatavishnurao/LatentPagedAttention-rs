"""CPU test of the dry-run checker embedded in scripts/e0b/run_e0b_portable.sh: it is extracted
and run against a synthetic results tree (no GPU), with B{b}_N{n} E4 labels."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "e0b" / "run_e0b_portable.sh"
sys.path.insert(0, str(ROOT / "scripts" / "e0b"))


def checker_source():
    text = RUNNER.read_text()
    marker = "<<'EOF'\nimport json\nimport sys\nfrom pathlib import Path\n\nout = Path("
    start = text.index(marker) + len("<<'EOF'\n")
    return text[start : text.index("\nEOF\n", start)]


def jl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def build(out, e4_config="B1_N1024", step_extra=None):
    camp = out / "campaign"
    (camp / "p0").mkdir(parents=True)
    (camp / "manifest.json").write_text(
        json.dumps({"runs": [{"label": "p0", "returncode": 0, "start_condition": "cool-start"}]})
    )
    ok = {"pass": True}
    (camp / "p0" / "correctness.json").write_text(
        json.dumps({k: ok for k in ("A1_reference", "B1_reference", "C1_reference", "C1_B1")})
    )
    jl(camp / "p0" / "samples.jsonl", [{"sm_clock_pre_mhz": 2000, "mem_clock_post_mhz": 6000}])
    (camp / "p0_telemetry.csv").write_text("x\n" * 6)
    (camp / "e0b_analysis.json").write_text(json.dumps({"results": {"1024": {}}}))
    (out / "mma_probe").mkdir()
    (out / "mma_probe" / "mma_probe_summary.json").write_text(
        json.dumps([{"kernel": "k_f16", "sass_mma_instructions": 4}])
    )
    (out / "e3_correctness.log").write_text("E3_SPLITK_GPU_OK=1\n")
    e3e4 = out / "e3e4"
    runs = [
        {"label": "e3_timing_N1024_p0", "kind": "e3_timing", "config": "N1024", "returncode": 0},
        {
            "label": f"e4_timing_{e4_config}_p0",
            "kind": "e4_timing",
            "config": e4_config,
            "returncode": 0,
        },
    ]
    for run in runs:
        (e3e4 / run["label"]).mkdir(parents=True)
        (e3e4 / run["label"] / "correctness.json").write_text(json.dumps({"x_max_abs_error": 0}))
        (e3e4 / run["label"] / "timing_meta.json").write_text(
            json.dumps({"footprint_bytes": {}, "bytes_per_layer_analytical": {}})
        )
    jl(
        e3e4 / runs[0]["label"] / "samples.jsonl",
        [
            {"variant": v, "component": "pipeline", "sm_clock_pre_mhz": 2000}
            for v in ("a3pad", "c3_fast", "c3_accurate")
        ],
    )
    step = {"seq": 1024, "bytes_analytical": 10, "achieved_gbps": 1.0, **(step_extra or {})}
    jl(
        e3e4 / runs[1]["label"] / "samples.jsonl",
        [{"variant": v, "component": "step_total", **step} for v in ("a3pad", "c3_accurate")]
        + [{"variant": "a3pad", "component": "layer", "sm_clock_pre_mhz": 2000}],
    )
    (e3e4 / "manifest.json").write_text(json.dumps({"runs": runs}))
    from analyze_variants import parse_config

    cfg = parse_config("e4_timing", e4_config)
    (e3e4 / "variants_analysis.json").write_text(
        json.dumps({"results": {"e3_timing/N1024": {}, f"e4_timing/{e4_config}": cfg}})
    )


def run_checker(out):
    return subprocess.run(
        [sys.executable, "-c", checker_source(), str(out)], capture_output=True, text=True
    )


def test_checker_accepts_new_labels(tmp_path):
    build(tmp_path)
    r = run_checker(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "DRY_RUN_OK" in r.stdout


@pytest.mark.parametrize(
    "kw, needle",
    [
        ({"e4_config": "B1"}, "config labels not B{b}_N{n}"),
        ({"step_extra": {"achieved_gbps": None}}, "missing seq/bytes_analytical/achieved_gbps"),
    ],
)
def test_checker_rejects_old_labels_and_missing_bytes(tmp_path, kw, needle):
    build(tmp_path, **kw)
    r = run_checker(tmp_path)
    assert r.returncode == 1
    assert needle in r.stdout


def test_checker_skip_e0b_needs_no_campaign(tmp_path):
    import shutil

    build(tmp_path)
    shutil.rmtree(tmp_path / "campaign")
    r = subprocess.run(
        [sys.executable, "-c", checker_source(), str(tmp_path), "skip-e0b"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "E0b: skipped" in r.stdout and "DRY_RUN_OK" in r.stdout
    assert run_checker(tmp_path).returncode != 0  # without the flag the campaign is required
