"""CPU tests for scripts/e0b/e3e4_campaign.py argument handling and the run plan (no GPU)."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "e0b" / "e3e4_campaign.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("e3e4_campaign", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ns(**kw):
    base = {
        "e3_lengths": "1024,8192,32768",
        "e4_batches": "1,8,32",
        "e4_seqs": "1024",
        "max_footprint_gib": 20.0,
        "skip_e4_correctness": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_default_plan_matches_original_protocol():
    mod = load()
    correctness, configs = mod.plan(ns())
    # One E4 correctness process (N = 1024, B = 1,8,32, the original suite), before timing.
    assert [c[0] for c in correctness] == ["e4_correctness_N1024"]
    assert correctness[0][2][:6] == ["--suite", "e4", "--seq", "1024", "--batches", "1,8,32"]
    e3 = [c for c in configs if c[0] == "e3_timing"]
    e4 = [c for c in configs if c[0] == "e4_timing"]
    assert [c[1] for c in e3] == ["N1024", "N8192", "N32768"]
    assert [c[1] for c in e4] == ["B1_N1024", "B8_N1024", "B32_N1024"]
    assert e4[1][2][:6] == ["--suite", "e4-timing", "--batch", "8", "--seq", "1024"]
    assert all("--max-footprint-gib" in c[2] for c in e4)


def test_grid_is_batch_by_seq():
    mod = load()
    correctness, configs = mod.plan(ns(e4_seqs="1024,8192,32768"))
    assert [c[1] for c in correctness] == ["N1024", "N8192", "N32768"]
    labels = [c[1] for c in configs if c[0] == "e4_timing"]
    assert labels == [f"B{b}_N{n}" for n in (1024, 8192, 32768) for b in (1, 8, 32)]
    assert len(set(labels)) == 9


def test_skip_correctness_and_empty_lists():
    mod = load()
    correctness, configs = mod.plan(ns(skip_e4_correctness=True))
    assert correctness == []
    correctness, configs = mod.plan(ns(e4_batches=""))
    assert correctness == [] and all(c[0] == "e3_timing" for c in configs)


@pytest.mark.parametrize(
    "kw",
    [
        {"e4_seqs": "1000"},
        {"e4_seqs": "1024,1024"},
        {"e4_seqs": "abc"},
        {"e4_batches": "0,8"},
        {"e3_lengths": "-16"},
    ],
)
def test_rejects_bad_lists(kw):
    mod = load()
    with pytest.raises(SystemExit):
        mod.plan(ns(**kw))


def test_print_plan_cli_needs_no_gpu_or_binary(tmp_path):
    out = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(tmp_path / "never"),
            "--print-plan",
            "--e4-seqs",
            "1024,8192",
            "--e4-batches",
            "8",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(out.stdout)
    assert [c[1] for c in plan["timing"] if c[0] == "e4_timing"] == ["B8_N1024", "B8_N8192"]
    assert not (tmp_path / "never").exists()


def test_host_check_passes_and_fails():
    mod = load()
    assert mod.host_check("") is None
    assert mod.host_check("echo HOST_CHECK_OK") == "HOST_CHECK_OK"
    with pytest.raises(mod.HostCheckFailed, match="running: brave"):
        mod.host_check("echo 'HOST_CHECK_FAILED: running: brave'; exit 1")


@pytest.mark.parametrize(
    "running, code, needle", [("", 0, "HOST_CHECK_OK"), ("brave\r\nbrave\r\n", 1, "running: brave")]
)
def test_wsl_host_check_script(tmp_path, running, code, needle):
    fake = tmp_path / "powershell.exe"
    fake.write_text(f"#!/usr/bin/env bash\nprintf '{running}'\n")
    fake.chmod(0o755)
    env = {"PATH": f"{tmp_path}:/usr/bin:/bin"}
    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "e0b" / "wsl_host_check.sh")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == code, r.stdout + r.stderr
    assert needle in r.stdout
