"""CPU tests of the --allow-wsl opt-in in scripts/e0b/run_e0b_portable.sh and the host_label
field in the analyzers. The WSL kernel is simulated through E0B_PROC_VERSION; the toolchain is
pointed at a missing directory so the runner stops at the first tool check after the WSL gate."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "e0b" / "run_e0b_portable.sh"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_analyze_variants import campaign, load  # noqa: E402


def run(tmp_path, *args, wsl=True):
    proc_version = tmp_path / "proc_version"
    kernel = "Linux version 6.6 (microsoft-standard-WSL2)" if wsl else "Linux version 6.8"
    proc_version.write_text(kernel + "\n")
    env = {
        **os.environ,
        "E0B_PROC_VERSION": str(proc_version),
        "CUDA_TOOLKIT_PATH": str(tmp_path / "no_cuda"),
    }
    env.pop("E0B_HOST_LABEL", None)
    return subprocess.run(["bash", str(RUNNER), *args], capture_output=True, text=True, env=env)


def test_wsl_refused_without_flag(tmp_path):
    r = run(tmp_path, "--out-dir", str(tmp_path / "out"))
    assert r.returncode == 1
    assert "WSL detected" in r.stderr
    assert "--allow-wsl" in r.stderr
    assert "WARNING: --allow-wsl" not in r.stderr


def test_allow_wsl_skips_only_the_wsl_refusal(tmp_path):
    r = run(tmp_path, "--out-dir", str(tmp_path / "out"), "--allow-wsl")
    assert "WSL detected" not in r.stderr
    assert "nvcc not found" in r.stderr  # the next preflight check still runs
    assert r.stderr.count("WARNING: --allow-wsl") == 2  # at start and at exit
    assert "not a cloud result" in r.stderr
    assert not (tmp_path / "out").exists()


def test_allow_wsl_still_refuses_mnt(tmp_path):
    r = run(tmp_path, "--out-dir", "/mnt/c/results_x", "--allow-wsl")
    assert r.returncode == 1
    assert "--out-dir is under /mnt" in r.stderr


def test_native_linux_prints_no_wsl_warning(tmp_path):
    r = run(tmp_path, "--out-dir", str(tmp_path / "out"), "--allow-wsl", wsl=False)
    assert "nvcc not found" in r.stderr
    assert "WARNING: --allow-wsl" not in r.stderr


def test_host_label_written_to_variants_analysis(tmp_path, monkeypatch):
    mod = load()
    root = campaign(tmp_path, [("e3_timing", "N1024", 0.9, "cool-start", {})])
    monkeypatch.setenv("E0B_HOST_LABEL", "laptop RTX 4060, WSL2, not a cloud result")
    mod.analyze(root)
    report = json.loads((root / "variants_analysis.json").read_text())
    assert report["host_label"] == "laptop RTX 4060, WSL2, not a cloud result"


def test_no_host_label_when_unset(tmp_path, monkeypatch):
    mod = load()
    root = campaign(tmp_path, [("e3_timing", "N1024", 0.9, "cool-start", {})])
    monkeypatch.delenv("E0B_HOST_LABEL", raising=False)
    assert "host_label" not in mod.analyze(root)
