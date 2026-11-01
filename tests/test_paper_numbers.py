"""CPU tests for scripts/paper/make_tables.py and scripts/paper/check_numbers.py."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "scripts" / "paper" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_literal_extraction_skips_code_labels_and_hashes():
    cn = load("check_numbers")
    text = (
        "# 3 Results\n"
        "C3 is 0.893 of A3 at N=8192 and moves 13.2x fewer bytes (11 %).\n"
        "Labels B32_N4096, sm_89, commit 507dea2 and `--iterations 50` are not numbers.\n"
        "<!-- 999 in a comment -->\n"
        "```\n777 in a fence\n```\n"
        "See [run](reports/x_20261101/a.json).\n"
    )
    lits = [lit for _, line in cn.strip_prose(text) for lit in cn.literals(line)]
    assert lits == ["0.893", "8192", "13.2", "11"]


def test_check_flags_untraceable_and_accepts_registered(tmp_path):
    cn = load("check_numbers")
    doc = tmp_path / "p.md"
    doc.write_text("ratio 0.893, invented 0.777, constant 512\n")
    bad = cn.check([doc], registry={"0.893"}, whitelist={"512"}, tokens=set())
    assert bad == [f"{doc}:1: 0.777"]
    assert cn.check([doc], {"0.893"}, {"512"}, tokens={"0.777"}) == []


def test_whitelist_requires_a_reason(tmp_path):
    cn = load("check_numbers")
    wl = tmp_path / "wl.txt"
    wl.write_text("512  # observed HMMA threshold\n")
    assert cn.load_whitelist(wl) == {"512"}
    wl.write_text("512\n")
    with pytest.raises(SystemExit, match="without a reason"):
        cn.load_whitelist(wl)


def test_make_tables_reads_committed_results(tmp_path):
    mt = load("make_tables")
    mt.main(["--out", str(tmp_path)])
    numbers = json.loads((tmp_path / "numbers.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["label"] == "laptop RTX 4060, WSL2, not a cloud result"
    assert len(summary["claim_eligible"]) == 4 and len(summary["descriptive"]) == 7
    for value in ("0.893", "0.865", "1.030", "0.972", "0.924"):
        assert value in numbers, value
    timing = (tmp_path / "tab_timing.md").read_text()
    assert "not enough clean processes" in timing and "claim-eligible" in timing
