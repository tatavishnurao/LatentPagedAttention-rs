#!/usr/bin/env python3
"""Fig. 2, clock traces: three single processes in time order, SM clock (top) and per-layer time
(bottom, log): (a) the VOID dry run (per-sample NVML reasons query), (b) the fix-1 instrument
failure, (c) a clean fix-2 process. Reads only committed per-sample E4 "layer" rows: a
samples.jsonl or samples.jsonl.xz path, or TARBALL.tar.xz::MEMBER for a file inside a committed
raw tarball (read in memory, nothing extracted).

Ported from the 2026-11-03 redraft's figure-sources/fig_clock_traces.py (same panels and labels);
`--stats` prints the numbers the caption quotes, computed from the same rows.

usage: fig_clock_traces.py --out fig_clock_traces.png [--stats]
       (defaults are the three committed processes the paper shows)
"""

import argparse
import io
import json
import lzma
import statistics
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import COLORS, INK, NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "reports" / "local_4060_20261101"
RAW = LOCAL / "raw"
DEFAULTS = (
    str(LOCAL / "instrument_checks" / "e4_B1_N1024_dryrun_VOID" / "samples.jsonl.xz"),
    f"{RAW / 'local4060_b2r_e4n1024_20261101_051528.tar.xz'}::"
    "local4060_b2r_e4n1024_20261101_051528/e3e4/e4_timing_B1_N1024_p3/samples.jsonl",
    f"{RAW / 'local4060_e4n8192_20261101_054853.tar.xz'}::"
    "local4060_e4n8192_20261101_054853/e3e4/e4_timing_B32_N8192_p0/samples.jsonl",
)
PANELS = (
    ("(a) No fix: telemetry call after every sample", "VOID dry run, B=1, N=1024"),
    ("(b) Fix 1: short re-warm bursts", "instrument failure, B=1, N=1024"),
    ("(c) Fix 2: saturating re-warm", "clean, B=32, N=8192"),
)
SHORT = {"a3pad": "GQA, padded", "c3_accurate": "Latent (hi/lo)"}


def read_text(spec):
    if "::" in spec:
        tar_path, member = spec.split("::", 1)
        with tarfile.open(tar_path, "r:xz") as tf:
            return tf.extractfile(member).read().decode()
    path = Path(spec)
    if path.suffix == ".xz":
        return lzma.open(path, "rt").read()
    return path.read_text()


def layer_rows(spec):
    """[(index, variant, sm_clock_pre_mhz, latency_us)] for E4 layer rows with a clock read."""
    out = []
    for line in io.StringIO(read_text(spec)):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("component") != "layer" or r.get("sm_clock_pre_mhz") is None:
            continue
        out.append((len(out), r["variant"], r["sm_clock_pre_mhz"], r["latency_ms"] * 1e3))
    return out


def stats(rows):
    clocks = [c for _, _, c, _ in rows]
    first_210 = next((i for i, _, c, _ in rows if c == 210), None)
    return {
        "samples": len(rows),
        "share_at_210_pct": round(100 * sum(c == 210 for c in clocks) / len(clocks), 1),
        "first_index_at_210": first_210,
        "min_mhz": min(clocks),
        "max_mhz": max(clocks),
        "max_after_first_mhz": max(clocks[1:]),
        "mode_mhz": statistics.mode(clocks),
    }


def render(traces, out):
    from plotstyle import setup

    plt = setup()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8,
        }
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(7.1, 3.0), dpi=300, sharex="col", gridspec_kw={"height_ratios": [1, 1.3]}
    )
    for c, (tr, (title, sub)) in enumerate(zip(traces, PANELS, strict=True)):
        top, bot = axes[0][c], axes[1][c]
        top.plot([t[0] for t in tr], [t[2] for t in tr], color=INK["secondary"], lw=0.9)
        top.set_ylim(0, 2800)
        top.set_title(title + "\n" + sub, fontsize=7.5, loc="left")
        for v in ("a3pad", "c3_accurate"):
            sel = [t for t in tr if t[1] == v]
            bot.scatter(
                [t[0] for t in sel],
                [t[3] for t in sel],
                s=2.5,
                lw=0,
                color=COLORS[v],
                label=SHORT.get(v, NAMES[v]),
            )
        bot.set_yscale("log")
        bot.set_ylim(15, 8000)
        bot.set_xlabel("timed sample (in order)")
        if c == 0:
            top.set_ylabel("SM clock (MHz)")
            bot.set_ylabel("layer time (µs)")
            bot.legend(fontsize=6.5, loc="upper right", markerscale=3)
    fig.tight_layout(pad=0.4)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"FIG_CLOCK_TRACES_OK {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("samples", nargs="*", default=list(DEFAULTS))
    ap.add_argument("--out")
    ap.add_argument("--stats", action="store_true", help="print the caption's numbers")
    a = ap.parse_args(argv)
    if len(a.samples) != 3:
        ap.error("need exactly three sample specs")
    traces = [layer_rows(s) for s in a.samples]
    if a.stats:
        for (title, _), tr in zip(PANELS, traces, strict=True):
            print(title.split(")")[0] + ")", json.dumps(stats(tr)))
    if a.out:
        render(traces, a.out)


if __name__ == "__main__":
    main()
