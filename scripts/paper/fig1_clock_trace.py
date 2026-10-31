#!/usr/bin/env python3
"""Fig. 1, clock trace: SM clock (top) and latency (bottom) per sample for one process, two
panels on a shared sample axis (no dual y-axis). Reads one samples.jsonl. Pass --component to
select E3 "pipeline" or E4 "step_total"/"layer" rows; a laptop trace can be drawn from existing
E0 samples the same way if they carry sm_clock_pre_mhz.

usage: fig1_clock_trace.py SAMPLES_JSONL --out fig1.png [--component pipeline] [--title TEXT]
       fig1_clock_trace.py --self-test --out /tmp/fig1_selftest.png
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import COLORS, INK, NAMES, clock_trace, synthetic_samples  # noqa: E402


def render(trace, out, title):
    from plotstyle import setup

    plt = setup()
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(5.2, 3.6), dpi=200, sharex=True, gridspec_kw={"height_ratios": [1, 1.4]}
    )
    top.plot([t["i"] for t in trace], [t["sm_mhz"] for t in trace], color=INK["secondary"], lw=1.2)
    top.set_ylabel("SM clock (MHz)")
    for v in sorted({t["variant"] for t in trace}):
        sel = [t for t in trace if t["variant"] == v]
        bottom.scatter(
            [t["i"] for t in sel],
            [t["us"] for t in sel],
            s=8,
            color=COLORS.get(v, INK["secondary"]),
            label=NAMES.get(v, v),
        )
    bottom.set_ylabel("latency (us)")
    bottom.set_xlabel("sample (in time order)")
    bottom.legend(fontsize=7)
    top.set_title(title, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("samples", nargs="?")
    ap.add_argument("--component")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        with tempfile.TemporaryDirectory() as d:
            trace = clock_trace(synthetic_samples(Path(d) / "s.jsonl"))
        title = "SYNTHETIC SELF-TEST DATA - not a result"
    else:
        if not a.samples:
            ap.error("SAMPLES_JSONL is required")
        trace = clock_trace(a.samples, a.component)
        title = a.title
    if not trace:
        raise SystemExit("no samples with sm_clock_pre_mhz and latency_ms")
    print(render(trace, a.out, title))


if __name__ == "__main__":
    main()
