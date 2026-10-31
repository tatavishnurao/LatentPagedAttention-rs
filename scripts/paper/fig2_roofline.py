#!/usr/bin/env python3
"""Fig. 2, roofline position: clean median step time vs analytic bytes moved per step (log-log),
per variant, with the DRAM floor t = bytes / (--peak-gbps * 1e9). The peak is an input (state its
source in the caption, e.g. a measured copy bandwidth or the vendor figure); this script measures
nothing. Reads only variants_analysis.json.

usage: fig2_roofline.py ANALYSIS_JSON_OR_DIR --peak-gbps 300 --out fig2.png [--kind e4_timing]
       fig2_roofline.py --self-test --out /tmp/fig2_selftest.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import (  # noqa: E402
    COLORS,
    INK,
    MIB,
    NAMES,
    load_analysis,
    roofline_points,
    synthetic_analysis,
)


def render(points, peak_gbps, out, title):
    from plotstyle import setup

    plt = setup()
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    for v in sorted({p["variant"] for p in points}):
        sel = sorted((p for p in points if p["variant"] == v), key=lambda p: p["bytes"])
        ax.plot(
            [p["bytes"] / MIB for p in sel],
            [p["us"] for p in sel],
            marker="o",
            ms=5,
            lw=1.2,
            color=COLORS.get(v, INK["secondary"]),
            label=NAMES.get(v, v),
        )
    lo = min(p["bytes"] for p in points) / 2
    hi = max(p["bytes"] for p in points) * 2
    ax.plot(
        [lo / MIB, hi / MIB],
        [lo / (peak_gbps * 1e9) * 1e6, hi / (peak_gbps * 1e9) * 1e6],
        ls="--",
        lw=1.0,
        color=INK["primary"],
        label=f"DRAM floor at {peak_gbps:g} GB/s",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("analytic bytes moved per step (MiB, log)")
    ax.set_ylabel("clean median step time (us, log)")
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("analysis", nargs="?")
    ap.add_argument("--peak-gbps", type=float)
    ap.add_argument("--kind", default="e4_timing")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        points = roofline_points(synthetic_analysis())
        title, peak = "SYNTHETIC SELF-TEST DATA - not a result", a.peak_gbps or 300.0
    else:
        if not a.analysis or a.peak_gbps is None:
            ap.error("ANALYSIS and --peak-gbps are required")
        points = roofline_points(load_analysis(a.analysis), a.kind)
        title, peak = a.title, a.peak_gbps
    if not points:
        raise SystemExit("no clean medians with analytic bytes")
    print(render(points, peak, a.out, title))


if __name__ == "__main__":
    main()
