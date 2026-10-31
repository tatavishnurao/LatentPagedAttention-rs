#!/usr/bin/env python3
"""Fig. 3, regime map: C3-accurate / padded-A3 time ratio with 95 % CI vs working set (MiB, log
x). Reference lines: L2 size (--l2-mib, required unless --self-test), ratio 1.0, and the
bandwidth-bound ratio 1 / (analytic byte ratio) per point. Filled markers are claim-eligible
(>= 5 clean processes); hollow markers are not. Reads only variants_analysis.json.

usage: fig3_regime_map.py ANALYSIS_JSON_OR_DIR --l2-mib 46 --out fig3.png [--title TEXT]
       fig3_regime_map.py --self-test --out /tmp/fig3_selftest.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import COLORS, INK, load_analysis, regime_points, synthetic_analysis  # noqa: E402

MARKERS = {"e3_timing": "s", "e4_timing": "o"}


def render(points, l2_mib, out, title):
    from plotstyle import setup

    plt = setup()
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    color = COLORS["c3_accurate"]
    for kind, label in (("e4_timing", "E4 (B x N step)"), ("e3_timing", "E3 (single sequence)")):
        pts = [p for p in points if p["kind"] == kind]
        if not pts:
            continue
        for p in pts:
            if p["ci"]:
                ax.plot([p["working_set_mib"]] * 2, p["ci"], color=color, lw=1.2, alpha=0.8)
        for eligible in (True, False):
            sel = [p for p in pts if p["claim_eligible"] == eligible]
            if sel:
                ax.scatter(
                    [p["working_set_mib"] for p in sel],
                    [p["ratio"] for p in sel],
                    s=36,
                    marker=MARKERS[kind],
                    zorder=3,
                    linewidths=1.5,
                    facecolors=color if eligible else INK["surface"],
                    edgecolors=color,
                    label=label + ("" if eligible else ", < 5 clean processes"),
                )
    bw = sorted(
        (p["working_set_mib"], p["bandwidth_bound_ratio"])
        for p in points
        if p["bandwidth_bound_ratio"]
    )
    if bw:
        ax.plot(
            [x for x, _ in bw],
            [y for _, y in bw],
            ls=":",
            color=INK["secondary"],
            lw=1.2,
            label="bandwidth-bound ratio (1 / byte ratio)",
        )
    ax.axhline(1.0, color=INK["primary"], lw=0.8)
    ax.axvline(l2_mib, color=INK["secondary"], lw=0.8, ls="--")
    ax.annotate(
        f"L2 {l2_mib:g} MiB",
        (l2_mib, ax.get_ylim()[1]),
        xytext=(3, -10),
        textcoords="offset points",
        fontsize=8,
        color=INK["secondary"],
    )
    ax.set_xscale("log")
    ax.set_xlabel("working set per layer, padded A3 (MiB, log scale)")
    ax.set_ylabel("time ratio C3 accurate / padded A3")
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("analysis", nargs="?")
    ap.add_argument("--l2-mib", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        points = regime_points(synthetic_analysis())
        title, l2 = "SYNTHETIC SELF-TEST DATA - not a result", a.l2_mib or 46.0
    else:
        if not a.analysis or a.l2_mib is None:
            ap.error("ANALYSIS and --l2-mib are required")
        points = regime_points(load_analysis(a.analysis))
        title, l2 = a.title, a.l2_mib
    if not points:
        raise SystemExit("no configurations with a ratio and analytic bytes")
    print(render(points, l2, a.out, title))


if __name__ == "__main__":
    main()
