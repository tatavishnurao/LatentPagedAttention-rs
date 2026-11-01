#!/usr/bin/env python3
"""Fig. 4, numerics: (a) max |error| vs N on heavy-tailed and uniform inputs, total (solid) and
kernel-attributable (dotted) per variant, from the GPU scale suite; (b) the CPU emulation that
isolates which FP16 rounding causes the heavy-tailed error. Reads only committed result files:
reports/e3_splitk_20261002/scale_c3_modes/scale_suite.log and scale/emulate_c3_numerics.out.

usage: fig4_numerics.py --out reports/local_4060_20261101/figures/fig4_numerics.png
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tables import E3REP, scale_rows  # noqa: E402
from paper_data import COLORS, INK, NAMES  # noqa: E402

VARIANTS = ("c3_fast", "c3_accurate", "a3pad")


def emulation_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"(.+?)\s+max=(\S+) mean=(\S+)", line)
        if m:
            rows.append((m.group(1).strip(), float(m.group(2))))
    return rows


def render(rows, emu, out):
    from plotstyle import setup

    plt = setup()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), dpi=200, width_ratios=[1.25, 1])
    for inputs, marker, alpha in (("heavy_tailed", "o", 1.0), ("uniform", "s", 0.45)):
        for v in VARIANTS:
            sel = sorted((r for r in rows if r["inputs"] == inputs and r["variant"] == v),
                         key=lambda r: r["n"])
            ns = [r["n"] for r in sel]
            label = f"{NAMES[v]}" if inputs == "heavy_tailed" else None
            ax.plot(ns, [r["total"] for r in sel], marker=marker, ms=4, color=COLORS[v],
                    alpha=alpha, label=label)
            ax.plot(ns, [r["kernel"] for r in sel], ls=":", marker=marker, ms=3,
                    color=COLORS[v], alpha=alpha)
    ax.axhline(3e-3, color=INK["secondary"], lw=0.8, ls="--")
    ax.text(4096, 3.15e-3, "3e-3 gate", fontsize=7, color=INK["secondary"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context length N")
    ax.set_ylabel("max |error| vs exact reference")
    ax.set_title("(a) solid: total, dotted: kernel-attributable\n"
                 "circles: heavy-tailed, faded squares: uniform", fontsize=7.5)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", ncol=3, fontsize=7,
               bbox_to_anchor=(0.01, 0.0))

    labels = [lab.replace(" (online softmax only)", "").replace(" (C3 numerics)", "")
              for lab, _ in emu]
    bx.barh(range(len(emu)), [v for _, v in emu], color=COLORS["c3_fast"])
    bx.set_yticks(range(len(emu)), labels, fontsize=7)
    bx.invert_yaxis()
    bx.set_xscale("log")
    bx.set_xlabel("max |error| (CPU emulation)")
    bx.set_title("(b) heavy-tailed, N=8192: which rounding", fontsize=7.5)
    fig.text(0.99, 0.02, "laptop RTX 4060, WSL2, not a cloud result; no timing",
             ha="right", fontsize=6, color=INK["secondary"])
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"FIG4_OK {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = scale_rows(E3REP / "scale_c3_modes" / "scale_suite.log")
    emu = emulation_rows(E3REP / "scale" / "emulate_c3_numerics.out")
    if not rows or not emu:
        raise SystemExit("no rows parsed from the scale log or emulation output")
    render(rows, emu, args.out)


if __name__ == "__main__":
    main()
