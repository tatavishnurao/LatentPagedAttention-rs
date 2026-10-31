"""Shared matplotlib styling (imported lazily by the figure scripts)."""

from paper_data import INK


def setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "savefig.facecolor": INK["surface"],
            "axes.edgecolor": INK["secondary"],
            "axes.labelcolor": INK["primary"],
            "text.color": INK["primary"],
            "xtick.color": INK["secondary"],
            "ytick.color": INK["secondary"],
            "axes.grid": True,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )
    return plt
