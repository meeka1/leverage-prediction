"""
Shared plotting theme and palette.

Colours are the data-viz reference palette, used **unmodified and in its documented slot
order**. That ordering is the colourblind-safety mechanism, not a cosmetic choice, so slots are
assigned by position and never cycled or re-ordered.

Scope of validity, per the palette's own documentation:
  * the fixed order clears the CVD and normal-vision gates on the *adjacent* pairlist, which
    covers lines, bars and stacks -- the forms used here;
  * for all-pairs forms (scatter, small multiples where every series meets every other) only
    the first three slots clear the floors, hence MAX_ALL_PAIRS_SERIES.

Figures are committed to a light surface: they are going into a printed dissertation, so a
dark variant would never be seen and is deliberately not generated.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- surfaces and ink ---
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#7a7975"
GRID = "#e3e2de"

# --- categorical slots, fixed order (never cycle, never re-order) ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
MAX_ALL_PAIRS_SERIES = 3

# --- sequential: one hue, light -> dark ---
SEQUENTIAL_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_STEPS)

# --- diverging: two opposed hues with a neutral gray midpoint (never a hue at the middle) ---
DIVERGING = LinearSegmentedColormap.from_list(
    "div_red_blue", ["#e34948", "#f0efec", "#2a78d6"]
)

# Dickinson stages in life cycle order, each pinned to a slot so a stage keeps its colour
# across every figure even when a chart omits some stages.
STAGE_ORDER = ["Introduction", "Growth", "Mature", "Shakeout", "Decline"]
STAGE_COLORS = dict(zip(STAGE_ORDER, SERIES))


def apply_theme() -> None:
    """Recessive chrome, thin marks, ink-coloured text."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 2.0,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": TEXT_SECONDARY,
        "font.size": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def strip_spines(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def caption(fig, text: str, y: float = -0.02) -> None:
    """
    One line of context under the figure, in secondary ink.

    `y` is in figure coordinates. Push it lower on figures whose rotated tick labels hang far
    below the axes, or the caption lands on top of them.
    """
    fig.text(0.0, y, text, ha="left", va="top", fontsize=8, color=TEXT_SECONDARY, wrap=True)


def year_axis(ax, years, step: int = 2) -> None:
    """Integer year ticks. Matplotlib otherwise interpolates years to 2007.5 and similar."""
    years = sorted(set(int(y) for y in years))
    ticks = [y for y in years if y % step == 0] or years
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(y) for y in ticks])
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)


def shade_test_period(ax, start_year: int, end_year: int, label: bool = False) -> None:
    """Mark the out-of-sample window so every time series carries the same reference."""
    ax.axvspan(start_year - 0.5, end_year + 0.5, color=GRID, alpha=0.6, zorder=0)
    if label:
        ax.annotate("test period", xy=(start_year - 0.3, ax.get_ylim()[1]),
                    xytext=(0, -4), textcoords="offset points",
                    fontsize=8, color=TEXT_SECONDARY, va="top")


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)
