#!/usr/bin/env python
"""Paper Y3 -- Figure F3 (money figure): the E3 regime map.

Two heatmaps (one per campus, c9 / c10). Axes: induced utilisation (columns,
u-levels 70/90/100/110/130 labelled by util_pool ~0.70..1.30) x recoverable
share beta (rows, 0 / 0.5 / 1.0). Colour = percentage reduction in TWT*(d*)
that SURGE WITH the supervisor (SURGE+SUP) achieves over the tuned
rule WITH the same supervisor (RULE+SUP), a ratio of pooled means over 3 seeds
x 10 held-out instances per cell. The manuscript also reports the layer against
the rule with no supervisor, so the colourbar names this contrast in words
rather than leaving it to the caption.

Colour: a LIGHT sequential ramp in the house blue (#f7fafd -> #7ea9ce). Every
cell is a pale tint, so every numeral is set in near-black ink on top of it; the
darkest cell still gives 7:1 contrast against the ink. Luminance falls
monotonically across the ramp, so the map survives greyscale, and the numeral
printed in every cell is the redundant channel that makes the value readable
with no colour at all. No dark saturated blocks, no white knockout numerals.

The realistic-load band (util 0.90-1.00) is boxed, and its name sits INSIDE the
box, in a header lane inside the axes, so no annotation floats outside the
plotting area. The beta=0 boundary (no recoverable urgency) collapses to ~0 gain
and is outlined separately; its name sits horizontally in a matching lane INSIDE
the axes directly beneath that row, so it reads as a note on the row and cannot
be mistaken for a second line of the y-axis label.

Physical placed size: 16.46 cm wide (CAS single-column text width; placed at
\\linewidth). 1:1, so on-figure pt == placed pt.

Typography: TeX Gyre Termes text with STIX mathtext (Times-metric, matches the
manuscript newtx body); one shared type scale/palette across F2-F5. All text is
near-black: grey is used only for non-text marks (cell separators, the inert-row
outline).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# The regime map is reported under the deployable review policy, which is the
# protocol the paper proposes; the earlier oracle-informed run is retained in
# results/y3_p4/e3_map.csv as the upper reference.
CSV = Path("results/y3_w1b/e3_map_deployable.csv")
OUT = Path("paper/figures/f3_regime_map.pdf")

CM = 1 / 2.54
FIG_W = 16.46 * CM      # placed width (in), CAS single-column text width
FIG_H = 7.3 * CM        # room for the panels plus the horizontal colourbar

U_LEVELS = [70, 90, 100, 110, 130]
BETAS = [0.0, 0.5, 1.0]        # bottom -> top
CAMPUSES = [9, 10]
VMIN, VMAX = 0.0, 95.0

# Light sequential ramp, house blue. Monotonic luminance 0.95 -> 0.37, so the
# ramp reads in greyscale; the deepest tint still clears 7:1 against the ink.
CMAP = LinearSegmentedColormap.from_list(
    "y3_light_blue", ["#f7fafd", "#dce9f5", "#a8c6e3", "#7ea9ce"], N=256)

INK = "#1A1A1A"         # every piece of figure text
C_SEP = "#aab3bc"       # cell separators (non-text mark)
C_INERT = "#6f7880"     # inert-row outline (non-text mark)
C_BAND = "#D55E00"      # vermillion -- the realistic-load band, the one accent

STRIP = 0.37            # headroom above the top row, in cell units, so the
                        # band's name sits inside the axes just above its box
STRIP_B = 0.44          # matching lane below the beta=0 row, in cell units, so
                        # "inert boundary" sits inside the axes directly under
                        # the row it names (horizontal, clear of every cell)
# Both lanes are paid for out of cell height, not figure height: the placed size
# stays 16.46 x 7.30 cm, so adding the lower lane cannot move a page break in the
# manuscript. STRIP is raised from 0.34 to 0.37 so that, at the smaller cell
# height, the band's name keeps the same physical clearance it had before.

# ---- Times-family serif typography (matches the manuscript newtx/Termes body) ---
_TG = Path.home() / ".TinyTeX/texmf-dist/fonts/opentype/public/tex-gyre"
for _f in ("texgyretermes-regular.otf", "texgyretermes-bold.otf",
           "texgyretermes-italic.otf", "texgyretermes-bolditalic.otf"):
    if (_TG / _f).exists():
        fm.fontManager.addfont(str(_TG / _f))
_SERIF = "TeX Gyre Termes" if (_TG / "texgyretermes-regular.otf").exists() else "Nimbus Roman"

# Shared type scale (absolute pt at 1:1 placement) -- one visual system across
# F2-F4. Two sizes; nothing set below 8 pt at the placed width.
FS_TITLE = 9.0          # panel titles
FS_AXIS = 9.0           # axis labels
FS_CBAR = 8.0           # colourbar (legend) label
FS_TICK = 8.0           # tick labels / in-cell values
FS_SMALL = 8.0          # smallest tier (colourbar ticks, boundary labels)

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": [_SERIF, "Nimbus Roman", "Liberation Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8.0, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "figure.dpi": 300,
})


def load_grid():
    d = pd.read_csv(CSV)
    up = d.groupby("u").util_pool.mean().round(2).to_dict()
    grids = {}
    for c in CAMPUSES:
        g = np.full((len(BETAS), len(U_LEVELS)), np.nan)
        for i, b in enumerate(BETAS):
            for j, u in enumerate(U_LEVELS):
                sub = d[(d.campus == c) & (d.u == u) & (d.beta == b)]
                rs, ms = sub.rule_sup.mean(), sub.m0_sup.mean()
                g[i, j] = 100.0 * (rs - ms) / rs
        grids[c] = g
    return grids, up


def main():
    grids, up = load_grid()
    util_labels = [f"{up[u]:.2f}" for u in U_LEVELS]

    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H), sharey=True,
        gridspec_kw=dict(left=0.072, right=0.992, bottom=0.395, top=0.895,
                         wspace=0.085),
    )
    band_cols = [1, 2]     # util 0.90, 1.00 = the realistic-load band
    y_top = len(BETAS) - 0.5 + STRIP
    y_bot = -0.5 - STRIP_B

    for k, (ax, c) in enumerate(zip(axes, CAMPUSES)):
        g = grids[c]
        ax.imshow(g, origin="lower", cmap=CMAP, vmin=VMIN, vmax=VMAX,
                  aspect="auto", interpolation="nearest",
                  extent=(-0.5, len(U_LEVELS) - 0.5, -0.5, len(BETAS) - 0.5))
        ax.set_xticks(range(len(U_LEVELS)))
        ax.set_xticklabels(util_labels, fontsize=FS_TICK)
        ax.set_yticks(range(len(BETAS)))
        ax.set_yticklabels([f"{b:g}" for b in BETAS], fontsize=FS_TICK)
        ax.set_xlabel("utilisation (induced load $u$)", fontsize=FS_AXIS, labelpad=3)
        ax.set_title(f"Campus C{c}", fontsize=FS_TITLE, pad=4)
        # thin grey separators between cells (non-text marks)
        for x in np.arange(0.5, len(U_LEVELS) - 0.5, 1):
            ax.plot([x, x], [-0.5, len(BETAS) - 0.5], color=C_SEP, lw=0.5,
                    zorder=3)
        for y in np.arange(0.5, len(BETAS) - 0.5, 1):
            ax.plot([-0.5, len(U_LEVELS) - 0.5], [y, y], color=C_SEP, lw=0.5,
                    zorder=3)
        ax.tick_params(which="major", length=2)
        # annotate every cell: the redundant channel that carries the value in
        # greyscale and for a colour-blind reader. Always near-black ink.
        for i in range(len(BETAS)):
            for j in range(len(U_LEVELS)):
                v = g[i, j]
                txt = f"{v:.0f}"
                if txt == "-0":
                    txt = "0"
                # true minus sign, as in the manuscript body (Termes has U+2212,
                # verified with pdffonts: no extra family is pulled in)
                txt = txt.replace("-", "−")
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=FS_TICK, color=INK, zorder=6)
        # beta=0 boundary: outline around the inert row
        ax.add_patch(Rectangle((-0.5, -0.5), len(U_LEVELS), 1,
                               fill=False, edgecolor=C_INERT, linewidth=1.0,
                               zorder=4))
        # realistic-load band: dashed box around the two band columns. Its name
        # sits inside the axes, in the headroom directly above that box and
        # centred on it, so nothing floats outside the plotting area and the
        # name cannot be read off the wrong columns.
        ax.add_patch(Rectangle((band_cols[0] - 0.5, -0.5), len(band_cols),
                               len(BETAS),
                               fill=False, edgecolor=C_BAND, linewidth=1.3,
                               linestyle=(0, (4, 2)), zorder=5))
        if k == 0:
            ax.text(np.mean(band_cols), len(BETAS) - 0.5 + STRIP / 2,
                    "realistic-load band", ha="center", va="center",
                    fontsize=FS_SMALL, color=INK, fontweight="bold", zorder=6)
        # beta=0 boundary marker: a horizontal note in the lane inside the axes
        # directly below the outlined row, aligned with the row's left edge. It
        # is the only text in that lane, so it names the row rather than the
        # axis; the left margin now holds the axis label alone.
        if k == 0:
            ax.text(-0.42, -0.5 - STRIP_B / 2, "inert boundary", ha="left",
                    va="center", fontsize=FS_SMALL, color=INK, zorder=6)
        ax.set_xlim(-0.5, len(U_LEVELS) - 0.5)
        ax.set_ylim(y_bot, y_top)

    # the only text left of the axes, so matplotlib's automatic placement (just
    # clear of the tick labels) is now correct and the margin carries no second
    # rotated line that could be read as part of the axis name.
    axes[0].set_ylabel("recoverable share  $\\beta$", fontsize=FS_AXIS)

    # ---- shared horizontal colourbar, naming the contrast in words -----------
    cax = fig.add_axes([0.285, 0.195, 0.430, 0.037])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(VMIN, VMAX),
                                            cmap=CMAP),
                      cax=cax, orientation="horizontal")
    cb.set_label(
        "reduction in true weighted tardiness (%):\n"
        "SURGE with the supervisor (SURGE+SUP) "
        "against the tuned rule with the same supervisor (RULE+SUP)",
        fontsize=FS_CBAR, labelpad=3, color=INK, linespacing=1.35)
    cb.ax.tick_params(labelsize=FS_SMALL, length=2, color=INK,
                      labelcolor=INK)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf")
    print(f"wrote {OUT}  figsize={FIG_W/CM:.2f}x{FIG_H/CM:.2f} cm  serif={_SERIF}")
    for c in CAMPUSES:
        print(f"C{c} cell values (rows beta={BETAS}, cols u={U_LEVELS}):")
        print(np.round(grids[c], 1))


if __name__ == "__main__":
    main()
