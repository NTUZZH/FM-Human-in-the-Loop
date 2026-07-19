#!/usr/bin/env python
"""Paper Y3 -- Figure F3: the E3 regime map.

Two heatmaps (one per campus, c9 / c10). Axes: induced utilisation (columns,
u-levels 70/90/100/110/130 labelled by util_pool ~0.70..1.30) x recoverable
share beta (rows, 0 / 0.5 / 1.0). Colour = M0+SUP percentage reduction in
TWT*(d*) relative to RULE+SUP (ratio of pooled means over 3 seeds x 10 held-out
instances per cell). Perceptually-uniform, colourblind-safe sequential map
(viridis). The realistic-load band (util ~0.9-1.0) is boxed; the beta=0 boundary
(no recoverable urgency) collapses to ~0 gain.

Physical placed size: 16.46 cm wide (CAS single-column text width; placed at
\\linewidth). 1:1, so on-figure pt == placed pt.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

CSV = Path("results/y3_p4/e3_map.csv")
OUT = Path("paper/figures/f3_regime_map.pdf")

CM = 1 / 2.54
FIG_W = 16.46 * CM      # placed width (in), CAS single-column text width
FIG_H = 7.9 * CM        # taller bottom margin so the band label clears the x-title

U_LEVELS = [70, 90, 100, 110, 130]
BETAS = [0.0, 0.5, 1.0]        # bottom -> top
CAMPUSES = [9, 10]
VMIN, VMAX = 0.0, 95.0
CMAP = plt.get_cmap("viridis")

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.0, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
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


def lum(rgba):
    r, gg, b = rgba[0], rgba[1], rgba[2]
    return 0.2126 * r + 0.7152 * gg + 0.0722 * b


def main():
    grids, up = load_grid()
    util_labels = [f"{up[u]:.2f}" for u in U_LEVELS]

    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H), sharey=True,
        gridspec_kw=dict(left=0.115, right=0.865, bottom=0.30, top=0.86, wspace=0.12),
    )
    band_cols = [1, 2]     # util 0.90, 1.00 = the realistic-load band

    for ax, c in zip(axes, CAMPUSES):
        g = grids[c]
        im = ax.imshow(g, origin="lower", cmap=CMAP, vmin=VMIN, vmax=VMAX,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(U_LEVELS)))
        ax.set_xticklabels(util_labels, fontsize=7)
        ax.set_yticks(range(len(BETAS)))
        ax.set_yticklabels([f"{b:g}" for b in BETAS], fontsize=7)
        ax.set_xlabel("utilisation (induced load $u$)", fontsize=7.5, labelpad=2)
        ax.set_title(f"Campus C{c}", fontsize=8.5, pad=4)
        # thin white gridlines between cells
        ax.set_xticks(np.arange(-0.5, len(U_LEVELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(BETAS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=2)
        # annotate every cell
        for i in range(len(BETAS)):
            for j in range(len(U_LEVELS)):
                v = g[i, j]
                frac = np.clip((v - VMIN) / (VMAX - VMIN), 0, 1)
                tc = "white" if lum(CMAP(frac)) < 0.55 else "black"
                txt = f"{v:.0f}"
                if txt == "-0":
                    txt = "0"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=tc)
        # beta=0 boundary: solid gray outline around the inert row
        ax.add_patch(Rectangle((-0.5, -0.5), len(U_LEVELS), 1,
                               fill=False, edgecolor="#777777", linewidth=1.0,
                               zorder=4, clip_on=False))
        # realistic-load band box (vermillion dashed)
        ax.add_patch(Rectangle((band_cols[0] - 0.5, -0.5), len(band_cols), len(BETAS),
                               fill=False, edgecolor="#D55E00", linewidth=1.6,
                               linestyle=(0, (4, 2)), zorder=5, clip_on=False))
        ax.set_xlim(-0.5, len(U_LEVELS) - 0.5)
        ax.set_ylim(-0.5, len(BETAS) - 0.5)

    axes[0].set_ylabel("recoverable share  $\\beta$", fontsize=7.5)

    # beta=0 boundary marker: compact vertical label in the left margin, row 0
    axes[0].text(-1.02, 0, "inert boundary", rotation=90, ha="center",
                 va="center", fontsize=6.3, color="#555555", clip_on=False)

    # realistic-load band label under panel 1, dropped clear of the x-axis title
    axes[0].text(1.5, -1.35, "realistic-load band", ha="center", va="top",
                 fontsize=6.5, color="#D55E00", fontweight="bold",
                 clip_on=False)

    # shared colourbar
    cax = fig.add_axes([0.885, 0.30, 0.022, 0.56])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(VMIN, VMAX),
                                            cmap=CMAP), cax=cax)
    cb.set_label("M0+SUP reduction in TWT* vs RULE+SUP  (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf")
    print(f"wrote {OUT}  figsize={FIG_W/CM:.2f}x{FIG_H/CM:.2f} cm")


if __name__ == "__main__":
    main()
