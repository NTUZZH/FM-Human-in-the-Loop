#!/usr/bin/env python
"""Paper Y3 -- Figure F5: system schematic of the human-in-the-loop correction loop.

recommend -> review (budget rho) -> override (private urgency s) -> learn
(correction layer updates the urgency estimator hat_s) -> transfer to the
unreviewed decisions, with a feedback arrow closing the loop back to recommend.
Automated stages are drawn in a light blue tint; the human-in-the-loop stage in a
light orange tint (Okabe-Ito, CVD-safe), on a neutral background.

Physical placed size: 16.46 cm wide (CAS single-column text width), placed 1:1.
"""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("paper/figures/f5_schematic.pdf")
CM = 1 / 2.54
FIG_W = 16.46 * CM
FIG_H = 5.2 * CM

BLUE = "#0072B2"        # automated edges / accents
BLUE_FILL = "#DCEAF6"   # automated stage fill
ORANGE = "#B8620A"      # human edge
ORANGE_FILL = "#FBE3C2" # human stage fill
INK = "#1a1a1a"

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.0,
})

# stage: (title, subtitle, is_human)
STAGES = [
    ("Recommend", "rule picks the\nnext work order", False),
    ("Review", "checks a share $\\rho$\nof the picks", True),
    ("Override", "corrects a pick from\nprivate urgency $s$", True),
    ("Learn", "update urgency\nestimate $\\hat{s}$", False),
    ("Transfer", "apply $\\hat{s}$ to all\nunreviewed picks", False),
]


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    n = len(STAGES)
    bw, bh = 17.6, 15.0
    gap = (100 - n * bw) / (n + 1)
    y0 = 17.0                      # box bottom
    ymid = y0 + bh / 2
    centers = []
    for k, (title, sub, human) in enumerate(STAGES):
        x0 = gap + k * (bw + gap)
        cx = x0 + bw / 2
        centers.append(cx)
        fc = ORANGE_FILL if human else BLUE_FILL
        ec = ORANGE if human else BLUE
        box = FancyBboxPatch((x0, y0), bw, bh,
                             boxstyle="round,pad=0.15,rounding_size=1.6",
                             linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2)
        ax.add_patch(box)
        ax.text(cx, y0 + bh - 3.4, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=INK, zorder=3)
        ax.text(cx, y0 + 4.7, sub, ha="center", va="center",
                fontsize=6.5, color=INK, zorder=3, linespacing=1.15)

    # forward arrows between stages
    for k in range(n - 1):
        xa = centers[k] + bw / 2
        xb = centers[k + 1] - bw / 2
        ax.add_patch(FancyArrowPatch((xa, ymid), (xb, ymid),
                     arrowstyle="-|>", mutation_scale=11, lw=1.2,
                     color=BLUE, zorder=1, shrinkA=0, shrinkB=0))

    # human-in-the-loop bracket over Review+Override
    hx0 = centers[1] - bw / 2
    hx1 = centers[2] + bw / 2
    ytop = y0 + bh
    ax.plot([hx0, hx0, hx1, hx1], [ytop + 1.4, ytop + 3.0, ytop + 3.0,
            ytop + 1.4], color=ORANGE, lw=0.9, zorder=1)
    ax.text((hx0 + hx1) / 2, ytop + 3.6, "human in the loop",
            ha="center", va="bottom", fontsize=6.8, color=ORANGE,
            fontweight="bold")

    # feedback loop: Transfer -> Recommend, routed clearly BELOW the boxes
    yL = 9.0
    ax.plot([centers[-1], centers[-1]], [y0, yL], color=BLUE, lw=1.3, zorder=1)
    ax.plot([centers[-1], centers[0]], [yL, yL], color=BLUE, lw=1.3, zorder=1)
    ax.add_patch(FancyArrowPatch((centers[0], yL), (centers[0], y0),
                 arrowstyle="-|>", mutation_scale=12, lw=1.3, color=BLUE,
                 zorder=1, shrinkA=0, shrinkB=0))
    ax.text(50, yL - 1.2, "corrected urgency augments the rule; the loop repeats",
            ha="center", va="top", fontsize=6.8, color=BLUE, style="italic")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf")
    print(f"wrote {OUT}  figsize={FIG_W/CM:.2f}x{FIG_H/CM:.2f} cm")


if __name__ == "__main__":
    main()
