#!/usr/bin/env python
"""Paper Y3 -- Figure F4 (money figure): recovery / information-budget curve.

x = cumulative number of supervisor overrides the correction layer has seen,
shared in meaning across both panels.
Panel (a) LEFT : fraction of the gap between the tuned rule and the omniscient
                 reference that the deployed correction layer closes.
Panel (b) RIGHT: sign accuracy of the recovered latent urgency estimate hat_s.
One line per recoverable share beta (0.5 / 0.75 / 1.0). Points are the mean over
3 seeds at each training round; band = +/-1 s.d. across seeds. The three series
are drawn in the paper's two house accents plus near-black, so no accent that
carries a fixed meaning elsewhere in the paper (the bluish-green of the
correction layer in F2) is spent on a beta level here; visual weight rises with
beta, and distinct markers and line styles keep identity from ever resting on
colour alone. Source: live_m0 rows of e4_recovery.csv.

FULL TRAJECTORY, NOT A SATURATION CURVE. All eight training rounds of every beta
are plotted, unsmoothed, and check_no_crop() asserts that every mean AND every
one-s.d. band edge falls inside the axis limits, so the peak-then-decline shape
the manuscript describes (later rounds add mostly confirmations) is visible
rather than cropped or flattened away. Axis names are the caption's words.

Text colour: every label is near-black INK. Grey is used only for non-text marks
(the two neutral reference lines and the gridlines).

Physical placed size: 16.46 cm wide (CAS text width), placed 1:1 at \\linewidth in
a figure* float (like F2/F3), two panels SIDE BY SIDE so all text renders at its
native size. Typography: TeX Gyre Termes text with STIX mathtext (Times-metric,
matches the manuscript newtx body); one shared type scale/palette across F2-F5.
Smallest text >= 6.8 pt at the placed width.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

CSV = Path("results/y3_p5/ablations/e4_recovery.csv")
OUT = Path("paper/figures/f4_recovery.pdf")

CM = 1 / 2.54
FIG_W = 16.46 * CM      # full text width; placed 1:1 at \linewidth in figure*
FIG_H = 6.9 * CM        # side-by-side two-panel layout (matches the F2 aspect)

BETAS = [0.5, 0.75, 1.0]
# The two house accents plus near-black, ordered so visual weight rises with the
# recoverable share: vermillion (lightest, relative luminance 0.22) -> house blue
# (0.15) -> near-black (0.01). All three are CVD-safe against one another and all
# three print at full depth; none is a pastel. Marker shape and line style repeat
# the same ordering, so identity survives greyscale and colour-blind reading.
STYLE = {
    0.5:  dict(color="#D55E00", marker="o", ls=(0, (1, 1)), label=r"$\beta=0.5$"),
    0.75: dict(color="#0072B2", marker="s", ls=(0, (5, 2)), label=r"$\beta=0.75$"),
    1.0:  dict(color="#1A1A1A", marker="^", ls="-",          label=r"$\beta=1.0$"),
}

INK = "#1A1A1A"         # every piece of figure text; no grey type anywhere
C_REF = "#666666"       # neutral reference lines -- LINE colour only

# Axis limits, chosen so that every mean and every band edge fits inside them
# (asserted in check_no_crop) and so each panel keeps a clear top lane for its
# note. Nothing is clipped, smoothed or truncated.
YLIM_A = (0.54, 1.10)
YLIM_B = (0.48, 0.96)
XLIM = (500, 2500)

# ---- Times-family serif typography (matches the manuscript newtx/Termes body) ---
_TG = Path.home() / ".TinyTeX/texmf-dist/fonts/opentype/public/tex-gyre"
for _f in ("texgyretermes-regular.otf", "texgyretermes-bold.otf",
           "texgyretermes-italic.otf", "texgyretermes-bolditalic.otf"):
    if (_TG / _f).exists():
        fm.fontManager.addfont(str(_TG / _f))
_SERIF = "TeX Gyre Termes" if (_TG / "texgyretermes-regular.otf").exists() else "Nimbus Roman"

# Shared type scale (absolute pt at 1:1 placement) -- one visual system across
# F2-F4. Two sizes; nothing set below 8 pt at the placed width (the only glyphs
# under 8 pt are mathtext accents/superscripts, 0.7x their base by construction).
FS_TITLE = 9.0          # panel titles "(a) ..."
FS_AXIS = 9.0           # axis labels
FS_TICK = 8.0           # tick labels
FS_LEG = 8.0            # legend
FS_SMALL = 8.0          # smallest tier (reference labels, panel notes)
LW_DATA = 1.7           # data line
LW_REF = 1.0            # thin reference line
MS_DATA = 4.6           # marker diameter (pt)
BAND_A = 0.14           # 1-sigma band fill alpha (three overlap; kept a light wash)

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


def agg():
    d = pd.read_csv(CSV)
    m = d[d.source == "live_m0"].copy()
    m["gap_closed"] = (m.rule_twt - m.m0_twt_mean) / (m.rule_twt - m.oracle_twt)
    out = {}
    for b in BETAS:
        g = m[m.beta == b]
        a = g.groupby("iter").agg(
            cum=("cum_overrides", "mean"),
            gc=("gap_closed", "mean"), gc_sd=("gap_closed", "std"),
            sa=("sign_acc", "mean"), sa_sd=("sign_acc", "std"),
        ).reset_index().sort_values("cum")
        out[b] = a
    return out


def check_no_crop(data):
    """Fail loudly if any plotted point or band edge would fall outside the axes.

    The manuscript reads this figure for a shape (recovery peaks early, then
    falls back as later rounds add mostly confirmations), so a limit that
    silently clipped a round would change the claim the figure supports.
    """
    for b, a in data.items():
        assert len(a) == 8, f"beta={b}: expected 8 training rounds, got {len(a)}"
        lo_a, hi_a = (a.gc - a.gc_sd).min(), (a.gc + a.gc_sd).max()
        lo_b, hi_b = (a.sa - a.sa_sd).min(), (a.sa + a.sa_sd).max()
        assert XLIM[0] <= a.cum.min() and a.cum.max() <= XLIM[1], \
            f"beta={b}: x range {a.cum.min():.0f}-{a.cum.max():.0f} outside {XLIM}"
        assert YLIM_A[0] <= lo_a and hi_a <= YLIM_A[1], \
            f"beta={b}: panel (a) band {lo_a:.3f}-{hi_a:.3f} outside {YLIM_A}"
        assert YLIM_B[0] <= lo_b and hi_b <= YLIM_B[1], \
            f"beta={b}: panel (b) band {lo_b:.3f}-{hi_b:.3f} outside {YLIM_B}"
        print(f"beta={b}: rounds={len(a)}  x {a.cum.min():.0f}-{a.cum.max():.0f}"
              f"  (a) band {lo_a:.3f}-{hi_a:.3f}  peak {a.gc.max():.3f}"
              f" at round {int(a.gc.idxmax())}, deployed {a.gc.iloc[-1]:.3f}"
              f"  |  (b) band {lo_b:.3f}-{hi_b:.3f}  peak {a.sa.max():.3f}"
              f" at round {int(a.sa.idxmax())}, deployed {a.sa.iloc[-1]:.3f}")


def main():
    data = agg()
    check_no_crop(data)
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H),
        # wspace leaves room for panel (b)'s two-line y-label; right stops short
        # of the edge so the last x tick label is not clipped.
        # left reserves room for a two-line rotated y-label at 9 pt; wspace does
        # the same for panel (b)'s; right stops short of the edge so the last x
        # tick label is not clipped.
        gridspec_kw=dict(left=0.100, right=0.968, bottom=0.150, top=0.900,
                         wspace=0.280),
    )

    for b in BETAS:
        a, s = data[b], STYLE[b]
        # (a) gap closed
        axa.fill_between(a.cum, a.gc - a.gc_sd, a.gc + a.gc_sd,
                         color=s["color"], alpha=BAND_A, linewidth=0, zorder=2)
        axa.plot(a.cum, a.gc, color=s["color"], marker=s["marker"], ls=s["ls"],
                 ms=MS_DATA, mew=0.7, mec="white", lw=LW_DATA, label=s["label"],
                 zorder=4)
        # (b) sign accuracy
        axb.fill_between(a.cum, a.sa - a.sa_sd, a.sa + a.sa_sd,
                         color=s["color"], alpha=BAND_A, linewidth=0, zorder=2)
        axb.plot(a.cum, a.sa, color=s["color"], marker=s["marker"], ls=s["ls"],
                 ms=MS_DATA, mew=0.7, mec="white", lw=LW_DATA, zorder=4)

    # ---- panel (a): deployed value -----------------------------------------
    axa.axhline(1.0, color=C_REF, lw=LW_REF, ls=(0, (2, 2)), zorder=1)
    axa.text(2470, 0.992, "omniscient reference", ha="right", va="top",
             fontsize=FS_SMALL, color=INK)
    axa.set_ylabel("fraction of the rule-to-reference\ngap closed",
                   fontsize=FS_AXIS)
    axa.set_ylim(*YLIM_A)
    axa.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    axa.tick_params(labelsize=FS_TICK, length=2)
    axa.grid(True, color="#e8e8e8", lw=0.5)
    axa.set_axisbelow(True)
    axa.set_xlabel("cumulative supervisor overrides seen", fontsize=FS_AXIS)
    axa.set_xlim(*XLIM)
    axa.set_title("(a) deployed value: dispatch-quality gap closed",
                  fontsize=FS_TITLE, loc="left", pad=3)
    # What the first round already buys: true of all three series at round 0
    # (0.81 / 0.65 / 0.63), so it is a panel note in the clear lane above the
    # reference line, not a leader to one curve, and it crosses nothing.
    axa.text(530, 1.093, "the first round already closes\n"
                         "most of the gap at every $\\beta$",
             ha="left", va="top", fontsize=FS_SMALL, color=INK)

    # ---- panel (b): mechanism ----------------------------------------------
    axb.axhline(0.5, color=C_REF, lw=LW_REF, ls=(0, (2, 2)), zorder=1)
    axb.text(2470, 0.508, "chance", ha="right", va="bottom",
             fontsize=FS_SMALL, color=INK)
    axb.set_ylabel("sign accuracy of the recovered\nurgency estimate $\\hat{s}$",
                   fontsize=FS_AXIS)
    axb.set_ylim(*YLIM_B)
    axb.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    axb.tick_params(labelsize=FS_TICK, length=2)
    axb.grid(True, color="#e8e8e8", lw=0.5)
    axb.set_axisbelow(True)
    axb.set_xlabel("cumulative supervisor overrides seen", fontsize=FS_AXIS)
    axb.set_xlim(*XLIM)
    axb.set_title("(b) mechanism: recovery of the hidden urgency",
                  fontsize=FS_TITLE, loc="left", pad=3)
    # The shape the manuscript now argues for: recovery does not improve with
    # more review. Sits in the panel's clear top lane, above every band.
    axb.text(530, 0.950, "recovery peaks early and does\n"
                         "not improve with more review",
             ha="left", va="top", fontsize=FS_SMALL, color=INK)
    # single legend for both panels, seated in panel (b)'s clear upper-right
    # lane: every band there tops out below 0.81, and unlike the lower-left it
    # covers no part of the chance reference line. It carries identity for both
    # panels, so its handles come from panel (a) where the series were labelled.
    handles, labels = axa.get_legend_handles_labels()
    leg = axb.legend(handles, labels, fontsize=FS_LEG, loc="upper right",
                     handlelength=2.2, labelspacing=0.28, borderaxespad=0.5,
                     frameon=True, framealpha=1.0, edgecolor="none")
    leg.get_frame().set_facecolor("white")
    leg.set_zorder(6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf")
    print(f"wrote {OUT}  figsize={FIG_W/CM:.2f}x{FIG_H/CM:.2f} cm  serif={_SERIF}")


if __name__ == "__main__":
    main()
