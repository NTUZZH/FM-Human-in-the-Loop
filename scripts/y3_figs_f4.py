#!/usr/bin/env python
"""Paper Y3 -- Figure F4: recovery / information-budget curve.

x = cumulative number of supervisor overrides the correction layer has seen.
Top panel  (a): fraction of the RULE -> ORACLE TWT* gap the deployed correction
                layer closes.
Bottom (b): sign accuracy of the recovered latent urgency estimate hat_s.
One line per recoverable share beta (0.5 / 0.75 / 1.0). Points are the mean over
3 seeds at each DAgger iteration; band = +/-1 s.d. across seeds. Colourblind-safe
Okabe-Ito hues plus distinct markers and line styles (identity is never
colour-alone). Source: live_m0 rows of e4_recovery.csv.

Physical placed size: 8.4 cm wide (CAS single-column), placed 1:1.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CSV = Path("results/y3_p5/ablations/e4_recovery.csv")
OUT = Path("paper/figures/f4_recovery.pdf")

CM = 1 / 2.54
FIG_W = 8.4 * CM
FIG_H = 9.4 * CM

BETAS = [0.5, 0.75, 1.0]
# Okabe-Ito CVD-safe hues (blue / orange / bluish-green), one per beta.
STYLE = {
    0.5:  dict(color="#009E73", marker="o", ls=(0, (1, 1)), label=r"$\beta=0.5$"),
    0.75: dict(color="#E69F00", marker="s", ls=(0, (5, 2)), label=r"$\beta=0.75$"),
    1.0:  dict(color="#0072B2", marker="^", ls="-",          label=r"$\beta=1.0$"),
}

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.0, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
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


def main():
    data = agg()
    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(FIG_W, FIG_H), sharex=True,
        gridspec_kw=dict(left=0.155, right=0.965, bottom=0.115, top=0.905,
                         hspace=0.16),
    )

    for b in BETAS:
        a, s = data[b], STYLE[b]
        # (a) gap closed
        axa.fill_between(a.cum, a.gc - a.gc_sd, a.gc + a.gc_sd,
                         color=s["color"], alpha=0.15, linewidth=0)
        axa.plot(a.cum, a.gc, color=s["color"], marker=s["marker"], ls=s["ls"],
                 ms=4.5, mew=0.6, mec="white", lw=1.6, label=s["label"])
        # (b) sign accuracy
        axb.fill_between(a.cum, a.sa - a.sa_sd, a.sa + a.sa_sd,
                         color=s["color"], alpha=0.15, linewidth=0)
        axb.plot(a.cum, a.sa, color=s["color"], marker=s["marker"], ls=s["ls"],
                 ms=4.5, mew=0.6, mec="white", lw=1.6)

    # reference lines
    axa.axhline(1.0, color="#555555", lw=0.8, ls=(0, (2, 2)), zorder=1)
    axa.text(2380, 1.0, "omniscient\nreference", ha="right", va="top",
             fontsize=6, color="#555555")
    axb.axhline(0.5, color="#555555", lw=0.8, ls=(0, (2, 2)), zorder=1)
    axb.text(2380, 0.505, "chance", ha="right", va="bottom",
             fontsize=6, color="#555555")

    axa.set_ylabel("fraction of RULE$\\to$ORACLE\ngap closed", fontsize=7.5)
    axa.set_ylim(0.55, 1.03)
    axa.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    axa.tick_params(labelsize=7, length=2)
    axa.grid(True, color="#e8e8e8", lw=0.5)
    axa.set_axisbelow(True)
    leg = axa.legend(fontsize=6.8, loc="lower right", handlelength=2.2,
                     labelspacing=0.25, borderaxespad=0.4, frameon=True,
                     framealpha=0.9, edgecolor="none")
    leg.get_frame().set_facecolor("white")
    axa.set_title("(a) deployed value: dispatch-quality gap closed",
                  fontsize=7.5, loc="left", pad=3)

    axb.set_ylabel("$\\hat{s}$ sign accuracy", fontsize=7.5)
    axb.set_ylim(0.48, 0.86)
    axb.set_yticks([0.5, 0.6, 0.7, 0.8])
    axb.tick_params(labelsize=7, length=2)
    axb.grid(True, color="#e8e8e8", lw=0.5)
    axb.set_axisbelow(True)
    axb.set_xlabel("cumulative supervisor overrides seen", fontsize=7.5)
    axb.set_title("(b) mechanism: recovery of the latent urgency",
                  fontsize=7.5, loc="left", pad=3)
    axb.set_xlim(500, 2450)

    # highlight the small-budget knee
    axa.annotate("most of the gap closed\nby the first review budget",
                 xy=(600, data[1.0].gc.iloc[0]), xytext=(950, 0.600),
                 fontsize=6, color="#333333", ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=0.7))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf")
    print(f"wrote {OUT}  figsize={FIG_W/CM:.2f}x{FIG_H/CM:.2f} cm")


if __name__ == "__main__":
    main()
