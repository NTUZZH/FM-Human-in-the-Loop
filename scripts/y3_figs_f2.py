#!/usr/bin/env python
"""Paper Y3 -- Figure F2: closing the loop (H3 override burden + dispatch quality).

Two panels side by side, full text width (16.46 cm wide, placed 1:1 at \\linewidth
in a figure* float), so all text renders at its native size (>= 6.8 pt).

(a) override burden. Supervisor override rate per training round (x = round 0..7;
    one outer data-aggregation round of the training loop). The learned policy M1
    (mean over 10 primary-cell seeds 301-310, band = +/-1 s.d.) falls from ~0.052
    to ~0.003, against the FLAT tuned-rule-plus-supervisor override line at 0.097.
    The falling-vs-flat contrast is H3.

(b) quality trajectory. The policy's true objective TWT*(w*,d*) over the same
    training rounds (mean over the 10 seeds, band = +/-1 s.d.), measured on the
    TRAINING ROLLOUTS. Dashed reference lines are HELD-OUT evaluations at the same
    cell: RULE 3644.8, RULE+SUP 2692.8, M0 (correction layer) 1991.5, ORACLE
    1815.0. The curve is a training-rollout average and the lines are held-out, so
    the curve is read for its trend and its position between the lines, not
    point-for-point.

Text colour: every label is near-black INK; the neutral reference LINES stay grey
(grey is allowed for non-text marks only). Series identity is carried by direct
labels, mark type and line style, never by lightness.

Primary cell: c9 storm2 u100 beta=1.0 rho=0.25 eps=0 TARGETED, fair-M1
(deadline_head=True). Data:
  - override_rate / true_twt trajectories: the 10 completed runs
    train_log/y3_sweep/m1_c9_u100_b1_r0.25_s301..s310/metrics.csv
  - flat RULE+SUP override rate + the four TWT*(w*,d*) reference means:
    results/y3_p5/harvest/primary_multiseed_summary.json
    (the committed 10-seed harvest; std convention = population std, ddof=0).

Typography: TeX Gyre Termes (the serif newtxtext is built on) with STIX mathtext,
matching the manuscript's newtx Times body; one shared type scale and palette
across F2-F5 (Okabe-Ito hues, identity never by colour alone; series labelled
directly, no legend). Smallest text >= 6.5 pt at the placed width.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SWEEP = Path("train_log/y3_sweep")
SUMMARY = Path("results/y3_p5/harvest/primary_multiseed_summary.json")
OUT = Path("paper/figures/f2_loop.pdf")

SEEDS = list(range(301, 311))          # the 10 completed primary-cell runs
CELL_DIR = "m1_c9_u100_b1_r0.25_s{s}"

CM = 1 / 2.54
FIG_W = 16.46 * CM      # full text width; placed 1:1 at \linewidth in figure*
FIG_H = 6.8 * CM        # side-by-side two-panel layout

# Okabe-Ito CVD-safe hues, matched to F3/F4/F5:
C_M1 = "#0072B2"        # blue  -- the learned policy (hero series, both panels)
C_SUP = "#D55E00"       # vermillion -- RULE+SUP status quo (both panels)
C_M0 = "#009E73"        # bluish-green -- the correction layer M0
C_GREY = "#666666"      # neutral bounds (RULE, ORACLE) -- LINE colour only
INK = "#1A1A1A"         # every piece of figure text; no grey type anywhere

# ---- Times-family serif typography (matches the manuscript newtx/Termes body) ---
# Register TeX Gyre Termes (the font newtxtext is built on) from the local TinyTeX
# tree; pair it with STIX for math so TWT*(w*,d*), beta, rho, s-hat all render in a
# Times-metric serif. No DejaVu anywhere. Falls back to Nimbus Roman (also
# Times-metric) if the Termes OTFs are not present.
_TG = Path.home() / ".TinyTeX/texmf-dist/fonts/opentype/public/tex-gyre"
for _f in ("texgyretermes-regular.otf", "texgyretermes-bold.otf",
           "texgyretermes-italic.otf", "texgyretermes-bolditalic.otf"):
    if (_TG / _f).exists():
        fm.fontManager.addfont(str(_TG / _f))
_SERIF = "TeX Gyre Termes" if (_TG / "texgyretermes-regular.otf").exists() else "Nimbus Roman"

# Shared type scale (absolute pt at 1:1 placement) -- one visual system across
# F2-F4. Three sizes; nothing set below 8 pt at the placed width. The only
# glyphs under 8 pt are mathtext superscripts, which are 0.7x their base by
# construction; FS_AXIS_SUP raises the base of the one label that carries them.
FS_TITLE = 9.0          # panel titles "(a) ..."
FS_AXIS = 9.0           # axis labels
FS_AXIS_SUP = 9.7       # axis label carrying superscripts (0.7x -> 6.8 pt)
FS_TICK = 8.0           # tick labels
FS_LABEL = 8.0          # direct series labels
FS_SMALL = 8.0          # smallest tier (reference-line labels)
LW_DATA = 1.7           # data line
LW_REF = 1.0            # thin reference line
MS_DATA = 4.6           # marker diameter (pt)
BAND_A = 0.18           # 1-sigma band fill alpha

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": [_SERIF, "Nimbus Roman", "Liberation Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8.0, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 300,
})


def load_trajectories():
    """Mean and population s.d. (ddof=0) over the 10 seeds, per iteration."""
    orr, twt = [], []
    for s in SEEDS:
        f = SWEEP / CELL_DIR.format(s=s) / "metrics.csv"
        d = pd.read_csv(f).sort_values("iter")
        assert len(d) == 8, f"{f}: expected 8 iters, got {len(d)}"
        orr.append(d["override_rate"].to_numpy())
        twt.append(d["true_twt"].to_numpy())
    orr = np.vstack(orr)               # [10, 8]
    twt = np.vstack(twt)
    it = np.arange(orr.shape[1])
    return (it,
            orr.mean(0), orr.std(0),   # ddof=0, committed convention
            twt.mean(0), twt.std(0))


def load_refs():
    d = json.loads(SUMMARY.read_text())
    lad = d["ladder"]
    refs = {
        "RULE": lad["rule"]["twt_mean"],
        "RULE+SUP": lad["rule_sup"]["twt_mean"],
        "M0": lad["m0_alone"]["twt_mean"],
        "ORACLE": lad["oracle"]["twt_mean"],
    }
    flat_or = d["H3_override_rate"]["rule_sup_override_rate_flat_mean"]
    return refs, flat_or


def main():
    it, or_m, or_s, tw_m, tw_s = load_trajectories()
    refs, flat_or = load_refs()

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H),
        gridspec_kw=dict(left=0.078, right=0.995, bottom=0.150, top=0.900,
                         wspace=0.235),
    )

    # ---- panel (a): override rate -------------------------------------------
    axa.axhline(flat_or, color=C_SUP, lw=1.4, ls=(0, (5, 2)), zorder=2)
    axa.fill_between(it, or_m - or_s, or_m + or_s, color=C_M1, alpha=BAND_A,
                     linewidth=0, zorder=2)
    axa.plot(it, or_m, color=C_M1, marker="^", ls="-", ms=MS_DATA, mew=0.7,
             mec="white", lw=LW_DATA, zorder=4)

    # direct labels (no legend); every label sits on the mark it names, so
    # position carries identity and all text can stay near-black.
    axa.text(6.9, flat_or + 0.0035, "fixed rule + supervisor",
             ha="right", va="bottom", fontsize=FS_LABEL, color=INK,
             fontweight="bold")
    axa.text(2.55, 0.0245, "learned policy (M1)", ha="left", va="center",
             fontsize=FS_LABEL, color=INK, fontweight="bold")
    axa.annotate("", xy=(1.02, or_m[1] + 0.0015), xytext=(2.5, 0.0225),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.7))

    axa.set_ylim(0.0, 0.108)
    axa.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    axa.set_ylabel("supervisor override rate", fontsize=FS_AXIS)
    axa.tick_params(labelsize=FS_TICK, length=2)
    axa.grid(True, color="#e8e8e8", lw=0.5)
    axa.set_axisbelow(True)
    axa.set_xlabel("training round", fontsize=FS_AXIS)
    axa.set_xlim(-0.25, 7.25)
    axa.set_xticks(range(8))
    axa.set_title("(a) override burden: learner falls, rule stays flat",
                  fontsize=FS_TITLE, loc="left", pad=4)

    # ---- panel (b): true objective TWT*(w*,d*) ------------------------------
    ref_style = {
        "RULE":     dict(color=C_GREY, ls=(0, (4, 2))),
        "RULE+SUP": dict(color=C_SUP,  ls=(0, (5, 2))),
        "M0":       dict(color=C_M0,   ls=(0, (4, 2))),
        "ORACLE":   dict(color=C_GREY, ls=(0, (1, 1.2))),
    }
    ref_lab = {
        "RULE": "RULE", "RULE+SUP": "RULE+SUP",
        "M0": "SURGE", "ORACLE": "ORACLE",
    }
    # M0 (1991.5) and ORACLE (1815.0) lines are close; offset their labels so
    # they never collide (M0 label sits higher above its line).
    ref_dy = {"RULE": 26, "RULE+SUP": 26, "M0": 40, "ORACLE": 24}
    for name, y in refs.items():
        st = ref_style[name]
        axb.axhline(y, color=st["color"], lw=LW_REF, ls=st["ls"], zorder=1)
        # Label text is near-black for every line, including the two neutral
        # bounds: hierarchy is bold (the two lines the argument turns on) against
        # regular, plus each label sitting directly on its own line.
        axb.text(6.95, y + ref_dy[name], ref_lab[name], ha="right", va="bottom",
                 fontsize=FS_SMALL, color=INK,
                 fontweight="bold" if name in ("RULE+SUP", "M0") else "normal")

    axb.fill_between(it, tw_m - tw_s, tw_m + tw_s, color=C_M1, alpha=BAND_A,
                     linewidth=0, zorder=2)
    axb.plot(it, tw_m, color=C_M1, marker="^", ls="-", ms=MS_DATA, mew=0.7,
             mec="white", lw=LW_DATA, zorder=4)
    axb.text(0.08, tw_m[0] + 75, "M1 (training rollout)", ha="left",
             va="bottom", fontsize=FS_LABEL, color=INK, fontweight="bold")

    axb.set_ylim(1750, 3820)
    axb.set_yticks([2000, 2500, 3000, 3500])
    # notation exactly as the manuscript body sets it: \mathrm{TWT}^*(w^*,d^*),
    # with the star superscripted on TWT too and no gap before the comma
    axb.set_ylabel("true weighted tardiness  $\\mathrm{TWT}^{*}(w^{*}\\!,d^{*})$",
                   fontsize=FS_AXIS_SUP)
    axb.tick_params(labelsize=FS_TICK, length=2)
    axb.grid(True, color="#e8e8e8", lw=0.5)
    axb.set_axisbelow(True)
    axb.set_xlabel("training round", fontsize=FS_AXIS)
    axb.set_xlim(-0.25, 7.25)
    axb.set_xticks(range(8))
    axb.set_title("(b) quality trajectory: objective falls each round",
                  fontsize=FS_TITLE, loc="left", pad=4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf")
    print(f"wrote {OUT}  figsize={FIG_W/CM:.2f}x{FIG_H/CM:.2f} cm  serif={_SERIF}")
    print("override mean :", [round(float(x), 5) for x in or_m])
    print("override sd   :", [round(float(x), 6) for x in or_s])
    print("flat RULE+SUP :", round(float(flat_or), 5))
    print("true_twt mean :", [round(float(x), 1) for x in tw_m])
    print("true_twt sd   :", [round(float(x), 1) for x in tw_s])
    print("refs          :", {k: round(v, 1) for k, v in refs.items()})


if __name__ == "__main__":
    main()
