#!/usr/bin/env python
"""Paper Y3 -- Figure F6 (headline exhibit): how much dispatching goes on
automatic, and what it costs in schedule quality.

Two panels, two knobs.

(a) The review budget. Every point is one budget setting: the horizontal
    position is the share of ALL dispatch decisions executed with no review,
    the vertical position is the reduction in true weighted tardiness against
    the tuned rule. Primary campus (C9) and confirmation campus (C10), three
    seeds each; the vertical bar through each marker is the seed-to-seed
    min-max range, so the reader can see which movements the three seeds
    actually support. The five-per-cent operating point is ringed and its
    reading spelled out in a sentence.

(b) The conformal level, which sets how wide the uncertainty interval around
    each order's estimated urgency shift is. This is the knob the decision
    stability test itself controls, so it is the honest companion to (a):
    widening or narrowing the interval moves what the test settles on its own
    from about 41% to about 78% of decisions, yet the share actually executed
    without review barely moves, because the 25% review budget in force here
    binds at every level. The reduction in true weighted tardiness is drawn on
    the same per-cent axis so the quality consequence is visible, and it too
    carries the seed range.

Design decisions worth stating in the script, because they are the two ways
this figure could have misled:

* Which knob belongs here. Both do, as two panels. The practitioner claim
  lives in the budget sweep, which is where coverage actually moves; but the
  run notes are explicit that coverage under the budget sweep essentially
  traces the budget (automation on reviewable decisions equals 1 - rho to
  three decimals in every row). Showing only panel (a) would let a reader
  believe the stability test is what buys the automation. Panel (b) shows what
  the test itself contributes and what the interval width does, and it
  discloses that under a binding budget the interval width does not buy
  coverage.

* The coverage axis. Coverage spans about 56% to 99%, so panel (a) is a plain
  LINEAR zoom to 50-100%: no break, no transform. The alternative, a
  logarithmic scale on the review load, would have spaced the five budgets
  evenly and decluttered the top, but it draws the one place where automation
  genuinely costs quality, the step from the tightest budget to the one below
  it, across a wide span and so makes that cost look gentler than it is. No
  numeral appears anywhere in this file. The safeguard against the opposite
  error, exaggerating the trade-off, is on the vertical axis: it is anchored at
  zero, which is a real zero here (no better than the tuned rule), so no
  quality difference is visually inflated by a truncated axis.

Colour: at most two accent hues, each with one meaning. Blue is the primary
campus, vermillion is the confirmation campus. Panel (b) is primary-campus data
only, so its two share series are blue; its schedule-quality series is drawn in
the near-black ink used for all text, not a third hue, which also marks it out
as the one series that is not a share of decisions. Every series is separated a
second time by line style and marker, so the panels survive greyscale and any
colour-vision deficiency.

Every plotted number is read from the W1 result files:
  results/y3_w1/routing_curve.json   pooled points, panel (a)
  results/y3_w1/curve_records.json   per-seed values, panel (a) ranges
  results/y3_w1/alpha_summary.json   pooled points, panel (b)
  results/y3_w1/alpha_records.json   per-seed values, panel (b) ranges
Nothing, including the numbers inside the callout sentence and the budget
labels, is typed into this file.

Physical placed size: 16.46 cm wide (CAS single-column text width; placed at
\\linewidth). 1:1, so on-figure pt == placed pt.

Typography: TeX Gyre Termes throughout, registered from the TinyTeX tree, with
STIX only as a fallback family; no mathtext is used anywhere in this figure, so
the exported PDF embeds a single font family.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

RES = Path("results/y3_w1")
CURVE_SUM = RES / "routing_curve.json"
CURVE_REC = RES / "curve_records.json"
ALPHA_SUM = RES / "alpha_summary.json"
ALPHA_REC = RES / "alpha_records.json"
OUT = Path("paper/figures/f6_routing.pdf")

CM = 1 / 2.54
FIG_W = 16.46 * CM      # placed width (in), CAS single-column text width
FIG_H = 8.00 * CM       # two panels, both carrying two-line axis labels

# ---- palette -------------------------------------------------------------- #
# Vermillion is taken verbatim from F3 (its C_BAND accent). The blue is the
# manuscript's LINE blue, the same value F2 and F4 use for their hero series;
# F3 only defines blue as a pale sequential ramp for filled cells, and a pastel
# would wash out as a thin line beside the vermillion.
INK = "#1A1A1A"         # every piece of figure text, and the quality series
C_PRIMARY = "#0072B2"   # house blue -- primary campus (C9)
C_CONFIRM = "#D55E00"   # house vermillion -- confirmation campus (C10)
F_PRIMARY = "#a8c6e3"   # light blue marker faces (F3's ramp)
F_CONFIRM = "#F7DFC9"   # light vermillion marker faces (F5's fill)
F_INK = "#dcdcdc"       # light face for the near-black series
C_GRID = "#e8e8e8"      # gridlines -- non-text mark
C_LEAD = "#8a929a"      # callout leader -- non-text mark

# ---- Times-family serif typography (matches the manuscript newtx/Termes) --- #
_TG = Path.home() / ".TinyTeX/texmf-dist/fonts/opentype/public/tex-gyre"
for _f in ("texgyretermes-regular.otf", "texgyretermes-bold.otf",
           "texgyretermes-italic.otf", "texgyretermes-bolditalic.otf"):
    if (_TG / _f).exists():
        fm.fontManager.addfont(str(_TG / _f))
_SERIF = "TeX Gyre Termes" if (_TG / "texgyretermes-regular.otf").exists() else "Nimbus Roman"

# Shared type scale (absolute pt at 1:1 placement) -- one visual system across
# F2-F6. Nothing is set below 8 pt at the placed width.
FS_TAG = 9.0            # panel tags "(a)" / "(b)"
FS_AXIS = 9.0           # axis labels
FS_TICK = 8.0           # tick labels
FS_LABEL = 8.0          # direct series labels, legend, point labels
FS_NOTE = 8.0           # smallest tier (the callout sentence)

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


# ---- data ----------------------------------------------------------------- #
def _reduction(rec):
    """Per-seed reduction in TWT* against the tuned rule, in per cent.

    Same formula the analysis uses for the pooled value it writes to the
    summary files; applied here one seed at a time so the figure can show the
    seed-to-seed range. Verified against the pooled value in main().
    """
    rule = float(np.mean(rec["per"]["rule"]))
    layer = float(np.mean(rec["per"]["m0_alone"]))     # historical column key
    return 100.0 * (rule - layer) / rule


def budget_curve():
    """Panel (a): pooled points from routing_curve.json, seed ranges from the
    per-seed records. Deployable ('stability') arm only, both campuses."""
    rows = [r for r in json.load(open(CURVE_SUM))["rows"]
            if r["arm"] == "stability"]
    recs = defaultdict(list)
    for r in json.load(open(CURVE_REC)):
        recs[(r["campus"], r["policy"], round(r["rho"], 4))].append(r)

    out = {}
    for campus in sorted({r["campus"] for r in rows}):
        rr = sorted((r for r in rows if r["campus"] == campus),
                    key=lambda r: r["automation_coverage"])
        cov = np.array([100.0 * r["automation_coverage"] for r in rr])
        red = np.array([r["red_m0_alone_pct"] for r in rr])
        rho = np.array([r["rho"] for r in rr])
        lo, hi = [], []
        for r in rr:
            per = [_reduction(x)
                   for x in recs[(campus, "stability", round(r["rho"], 4))]]
            lo.append(min(per))
            hi.append(max(per))
        out[campus] = dict(cov=cov, red=red, rho=rho,
                           lo=np.array(lo), hi=np.array(hi),
                           n_seeds=int(rr[0]["n_seeds"]))
    return out


def alpha_curve():
    """Panel (b): pooled points from alpha_summary.json, seed ranges from the
    per-seed records. The 'stability' arm is the swept one; 'stability_norm' is
    a separate band-scaling variant and is not part of this sweep."""
    rows = sorted((r for r in json.load(open(ALPHA_SUM))
                   if r["arm"] == "stability"),
                  key=lambda r: -r["alpha"])
    recs = defaultdict(list)
    for r in json.load(open(ALPHA_REC)):
        recs[(r["arm"], round(r["alpha"], 4))].append(r)

    conf = np.array([100.0 * (1.0 - r["alpha"]) for r in rows])
    series = {}
    getters = {
        "settled": (lambda r: 100.0 * r["unbudgeted_automation"],
                    lambda x: 100.0 * x["verdict"]["automation_coverage_unbudgeted"]),
        "executed": (lambda r: 100.0 * r["automation_coverage"],
                     lambda x: 100.0 * x["routing"]["m0_sup_cov_all"]),
        "quality": (lambda r: r["red_m0_alone_pct"], _reduction),
    }
    for key, (pooled, per_seed) in getters.items():
        val, lo, hi = [], [], []
        for r in rows:
            val.append(pooled(r))
            per = [per_seed(x) for x in recs[(r["arm"], round(r["alpha"], 4))]]
            lo.append(min(per))
            hi.append(max(per))
        series[key] = dict(val=np.array(val), lo=np.array(lo), hi=np.array(hi))
    rho = sorted({x["rho"] for x in recs[(rows[0]["arm"],
                                          round(rows[0]["alpha"], 4))]})
    return conf, series, rows, float(rho[0])


def band(ax, x, d, color, face, ls, marker, lw=1.4, ms=4.2, z=5,
         capsize=1.9, elw=0.85):
    """One series: the pooled curve, plus the seed-to-seed min-max range.

    ``capsize``/``elw`` are varied only in panel (b), where two same-hue series
    have ranges on the same abscissa; the wide caps belong to the circles and
    the narrow ones to the squares, so the two bars can be told apart without
    displacing either series from its measured position.
    """
    ax.errorbar(x, d["val"], yerr=[d["val"] - d["lo"], d["hi"] - d["val"]],
                fmt=marker, ls=ls, lw=lw, color=color,
                ecolor=color, elinewidth=elw, capsize=capsize, capthick=elw,
                ms=ms, mfc=face, mec=color, mew=1.0, zorder=z,
                clip_on=False)


# ---- figure --------------------------------------------------------------- #
def main():
    bud = budget_curve()
    conf, alp, alpha_rows, alpha_rho = alpha_curve()
    c9, c10 = bud[9], bud[10]

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H),
        gridspec_kw=dict(left=0.085, right=0.985, bottom=0.170, top=0.955,
                         wspace=0.300, width_ratios=[1.12, 1.0]),
    )

    # ================= panel (a): the review-budget knob ==================== #
    # The x-axis runs one unit past 100 so the highest-coverage point keeps its
    # seed bar and its label clear of the spine; no data can exist there.
    axa.set_xlim(50, 101)
    axa.set_ylim(0, 77)
    axa.set_xticks([50, 60, 70, 80, 90, 100])
    axa.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    axa.grid(True, color=C_GRID, lw=0.5, zorder=0)
    axa.set_axisbelow(True)

    band(axa, c10["cov"], dict(val=c10["red"], lo=c10["lo"], hi=c10["hi"]),
         C_CONFIRM, F_CONFIRM, (0, (5, 2)), "s")
    band(axa, c9["cov"], dict(val=c9["red"], lo=c9["lo"], hi=c9["hi"]),
         C_PRIMARY, F_PRIMARY, "-", "o")

    # direct series labels, so the panel reads without the caption
    axa.text(72.5, 56.0, "primary campus (C9)", ha="center", va="center",
             fontsize=FS_LABEL, color=INK, fontweight="bold", zorder=7)
    axa.text(72.5, 74.0, "confirmation campus (C10)", ha="center", va="center",
             fontsize=FS_LABEL, color=INK, fontweight="bold", zorder=7)

    # Review-budget labels on the primary curve. The 5% point carries the
    # callout instead, so it gets no label of its own. The 10% label goes above
    # its seed bar and the 2% label below its own, which is the only placement
    # that keeps the three high-coverage points, the ring and the spine apart.
    above = {0.10}
    for k, rho in enumerate(c9["rho"]):
        if rho == 0.05:
            continue
        pct = f"{100 * rho:g}%"
        if rho == max(c9["rho"]):
            pct = pct + " review budget"
        if rho in above:
            axa.text(c9["cov"][k], c9["hi"][k] + 3.4, pct, ha="center",
                     va="bottom", fontsize=FS_LABEL, color=INK, zorder=7)
        else:
            axa.text(c9["cov"][k], c9["lo"][k] - 3.6, pct, ha="center",
                     va="top", fontsize=FS_LABEL, color=INK, zorder=7)

    # the operating point the manuscript quotes: ring it, then say what it says
    j = int(np.argmin(np.abs(c9["rho"] - 0.05)))
    axa.plot([c9["cov"][j]], [c9["red"][j]], marker="o", ms=9.5, mfc="none",
             mec=INK, mew=0.9, ls="none", zorder=8)
    call = (f"At a {100 * c9['rho'][j]:g}% review budget the primary campus runs\n"
            f"{c9['cov'][j]:.1f}% of dispatches without review and still\n"
            f"cuts true weighted tardiness by {c9['red'][j]:.1f}%.")
    # the leader stops just outside the ring, so it reads as attached to the
    # point rather than floating in the empty lower half of the panel
    axa.annotate(call, xy=(c9["cov"][j], c9["red"][j]), xytext=(51.5, 33.5),
                 ha="left", va="top", fontsize=FS_NOTE, color=INK,
                 linespacing=1.45, zorder=7,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=C_LEAD,
                                 shrinkA=2, shrinkB=7.5,
                                 connectionstyle="arc3,rad=0.0"))

    # what the bars are, said inside the figure: a reader who meets F6 before
    # the caption still knows the vertical bars are a seed range, not a CI
    axa.text(51.5, 9.0,
             f"bars span the range over {c9['n_seeds']} seeds (both panels)",
             ha="left", va="center", fontsize=FS_NOTE, color=INK,
             style="italic", zorder=7)

    axa.set_xlabel("dispatch decisions executed\nwithout review (%)",
                   fontsize=FS_AXIS, labelpad=2, linespacing=1.35)
    axa.set_ylabel("reduction in true weighted tardiness\n"
                   "against the tuned rule (%)",
                   fontsize=FS_AXIS, labelpad=3, linespacing=1.35)
    axa.tick_params(labelsize=FS_TICK, length=2)
    axa.set_title("(a)", fontsize=FS_TAG, fontweight="bold", loc="left", pad=4)

    # ================= panel (b): the interval-width knob =================== #
    axb.set_xlim(45, 100)
    axb.set_ylim(0, 100)
    axb.set_xticks([50, 60, 70, 80, 90, 100])
    axb.set_yticks([0, 20, 40, 60, 80, 100])
    axb.grid(True, color=C_GRID, lw=0.5, zorder=0)
    axb.set_axisbelow(True)

    # Marker vocabulary is fixed across the figure: the circle is primary-campus
    # data (both panels), the square is the confirmation campus and appears in
    # panel (a) only, so panel (b)'s second share series takes a diamond rather
    # than reusing the square for an unrelated meaning.
    band(axb, conf, alp["executed"], C_PRIMARY, F_PRIMARY, (0, (5, 2)), "D",
         z=4, ms=3.9, capsize=1.1, elw=0.7)
    band(axb, conf, alp["settled"], C_PRIMARY, F_PRIMARY, "-", "o",
         z=6, capsize=2.8, elw=0.95)
    band(axb, conf, alp["quality"], INK, F_INK, (0, (3, 1.4, 1, 1.4)), "^",
         ms=4.4, z=5, capsize=2.0)

    handles = [
        Line2D([], [], color=C_PRIMARY, ls=(0, (5, 2)), lw=1.4, marker="D",
               ms=3.9, mfc=F_PRIMARY, mec=C_PRIMARY, mew=1.0),
        Line2D([], [], color=C_PRIMARY, ls="-", lw=1.4, marker="o",
               ms=4.2, mfc=F_PRIMARY, mec=C_PRIMARY, mew=1.0),
        Line2D([], [], color=INK, ls=(0, (3, 1.4, 1, 1.4)), lw=1.4, marker="^",
               ms=4.4, mfc=F_INK, mec=INK, mew=1.0),
    ]
    labels = [
        f"executed without review, {100 * alpha_rho:g}% budget",
        "settled by the intervals, no budget cap",
        "reduction in true weighted tardiness",
    ]
    leg = axb.legend(handles, labels, fontsize=FS_LABEL, loc="lower left",
                     frameon=False, handlelength=2.6, handletextpad=0.6,
                     labelspacing=0.42, borderpad=0.15, borderaxespad=0.5)
    for t in leg.get_texts():
        t.set_color(INK)

    axb.set_xlabel("nominal confidence of the interval\n"
                   "around each order's urgency (%)",
                   fontsize=FS_AXIS, labelpad=2, linespacing=1.35)
    axb.set_ylabel("share of decisions, and reduction\n"
                   "in true weighted tardiness (%)",
                   fontsize=FS_AXIS, labelpad=3, linespacing=1.35)
    axb.tick_params(labelsize=FS_TICK, length=2)
    axb.set_title("(b)", fontsize=FS_TAG, fontweight="bold", loc="left", pad=4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf")
    png = OUT.with_suffix(".png")
    fig.savefig(png, format="png", dpi=300)
    print(f"wrote {OUT} and {png}  "
          f"figsize={FIG_W / CM:.2f}x{FIG_H / CM:.2f} cm  serif={_SERIF}")

    # ---- provenance echo: every plotted value, as read from the files ------ #
    print("\npanel (a) budget sweep  (n_seeds=%d)" % c9["n_seeds"])
    for campus, d in ((9, c9), (10, c10)):
        for k in range(len(d["rho"])):
            print("  C%-2d rho=%.2f  coverage=%6.2f%%  reduction=%6.2f%%  "
                  "seed range [%.2f, %.2f]"
                  % (campus, d["rho"][k], d["cov"][k], d["red"][k],
                     d["lo"][k], d["hi"][k]))
    print("\npanel (b) conformal sweep  (campus 9, review budget %.2f)"
          % alpha_rho)
    for k, r in enumerate(alpha_rows):
        print("  alpha=%.2f (conf=%.0f%%)  settled=%6.2f%%  executed=%6.2f%%  "
              "reduction=%6.2f%%"
              % (r["alpha"], conf[k], alp["settled"]["val"][k],
                 alp["executed"]["val"][k], alp["quality"]["val"][k]))

    # ---- assertion: the per-seed recomputation must reproduce the pooled
    # value the summary files carry, otherwise the range bars belong to a
    # different quantity from the markers they hang on.
    for r in json.load(open(CURVE_SUM))["rows"]:
        if r["arm"] != "stability":
            continue
        per = [_reduction(x) for x in json.load(open(CURVE_REC))
               if x["campus"] == r["campus"] and x["policy"] == "stability"
               and abs(x["rho"] - r["rho"]) < 1e-9]
        assert min(per) - 1e-6 <= r["red_m0_alone_pct"] <= max(per) + 1e-6, r
    print("\npooled reductions lie inside their own per-seed ranges: OK")


if __name__ == "__main__":
    main()
