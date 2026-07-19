#!/usr/bin/env python
"""
P5 - All figures for the manuscript (cas-dc, double column).

One script, all figures. Renders <fig>.pdf (vector, for LaTeX) + <fig>.png (QA).
Design follows the dataviz skill: fixed categorical order, CVD-validated palette
(worst adjacent dE 16.6 on the tie-bundle; warm outliers 11.2 are mitigated by
direct labels per the skill's secondary-encoding rule), thin marks, hairline
recessive axes, selective direct labels, legend for >=2 series, no dual axes.

Method -> color map is FROZEN and identical across every figure.

  conda activate fjsp && python scripts/p5_figures.py [f1 f2 f3 f4 f5 f6]
"""
import sys, math
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
import matplotlib.ticker as mticker

from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
FIGDIR = f"{ROOT}/paper/figures"

# ----------------------------------------------------------------------------
# Design tokens (dataviz reference palette, light surface)
# ----------------------------------------------------------------------------
INK      = "#0b0b0b"   # primary text
INK2     = "#52514e"   # secondary text
MUTE     = "#898781"   # axis / muted labels
GRID     = "#e1e0d9"   # hairline gridline
AXIS     = "#c3c2b7"   # baseline / axis
SURF     = "#fcfcfb"   # chart surface
TIEGRAY  = "#c3c2b7"   # "top tier" shared color (neutral -> reads as "no winner")
TIEFILL  = "#d9d8d1"   # lighter tie fill for map cells

# FROZEN method -> color (validated). Order = paper legend order.
CMAP = {
    "edd":   "#008300",  # green
    "pfifo": "#8f8d86",  # muted gray (non-focal; ~edd by construction)
    "atc":   "#4a3aa7",  # violet
    "wspt":  "#eb6834",  # orange
    "mor":   "#e34948",  # red (collapse)
    "random":"#c0beb6",  # light muted gray (non-focal floor)
    "ga":    "#eda100",  # yellow
    "cpsat": "#e87ba4",  # magenta (cpsat60 / cpsat300)
    "roll":  "#1baf7a",  # aqua (Rolling CP-SAT)
    "policy":"#2a78d6",  # blue (learned policy, protagonist)
}
PRETTY = {"edd":"EDD","pfifo":"pFIFO","atc":"ATC","wspt":"WSPT","mor":"MOR",
          "random":"Random","ga":"GA","cpsat":"CP-SAT","cpsat60":"CP-SAT 60s",
          "cpsat300":"CP-SAT 300s","roll":"Rolling CP-SAT","policy":"Policy"}

def mcol(m):
    """color for a raw method token from any results file."""
    m = m.lower()
    if m.startswith("v2rl") or m in ("policy",): return CMAP["policy"]
    if m.startswith("rl3"): return CMAP["policy"]
    if m.startswith("rollcp") or m == "roll": return CMAP["roll"]
    if m.startswith("cpsat"): return CMAP["cpsat"]
    return CMAP.get(m, MUTE)

# ----------------------------------------------------------------------------
# Matplotlib rcParams tuned for cas-dc print sizes
# ----------------------------------------------------------------------------
def set_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        # Times New Roman everywhere (journal body font). "Nimbus Roman" is the
        # URW clone metrically identical to Times New Roman (what Linux ships);
        # STIX is the Times-compatible math companion.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.2, "axes.titlesize": 7.8, "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.6,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.5,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTE, "ytick.color": MUTE,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.4, "lines.markersize": 4.5,
        "legend.frameon": False, "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5, "legend.columnspacing": 1.1,
        "legend.labelspacing": 0.35, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.axisbelow": True, "savefig.facecolor": SURF, "figure.facecolor": SURF,
    })

MM = 1/25.4
def figsize(w_mm, h_mm): return (w_mm*MM, h_mm*MM)

def style_ax(ax):
    ax.set_facecolor(SURF)
    for s in ("left","bottom"): ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(0.5)
    ax.tick_params(length=2.2, width=0.5, colors=MUTE)
    return ax

def save(fig, name, tight=True):
    kw = dict(bbox_inches="tight", pad_inches=0.02) if tight else {}
    fig.savefig(f"{FIGDIR}/{name}.pdf", **kw)
    fig.savefig(f"{FIGDIR}/{name}.png", dpi=300, **kw)
    plt.close(fig)
    print(f"  wrote {name}.pdf + .png")

def fmt_wwt(v):
    if v < 10: return f"{v:.1f}"
    if v < 1000: return f"{v:.0f}"
    return f"{v/1000:.1f}k"

# ============================================================================
# F1  pipeline schematic
# ============================================================================
def fig1_pipeline():
    fig = plt.figure(figsize=figsize(180, 74))
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,180); ax.set_ylim(0,74); ax.axis("off")

    def box(x, y, w, h, title, sub=None, fc=SURF, ec=AXIS, tc=INK, lw=0.8, accent=None, fs=7.2):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.8",
                           linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
        ax.add_patch(p)
        if accent:  # left accent bar
            ax.add_patch(Rectangle((x, y), 1.6, h, facecolor=accent, edgecolor="none", zorder=3))
        cy = y + h/2 + (1.7 if sub else 0)
        ax.text(x+w/2+ (0.8 if accent else 0), cy, title, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold", zorder=4)
        if sub:
            ax.text(x+w/2+(0.8 if accent else 0), y+h/2-2.4, sub, ha="center", va="center",
                    fontsize=6.1, color=INK2, zorder=4)
        return (x, y, w, h)

    def arrow(a, b, side_a="r", side_b="l", color=MUTE, lw=1.0, rad=0.0):
        ax_, ay_, aw, ah = a; bx_, by_, bw, bh = b
        pa = {"r":(ax_+aw, ay_+ah/2), "l":(ax_, ay_+ah/2), "t":(ax_+aw/2, ay_+ah), "b":(ax_+aw/2, ay_)}[side_a]
        pb = {"r":(bx_+bw, by_+bh/2), "l":(bx_, by_+bh/2), "t":(bx_+bw/2, by_+bh), "b":(bx_+bw/2, by_)}[side_b]
        ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-|>", mutation_scale=8,
                     linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=1))

    # --- stage 1: source + cleaning (left column) ---
    b_src = box(3, 46, 30, 15, "FMUCD", "raw work-order log", accent=CMAP["policy"])
    b_cln = box(3, 22, 30, 15, "Cleaning R1-R7", "de-dup, priority + trade maps", accent=CMAP["policy"])
    arrow(b_src, b_cln, "b", "t", lw=1.1)

    # --- stage 2: two tracks ---
    b_rep = box(43, 49, 40, 13, "[R] Replay track", "first-N releases, non-overlap", accent=CMAP["edd"])
    b_gen = box(43, 22, 40, 13, "[C] Generator track", "fitted packs + contention knobs", accent=CMAP["wspt"])
    arrow(b_cln, b_rep, "r", "l", rad=-0.18, lw=1.1)
    arrow(b_cln, b_gen, "r", "l", rad=0.12, lw=1.1)

    # --- stage 3: instances (merge) ---
    # counts are COMPUTED from the released result files so the schematic can
    # never drift out of sync with what was actually evaluated.
    _e1 = pd.read_csv(f"{ROOT}/results/e1_static/results.csv", usecols=["id"])
    _dy = pd.read_csv(f"{ROOT}/results/p4_dyneval/results.csv",
                      usecols=["id", "regime", "crew_multiplier",
                               "arrival_multiplier", "pm_share_override"])
    _n_static = _e1["id"].nunique()
    _n_dyn = len(_dy.fillna(-1).drop_duplicates())
    del _e1, _dy
    b_ins = box(92, 35, 30, 15, "Benchmark\ninstances",
                f"{_n_static:,} static / {_n_dyn:,} dynamic", accent=INK2)
    arrow(b_rep, b_ins, "r", "l", rad=0.14, lw=1.1)
    arrow(b_gen, b_ins, "r", "l", rad=-0.14, lw=1.1)

    # --- stage 4: methods suite ---
    b_met = box(131, 35, 46, 15, "", None, accent=INK2)
    ax.text(155.5, 47.4, "Methods suite", ha="center", va="center", fontsize=7.2, color=INK, weight="bold", zorder=5)
    chips = [("EDD","edd"),("ATC","atc"),("WSPT","wspt"),("MOR","mor"),("pFIFO","pfifo"),
             ("Rand","random"),("GA","ga"),("CP","cpsat"),("Roll","roll"),("Policy","policy")]
    col_x = [133.5, 142.1, 150.7, 159.3, 167.9]; row_y = [43.0, 38.4]
    for i,(lab,key) in enumerate(chips):
        xx = col_x[i % 5]; yy = row_y[i // 5]
        ax.add_patch(Rectangle((xx, yy), 1.9, 1.9, facecolor=mcol(key), edgecolor="none", zorder=4))
        ax.text(xx+2.3, yy+0.95, lab, ha="left", va="center", fontsize=5.6, color=INK2, zorder=4)
    arrow(b_ins, b_met, "r", "l", lw=1.1)

    # --- stage 5: validator + outputs (down) ---
    b_val = box(131, 11, 46, 13, "Independent validator", "feasibility + weighted tardiness", accent=CMAP["mor"])
    arrow(b_met, b_val, "b", "t", lw=1.1)
    b_out = box(92, 11, 30, 13, "Metrics +\ndecision map", None, accent=CMAP["policy"])
    arrow(b_val, b_out, "l", "r", lw=1.1)

    # flow band labels
    ax.text(18, 66.5, "1  Data + cleaning", fontsize=6.4, color=MUTE, ha="center", style="italic")
    ax.text(63, 66.5, "2  Two instance tracks", fontsize=6.4, color=MUTE, ha="center", style="italic")
    ax.text(107, 55.5, "3  Instances", fontsize=6.4, color=MUTE, ha="center", style="italic")
    ax.text(154, 55.5, "4  Schedulers", fontsize=6.4, color=MUTE, ha="center", style="italic")
    ax.text(129, 4.5, "5  Score + characterize (independent of the schedulers)", fontsize=6.4, color=MUTE, ha="center", style="italic")

    save(fig, "f1_pipeline")

# ============================================================================
# F2  static benchmark (2 panels)
# ============================================================================
def fig2_static():
    e1 = pd.read_csv(f"{ROOT}/results/e1_static/results.csv")
    piv = e1.pivot_table(index="id", columns="method", values="wwt", aggfunc="first")
    bk = piv.min(axis=1); nz = bk > 1e-9
    gap = piv.sub(bk, axis=0)
    lat = e1.groupby("method")["wall_seconds"].mean() * 1000.0  # ms

    order = ["cpsat300","cpsat60","ga","atc","wspt","pfifo","edd","random","mor"]
    gaps = {m: gap[m][nz].mean() for m in order}
    ga_beats = int((piv["ga"] < piv["cpsat60"] - 1.0).sum())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize(180, 66), gridspec_kw=dict(wspace=0.42))

    # ---- Panel A: ordered horizontal bars, log-x gap ----
    style_ax(axA)
    ypos = np.arange(len(order))[::-1]
    focal = {"cpsat300","cpsat60","ga","atc","mor"}     # emphasize; mute the rest
    for y, m in zip(ypos, order):
        c = mcol(m) if m in focal else MUTE
        alpha = 1.0 if m in focal else 0.55
        axA.barh(y, gaps[m], height=0.62, color=c, alpha=alpha, edgecolor=SURF, linewidth=0.6, zorder=3)
        axA.text(gaps[m]*1.14, y, fmt_wwt(gaps[m]), va="center", ha="left", fontsize=6.0, color=INK2, zorder=4)
    axA.set_yticks(ypos)
    axA.set_yticklabels([PRETTY.get(m,m) for m in order], fontsize=6.8, color=INK)
    axA.set_xscale("log")
    axA.set_xlim(0.1, 2200)
    axA.set_xlabel("Mean gap to best-known TWT  (weighted tardiness, log)", fontsize=6.9)
    axA.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))
    axA.grid(axis="x", which="major", color=GRID, linewidth=0.5)
    axA.set_axisbelow(True)
    # tier brackets
    axA.text(0.14, 8.55, "exact / near-exact", fontsize=6.1, color=INK2, style="italic")
    axA.text(15, 5.55, "dispatch rules", fontsize=6.1, color=INK2, style="italic")
    axA.text(120, 1.35, "naive floor", fontsize=6.1, color=INK2, style="italic")
    # GA beats cpsat60 annotation
    # leader lands INSIDE the GA bar (not at its tip) so the curve stays well
    # clear of the '4.3' value label sitting just right of the bar end.
    axA.annotate(f"GA beats CP-SAT 60s\non {ga_beats} hard instances",
                 xy=(gaps["ga"]*0.55, ypos[order.index("ga")]),
                 xytext=(9, 7.9), fontsize=6.1, color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.6,
                                 connectionstyle="arc3,rad=-0.2"))
    axA.set_title(f"(a)  Solution quality on nonzero instances (n={int(nz.sum()):,})", loc="left",
                  fontsize=7.4, color=INK, weight="bold", pad=6)

    # ---- Panel B: latency vs quality scatter ----
    style_ax(axB)
    # EDD and pFIFO coincide by construction (gap 73 vs 72, same latency) -> one dot,
    # drawn in EDD green and labelled 'EDD / pFIFO'. Panel (a) keeps both bars.
    order_b = [m for m in order if m != "pfifo"]
    LAB_B = dict(PRETTY); LAB_B["edd"] = "EDD / pFIFO"
    OFF_B = {  # deterministic label offsets in POINTS: (dx, dy, ha, va)
        "mor":      ( 7,  0, "left",   "center"),
        "random":   ( 4,  6, "left",   "bottom"),
        "edd":      (12, 11, "left",   "bottom"),  # up-right into clear gap (frontier line + WSPT box it in at dot level)
        "wspt":     ( 7,  0, "left",   "center"),
        "atc":      ( 7,  0, "left",   "center"),
        "ga":       ( 7,  0, "left",   "center"),
        "cpsat60":  ( 0,  8, "right",  "bottom"),  # up-left (right label overruns the panel edge)
        "cpsat300": (-7,  0, "right",  "center"),
    }
    for m in order_b:
        x, y = lat[m], gaps[m]
        axB.scatter(x, y, s=42, color=mcol(m), edgecolor=SURF, linewidth=0.9, zorder=4)
        dx, dy, ha, va = OFF_B[m]
        axB.annotate(LAB_B.get(m, m), (x, y), textcoords="offset points", xytext=(dx, dy),
                     fontsize=6.1, color=INK2, ha=ha, va=va)
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlim(0.25, 12000); axB.set_ylim(0.1, 1200)
    axB.set_xlabel("Decision latency per instance  (ms, log)", fontsize=6.9)
    axB.set_ylabel("Mean gap to best-known TWT  (log)", fontsize=6.9)
    axB.grid(True, which="major", color=GRID, linewidth=0.5)
    # guide regions
    axB.text(0.7, 0.16, "fast, coarse", fontsize=6.1, color=MUTE, style="italic")
    axB.text(1600, 300, "slow, exact", fontsize=6.1, color=MUTE, style="italic", ha="center")
    axB.annotate("", xy=(3500, 0.3), xytext=(0.5, 300),
                 arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
    axB.text(35, 3.2, "quality / latency\nfrontier", fontsize=6.1, color=INK2, style="italic", ha="center", rotation=-33)
    axB.set_title("(b)  Latency vs quality trade-off", loc="left",
                  fontsize=7.4, color=INK, weight="bold", pad=6)
    save(fig, "f2_static")

# ============================================================================
# F3  intensity curves (4 small multiples)
# ============================================================================
def fig3_curves():
    c = pd.read_csv(f"{ROOT}/results/p4_dyneval/e2_curve.csv")
    # Policy = MEAN across the ten MLP seeds (protocol forbids best-of-seeds);
    # a min..max band across seeds shows honest seed spread (widens on campus 5 u>=1).
    POL_SEEDS = [f"v2rl{s}" for s in range(301, 311)]
    series = [("edd","EDD"),("atc","ATC"),("wspt","WSPT"),("mor","MOR"),("policy","Policy (10 seeds)")]
    # Per-series styles so tied lines that coincide exactly stay distinguishable:
    # thick EDD underneath, dashed ATC over it (gaps reveal green), Policy markers on top.
    STYLE = {
        "edd":    dict(ls="-",         lw=2.4, alpha=0.85, zorder=3, marker="",  ms=0),
        "atc":    dict(ls=(0,(4,1.8)), lw=1.5, alpha=1.0,  zorder=4, marker="",  ms=0),
        "wspt":   dict(ls="-",         lw=1.3, alpha=1.0,  zorder=3, marker="",  ms=0),
        "mor":    dict(ls="-",         lw=1.3, alpha=1.0,  zorder=4, marker="",  ms=0),
        "policy": dict(ls="-",         lw=1.2, alpha=1.0,  zorder=5, marker="o", ms=3.2),
    }
    campuses = [5.0, 9.0, 10.0, 12.0]
    fig, axes = plt.subplots(1, 4, figsize=figsize(180, 60), sharey=True,
                             gridspec_kw=dict(wspace=0.12))
    ymax = 260000
    for ax, camp in zip(axes, campuses):
        style_ax(ax)
        sub = c[c.campus == camp]
        # shade u>1 (overload)
        ax.axvspan(1.0, 1.35, color=GRID, alpha=0.6, zorder=0, linewidth=0)
        for key, lab in series:
            st = STYLE[key]; col = mcol(key)
            if key == "policy":
                pm = (sub[sub.method.isin(POL_SEEDS)]
                      .pivot_table(index="u_target", columns="method", values="mean_wwt")
                      .sort_index())
                xs = pm.index.values
                lo = pm.min(axis=1).values; hi = pm.max(axis=1).values; mn = pm.mean(axis=1).values
                ax.fill_between(xs, lo, hi, color=col, alpha=0.18, zorder=2, linewidth=0)
                ax.plot(xs, mn, color=col, ls=st["ls"], lw=st["lw"], marker=st["marker"],
                        markersize=st["ms"], markeredgecolor=SURF, markeredgewidth=0.5,
                        zorder=st["zorder"], alpha=st["alpha"])
            else:
                d = sub[sub.method == key].sort_values("u_target")
                ax.plot(d.u_target, d.mean_wwt, color=col, ls=st["ls"], lw=st["lw"],
                        marker=st["marker"], markersize=st["ms"], markeredgecolor=SURF,
                        markeredgewidth=0.5, zorder=st["zorder"], alpha=st["alpha"])
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.5)
        ax.set_ylim(0, ymax)
        ax.set_xlim(0.65, 1.35)
        ax.set_xticks([0.7, 0.9, 1.1, 1.3])
        ax.set_title(f"Campus {int(camp)}", loc="left", fontsize=7.2, color=INK, weight="bold", pad=3)
        ax.grid(axis="y", which="major", color=GRID, linewidth=0.4)
        ax.tick_params(axis="x", labelsize=6.2)
    # y ticks ONCE after the loop: with sharey=True a later set_yticks([]) on
    # any panel clears the shared locator for every panel (the bug that shipped
    # an unlabelled axis); label the left panel, hide labels elsewhere.
    axes[0].set_yticks([0, 1, 100, 10000, 100000])
    axes[0].set_yticklabels(["0", "1", "100", "10k", "100k"], fontsize=6.2)
    axes[0].set_ylabel("Mean TWT  (symlog)", fontsize=6.9)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    # overload label: top of the shaded band on campus 5, where no curve passes
    axes[0].text(1.17, 1.5e5, "overload\n$u>1$", fontsize=6.1, color=MUTE, ha="center", va="top")
    # MOR callout on campus 12: clear space above-left of the red curve, not touching it
    axes[3].text(0.68, 2.2e5, "MOR collapses", fontsize=6.1, color=CMAP["mor"],
                 ha="left", va="top", weight="bold")
    # top-tier tie callout: text in empty upper-left of campus 9, leader to the tied bundle
    axes[1].annotate("top-tier tie:\nEDD $\\approx$ ATC $\\approx$ Policy",
                     xy=(1.22, 950), xycoords="data",
                     xytext=(0.04, 0.98), textcoords="axes fraction",
                     fontsize=5.6, color=INK, ha="left", va="top",
                     arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.6,
                                     connectionstyle="arc3,rad=0.18"))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.30, wspace=0.12)
    fig.text(0.53, 0.155, "target utilization  $u$", ha="center", va="center", fontsize=7.0, color=INK)
    # shared legend (styles mirror the plotted series so each stays identifiable)
    handles = [
        Line2D([0],[0], color=mcol("edd"),    lw=2.4, ls="-",         label="EDD"),
        Line2D([0],[0], color=mcol("atc"),    lw=1.5, ls=(0,(4,1.8)), label="ATC"),
        Line2D([0],[0], color=mcol("wspt"),   lw=1.3, ls="-",         label="WSPT"),
        Line2D([0],[0], color=mcol("mor"),    lw=1.3, ls="-",         label="MOR"),
        Line2D([0],[0], color=mcol("policy"), lw=1.2, ls="-", marker="o", markersize=3.6,
               markeredgecolor=SURF, label="Policy (10 seeds)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.6,
               bbox_to_anchor=(0.53, 0.02), frameon=False)
    save(fig, "f3_curves")

# ============================================================================
# F4  decision map (money figure)
#   PRIMARY (cell colour) = 4-way winner over {EDD, ATC, WSPT, Policy} on ALL
#     instances in the cell.  Policy = per-instance MEAN across the three RL
#     seeds v2rl301/302/303 (mean, never best-of-seeds).
#   SECONDARY (dagger) = the fair 5-way match that additionally includes Rolling
#     CP-SAT, computed on the CP-SAT-replan common id subset (n=8 per campus);
#     a cell is daggered where that 5-way winner differs from the 4-way winner.
# ============================================================================
POL_SEEDS = [f"v2rl{s}" for s in range(301, 311)]
FOCAL4 = ["edd", "atc", "wspt", "policy"]
F4LAB = {"edd": "EDD", "atc": "ATC", "wspt": "WSPT", "policy": "Policy", "roll": "Rolling"}

def _policy_per_id(df):
    """per-instance policy TWT = MEAN over the available RL seeds (never best-of)."""
    p = df[df.method.isin(POL_SEEDS)]
    return p.groupby("id")["wwt"].mean()

def _winner4(df, tol=0.01):
    """PRIMARY: 4-way winner over {EDD, ATC, WSPT, Policy} on ALL cell instances."""
    pol = _policy_per_id(df)
    rp = (df[df.method.isin(["edd", "atc", "wspt"])]
          .pivot_table(index="id", columns="method", values="wwt", aggfunc="mean"))
    ids = rp.dropna().index.intersection(pol.index)
    if len(ids) == 0: return None
    mw = pd.Series({"edd": rp.loc[ids, "edd"].mean(),
                    "atc": rp.loc[ids, "atc"].mean(),
                    "wspt": rp.loc[ids, "wspt"].mean(),
                    "policy": pol.loc[ids].mean()}).reindex(FOCAL4)
    best = mw.min()
    winners = [m for m in mw.index if mw[m] <= best*(1+tol)+1e-9]
    return best, winners, len(ids), mw

def _winner5(df, tol=0.01):
    """SECONDARY: 5-way (+ Rolling CP-SAT) on the rollcp2-common id subset."""
    ids = set(df.loc[df.method == "rollcp2", "id"].unique())
    if not ids: return None
    sub = df[df.id.isin(ids)]
    pol = _policy_per_id(sub)
    rp = (sub[sub.method.isin(["edd", "atc", "wspt", "rollcp2"])]
          .pivot_table(index="id", columns="method", values="wwt", aggfunc="mean"))
    aids = rp.dropna().index.intersection(pol.index)
    if len(aids) == 0: return None
    mw = pd.Series({"edd": rp.loc[aids, "edd"].mean(),
                    "atc": rp.loc[aids, "atc"].mean(),
                    "wspt": rp.loc[aids, "wspt"].mean(),
                    "roll": rp.loc[aids, "rollcp2"].mean(),
                    "policy": pol.loc[aids].mean()})
    best = mw.min()
    winners = [m for m in mw.index if mw[m] <= best*(1+tol)+1e-9]
    return best, winners, len(aids)

def _top(winners):
    return "tie" if len(winners) > 1 else winners[0]

def _cell_color(winners):
    if winners is None: return SURF, ""
    if len(winners) > 1: return TIEFILL, "tie"
    w = winners[0]
    return mcol("policy" if w == "policy" else ("roll" if w == "roll" else w)), F4LAB[w]

def _draw_cell(ax, x, y, w, h, winners, best, held=False, dagger=False):
    fc, lab = _cell_color(winners)
    ax.add_patch(FancyBboxPatch((x+0.03, y+0.03), w-0.06, h-0.06,
        boxstyle="round,pad=0,rounding_size=0.03", facecolor=fc, edgecolor=SURF, linewidth=1.2, zorder=2))
    if held:
        ax.add_patch(FancyBboxPatch((x+0.03, y+0.03), w-0.06, h-0.06,
            boxstyle="round,pad=0,rounding_size=0.03", facecolor="none",
            edgecolor=INK, linewidth=1.1, linestyle=(0,(2,1)), zorder=4))
    # text color: white on saturated policy/roll/atc/wspt, ink on tie/edd/ga
    dark = fc not in (TIEFILL, SURF, CMAP["ga"])
    tc = "#ffffff" if dark else INK
    if lab:
        ax.text(x+w/2, y+h*0.60, lab, ha="center", va="center", fontsize=6.6,
                color=tc, weight="bold", zorder=5)
        ax.text(x+w/2, y+h*0.30, fmt_wwt(best), ha="center", va="center", fontsize=6.1,
                color=tc if dark else INK2, zorder=5)
    if dagger:  # 5-way (+Rolling) match on n=8 subset picks a different winner
        ax.text(x+w-0.14, y+h-0.13, "†", ha="right", va="top", fontsize=7.4,
                color=tc if lab else INK, weight="bold", zorder=6)

def fig4_map():
    d = pd.read_csv(f"{ROOT}/results/p4_dyneval/results.csv")
    fig = plt.figure(figsize=figsize(180, 80))
    axA = fig.add_axes([0.030, 0.02, 0.26, 0.95]); axA.set_xlim(-1.15, 3.05); axA.set_ylim(-0.95, 3.55); axA.axis("off"); axA.set_aspect("equal"); axA.set_anchor("NW")
    axB = fig.add_axes([0.335, 0.02, 0.29, 0.95]); axB.set_xlim(-1.35, 3.05); axB.set_ylim(-1.05, 6.55); axB.axis("off"); axB.set_aspect("equal"); axB.set_anchor("W")
    tally = {}         # PRIMARY 4-way tally (drives colour + headline)
    tally_old5 = {}    # published 5-way(best-seed) tally, for the before/after print
    n_dagger = 0
    dagger_sets = []   # 5-way winner sets of daggered cells (caption guard)

    def _old5(df, tol=0.01):
        """published winner: {edd,atc,wspt,v2rl302,rollcp2} on rollcp2-common ids."""
        ids = set(df.loc[df.method == "rollcp2", "id"].unique())
        sub = df[df.id.isin(ids)] if ids else df
        F5 = ["edd", "atc", "wspt", "v2rl302", "rollcp2"]
        mw = sub[sub.method.isin(F5)].groupby("method")["wwt"].mean().reindex(F5).dropna()
        if mw.empty: return "tie"
        best = mw.min(); ws = [m for m in mw.index if mw[m] <= best*(1+tol)+1e-9]
        if len(ws) > 1: return "tie"
        return {"v2rl302": "policy", "rollcp2": "roll"}.get(ws[0], ws[0])

    def _place(ax, cc, yy, cell, held=False, name=""):
        nonlocal n_dagger
        res4 = _winner4(cell)
        if not res4: return
        best, winners, n, mw = res4
        res5 = _winner5(cell)
        dag = (res5 is not None) and (_top(res5[1]) != _top(winners))
        _draw_cell(ax, cc, yy, 1, 1, winners, best, held=held, dagger=dag)
        top = _top(winners); tally[top] = tally.get(top, 0) + 1
        o = _old5(cell); tally_old5[o] = tally_old5.get(o, 0) + 1
        if dag:
            n_dagger += 1
            dagger_sets.append((name, tuple(sorted(res5[1]))))

    # ---- Panel (a): pmmix, campuses pooled, pm_share x crew ----
    pm = d[d.regime == "pmmix"]
    pms = [0.2, 0.5, 0.8]; crews = [0.6, 0.8, 1.0]
    for r, pmv in enumerate(pms):
        yy = 2 - r  # top row = pm 0.2
        for cc, cr in enumerate(crews):
            cell = pm[(pm.pm_share_override == pmv) & (pm.crew_multiplier == cr)]
            _place(axA, cc, yy, cell, name=f"pm{pmv} m{cr}")
    for cc, cr in enumerate(crews):
        axA.text(cc + 0.5, -0.12, f"{cr:g}", ha="center", va="top", fontsize=6.6, color=INK2)
    for r, pmv in enumerate(pms):
        axA.text(-0.10, (2 - r) + 0.5, f"{pmv:g}", ha="right", va="center", fontsize=6.6, color=INK2)
    axA.text(1.5, -0.60, "crew multiplier", ha="center", va="top", fontsize=6.9, color=INK)
    axA.text(-0.78, 1.5, "PM share", ha="center", va="center", rotation=90, fontsize=6.9, color=INK)
    axA.text(-1.1, 3.42, "(a)  Generator contention", ha="left", va="bottom", fontsize=7.6, color=INK, weight="bold")
    axA.text(-1.1, 3.15, "campuses 5/9/10/12 pooled", ha="left", va="bottom", fontsize=6.3, color=INK2, style="italic")

    # ---- Panel (b): replay, campus x crew ----
    camps = [1, 2, 5, 9, 10, 12]
    colspec = [("replay-default", 1.0, "1.0\ndefault"), ("replay-tight", 0.8, "0.8\ntight"), ("replay-tight", 0.6, "0.6\ntight")]
    for ri, camp in enumerate(camps):
        yy = 5 - ri
        held = camp in (1, 2)
        for ci, (reg, cr, _) in enumerate(colspec):
            rr = d[d.regime == reg]
            cell = rr[(rr.campus == camp) & (rr.crew_multiplier == cr)] if reg == "replay-tight" else rr[rr.campus == camp]
            _place(axB, ci, yy, cell, held=held, name=f"c{camp} m{cr}")
        axB.text(-0.12, yy + 0.5, f"c{camp}", ha="right", va="center", fontsize=6.6,
                 color=INK, weight="bold" if held else "normal")
    # held-out bracket
    axB.plot([-0.78, -0.78], [4.05, 5.95], color=CMAP["mor"], lw=1.0)
    axB.plot([-0.78, -0.68], [5.95, 5.95], color=CMAP["mor"], lw=1.0)
    axB.plot([-0.78, -0.68], [4.05, 4.05], color=CMAP["mor"], lw=1.0)
    axB.text(-0.95, 5.0, "held out", ha="center", va="center", rotation=90, fontsize=6.0, color=CMAP["mor"], weight="bold")
    for ci, (_, _, lab) in enumerate(colspec):
        axB.text(ci + 0.5, -0.12, lab, ha="center", va="top", fontsize=6.2, color=INK2)
    axB.text(1.5, -0.92, "crew multiplier  (regime)", ha="center", va="top", fontsize=6.9, color=INK)
    axB.text(-1.3, 6.35, "(b)  Replay contention", ha="left", va="bottom", fontsize=7.6, color=INK, weight="bold")
    axB.text(-1.3, 6.08, "per campus; c1/c2 held out from training", ha="left", va="bottom", fontsize=6.3, color=INK2, style="italic")

    # ---- right-hand text column: headline, legend, how-to-read ------------
    # Every displayed count is COMPUTED (tally, len(POL_SEEDS)) so the figure
    # can never drift out of sync with the data it plots.
    import textwrap
    x0 = 0.640
    n_tie = tally.get("tie", 0); n_tot = sum(tally.values())
    fig.text(x0, 0.955, f"{n_tie} of {n_tot} cells:  top-tier tie", fontsize=9.5,
             color=INK, weight="bold", ha="left", va="top")
    fig.text(x0, 0.860,
             "rules $\\approx$ policy $\\approx$ optimiser; ATC leads\n"
             "outright on c2, and on c5/c10 at the\ntightest crew.",
             fontsize=6.9, color=INK2, ha="left", va="top")
    leg = [Patch(facecolor=TIEFILL, edgecolor=SURF, label="top tier: rules $\\approx$ policy (tie $\\leq$1%)"),
           Patch(facecolor=CMAP["atc"], edgecolor=SURF, label="ATC wins outright"),
           Line2D([0], [0], marker=r"$\dagger$", color=INK, linestyle="none", markersize=7,
                  label="5-way (+Rolling) subsample differs"),
           Patch(facecolor="none", edgecolor=INK, linestyle=(0, (2, 1)), label="held-out campus")]
    fig.legend(handles=leg, loc="upper left", ncol=1, fontsize=6.4,
               bbox_to_anchor=(x0 - 0.012, 0.700), frameon=False,
               labelspacing=0.55, handletextpad=0.6)
    note = (f"Cell colour = lowest mean TWT among {{EDD, ATC, WSPT, Policy}} over all "
            f"instances in the cell; Policy = per-instance mean of the {len(POL_SEEDS)} "
            "seeds, never a best-of. The winner is labelled with its mean TWT; methods "
            "within 1% of the best share the top-tier colour. \u2020 marks cells where the "
            "five-way match that adds Rolling CP-SAT (8 instances per campus-size "
            "cell) ends in a top-tier tie instead. Policy never wins a cell outright.")
    fig.text(x0, 0.415, "\n".join(textwrap.wrap(note, 52)), fontsize=5.9,
             color=MUTE, ha="left", va="top", linespacing=1.35)
    # LITERAL GUARD: fig_map.tex caption names the dagger outcomes (3 cells,
    # each a 5-way top-tier tie: {ATC,Rolling} on c2-default and c5-m0.6,
    # {ATC,WSPT} on c10-m0.6). Fail the build if the data drifts from that prose.
    assert sorted(dagger_sets) == [
        ("c10 m0.6", ("atc", "wspt")),
        ("c2 m1.0", ("atc", "roll")),
        ("c5 m0.6", ("atc", "roll")),
    ], f"dagger outcomes drifted from the fig_map.tex caption: {dagger_sets}"
    save(fig, "f4_map")
    print("   F4 winner tally  BEFORE (published 5-way, best-seed):", tally_old5)
    print("   F4 winner tally  AFTER  (4-way full-cell, seed-mean) :", tally, "| dagger cells:", n_dagger)

# ============================================================================
# F5  transfer (single column)
# ============================================================================
def fig5_transfer():
    d = pd.read_csv(f"{ROOT}/results/p4_dyneval/results.csv")
    rules = ["edd","wspt","atc","pfifo","mor"]; pol = [f"v2rl{s}" for s in range(301, 311)]
    camps = [1,2,5,9,10,12]
    def ratios(reg, crew):
        dd = d[d.regime==reg]
        if crew is not None: dd = dd[dd.crew_multiplier==crew]
        out = {}
        for camp in camps:
            cc = dd[dd.campus==camp]
            if not len(cc): continue
            mw = cc.groupby("method")["wwt"].mean()
            br = mw.reindex(rules).min()
            rr = [mw.get(p, np.nan)/br for p in pol]
            out[camp] = (np.nanmean(rr), np.nanmin(rr), np.nanmax(rr))
        return out
    dft = ratios("replay-default", None)
    tgt = ratios("replay-tight", 0.8)

    fig, ax = plt.subplots(figsize=figsize(88, 66)); style_ax(ax)
    x = np.arange(len(camps)); w = 0.38
    cA, cB = CMAP["policy"], "#9cc0ef"   # two shades of the policy hue
    base = 1.0
    for i,(dct,off,col,lab) in enumerate([(dft,-w/2,cA,"replay-default"),(tgt,+w/2,cB,"replay-tight m0.8")]):
        for j,camp in enumerate(camps):
            if camp not in dct: continue
            mean,lo,hi = dct[camp]
            ax.bar(x[j]+off, mean-base, width=w, bottom=base, color=col, edgecolor=SURF,
                   linewidth=0.6, zorder=3, label=lab if j==0 else None)
            # whiskers (3-seed min/max)
            ax.plot([x[j]+off, x[j]+off], [lo, hi], color=INK2, lw=0.8, zorder=5)
            ax.plot([x[j]+off-0.06, x[j]+off+0.06], [hi,hi], color=INK2, lw=0.8, zorder=5)
            ax.plot([x[j]+off-0.06, x[j]+off+0.06], [lo,lo], color=INK2, lw=0.8, zorder=5)
            if mean > 1.10:
                # value label sits ABOVE the whisker cap (+0.12 pad); the c2 pair is at
                # very different heights (2.2x vs 3.2x) so no x-separation is needed.
                ax.text(x[j]+off, hi+0.12, f"{mean:.1f}x", ha="center", va="bottom",
                        fontsize=6.2, color=INK, weight="bold")
    ax.axhline(base, color=INK2, lw=0.9, zorder=4)
    ax.text(5.52, base+0.02, "parity", ha="right", va="bottom", fontsize=6.0, color=INK2, style="italic")
    ax.set_ylim(0.9, 4.9)
    ax.set_yticks([1,2,3,4]); ax.set_yticklabels(["1x","2x","3x","4x"], fontsize=6.4)
    ax.set_ylabel("Policy TWT / best-rule TWT", fontsize=7.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{c}" for c in camps], fontsize=6.8, color=INK)
    # held-out campuses read from the shaded span + italic header (redundant red
    # 'held-out' tick labels removed to declutter the axis).
    ax.axvspan(-0.5, 1.5, color=GRID, alpha=0.45, zorder=0, linewidth=0)
    ax.text(0.5, 4.8, "held out from training", ha="center", va="top", fontsize=6.1, color=INK2, style="italic")
    ax.text(3.75, 4.8, "training campuses  (parity holds)", ha="center", va="top", fontsize=6.1, color=INK2, style="italic")
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.5)
    ax.set_xlim(-0.6, 5.6)
    ax.legend(loc="center right", fontsize=6.4, bbox_to_anchor=(1.0, 0.56))
    ax.set_title("Transfer to held-out campuses", loc="left", fontsize=7.6, color=INK, weight="bold", pad=5)
    save(fig, "f5_transfer")

# ============================================================================
# F6  sensitivity (Kendall tau matrix, single column) -- frozen summary values
# ============================================================================
def fig6_sensitivity():
    cols = ["c5/150","c5/400","c9/150","c9/400","c10/150","c10/400","c12/150","c12/400","pooled"]
    rows = ["sla0.5","sla1.5","crew0.75","crew1.25"]
    rowlab = ["SLA x0.5","SLA x1.5","crew x0.75","crew x1.25"]
    NA = np.nan
    T = np.array([
        [NA, NA, 0.81, 0.81, 0.72, 0.40, 0.72, 0.20, 0.83],
        [NA, NA, 0.85, 1.00, 1.00, 0.85, 0.82, 0.09, 0.83],
        [NA, NA, 0.87, 0.77, 0.87, 0.31, 0.85, 0.49, 0.77],
        [NA, NA, 0.85, 0.98, 0.97, 0.90, 1.00, 0.20, 0.83],
    ])
    # LITERAL GUARD: the frozen pooled column must match the released
    # tab_sensitivity.tex; refuse to draw a stale figure.
    import re as _re
    _tab = open(f"{ROOT}/results/p4_sensitivity/tab_sensitivity.tex").read()
    _pooled = [float(m) for m in _re.findall(r"&\s*([01]\.\d+)\s*&\s*[01]\.\d+\s*\\\\", _tab)]
    _frozen = [row[-1] for row in T]
    assert all(abs(a - b) < 1e-9 for a, b in zip(_frozen, _pooled)), \
        f"f6 frozen pooled taus {_frozen} != released {_pooled}"
    nr, nc = len(rows), len(cols)
    # fixed single-column layout (saved WITHOUT tight-crop so it renders 1:1 at
    # column width and the label fonts do not shrink below the print floor)
    fig = plt.figure(figsize=figsize(88, 59))
    ax = fig.add_axes([0.185, 0.160, 0.790, 0.615]); ax.set_xlim(0, nc); ax.set_ylim(0, nr); ax.axis("off")
    from matplotlib.colors import LinearSegmentedColormap
    blues = LinearSegmentedColormap.from_list("blu", ["#eef4fb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95"])
    def _lum(rgba):
        f = lambda c: c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4
        R, G, B = (f(rgba[0]), f(rgba[1]), f(rgba[2])); return 0.2126*R + 0.7152*G + 0.0722*B
    for i in range(nr):
        yy = nr - 1 - i
        for j in range(nc):
            v = T[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle((j+0.04, yy+0.04), 0.92, 0.92, facecolor="#f0efec", edgecolor=SURF, linewidth=1.2))
                ax.text(j+0.5, yy+0.5, "n/a", ha="center", va="center", fontsize=6.6, color=MUTE, style="italic")
                continue
            fill = blues(float(min(1.0, max(0.0, v))))  # float() so v==1.0 is a value, not a LUT index
            ax.add_patch(Rectangle((j+0.04, yy+0.04), 0.92, 0.92, facecolor=fill, edgecolor=SURF, linewidth=1.2))
            if v < 0.8:   # below-threshold: 45-deg hatch as the secondary (print-safe) channel
                ax.add_patch(Rectangle((j+0.04, yy+0.04), 0.92, 0.92, facecolor="none",
                             edgecolor="#2b2b2b", linewidth=0.3, hatch="/////"))
            tc = "#ffffff" if _lum(fill) < 0.45 else INK
            ax.text(j+0.5, yy+0.5, f"{v:.2f}", ha="center", va="center", fontsize=6.9,
                    color=tc, weight="bold" if j == nc-1 else "normal")
    # column labels: two-line horizontal (campus over size), no rotation overhang
    for j, cl in enumerate(cols):
        lab = cl.replace("/", "\n")
        ax.text(j+0.5, nr + 0.06, lab, ha="center", va="bottom", fontsize=6.4,
                color=INK if cl == "pooled" else INK2, linespacing=0.92,
                weight="bold" if cl == "pooled" else "normal", clip_on=False)
    for i, rl in enumerate(rowlab):
        ax.text(-0.12, (nr-1-i)+0.5, rl, ha="right", va="center", fontsize=6.6, color=INK2, clip_on=False)
    ax.plot([8, 8], [0, nr], color=AXIS, lw=0.7)   # separator before pooled
    fig.text(0.020, 0.965, "Ranking robustness (Kendall $\\tau_b$)", ha="left", va="top",
             fontsize=8.0, color=INK, weight="bold")
    fig.text(0.020, 0.025, "hatched: $\\tau < 0.8$ (ranking perturbed).\nn/a: degenerate cell (baseline fully tied).",
             ha="left", va="bottom", fontsize=6.6, color=INK2)
    save(fig, "f6_sensitivity", tight=False)

# ============================================================================
MAIN = {"f1":fig1_pipeline, "f2":fig2_static, "f3":fig3_curves,
        "f4":fig4_map, "f5":fig5_transfer, "f6":fig6_sensitivity}
if __name__ == "__main__":
    set_style()
    which = sys.argv[1:] or list(MAIN)
    for k in which:
        print(f"[{k}]"); MAIN[k]()
    print("done.")
