#!/usr/bin/env python
"""P5 EXTRA - four added figures for the manuscript.

f7_data      FMUCD at a glance          (double col, 4 panels)
f8_priority  raw priority is not naive  (single col, 2 panels)
f9_training  what training discriminates (single col, 2 panels)
f10_rolling  replan on a clock          (double col, 2 panels)

Style is inherited verbatim from scripts/p5_figures.py (Times-serif, hairline
axes, direct labels).  Every plotted value comes from a results/ file:
  f7 : results/p0_profile/{arrivals,per_campus,trades}.csv + labor_hist.csv
  f8 : results/p1_calib/priority_mapping.csv
  f9 : results/p3_train/v2/seed{301,302,303}/curves.csv
  f10: results/p4_dyneval/rolling_diag.json

  python scripts/p5_figures_extra.py [f7 f8 f9 f10]
"""
import json
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
import matplotlib.ticker as mticker

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from p5_figures import (  # noqa: E402
    set_style, save, style_ax, figsize, CMAP, PRETTY,
    INK, INK2, MUTE, GRID, AXIS, SURF, fmt_wwt, mcol,
)

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
P0 = f"{ROOT}/results/p0_profile"

# Okabe-Ito qualitative palette (colour-blind safe) for the 6 campuses -- kept
# deliberately DISJOINT from the frozen method palette in p5_figures.
CAMP_COL = {1: "#E69F00", 2: "#56B4E9", 5: "#009E73",
            9: "#0072B2", 10: "#D55E00", 12: "#CC79A7"}
SCHEDULABLE = [1, 2, 5, 9, 10, 12]
POLICY_BLUE = CMAP["policy"]     # #2a78d6
ROLL_TEAL = CMAP["roll"]         # #1baf7a


def _kfmt(n):
    """126000 -> '126k', 41000 -> '41k', <1000 -> '900'."""
    n = float(n)
    if n >= 1000:
        return f"{n/1000:.0f}k"
    return f"{n:.0f}"


# ============================================================================
# F7  FMUCD at a glance (4 panels)
# ============================================================================
def fig7_data():
    ar = pd.read_csv(f"{P0}/arrivals.csv")
    pc = pd.read_csv(f"{P0}/per_campus.csv")
    tr = pd.read_csv(f"{P0}/trades.csv")
    lh = pd.read_csv(f"{P0}/labor_hist.csv")

    fig = plt.figure(figsize=figsize(180, 62))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.32, 1.12, 1.02, 1.06],
                          left=0.045, right=0.995, top=0.86, bottom=0.155,
                          wspace=0.42)
    axA, axB, axC, axD = (fig.add_subplot(gs[0, i]) for i in range(4))

    # ---- (a) monthly arrivals per campus (log-y, thin lines, direct labels) --
    style_ax(axA)
    ar6 = ar[ar.UniversityID.isin(SCHEDULABLE)]
    g = ar6.groupby(["UniversityID", "month"])["rows"].sum().reset_index()
    g["yr"] = g["month"].str.slice(0, 4).astype(int) + \
        (g["month"].str.slice(5, 7).astype(int) - 1) / 12.0
    ends = []
    for c in SCHEDULABLE:
        s = g[g.UniversityID == c].sort_values("yr")
        axA.plot(s.yr, s.rows, "-", color=CAMP_COL[c], lw=0.9, alpha=0.95,
                 zorder=3, solid_capstyle="round")
        # label at the recent-plateau level (median of last 6 months), not the
        # partial final-month cliff, so the tag sits where the line reads.
        ends.append((c, float(s.yr.iloc[-1]), float(s.rows.tail(6).median())))
    axA.set_yscale("log")
    axA.set_ylim(0.7, 40000)
    axA.set_xlim(2002, 2024.3)
    axA.set_xticks([2004, 2008, 2012, 2016, 2020])
    axA.set_xticklabels(["'04", "'08", "'12", "'16", "'20"], fontsize=6.2)
    axA.set_yticks([1, 10, 100, 1000, 10000])
    axA.set_yticklabels(["1", "10", "100", "1k", "10k"], fontsize=6.2)
    axA.grid(axis="y", which="major", color=GRID, linewidth=0.4)
    # direct labels near each line end; the four campuses that converge to ~1.8k
    # WOs/month get nudged apart in log space and a white halo so each c-tag
    # stays legible over the tangle (a small vertical float is unavoidable when
    # lines converge -- colour + x-position tie each tag to its line).
    # four campuses converge to ~1.8k WOs/mo; direct tags are pushed apart in
    # log space to a spacing that clears the digit cap-height (no ink overlap),
    # smaller font + a stronger white halo keep each tag legible over the tangle.
    ends.sort(key=lambda t: t[2])
    last_ly = -1e9
    for c, xe, ye in ends:
        ly = np.log10(max(ye, 1.0))
        if ly - last_ly < 0.22:
            ly = last_ly + 0.22
        last_ly = ly
        axA.text(xe + 0.3, 10 ** ly, f"c{c}", color=CAMP_COL[c],
                 fontsize=5.6, ha="left", va="center", weight="bold",
                 path_effects=[pe.withStroke(linewidth=2.4, foreground=SURF)])
    axA.set_ylabel("work orders / month  (log)", fontsize=6.9)
    axA.set_title("(a)  Monthly arrivals", loc="left", fontsize=7.4,
                  color=INK, weight="bold", pad=5)

    # ---- (b) labour hours per order (log-x histogram) -----------------------
    style_ax(axB)
    h = lh[lh.kind == "hist"].copy()
    edges = np.append(h.bin_lo.astype(float).to_numpy(),
                      float(h.bin_hi.astype(float).iloc[-1]))
    counts = h["count"].astype(float).to_numpy()
    axB.stairs(counts, edges, fill=True, color=POLICY_BLUE, alpha=0.55,
               edgecolor=POLICY_BLUE, linewidth=0.6, zorder=3)
    axB.set_xscale("log")
    axB.set_xlim(0.03, 200)
    q = {r.label: float(r.value) for _, r in lh[lh.kind == "quant"].iterrows()}
    med, p90, p99 = q["six_p50"], q["six_p90"], q["six_p99"]
    ymax = counts.max() * 1.16
    axB.set_ylim(0, ymax)
    # 'median' is the widest tag and its line sits closest to p90; place it to
    # the LEFT of its dashed line (clear top-left space) so it never reaches the
    # p90 line. p90 / p99 stay to the right of their lines.
    for xv, lab, side in [(med, "median\n1.0 h", "l"), (p90, "p90\n6 h", "r"),
                          (p99, "p99\n49 h", "r")]:
        axB.axvline(xv, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=4)
        if side == "l":
            axB.text(xv * 0.92, ymax * 0.93, lab, fontsize=5.7, color=INK2,
                     ha="right", va="top", linespacing=0.9)
        else:
            axB.text(xv * 1.08, ymax * 0.93, lab, fontsize=5.7, color=INK2,
                     ha="left", va="top", linespacing=0.9)
    axB.set_xticks([0.1, 1, 10, 100])
    axB.set_xticklabels(["0.1", "1", "10", "100"], fontsize=6.2)
    axB.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, p: f"{v/1000:.0f}k" if v >= 1000 else "0"))
    axB.tick_params(axis="y", labelsize=6.2)
    axB.set_xlabel("labour hours  (log)", fontsize=6.9)
    axB.set_ylabel("work orders", fontsize=6.9)
    axB.set_title("(b)  Labour hours per order", loc="left", fontsize=7.4,
                  color=INK, weight="bold", pad=5)

    # ---- (c) preventive share per campus (stacked horizontal) ---------------
    style_ax(axC)
    pc6 = pc[pc.UniversityID.isin(SCHEDULABLE)].sort_values("pm_share")
    cm_col, pm_col = CMAP["wspt"], "#8aa0b6"   # corrective orange, preventive gray-blue
    ypos = np.arange(len(pc6))
    for y, (_, r) in zip(ypos, pc6.iterrows()):
        pm = float(r.pm_share)
        axC.barh(y, 1 - pm, height=0.66, left=0.0, color=cm_col, alpha=0.9,
                 edgecolor=SURF, linewidth=0.5, zorder=3)
        axC.barh(y, pm, height=0.66, left=1 - pm, color=pm_col, alpha=0.95,
                 edgecolor=SURF, linewidth=0.5, zorder=3)
        axC.text(1.02, y, _kfmt(r.rows), va="center", ha="left",
                 fontsize=6.0, color=INK2)
    # direct segment labels once, on the top bar (highest pm share)
    ytop = ypos[-1]
    rtop = pc6.iloc[-1]
    axC.text((1 - rtop.pm_share) / 2, ytop + 0.52, "corrective", ha="center",
             va="bottom", fontsize=5.8, color=cm_col, weight="bold")
    axC.text(1 - rtop.pm_share / 2, ytop + 0.52, "preventive", ha="center",
             va="bottom", fontsize=5.8, color="#5f7a94", weight="bold")
    axC.set_yticks(ypos)
    axC.set_yticklabels([f"c{int(c)}" for c in pc6.UniversityID], fontsize=6.6,
                        color=INK)
    axC.set_xlim(0, 1.16)
    axC.set_xticks([0, 0.5, 1.0])
    axC.set_xticklabels(["0", ".5", "1"], fontsize=6.2)
    axC.set_ylim(-0.6, len(pc6) - 0.1)
    axC.set_xlabel("share of work orders", fontsize=6.9)
    axC.set_title("(c)  Preventive share", loc="left", fontsize=7.4,
                  color=INK, weight="bold", pad=5)

    # ---- (d) trade mix (top 8 + other) --------------------------------------
    # code as the (short) y-tick; description + share at the bar end, so the
    # long trade names never reach back into panel (c).
    style_ax(axD)
    tot = float(tr.rows.sum())
    top = tr.sort_values("rows", ascending=False).head(8)
    other = tot - float(top.rows.sum())
    DESC = {"D30": "HVAC", "D50": "Electrical", "D20": "Plumbing",
            "D40": "Fire prot.", "C10": "Interior", "E10": "Equipment",
            "B20": "Exterior", "C30": "Finishes"}
    rows_ = [(r.trade, DESC.get(r.trade, str(r.description)), float(r.rows) / tot)
             for _, r in top.iterrows()]
    rows_.append(("other", "", other / tot))
    ypos = np.arange(len(rows_))[::-1]
    for y, (code, desc, sh) in zip(ypos, rows_):
        c = MUTE if code == "other" else "#6b7a86"
        axD.barh(y, sh * 100, height=0.68, color=c, alpha=0.92,
                 edgecolor=SURF, linewidth=0.5, zorder=3)
        lab = f"{desc}  {sh*100:.0f}%" if desc else f"{sh*100:.0f}%"
        axD.text(sh * 100 + 0.8, y, lab, va="center", ha="left",
                 fontsize=5.7, color=INK2)
    axD.set_yticks(ypos)
    axD.set_yticklabels([r[0] for r in rows_], fontsize=6.0, color=INK)
    axD.set_xlim(0, 62)
    axD.set_xticks([0, 10, 20, 30])
    axD.set_xticklabels(["0", "10", "20", "30"], fontsize=6.2)
    axD.set_ylim(-0.6, len(rows_) - 0.4)
    axD.set_xlabel("share of all work orders (%)", fontsize=6.9)
    axD.grid(axis="x", which="major", color=GRID, linewidth=0.4)
    axD.set_title("(d)  Trade mix", loc="left", fontsize=7.4,
                  color=INK, weight="bold", pad=5)
    save(fig, "f7_data")


# ============================================================================
# F8  the raw priority field cannot be read naively (2 panels)
# ============================================================================
def fig8_priority():
    pm = pd.read_csv(f"{ROOT}/results/p1_calib/priority_mapping.csv")
    cm = pm[pm.is_pm_split == "cm"].copy()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize(88, 62),
                                   gridspec_kw=dict(wspace=0.18))

    # ---- (a) campus 2, labelled scale, bars ordered by realized duration ----
    style_ax(axA)
    c2 = cm[(cm.campus == 2)].dropna(subset=["median_cm_duration_days"])
    c2 = c2.sort_values("median_cm_duration_days")
    labels = [r.raw_value.split("-", 1)[-1].title() for _, r in c2.iterrows()]
    dur = c2.median_cm_duration_days.to_numpy()
    klass = c2.mapped_class.to_numpy()
    ypos = np.arange(len(c2))
    kcol = {1: CMAP["mor"], 2: CMAP["wspt"], 3: "#5f8fce", 4: MUTE}
    # Value labels go INSIDE the bar (white, right-aligned) whenever the bar is
    # long enough to hold them; only short bars get an outside label. This keeps
    # the whole right margin of panel (a) clear of panel (b)'s vertical y-label.
    for y, d, lab, k in zip(ypos, dur, labels, klass):
        axA.barh(y, d, height=0.62, color=kcol.get(int(k), MUTE), alpha=0.9,
                 edgecolor=SURF, linewidth=0.5, zorder=3)
        txt = f"{d:.1f} d  →P{int(k)}"
        if d >= 3.5:  # inside the bar, white
            axA.text(d - 0.18, y, txt, va="center", ha="right", fontsize=5.9,
                     color="#ffffff", weight="bold", zorder=4)
        else:         # short bar: outside, ink (stays well inside panel a)
            axA.text(d + 0.14, y, txt, va="center", ha="left", fontsize=5.9,
                     color=INK2, zorder=4)
    axA.set_yticks(ypos)
    axA.set_yticklabels(labels, fontsize=6.4, color=INK)
    axA.set_xlim(0, 7.0)
    axA.set_xticks([0, 2, 4, 6])
    axA.set_xlabel("median close time (days)", fontsize=6.8)
    axA.set_ylim(-0.6, len(c2) - 0.4)
    axA.grid(axis="x", which="major", color=GRID, linewidth=0.4)
    axA.set_title("(a)  Campus 2: text scale", loc="left", fontsize=7.3,
                  color=INK, weight="bold", pad=4)
    axA.text(0.97, 0.06, "labels sort\nby urgency", transform=axA.transAxes,
             fontsize=5.8, color=INK2, style="italic", va="bottom", ha="right",
             linespacing=0.95)

    # ---- (b) campus 12 numeric codes invert urgency (+ campus 1 contrast) ---
    style_ax(axB)
    for camp, col, faint, tag in [(1, MUTE, True, "c1"),
                                  (12, ROLL_TEAL, False, "c12")]:
        s = cm[(cm.campus == camp) & (cm.rule == "r5c")].copy()
        s["code"] = s.raw_value.astype(float)
        s = s.sort_values("code")
        rho = float(s.spearman_rho.iloc[0])
        lw = 1.0 if faint else 1.6
        alpha = 0.5 if faint else 1.0
        axB.plot(s.code, s.median_cm_duration_days, "-o", color=col, lw=lw,
                 markersize=3.0 if faint else 4.0, markeredgecolor=SURF,
                 markeredgewidth=0.5, alpha=alpha, zorder=3 if faint else 5)
        # tag the series at its last point
        xe = float(s.code.iloc[-1]); ye = float(s.median_cm_duration_days.iloc[-1])
        axB.annotate(f"{tag}  $\\rho$={rho:+.2f}",
                     (xe, ye), xytext=(6 if camp == 12 else 2,
                                       -8 if camp == 12 else 9),
                     textcoords="offset points",
                     fontsize=6.0, color=col, weight="bold",
                     ha="left", va="center")
    axB.annotate("higher code =\nmore urgent", xy=(50, 1.5), xytext=(40, 33),
                 fontsize=5.8, color=ROLL_TEAL, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-|>", color=ROLL_TEAL, lw=0.8,
                                 connectionstyle="arc3,rad=-0.25"))
    axB.set_xlim(-4, 56)
    axB.set_ylim(0, 72)
    axB.set_xlabel("raw priority code (campus-specific)", fontsize=6.8)
    axB.set_ylabel("median close time (days)", fontsize=6.8)
    axB.grid(axis="y", which="major", color=GRID, linewidth=0.4)
    axB.set_title("(b)  Numeric codes invert", loc="left", fontsize=7.3,
                  color=INK, weight="bold", pad=4)
    save(fig, "f8_priority")


# ============================================================================
# F9  what training can and cannot discriminate (2 stacked panels, shared x)
# ============================================================================
def fig9_training():
    seeds = list(range(301, 311))
    curves = {s: pd.read_csv(f"{ROOT}/results/p3_train/v2/seed{s}/curves.csv")
              for s in seeds}
    import matplotlib as _mpl
    _blu = _mpl.colormaps["Blues"]
    blues = [_blu(0.35 + 0.06 * i) for i in range(len(seeds))]

    fig, (axA, axB) = plt.subplots(2, 1, figsize=figsize(88, 60), sharex=True,
                                   gridspec_kw=dict(hspace=0.28, height_ratios=[1, 1]))

    # ---- (a) default-capacity dev: flat plateau ----------------------------
    style_ax(axA)
    # LITERAL GUARD: the plateau band (macros \ablplateaulo/hi = 409-411) must
    # contain every v2 seed's best default-capacity dev value.
    _floors = [curves[s]["dev_wwt_mean"].min() for s in seeds]
    assert all(408.5 <= f <= 411.5 for f in _floors), f"plateau drifted: {_floors}"
    axA.axhspan(409, 411, color=POLICY_BLUE, alpha=0.13, zorder=0, linewidth=0)
    for s, col in zip(seeds, blues):
        d = curves[s]
        axA.plot(d["update"], d["dev_wwt_mean"], "-", color=col, lw=0.9,
                 alpha=0.85, zorder=3, label="_nolegend_")
    axA.set_ylim(408, 426)
    axA.set_yticks([410, 415, 420, 425])
    axA.tick_params(axis="y", labelsize=6.2)
    axA.text(595, 411.6, "plateau: all variants 409–411", fontsize=5.9,
             color=INK2, ha="right", va="bottom", style="italic")
    axA.set_ylabel("dev TWT", fontsize=6.9)
    axA.set_title("(a)  default-capacity dev set $\\cdot$ 10 seeds", loc="left", fontsize=7.3,
                  color=INK, weight="bold", pad=4)

    # ---- (b) tight-capacity dev (m=0.6): declining + selected checkpoints ---
    style_ax(axB)
    for s, col in zip(seeds, blues):
        d = curves[s]
        raw = d["dev_wwt_tight"]
        roll = raw.rolling(window=15, center=True, min_periods=1).mean()
        axB.plot(d["update"], raw, "-", color=col, lw=0.5, alpha=0.25, zorder=2)
        axB.plot(d["update"], roll, "-", color=col, lw=1.2, alpha=0.98,
                 zorder=4)
        imin = raw.idxmin()
        axB.plot(d["update"][imin], raw[imin], marker="v", color=col,
                 markersize=4.6, markeredgecolor=SURF, markeredgewidth=0.6,
                 zorder=6)
    axB.set_ylim(423, 500)
    axB.set_yticks([440, 460, 480, 500])
    axB.tick_params(axis="y", labelsize=6.2)
    axB.text(300, 494, "selected checkpoint = per-seed minimum ($\\blacktriangledown$)",
             fontsize=5.8, color=INK2, ha="center", va="center", style="italic")
    axB.set_ylabel("dev TWT", fontsize=6.9)
    axB.set_xlim(0, 600)
    axB.set_xticks([0, 150, 300, 450, 600])
    axB.tick_params(axis="x", labelsize=6.2)
    axB.set_xlabel("PPO update", fontsize=6.9)
    axB.set_title("(b)  tight-capacity dev set (m=0.6) $\\cdot$ 10 seeds", loc="left",
                  fontsize=7.3, color=INK, weight="bold", pad=4)
    save(fig, "f9_training")


# ============================================================================
# F10  replan on a clock, not only on arrivals (2 panels)
# ============================================================================
def fig10_rolling():
    recs = json.load(open(f"{ROOT}/results/p4_dyneval/rolling_diag.json"))
    by = {(r["short"], r["variant"]): r for r in recs}

    fig = plt.figure(figsize=figsize(180, 58))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], left=0.055,
                          right=0.975, top=0.85, bottom=0.17, wspace=0.24)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ---- (a) replan-event timeline for id-0102 -----------------------------
    style_ax(axA)
    ao = by[("0102", "arrival-only")]
    pe = by[("0102", "periodic")]
    horizon = max(ao["makespan"], pe["makespan"]) * 1.02
    lanes = [(1, ao, "arrival-only trigger", CMAP["mor"]),
             (0, pe, "periodic + arrival", INK)]
    for y, rec, name, col in lanes:
        t = sorted(rec["replan_times_bh"])
        # longest no-replan span (last tick -> makespan)
        ext = t + [rec["makespan"]]
        gaps = np.diff(ext)
        gi = int(np.argmax(gaps))
        if y == 1:  # shade the stale span in the arrival-only lane only
            axA.add_patch(Rectangle((ext[gi], y - 0.28), gaps[gi], 0.56,
                          facecolor=CMAP["mor"], alpha=0.14, edgecolor="none",
                          zorder=1))
            axA.text((ext[gi] + ext[gi + 1]) / 2, y + 0.32,
                     "stale plan executes uncorrected",
                     ha="center", va="bottom", fontsize=5.9, color=CMAP["mor"],
                     style="italic")
        axA.hlines(y, 0, rec["makespan"], color=AXIS, lw=0.6, zorder=2)
        for tv in t:
            axA.vlines(tv, y - 0.17, y + 0.17, color=col, lw=0.7, zorder=4)
        axA.plot(rec["makespan"], y, marker="|", color=col, markersize=7,
                 markeredgewidth=1.1, zorder=5)
        wwt_txt = format(int(round(rec["wwt"])), ",")
        # place the label so its white box sits fully clear of the shaded
        # stale-span rectangle's right edge (the pink ends at the makespan).
        axA.text(horizon * 0.95, y, f"TWT {wwt_txt}",
                 ha="right", va="center", fontsize=7.0, weight="bold",
                 color=col, zorder=6,
                 bbox=dict(boxstyle="round,pad=0.16", fc=SURF, ec="none"))
        axA.text(-horizon * 0.012, y, name, ha="right", va="center",
                 fontsize=6.3, color=col)
    axA.set_xlim(-horizon * 0.16, horizon)
    axA.set_ylim(-0.7, 1.7)
    axA.set_yticks([])
    for sp in ("left",):
        axA.spines[sp].set_visible(False)
    axA.set_xlabel("business hours", fontsize=6.9)
    axA.set_xticks([0, 100, 200, 300])
    axA.tick_params(axis="x", labelsize=6.2)
    axA.text(0, 1.62, "campus 9 · size 400 · m=0.6 · id 0102",
             fontsize=6.0, color=INK2, ha="left", va="center")
    axA.set_title("(a)  Replan timeline", loc="left", fontsize=7.4, color=INK,
                  weight="bold", pad=6, x=-0.16)

    # ---- (b) outcome slope chart (arrival-only -> periodic), log-y ----------
    style_ax(axB)
    order = ["0102", "0105", "0107"]
    xa, xp = 0.0, 1.0
    FLOOR = 6.0
    # Hand-tuned label placement: the two crossing slope lines, the near-coincident
    # periodic dots (402 vs 268) and the two near-equal EDD references force
    # mutually-clear, per-instance positions (all verified collision-free).
    IDLAB = {"0102": (0.16, "left"),    # nudged right of the plunging id-0105 line
             "0105": (0.0, "center"), "0107": (0.0, "center")}
    PVAL = {"0102": (0.72, 468.0),      # up-left into the line wedge
            "0107": (0.80, 175.0),      # down-left, clear of the id-0107 line
            "0105": (xp - 0.17, None)}
    EDDVA = {"0102": "bottom", "0107": "top"}  # split the 402/268 EDD notes apart
    for i, sid in enumerate(order):
        a = by[(sid, "arrival-only")]
        p = by[(sid, "periodic")]
        ya, yp, ye = a["wwt"], p["wwt"], a["edd_wwt"]
        axB.plot([xa, xp], [ya, yp], "-", color=MUTE, lw=1.0, zorder=2)
        axB.plot(xa, ya, "o", color=CMAP["mor"], markersize=5.0,
                 markeredgecolor=SURF, markeredgewidth=0.7, zorder=5)
        axB.plot(xp, yp, "o", color=ROLL_TEAL, markersize=5.0,
                 markeredgecolor=SURF, markeredgewidth=0.7, zorder=5)
        # EDD reference tick (dashed) at the periodic column; periodic rolling
        # lands ON the EDD reference for the non-pathological cases, so only the
        # tick + a short "EDD" note is drawn (the number is not repeated). The
        # note sits above (0102) / below (0107) its tick so the two nearly-equal
        # references never overprint.
        if ye > 1e-9:
            axB.plot([xp - 0.14, xp + 0.14], [ye, ye], ls=(0, (2, 1.5)),
                     color=INK2, lw=0.9, zorder=4)
            axB.text(xp + 0.19, ye, "EDD", fontsize=5.5, color=INK2,
                     va=EDDVA.get(sid, "center"), ha="left")
        else:  # EDD == 0: clip to the log floor, annotate honestly
            axB.plot([xp - 0.14, xp + 0.14], [FLOOR, FLOOR], ls=(0, (2, 1.5)),
                     color=INK2, lw=0.9, zorder=4)
            axB.text(xp + 0.19, FLOOR, "EDD 0\n(log floor)", fontsize=5.2,
                     color=INK2, va="center", ha="left", linespacing=0.9)
        # arrival-only value (left of the red dot)
        axB.text(xa - 0.07, ya, fmt_wwt(ya), fontsize=5.9, color=CMAP["mor"],
                 ha="right", va="center", weight="bold")
        # periodic value (teal): hand-placed left of the EDD tick, clear of the
        # crossing lines and of the neighbouring periodic label
        pvx, pvy = PVAL[sid]
        axB.text(pvx, yp if pvy is None else pvy, fmt_wwt(yp), fontsize=5.9,
                 color=ROLL_TEAL, ha="right", va="center", weight="bold")
        # instance tag near the arrival-only dot
        idx, idha = IDLAB[sid]
        axB.text(idx, ya * 1.52, f"id {sid}", fontsize=5.6, color=INK2,
                 ha=idha, va="bottom")
    axB.set_yscale("log")
    axB.set_ylim(FLOOR * 0.8, 20000)
    axB.set_xlim(-0.5, 1.6)
    axB.set_xticks([xa, xp])
    axB.set_xticklabels(["arrival\nonly", "periodic"], fontsize=6.2)
    axB.set_ylabel("episode TWT  (log)", fontsize=6.9)
    axB.grid(axis="y", which="major", color=GRID, linewidth=0.4)
    axB.set_title("(b)  Outcome", loc="left", fontsize=7.4, color=INK,
                  weight="bold", pad=6)
    save(fig, "f10_rolling")


MAIN = {"f7": fig7_data, "f8": fig8_priority, "f9": fig9_training,
        "f10": fig10_rolling}
if __name__ == "__main__":
    set_style()
    which = sys.argv[1:] or list(MAIN)
    for k in which:
        print(f"[{k}]")
        MAIN[k]()
    print("done.")
