#!/usr/bin/env python
"""W1 analysis: headline table, routing curve, coverage, and the macro block.

Reads the per-cell records written by scripts/y3_w1_sweep.py (from the result
cache, never from the CSV, so a half-written CSV cannot enter a statistic) and
produces:

  head    the headline table at the manuscript's headline cell: per-arm TWT*
          ladder, the three gate contrasts, paired Wilcoxon and Holm.
  grid    the same three contrasts over the eight-cell contention grid, Holm
          within each contrast type ACROSS the eight cells, which is the family
          structure Section "Objective, compute, seeds, and statistics" declares.
  curve   (rho, automation coverage, TWT* reduction) at both cells.
  alpha   the conformal level sweep.
  cov     band coverage against the true shift per beta (evaluation only).
  macros  a \\newcommand block for every number the manuscript should quote,
          each with a trailing comment naming the results file and field.

The statistics are IMPORTED from scripts/y3_p4_m0grid.py (paired_wilcoxon with
zero_method='pratt', win_tie_loss, holm), not reimplemented, so a contrast here
is computed by the same code that produced the committed numbers.

Run:  PYTHONPATH=src python scripts/y3_w1_analyze.py --part all
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import y3_p4_m0grid as P4                                           # noqa: E402
import y3_w1_sweep as S                                             # noqa: E402

_OUT = S._OUT
DEC = S.DECIDERS

CONTRASTS = [("M0_vs_RULE", "m0_alone", "rule"),
             ("M0sup_vs_RULEsup", "m0_sup", "rule_sup"),
             ("M0sup_vs_ORACLE", "m0_sup", "oracle")]


def load(tasks, require=False):
    """(task, record) pairs for the tasks whose cache record exists."""
    out, missing = [], 0
    for t in tasks:
        p = os.path.join(S._CACHE, "%s.json" % S._cell_sig(t))
        if not os.path.exists(p):
            missing += 1
            continue
        with open(p) as fh:
            out.append((t, json.load(fh)))
    if missing:
        msg = "%d of %d cache records missing" % (missing, len(tasks))
        if require:
            raise RuntimeError(msg)
        print("  [warn] %s" % msg)
    return out


def _check_split(recs):
    """Every record must score the same held-out instance ids."""
    ref = recs[0][1]["inst_ids"]
    for _t, r in recs:
        if r["inst_ids"] != ref:
            raise RuntimeError("held-out instance ids differ across records")
    return ref


def stack(recs, decider):
    recs = sorted(recs, key=lambda tr: tr[1]["seed"])
    ids = _check_split(recs)
    mat = []
    for _t, r in recs:
        idx = {iid: i for i, iid in enumerate(r["inst_ids"])}
        mat.append([r["per"][decider][idx[i]] for i in ids])
    return np.asarray(mat, float)


def ladder(recs):
    out = {}
    for d in DEC:
        m = stack(recs, d)
        seed_means = m.mean(axis=1)
        out[d] = {"twt_mean": float(seed_means.mean()),
                  "twt_std": float(seed_means.std(ddof=0)),
                  "n_seeds": int(m.shape[0])}
    rule = out["rule"]["twt_mean"]
    for d in DEC:
        out[d]["pct_below_rule"] = 100.0 * (rule - out[d]["twt_mean"]) / rule
    return out


def contrast(recs, test, comp):
    a = stack(recs, test).mean(axis=0)
    b = stack(recs, comp).mean(axis=0)
    am, bm = float(a.mean()), float(b.mean())
    return {"test": test, "comparator": comp, "test_mean": am,
            "comparator_mean": bm,
            "pct_vs_comparator": 100.0 * (bm - am) / bm if abs(bm) > 1e-12 else 0.0,
            "wtl": P4.win_tie_loss(a, b), "wilcoxon_p": P4.paired_wilcoxon(a, b),
            "n_instances": int(a.size)}


def routing_means(recs):
    keys = ["m0_sup_revfrac_mean", "m0_sup_revfrac_all_mean",
            "m0_sup_undetermined", "m0_sup_cov_all", "m0_sup_cov_reviewable",
            "rule_sup_revfrac_mean", "rule_sup_undetermined"]
    out = {}
    for k in keys:
        v = [r["routing"].get(k) for _t, r in recs]
        v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
        out[k] = float(np.mean(v)) if v else float("nan")
    for k in ("band_q", "coverage_true", "coverage_true_nonzero",
              "coverage_weak", "coverage_weak_override", "coverage_weak_confirm",
              "mean_half_width", "mean_abs_error", "n_orders", "n_weak"):
        v = [r["coverage"].get(k) for _t, r in recs if r.get("coverage")]
        v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
        out[k] = float(np.mean(v)) if v else float("nan")
    v = [r["verdict"].get("automation_coverage_unbudgeted")
         for _t, r in recs if r.get("verdict")]
    out["automation_coverage_unbudgeted"] = float(np.mean(v)) if v else float("nan")
    for k in ("pearson_r", "sign_acc_nonzero", "exact_class_acc", "override_rate"):
        out[k] = float(np.mean([r["m0_final"][k] for _t, r in recs
                                if r["m0_final"].get(k) is not None]))
    return out


# --------------------------------------------------------------------------- #
def analyze_head():
    print("=" * 90)
    print("HEADLINE CELL (c9 storm2 u100, beta 1.00, rho 0.25, eps 0), seeds 301-310")
    print("=" * 90)
    by_arm = {}
    for arm, kw in S.ARMS:
        recs = load([t for t in S.tasks_head() if t["arm"] == arm])
        if not recs:
            continue
        by_arm[arm] = {"ladder": ladder(recs), "routing": routing_means(recs),
                       "contrasts": {n: contrast(recs, a, b)
                                     for n, a, b in CONTRASTS},
                       "n_seeds": len(recs)}
    print("\n%-14s %9s %9s %9s %9s %9s   %8s %8s" %
          ("arm", "RULE", "M0", "M0+SUP", "RULE+SUP", "ORACLE",
           "M0 gain", "M0S gain"))
    for arm in by_arm:
        L = by_arm[arm]["ladder"]
        print("%-14s %9.1f %9.1f %9.1f %9.1f %9.1f   %7.2f%% %7.2f%%"
              % (arm, L["rule"]["twt_mean"], L["m0_alone"]["twt_mean"],
                 L["m0_sup"]["twt_mean"], L["rule_sup"]["twt_mean"],
                 L["oracle"]["twt_mean"], L["m0_alone"]["pct_below_rule"],
                 L["m0_sup"]["pct_below_rule"]))

    # Holm within each contrast type, across the arms compared at this cell.
    holm = {}
    arms = [a for a, _ in S.ARMS if a in by_arm]
    for name, _a, _b in CONTRASTS:
        pv = [by_arm[a]["contrasts"][name]["wilcoxon_p"] for a in arms]
        adj = P4.holm(pv)
        holm[name] = {a: {"raw_p": pv[i], "holm_p": adj[i],
                          "pct": by_arm[a]["contrasts"][name]["pct_vs_comparator"],
                          "wtl": by_arm[a]["contrasts"][name]["wtl"]}
                      for i, a in enumerate(arms)}
    print("\nContrasts (seed-averaged per-instance paired Wilcoxon, n=10 instances;"
          "\nHolm within each contrast type across the %d arms compared here)" % len(arms))
    for name, _a, _b in CONTRASTS:
        print("  %s" % name)
        for a in arms:
            h = holm[name][a]
            print("    %-14s %+7.2f%%  W/T/L %2d/%d/%-2d  raw p=%.4g  Holm p=%.4g"
                  % (a, h["pct"], h["wtl"]["W"], h["wtl"]["T"], h["wtl"]["L"],
                     h["raw_p"], h["holm_p"]))

    print("\nRouting telemetry (seed means over the ten held-out instances)")
    print("  %-14s %8s %8s %8s %8s %8s %8s" %
          ("arm", "revfrac", "undet", "cov_all", "band q", "cov_true", "cov_weak"))
    for arm in by_arm:
        r = by_arm[arm]["routing"]
        print("  %-14s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f"
              % (arm, r["m0_sup_revfrac_mean"], r["m0_sup_undetermined"],
                 r["m0_sup_cov_all"], r["band_q"], r["coverage_true"],
                 r["coverage_weak"]))

    # The measured price of deployability: matched protocol, policy only, and
    # tested directly on the same ten held-out instances rather than read off
    # two independently-computed percentages.
    price = {}
    arm_recs = {a: load([t for t in S.tasks_head() if t["arm"] == a])
                for a in arms}
    if "stability" in by_arm and "targeted" in by_arm:
        print("\nDirect arm-vs-arm tests (same instances, same seeds; the ONLY"
              "\ndifference between two split arms is the review policy)")
        pairs = [("stability", "targeted"), ("stability", "margin"),
                 ("stability", "random"), ("stability", "targeted_pub")]
        for key, lab in (("m0_alone", "M0 alone"), ("m0_sup", "M0+SUP")):
            for a, b in pairs:
                if a not in arm_recs or b not in arm_recs:
                    continue
                x = stack(arm_recs[a], key).mean(axis=0)
                y = stack(arm_recs[b], key).mean(axis=0)
                p = P4.paired_wilcoxon(x, y)
                wtl = P4.win_tie_loss(x, y)
                pa = by_arm[a]["ladder"][key]["pct_below_rule"]
                pb = by_arm[b]["ladder"][key]["pct_below_rule"]
                price["%s:%s_vs_%s" % (key, a, b)] = {
                    "a_pct": pa, "b_pct": pb, "delta_pp": pa - pb,
                    "twt_pct_change": 100.0 * (float(y.mean()) - float(x.mean()))
                    / float(y.mean()), "wilcoxon_p": p, "wtl": wtl}
                print("  %-8s %-11s vs %-13s %+6.2f pp (%.2f%% vs %.2f%%)  "
                      "W/T/L %2d/%d/%-2d  p=%.4g"
                      % (lab, a, b, pa - pb, pa, pb, wtl["W"], wtl["T"],
                         wtl["L"], p))
        for key, lab in (("m0_alone", "M0 alone"), ("m0_sup", "M0+SUP")):
            s = by_arm["stability"]["ladder"][key]["pct_below_rule"]
            t = by_arm["targeted"]["ladder"][key]["pct_below_rule"]
            price[key] = {"stability_pct": s, "targeted_pct": t,
                          "price_pp": t - s}
            print("  price of deployability, %-8s: %+.2f percentage points "
                  "(positive = the deployable policy costs)" % (lab, t - s))
    out = {"cell": S.HEAD, "arms": by_arm, "holm": holm, "price": price}
    with open(os.path.join(_OUT, "head_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\n[head] wrote results/y3_w1/head_summary.json")
    return out


def analyze_grid():
    print("=" * 90)
    print("EIGHT-CELL CONTENTION GRID, Holm within each contrast type across cells")
    print("=" * 90)
    tasks = S.tasks_grid()
    out = {}
    for arm in sorted({t["arm"] for t in tasks}):
        cells = {}
        for cell in S.GRID_CELLS:
            recs = load([t for t in tasks if t["arm"] == arm
                         and t["u"] == cell["u"] and t["beta"] == cell["beta"]
                         and t["rho"] == cell["rho"]])
            if not recs:
                continue
            ck = "c9_u%d_b%.2f_r%.2f" % (cell["u"], cell["beta"], cell["rho"])
            cells[ck] = {"ladder": ladder(recs),
                         "contrasts": {n: contrast(recs, a, b)
                                       for n, a, b in CONTRASTS},
                         "routing": routing_means(recs), "n_seeds": len(recs)}
        if not cells:
            continue
        holm = {}
        keys = sorted(cells)
        for name, _a, _b in CONTRASTS:
            pv = [cells[k]["contrasts"][name]["wilcoxon_p"] for k in keys]
            adj = P4.holm(pv)
            holm[name] = {k: {"raw_p": pv[i], "holm_p": adj[i],
                              "pct": cells[k]["contrasts"][name]["pct_vs_comparator"],
                              "wtl": cells[k]["contrasts"][name]["wtl"]}
                          for i, k in enumerate(keys)}
            holm[name]["_n_cells"] = len(keys)
            holm[name]["_n_sig_holm_0.05"] = int(sum(
                1 for i, _k in enumerate(keys)
                if adj[i] is not None and not np.isnan(adj[i]) and adj[i] < 0.05))
        out[arm] = {"cells": cells, "holm": holm}
        print("\narm = %s" % arm)
        print("  %-22s %8s %8s %9s %9s %9s" %
              ("cell", "M0 gain", "M0S gain", "M0vRULE p", "Holm p", "W/T/L"))
        for k in keys:
            c = cells[k]
            h = holm["M0_vs_RULE"][k]
            print("  %-22s %7.2f%% %7.2f%% %9.4g %9.4g   %d/%d/%d"
                  % (k, c["ladder"]["m0_alone"]["pct_below_rule"],
                     c["ladder"]["m0_sup"]["pct_below_rule"], h["raw_p"],
                     h["holm_p"], h["wtl"]["W"], h["wtl"]["T"], h["wtl"]["L"]))
        for name, _a, _b in CONTRASTS:
            print("    %s: %d of %d cells significant at Holm 0.05"
                  % (name, holm[name]["_n_sig_holm_0.05"], holm[name]["_n_cells"]))
    # The price of deployability, per cell: the SAME instances and seeds, with
    # the review policy as the only difference. Holm across the eight cells.
    if "stability" in out and "targeted" in out:
        print("\nPrice of deployability per cell (oracle-informed minus deployable;"
              "\npositive means the deployable policy costs; Holm across the cells)")
        rows = {}
        for cell in GRID_CELLS_SORTED(tasks):
            ck, s_recs, t_recs = cell
            if not s_recs or not t_recs:
                continue
            entry = {}
            for key in ("m0_alone", "m0_sup"):
                x = stack(s_recs, key).mean(axis=0)
                y = stack(t_recs, key).mean(axis=0)
                entry[key] = {
                    "stability_pct": out["stability"]["cells"][ck]["ladder"][key]["pct_below_rule"],
                    "targeted_pct": out["targeted"]["cells"][ck]["ladder"][key]["pct_below_rule"],
                    "wilcoxon_p": P4.paired_wilcoxon(x, y),
                    "wtl": P4.win_tie_loss(x, y)}
                entry[key]["price_pp"] = (entry[key]["targeted_pct"]
                                          - entry[key]["stability_pct"])
            rows[ck] = entry
        keys = sorted(rows)
        for key in ("m0_alone", "m0_sup"):
            adj = P4.holm([rows[k][key]["wilcoxon_p"] for k in keys])
            for i, k in enumerate(keys):
                rows[k][key]["holm_p"] = adj[i]
        print("  %-22s %10s %10s %8s %10s %9s" %
              ("cell", "deployable", "oracle-inf", "price", "W/T/L", "Holm p"))
        for k in keys:
            e = rows[k]["m0_alone"]
            print("  %-22s %9.2f%% %9.2f%% %+7.2fpp %4d/%d/%-3d %9.4g"
                  % (k, e["stability_pct"], e["targeted_pct"], e["price_pp"],
                     e["wtl"]["W"], e["wtl"]["T"], e["wtl"]["L"], e["holm_p"]))
        mean_price = float(np.mean([rows[k]["m0_alone"]["price_pp"] for k in keys]))
        print("  mean price over the %d cells: %+.2f percentage points" %
              (len(keys), mean_price))
        out["price_per_cell"] = rows
        out["mean_price_pp_m0_alone"] = mean_price
    with open(os.path.join(_OUT, "grid_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\n[grid] wrote results/y3_w1/grid_summary.json")
    return out


def GRID_CELLS_SORTED(tasks):
    """(cell key, stability records, targeted records) for each grid cell."""
    out = []
    for cell in S.GRID_CELLS:
        ck = "c9_u%d_b%.2f_r%.2f" % (cell["u"], cell["beta"], cell["rho"])
        sel = lambda arm: load([t for t in tasks if t["arm"] == arm  # noqa: E731
                                and t["u"] == cell["u"]
                                and t["beta"] == cell["beta"]
                                and t["rho"] == cell["rho"]])
        out.append((ck, sel("stability"), sel("targeted")))
    return out


def analyze_curve():
    print("=" * 90)
    print("ROUTING CURVE: automation coverage against true weighted tardiness")
    print("=" * 90)
    tasks = S.tasks_curve()
    rows = []
    for campus in (9, 10):
        for arm in ("stability", "targeted"):
            for rho in sorted({t["rho"] for t in tasks}):
                recs = load([t for t in tasks if t["campus"] == campus
                             and t["arm"] == arm and t["rho"] == rho])
                if not recs:
                    continue
                L = ladder(recs)
                r = routing_means(recs)
                rows.append({
                    "campus": campus, "arm": arm, "rho": rho,
                    "n_seeds": len(recs),
                    "automation_coverage": r["m0_sup_cov_all"],
                    "automation_coverage_reviewable": r["m0_sup_cov_reviewable"],
                    "reviewed_fraction": r["m0_sup_revfrac_mean"],
                    "undetermined_rate": r["m0_sup_undetermined"],
                    "unbudgeted_automation": r["automation_coverage_unbudgeted"],
                    "twt_rule": L["rule"]["twt_mean"],
                    "twt_m0_alone": L["m0_alone"]["twt_mean"],
                    "twt_m0_sup": L["m0_sup"]["twt_mean"],
                    "twt_rule_sup": L["rule_sup"]["twt_mean"],
                    "twt_oracle": L["oracle"]["twt_mean"],
                    "red_m0_alone_pct": L["m0_alone"]["pct_below_rule"],
                    "red_m0_sup_pct": L["m0_sup"]["pct_below_rule"],
                    "red_rule_sup_pct": L["rule_sup"]["pct_below_rule"],
                    "band_q": r["band_q"]})
    print("\n%-3s %-10s %6s %6s %9s %9s %9s %9s %9s" %
          ("C", "arm", "rho", "seeds", "coverage", "undet", "M0 red%",
           "M0+SUP%", "RULE+SUP%"))
    for w in rows:
        print("%-3d %-10s %6.2f %6d %9.4f %9.4f %8.2f%% %8.2f%% %8.2f%%"
              % (w["campus"], w["arm"], w["rho"], w["n_seeds"],
                 w["automation_coverage"], w["undetermined_rate"],
                 w["red_m0_alone_pct"], w["red_m0_sup_pct"],
                 w["red_rule_sup_pct"]))
    import csv as _csv
    p = os.path.join(_OUT, "routing_curve.csv")
    with open(p, "w", newline="") as fh:
        wtr = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        for w in rows:
            wtr.writerow(w)
    with open(os.path.join(_OUT, "routing_curve.json"), "w") as fh:
        json.dump({"rows": rows,
                   "note": "automation_coverage = share of ALL dispatch decisions "
                           "executed without review (1 - reviewed fraction over all "
                           "decisions); undetermined_rate is the share of "
                           "multi-candidate decisions the stability test refers, "
                           "before the budget cap"}, fh, indent=1)
    print("\n[curve] wrote results/y3_w1/routing_curve.{csv,json}")
    return rows


def analyze_alpha():
    print("=" * 90)
    print("CONFORMAL LEVEL SWEEP (headline cell, rho 0.25, seeds 301-303)")
    print("=" * 90)
    tasks = S.tasks_alpha()
    rows = []
    for arm in sorted({t["arm"] for t in tasks}):
        for a in sorted({t["alpha"] for t in tasks if t["arm"] == arm}):
            recs = load([t for t in tasks if t["arm"] == arm and t["alpha"] == a])
            if not recs:
                continue
            L = ladder(recs); r = routing_means(recs)
            rows.append({"arm": arm, "alpha": a, "n_seeds": len(recs),
                         "band_q": r["band_q"],
                         "undetermined_rate": r["m0_sup_undetermined"],
                         "unbudgeted_automation": r["automation_coverage_unbudgeted"],
                         "automation_coverage": r["m0_sup_cov_all"],
                         "coverage_weak": r["coverage_weak"],
                         "coverage_true": r["coverage_true"],
                         "coverage_true_nonzero": r["coverage_true_nonzero"],
                         "red_m0_alone_pct": L["m0_alone"]["pct_below_rule"],
                         "red_m0_sup_pct": L["m0_sup"]["pct_below_rule"]})
    print("\n%-16s %6s %7s %8s %10s %9s %9s %9s" %
          ("arm", "alpha", "q", "undet", "unbudg.aut", "cov_weak", "cov_true",
           "M0 red%"))
    for w in rows:
        print("%-16s %6.2f %7.3f %8.4f %10.4f %9.4f %9.4f %8.2f%%"
              % (w["arm"], w["alpha"], w["band_q"], w["undetermined_rate"],
                 w["unbudgeted_automation"], w["coverage_weak"],
                 w["coverage_true"], w["red_m0_alone_pct"]))
    with open(os.path.join(_OUT, "alpha_summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\n[alpha] wrote results/y3_w1/alpha_summary.json")
    return rows


def analyze_cov():
    print("=" * 90)
    print("BAND COVERAGE PER BETA (EVALUATION ONLY; the band never sees the latent)")
    print("=" * 90)
    tasks = S.tasks_cov()
    rows = []
    for b in sorted({t["beta"] for t in tasks}):
        recs = load([t for t in tasks if t["beta"] == b])
        if not recs:
            continue
        r = routing_means(recs)
        L = ladder(recs)
        rows.append({"beta": b, "n_seeds": len(recs), "band_q": r["band_q"],
                     "coverage_weak": r["coverage_weak"],
                     "coverage_weak_override": r["coverage_weak_override"],
                     "coverage_weak_confirm": r["coverage_weak_confirm"],
                     "coverage_true": r["coverage_true"],
                     "coverage_true_nonzero": r["coverage_true_nonzero"],
                     "mean_half_width": r["mean_half_width"],
                     "mean_abs_error": r["mean_abs_error"],
                     "pearson_r": r["pearson_r"],
                     "sign_acc_nonzero": r["sign_acc_nonzero"],
                     "undetermined_rate": r["m0_sup_undetermined"],
                     "red_m0_alone_pct": L["m0_alone"]["pct_below_rule"],
                     "n_orders": r["n_orders"], "n_weak": r["n_weak"]})
    print("\n%6s %6s %9s %10s %10s %10s %10s %9s" %
          ("beta", "q", "cov_weak", "cov_w_over", "cov_true", "cov_true_nz",
           "mean|err|", "M0 red%"))
    for w in rows:
        print("%6.2f %6.3f %9.4f %10.4f %10.4f %10.4f %10.4f %8.2f%%"
              % (w["beta"], w["band_q"], w["coverage_weak"],
                 w["coverage_weak_override"], w["coverage_true"],
                 w["coverage_true_nonzero"], w["mean_abs_error"],
                 w["red_m0_alone_pct"]))
    with open(os.path.join(_OUT, "coverage_summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\n[cov] wrote results/y3_w1/coverage_summary.json")
    return rows


# --------------------------------------------------------------------------- #
def emit_macros(head, curve, alpha, cov, grid):
    """The \\newcommand block, each line carrying its results file and field."""
    L = []
    a = lambda arm: head["arms"][arm]                      # noqa: E731

    def pct(x):
        return "%.1f\\%%" % x

    if "stability" in head["arms"]:
        s = a("stability")
        L.append((r"\newcommand{\RoutedGain}{%s}" % pct(s["ladder"]["m0_alone"]["pct_below_rule"]),
                  "head_summary.json:arms.stability.ladder.m0_alone.pct_below_rule %.4f"
                  % s["ladder"]["m0_alone"]["pct_below_rule"]))
        # Seed spread in PERCENTAGE POINTS of RULE, the convention \MzeroGainStd
        # uses (committed macros.tex: "m0_alone twt_std_pop 62.31 / RULE").
        L.append((r"\newcommand{\RoutedGainStd}{%.1f}"
                  % (100.0 * s["ladder"]["m0_alone"]["twt_std"] / s["ladder"]["rule"]["twt_mean"]),
                  "head_summary.json:arms.stability.ladder.m0_alone.twt_std %.2f / RULE %.2f"
                  % (s["ladder"]["m0_alone"]["twt_std"], s["ladder"]["rule"]["twt_mean"])))
        L.append((r"\newcommand{\RoutedSupGainStd}{%.1f}"
                  % (100.0 * s["ladder"]["m0_sup"]["twt_std"] / s["ladder"]["rule"]["twt_mean"]),
                  "head_summary.json:arms.stability.ladder.m0_sup.twt_std %.2f / RULE"
                  % s["ladder"]["m0_sup"]["twt_std"]))
        hso = head["holm"]["M0sup_vs_ORACLE"]["stability"]
        L.append((r"\newcommand{\RoutedSupVsOracleWTL}{%d/%d/%d}"
                  % (hso["wtl"]["W"], hso["wtl"]["T"], hso["wtl"]["L"]),
                  "head_summary.json:holm.M0sup_vs_ORACLE.stability.wtl "
                  "(M0+SUP against the omniscient reference)"))
        L.append((r"\newcommand{\RoutedSupVsOracleP}{%.2f}" % hso["raw_p"],
                  "head_summary.json:holm.M0sup_vs_ORACLE.stability.raw_p %.6g "
                  "(raw, 10 seeds)" % hso["raw_p"]))
        L.append((r"\newcommand{\RoutedSupGain}{%s}" % pct(s["ladder"]["m0_sup"]["pct_below_rule"]),
                  "head_summary.json:arms.stability.ladder.m0_sup.pct_below_rule %.4f"
                  % s["ladder"]["m0_sup"]["pct_below_rule"]))
        h = head["holm"]["M0_vs_RULE"]["stability"]
        L.append((r"\newcommand{\RoutedRawP}{%.4g}" % h["raw_p"],
                  "head_summary.json:holm.M0_vs_RULE.stability.raw_p"))
        L.append((r"\newcommand{\RoutedHolmP}{%.3f}" % h["holm_p"],
                  "head_summary.json:holm.M0_vs_RULE.stability.holm_p %.6g" % h["holm_p"]))
        L.append((r"\newcommand{\RoutedWTL}{%d/%d/%d}"
                  % (h["wtl"]["W"], h["wtl"]["T"], h["wtl"]["L"]),
                  "head_summary.json:holm.M0_vs_RULE.stability.wtl"))
        r = s["routing"]
        L.append((r"\newcommand{\BandAlpha}{0.1}", "sweep task field alpha (conformal level)"))
        L.append((r"\newcommand{\BandQ}{%.2f}" % r["band_q"],
                  "head_summary.json:arms.stability.routing.band_q (class-shift units)"))
        L.append((r"\newcommand{\BandCovWeak}{%s}" % pct(100 * r["coverage_weak"]),
                  "head_summary.json:arms.stability.routing.coverage_weak %.4f"
                  % r["coverage_weak"]))
        L.append((r"\newcommand{\BandCovTrue}{%s}" % pct(100 * r["coverage_true"]),
                  "head_summary.json:arms.stability.routing.coverage_true %.4f"
                  % r["coverage_true"]))
        L.append((r"\newcommand{\BandCovTrueNz}{%s}" % pct(100 * r["coverage_true_nonzero"]),
                  "head_summary.json:arms.stability.routing.coverage_true_nonzero %.4f"
                  % r["coverage_true_nonzero"]))
        L.append((r"\newcommand{\UndetRate}{%s}" % pct(100 * r["m0_sup_undetermined"]),
                  "head_summary.json:arms.stability.routing.m0_sup_undetermined %.4f"
                  % r["m0_sup_undetermined"]))
        L.append((r"\newcommand{\AutoCoverage}{%s}" % pct(100 * r["m0_sup_cov_all"]),
                  "head_summary.json:arms.stability.routing.m0_sup_cov_all %.4f"
                  % r["m0_sup_cov_all"]))
        L.append((r"\newcommand{\AutoCoverageUnbudgeted}{%s}"
                  % pct(100 * r["automation_coverage_unbudgeted"]),
                  "head_summary.json:arms.stability.routing.automation_coverage_unbudgeted %.4f"
                  % r["automation_coverage_unbudgeted"]))
    if "targeted" in head["arms"]:
        t = a("targeted")
        L.append((r"\newcommand{\OracleRoutedGain}{%s}"
                  % pct(t["ladder"]["m0_alone"]["pct_below_rule"]),
                  "head_summary.json:arms.targeted.ladder.m0_alone.pct_below_rule %.4f "
                  "(ORACLE-INFORMED UPPER REFERENCE)"
                  % t["ladder"]["m0_alone"]["pct_below_rule"]))
    if head.get("price"):
        p = head["price"].get("m0_alone")
        if p:
            L.append((r"\newcommand{\DeployPrice}{%.1f}" % (p["targeted_pct"] - p["stability_pct"]),
                      "head_summary.json:price.m0_alone (percentage points, "
                      "oracle-informed minus deployable; NEGATIVE means the "
                      "deployable policy is better)"))
    if "margin" in head["arms"]:
        L.append((r"\newcommand{\MarginGain}{%s}"
                  % pct(a("margin")["ladder"]["m0_alone"]["pct_below_rule"]),
                  "head_summary.json:arms.margin.ladder.m0_alone.pct_below_rule %.4f "
                  "(observable margin-only control)"
                  % a("margin")["ladder"]["m0_alone"]["pct_below_rule"]))
    if "random" in head["arms"]:
        L.append((r"\newcommand{\RandomRoutedGain}{%s}"
                  % pct(a("random")["ladder"]["m0_alone"]["pct_below_rule"]),
                  "head_summary.json:arms.random.ladder.m0_alone.pct_below_rule %.4f "
                  "(lower control)"
                  % a("random")["ladder"]["m0_alone"]["pct_below_rule"]))
    if head.get("price"):
        d = head["price"].get("m0_alone:stability_vs_targeted")
        if d:
            L.append((r"\newcommand{\DeployPriceP}{%.3f}" % d["wilcoxon_p"],
                      "head_summary.json:price['m0_alone:stability_vs_targeted']"
                      ".wilcoxon_p (raw; NOT corrected for the eight arm-vs-arm "
                      "tests, where it would not survive)"))
            L.append((r"\newcommand{\DeployPriceWTL}{%d/%d/%d}"
                      % (d["wtl"]["W"], d["wtl"]["T"], d["wtl"]["L"]),
                      "head_summary.json:price['m0_alone:stability_vs_targeted'].wtl"))
    if grid and "mean_price_pp_m0_alone" in grid:
        L.append((r"\newcommand{\DeployPriceGrid}{%.1f}" % grid["mean_price_pp_m0_alone"],
                  "grid_summary.json:mean_price_pp_m0_alone (percentage points, "
                  "mean over the eight-cell grid; NEGATIVE means the deployable "
                  "policy is better)"))
        pp = [v["m0_alone"] for v in grid["price_per_cell"].values()]
        L.append((r"\newcommand{\DeployPriceGridRange}{$%+.1f$ to $%+.1f$}"
                  % (min(x["price_pp"] for x in pp), max(x["price_pp"] for x in pp)),
                  "grid_summary.json:price_per_cell, per-cell range in percentage "
                  "points; no cell survives Holm (smallest Holm p = %.3f)"
                  % min(x["holm_p"] for x in pp)))
    if curve:
        c10 = [w for w in curve if w["campus"] == 10 and abs(w["rho"] - 0.25) < 1e-9]
        if c10:
            L.append((r"\newcommand{\RoutedGainCten}{%s}" % pct(c10[0]["red_m0_alone_pct"]),
                      "routing_curve.json:rows[c10,stability,rho=0.25].red_m0_alone_pct "
                      "%.4f (confirmation cell, 3 seeds)" % c10[0]["red_m0_alone_pct"]))
            L.append((r"\newcommand{\AutoCoverageCten}{%s}"
                      % pct(100 * c10[0]["automation_coverage"]),
                      "routing_curve.json:rows[c10,stability,rho=0.25]"
                      ".automation_coverage %.4f" % c10[0]["automation_coverage"]))
    diag = os.path.join(_OUT, "targeted_clause_diagnostic.json")
    if os.path.exists(diag):
        with open(diag) as fh:
            dg = json.load(fh)
        L.append((r"\newcommand{\OracleClauseRate}{%s}" % pct(100 * dg["has_plus2_rate"]),
                  "targeted_clause_diagnostic.json:has_plus2_rate %.4f -- share of "
                  "multi-candidate decisions the ORACLE-INFORMED policy flags, "
                  "against a budget of 0.25" % dg["has_plus2_rate"]))
    if grid:
        for arm in ("stability", "targeted"):
            if arm not in grid:
                continue
            h = grid[arm]["holm"]["M0_vs_RULE"]
            name = "Routed" if arm == "stability" else "OracleRouted"
            L.append((r"\newcommand{\%sGridSig}{%d}" % (name, h["_n_sig_holm_0.05"]),
                      "grid_summary.json:%s.holm.M0_vs_RULE._n_sig_holm_0.05 of %d cells"
                      % (arm, h["_n_cells"])))
            worst = max(v["holm_p"] for k, v in h.items() if not k.startswith("_"))
            L.append((r"\newcommand{\%sGridWorstHolmP}{%.4g}" % (name, worst),
                      "grid_summary.json:%s.holm.M0_vs_RULE, largest Holm p over the "
                      "eight-cell grid" % arm))
    if curve:
        for w in curve:
            if w["campus"] == 9 and w["arm"] == "stability" and abs(w["rho"] - 0.05) < 1e-9:
                L.append((r"\newcommand{\CoverageAtRhoFive}{%s}"
                          % pct(100 * w["automation_coverage"]),
                          "routing_curve.json:rows[c9,stability,rho=0.05].automation_coverage %.4f"
                          % w["automation_coverage"]))
                L.append((r"\newcommand{\GainAtRhoFive}{%s}" % pct(w["red_m0_alone_pct"]),
                          "routing_curve.json:rows[c9,stability,rho=0.05].red_m0_alone_pct %.4f"
                          % w["red_m0_alone_pct"]))
    if cov:
        b0 = [w for w in cov if abs(w["beta"]) < 1e-9]
        if b0:
            L.append((r"\newcommand{\BandCovTrueBetaZero}{%s}" % pct(100 * b0[0]["coverage_true"]),
                      "coverage_summary.json:[beta=0].coverage_true %.4f"
                      % b0[0]["coverage_true"]))
    if alpha:
        best = min(alpha, key=lambda w: abs(w["alpha"] - 0.5))
        L.append((r"\newcommand{\AutoCoverageLooseAlpha}{%s}"
                  % pct(100 * best["unbudgeted_automation"]),
                  "alpha_summary.json:[alpha=%.2f].unbudgeted_automation %.4f"
                  % (best["alpha"], best["unbudgeted_automation"])))

    txt = ["% W1 macros -- deployable review routing. Do NOT paste blindly:",
           "% every value below is traced to a file under results/y3_w1/.",
           ""]
    w = max(len(m) for m, _c in L) if L else 0
    for m, c in L:
        txt.append("%-*s %% %s" % (w, m, c))
    body = "\n".join(txt)
    p = os.path.join(_OUT, "macros_w1.tex")
    with open(p, "w") as fh:
        fh.write(body + "\n")
    print("\n" + body)
    print("\n[macros] wrote %s" % p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["head", "grid", "curve", "alpha", "cov", "macros",
                             "all"])
    args = ap.parse_args(argv)
    head = curve = alpha = cov = grid = None
    if args.part in ("head", "macros", "all"):
        head = analyze_head()
    if args.part in ("grid", "macros", "all"):
        try:
            grid = analyze_grid()
        except Exception as e:
            print("[grid] skipped (%s)" % e)
    if args.part in ("curve", "macros", "all"):
        try:
            curve = analyze_curve()
        except Exception as e:
            print("[curve] skipped (%s)" % e)
    if args.part in ("alpha", "macros", "all"):
        try:
            alpha = analyze_alpha()
        except Exception as e:
            print("[alpha] skipped (%s)" % e)
    if args.part in ("cov", "macros", "all"):
        try:
            cov = analyze_cov()
        except Exception as e:
            print("[cov] skipped (%s)" % e)
    if args.part in ("macros", "all") and head:
        emit_macros(head, curve, alpha, cov, grid)


if __name__ == "__main__":
    main()
