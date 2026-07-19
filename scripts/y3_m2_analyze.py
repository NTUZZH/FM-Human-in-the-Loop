#!/usr/bin/env python
"""Analyze the M2 ablation outputs into readable tables (parity + active).

Reads results/y3_p5/m2/{m2_summary.json, m2_recovery.csv} and prints:
  1. Parity ladder (RULE / M0 / M2-TGT / M2-ACT / ORACLE) + gap-closed% +
     M2t-vs-M0 and *-vs-RULE seed-averaged paired contrasts, per beta.
  2. Active-elicitation: reviewed fraction (matched budget), cumulative overrides,
     final recovery accuracy, M2a-vs-M2t TWT* contrast.
  3. Recovery curve (seed-averaged accuracy + held-out TWT* vs cum overrides) for
     M0 / M2t / M2a, and the override count each method needs to reach fixed
     accuracy thresholds (the active-elicitation efficiency lens).
"""
from __future__ import annotations
import csv
import json
import os
from collections import defaultdict

import numpy as np

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "y3_p5", "m2")
ORACLE_KEY = "oracle"
ALONE = "m0_alone"        # generic 'alone' key in the ladder dict


def load():
    with open(os.path.join(_OUT, "m2_summary.json")) as fh:
        summ = json.load(fh)
    rec = defaultdict(lambda: defaultdict(list))   # (beta,method) -> iter-> rows
    rows = []
    with open(os.path.join(_OUT, "m2_recovery.csv")) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return summ, rows


def fmt_contrast(c):
    w = c["wtl"]
    return "%+.1f%% %d/%d/%d p=%.3f" % (c["pct_gain"], w["W"], w["T"], w["L"],
                                        c["wilcoxon_p"])


def parity_tables(summ):
    print("\n" + "=" * 78)
    print("1. PARITY LADDER  (TWT*(w*,d*), c9 storm2 u100, rho0.25, held-out n=10)")
    print("=" * 78)
    for cell, c in sorted(summ["cells"].items()):
        beta = c["beta"]; ns = c["n_seeds"]
        lad = c["ladder"]
        rule = lad["m0"]["rule"]["twt_mean"]
        orac = lad["m0"]["oracle"]["twt_mean"]
        m0 = lad["m0"][ALONE]["twt_mean"]
        m2t = lad["m2t"][ALONE]["twt_mean"]
        m2a = lad["m2a"][ALONE]["twt_mean"]

        def gap(v):
            return 100.0 * (rule - v) / (rule - orac) if rule > orac else float("nan")
        print("\nbeta=%.2f  (n_seeds=%d, seeds %s)" % (beta, ns, c["seeds"]))
        print("  RULE=%.0f  ORACLE=%.0f   (gap = RULE-ORACLE = %.0f)"
              % (rule, orac, rule - orac))
        print("  %-8s %8s %10s %10s" % ("method", "TWT*", "%below RULE", "%gap closed"))
        for name, v in (("M0", m0), ("M2-TGT", m2t), ("M2-ACT", m2a)):
            print("  %-8s %8.0f %9.1f%% %10.0f%%"
                  % (name, v, 100.0 * (rule - v) / rule, gap(v)))
        ct = c["contrasts"]
        print("  contrasts (seed-averaged paired Wilcoxon, W=test lower TWT*):")
        print("    M0    vs RULE : %s" % fmt_contrast(ct["M0_alone_vs_RULE"]))
        print("    M2-TGT vs RULE: %s" % fmt_contrast(ct["M2t_alone_vs_RULE"]))
        print("    M2-TGT vs M0  : %s  <- PARITY TEST" % fmt_contrast(ct["M2t_vs_M0_alone"]))
        print("    M2-ACT vs M0  : %s" % fmt_contrast(ct["M2a_vs_M0_alone"]))
        print("    M2-ACT vs M2-TGT: %s  <- ACTIVE vs FIXED" % fmt_contrast(ct["M2a_vs_M2t_alone"]))


def active_tables(summ):
    print("\n" + "=" * 78)
    print("2. ACTIVE ELICITATION  (matched review budget rho=0.25)")
    print("=" * 78)
    for cell, c in sorted(summ["cells"].items()):
        beta = c["beta"]; lad = c["ladder"]
        print("\nbeta=%.2f" % beta)
        print("  %-8s %10s %10s %9s %8s %8s" %
              ("method", "rev.frac", "cum_over", "sign_acc", "pear_r", "TWT*"))
        for name, key in (("M0", "m0"), ("M2-TGT", "m2t"), ("M2-ACT", "m2a")):
            fr = lad[key]["final_recovery"]
            twt = lad[key][ALONE]["twt_mean"]
            rf = fr["reviewed_fraction"]
            rf_s = ("%.3f" % rf) if rf == rf else "n/a(=M2t)"
            print("  %-8s %10s %10.0f %9.3f %8.3f %8.0f"
                  % (name, rf_s, fr["cum_overrides"], fr["sign_acc_nonzero"],
                     fr["pearson_r"], twt))


def recovery_curves(rows):
    print("\n" + "=" * 78)
    print("3. RECOVERY CURVE  (seed-averaged accuracy & held-out TWT* vs cum overrides)")
    print("=" * 78)
    # group by (beta, method, iter) -> average over seeds
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["beta"], r["method"])
        g[key][int(r["iter"])].append(r)
    betas = sorted(set(r["beta"] for r in rows), key=float)
    for beta in betas:
        print("\nbeta=%s" % beta)
        for method in ("m0", "m2t", "m2a"):
            iters = g[(beta, method)]
            if not iters:
                continue
            print("  %-6s: iter  cum_over  sign_acc  pear_r  alone_TWT*" % method)
            for it in sorted(iters):
                rs = iters[it]
                cum = np.mean([float(x["cum_overrides"]) for x in rs])
                sa = np.mean([float(x["sign_acc_nonzero"]) for x in rs])
                pr = np.mean([float(x["pearson_r"]) for x in rs])
                tw = np.mean([float(x["alone_twt"]) for x in rs])
                print("          %2d %9.0f %9.3f %7.3f %10.0f" % (it, cum, sa, pr, tw))

    # matched-override-budget efficiency: overrides to reach accuracy thresholds
    print("\n  --- overrides to first reach a sign-acc / pearson-r threshold ---")
    for beta in betas:
        print("  beta=%s" % beta)
        for method in ("m0", "m2t", "m2a"):
            iters = g[(beta, method)]
            if not iters:
                continue
            pts = []
            for it in sorted(iters):
                rs = iters[it]
                pts.append((np.mean([float(x["cum_overrides"]) for x in rs]),
                            np.mean([float(x["sign_acc_nonzero"]) for x in rs]),
                            np.mean([float(x["pearson_r"]) for x in rs])))
            def first_over(thr, idx):
                for cum, sa, pr in pts:
                    if (sa if idx == 1 else pr) >= thr:
                        return cum
                return float("nan")
            print("    %-6s  overrides@sign>=0.70=%6.0f  @r>=0.30=%6.0f  (max r=%.3f)"
                  % (method, first_over(0.70, 1), first_over(0.30, 2),
                     max(p[2] for p in pts)))


def main():
    summ, rows = load()
    parity_tables(summ)
    active_tables(summ)
    recovery_curves(rows)
    print("\n[config]", json.dumps(summ["config"], indent=0)[:400])


if __name__ == "__main__":
    main()
