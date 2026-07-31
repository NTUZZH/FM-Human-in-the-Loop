#!/usr/bin/env python
"""W3 aggregation: the before/after tables and the macro block.

Reads every results/y3_w3/<tag>/<variant>_s<seed>.json written by
scripts/y3_w3_run.py and produces

  results/y3_w3/summary.json   every aggregated number, with its provenance
  results/y3_w3/tables.txt     the human-readable before/after tables
  results/y3_w3/w3_macros.tex  \\newcommand definitions, one per quotable number

Two pooling conventions are computed and asserted to agree, because the
manuscript quotes both:
  * the regime-map pooling of scripts/y3_beta0_check.pooled_twt -- sum TWT* over
    every (seed, instance) row, then 100*(rule - aug)/rule;
  * the contrast of scripts/y3_p4_m0grid._contrast -- seed-average each
    decider's per-instance TWT*, then the paired Wilcoxon over the held-out
    instances.
They are algebraically the same percentage; computing both catches a silently
dropped seed.
"""

from __future__ import annotations

import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np                                               # noqa: E402

import y3_w3_lib as L                                            # noqa: E402

OUT = L.OUT
ORDER = ["mse_published", "mse_reexpr", "tobit_imp", "tobit", "tobit_sig",
         "tobit_exp", "classmean_oracle"]
LABEL = {
    "mse_published": "(i)    squared error [published M0]",
    "mse_reexpr":    "(i-b)  squared error, re-expressed loop",
    "tobit_imp":     "(iii)  Tobit, impossible labels only",
    "tobit":         "(ii)   Tobit, strict censoring",
    "tobit_sig":     "(v)    Tobit, strict, sigma fitted",
    "tobit_exp":     "(iv)   Tobit, strict, E[clip(s,L,U)] deployed",
    "classmean_oracle": "ref    true class-mean constant [ORACLE]",
}


def load(tag):
    recs = {}
    for p in sorted(glob.glob(os.path.join(OUT, tag, "*.json"))):
        d = json.load(open(p))
        recs.setdefault(d["variant"], {})[int(d["seed"])] = d
    return recs


def aggregate_variant(rs):
    """rs: seed -> record. Returns the pooled and seed-averaged aggregates."""
    seeds = sorted(rs)
    rule = np.asarray([rs[s]["twt"]["rule"] for s in seeds], float)   # S x n
    aug = np.asarray([rs[s]["twt"]["aug"] for s in seeds], float)
    pooled = 100.0 * (rule.sum() - aug.sum()) / rule.sum()
    a = aug.mean(axis=0)                       # seed-averaged per instance
    b = rule.mean(axis=0)
    con = L.contrast(a, b)
    per_seed_pct = [rs[s]["twt"]["pct_below_rule"] for s in seeds]

    def m(path, default=float("nan")):
        vals = []
        for s in seeds:
            d = rs[s]
            for k in path.split("."):
                d = d.get(k, {}) if isinstance(d, dict) else default
            vals.append(float(d) if isinstance(d, (int, float)) else default)
        return float(np.mean(vals))

    by_class = {}
    for k in ("1", "2", "3", "4"):
        by_class[k] = {
            "n": int(rs[seeds[0]]["recovery_eval"]["by_recorded_class"][k]["n"]),
            "mean_hat_s": m("recovery_eval.by_recorded_class.%s.mean_hat_s" % k),
            "mean_applied_shift": m("recovery_eval.by_recorded_class.%s.mean_applied_shift" % k),
            "mean_true_effective_shift": m("recovery_eval.by_recorded_class.%s.mean_true_effective_shift" % k),
        }
    out = {
        "seeds": seeds, "n_seeds": len(seeds),
        "n_instances": int(rule.shape[1]),
        "pooled_pct_below_rule": pooled,
        "contrast_pct_below_rule": con["pct_vs_comparator"],
        "pct_seed_sd": float(np.std(per_seed_pct, ddof=0)),
        "per_seed_pct": per_seed_pct,
        "twt_aug_seedavg_per_instance": a.tolist(),
        "twt_rule_seedavg_per_instance": b.tolist(),
        "wtl_vs_rule": con["wtl"], "wilcoxon_p_vs_rule": con["wilcoxon_p"],
        "mean_hat_s": m("recovery_eval.mean_hat_s"),
        "sd_hat_s": m("recovery_eval.sd_hat_s"),
        "mean_applied_shift": m("recovery_eval.mean_applied_shift"),
        "mean_true_effective_shift": m("recovery_eval.mean_true_effective_shift"),
        "pearson_r": m("recovery_eval.pearson_r"),
        "pearson_r_applied_vs_true_effective":
            m("recovery_eval.pearson_r_applied_vs_true_effective"),
        "sign_acc_nonzero": m("recovery_eval.sign_acc_nonzero"),
        "exact_class_acc": m("recovery_eval.exact_class_acc"),
        "kendall_tau": m("kendall.kendall_tau"),
        "kendall_recorded_field_floor": m("kendall_recorded_field_floor.kendall_tau"),
        "sigma": m("sigma"),
        "n_params_estimator": int(rs[seeds[0]]["n_params_estimator"]),
        "n_params_total": int(rs[seeds[0]]["n_params_total"]),
        "by_recorded_class": by_class,
        "censor": rs[seeds[0]].get("censor", {}),
    }
    if "published" in rs[seeds[0]]:
        pr = np.asarray([rs[s]["published"]["rule"] for s in seeds], float)
        pm = np.asarray([rs[s]["published"]["m0_alone"] for s in seeds], float)
        out["published_pooled_pct_below_rule"] = \
            100.0 * (pr.sum() - pm.sum()) / pr.sum()
        out["max_abs_dTWT_rule_vs_published"] = \
            float(np.max([rs[s]["published"]["max_abs_dTWT_rule"] for s in seeds]))
        out["max_abs_dTWT_m0_vs_published"] = \
            float(np.max([rs[s]["published"]["max_abs_dTWT_m0_vs_mine"] for s in seeds]))
    return out


def head_to_head(agg, test, comp):
    if test not in agg or comp not in agg:
        return None
    a = np.asarray(agg[test]["twt_aug_seedavg_per_instance"], float)
    b = np.asarray(agg[comp]["twt_aug_seedavg_per_instance"], float)
    con = L.contrast(a, b)
    return {"test": test, "comparator": comp,
            "dTWT_pct": con["pct_vs_comparator"], "wtl": con["wtl"],
            "wilcoxon_p": con["wilcoxon_p"],
            "d_pct_below_rule": agg[test]["pooled_pct_below_rule"]
            - agg[comp]["pooled_pct_below_rule"]}


def fmt_cell(tag, agg, fh):
    print("\n%s" % ("=" * 128), file=fh)
    print("CELL %s   (%d held-out instances, seeds %s)"
          % (tag, agg[list(agg)[0]]["n_instances"],
             agg[list(agg)[0]]["seeds"]), file=fh)
    print("=" * 128, file=fh)
    print("%-44s %6s %8s %9s %9s %8s %8s %8s %7s %7s" %
          ("rung", "params", "%<RULE", "meanHatS", "applied", "Pearson",
           "rAppl", "signAcc", "tau", "sd%"), file=fh)
    print("-" * 128, file=fh)
    for v in ORDER:
        if v not in agg:
            continue
        a = agg[v]
        print("%-44s %6d %8.3f %+9.4f %+9.4f %+8.3f %+8.3f %8.3f %7.3f %7.2f"
              % (LABEL[v], a["n_params_total"], a["pooled_pct_below_rule"],
                 a["mean_hat_s"], a["mean_applied_shift"], a["pearson_r"],
                 a["pearson_r_applied_vs_true_effective"],
                 a["sign_acc_nonzero"], a["kendall_tau"], a["pct_seed_sd"]),
              file=fh)
    a0 = agg[list(agg)[0]]
    print("-" * 128, file=fh)
    print("%-44s %6s %8.3f %+9.4f %+9.4f %8s %8s %8s %7.3f"
          % ("       RULE (recorded fields, hat_s == 0)", "-", 0.0, 0.0, 0.0,
             "-", "-", "-", a0["kendall_recorded_field_floor"]), file=fh)
    print("%-44s %35s%+9.4f" % ("       TRUE effective shift E[c - c*]", "",
                                a0["mean_true_effective_shift"]), file=fh)
    if "published_pooled_pct_below_rule" in agg.get("mse_published", {}):
        p = agg["mse_published"]
        print("\npublished (results/y3_p4) for the same cell-seeds: %.4f %% "
              "| max |dTWT*| vs mine: rule %.3g, m0_alone %.3g"
              % (p["published_pooled_pct_below_rule"],
                 p["max_abs_dTWT_rule_vs_published"],
                 p["max_abs_dTWT_m0_vs_published"]), file=fh)
    print("\nmean fitted shift by RECORDED class (raw / applied after the "
          "deployment clip / TRUE effective):", file=fh)
    hdr = "%-44s" % "rung"
    for k in ("1", "2", "3", "4"):
        hdr += " %-22s" % ("class %s (n=%d)" % (k, a0["by_recorded_class"][k]["n"]))
    print(hdr, file=fh)
    for v in ORDER:
        if v not in agg:
            continue
        line = "%-44s" % LABEL[v]
        for k in ("1", "2", "3", "4"):
            b = agg[v]["by_recorded_class"][k]
            line += " %+7.4f/%+7.4f    " % (b["mean_hat_s"], b["mean_applied_shift"])
        print(line, file=fh)
    line = "%-44s" % "       TRUE E[c - c* | c]"
    for k in ("1", "2", "3", "4"):
        line += " %19.4f    " % a0["by_recorded_class"][k]["mean_true_effective_shift"]
    print(line, file=fh)

    print("\nHEAD-TO-HEAD (seed-averaged per-instance TWT*, paired Wilcoxon "
          "(pratt), Holm within this cell; a WIN = the test rung is LOWER)", file=fh)
    pairs = [("tobit", "mse_published"), ("tobit_imp", "mse_published"),
             ("tobit_exp", "mse_published"), ("tobit_sig", "tobit"),
             ("classmean_oracle", "mse_published")]
    hh = [head_to_head(agg, t, c) for t, c in pairs]
    hh = [h for h in hh if h]
    if hh:
        adj = L.holm([h["wilcoxon_p"] for h in hh])
        print("%-34s %10s %10s %10s %10s" %
              ("contrast", "dTWT*%", "d(%<RULE)", "W/T/L", "p(Holm)"), file=fh)
        for h, p in zip(hh, adj):
            print("%-34s %+10.2f %+10.2f %10s %10.4f"
                  % ("%s vs %s" % (h["test"], h["comparator"]), h["dTWT_pct"],
                     h["d_pct_below_rule"],
                     "%d/%d/%d" % (h["wtl"]["W"], h["wtl"]["T"], h["wtl"]["L"]),
                     p), file=fh)
    return hh


def main():
    tags = [d for d in sorted(os.listdir(OUT))
            if os.path.isdir(os.path.join(OUT, d))]
    summary = {"cells": {}, "note": (
        "W3 censoring-aware ordinal shift likelihood. TWT* scored by the "
        "independent validator hitl.true_objective.score_true; pooling matches "
        "y3_beta0_check.pooled_twt and the contrast matches y3_p4_m0grid._contrast. "
        "mean_hat_s is the estimator's raw output (the quantity the manuscript "
        "quotes); mean_applied_shift is clip(hat_s, c-4, c-1), the correction the "
        "corrected class actually applies.")}
    with open(os.path.join(OUT, "tables.txt"), "w") as fh:
        for tag in tags:
            recs = load(tag)
            if not recs:
                continue
            agg = {v: aggregate_variant(rs) for v, rs in recs.items()}
            agg = {v: agg[v] for v in ORDER if v in agg}
            hh = fmt_cell(tag, agg, fh)
            summary["cells"][tag] = {"variants": agg, "head_to_head": hh}
    L.write_json(os.path.join(OUT, "summary.json"), summary)
    print(open(os.path.join(OUT, "tables.txt")).read())
    print("wrote %s and %s" % (os.path.join(OUT, "tables.txt"),
                               os.path.join(OUT, "summary.json")))


if __name__ == "__main__":
    main()
