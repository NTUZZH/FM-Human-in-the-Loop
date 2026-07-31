#!/usr/bin/env python
"""W2: summarise the estimator ladder into the table and the macro block.

Reads results/y3_w2/<tag>.json (written by scripts/y3_w2_ladder.py) and writes
  results/y3_w2/<tag>_summary.json   machine-readable, one entry per rung
  results/y3_w2/<tag>_table.txt      the ladder table as it goes in the report
  results/y3_w2/<tag>_macros.tex     \\newcommand block, each with the results
                                     file and field it came from

Aggregation matches the paper: seed-average each decider's per-instance TWT*,
then a two-sided paired Wilcoxon signed-rank test (zero_method='pratt') over the
held-out instances, Holm-corrected within the family of contrasts reported here.
Recovery metrics are averaged over seeds and reported with the seed standard
deviation, because a single seed of this pipeline is not interpretable: a pure
torch thread-count change (1 -> 4) moves the seed-301 headline by 1.56
percentage points, which is the size of the published seed spread.
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np                                              # noqa: E402

import y3_w2_lib as L                                           # noqa: E402

LABEL = {
    "mse_published": "(i)    squared error, per-order  [published M0]",
    "mse_es":        "(i-es) squared error, per-order  + split/early-stop",
    "choice":        "(ii)   choice likelihood, per-order",
    "choice_queue":  "(iii)  choice likelihood + queue conditioning",
    "choice_tol":    "(iia)  choice likelihood, tolerance-aware confirmations",
    "choice_queue_tol": "(iiia) (iii) + tolerance-aware confirmations",
    "choice_queue_k64": "(iiib) (iii) with the choice set capped at K<=64",
}


def agg(per_seed, variant):
    """Seed-average per-instance TWT*, and mean/sd over seeds for the rest."""
    seeds = [s for s in sorted(per_seed, key=int) if variant in per_seed[s]]
    rows = [per_seed[s][variant] for s in seeds]
    if not rows:
        return None
    aug = np.mean([r["twt_aug"] for r in rows], axis=0)     # seed-averaged
    rule = np.mean([r["twt_rule"] for r in rows], axis=0)
    con = L.contrast(aug, rule)
    grab = lambda path: np.asarray(  # noqa: E731
        [_dig(r, path) for r in rows], float)
    out = {
        "variant": variant, "label": LABEL.get(variant, variant),
        "n_seeds": len(rows), "seeds": [int(s) for s in seeds],
        "twt_aug_by_seed": {s: r["twt_aug"] for s, r in zip(seeds, rows)},
        "twt_rule_by_seed": {s: r["twt_rule"] for s, r in zip(seeds, rows)},
        "twt_aug_seed_avg_per_instance": aug.tolist(),
        "twt_rule_seed_avg_per_instance": rule.tolist(),
        "twt_aug_mean": float(aug.mean()), "twt_rule_mean": float(rule.mean()),
        "pct_below_rule": con["pct_vs_comparator"],
        "wtl": con["wtl"], "wilcoxon_p": con["wilcoxon_p"],
        "n_instances": con["n_instances"],
        "pct_below_rule_per_seed": [
            100.0 * (np.mean(r["twt_rule"]) - np.mean(r["twt_aug"]))
            / np.mean(r["twt_rule"]) for r in rows],
        "n_params_estimator": rows[0]["n_params_estimator"],
        "n_params_total": rows[0]["n_params_total"],
    }
    out["pct_below_rule_seed_sd"] = float(np.std(out["pct_below_rule_per_seed"]))
    for name, path in (("pearson_r", "recovery.pearson_r"),
                       ("sign_acc", "recovery.sign_acc_nonzero"),
                       ("exact_acc", "recovery.exact_class_acc"),
                       ("mean_hat_s", "recovery.mean_hat_s"),
                       ("sd_hat_s", "recovery.sd_hat_s"),
                       ("kendall_tau", "kendall_eval.kendall_tau"),
                       ("choice_ll", "choice_ll.ll"),
                       ("choice_ll_overrides", "choice_ll.ll_overrides"),
                       ("choice_top1", "choice_ll.acc_top1"),
                       ("tau", "choice_tau_calibrated")):
        v = grab(path)
        out[name] = float(np.nanmean(v))
        out[name + "_sd"] = float(np.nanstd(v))
    if "ms_per_decision" in rows[0]:
        out["ms_per_decision"] = float(np.mean([r["ms_per_decision"] for r in rows]))
    return out


def _overrides(src):
    try:
        raw = json.load(open(src))
        return int(raw[sorted(raw, key=int)[0]]["mse_published"]["n_overrides_total"])
    except Exception:
        return -1


def _dig(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ladder")
    a = ap.parse_args()
    src = os.path.join(L.OUT, "%s.json" % a.tag)
    per_seed = json.load(open(src))
    seeds = sorted(per_seed, key=int)
    variants = [v for v in L.VARIANTS if v in per_seed[seeds[0]]]

    rows = [agg(per_seed, v) for v in variants]
    rows = [r for r in rows if r]
    # Holm within the family of "variant vs RULE" contrasts reported here.
    adj = L.holm([r["wilcoxon_p"] for r in rows])
    for r, p in zip(rows, adj):
        r["wilcoxon_p_holm"] = p

    # DIRECT rung-to-rung contrasts: the actual test of each separable claim.
    #   (i)   -> (i-es) : the fitting protocol
    #   (i-es)-> (ii)   : CLAIM A, the likelihood correction
    #   (ii)  -> (iii)  : CLAIM B, the queue conditioning
    by_v = {r["variant"]: r for r in rows}
    pairs = [("mse_es", "mse_published", "protocol (split + early stop)"),
             ("choice", "mse_es", "CLAIM A: likelihood correction"),
             ("choice_queue", "choice", "CLAIM B: queue conditioning"),
             ("choice_queue", "mse_published", "A+B vs the published M0"),
             ("choice_tol", "choice", "tolerance-aware confirmations"),
             ("choice_queue_tol", "choice_queue", "tolerance-aware confirmations"),
             ("choice_queue_k64", "choice_queue", "choice set capped at K<=64")]
    head_to_head = []
    for t, c, why in pairs:
        if t not in by_v or c not in by_v:
            continue
        # SEED-MATCHED: average only over the seeds both rungs were run on.
        # Robustness rows use fewer seeds, and the choice rungs' seed spread is
        # tens of percentage points, so an unmatched contrast would compare
        # seed sets rather than estimators.
        common = sorted(set(by_v[t]["twt_aug_by_seed"]) & set(by_v[c]["twt_aug_by_seed"]),
                        key=int)
        ta = np.mean([by_v[t]["twt_aug_by_seed"][s] for s in common], axis=0)
        tb = np.mean([by_v[c]["twt_aug_by_seed"][s] for s in common], axis=0)
        con = L.contrast(ta, tb)
        con.update({"test": t, "comparator": c, "claim": why,
                    "seeds_matched": [int(x) for x in common],
                    "d_pearson": by_v[t]["pearson_r"] - by_v[c]["pearson_r"],
                    "d_sign_acc": by_v[t]["sign_acc"] - by_v[c]["sign_acc"],
                    "d_kendall": by_v[t]["kendall_tau"] - by_v[c]["kendall_tau"],
                    "d_choice_ll": by_v[t]["choice_ll"] - by_v[c]["choice_ll"]})
        head_to_head.append(con)
    if head_to_head:
        adj2 = L.holm([h["wilcoxon_p"] for h in head_to_head])
        for h, p in zip(head_to_head, adj2):
            h["wilcoxon_p_holm"] = p

    ref = per_seed[seeds[0]].get("_recorded_reference", {})
    ref_ken = np.mean([per_seed[s]["_recorded_reference"]["kendall_eval"]["kendall_tau"]
                       for s in seeds])
    ref_ll = np.mean([per_seed[s]["_recorded_reference"]["choice_ll"]["ll"]
                      for s in seeds])
    ref_unif = np.mean([per_seed[s]["_recorded_reference"]["choice_ll"]["ll_uniform"]
                        for s in seeds])

    summary = {"source": src, "seeds": [int(s) for s in seeds],
               "config": per_seed[seeds[0]]["_config"],
               "testset_counts": per_seed[seeds[0]]["_testset_counts"],
               "recorded_reference": {
                   "kendall_tau": float(ref_ken), "choice_ll": float(ref_ll),
                   "choice_ll_uniform": float(ref_unif),
                   "note": ref.get("note", "")},
               "rows": rows, "head_to_head": head_to_head}

    out_json = os.path.join(L.OUT, "%s_summary.json" % a.tag)
    with open(out_json + ".tmp", "w") as fh:
        json.dump(summary, fh, indent=1)
    os.replace(out_json + ".tmp", out_json)

    # ---------------- table ------------------------------------------------- #
    hdr = ("%-52s %5s %6s %7s %7s %7s %8s %9s %7s %6s %8s"
           % ("rung", "seeds", "params", "Pearson", "signAcc", "Kendall",
              "heldLL", "TWT*", "%<RULE", "W/T/L", "p(Holm)"))
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append("%-52s %5d %6d %6.3f%s %6.3f%s %6.3f%s %8.3f %9.1f %6.2f%% "
                     "%2d/%d/%-2d %8.4f"
                     % (r["label"], r["n_seeds"], r["n_params_total"],
                        r["pearson_r"], "", r["sign_acc"], "", r["kendall_tau"], "",
                        r["choice_ll"], r["twt_aug_mean"], r["pct_below_rule"],
                        r["wtl"]["W"], r["wtl"]["T"], r["wtl"]["L"],
                        r["wilcoxon_p_holm"]))
    lines.append("-" * len(hdr))
    lines.append("%-52s %5s %6s %6s  %6s  %6.3f  %8.3f %9.1f %6.2f%%"
                 % ("       RULE (recorded-field ATC, hat_s == 0)", "-", "-", "-",
                    "-", ref_ken, ref_ll, rows[0]["twt_rule_mean"], 0.0))
    lines.append("%-52s %5s %6s %6s  %6s  %6s  %8.3f"
                 % ("       uniform-choice floor", "-", "-", "-", "-", "-", ref_unif))
    lines.append("")
    lines.append("seeds %s; %d held-out instances; seed-averaged per-instance TWT*, "
                 "paired Wilcoxon (pratt), Holm within this family."
                 % (summary["seeds"], rows[0]["n_instances"]))
    lines.append("seed sd of %%<RULE: " + ", ".join(
        "%s %.2f" % (r["variant"], r["pct_below_rule_seed_sd"]) for r in rows))
    lines.append("")
    lines.append("HEAD-TO-HEAD (seed-averaged per-instance TWT*, paired Wilcoxon, "
                 "Holm within this family; a WIN = the test rung is LOWER)")
    lines.append("%-40s %-30s %8s %7s %7s %8s %8s %s"
                 % ("contrast", "claim", "dTWT*%", "dPear", "dKend", "W/T/L",
                    "p(Holm)", "seeds"))
    for h in head_to_head:
        lines.append("%-40s %-30s %+7.2f%% %+7.3f %+7.3f %2d/%d/%-2d %7.4f  seeds=%s"
                     % ("%s vs %s" % (h["test"], h["comparator"]), h["claim"],
                        h["pct_vs_comparator"], h["d_pearson"], h["d_kendall"],
                        h["wtl"]["W"], h["wtl"]["T"], h["wtl"]["L"],
                        h["wilcoxon_p_holm"], h["seeds_matched"]))
    table = "\n".join(lines)
    with open(os.path.join(L.OUT, "%s_table.txt" % a.tag), "w") as fh:
        fh.write(table + "\n")
    print(table)

    # ---------------- macros ------------------------------------------------ #
    rel = "results/y3_w2/%s_summary.json" % a.tag
    M = []
    A = M.append
    A("%% W2 (queue-conditioned choice-model estimator). Source: %s" % rel)
    A("% Every macro's comment names the file and the field it came from.")
    name = {"mse_published": "Msq", "mse_es": "MsqEs", "choice": "Mcl",
            "choice_queue": "Mclq", "choice_tol": "MclTol",
            "choice_queue_tol": "MclqTol", "choice_queue_k64": "MclqKsf"}
    for r in rows:
        n = name.get(r["variant"])
        if not n:
            continue
        A("\\newcommand{\\%sPearson}{%.2f}    %% %s rows[%s].pearson_r"
          % (n, r["pearson_r"], rel, r["variant"]))
        A("\\newcommand{\\%sSignAcc}{%.3f}    %% %s rows[%s].sign_acc"
          % (n, r["sign_acc"], rel, r["variant"]))
        A("\\newcommand{\\%sKendall}{%.3f}    %% %s rows[%s].kendall_tau"
          % (n, r["kendall_tau"], rel, r["variant"]))
        A("\\newcommand{\\%sHeldLL}{%.3f}     %% %s rows[%s].choice_ll"
          % (n, r["choice_ll"], rel, r["variant"]))
        A("\\newcommand{\\%sGain}{%.1f\\%%}   %% %s rows[%s].pct_below_rule"
          % (n, r["pct_below_rule"], rel, r["variant"]))
        A("\\newcommand{\\%sGainStd}{%.1f}    %% %s rows[%s].pct_below_rule_seed_sd"
          % (n, r["pct_below_rule_seed_sd"], rel, r["variant"]))
        A("\\newcommand{\\%sParams}{%d}       %% %s rows[%s].n_params_total"
          % (n, r["n_params_total"], rel, r["variant"]))
        A("\\newcommand{\\%sP}{%.3f}          %% %s rows[%s].wilcoxon_p_holm"
          % (n, r["wilcoxon_p_holm"], rel, r["variant"]))
        A("\\newcommand{\\%sShiftSD}{%.2f}     %% %s rows[%s].sd_hat_s"
          % (n, r["sd_hat_s"], rel, r["variant"]))
        gap0, gap = 1.0 - ref_ken, 1.0 - r["kendall_tau"]
        A("\\newcommand{\\%sRankGap}{%.0f\\%%}  %% %s 1-(1-rows[%s].kendall_tau)"
          "/(1-recorded_reference.kendall_tau)" % (n, 100.0 * (gap0 - gap) / gap0,
                                                   rel, r["variant"]))
    A("\\newcommand{\\RuleKendall}{%.3f}      %% %s recorded_reference.kendall_tau"
      % (ref_ken, rel))
    A("\\newcommand{\\RuleHeldLL}{%.3f}       %% %s recorded_reference.choice_ll"
      % (ref_ll, rel))
    A("\\newcommand{\\UnifHeldLL}{%.3f}       %% %s recorded_reference.choice_ll_uniform"
      % (ref_unif, rel))
    A("\\newcommand{\\WtwoSeeds}{%d}            %% %s seeds (301-%d)"
      % (len(summary["seeds"]), rel, max(summary["seeds"])))
    A("\\newcommand{\\WtwoInstances}{%d}        %% %s rows[*].n_instances"
      % (rows[0]["n_instances"], rel))
    A("\\newcommand{\\WtwoOverrides}{%d}        %% results/y3_w2/ladder.json "
      "['301']['mse_published'].n_overrides_total (overrides available to the fit)"
      % _overrides(src))
    lat_path = os.path.join(L.OUT, "latency.json")
    if os.path.exists(lat_path):
        lat = json.load(open(lat_path))["summary"]
        A("\\newcommand{\\MsqMsPerDec}{%.3f}     %% results/y3_w2/latency.json "
          "summary.ms_per_decision_per_order_best (CONTENDED box: upper bound)"
          % lat["ms_per_decision_per_order_best"])
        A("\\newcommand{\\MclqMsPerDec}{%.3f}    %% results/y3_w2/latency.json "
          "summary.ms_per_decision_queue_best (CONTENDED box: upper bound)"
          % lat["ms_per_decision_queue_best"])
        A("\\newcommand{\\MclqCostRatio}{%.1f}   %% results/y3_w2/latency.json "
          "summary.ratio_queue_over_per_order (the reportable quantity)"
          % lat["ratio_queue_over_per_order"])
    with open(os.path.join(L.OUT, "%s_macros.tex" % a.tag), "w") as fh:
        fh.write("\n".join(M) + "\n")
    print("\nwrote %s, %s_table.txt, %s_macros.tex" % (out_json, a.tag, a.tag))


if __name__ == "__main__":
    sys.exit(main())
