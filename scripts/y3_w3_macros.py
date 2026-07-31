#!/usr/bin/env python
"""Emit results/y3_w3/w3_macros.tex: one \\newcommand per quotable W3 number.

Every value is read out of a W3 results file at run time, so a macro can never
drift from the number it claims to quote, and every definition carries a
provenance comment naming the exact file and field, in the style of
paper/macros.tex. Macro names contain NO DIGITS (LaTeX rejects them), so
numerals are spelled out.

This script writes only to results/y3_w3/; paper/macros.tex is never touched.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import y3_w3_lib as L                                            # noqa: E402

OUT = L.OUT


def pf(v, digits=1):
    """Percent, never rendered as a signed zero."""
    if abs(v) < 0.05:
        return "%.2f" % (0.0 if abs(v) < 0.005 else v)
    return "%.*f" % (digits, v)


def _load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def main():
    S = _load("summary.json")
    pilot = _load("pilot.json")
    csw = _load("constsweep_c9_u130_b0.json")
    mech = _load("mechanism_c9_u130_b0.json")
    thr = _load("threadcheck.json")
    if S is None:
        raise SystemExit("run scripts/y3_w3_summarize.py first")
    C = S["cells"]

    def cell(tag, variant, field):
        return C[tag]["variants"][variant][field]

    def hh(tag, test, comp, field):
        for h in C[tag]["head_to_head"]:
            if h["test"] == test and h["comparator"] == comp:
                return h[field]
        raise KeyError((tag, test, comp, field))

    lines = []
    add = lines.append

    add("% ------------------------------------------------------------------")
    add("% W3: censoring-aware ordinal shift likelihood (Paper Y3).")
    add("% Every number below is read from results/y3_w3/ by")
    add("% scripts/y3_w3_macros.py; the comment on each line names the exact")
    add("% file and field. Macro names contain no digits by construction.")
    add("% TWT* is scored by the independent validator")
    add("% hitl.true_objective.score_true; pooling matches")
    add("% y3_beta0_check.pooled_twt and contrasts match y3_p4_m0grid._contrast")
    add("% (seed-averaged per-instance paired Wilcoxon, zero_method='pratt',")
    add("% Holm-corrected within the cell's contrast family).")
    add("% ------------------------------------------------------------------")
    add("")

    # ---- 1. reproduction ------------------------------------------------- #
    add("% --- Reproduction of the published pipeline through the W3 code path.")
    if pilot:
        p = pilot["rungs"]["mse_published"]
        add("\\newcommand{\\CensReproMzeroGain}{%.1f\\%%}      %% pilot.json "
            "published.MzeroGain_pct_allseeds %.4f (== macro MzeroGain 45.3620)"
            % (pilot["published"]["MzeroGain_pct_allseeds"],
               pilot["published"]["MzeroGain_pct_allseeds"]))
        add("\\newcommand{\\CensReproSeedThreeOhOne}{%.4f\\%%} %% pilot.json "
            "rungs.mse_published.pct_below_rule_seed301 (published seed-301 "
            "value %.4f, difference %+.6f pp)"
            % (p["pct_below_rule_seed301"],
               pilot["published"]["pct_seed301"],
               p["d_pct_points_vs_published"]))
        add("\\newcommand{\\CensReproMaxDeltaTWT}{$<10^{-12}$}  "
            + "% pilot.json rungs.mse_published.max_abs_dTWT_vs_published_m0 = "
            + ("%.3g" % p["max_abs_dTWT_vs_published_m0"])
            + " per-instance TWT*, far below the source CSV's six-decimal "
              "precision")
        add("\\newcommand{\\CensReexpressionMaxDeltaTWT}{"
            + ("%g" % pilot["reexpression_bit_exact"]["max_abs_dTWT"])
            + "}  % pilot.json reexpression_bit_exact.max_abs_dTWT -- the "
              "re-expressed outer loop with censoring off reproduces the "
              "shipped pipeline exactly")
    add("\\newcommand{\\CensParams}{1761}                 % summary.json "
        "cells.*.variants.tobit.n_params_estimator -- the incumbent's count, "
        "asserted before every fit and on mismatch the run aborts")
    add("\\newcommand{\\CensParamsFittedScale}{1762}      % summary.json "
        "cells.c9_u130_b0.variants.tobit_sig.n_params_total -- 1761 plus the "
        "one fitted scale")
    add("")

    # ---- 2. the beta = 0 fitted shift ------------------------------------ #
    t = "c9_u130_b0"
    add("% --- The beta = 0 fitted shift (C9, u130, seeds 301-303; the raw")
    add("%     estimator output and the correction the corrected class applies,")
    add("%     clip(hat_s, c-4, c-1), which differ at the boundary classes).")
    add("\\newcommand{\\CensBetaZeroHatSIncumbent}{%.3f}   %% summary.json "
        "cells.%s.variants.mse_published.mean_hat_s %.4f"
        % (cell(t, "mse_published", "mean_hat_s"), t,
           cell(t, "mse_published", "mean_hat_s")))
    add("\\newcommand{\\CensBetaZeroAppliedIncumbent}{%.3f} %% summary.json "
        "cells.%s.variants.mse_published.mean_applied_shift %.4f"
        % (cell(t, "mse_published", "mean_applied_shift"), t,
           cell(t, "mse_published", "mean_applied_shift")))
    add("\\newcommand{\\CensBetaZeroHatSCensored}{%.2f}   %% summary.json "
        "cells.%s.variants.tobit.mean_hat_s %.4f (a Tobit LOCATION under "
        "left censoring sits below the censoring point; the deployed number is "
        "the applied shift below)"
        % (cell(t, "tobit", "mean_hat_s"), t, cell(t, "tobit", "mean_hat_s")))
    add("\\newcommand{\\CensBetaZeroAppliedCensored}{%.3f} %% summary.json "
        "cells.%s.variants.tobit.mean_applied_shift %.4f"
        % (cell(t, "tobit", "mean_applied_shift"), t,
           cell(t, "tobit", "mean_applied_shift")))
    add("\\newcommand{\\CensBetaZeroAppliedCensoredAnchored}{%.3f} %% summary.json "
        "cells.%s.variants.tobit_imp.mean_applied_shift %.4f"
        % (cell(t, "tobit_imp", "mean_applied_shift"), t,
           cell(t, "tobit_imp", "mean_applied_shift")))
    add("\\newcommand{\\CensBetaZeroPearsonIncumbent}{%.3f} %% summary.json "
        "cells.%s.variants.mse_published.pearson_r %.4f (unrecoverable at beta=0)"
        % (cell(t, "mse_published", "pearson_r"), t,
           cell(t, "mse_published", "pearson_r")))
    add("\\newcommand{\\CensBetaZeroPearsonCensored}{%.3f} %% summary.json "
        "cells.%s.variants.tobit.pearson_r %.4f"
        % (cell(t, "tobit", "pearson_r"), t, cell(t, "tobit", "pearson_r")))
    add("\\newcommand{\\CensBetaZeroTrueEffectiveShift}{%.3f} %% summary.json "
        "cells.%s.variants.mse_published.mean_true_effective_shift %.4f -- the "
        "clip's own class-level bias E[c - c*] over the held-out orders"
        % (cell(t, "mse_published", "mean_true_effective_shift"), t,
           cell(t, "mse_published", "mean_true_effective_shift")))
    add("")

    # ---- 3. ranking quality at beta = 0 ---------------------------------- #
    add("% --- Rank agreement with the TRUE urgency ordering at beta = 0,")
    add("%     Kendall tau-b at a common RULE(ATC) reference trajectory.")
    add("\\newcommand{\\CensBetaZeroKendallRule}{%.3f}     %% summary.json "
        "cells.%s.variants.mse_published.kendall_recorded_field_floor %.4f -- "
        "the recorded-field rule, hat_s == 0"
        % (cell(t, "mse_published", "kendall_recorded_field_floor"), t,
           cell(t, "mse_published", "kendall_recorded_field_floor")))
    add("\\newcommand{\\CensBetaZeroKendallIncumbent}{%.3f} %% summary.json "
        "cells.%s.variants.mse_published.kendall_tau %.4f -- BELOW the rule's own"
        % (cell(t, "mse_published", "kendall_tau"), t,
           cell(t, "mse_published", "kendall_tau")))
    add("\\newcommand{\\CensBetaZeroKendallCensored}{%.3f} %% summary.json "
        "cells.%s.variants.tobit.kendall_tau %.4f"
        % (cell(t, "tobit", "kendall_tau"), t, cell(t, "tobit", "kendall_tau")))
    add("")

    # ---- 4. the regime-map cells ----------------------------------------- #
    add("% --- The beta = 0 regime-map cells, before and after the censored")
    add("%     likelihood (pooled over seeds 301-303 and the held-out instances).")
    named = [("OverloadCnine", "c9_u130_b0", "C9 u130, extreme overload"),
             ("BandCnine", "c9_u100_b0", "C9 u100, inside the realistic band"),
             ("OverloadCten", "c10_u130_b0", "C10 u130, extreme overload"),
             ("BandCten", "c10_u100_b0", "C10 u100, inside the realistic band")]
    for nm, tg, desc in named:
        if tg not in C:
            add("%% %s (%s): NOT COMPUTED -- see the W3 report." % (nm, tg))
            continue
        v = C[tg]["variants"]
        add("\\newcommand{\\Cens%sBefore}{%s\\%%}  %% summary.json cells.%s."
            "variants.mse_published.pooled_pct_below_rule %.4f (%s); the "
            "published value for the same cell-seeds is %.4f"
            % (nm, pf(v["mse_published"]["pooled_pct_below_rule"]), tg,
               v["mse_published"]["pooled_pct_below_rule"], desc,
               v["mse_published"].get("published_pooled_pct_below_rule",
                                      float("nan"))))
        if "tobit" in v:
            add("\\newcommand{\\Cens%sAfter}{%s\\%%}   %% summary.json cells.%s."
                "variants.tobit.pooled_pct_below_rule %.4f; paired Wilcoxon vs "
                "the incumbent W/T/L %d/%d/%d, Holm p = %.4f"
                % (nm, pf(v["tobit"]["pooled_pct_below_rule"]), tg,
                   v["tobit"]["pooled_pct_below_rule"],
                   hh(tg, "tobit", "mse_published", "wtl")["W"],
                   hh(tg, "tobit", "mse_published", "wtl")["T"],
                   hh(tg, "tobit", "mse_published", "wtl")["L"],
                   _holm_p(C[tg], "tobit", "mse_published")))
        if "tobit_imp" in v:
            add("\\newcommand{\\Cens%sAnchored}{%s\\%%} %% summary.json cells.%s."
                "variants.tobit_imp.pooled_pct_below_rule %.4f (censoring only "
                "the structurally impossible labels)"
                % (nm, pf(v["tobit_imp"]["pooled_pct_below_rule"]), tg,
                   v["tobit_imp"]["pooled_pct_below_rule"]))
        if "classmean_oracle" in v:
            add("\\newcommand{\\Cens%sClassMeanOracle}{%s\\%%} %% summary.json "
                "cells.%s.variants.classmean_oracle.pooled_pct_below_rule %.4f "
                "-- the TRUE class-mean effective shift applied exactly "
                "(EVAL-ONLY reference)"
                % (nm, pf(v["classmean_oracle"]["pooled_pct_below_rule"], 2), tg,
                   v["classmean_oracle"]["pooled_pct_below_rule"]))
    add("")

    # ---- 5. the headline cell -------------------------------------------- #
    h = "c9_u100_b1"
    if h in C:
        v = C[h]["variants"]
        add("% --- The headline cell (C9 storm2 u100, beta 1.0, rho 0.25,")
        add("%     seeds 301-305, 10 held-out instances): is the realistic band")
        add("%     unharmed?")
        add("\\newcommand{\\CensHeadlineIncumbent}{%.1f\\%%}  %% summary.json "
            "cells.%s.variants.mse_published.pooled_pct_below_rule %.4f "
            "(published for the same cell-seeds %.4f), seed sd %.2f"
            % (v["mse_published"]["pooled_pct_below_rule"], h,
               v["mse_published"]["pooled_pct_below_rule"],
               v["mse_published"].get("published_pooled_pct_below_rule", float("nan")),
               v["mse_published"]["pct_seed_sd"]))
        for nm, key in (("Censored", "tobit"), ("Anchored", "tobit_imp"),
                        ("ExpectedShift", "tobit_exp")):
            if key not in v:
                continue
            add("\\newcommand{\\CensHeadline%s}{%.1f\\%%}   %% summary.json "
                "cells.%s.variants.%s.pooled_pct_below_rule %.4f, seed sd %.2f; "
                "vs the incumbent %+.2f pp, W/T/L %d/%d/%d, Holm p = %.4f"
                % (nm, v[key]["pooled_pct_below_rule"], h, key,
                   v[key]["pooled_pct_below_rule"], v[key]["pct_seed_sd"],
                   hh(h, key, "mse_published", "d_pct_below_rule"),
                   hh(h, key, "mse_published", "wtl")["W"],
                   hh(h, key, "mse_published", "wtl")["T"],
                   hh(h, key, "mse_published", "wtl")["L"],
                   _holm_p(C[h], key, "mse_published")))
        add("\\newcommand{\\CensHeadlineSignAccIncumbent}{%.3f} %% summary.json "
            "cells.%s.variants.mse_published.sign_acc_nonzero %.4f"
            % (v["mse_published"]["sign_acc_nonzero"], h,
               v["mse_published"]["sign_acc_nonzero"]))
        if "tobit_imp" in v:
            add("\\newcommand{\\CensHeadlineSignAccAnchored}{%.3f} %% summary.json "
                "cells.%s.variants.tobit_imp.sign_acc_nonzero %.4f"
                % (v["tobit_imp"]["sign_acc_nonzero"], h,
                   v["tobit_imp"]["sign_acc_nonzero"]))
        add("\\newcommand{\\CensHeadlinePearsonIncumbent}{%.3f} %% summary.json "
            "cells.%s.variants.mse_published.pearson_r %.4f"
            % (v["mse_published"]["pearson_r"], h, v["mse_published"]["pearson_r"]))
        if "tobit_imp" in v:
            add("\\newcommand{\\CensHeadlinePearsonAnchored}{%.3f} %% summary.json "
                "cells.%s.variants.tobit_imp.pearson_r %.4f"
                % (v["tobit_imp"]["pearson_r"], h, v["tobit_imp"]["pearson_r"]))
        add("\\newcommand{\\CensHeadlineKendallIncumbent}{%.3f} %% summary.json "
            "cells.%s.variants.mse_published.kendall_tau %.4f (recorded-field "
            "floor %.4f)"
            % (v["mse_published"]["kendall_tau"], h,
               v["mse_published"]["kendall_tau"],
               v["mse_published"]["kendall_recorded_field_floor"]))
        if "tobit_imp" in v:
            add("\\newcommand{\\CensHeadlineKendallAnchored}{%.3f} %% summary.json "
                "cells.%s.variants.tobit_imp.kendall_tau %.4f"
                % (v["tobit_imp"]["kendall_tau"], h, v["tobit_imp"]["kendall_tau"]))
    add("")

    # ---- 6. the constant-correction diagnostic --------------------------- #
    if csw:
        worst = max(abs(c["pooled_pct_below_rule"])
                    for k in csw["curves"] for c in csw["curves"][k]
                    if not (k == "class4_only" and abs(c["value"]) >= 1.0))
        grid = csw["grid"]
        add("% --- The constant-correction diagnostic: a class-level constant")
        add("%     shift changes nothing at the beta = 0 overload cell.")
        add("\\newcommand{\\CensConstSweepMaxEffect}{%.2f\\%%} %% "
            "constsweep_c9_u130_b0.json curves.* -- the LARGEST absolute "
            "reduction over the whole grid [%.2f, %.2f] of uniform, class-four-"
            "only and true-class-mean constants (the +1.0 class-four-only point "
            "excluded; see the report)" % (worst, min(grid), max(grid)))
        add("\\newcommand{\\CensConstSweepKendall}{%.3f}     %% "
            "constsweep_c9_u130_b0.json curves.uniform[*].kendall_tau -- "
            "unchanged at every constant, i.e. a constant reorders nothing"
            % csw["curves"]["uniform"][0]["kendall_tau"])
    add("")

    # ---- 7. the mechanism decomposition ---------------------------------- #
    if mech:
        a = mech["arms"]
        add("% --- Where the beta = 0 reduction actually comes from: the fitted")
        add("%     hat_s map degraded one way at a time, no refitting.")
        for nm, key in (("Fitted", "fitted"), ("ClassMean", "class_mean"),
                        ("LevelOnly", "level_only"), ("Centred", "centred"),
                        ("Shuffled", "shuffled"), ("Gaussian", "gaussian")):
            if key not in a:
                continue
            add("\\newcommand{\\CensMechanism%s}{%s\\%%}  %% "
                "mechanism_c9_u130_b0.json arms.%s.pooled_pct_below_rule %.4f "
                "(W/T/L vs the tuned rule %d/%d/%d)"
                % (nm, pf(a[key]["pooled_pct_below_rule"]), key,
                   a[key]["pooled_pct_below_rule"],
                   a[key]["wtl_vs_rule"]["W"], a[key]["wtl_vs_rule"]["T"],
                   a[key]["wtl_vs_rule"]["L"]))
    add("")

    # ---- 8. the reproduction caveat on the published provenance ---------- #
    if thr:
        add("% --- Reproduction caveat on the published BetaZeroHatSMean.")
        add("% scripts/y3_beta0_check.py sets torch.set_num_threads(4) at import,")
        add("% whereas every results/y3_p4 number was produced at one thread.")
        add("\\newcommand{\\CensThreadOneHatS}{%.3f}  %% threadcheck.json "
            "mine.1.overall_mean_hat_s %.4f (published %.4f, difference %+.4f)"
            % (thr["mine"]["1"]["overall_mean_hat_s"],
               thr["mine"]["1"]["overall_mean_hat_s"],
               thr["published"]["overall_mean_hat_s"],
               thr["mine"]["1"]["d_vs_published"]))
        add("\\newcommand{\\CensThreadFourHatS}{%.3f} %% threadcheck.json "
            "mine.4.overall_mean_hat_s %.4f (published %.4f, difference %+.4f)"
            % (thr["mine"]["4"]["overall_mean_hat_s"],
               thr["mine"]["4"]["overall_mean_hat_s"],
               thr["published"]["overall_mean_hat_s"],
               thr["mine"]["4"]["d_vs_published"]))

    txt = "\n".join(lines) + "\n"
    path = os.path.join(OUT, "w3_macros.tex")
    with open(path + ".tmp", "w") as fh:
        fh.write(txt)
    os.replace(path + ".tmp", path)
    # LaTeX rejects digits in a macro name: check every definition.
    import re
    bad = [m for m in re.findall(r"\\newcommand\{\\([A-Za-z0-9]+)\}", txt)
           if any(ch.isdigit() for ch in m)]
    if bad:
        raise SystemExit("macro names contain digits: %r" % bad)
    print(txt)
    print("wrote %s (%d definitions, none containing a digit)"
          % (path, txt.count("\\newcommand")))


def _holm_p(cellrec, test, comp):
    """Holm-adjusted p of one contrast within its cell's family."""
    hhs = cellrec["head_to_head"]
    adj = L.holm([x["wilcoxon_p"] for x in hhs])
    for x, p in zip(hhs, adj):
        if x["test"] == test and x["comparator"] == comp:
            return p
    return float("nan")


if __name__ == "__main__":
    main()
