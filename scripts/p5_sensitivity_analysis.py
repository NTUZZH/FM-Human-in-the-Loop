#!/usr/bin/env python
"""P5 / E5 sensitivity analysis: turn results/p4_sensitivity/results.csv into the
appendix robustness report (md + LaTeX fragment).

The paper's actual question (docs/protocol.md locked defaults): *do the benchmark
conclusions survive the calibration sweeps* -- a +-50% SLA perturbation and a
+-25% capacity perturbation?  A "conclusion" here is the method LEADERBOARD (which
dispatcher beats which).  We therefore quantify robustness as **Kendall's tau
between the baseline method ranking and each perturbed condition's ranking**:
tau = 1 means the ordering is perfectly preserved (conclusions survive), tau
near 0 means the sweep reshuffles the leaderboard.

Rankings are by mean WWT (lower is better) over the 9 online methods
(edd, wspt, atc, pfifo, mor, random, rl301..rl303).

Outputs (results/p4_sensitivity/)
---------------------------------
  sensitivity_summary.md
    1. Kendall-tau robustness: per (campus x size) cell AND pooled, for each of
       the four perturbed conditions, plus a plain-language verdict (is the
       baseline-best method still best?  are all pooled tau >= 0.8?).
    2. Mean WWT per method, per condition x campus x size.
    3. SLA breach-share-by-priority SHIFT table (baseline vs sla0.5 vs sla1.5,
       per priority class P1..P4) -- how the deadline sweep moves the breach
       burden across priorities.
  tab_sensitivity.tex
    booktabs fragment: Kendall tau (baseline vs each condition) by size and
    pooled, plus the worst-cell tau -- the headline robustness table.

Pooled tau
----------
Per-cell tau is Kendall's tau-b on the two mean-WWT-per-method vectors for that
(campus,size).  The POOLED tau is scale-free: within every cell each method gets
an integer rank (1 = lowest mean WWT); a method's cross-cell AVERAGE rank forms
the baseline and condition rank vectors, and tau-b is taken between them.  This
avoids a few high-WWT cells dominating a raw-mean pool.  Cells whose baseline
means are fully tied (e.g. capacity-adequate, all-zero WWT) yield an undefined
per-cell tau (reported '-') and are dropped from the mean-cell-tau summary; they
still contribute to the pooled ranks (ties are broken across cells).

Usage
-----
    PYTHONPATH=src python scripts/p5_sensitivity_analysis.py
        [--in DIR] [--csv PATH] [--rl-tag rl] [--tau-pass 0.8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = _ROOT / "results" / "p4_sensitivity"

ALL_PDRS = ["edd", "wspt", "atc", "pfifo", "mor", "random"]
RL_SEEDS = [301, 302, 303]
RL_TAG = "rl"
RL_METHODS = ["%s%d" % (RL_TAG, s) for s in RL_SEEDS]
METHODS = ALL_PDRS + RL_METHODS

BASELINE = "baseline"
SLA_CONDS = ["sla0.5", "sla1.5"]
PERTURBED = ["sla0.5", "sla1.5", "crew0.75", "crew1.25"]
COND_LABEL = {
    "sla0.5": "SLA $\\times0.5$", "sla1.5": "SLA $\\times1.5$",
    "crew0.75": "Crew $\\times0.75$", "crew1.25": "Crew $\\times1.25$",
}
METHOD_LABEL = {"edd": "EDD", "wspt": "WSPT", "atc": "ATC", "pfifo": "pFIFO",
                "mor": "MOR", "random": "Random"}
PRIORITIES = [1, 2, 3, 4]


def _configure_rl(tag):
    global RL_TAG, RL_METHODS, METHODS
    RL_TAG = str(tag)
    RL_METHODS = ["%s%d" % (RL_TAG, s) for s in RL_SEEDS]
    METHODS = ALL_PDRS + RL_METHODS
    for s in RL_SEEDS:
        METHOD_LABEL["%s%d" % (RL_TAG, s)] = "RL (seed %d)" % s


# --------------------------------------------------------------------------- #
# Core stats
# --------------------------------------------------------------------------- #
def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return ("%%.%df" % nd) % x


def cell_means(df, campus, size, condition):
    """method -> mean WWT over the feasible rows of one (campus,size,condition)."""
    sub = df[(df["campus"] == campus) & (df["size"] == size)
             & (df["condition"] == condition) & (df["feasible"] == 1)]
    means = {}
    for m in METHODS:
        rows = sub[sub["method"] == m]
        means[m] = float(rows["wwt"].mean()) if len(rows) else np.nan
    return means


def _tau(x, y):
    """Kendall tau-b of two aligned score vectors; None if degenerate.

    kendalltau returns nan when either vector is constant (a fully-tied ranking
    carries no ordering information) or too short; we surface that as None."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return None
    t, _p = kendalltau(x, y)
    return None if (t is None or not np.isfinite(t)) else float(t)


def cell_tau(df, campus, size, condition):
    """Per-cell Kendall tau: baseline vs condition method-ranking by mean WWT."""
    bm = cell_means(df, campus, size, BASELINE)
    cm = cell_means(df, campus, size, condition)
    x = [bm[m] for m in METHODS]
    y = [cm[m] for m in METHODS]
    return _tau(x, y)


def _avg_ranks(df, cells, condition):
    """Cross-cell average rank per method (1 = lowest mean WWT), scale-free.

    Within each cell methods are ranked by mean WWT (ties -> average rank); a
    method's mean of those per-cell ranks is returned.  Cells where the method
    is absent contribute no rank for it."""
    per_method = {m: [] for m in METHODS}
    for campus, size in cells:
        means = cell_means(df, campus, size, condition)
        vals = np.array([means[m] for m in METHODS], dtype=float)
        # rank finite values (average ranks for ties); leave NaN methods out
        finite = np.isfinite(vals)
        if finite.sum() < 1:
            continue
        order = pd.Series(vals[finite]).rank(method="average").to_numpy()
        idx = [i for i, f in enumerate(finite) if f]
        for j, i in enumerate(idx):
            per_method[METHODS[i]].append(order[j])
    return {m: (float(np.mean(v)) if v else np.nan)
            for m, v in per_method.items()}


def pooled_tau(df, cells, condition):
    """Scale-free pooled tau over ``cells``: tau of the cross-cell average-rank
    vectors (baseline vs condition)."""
    br = _avg_ranks(df, cells, BASELINE)
    cr = _avg_ranks(df, cells, condition)
    return _tau([br[m] for m in METHODS], [cr[m] for m in METHODS])


def best_method(df, cells, condition):
    """The method with the lowest cross-cell average rank under ``condition``."""
    ar = _avg_ranks(df, cells, condition)
    finite = {m: r for m, r in ar.items() if np.isfinite(r)}
    if not finite:
        return None
    return min(finite, key=finite.get)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="P5/E5 sensitivity analysis.")
    ap.add_argument("--in", dest="in_dir", default=str(OUT_DIR),
                    help="results root: read <dir>/results.csv, write the "
                         "sensitivity_* outputs there (default "
                         "results/p4_sensitivity)")
    ap.add_argument("--csv", default=None,
                    help="explicit results csv (overrides <in>/results.csv)")
    ap.add_argument("--rl-tag", default="rl",
                    help="method-column prefix of the RL policy (default 'rl')")
    ap.add_argument("--tau-pass", type=float, default=0.8,
                    help="pooled-tau threshold for the 'conclusions survive' "
                         "verdict (default 0.8)")
    args = ap.parse_args(argv)

    _configure_rl(args.rl_tag)
    out_dir = Path(args.in_dir)
    csv_path = Path(args.csv) if args.csv else out_dir / "results.csv"
    if not csv_path.exists():
        sys.exit("results csv not found: %s (run scripts/p4_sensitivity.py "
                 "first)" % csv_path)
    df = pd.read_csv(csv_path)

    n_rows = len(df)
    n_infeas = int((df["feasible"] != 1).sum())
    campuses = sorted(df["campus"].unique().tolist())
    sizes = sorted(df["size"].unique().tolist())
    cells = [(c, s) for c in campuses for s in sizes
             if not df[(df["campus"] == c) & (df["size"] == s)].empty]
    conds_present = [c for c in PERTURBED if c in set(df["condition"])]

    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# E5 sensitivity summary — SLA & capacity robustness")
    lines.append("")
    lines.append("Source: `%s` (%d rows; %d infeasible rows excluded from all "
                 "statistics)." % (csv_path.name, n_rows, n_infeas))
    lines.append("Base set: campuses %s x sizes %s (first-30 replay-test each); "
                 "methods %s." % (campuses, sizes, ", ".join(METHODS)))
    lines.append("Question (Appendix B): do the method-ranking conclusions "
                 "survive a +-50% SLA sweep and a +-25% capacity sweep?  "
                 "Robustness = Kendall's tau-b between the baseline method "
                 "ranking (by mean WWT) and each perturbed condition's ranking.")
    lines.append("")

    # ---- 1. Kendall-tau robustness -----------------------------------------
    lines.append("## 1. Ranking robustness (Kendall tau, baseline vs condition)")
    lines.append("")
    lines.append("Per-cell tau on the mean-WWT-per-method vectors; POOLED tau is "
                 "scale-free (tau of cross-cell average-rank vectors). "
                 "'-' = degenerate cell (baseline WWT fully tied, e.g. "
                 "capacity-adequate).")
    lines.append("")
    header = "| condition | " + " | ".join("c%d/%d" % (c, s) for c, s in cells) \
             + " | mean cell | pooled |"
    lines.append(header)
    lines.append("|" + "---|" * (len(cells) + 3))
    pooled = {}          # condition -> pooled tau over all cells
    pooled_by_size = {}  # (condition,size) -> pooled tau
    for cond in conds_present:
        cell_taus = [cell_tau(df, c, s, cond) for c, s in cells]
        finite = [t for t in cell_taus if t is not None]
        mean_cell = float(np.mean(finite)) if finite else None
        ptau = pooled_tau(df, cells, cond)
        pooled[cond] = ptau
        for s in sizes:
            scells = [(c, ss) for c, ss in cells if ss == s]
            pooled_by_size[(cond, s)] = pooled_tau(df, scells, cond)
        row = "| %s | %s | %s | %s |" % (
            cond, " | ".join(_fmt(t) for t in cell_taus),
            _fmt(mean_cell), _fmt(ptau))
        lines.append(row)
    lines.append("")

    # verdict
    base_best = best_method(df, cells, BASELINE)
    lines.append("### Verdict — do conclusions survive?")
    lines.append("")
    lines.append("Baseline best-ranked method (lowest pooled average rank): "
                 "**%s**." % (base_best or "n/a"))
    all_pass = True
    for cond in conds_present:
        bm = best_method(df, cells, cond)
        ptau = pooled[cond]
        survives = (ptau is not None and ptau >= args.tau_pass)
        all_pass = all_pass and survives and (bm == base_best)
        lines.append("* **%s**: pooled tau = %s (%s threshold %.2f); best "
                     "method now **%s** (%s). %s"
                     % (cond, _fmt(ptau),
                        ">=" if survives else "<", args.tau_pass, bm or "n/a",
                        "unchanged" if bm == base_best else "CHANGED",
                        "ranking preserved" if survives
                        else "ranking perturbed"))
    lines.append("")
    lines.append("**Overall: conclusions %s the calibration sweeps** "
                 "(all pooled tau >= %.2f and the best method is unchanged: %s)."
                 % ("SURVIVE" if all_pass else "are SENSITIVE to",
                    args.tau_pass, all_pass))
    lines.append("")

    # ---- 2. Mean WWT per method, per condition x campus x size -------------
    lines.append("## 2. Mean WWT per method (condition x campus x size)")
    lines.append("")
    conds_all = [BASELINE] + conds_present
    for cond in conds_all:
        lines.append("### Condition: %s" % cond)
        lines.append("")
        lines.append("| campus | size | n | " + " | ".join(METHODS) + " |")
        lines.append("|" + "---|" * (3 + len(METHODS)))
        for c, s in cells:
            means = cell_means(df, c, s, cond)
            sub = df[(df["campus"] == c) & (df["size"] == s)
                     & (df["condition"] == cond) & (df["method"] == "edd")
                     & (df["feasible"] == 1)]
            n = int(len(sub))
            lines.append("| %d | %d | %d | %s |"
                         % (c, s, n,
                            " | ".join(_fmt(means[m]) for m in METHODS)))
        lines.append("")

    # ---- 3. SLA breach-share-by-priority shift -----------------------------
    lines.append("## 3. SLA breach-share shift by priority (averaged over "
                 "methods & instances)")
    lines.append("")
    sla_present = [c for c in SLA_CONDS if c in set(df["condition"])]
    if sla_present:
        lines.append("Mean per-priority breach share over all methods and "
                     "feasible instances; delta vs baseline in parentheses. "
                     "Tighter SLA (x0.5) shrinks windows -> more breaches; "
                     "looser (x1.5) -> fewer.")
        lines.append("")
        cols = ["baseline"] + sla_present
        lines.append("| priority | " + " | ".join(
            ("%s" % c if c == "baseline" else "%s (delta)" % c)
            for c in cols) + " |")
        lines.append("|" + "---|" * (1 + len(cols)))

        def prio_breach(cond, p):
            col = "breach_p%d" % p
            sub = df[(df["condition"] == cond) & (df["feasible"] == 1)]
            v = pd.to_numeric(sub[col], errors="coerce")
            return float(v.mean()) if v.notna().any() else None

        for p in PRIORITIES:
            base_v = prio_breach(BASELINE, p)
            cells_txt = [_fmt(base_v, 3)]
            for cond in sla_present:
                v = prio_breach(cond, p)
                if v is None or base_v is None:
                    cells_txt.append("-")
                else:
                    cells_txt.append("%s (%+.3f)" % (_fmt(v, 3), v - base_v))
            lines.append("| P%d | %s |" % (p, " | ".join(cells_txt)))
        lines.append("")
        lines.append("(P1 emergency ... P4 planned/PM; weights 8/4/2/1.)")
    else:
        lines.append("(no SLA conditions present in the csv)")
    lines.append("")

    md_path = out_dir / "sensitivity_summary.md"
    md_path.write_text("\n".join(lines))

    # ---- LaTeX fragment: headline robustness table -------------------------
    tex = []
    tex.append("% E5 ranking robustness: Kendall tau-b between the baseline "
               "method ranking (by mean WWT) and each perturbed condition.")
    tex.append("% Generated by scripts/p5_sensitivity_analysis.py from "
               "results/p4_sensitivity/results.csv.")
    tex.append("% Pooled columns use scale-free cross-cell average ranks; "
               "'worst' is the minimum finite per-cell tau.")
    tex.append("\\begin{tabular}{lcccc}")
    tex.append("\\toprule")
    tex.append("Condition & $\\tau$ (150) & $\\tau$ (400) & "
               "$\\tau$ (pooled) & worst cell \\\\")
    tex.append("\\midrule")
    for cond in conds_present:
        worst = None
        cts = [cell_tau(df, c, s, cond) for c, s in cells]
        finite = [t for t in cts if t is not None]
        if finite:
            worst = min(finite)
        t150 = pooled_by_size.get((cond, 150))
        t400 = pooled_by_size.get((cond, 400))
        tex.append("%s & %s & %s & %s & %s \\\\"
                   % (COND_LABEL.get(cond, cond), _fmt(t150), _fmt(t400),
                      _fmt(pooled[cond]), _fmt(worst)))
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex_path = out_dir / "tab_sensitivity.tex"
    tex_path.write_text("\n".join(tex) + "\n")

    print("Wrote %s" % md_path)
    print("Wrote %s" % tex_path)
    print("Baseline best method: %s" % (base_best or "n/a"))
    for cond in conds_present:
        print("  %-9s pooled tau = %s  (best -> %s)"
              % (cond, _fmt(pooled[cond]), best_method(df, cells, cond)))


if __name__ == "__main__":
    main()
