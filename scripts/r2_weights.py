#!/usr/bin/env python
"""R2 / A4 -- priority-weight-vector sweep (robustness check: sweep the
objective weight vector w=(8,4,2,1)).

We re-simulate the SIX dispatching rules on the E5 base set under three tardiness
weight vectors mapped by priority class (P1,P2,P3,P4):

    baseline  (8, 4, 2, 1)   -- the paper's weights (SLA-tier doubling)
    flat      (4, 3, 2, 1)   -- near-linear, compresses the priority spread
    steep     (27, 9, 3, 1)  -- geometric x3, widens the priority spread

The weight vector is injected into BOTH the rule scoring AND the objective:
WSPT and ATC score with w/p, so their DISPATCH DECISIONS change with the vector
(EDD/pFIFO/MOR/random ignore weight, but their relative standing in the
weighted-tardiness objective still moves because the objective is reweighted).
due_bh (SLA per priority) is untouched -- only weight_j changes.

Every schedule has travel=0 (end-start==p_bh) so it is FEASIBLE and scored by the
independent validator (WWT). Consistency check: the (8,4,2,1) rerun reproduces the
existing results/p4_sensitivity baseline mean TWT for cell (campus 5, size 150),
per method, to < 1e-6.

Base set: campus {5,9,10,12} x size {150,400} x first-30 replay-test = 240
instances (mirror of scripts/p4_sensitivity.py).

Outputs (results/r2_sens/)
--------------------------
  weights.csv        id, campus, size, method, wvec, w1, w2, w3, w4, twt
  weights_summary.md Kendall tau-b (baseline vs each vector) per cell + pooled;
                     baseline-best vs swept-best method.
  tab_weights.tex    headline tau table (tab_sensitivity.tex layout).
"""

from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from fmwos import pdrs                       # noqa: E402
from fmwos.validator import validate          # noqa: E402

INST_ROOT = _ROOT / "data" / "processed" / "instances"
INDEX_CSV = INST_ROOT / "index.csv"
OUT_DIR = _ROOT / "results" / "r2_sens"
P4_BASELINE_CSV = _ROOT / "results" / "p4_sensitivity" / "results.csv"

CAMPUSES = [5, 9, 10, 12]
SIZES = [150, 400]
N_BASE = 30
RULES = ["edd", "wspt", "atc", "pfifo", "mor", "random"]
SEED = 301

# (label, (w_P1, w_P2, w_P3, w_P4)). "baseline" MUST be first (it is the
# reference ranking every tau is taken against).
WVECS = [
    ("baseline", (8.0, 4.0, 2.0, 1.0)),
    ("flat", (4.0, 3.0, 2.0, 1.0)),
    ("steep", (27.0, 9.0, 3.0, 1.0)),
]
WLABELS = [w[0] for w in WVECS]


def _base_rows():
    with open(INDEX_CSV, newline="") as f:
        idx = list(csv.DictReader(f))
    rows = []
    for campus in CAMPUSES:
        for size in SIZES:
            cell = [r for r in idx
                    if str(r.get("split", "")).strip().lower() == "test"
                    and str(r.get("track", "")).strip().lower() == "replay"
                    and int(r["campus"]) == campus
                    and int(r["size_class"]) == size]
            cell.sort(key=lambda r: r["id"])
            rows.extend(cell[:N_BASE])
    return rows


def _reweight(instance, wvec):
    """Deep-copy ``instance`` with every WO weight remapped to wvec[priority-1].

    priority is 1..4; a WO with an out-of-range priority keeps its weight
    (defensive; the calibrated instances only ever carry 1..4)."""
    inst = copy.deepcopy(instance)
    for wo in inst["work_orders"]:
        p = int(wo.get("priority", 0))
        if 1 <= p <= 4:
            wo["weight"] = float(wvec[p - 1])
    return inst


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _base_rows()
    print("A4 weights: %d base instances x %d rules x %d weight vectors"
          % (len(base), len(RULES), len(WVECS)))

    rows = []
    guard_mismatch = []
    for r in base:
        with open(INST_ROOT / r["path"]) as f:
            inst = json.load(f)
        campus = int(r["campus"])
        size = int(r["size_class"])
        for label, wvec in WVECS:
            winst = _reweight(inst, wvec)
            # Guard: the baseline vector must not change any weight vs the
            # on-disk instance (the paper's weights ARE (8,4,2,1)).
            if label == "baseline":
                for a, b in zip(inst["work_orders"], winst["work_orders"]):
                    if abs(float(a["weight"]) - float(b["weight"])) > 1e-9:
                        guard_mismatch.append((inst["meta"]["id"], a["id"]))
            for rule in RULES:
                sched = pdrs.dispatch(winst, rule, seed=SEED)
                res = validate(winst, sched)
                if not res["feasible"]:
                    print("[INFEASIBLE] %s %s %s" % (inst["meta"]["id"], label, rule))
                rows.append({
                    "id": inst["meta"]["id"], "campus": campus, "size": size,
                    "method": rule, "wvec": label,
                    "w1": wvec[0], "w2": wvec[1], "w3": wvec[2], "w4": wvec[3],
                    "twt": res["metrics"]["WWT"],
                })

    out_csv = OUT_DIR / "weights.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "campus", "size", "method",
                                          "wvec", "w1", "w2", "w3", "w4", "twt"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("Wrote %s (%d rows)" % (out_csv, len(rows)))
    print("baseline-weight identity guard: %d mismatched WO(s)"
          % len(guard_mismatch))

    ok = _regression_guard(rows)
    analyse(rows)
    return len(guard_mismatch), ok


def _regression_guard(rows):
    """Assert the baseline-vector rerun reproduces the p4_sensitivity baseline
    mean TWT for cell (campus 5, size 150), per method (< 1e-6)."""
    if not P4_BASELINE_CSV.exists():
        print("  [guard] p4_sensitivity results.csv absent -- skipping "
              "cross-check")
        return None
    df = pd.read_csv(P4_BASELINE_CSV)
    sub = df[(df["condition"] == "baseline") & (df["campus"] == 5)
             & (df["size"] == 150) & (df["feasible"] == 1)]
    ok = True
    print("  [guard] baseline rerun vs p4_sensitivity (campus 5, size 150):")
    for m in RULES:
        ref = sub[sub["method"] == m]["wwt"]
        mine = [r["twt"] for r in rows if r["wvec"] == "baseline"
                and r["campus"] == 5 and r["size"] == 150 and r["method"] == m]
        ref_mean = float(ref.mean()) if len(ref) else float("nan")
        my_mean = float(np.mean(mine)) if mine else float("nan")
        d = abs(ref_mean - my_mean)
        flag = "OK" if d < 1e-6 else "MISMATCH"
        if d >= 1e-6:
            ok = False
        print("    %-7s ref=%.6f mine=%.6f |d|=%.2e %s"
              % (m, ref_mean, my_mean, d, flag))
    return ok


# --------------------------------------------------------------------------- #
# Analysis (mirror of p5_sensitivity_analysis)
# --------------------------------------------------------------------------- #
def _tau(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return None
    t, _ = kendalltau(x, y)
    return None if (t is None or not np.isfinite(t)) else float(t)


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return ("%%.%df" % nd) % x


def _cell_means(rows, campus, size, wvec):
    return {m: (float(np.mean([r["twt"] for r in rows if r["campus"] == campus
                               and r["size"] == size and r["method"] == m
                               and r["wvec"] == wvec]))
                if any(r["campus"] == campus and r["size"] == size
                       and r["method"] == m and r["wvec"] == wvec for r in rows)
                else np.nan)
            for m in RULES}


def _avg_ranks(rows, cells, wvec):
    per = {m: [] for m in RULES}
    for c, s in cells:
        means = _cell_means(rows, c, s, wvec)
        vals = np.array([means[m] for m in RULES], float)
        finite = np.isfinite(vals)
        if finite.sum() < 1:
            continue
        order = pd.Series(vals[finite]).rank(method="average").to_numpy()
        idx = [i for i, f in enumerate(finite) if f]
        for j, i in enumerate(idx):
            per[RULES[i]].append(order[j])
    return {m: (float(np.mean(v)) if v else np.nan) for m, v in per.items()}


def _pooled_tau(rows, cells, wvec):
    br = _avg_ranks(rows, cells, "baseline")
    cr = _avg_ranks(rows, cells, wvec)
    return _tau([br[m] for m in RULES], [cr[m] for m in RULES])


def _best(rows, cells, wvec):
    ar = _avg_ranks(rows, cells, wvec)
    finite = {m: r for m, r in ar.items() if np.isfinite(r)}
    return min(finite, key=finite.get) if finite else None


def analyse(rows):
    cells = [(c, s) for c in CAMPUSES for s in SIZES]
    perturbed = [w for w in WLABELS if w != "baseline"]

    per_cell = {}
    mean_cell = {}
    pooled = {}
    pooled_by_size = {}
    for wl in perturbed:
        per_cell[wl] = {(c, s): _tau(
            [_cell_means(rows, c, s, "baseline")[m] for m in RULES],
            [_cell_means(rows, c, s, wl)[m] for m in RULES]) for c, s in cells}
        finite = [t for t in per_cell[wl].values() if t is not None]
        mean_cell[wl] = float(np.mean(finite)) if finite else None
        pooled[wl] = _pooled_tau(rows, cells, wl)
        for s in SIZES:
            pooled_by_size[(wl, s)] = _pooled_tau(
                rows, [(c, s) for c in CAMPUSES], wl)

    base_best = _best(rows, cells, "baseline")

    L = []
    L.append("# A4 priority-weight-vector sweep summary")
    L.append("")
    L.append("Source: `weights.csv`. Six PDRs on the E5 base set (campus "
             "{5,9,10,12} x size {150,400} x first-30 replay-test = 240 "
             "instances). Tardiness weights mapped by priority (P1,P2,P3,P4): "
             "baseline (8,4,2,1), flat (4,3,2,1), steep (27,9,3,1). The vector "
             "enters both the WSPT/ATC scores (dispatch decisions change) and the "
             "objective. All schedules travel=0 -> feasible -> scored by the "
             "independent validator.")
    L.append("")
    L.append("## Ranking robustness (Kendall tau-b, baseline weights vs sweep)")
    L.append("")
    L.append("Per-cell tau on the mean-TWT-per-method vectors; '-' = degenerate "
             "cell (baseline ranking fully tied, capacity-adequate campus 5). "
             "Pooled tau is the scale-free cross-cell average-rank tau "
             "(tab_sensitivity.tex methodology).")
    L.append("")
    L.append("| weights | " + " | ".join("c%d/%d" % (c, s) for c, s in cells)
             + " | mean cell | pooled |")
    L.append("|" + "---|" * (len(cells) + 3))
    for wl in perturbed:
        cts = [per_cell[wl][(c, s)] for c, s in cells]
        L.append("| %s | %s | %s | %s |" % (
            wl, " | ".join(_fmt(t) for t in cts),
            _fmt(mean_cell[wl]), _fmt(pooled[wl])))
    L.append("")
    L.append("### Verdict -- does the leaderboard survive?")
    L.append("")
    L.append("Baseline best-ranked method (lowest pooled average rank): **%s**."
             % (base_best or "n/a"))
    for wl in perturbed:
        b = _best(rows, cells, wl)
        L.append("* **%s**: pooled tau = %s; best method now **%s** (%s)."
                 % (wl, _fmt(pooled[wl]), b or "n/a",
                    "unchanged" if b == base_best else "CHANGED"))
    L.append("")
    (OUT_DIR / "weights_summary.md").write_text("\n".join(L))
    print("Wrote %s" % (OUT_DIR / "weights_summary.md"))

    T = []
    T.append("% A4 priority-weight-vector ranking robustness: Kendall tau-b "
             "between the baseline-weight method ranking (by mean TWT) and each "
             "swept weight vector.")
    T.append("% Generated by scripts/r2_weights.py from results/r2_sens/weights.csv.")
    T.append("% Pooled columns use scale-free cross-cell average ranks; 'worst' "
             "is the minimum finite per-cell tau.")
    T.append("\\begin{tabular}{lcccc}")
    T.append("\\toprule")
    T.append("Weight vector & $\\tau$ (150) & $\\tau$ (400) & "
             "$\\tau$ (pooled) & worst cell \\\\")
    T.append("\\midrule")
    labelmap = {"flat": "Flat $(4,3,2,1)$", "steep": "Steep $(27,9,3,1)$"}
    for wl in perturbed:
        cts = [per_cell[wl][(c, s)] for c, s in cells]
        finite = [t for t in cts if t is not None]
        worst = min(finite) if finite else None
        T.append("%s & %s & %s & %s & %s \\\\" % (
            labelmap.get(wl, wl), _fmt(pooled_by_size.get((wl, 150))),
            _fmt(pooled_by_size.get((wl, 400))), _fmt(pooled[wl]), _fmt(worst)))
    T.append("\\bottomrule")
    T.append("\\end{tabular}")
    (OUT_DIR / "tab_weights.tex").write_text("\n".join(T) + "\n")
    print("Wrote %s" % (OUT_DIR / "tab_weights.tex"))

    print("\nBaseline best method: %s" % base_best)
    for wl in perturbed:
        print("  %-6s pooled tau = %s  mean-cell = %s  best -> %s"
              % (wl, _fmt(pooled[wl]), _fmt(mean_cell[wl]),
                 _best(rows, cells, wl)))


if __name__ == "__main__":
    n_mis, guard_ok = run()
    sys.exit(1 if (n_mis or guard_ok is False) else 0)
