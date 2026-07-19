#!/usr/bin/env python
"""P0 - post-cleaning LaborHours histogram + quantiles for Figure 7(b).

Loads the raw FMUCD CSV, applies the locked cleaning pipeline (fmwos.io.clean:
R2-R4/R6/R7), then computes the per-work-order LaborHours distribution on
log-spaced bins RESTRICTED to the six schedulable campuses (1,2,5,9,10,12),
plus the quantiles (p50/p90/p95/p99/mean) both on those six campuses (plotted in
the figure) and on ALL cleaned campuses (used to cross-check the manuscript values,
which are all-campus values from results/p0_profile/overview.json).

Output: results/p0_profile/labor_hist.csv  (tidy long format, one file)
  kind='hist'  : the 60 log bins  (label=bin index, bin_lo, bin_hi, count)
  kind='edge'  : underflow (<0.01) / overflow (>200) counts (provenance)
  kind='quant' : six_*/all_* quantiles (label, value)

Run (a few minutes; loads a 1.4 GB CSV):
  PYTHONPATH=src python scripts/p0_labor_hist.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmwos.io import clean, load_raw  # noqa: E402

RAW = ROOT / "data" / "raw" / "FMUCD.csv"
OUT = ROOT / "results" / "p0_profile" / "labor_hist.csv"
SCHEDULABLE = [1, 2, 5, 9, 10, 12]

# Histogram support: 60 log-spaced bins over [0.01, 200] + under/overflow.
NBINS = 60
LO, HI = 0.01, 200.0

# All-campus quantile targets from the manuscript values (results/p0_profile/overview.json).
MACROS = {"p50": 1.0, "p90": 6.0, "p95": 11.0, "p99": 49.25, "mean": 3.2188530052}
TOL = 0.02  # absolute tolerance on the all-campus cross-check


def _quantiles(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "mean": float(np.mean(x)),
        "n": int(x.size),
    }


def main() -> int:
    print(f"[labor_hist] loading raw CSV {RAW} ...", flush=True)
    df = load_raw(RAW)
    print(f"[labor_hist] raw rows = {len(df):,}; cleaning ...", flush=True)
    dfc, audit = clean(df)
    print(f"[labor_hist] clean work orders = {len(dfc):,}  "
          f"(cap={audit['R4_labor_cap_hours']:.4f} h, "
          f"pm_share={audit['pm_share']:.4f})", flush=True)

    lh_all = dfc["LaborHours"].to_numpy(dtype=float)
    mask6 = dfc["UniversityID"].isin(SCHEDULABLE).to_numpy()
    lh_6 = lh_all[mask6]
    print(f"[labor_hist] all-campus n = {lh_all.size:,}; "
          f"six-campus n = {lh_6.size:,}", flush=True)

    q_all = _quantiles(lh_all)
    q_6 = _quantiles(lh_6)

    # ---- cross-check ALL-campus quantiles vs macros (STOP on real mismatch) ---
    print("[labor_hist] all-campus quantiles vs the manuscript:")
    disagree = []
    for k, tgt in MACROS.items():
        got = q_all[k]
        ok = abs(got - tgt) <= TOL
        print(f"    {k:>4}: recomputed={got:.5f}  macro={tgt:.5f}  "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            disagree.append((k, got, tgt))
    if disagree:
        print("[labor_hist] STOP: recomputed all-campus quantiles disagree with "
              "the manuscript:", disagree, flush=True)
        return 2
    print("[labor_hist] cross-check PASSED (all-campus == macros within tol).",
          flush=True)
    print(f"[labor_hist] SIX-campus quantiles (plotted): {q_6}", flush=True)

    # ---- histogram on the six schedulable campuses -------------------------
    edges = np.logspace(np.log10(LO), np.log10(HI), NBINS + 1)
    counts, _ = np.histogram(lh_6, bins=edges)
    underflow = int((lh_6 < LO).sum())
    overflow = int((lh_6 >= HI).sum())
    assert int(counts.sum()) + underflow + overflow == lh_6.size, "count mismatch"
    print(f"[labor_hist] hist counts sum={int(counts.sum())}  "
          f"underflow(<{LO})={underflow}  overflow(>={HI})={overflow}",
          flush=True)

    # ---- write single tidy CSV --------------------------------------------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "label", "bin_lo", "bin_hi", "count", "value"])
        for i in range(NBINS):
            w.writerow(["hist", i, f"{edges[i]:.6g}", f"{edges[i + 1]:.6g}",
                        int(counts[i]), ""])
        w.writerow(["edge", "underflow", 0.0, f"{LO:.6g}", underflow, ""])
        w.writerow(["edge", "overflow", f"{HI:.6g}", "inf", overflow, ""])
        for scope, q in (("six", q_6), ("all", q_all)):
            for k in ("p50", "p90", "p95", "p99", "mean", "n"):
                w.writerow(["quant", f"{scope}_{k}", "", "", "", q[k]])
    print(f"[labor_hist] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
