#!/usr/bin/env python
"""W3 R6: what a CONSTANT correction does at beta = 0.

The package's diagnostic question. At beta = 0 nothing per-order is
recoverable, yet the fitted layer reduces true weighted tardiness by 15% (C9)
and 33% (C10) in the extreme-overload columns. Two explanations are compatible
with that, and they imply opposite rewrites of the manuscript's Section 6.6:

  (a) the estimator partially captures the clip's class-level bias -- true
      class-4 orders are on average 0.37 classes more urgent than recorded --
      so the reduction is a coarse but real correction;
  (b) the reduction is a property of the CONSTANT ITSELF: under extreme
      overload a uniform pull-in of the dominant class's deadlines reorders the
      queue in a way that happens to reduce TWT*, whatever the constant means.

This script separates them without fitting anything. It sweeps a constant
``hat_s`` -- the same number for every order, so it contains no information at
all -- through the deployed augmented-ATC decider on the held-out instances,
and reports the reduction against the tuned rule as a function of that
constant. It also runs the class-wise variants (the true class-mean constant,
and a shift applied to the recorded class-4 orders only), so the contribution
of each class is visible.

Under (a) the curve should peak at the true class-mean correction. Under (b) it
should peak at a small positive constant unrelated to it, and the true
class-mean correction should sit off the peak.

Nothing here is fitted, nothing here is a deliverable, and no number from it is
a headline. Run:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-23 \
    python scripts/y3_w3_constsweep.py --cell c9_u130_b0 --workers 4
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse                                                  # noqa: E402
import sys                                                       # noqa: E402
import time                                                      # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np                                               # noqa: E402

import y3_w3_lib as L                                            # noqa: E402
from y3_w3_run import CELLS                                      # noqa: E402

GRID = [-0.40, -0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.08, 0.10, 0.15,
        0.20, 0.30, 0.40, 0.60, 1.00]


def _cell(tag, seed):
    spec = dict(CELLS[tag])
    spec.pop("published_csv"); spec.pop("protocol")
    return L.build_cell(seed=seed, **spec)


def job(args):
    tag, seed, kind, value = args
    L.set_threads(1)
    cell = _cell(tag, seed)
    if kind == "uniform":
        const = {1: value, 2: value, 3: value, 4: value}
    elif kind == "class4_only":
        const = {1: 0.0, 2: 0.0, 3: 0.0, 4: value}
    elif kind == "true_classmean":
        const = L.true_class_mean_effective_shift(cell["train"] + cell["probe"],
                                                  cell["overlay"])
        const = {k: value * v for k, v in const.items()}
    else:
        raise ValueError(kind)
    model = L.Fitted("constant", const_by_class=const)
    dep = L.deployed_twt(model, cell["eval"], cell["overlay"],
                         channel=cell["channel"], seed=seed)
    ken = L.kendall_on_instances(model, cell["eval"], cell["overlay"],
                                 channel=cell["channel"], seed=seed)
    rule = np.asarray(dep["rule"]); aug = np.asarray(dep["aug"])
    return {"tag": tag, "seed": seed, "kind": kind, "value": value,
            "const_by_class": {str(k): float(v) for k, v in const.items()},
            "rule": rule.tolist(), "aug": aug.tolist(),
            "pct_below_rule": 100.0 * (rule.mean() - aug.mean()) / rule.mean(),
            "kendall_tau": ken["kendall_tau"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--seeds", default="301,302,303")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs = ([(a.cell, s, "uniform", v) for s in seeds for v in GRID]
            + [(a.cell, s, "class4_only", v) for s in seeds for v in GRID]
            + [(a.cell, s, "true_classmean", v) for s in seeds
               for v in (0.0, 0.25, 0.5, 1.0)])
    print("[constsweep] cell=%s  %d jobs" % (a.cell, len(jobs)), flush=True)
    recs, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fut = [ex.submit(job, j) for j in jobs]
        for i, f in enumerate(as_completed(fut)):
            r = f.result(); recs.append(r)
            print("  [%3d/%3d] %-15s %+6.3f s%d -> %7.3f%% <RULE  tau=%.3f "
                  "(wall %.0fs)" % (i + 1, len(jobs), r["kind"], r["value"],
                                    r["seed"], r["pct_below_rule"],
                                    r["kendall_tau"], time.time() - t0),
                  flush=True)

    out = {"cell": a.cell, "seeds": seeds, "grid": GRID, "records": recs,
           "note": ("A CONSTANT hat_s contains no per-order information. Its "
                    "reduction against the tuned rule therefore measures what a "
                    "uniform deadline/weight nudge does under this load, not "
                    "recovery. 'true_classmean' scales the TRUE population mean "
                    "effective shift E[c - c* | c], read from the simulator for "
                    "this diagnostic only.")}
    curves = {}
    for kind in ("uniform", "class4_only", "true_classmean"):
        rows = [r for r in recs if r["kind"] == kind]
        vals = sorted({r["value"] for r in rows})
        curve = []
        for v in vals:
            sel = [r for r in rows if r["value"] == v]
            rule = np.asarray([r["rule"] for r in sel], float)
            aug = np.asarray([r["aug"] for r in sel], float)
            curve.append({"value": v,
                          "pooled_pct_below_rule":
                              100.0 * (rule.sum() - aug.sum()) / rule.sum(),
                          "kendall_tau": float(np.mean([r["kendall_tau"] for r in sel])),
                          "n_seeds": len(sel)})
        curves[kind] = curve
    out["curves"] = curves
    L.write_json(os.path.join(L.OUT, "constsweep_%s.json" % a.cell), out)

    print("\nCONSTANT-CORRECTION SWEEP, cell %s (pooled over seeds %s)"
          % (a.cell, seeds))
    for kind, curve in curves.items():
        print("\n  %s" % kind)
        print("    %8s %12s %8s" % ("const", "%<RULE", "tau"))
        for c in curve:
            print("    %+8.3f %12.3f %8.3f"
                  % (c["value"], c["pooled_pct_below_rule"], c["kendall_tau"]))
    print("\nwrote", os.path.join(L.OUT, "constsweep_%s.json" % a.cell))


if __name__ == "__main__":
    main()
