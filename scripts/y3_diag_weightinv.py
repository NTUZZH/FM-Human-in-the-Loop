"""Y3 diagnostic: INDEPENDENT verification of the non-myopic weight-INVARIANCE claim.

Claim under test (notes/decisions.md PIVOTAL FINDING):
  "the schedule that is provably OPTIMAL under RECORDED weights is ALSO optimal
   under the TRUE objective TWT* on this benchmark" -- exact 0.0 gap.

Method (per the task brief, fully independent of scripts/y3_headroom_probe2.py):
  For each size-150 LOADED instance (campuses {9,12}; storm a200/a300 and
  pmmix c60):
    * Schedule A = static CP-SAT minimizing sum_j w(c_j) * T_j
                   (RECORDED weights = the instance's own `weight` field),
                   recorded due dates d_j.
    * Schedule B = static CP-SAT minimizing sum_j w*(c*_j) * T_j
                   (TRUE weights: deep-copy the instance, overwrite each
                   wo["weight"] with w*(c*_j) from the LOCKED overlay draw),
                   same recorded due dates d_j.
    * Score BOTH on the TRUE objective, computed here directly (the small
      scoring function copied out of hitl/true_objective.py, NOT imported, so
      this check is independent of that module):
          TWT* = sum_j w*(c*_j) * max(0, C_j - d_j),   d_j = recorded due_bh.
  gap = TWT*(A) - TWT*(B).  B minimizes exactly TWT* (on the centi grid), so
  TWT*(B) is the true optimum and gap >= 0 up to sub-centi rounding.  gap == 0
  means the recorded-weight optimum is ALSO true-objective optimal
  (weight-invariant optimum).

  How CP-SAT reads weights: src/fmwos/cpsat.py line 107
      wt = [int(round(w["weight"])) for w in work_orders]
  and due dates line 106  due_c = [_centi_round(w["due_bh"]) ...].  Verified:
  the instance's recorded `weight` == W_OF_CLASS[priority] and `due_bh` ==
  release_bh + SLA(priority); B changes ONLY the weight field.

beta = 1.0 (strongest recoverable information).  Reports the per-instance gap
distribution and the count of solves that returned OPTIMAL.

Run:  PYTHONPATH=src OMP_NUM_THREADS=1 nice python scripts/y3_diag_weightinv.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import copy
import csv
import glob
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import cpsat                      # noqa: E402  (locked; read/import OK)
from fmwos.hitl import overlay as ov         # noqa: E402  (locked; read/import OK)

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUTDIR = os.path.join(_ROOT, "results", "y3_diag", "weightinv")

MASTER_SEED = 12345
FAMILY = "F-NL"
BETA = 1.0
SIGMA_S = 1.0                    # for the assertion below
N_INST = 20
CAMPUSES = (9, 12)
TIME_LIMIT_S = 60.0              # generous; instances solve in <1 s at this size
CPSAT_WORKERS = 1               # single-threaded solve; parallelism is over instances
MAX_WORKERS = 8
TIE_TOL = 1e-6                   # gap this small counts as EXACT zero

# Loaded size-150 cells:  storm a200/a300 (crew c80 = reduced),  pmmix c60.
CELLS = [
    ("storm", "150", "storm/150/*a200_c80*"),
    ("storm", "150", "storm/150/*a300_c80*"),
    ("pmmix", "150", "pmmix/150/*p20_c60*"),
]

# Locked constants (Appendix B), asserted against overlay.py at import time.
W_OF_CLASS = {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}
SLA = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}


def _assert_constants():
    assert ov.W_OF_CLASS == W_OF_CLASS, ("w(c) mismatch vs overlay.py: %r"
                                         % (ov.W_OF_CLASS,))
    assert abs(ov.SIGMA_S - SIGMA_S) < 1e-12, "sigma_s mismatch"
    assert ov.OverlayParams(beta=BETA).master_seed == MASTER_SEED


def _twt_star(inst, sched, wstar):
    """TRUE objective of a schedule: sum_j w*(c*_j) * max(0, C_j - d_j).

    Copied definition (independent of hitl/true_objective.py): C_j = end_bh,
    d_j = recorded due_bh, w*(c*_j) from the overlay draw.  access_alpha == 0
    so there is no access penalty term.
    """
    wo_by = {w["id"]: w for w in inst["work_orders"]}
    tot = 0.0
    for a in sched.get("assignments", []):
        wo = wo_by[a["wo"]]
        end = float(a["end_bh"])
        due = float(wo["due_bh"])
        tot += float(wstar[wo["id"]]) * max(0.0, end - due)
    return tot


def _twt_recorded(inst, sched):
    """Recorded objective: sum_j w(c_j) * max(0, C_j - d_j)."""
    wo_by = {w["id"]: w for w in inst["work_orders"]}
    tot = 0.0
    for a in sched.get("assignments", []):
        wo = wo_by[a["wo"]]
        end = float(a["end_bh"])
        due = float(wo["due_bh"])
        tot += float(wo["weight"]) * max(0.0, end - due)
    return tot


def _run_instance(args):
    campus, track, size, path = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]
    nwo = len(inst["work_orders"])

    ovl = ov.Overlay(ov.OverlayParams(beta=BETA, family=FAMILY,
                                      master_seed=MASTER_SEED))
    applied = ovl.apply(inst)
    wstar = applied["w_star"]

    # sanity within worker: recorded weight == W(priority), due == rel + SLA(prio)
    for wo in inst["work_orders"]:
        assert abs(float(wo["weight"]) - W_OF_CLASS[int(wo["priority"])]) < 1e-9
        assert abs(float(wo["due_bh"]) -
                   (float(wo["release_bh"]) + SLA[int(wo["priority"])])) < 1e-6

    n_weights_changed = sum(
        1 for wo in inst["work_orders"]
        if abs(float(wo["weight"]) - float(wstar[wo["id"]])) > 1e-9)

    # Schedule A: recorded weights (instance as-is).
    t0 = time.perf_counter()
    A = cpsat.solve(inst, time_limit_s=TIME_LIMIT_S, workers=CPSAT_WORKERS)
    tA = time.perf_counter() - t0

    # Schedule B: true weights (overwrite the weight field only).
    itB = copy.deepcopy(inst)
    for wo in itB["work_orders"]:
        wo["weight"] = float(wstar[wo["id"]])
    t0 = time.perf_counter()
    B = cpsat.solve(itB, time_limit_s=TIME_LIMIT_S, workers=CPSAT_WORKERS)
    tB = time.perf_counter() - t0

    twt_star_A = _twt_star(inst, A, wstar)
    twt_star_B = _twt_star(inst, B, wstar)
    gap = twt_star_A - twt_star_B

    return {
        "campus": campus, "track": track, "size": size,
        "cell": inst_id.rsplit("_", 1)[0], "inst_id": inst_id, "nwo": nwo,
        "n_weights_changed": n_weights_changed,
        "status_A": A["status"], "status_B": B["status"],
        "obj_grid_A_recorded": A["objective_bh"],   # A's grid objective (recorded w)
        "obj_grid_B_true": B["objective_bh"],        # B's grid objective (true w*)
        "twt_recorded_A": _twt_recorded(inst, A),
        "twt_recorded_B": _twt_recorded(inst, B),
        "twt_star_A": twt_star_A,
        "twt_star_B": twt_star_B,
        "gap": gap,
        "wall_A": tA, "wall_B": tB,
    }


def main():
    _assert_constants()
    tasks = []
    for campus in CAMPUSES:
        cdir = "c%02d" % campus
        for track, size, tail in CELLS:
            files = sorted(glob.glob(os.path.join(_INST, cdir, tail + "*.json")))
            for p in files[:N_INST]:
                tasks.append((campus, track, size, p))
    print("weight-invariance verify: %d instances (beta=%.2f, n<=%d/cell, "
          "%d solves total)" % (len(tasks), BETA, N_INST, 2 * len(tasks)))

    records = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        done = 0
        for r in ex.map(_run_instance, tasks):
            records.append(r)
            done += 1
            if done % 20 == 0:
                print("  %d/%d instances" % (done, len(tasks)))

    os.makedirs(_OUTDIR, exist_ok=True)
    out_csv = os.path.join(_OUTDIR, "weightinv_perinstance.csv")
    cols = ["campus", "track", "size", "cell", "inst_id", "nwo",
            "n_weights_changed", "status_A", "status_B",
            "obj_grid_A_recorded", "obj_grid_B_true",
            "twt_recorded_A", "twt_recorded_B", "twt_star_A", "twt_star_B",
            "gap", "wall_A", "wall_B"]
    with open(out_csv, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        for r in sorted(records, key=lambda x: (x["campus"], x["cell"],
                                                x["inst_id"])):
            row = dict(r)
            for k in ("obj_grid_A_recorded", "obj_grid_B_true",
                      "twt_recorded_A", "twt_recorded_B",
                      "twt_star_A", "twt_star_B", "gap"):
                row[k] = "" if row[k] is None else "%.6f" % row[k]
            row["wall_A"] = "%.3f" % row["wall_A"]
            row["wall_B"] = "%.3f" % row["wall_B"]
            wtr.writerow({c: row[c] for c in cols})
    print("wrote %s (%d rows)" % (out_csv, len(records)))

    # ------- aggregate report -------
    n_solves = 2 * len(records)
    opt_A = sum(1 for r in records if r["status_A"] == "OPTIMAL")
    opt_B = sum(1 for r in records if r["status_B"] == "OPTIMAL")
    n_opt = opt_A + opt_B

    exact_zero = sum(1 for r in records if abs(r["gap"]) <= TIE_TOL)
    pos = sum(1 for r in records if r["gap"] > TIE_TOL)      # A worse (claim broken)
    neg = sum(1 for r in records if r["gap"] < -TIE_TOL)     # A better (rounding)
    max_abs_gap = max((abs(r["gap"]) for r in records), default=0.0)
    tot_wstar_changed = sum(r["n_weights_changed"] for r in records)

    print("\n=== SOLVE STATUS ===")
    print("total solves: %d   OPTIMAL: %d/%d  (A: %d/%d, B: %d/%d)"
          % (n_solves, n_opt, n_solves, opt_A, len(records),
             opt_B, len(records)))
    non_opt = [(r["inst_id"], r["status_A"], r["status_B"]) for r in records
               if r["status_A"] != "OPTIMAL" or r["status_B"] != "OPTIMAL"]
    if non_opt:
        print("  NON-OPTIMAL solves:")
        for iid, sA, sB in non_opt:
            print("    %s  A=%s B=%s" % (iid, sA, sB))

    print("\n=== GAP DISTRIBUTION (TWT*(A) - TWT*(B)) ===")
    print("instances: %d   mean w* relabelled: %.1f/150"
          % (len(records), tot_wstar_changed / max(1, len(records))))
    print("exact-zero (|gap|<=%.0e): %d/%d" % (TIE_TOL, exact_zero, len(records)))
    print("gap>0 (recorded-opt WORSE on TWT*): %d" % pos)
    print("gap<0 (recorded-opt better; grid-rounding artifact): %d" % neg)
    print("max |gap|: %.6g weighted units" % max_abs_gap)

    print("\n=== PER (campus, cell): mean TWT* and gap ===")
    print("campus cell                         n  meanTWT*A  meanTWT*B   "
          "meanGap  maxGap  exact0  OPT_AB")
    cells = sorted({(r["campus"], r["cell"]) for r in records})
    for campus, cell in cells:
        sub = [r for r in records if r["campus"] == campus and r["cell"] == cell]
        n = len(sub)
        mA = sum(r["twt_star_A"] for r in sub) / n
        mB = sum(r["twt_star_B"] for r in sub) / n
        mg = sum(r["gap"] for r in sub) / n
        mxg = max(abs(r["gap"]) for r in sub)
        ez = sum(1 for r in sub if abs(r["gap"]) <= TIE_TOL)
        oab = sum(1 for r in sub if r["status_A"] == "OPTIMAL"
                  and r["status_B"] == "OPTIMAL")
        print("%6d %-30s %2d %10.3f %10.3f %8.4f %7.4f  %2d/%d  %2d/%d"
              % (campus, cell, n, mA, mB, mg, mxg, ez, n, oab, n))

    verdict = ("CONFIRMED: exact 0.0 gap on all solves (weight-invariant optimum)"
               if (exact_zero == len(records) and pos == 0 and neg == 0)
               else "NOT exact-0 on all instances -- see distribution above")
    print("\n=== VERDICT: %s ===" % verdict)

    import pickle
    with open(os.path.join(_OUTDIR, "_weightinv_rows.pkl"), "wb") as fh:
        pickle.dump(records, fh)


if __name__ == "__main__":
    main()
