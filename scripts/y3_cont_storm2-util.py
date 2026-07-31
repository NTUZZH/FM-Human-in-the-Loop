#!/usr/bin/env python
"""Y3 continuation: full-class-shift ORACLE-vs-RULE headroom on the storm2
utilization ladder (regime STORM2).

Operationalization under test -- FULL-CLASS-SHIFT (the deadline moves).

The supervisor overlay (overlay.py, F-NL, master_seed=12345, sigma_s=1.0) draws,
per order j, a class shift s_j and the realized TRUE class

    c*_j = clip(c_j - s_j, 1, 4)                      (positive s = more urgent).

Recorded fields the deployed rule sees:  w(c) = 8/4/2/1,  SLA(c) = 8/24/80/171.4
business-hours, recorded due d_j = r_j + SLA(c_j)  (holds exactly on these
instances, verified).  The TRUE quantities the supervisor sees:

    TRUE weight   w*_j = w(c*_j)
    TRUE deadline d*_j = r_j + SLA(c*_j)              (r_j = release_bh)

Everything is scored on the TRUE objective

    TWT* = sum_j w*_j * max(0, C_j - d*_j)

with C_j the realized completion of order j and feasibility from the independent
validator (unchanged).  This is the FULL-class-shift objective: the latent moves
BOTH the cost of lateness (w -> w*) AND the clock (d -> d*).

What is compared, in the actual dynamic dispatch env (the headline pipeline, no
solver -- storm2 is CP-SAT-intractable):

  RULE   = ATC (and EDD) on the RECORDED fields (recorded w, recorded d).  The
           deployed myopic rule; it does NOT see the latent.
  ORACLE = the SAME myopic ATC dispatcher computed with the TRUE class: true
           weight w* AND true deadline d*.  Built from the supervisor's
           preferred-pick path (deciders.run_oracle_greedy / Supervisor.
           preferred_pick), which already injects w*; here we ALSO inject d* by
           overriding the supervisor's per-order due map with d* (the locked
           supervisor.py only injects w*).  This is the full-information ceiling
           for a myopic dispatcher.

Reported per (campus/pool, u-level, beta):
  headroom = (TWT*_RULE - TWT*_ORACLE) / TWT*_RULE, pooled ratio of means,
  per-instance W/T/L (tie band |diff| <= 1.0 weighted unit),
  utilization = sum p_bh of a trade / (crew_of_trade * horizon_bh), pooled and
  worst-trade, and the incomplete-at-horizon count (orders finishing after the
  80 bh window) for the deployed ATC schedule -- so the realism of each u-rung
  is explicit.

storm2 is a fixed 80 bh window with a Poisson workload rate-scaled to a targeted
utilization u_target in {0.7,0.9,1.0,1.1,1.3} (u70..u130); the work-order count n
GROWS with u, so higher u is genuine sustained overload.

Run:  PYTHONPATH=src nice python scripts/y3_cont_storm2-util.py [--workers 8]
                                                                [--n-per-cell 12]
"""

from __future__ import annotations

import os

# Single-threaded numeric libs BEFORE numpy import (worker processes).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import glob
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.env import DispatchEnv                 # noqa: E402
from fmwos.hitl import deciders as dec            # noqa: E402
from fmwos.hitl import overlay as ov              # noqa: E402
from fmwos.hitl.supervisor import Supervisor      # noqa: E402
from fmwos import validator as _validator         # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_cont", "storm2-util")

SEED = 301
MASTER_SEED = 12345
FAMILY = "F-NL"
BETAS = (0.25, 0.5, 0.75, 1.0)

# {9,12} primary; {5,10} for the sign-stability cross-check.
CAMPUSES = (9, 12, 5, 10)
PRIMARY = (9, 12)
SIGNCHK = (5, 10)
U_LEVELS = (70, 90, 100, 110, 130)          # u_target*100
TIE_TOL = 1.0

# Recorded/true class constants (verified against overlay.py + a storm2 sample).
SLA = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}          # business-hours
W_OF_CLASS = {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}
HORIZON_BH = 80.0                                    # storm2 fixed window

RULES = ("atc", "edd")

_OVERLAYS = {}


def _overlay(beta):
    o = _OVERLAYS.get(beta)
    if o is None:
        o = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                        master_seed=MASTER_SEED))
        _OVERLAYS[beta] = o
    return o


def _dstar_map(inst, applied):
    """True deadline d*_j = release_bh + SLA(c*_j) per order."""
    cstar = applied["c_star"]
    return {w["id"]: float(w["release_bh"]) + SLA[cstar[w["id"]]]
            for w in inst["work_orders"]}


def _score_twt_star(inst, sched, applied, dstar, wo_rel):
    """TRUE objective TWT* = sum_j w*_j * max(0, C_j - d*_j).  Independent copy of
    the true_objective.py scoring block, but reading d* (NOT recorded due) and w*.
    Returns (TWT*, feasible)."""
    base = _validator.validate(inst, sched)
    wstar = applied["w_star"]
    twt = 0.0
    for a in sched.get("assignments", []) or []:
        wid = a.get("wo")
        end = a.get("end_bh")
        if wid is None or end is None or wid not in dstar:
            continue
        twt += wstar[wid] * max(0.0, float(end) - dstar[wid])
    return twt, bool(base["feasible"])


def _utilization(inst):
    """Pooled and worst-trade utilization over the fixed HORIZON_BH window, plus
    the pooled crew/work totals.  util_g = sum p_bh in trade g / (k_g * H)."""
    kg = defaultdict(int)
    for t in inst["technicians"]:
        kg[t["trade"]] += 1
    pg = defaultdict(float)
    for w in inst["work_orders"]:
        pg[w["trade"]] += float(w["p_bh"])
    total_crew = sum(kg.values())
    total_work = sum(pg.values())
    util_pool = total_work / (total_crew * HORIZON_BH) if total_crew else 0.0
    worst = 0.0
    worst_trade = None
    for g, work in pg.items():
        k = kg.get(g, 0)
        if k <= 0:
            continue
        u = work / (k * HORIZON_BH)
        if u > worst:
            worst, worst_trade = u, g
    return util_pool, worst, worst_trade, total_work, total_crew


def _incomplete_at_horizon(sched):
    """Count of orders finishing AFTER the 80 bh window, and the makespan."""
    inc = 0
    mk = 0.0
    for a in sched.get("assignments", []) or []:
        e = float(a["end_bh"])
        if e > HORIZON_BH:
            inc += 1
        if e > mk:
            mk = e
    return inc, mk


def _process_instance(args):
    campus, u, path = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]
    n_wos = len(inst["work_orders"])
    wo_rel = {w["id"]: float(w["release_bh"]) for w in inst["work_orders"]}

    util_pool, util_worst, worst_trade, total_work, total_crew = _utilization(inst)

    # RULE schedules on RECORDED fields -- beta-independent (rules never see w*/d*).
    rule_sched = {r: dec.run_rule(DispatchEnv(inst), r, seed=SEED) for r in RULES}
    inc_atc, mk_atc = _incomplete_at_horizon(rule_sched["atc"])

    rec = {
        "campus": campus, "u": u, "inst_id": inst_id, "n_wos": n_wos,
        "util_pool": util_pool, "util_worst": util_worst,
        "worst_trade": worst_trade,
        "inc_atc": inc_atc, "inc_atc_frac": inc_atc / n_wos if n_wos else 0.0,
        "makespan_atc": mk_atc, "window_bh": HORIZON_BH,
        "per_beta": {},
    }

    for beta in BETAS:
        overlay = _overlay(beta)
        applied = overlay.apply(inst)
        dstar = _dstar_map(inst, applied)

        # ORACLE = true-weight-and-true-deadline ATC (myopic-greedy skyline).
        # Supervisor.preferred_pick already uses w* (self.wstar); inject d* by
        # overriding its per-order due map with d* (locked file only injects w*).
        sup = Supervisor(overlay, inst, rho=0.0, applied=applied)
        sup.due = dstar
        or_sched = dec.run_oracle_greedy(DispatchEnv(inst), sup, seed=SEED)
        twt_or, feas_or = _score_twt_star(inst, or_sched, applied, dstar, wo_rel)

        row = {"twt_oracle": twt_or, "feas_oracle": feas_or}
        for r in RULES:
            twt_r, feas_r = _score_twt_star(inst, rule_sched[r], applied, dstar,
                                            wo_rel)
            row["twt_" + r] = twt_r
            row["feas_" + r] = feas_r
        rec["per_beta"][beta] = row
    return rec


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _wtl(records, rule):
    """Per-instance win/tie/loss of ORACLE vs `rule` on TWT* (positive diff =>
    oracle better).  Tie band = TIE_TOL weighted units."""
    w = t = l = 0
    for r in records:
        diff = r["twt_" + rule] - r["twt_oracle"]
        if abs(diff) <= TIE_TOL:
            t += 1
        elif diff > TIE_TOL:
            w += 1
        else:
            l += 1
    return w, t, l


def _cell_stats(recs_beta, rule):
    """recs_beta: list of per-instance per-beta rows (already merged with the
    instance-level util fields).  Returns the aggregate dict for one cell."""
    n = len(recs_beta)
    mean_rule = sum(r["twt_" + rule] for r in recs_beta) / n
    mean_or = sum(r["twt_oracle"] for r in recs_beta) / n
    hr = (100.0 * (mean_rule - mean_or) / mean_rule) if mean_rule > 1e-9 else 0.0
    w, t, l = _wtl(recs_beta, rule)
    return dict(n=n, mean_rule=mean_rule, mean_oracle=mean_or, pct_headroom=hr,
                wins=w, ties=t, losses=l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-per-cell", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(_OUT, exist_ok=True)

    tasks = []
    for campus in CAMPUSES:
        cdir = "c%02d" % campus
        for u in U_LEVELS:
            fs = sorted(glob.glob(os.path.join(
                _INST, cdir, "storm2", "w80",
                "%s_storm2_w80_u%d_*.json" % (cdir, u))))[:args.n_per_cell]
            for p in fs:
                tasks.append((campus, u, p))
    print("storm2-util: %d instance-tasks (x%d betas), campuses=%s u=%s n/cell=%d"
          % (len(tasks), len(BETAS), CAMPUSES, U_LEVELS, args.n_per_cell),
          flush=True)

    t0 = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        done = 0
        for rec in ex.map(_process_instance, tasks, chunksize=2):
            records.append(rec)
            done += 1
            if done % 40 == 0:
                print("  processed %d/%d  (%.0fs)"
                      % (done, len(tasks), time.time() - t0), flush=True)
    print("compute done in %.0fs" % (time.time() - t0), flush=True)

    # ---- flatten to per-(instance,beta) rows carrying util fields ----------
    flat = []          # one row per (instance, beta)
    for rec in records:
        for beta in BETAS:
            row = dict(rec["per_beta"][beta])
            row.update(campus=rec["campus"], u=rec["u"], beta=beta,
                       inst_id=rec["inst_id"], n_wos=rec["n_wos"],
                       util_pool=rec["util_pool"], util_worst=rec["util_worst"],
                       inc_atc=rec["inc_atc"], inc_atc_frac=rec["inc_atc_frac"],
                       makespan_atc=rec["makespan_atc"])
            flat.append(row)

    # ---- per-instance CSV ---------------------------------------------------
    inst_csv = os.path.join(_OUT, "per_instance.csv")
    cols = ["campus", "u", "beta", "inst_id", "n_wos", "util_pool", "util_worst",
            "inc_atc", "inc_atc_frac", "makespan_atc",
            "twt_atc", "twt_edd", "twt_oracle",
            "feas_atc", "feas_edd", "feas_oracle"]
    with open(inst_csv, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=cols)
        wcsv.writeheader()
        for r in flat:
            wcsv.writerow({c: r.get(c) for c in cols})
    print("wrote %s (%d rows)" % (inst_csv, len(flat)))

    # ---- cell aggregates ----------------------------------------------------
    def _subset(scope_campuses, u, beta):
        return [r for r in flat if r["campus"] in scope_campuses
                and r["u"] == u and r["beta"] == beta]

    def _util_mean(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    cell_rows = []   # for CSV + summary
    scopes = [("pool_9+12", PRIMARY), ("pool_5+10", SIGNCHK),
              ("c9", (9,)), ("c12", (12,)), ("c5", (5,)), ("c10", (10,))]
    for scope_name, camps in scopes:
        for u in U_LEVELS:
            for beta in BETAS:
                sub = _subset(camps, u, beta)
                if not sub:
                    continue
                base = dict(scope=scope_name, u=u, beta=beta,
                            util_pool=_util_mean(sub, "util_pool"),
                            util_worst=_util_mean(sub, "util_worst"),
                            inc_atc_frac=_util_mean(sub, "inc_atc_frac"),
                            inc_atc=_util_mean(sub, "inc_atc"),
                            n_wos=_util_mean(sub, "n_wos"))
                for rule in RULES:
                    st = _cell_stats(sub, rule)
                    row = dict(base)
                    row["rule"] = rule
                    row.update(st)
                    cell_rows.append(row)

    cell_csv = os.path.join(_OUT, "cells.csv")
    ccols = ["scope", "rule", "u", "beta", "util_pool", "util_worst",
             "inc_atc_frac", "inc_atc", "n_wos", "n", "mean_rule", "mean_oracle",
             "pct_headroom", "wins", "ties", "losses"]
    with open(cell_csv, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=ccols)
        wcsv.writeheader()
        for r in cell_rows:
            out = dict(r)
            for k in ("util_pool", "util_worst", "inc_atc_frac", "n_wos",
                      "mean_rule", "mean_oracle", "pct_headroom", "inc_atc"):
                out[k] = "%.4f" % out[k]
            wcsv.writerow({c: out[c] for c in ccols})
    print("wrote %s (%d rows)" % (cell_csv, len(cell_rows)))

    # ---- console ladder (ATC, primary pool) --------------------------------
    print("\n=== ATC headroom ladder, pooled {9,12} ===")
    print("  u  util_pool util_worst inc@80  " + "  ".join("b%.2f" % b for b in BETAS))
    for u in U_LEVELS:
        line = "u%-3d" % u
        up = uw = incf = 0.0
        hrs = []
        for beta in BETAS:
            sub = _subset(PRIMARY, u, beta)
            st = _cell_stats(sub, "atc")
            hrs.append(st["pct_headroom"])
            up = _util_mean(sub, "util_pool")
            uw = _util_mean(sub, "util_worst")
            incf = _util_mean(sub, "inc_atc_frac")
        print("  %s %8.3f %9.3f %5.0f%%  %s"
              % (line, up, uw, 100 * incf,
                 "  ".join("%5.1f" % h for h in hrs)))

    # ---- JSON summary -------------------------------------------------------
    summary = {"betas": list(BETAS), "campuses": list(CAMPUSES),
               "u_levels": list(U_LEVELS), "n_per_cell": args.n_per_cell,
               "tie_tol": TIE_TOL, "seed": SEED, "master_seed": MASTER_SEED,
               "family": FAMILY, "horizon_bh": HORIZON_BH,
               "cells": cell_rows}
    with open(os.path.join(_OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote %s" % os.path.join(_OUT, "summary.json"))


if __name__ == "__main__":
    main()
