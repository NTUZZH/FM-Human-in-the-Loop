#!/usr/bin/env python
"""y3_verc TASK 3 -- H1 room.  Campus 9, storm2 u100, beta=1.0, rho=0.25 TARGETED,
epsilon=0.  Ladder scored on TWT*(w*,d*), 12 instances:

   RULE      = ATC on recorded fields, no supervisor.
   RULE+SUP  = ATC + simulated supervisor reviewing up to 25% of decisions and
               overriding toward the true-(w*,d*) preferred pick.  The supervisor
               due map is overridden to d* so its preferred pick and its override
               improvement both use the TRUE deadline (locked file injects only
               w*); epsilon=0, theta = locked default (1.0).
   ORACLE    = myopic ATC on true w*,d* at every decision (full info ceiling).

The H1 question: does 25%-review RULE+SUP leave a SUBSTANTIAL gap to ORACLE that a
learned policy could fill?  We report the three levels + the fraction of the
RULE->ORACLE gap that 25% review closes.  theta=0 also reported as a max-override
sensitivity (every beneficial override at 25% review fires).
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import glob, json, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from fmwos.env import DispatchEnv
from fmwos.hitl import deciders as dec
from fmwos.hitl import overlay as ov
from fmwos.hitl.supervisor import Supervisor
from fmwos import validator as _validator

_INST = os.path.join(_ROOT, "data", "processed", "instances")
SEED = 301
SLA = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}
BETA = 1.0
RHO = 0.25
N = 12


def _dstar_map(inst, applied):
    cs = applied["c_star"]
    return {w["id"]: float(w["release_bh"]) + SLA[cs[w["id"]]]
            for w in inst["work_orders"]}


def _score(inst, sched, wstar, dstar):
    base = _validator.validate(inst, sched)
    twt = 0.0
    for a in sched.get("assignments", []) or []:
        wid = a.get("wo"); end = a.get("end_bh")
        if wid is None or end is None or wid not in dstar:
            continue
        twt += wstar[wid] * max(0.0, float(end) - dstar[wid])
    return twt, bool(base["feasible"])


def run_cell(campus=9, u=100):
    cdir = "c%02d" % campus
    fs = sorted(glob.glob(os.path.join(_INST, cdir, "storm2", "w80",
                "%s_storm2_w80_u%d_*.json" % (cdir, u))))[:N]
    overlay = ov.Overlay(ov.OverlayParams(beta=BETA, family="F-NL", master_seed=12345))

    rows = []
    for p in fs:
        inst = json.load(open(p))
        applied = overlay.apply(inst)
        wstar = applied["w_star"]
        dstar = _dstar_map(inst, applied)

        # RULE
        rule_sched = dec.run_rule(DispatchEnv(inst), "atc", seed=SEED)
        twt_rule, _ = _score(inst, rule_sched, wstar, dstar)

        # RULE+SUP (theta=1.0 default, and theta=0 max-override)
        supdata = {}
        for theta in (1.0, 0.0):
            sup = Supervisor(overlay, inst, rho=RHO, epsilon=0.0, theta=theta,
                             mechanism="targeted", seed=SEED, applied=applied)
            sup.due = dstar
            sched, log = dec.run_rule_sup(DispatchEnv(inst), "atc", sup, seed=SEED)
            twt, _ = _score(inst, sched, wstar, dstar)
            summ = sup.summary()
            supdata[theta] = dict(twt=twt, rev_frac=summ["reviewed_fraction"],
                                  n_rev=summ["n_reviews"],
                                  n_over=summ["n_overrides"],
                                  n_reviewable=summ["n_reviewable"])

        # ORACLE
        osup = Supervisor(overlay, inst, rho=0.0, applied=applied)
        osup.due = dstar
        or_sched = dec.run_oracle_greedy(DispatchEnv(inst), osup, seed=SEED)
        twt_or, _ = _score(inst, or_sched, wstar, dstar)

        rows.append(dict(inst=inst["meta"]["id"], n_wos=len(inst["work_orders"]),
                         rule=twt_rule, sup1=supdata[1.0]["twt"],
                         sup0=supdata[0.0]["twt"], oracle=twt_or,
                         rev_frac1=supdata[1.0]["rev_frac"],
                         n_over1=supdata[1.0]["n_over"],
                         rev_frac0=supdata[0.0]["rev_frac"],
                         n_over0=supdata[0.0]["n_over"]))
    return rows


if __name__ == "__main__":
    rows = run_cell(9, 100)
    m_rule = np.mean([r["rule"] for r in rows])
    m_sup1 = np.mean([r["sup1"] for r in rows])
    m_sup0 = np.mean([r["sup0"] for r in rows])
    m_or = np.mean([r["oracle"] for r in rows])
    revf1 = np.mean([r["rev_frac1"] for r in rows])
    revf0 = np.mean([r["rev_frac0"] for r in rows])
    over1 = np.mean([r["n_over1"] for r in rows])
    over0 = np.mean([r["n_over0"] for r in rows])

    gap = m_rule - m_or
    closed1 = 100.0 * (m_rule - m_sup1) / gap if gap > 1e-9 else 0.0
    closed0 = 100.0 * (m_rule - m_sup0) / gap if gap > 1e-9 else 0.0

    print("=== TASK 3 H1 ladder: c9 storm2 u100 beta=1.0 rho=0.25 targeted eps=0 (n=%d) ==="
          % len(rows))
    print(" TWT*(w*,d*) pooled means:")
    print("   RULE (ATC, no sup)        = %.1f" % m_rule)
    print("   RULE+SUP (theta=1.0,def)  = %.1f   [review_frac=%.3f, overrides/inst=%.0f, gap closed=%.1f%%]"
          % (m_sup1, revf1, over1, closed1))
    print("   RULE+SUP (theta=0, maxov) = %.1f   [review_frac=%.3f, overrides/inst=%.0f, gap closed=%.1f%%]"
          % (m_sup0, revf0, over0, closed0))
    print("   ORACLE (full info)        = %.1f" % m_or)
    print(" RULE->ORACLE gap = %.1f (%.1f%% of RULE); 25%% review closes %.1f%% (theta=1) / %.1f%% (theta=0)"
          % (gap, 100.0 * gap / m_rule, closed1, closed0))
    print(" REMAINING gap SUP(theta=1)->ORACLE = %.1f (%.1f%% of RULE)"
          % (m_sup1 - m_or, 100.0 * (m_sup1 - m_or) / m_rule))
    out = dict(cell="c9_storm2_u100_b1.0_rho0.25", n=len(rows),
               mean_rule=m_rule, mean_sup_theta1=m_sup1, mean_sup_theta0=m_sup0,
               mean_oracle=m_or, review_frac_theta1=revf1, review_frac_theta0=revf0,
               overrides_per_inst_theta1=over1, overrides_per_inst_theta0=over0,
               gap_closed_pct_theta1=closed1, gap_closed_pct_theta0=closed0,
               rows=rows)
    with open(os.path.join(_ROOT, "results", "y3_verc", "task3_h1_ladder.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote task3_h1_ladder.json")
