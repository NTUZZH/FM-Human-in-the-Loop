#!/usr/bin/env python
"""y3_verc TASK 1 -- non-myopic / strong-baseline confirmation.

Does the ORACLE (myopic ATC on TRUE w*,d*) still beat the BEST RECORDED planner
by a wide, sign-stable margin, scored on the TRUE objective TWT*(w*,d*)?

Recorded planners:
  * FULL PDR portfolio on RECORDED fields: atc, edd, wspt, pfifo, mor (non-delay
    dispatch).  best_portfolio = per-instance min TWT* over these (an ORACLE
    SELECTION over recorded rules -- generous to the recorded side).
  * GA on RECORDED (w,d): gap-aware INSERTION decoder (strictly stronger than
    non-delay), a genuinely NON-MYOPIC recorded planner.  Scored on TWT*.
ORACLE:
  * dec.run_oracle_greedy with Supervisor(rho=0) whose due map is overridden to
    d* (true deadline) and whose weights are w* -- the full-information myopic
    ceiling, IDENTICAL construction to scripts/y3_cont_storm2-util.py.

Cells: campuses {9,10} x u in {90,100}, beta=1.0, first 12 instances each.
GA subset: 6 instances of c9 u100 and 6 of c10 u100 (the headline cells).
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import argparse, csv, glob, json, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from fmwos.env import DispatchEnv
from fmwos.hitl import deciders as dec
from fmwos.hitl import overlay as ov
from fmwos.hitl.supervisor import Supervisor
from fmwos import validator as _validator
from fmwos import ga as _ga

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_verc")
SEED = 301
MASTER_SEED = 12345
FAMILY = "F-NL"
BETA = 1.0
SLA = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}
RULES = ("atc", "edd", "wspt", "pfifo", "mor")
TIE_TOL = 1.0
CELLS = [(9, 90), (9, 100), (10, 90), (10, 100)]
GA_CELLS = [(9, 100), (10, 100)]
GA_N = 6
GA_BUDGET = 60.0

_OVERLAY = None


def _overlay():
    global _OVERLAY
    if _OVERLAY is None:
        _OVERLAY = ov.Overlay(ov.OverlayParams(beta=BETA, family=FAMILY,
                                               master_seed=MASTER_SEED))
    return _OVERLAY


def _dstar_map(inst, applied):
    cstar = applied["c_star"]
    return {w["id"]: float(w["release_bh"]) + SLA[cstar[w["id"]]]
            for w in inst["work_orders"]}


def _score_twt_star(inst, sched, wstar, dstar):
    base = _validator.validate(inst, sched)
    twt = 0.0
    for a in sched.get("assignments", []) or []:
        wid = a.get("wo"); end = a.get("end_bh")
        if wid is None or end is None or wid not in dstar:
            continue
        twt += wstar[wid] * max(0.0, float(end) - dstar[wid])
    return twt, bool(base["feasible"])


def _process(args):
    campus, u, path, do_ga = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]
    n = len(inst["work_orders"])
    overlay = _overlay()
    applied = overlay.apply(inst)
    wstar = applied["w_star"]
    dstar = _dstar_map(inst, applied)

    row = {"campus": campus, "u": u, "inst_id": inst_id, "n_wos": n}
    # recorded PDR portfolio
    for r in RULES:
        sched = dec.run_rule(DispatchEnv(inst), r, seed=SEED)
        twt, feas = _score_twt_star(inst, sched, wstar, dstar)
        row["twt_" + r] = twt
        row["feas_" + r] = feas
    # ORACLE (true w*,d*)
    sup = Supervisor(overlay, inst, rho=0.0, applied=applied)
    sup.due = dstar
    or_sched = dec.run_oracle_greedy(DispatchEnv(inst), sup, seed=SEED)
    twt_or, feas_or = _score_twt_star(inst, or_sched, wstar, dstar)
    row["twt_oracle"] = twt_or
    row["feas_oracle"] = feas_or
    # GA on recorded (w,d), scored on TWT*
    if do_ga:
        t0 = time.perf_counter()
        gs = _ga.solve_ga(inst, budget_s=GA_BUDGET, seed=SEED, pop=100)
        twt_ga, feas_ga = _score_twt_star(inst, gs, wstar, dstar)
        row["twt_ga"] = twt_ga
        row["feas_ga"] = feas_ga
        row["ga_gens"] = gs.get("generations")
        row["ga_wall"] = time.perf_counter() - t0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-per-cell", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(_OUT, exist_ok=True)

    tasks = []
    ga_set = set()
    for (campus, u) in GA_CELLS:
        cdir = "c%02d" % campus
        fs = sorted(glob.glob(os.path.join(_INST, cdir, "storm2", "w80",
                    "%s_storm2_w80_u%d_*.json" % (cdir, u))))[:GA_N]
        for p in fs:
            ga_set.add(p)
    for (campus, u) in CELLS:
        cdir = "c%02d" % campus
        fs = sorted(glob.glob(os.path.join(_INST, cdir, "storm2", "w80",
                    "%s_storm2_w80_u%d_*.json" % (cdir, u))))[:args.n_per_cell]
        for p in fs:
            tasks.append((campus, u, p, p in ga_set))
    print("task1: %d instance-tasks (GA on %d), workers=%d"
          % (len(tasks), len(ga_set), args.workers), flush=True)

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_process, tasks, chunksize=1):
            rows.append(r)
    print("compute done in %.0fs" % (time.time() - t0), flush=True)

    # per-instance CSV
    cols = ["campus", "u", "inst_id", "n_wos"] + \
           ["twt_" + r for r in RULES] + ["twt_oracle", "twt_ga",
            "ga_gens", "ga_wall", "feas_oracle"]
    with open(os.path.join(_OUT, "task1_per_instance.csv"), "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=cols)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow({c: r.get(c) for c in cols})

    # cell aggregates
    def wtl(sub, keyfn):
        w = t = l = 0
        for r in sub:
            diff = keyfn(r) - r["twt_oracle"]
            if abs(diff) <= TIE_TOL:
                t += 1
            elif diff > TIE_TOL:
                w += 1
            else:
                l += 1
        return w, t, l

    def best_rec(r, include_ga=True):
        vals = [r["twt_" + x] for x in RULES]
        if include_ga and ("twt_ga" in r) and (r.get("twt_ga") is not None):
            vals.append(r["twt_ga"])
        return min(vals)

    def best_rule_name(r):
        pairs = [(r["twt_" + x], x) for x in RULES]
        if ("twt_ga" in r) and (r.get("twt_ga") is not None):
            pairs.append((r["twt_ga"], "ga"))
        return min(pairs)[1]

    cell_lines = []
    summary_cells = []
    for (campus, u) in CELLS:
        sub = [r for r in rows if r["campus"] == campus and r["u"] == u]
        n = len(sub)
        m_or = sum(r["twt_oracle"] for r in sub) / n
        m_atc = sum(r["twt_atc"] for r in sub) / n
        m_bp = sum(best_rec(r, include_ga=False) for r in sub) / n   # portfolio only
        has_ga = any("twt_ga" in r and r.get("twt_ga") is not None for r in sub)
        m_ball = sum(best_rec(r, include_ga=True) for r in sub) / n  # incl GA
        hr_atc = 100.0 * (m_atc - m_or) / m_atc if m_atc > 1e-9 else 0.0
        hr_bp = 100.0 * (m_bp - m_or) / m_bp if m_bp > 1e-9 else 0.0
        hr_ball = 100.0 * (m_ball - m_or) / m_ball if m_ball > 1e-9 else 0.0
        w_bp, t_bp, l_bp = wtl(sub, lambda r: best_rec(r, include_ga=False))
        w_ba, t_ba, l_ba = wtl(sub, lambda r: best_rec(r, include_ga=True))
        # which rule wins the portfolio how often
        bestcount = defaultdict(int)
        for r in sub:
            bestcount[best_rule_name(r)] += 1
        # GA stats
        ga_rows = [r for r in sub if r.get("twt_ga") is not None]
        m_ga = (sum(r["twt_ga"] for r in ga_rows) / len(ga_rows)) if ga_rows else None
        ga_gens = (sum(r["ga_gens"] for r in ga_rows) / len(ga_rows)) if ga_rows else None
        c = dict(campus=campus, u=u, n=n, mean_oracle=m_or, mean_atc=m_atc,
                 mean_best_portfolio=m_bp, mean_best_all=m_ball, mean_ga=m_ga,
                 hr_vs_atc=hr_atc, hr_vs_best_portfolio=hr_bp,
                 hr_vs_best_all=hr_ball,
                 wtl_vs_bp="%d/%d/%d" % (w_bp, t_bp, l_bp),
                 wtl_vs_best_all="%d/%d/%d" % (w_ba, t_ba, l_ba),
                 best_rule_counts=dict(bestcount), ga_n=len(ga_rows),
                 ga_mean_gens=ga_gens)
        summary_cells.append(c)
        cell_lines.append(
            "c%d u%d n=%d | ORACLE=%.1f ATC=%.1f bestPDR=%.1f bestALL=%.1f%s "
            "| HR vs ATC=%.1f%% vs bestPDR=%.1f%% vs bestALL=%.1f%% "
            "| W/T/L(o vs bestPDR)=%d/%d/%d (vs bestALL)=%d/%d/%d "
            "| bestrule=%s | GA(n=%d,gens~%s,mean=%s)"
            % (campus, u, n, m_or, m_atc, m_bp, m_ball,
               "" if m_ga is None else " GA=%.1f" % m_ga,
               hr_atc, hr_bp, hr_ball, w_bp, t_bp, l_bp, w_ba, t_ba, l_ba,
               dict(bestcount), len(ga_rows),
               "%.0f" % ga_gens if ga_gens else "NA",
               "%.1f" % m_ga if m_ga is not None else "NA"))

    print("\n=== TASK 1 cell summary (beta=1.0) ===")
    for ln in cell_lines:
        print(ln)
    with open(os.path.join(_OUT, "task1_summary.json"), "w") as fh:
        json.dump(summary_cells, fh, indent=1, default=str)
    print("\nwrote task1_per_instance.csv + task1_summary.json")


if __name__ == "__main__":
    main()
