#!/usr/bin/env python
"""Y3 continuation, CREW-STARVATION regime (full-class-shift operationalization).

Question
--------
Under the full-class-shift latent (the supervisor's true class c* moves BOTH the
tardiness weight w* AND the SLA deadline d*), is there real information headroom
for a myopic dispatcher when the crew is starved? We shrink the crew of size-400
replay and generator instances (campuses 9, 12) down a multiplier ladder and, at
each level, compare:

  RULE   : the deployed dispatcher on the RECORDED fields (recorded w, recorded
           d = r + SLA(c)).  Sees nothing latent.
  ORACLE : the SAME dispatcher computed with the TRUE class -- true weight w*(c*)
           AND true deadline d* = r + SLA(c*).  The full-information ceiling for a
           myopic dispatcher.

Both schedules are scored on the SINGLE true objective
    TWT* = sum_j w*(c*_j) * max(0, C_j - d*_j)
(the deadline term reads d*, not the recorded due).  Headroom per cell =
(TWT*_RULE - TWT*_ORACLE) / TWT*_RULE.

Headline rule = ATC; EDD reported alongside.

Latent (UNCHANGED from overlay.py)
----------------------------------
Overlay(F-NL, master_seed=12345, sigma_s=1.0).  c*_j = clip(c_j - s_j, 1, 4).
w(1..4) = 8/4/2/1 ; SLA(1..4) = 8/24/80/171.4 bh ; recorded d_j = r_j + SLA(c_j)
(verified exact on a sample of both tracks).  The latent is applied to the
ORIGINAL instance so c*/w*/d* are IDENTICAL across the whole crew ladder -- the
only thing that varies within a (source, campus) is the crew multiplier.

Crew knob
---------
crew_multiplier m applied EXACTLY as scripts/p4_dyneval.py applies it to replay
(``fmwos.tightness.scale_crew``): per-trade technician count -> max(1, round(cnt*m)).
m = 1.0 uses the original instance untouched (the p4 replay-default reference);
m < 1.0 uses scale_crew(m).  Ladder m in {1.0, 0.75, 0.5, 0.35, 0.25}.

ORACLE decider
--------------
ATC oracle: the supervisor preferred-pick path (deciders.run_oracle_greedy over
fmwos.hitl.supervisor.Supervisor).  Supervisor.preferred_pick already injects w*;
its due-date term reads the RECORDED due, so we OVERRIDE sup.due with d* in this
scratch wrapper (the locked file is untouched).  EDD oracle: a plain earliest-d*
decider driven through env.run_supervised (no locked file touched).

Utilization
-----------
Per instance, horizon = meta.window_bh (== the release span).  Per trade
u_trade = sum p_bh(trade) / (crew(trade) * window).  pooled = sum p_bh /
(crew_total * window) ; worst-trade = max over trades.  Reported as the MEDIAN
over the cell's instances (robust to replay's degenerate tiny-window instances).
Also reported: incomplete-at-horizon = mean count of orders finishing after
window_bh under RULE-ATC (realized backlog).

Run:  PYTHONPATH=src OMP_NUM_THREADS=1 nice python scripts/y3_cont_crew-starvation.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import glob
import json
import sys
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import pdrs, tightness            # noqa: E402
from fmwos.env import DispatchEnv            # noqa: E402
from fmwos.validator import validate         # noqa: E402
from fmwos.hitl import deciders as dec       # noqa: E402
from fmwos.hitl import overlay as ov         # noqa: E402
from fmwos.hitl.supervisor import Supervisor # noqa: E402

# --------------------------------------------------------------------------- #
# Locked constants (verified against overlay.py + a sample of both tracks).
# --------------------------------------------------------------------------- #
SLA = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}     # business hours, per class
W_OF = {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}
SEED = 301
MASTER_SEED = 12345
FAMILY = "F-NL"
SIGMA_S = 1.0

SOURCES = ("replay", "generator")
CAMPUSES = (9, 12)
SIZE = 400
LADDER = (1.0, 0.75, 0.5, 0.35, 0.25)
BETAS = (0.5, 1.0)
N_PER_CELL = 40
TIE_TOL = 1.0
MAX_WORKERS = 8

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_cont", "crew-starvation")

_OVERLAYS = {}


def _overlay(beta):
    o = _OVERLAYS.get(beta)
    if o is None:
        o = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                        master_seed=MASTER_SEED, sigma_s=SIGMA_S))
        _OVERLAYS[beta] = o
    return o


# --------------------------------------------------------------------------- #
# True-objective scorer (copied from true_objective.py, but d* not recorded due).
# --------------------------------------------------------------------------- #
def score_true_dstar(instance, schedule, wstar, dstar):
    """TWT* = sum w*(c*) * max(0, end - d*).  Feasibility from the independent
    validator.  Also returns #orders finishing after window_bh."""
    base = validate(instance, schedule)
    wo_by = {w["id"]: w for w in instance.get("work_orders", []) or []}
    win = float(instance["meta"]["window_bh"])
    twt = 0.0
    incomplete = 0
    for a in schedule.get("assignments", []) or []:
        wid = a.get("wo")
        end = a.get("end_bh")
        if wid not in wo_by or end is None:
            continue
        end = float(end)
        twt += wstar[wid] * max(0.0, end - dstar[wid])
        if end > win:
            incomplete += 1
    return {"feasible": bool(base["feasible"]), "TWT_star": twt,
            "incomplete_at_horizon": incomplete}


# --------------------------------------------------------------------------- #
# Utilization.
# --------------------------------------------------------------------------- #
def utilization(instance):
    win = float(instance["meta"]["window_bh"])
    p_by = defaultdict(float)
    for w in instance["work_orders"]:
        p_by[w["trade"]] += float(w["p_bh"])
    c_by = Counter(t["trade"] for t in instance["technicians"])
    tot_p = sum(p_by.values())
    tot_c = len(instance["technicians"])
    pooled = tot_p / (tot_c * win) if tot_c * win > 0 else float("inf")
    worst = 0.0
    for tr, pp in p_by.items():
        nc = c_by.get(tr, 0)
        u = pp / (nc * win) if nc > 0 else float("inf")
        worst = max(worst, u)
    return pooled, worst


# --------------------------------------------------------------------------- #
# ORACLE deciders (locked files untouched).
# --------------------------------------------------------------------------- #
def _oracle_atc_schedule(inst_run, orig_instance, applied, dstar):
    """ATC computed with true w* and true d*, via the supervisor preferred-pick
    path.  sup.wstar already carries w*; we override sup.due -> d*."""
    sup = Supervisor(_overlay_dummy(), orig_instance, rho=0.0, applied=applied)
    sup.due = dict(dstar)                       # <-- inject true deadline d*
    return dec.run_oracle_greedy(DispatchEnv(inst_run), sup, seed=SEED)


def _overlay_dummy():
    # Supervisor stores overlay only for the (unused-in-oracle) review path;
    # any bound overlay works.  Reuse the beta-0.5 overlay object.
    return _overlay(0.5)


def _oracle_edd_schedule(inst_run, dstar):
    """EDD computed with true d*: earliest true deadline first (tie: id)."""
    env = DispatchEnv(inst_run)

    def decider(queue, t, rng):
        pick = min(queue, key=lambda j: (dstar[j["id"]], j["id"]))
        return pick, pdrs._BIG_MARGIN

    sched, _ = env.run_supervised(decider, supervisor=None,
                                  method="oracle_edd", seed=SEED)
    return sched


# --------------------------------------------------------------------------- #
# One instance x one crew level -> both betas x {atc, edd} x {rule, oracle}.
# --------------------------------------------------------------------------- #
def _process(task):
    source, campus, m, path = task
    orig = json.load(open(path))
    inst_id = orig["meta"]["id"]
    inst_run = orig if m == 1.0 else tightness.scale_crew(orig, m)

    pooled_u, worst_u = utilization(inst_run)

    # RULE schedules are latent-independent (recorded fields) and beta-independent.
    s_rule_atc = dec.run_rule(DispatchEnv(inst_run), "atc", seed=SEED)
    s_rule_edd = dec.run_rule(DispatchEnv(inst_run), "edd", seed=SEED)

    recs = []
    for beta in BETAS:
        applied = _overlay(beta).apply(orig)     # latent on ORIGINAL instance
        wstar = applied["w_star"]
        cstar = applied["c_star"]
        dstar = {w["id"]: float(w["release_bh"]) + SLA[cstar[w["id"]]]
                 for w in orig["work_orders"]}

        r_atc = score_true_dstar(inst_run, s_rule_atc, wstar, dstar)
        r_edd = score_true_dstar(inst_run, s_rule_edd, wstar, dstar)

        s_or_atc = _oracle_atc_schedule(inst_run, orig, applied, dstar)
        s_or_edd = _oracle_edd_schedule(inst_run, dstar)
        o_atc = score_true_dstar(inst_run, s_or_atc, wstar, dstar)
        o_edd = score_true_dstar(inst_run, s_or_edd, wstar, dstar)

        recs.append({
            "source": source, "campus": campus, "crew_m": m, "beta": beta,
            "inst_id": inst_id, "window_bh": float(orig["meta"]["window_bh"]),
            "pooled_u": pooled_u, "worst_u": worst_u,
            "twt_rule_atc": r_atc["TWT_star"], "twt_oracle_atc": o_atc["TWT_star"],
            "twt_rule_edd": r_edd["TWT_star"], "twt_oracle_edd": o_edd["TWT_star"],
            "feas_rule_atc": int(r_atc["feasible"]),
            "feas_oracle_atc": int(o_atc["feasible"]),
            "feas_rule_edd": int(r_edd["feasible"]),
            "feas_oracle_edd": int(o_edd["feasible"]),
            "incomplete_rule_atc": r_atc["incomplete_at_horizon"],
        })
    return recs


def _tasks():
    out = []
    for source in SOURCES:
        for campus in CAMPUSES:
            d = os.path.join(_INST, "c%02d" % campus, source, str(SIZE))
            files = sorted(glob.glob(os.path.join(d, "*.json")))[:N_PER_CELL]
            for m in LADDER:
                for p in files:
                    out.append((source, campus, m, p))
    return out


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #
def _cell_stats(records, rule):
    """Aggregate a list of per-instance records for one rule ('atc'|'edd')."""
    n = len(records)
    rk, ok = "twt_rule_%s" % rule, "twt_oracle_%s" % rule
    sum_r = sum(r[rk] for r in records)
    sum_o = sum(r[ok] for r in records)
    mean_r = sum_r / n if n else 0.0
    mean_o = sum_o / n if n else 0.0
    headroom = (100.0 * (mean_r - mean_o) / mean_r) if mean_r > 1e-9 else 0.0
    wins = ties = losses = 0
    for r in records:
        diff = r[rk] - r[ok]                     # positive => oracle better
        if abs(diff) <= TIE_TOL:
            ties += 1
        elif diff > TIE_TOL:
            wins += 1
        else:
            losses += 1
    pooled_u = float(np.median([r["pooled_u"] for r in records])) if n else 0.0
    worst_u = float(np.median([r["worst_u"] for r in records])) if n else 0.0
    incomplete = float(np.mean([r["incomplete_rule_atc"] for r in records])) if n else 0.0
    infeas = sum(1 for r in records
                 if not (r["feas_rule_%s" % rule] and r["feas_oracle_%s" % rule]))
    return dict(n=n, mean_rule=mean_r, mean_oracle=mean_o, pct_headroom=headroom,
                wins=wins, ties=ties, losses=losses, pooled_u=pooled_u,
                worst_u=worst_u, incomplete=incomplete, infeasible=infeas)


def main():
    os.makedirs(_OUT, exist_ok=True)
    tasks = _tasks()
    print("crew-starvation: %d instance-tasks (%d cells x betas)"
          % (len(tasks), len(SOURCES) * len(CAMPUSES) * len(LADDER)))

    records = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        done = 0
        for res in ex.map(_process, tasks, chunksize=4):
            records.extend(res)
            done += 1
            if done % 200 == 0:
                print("  processed %d/%d instance-tasks" % (done, len(tasks)))

    # dump raw per-instance records
    raw_csv = os.path.join(_OUT, "records.csv")
    cols = ["source", "campus", "crew_m", "beta", "inst_id", "window_bh",
            "pooled_u", "worst_u", "twt_rule_atc", "twt_oracle_atc",
            "twt_rule_edd", "twt_oracle_edd", "feas_rule_atc", "feas_oracle_atc",
            "feas_rule_edd", "feas_oracle_edd", "incomplete_rule_atc"]
    with open(raw_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({c: r[c] for c in cols})
    print("wrote %s (%d rows)" % (raw_csv, len(records)))

    # per-cell aggregation (source, campus, crew_m, beta) for atc + edd
    cells = []
    for source in SOURCES:
        for campus in CAMPUSES:
            for m in LADDER:
                for beta in BETAS:
                    sub = [r for r in records if r["source"] == source
                           and r["campus"] == campus and r["crew_m"] == m
                           and r["beta"] == beta]
                    if not sub:
                        continue
                    for rule in ("atc", "edd"):
                        st = _cell_stats(sub, rule)
                        cells.append(dict(source=source, campus=campus,
                                          crew_m=m, beta=beta, rule=rule, **st))

    cell_csv = os.path.join(_OUT, "cells.csv")
    ccols = ["source", "campus", "crew_m", "beta", "rule", "n", "pooled_u",
             "worst_u", "mean_rule", "mean_oracle", "pct_headroom", "wins",
             "ties", "losses", "incomplete", "infeasible"]
    with open(cell_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ccols)
        w.writeheader()
        for c in cells:
            out = dict(c)
            for k in ("pooled_u", "worst_u", "mean_rule", "mean_oracle",
                      "pct_headroom", "incomplete"):
                out[k] = "%.4f" % out[k]
            w.writerow({k: out[k] for k in ccols})
    print("wrote %s (%d cells)" % (cell_csv, len(cells)))

    with open(os.path.join(_OUT, "cells.json"), "w") as fh:
        json.dump(cells, fh, indent=1)

    # console summary: ATC headline, per (source, campus, crew_m), both betas
    print("\n=== ATC headline: pct_headroom by crew level ===")
    print("%-9s c%-3s %-6s | %-7s %-7s | b=0.5 head%%  W/T/L | b=1.0 head%%  W/T/L"
          % ("source", "", "m", "pooled_u", "worst_u"))
    atc = [c for c in cells if c["rule"] == "atc"]
    for source in SOURCES:
        for campus in CAMPUSES:
            for m in LADDER:
                b05 = next((c for c in atc if c["source"] == source
                            and c["campus"] == campus and c["crew_m"] == m
                            and c["beta"] == 0.5), None)
                b10 = next((c for c in atc if c["source"] == source
                            and c["campus"] == campus and c["crew_m"] == m
                            and c["beta"] == 1.0), None)
                if not b05:
                    continue
                print("%-9s c%02d  %-6.2f | %-7.2f %-7.2f | %+7.2f  %d/%d/%d | %+7.2f  %d/%d/%d"
                      % (source, campus, m, b05["pooled_u"], b05["worst_u"],
                         b05["pct_headroom"], b05["wins"], b05["ties"], b05["losses"],
                         b10["pct_headroom"], b10["wins"], b10["ties"], b10["losses"]))
    print("\ndone.")


if __name__ == "__main__":
    main()
