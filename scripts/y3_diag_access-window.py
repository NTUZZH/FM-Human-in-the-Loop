"""Y3 diagnostic: the ACCESS-WINDOW latent channel (proposal secondary channel).

Question. The weight-only latent (PIVOTAL FINDING, notes/decisions.md) has ~0
decision leverage: reweighting the cost of an already-near-optimal tardy set does
not change the best schedule. The access-window channel is a DIFFERENT kind of
latent: a per-building timing/feasibility constraint the dispatcher CONTROLS.
Each restricted building is enterable only on weekday mornings (business-hour of
day < 4). Dispatching a job in a restricted building at an afternoon start incurs
a FIXED true-objective penalty (Appendix B: weight-8 x 8 bh = 64 units;
overlay.ACCESS_PENALTY). The system does not see windows; the supervisor does.

Objective for THIS variant (task brief):
    TWT*_access = recorded TWT  +  sum of access penalties for window violations
Both the RULE and the ORACLE are scored on it. The tardiness term uses RECORDED
weights (so this variant is decoupled from beta / the weight channel: the
restricted set depends only on (master_seed, instance_id, building, alpha), never
on beta). We therefore report at a single nominal beta and flag beta-invariance.

Deciders.
* RULE          -- plain ATC on the recorded field, run in the LOCKED non-delay
                   env (dec.run_rule). Ignores windows -> pays penalties.
* ORACLE-REORDER-- window-aware ATC run in the SAME locked non-delay env
                   (env.run_policy custom pick): in a morning window prefer
                   restricted-building jobs (use the open window); in an
                   afternoon window prefer non-restricted jobs (leave restricted
                   in queue). Faithful to the benchmark's non-delay dispatch, but
                   it can only REORDER, never idle.
* ORACLE-DEFER  -- window-as-hard-constraint dispatcher (this file, a from-scratch
                   copy of the pdrs event loop): a restricted job may START only
                   in a morning window; when a technician is free in an afternoon
                   and only restricted jobs remain for its trade, the technician
                   IDLES until the next morning boundary (deferral / idle
                   insertion). Access penalty is 0 by construction; the only cost
                   is any recorded tardiness the idle time induces. This is the
                   channel's true information CEILING. It departs from non-delay
                   dispatch (idle insertion), which the locked env cannot express.

Scoring reuses the INDEPENDENT validator (fmwos.validator.validate) for
feasibility and overlay.access_penalty for the penalty; the recorded-TWT block is
copied here (true_objective.py is not edited).

STRUCTURAL NOTE (verified in main): the `building` field is populated ONLY on
REPLAY instances of campuses c05 and c10 (the over-resourced campuses). It is
None on every synthetic track (generator/storm/storm2/pmmix) and on c09/c12
replay. So the access channel is INERT on the task's specified loaded cells
(c09/c12 storm/pmmix) and everywhere else with no buildings; it can only fire on
c05/c10 replay. Both facts are reported.

Run:  PYTHONPATH=src nice python scripts/y3_diag_access-window.py
Out:  results/y3_diag/access-window/{coverage.csv,headroom.csv,summary.json}
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import glob
import heapq
import itertools
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.env import DispatchEnv                 # noqa: E402
from fmwos.hitl import deciders as dec            # noqa: E402
from fmwos.hitl import overlay as ov              # noqa: E402
from fmwos import validator as _validator         # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_diag", "access-window")
SEED = 301
MASTER_SEED = 12345
FAMILY = "F-NL"
BETA = 1.0                      # nominal; access channel is beta-independent
ALPHAS = (0.1, 0.2)
TIE_TOL = 1.0                   # tie band on the per-instance TWT*_access diff
N_PER_CELL = 100
MAX_WORKERS = 8
_DAY_BH = 8.0
_MORNING_END = 4.0              # matches overlay._violates_window / restricted_buildings

# Cells that CAN carry the channel (buildings present): c05/c10 replay.
BUILDINGED_CELLS = [(5, "replay", 150), (5, "replay", 400),
                    (10, "replay", 150), (10, "replay", 400)]
# Task's specified loaded cells (no buildings -> inert; reported as a control).
LOADED_CELLS = [(9, "storm", 150), (12, "storm", 150),
                (9, "pmmix", 150), (12, "pmmix", 150)]


# --------------------------------------------------------------------------- #
# Scoring (copied from true_objective.py; recorded-TWT + access penalty)       #
# --------------------------------------------------------------------------- #
def recorded_twt(instance, schedule):
    wo_by_id = {w["id"]: w for w in instance.get("work_orders", []) or []}
    twt = 0.0
    for a in schedule.get("assignments", []) or []:
        wo = wo_by_id.get(a.get("wo"))
        end = a.get("end_bh")
        if wo is None or end is None:
            continue
        twt += float(wo["weight"]) * max(0.0, float(end) - float(wo["due_bh"]))
    return twt


def score_access(instance, schedule, overlay):
    """TWT*_access = recorded TWT + access penalty. Also returns feasibility."""
    base = _validator.validate(instance, schedule)
    twt = recorded_twt(instance, schedule)
    pen = overlay.access_penalty(instance, schedule)
    return {"feasible": base["feasible"], "rec_twt": twt, "access": pen,
            "twt_access": twt + pen}


# --------------------------------------------------------------------------- #
# ATC helper (recorded weights), shared by reorder-pick and defer dispatcher.  #
# Mirrors pdrs._pick_atc exactly: pbar over the full trade queue, k=2, id tie.  #
# --------------------------------------------------------------------------- #
def _atc_argmin(cands, full_queue, now):
    pbar = sum(j["p_bh"] for j in full_queue) / len(full_queue)
    denom = 2.0 * pbar

    def key(j):
        slack = max(0.0, j["due_bh"] - now - j["p_bh"])
        score = (j["weight"] / j["p_bh"]) * math.exp(-slack / denom)
        return (-score, j["id"])

    return min(cands, key=key)


# --------------------------------------------------------------------------- #
# ORACLE-REORDER: window-aware ATC pick for env.run_policy (non-delay env).     #
# --------------------------------------------------------------------------- #
def make_reorder_pick(restricted_ids):
    def pick(queue, t, rng):
        morning = (t % _DAY_BH) < _MORNING_END
        if morning:                 # use the open window: restricted jobs first
            cands = [j for j in queue if j["id"] in restricted_ids] or queue
        else:                       # protect restricted: serve non-restricted now
            cands = [j for j in queue if j["id"] not in restricted_ids] or queue
        return _atc_argmin(cands, queue, t)
    return pick


# --------------------------------------------------------------------------- #
# ORACLE-DEFER: window-as-hard-constraint dispatcher (idle insertion).         #
# From-scratch copy of the pdrs event loop; reduces EXACTLY to pdrs.dispatch    #
# (== env ATC) when restricted_ids is empty (asserted in the smoke).           #
# --------------------------------------------------------------------------- #
_FREE, _RELEASE = 0, 1


def dispatch_defer(instance, restricted_ids, seed=0):
    queue = defaultdict(list)
    idle = defaultdict(list)
    counter = itertools.count()
    events = []
    for tech in instance["technicians"]:
        heapq.heappush(events, (0.0, next(counter), _FREE, tech["id"], tech["trade"]))
    for wo in instance["work_orders"]:
        heapq.heappush(events, (float(wo["release_bh"]), next(counter), _RELEASE, wo))

    assignments = []

    def next_morning(now):
        day = math.floor(now / _DAY_BH)
        return (day + 1) * _DAY_BH          # start of next day = next morning window

    def try_dispatch(trade, now):
        q = queue[trade]
        free = idle[trade]
        while free and q:
            morning = (now % _DAY_BH) < _MORNING_END
            if morning:
                restr = [j for j in q if j["id"] in restricted_ids]
                cands = restr if restr else q
            else:
                nonr = [j for j in q if j["id"] not in restricted_ids]
                if nonr:
                    cands = nonr
                else:
                    # only restricted jobs remain in an afternoon window: DEFER.
                    m = next_morning(now)
                    while free:
                        tid = heapq.heappop(free)
                        heapq.heappush(events, (m, next(counter), _FREE, tid, trade))
                    return
            job = _atc_argmin(cands, q, now)
            q.remove(job)
            tid = heapq.heappop(free)
            start = float(now)
            end = start + float(job["p_bh"])
            assignments.append({"wo": job["id"], "tech": tid,
                                "start_bh": start, "end_bh": end})
            heapq.heappush(events, (end, next(counter), _FREE, tid, trade))

    while events:
        now = events[0][0]
        touched = set()
        while events and events[0][0] == now:
            _, _, kind, *payload = heapq.heappop(events)
            if kind == _FREE:
                tid, trade = payload
                heapq.heappush(idle[trade], tid)
                touched.add(trade)
            else:
                wo = payload[0]
                queue[wo["trade"]].append(wo)
                touched.add(wo["trade"])
        for trade in sorted(touched):
            try_dispatch(trade, now)

    return {"instance_id": instance["meta"]["id"], "method": "defer_oracle",
            "seed": seed, "wall_seconds": 0.0, "decisions": len(assignments),
            "assignments": assignments}


# --------------------------------------------------------------------------- #
# Per-instance work                                                            #
# --------------------------------------------------------------------------- #
def _restricted_wo_ids(instance, overlay):
    restr_b = overlay.restricted_buildings(instance)
    return {w["id"] for w in instance["work_orders"]
            if w.get("building") in restr_b}, len(restr_b)


def _process_instance(args):
    campus, track, size, path = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]
    n_bldg = len({w.get("building") for w in inst["work_orders"]
                  if w.get("building") is not None})

    # RULE schedule (plain ATC, locked non-delay env) is alpha-independent.
    s_rule = dec.run_rule(DispatchEnv(inst), "atc", seed=SEED)

    out = []
    for alpha in ALPHAS:
        overlay = ov.Overlay(ov.OverlayParams(beta=BETA, family=FAMILY,
                                              master_seed=MASTER_SEED,
                                              access_alpha=alpha))
        restr_ids, n_restr_b = _restricted_wo_ids(inst, overlay)

        rule = score_access(inst, s_rule, overlay)

        s_reorder = DispatchEnv(inst).run_policy(make_reorder_pick(restr_ids),
                                                 method="reorder", seed=SEED)
        reorder = score_access(inst, s_reorder, overlay)

        s_defer = dispatch_defer(inst, restr_ids, seed=SEED)
        defer = score_access(inst, s_defer, overlay)

        out.append({
            "campus": campus, "track": track, "size": size, "alpha": alpha,
            "inst_id": inst_id, "n_bldg": n_bldg, "n_restr_bldg": n_restr_b,
            "n_restr_jobs": len(restr_ids),
            "rule_rec": rule["rec_twt"], "rule_access": rule["access"],
            "rule_twt": rule["twt_access"], "rule_feas": rule["feasible"],
            "reorder_twt": reorder["twt_access"], "reorder_access": reorder["access"],
            "reorder_feas": reorder["feasible"],
            "defer_twt": defer["twt_access"], "defer_rec": defer["rec_twt"],
            "defer_access": defer["access"], "defer_feas": defer["feasible"],
        })
    return out


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #
def _headroom_stats(records, oracle_key):
    """ORACLE vs RULE on TWT*_access. Positive gap => oracle better (lower)."""
    n = len(records)
    if not n:
        return None
    sum_rule = sum(r["rule_twt"] for r in records)
    sum_or = sum(r[oracle_key] for r in records)
    mean_rule = sum_rule / n
    mean_or = sum_or / n
    abs_gap = mean_rule - mean_or
    pct = (100.0 * abs_gap / mean_rule) if mean_rule > 1e-9 else 0.0
    wins = ties = losses = 0
    for r in records:
        diff = r["rule_twt"] - r[oracle_key]       # positive => oracle better
        if abs(diff) <= TIE_TOL:
            ties += 1
        elif diff > TIE_TOL:
            wins += 1
        else:
            losses += 1
    return {"n": n, "mean_rule": mean_rule, "mean_oracle": mean_or,
            "abs_gap": abs_gap, "pct": pct,
            "wins": wins, "ties": ties, "losses": losses}


def main():
    # ------------------------------------------------------------------ #
    # (0) building-coverage map (structural finding)                     #
    # ------------------------------------------------------------------ #
    cov_rows = []
    for c in (5, 9, 10, 12):
        for track in ("replay", "generator", "storm", "storm2", "pmmix"):
            for size in ("150", "400", "w80"):
                d = os.path.join(_INST, "c%02d" % c, track, size)
                if not os.path.isdir(d):
                    continue
                files = sorted(glob.glob(os.path.join(d, "*.json")))
                if not files:
                    continue
                inst = json.load(open(files[0]))
                nb = len({w.get("building") for w in inst["work_orders"]
                          if w.get("building") is not None})
                cov_rows.append({"campus": c, "track": track, "size": size,
                                 "n_files": len(files), "distinct_bldg": nb})
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "coverage.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["campus", "track", "size",
                                           "n_files", "distinct_bldg"])
        w.writeheader()
        w.writerows(cov_rows)
    buildinged = [r for r in cov_rows if r["distinct_bldg"] > 0]
    print("building coverage: %d/%d cells carry buildings:" %
          (len(buildinged), len(cov_rows)))
    for r in buildinged:
        print("  c%02d/%s/%s : %d buildings" %
              (r["campus"], r["track"], r["size"], r["distinct_bldg"]))

    # ------------------------------------------------------------------ #
    # (1) enumerate tasks: buildinged cells (fire) + loaded cells (inert) #
    # ------------------------------------------------------------------ #
    tasks = []
    for campus, track, size in BUILDINGED_CELLS + LOADED_CELLS:
        d = os.path.join(_INST, "c%02d" % campus, track, str(size))
        files = sorted(glob.glob(os.path.join(d, "*.json")))[:N_PER_CELL]
        for p in files:
            tasks.append((campus, track, size, p))
    print("total instance-tasks: %d (x%d alphas)" % (len(tasks), len(ALPHAS)))

    records = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        done = 0
        for res in ex.map(_process_instance, tasks, chunksize=4):
            records.extend(res)
            done += 1
            if done % 200 == 0:
                print("  processed %d/%d instances" % (done, len(tasks)))

    # infeasibility guard (should never trip)
    infeas = [r for r in records if not (r["rule_feas"] and r["reorder_feas"]
                                         and r["defer_feas"])]
    if infeas:
        print("WARNING: %d infeasible schedules!" % len(infeas))

    # ------------------------------------------------------------------ #
    # (2) per-cell headroom table                                        #
    # ------------------------------------------------------------------ #
    cells = sorted({(r["campus"], r["track"], r["size"]) for r in records})
    rows = []
    for (campus, track, size) in cells:
        for alpha in ALPHAS:
            sub = [r for r in records if r["campus"] == campus
                   and r["track"] == track and r["size"] == size
                   and r["alpha"] == alpha]
            if not sub:
                continue
            reo = _headroom_stats(sub, "reorder_twt")
            dfr = _headroom_stats(sub, "defer_twt")
            mean_rule_access = sum(r["rule_access"] for r in sub) / len(sub)
            mean_rule_rec = sum(r["rule_rec"] for r in sub) / len(sub)
            mean_defer_rec = sum(r["defer_rec"] for r in sub) / len(sub)
            mean_defer_access = sum(r["defer_access"] for r in sub) / len(sub)
            mean_restr_jobs = sum(r["n_restr_jobs"] for r in sub) / len(sub)
            rows.append({
                "campus": campus, "track": track, "size": size, "alpha": alpha,
                "n": reo["n"],
                "mean_restr_jobs": round(mean_restr_jobs, 2),
                "mean_rule_rec_twt": round(mean_rule_rec, 3),
                "mean_rule_access": round(mean_rule_access, 2),
                "mean_rule_twt_access": round(reo["mean_rule"], 3),
                # reorder oracle
                "reorder_mean_twt": round(reo["mean_oracle"], 3),
                "reorder_pct_headroom": round(reo["pct"], 3),
                "reorder_W_T_L": "%d/%d/%d" % (reo["wins"], reo["ties"], reo["losses"]),
                # defer oracle (ceiling)
                "defer_mean_twt": round(dfr["mean_oracle"], 3),
                "defer_mean_rec_twt": round(mean_defer_rec, 3),
                "defer_mean_access": round(mean_defer_access, 2),
                "defer_pct_headroom": round(dfr["pct"], 3),
                "defer_W_T_L": "%d/%d/%d" % (dfr["wins"], dfr["ties"], dfr["losses"]),
            })
            print("c%02d/%s/%s a%.1f n=%d | RULE twt*=%.1f (rec %.2f + acc %.1f) "
                  "| REORDER %.1f (%.2f%%) | DEFER %.1f (%.2f%%, rec %.2f acc %.1f) W/T/L %s"
                  % (campus, track, size, alpha, reo["n"], reo["mean_rule"],
                     mean_rule_rec, mean_rule_access, reo["mean_oracle"], reo["pct"],
                     dfr["mean_oracle"], dfr["pct"], mean_defer_rec, mean_defer_access,
                     "%d/%d/%d" % (dfr["wins"], dfr["ties"], dfr["losses"])))

    with open(os.path.join(_OUT, "headroom.csv"), "w", newline="") as fh:
        cols = ["campus", "track", "size", "alpha", "n", "mean_restr_jobs",
                "mean_rule_rec_twt", "mean_rule_access", "mean_rule_twt_access",
                "reorder_mean_twt", "reorder_pct_headroom", "reorder_W_T_L",
                "defer_mean_twt", "defer_mean_rec_twt", "defer_mean_access",
                "defer_pct_headroom", "defer_W_T_L"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # per-instance dump for auditing
    with open(os.path.join(_OUT, "per_instance.csv"), "w", newline="") as fh:
        cols = list(records[0].keys())
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(records)

    max_defer = max((r["defer_pct_headroom"] for r in rows
                     if r["mean_rule_twt_access"] > 1e-9), default=0.0)
    max_reorder = max((r["reorder_pct_headroom"] for r in rows
                       if r["mean_rule_twt_access"] > 1e-9), default=0.0)
    summary = {
        "variant": "access-window",
        "beta_independent": True,
        "buildinged_cells": [{"campus": r["campus"], "track": r["track"],
                              "size": r["size"], "distinct_bldg": r["distinct_bldg"]}
                             for r in buildinged],
        "max_defer_pct_headroom": max_defer,
        "max_reorder_pct_headroom": max_reorder,
        "n_cells": len(rows),
        "rows": rows,
    }
    with open(os.path.join(_OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote %s" % os.path.join(_OUT, "headroom.csv"))
    print("MAX defer %% headroom = %.2f ; MAX reorder %% headroom = %.2f"
          % (max_defer, max_reorder))


if __name__ == "__main__":
    main()
