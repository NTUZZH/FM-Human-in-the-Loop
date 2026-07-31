"""Y3 continuation: FULL-CLASS-SHIFT ORACLE-vs-RULE on the STORM arrival track.

Regime STORM (arrival-multiplier knob). This scan tests the DEADLINE channel of
the full-class-shift latent -- the piece the prior weight-only scan
(results/y3_p1/headroom_scan2.csv) never touched -- at BOTH size 150 and size
400, on the tighter crew (crew_multiplier 0.8 -> id tail c80), across the four
storm arrival multipliers a125/a150/a200/a300 (arrival_multiplier 1.25/1.5/2/3).

Operationalization (verified against src/fmwos/hitl/overlay.py and a sample
instance; identical wiring to scripts/y3_diag_full-class-shift.py, which is the
author-blessed reference for this fix):

    s_j   = clip(round(sigma_s * xi_j), -2, +2)      (overlay F-NL, seed 12345)
    c*_j  = clip(c_j - s_j, 1, 4)                     (positive shift => urgent)
    w*_j  = w(c*_j)          with w   = 8/4/2/1  for class 1..4
    d*_j  = r_j + SLA(c*_j)  with SLA = 8/24/80/171.4 bh for class 1..4

    TRUE OBJECTIVE   TWT* = sum_j w*_j * max(0, C_j - d*_j)   (uses d*, NOT d)

    RULE   R  = R computed on the RECORDED fields (recorded w, recorded d);
                the deployed dispatcher never sees the latent.
    ORACLE R  = R computed with the TRUE class (true w*, true d*); the
                full-information myopic ceiling.

    headroom = (TWT*_RULE - TWT*_ORACLE) / TWT*_RULE     (per cell, per beta)

R in {ATC (headline), EDD}. The ATC ORACLE is the true-weight/true-deadline ATC
score (the supervisor's preferred-pick logic, but with the due-date term reading
d* -- deciders.py only injects w*, so d* is added here in the oracle decider, not
in the locked file). A weight-only ORACLE (w*, recorded d) is also scored for
ATC, to isolate how much the DEADLINE moving adds on top of the (near-dead)
weight channel.

CONTENTION. Per instance we report utilization over the arrival window
    u = sum_trade p_bh / (crew_of_trade * window_bh)
pooled (all trades) and worst-trade, plus makespan and the count of orders
completing after the arrival window (realized backlog). storm is a fixed-N
TRANSIENT burst (arrival_multiplier compresses the arrival window; total work is
~constant), so it is NOT permanent overload -- the honest label is a busy /
bottleneck-saturated team during the burst, not an infinite-backlog overload.

Outputs (results/y3_cont/storm-arrival/):
  per_instance.csv, cells.csv, summary.json

Run:  PYTHONPATH=src OMP_NUM_THREADS=1 nice python scripts/y3_cont_storm-arrival.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import glob
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import validator as _validator            # noqa: E402
from fmwos.env import DispatchEnv                     # noqa: E402
from fmwos.hitl import deciders as dec                # noqa: E402
from fmwos.hitl import overlay as ov                  # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_cont", "storm-arrival")
os.makedirs(_OUT, exist_ok=True)

SEED = 301
MASTER_SEED = 12345
FAMILY = "F-NL"
BETAS = (0.25, 0.5, 1.0)
CAMPUSES = (9, 12, 5, 10)          # {9,12} loaded/headline; {5,10} sign check
LOADED = (9, 12)
SIGN = (5, 10)
SIZES = (150, 400)
ARRIVALS = ("a125", "a150", "a200", "a300")   # arrival_multiplier 1.25/1.5/2/3
CREW_TAG = "c80"                    # crew_multiplier 0.8 (the tighter crew)
RULES = ("atc", "edd")
TIE_TOL = 1.0                      # tie band on the per-instance TWT* diff
ATC_K = 2.0
_BIG = 1e9
N_PER_CELL = 60                    # up to 60; only 30 exist per cell on disk

# SLA(class 1..4) in business-hours (Appendix B; VERIFIED d_j = r_j + SLA(c_j)).
SLA_OF_CLASS = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}
W_OF_CLASS = dict(ov.W_OF_CLASS)   # 8/4/2/1

_OVERLAYS = {}


def _overlay(beta):
    o = _OVERLAYS.get(beta)
    if o is None:
        o = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                        master_seed=MASTER_SEED))
        _OVERLAYS[beta] = o
    return o


# --------------------------------------------------------------------------- #
# Full-class-shift maps and TWT* scorer (mirrors y3_diag_full-class-shift.py). #
# --------------------------------------------------------------------------- #
def _fullshift_maps(inst, applied):
    cstar = applied["c_star"]
    wstar = dict(applied["w_star"])
    dstar = {}
    for wo in inst["work_orders"]:
        wid = wo["id"]
        dstar[wid] = float(wo["release_bh"]) + SLA_OF_CLASS[int(cstar[wid])]
    return wstar, dstar


def score_twt(instance, schedule, wstar, due_map):
    """TWT* = sum_j w*_j * max(0, C_j - due_j). due_map is d* (full-class-shift)
    or the recorded due (weight-only reference)."""
    wo_by_id = {wo["id"]: wo for wo in instance.get("work_orders", []) or []}
    twt = 0.0
    for a in schedule.get("assignments", []) or []:
        wid = a.get("wo")
        wo = wo_by_id.get(wid)
        end = a.get("end_bh")
        if wo is None or end is None:
            continue
        twt += wstar[wid] * max(0.0, float(end) - due_map[wid])
    return twt


# --------------------------------------------------------------------------- #
# ORACLE deciders: the rule computed with the TRUE class (w*, d*).            #
# --------------------------------------------------------------------------- #
def make_oracle_atc(wstar, due_map, k=ATC_K):
    def decider(queue, t, rng):
        pbar = sum(j["p_bh"] for j in queue) / len(queue)
        denom = k * pbar

        def key(j):
            jid = j["id"]
            slack = max(0.0, due_map[jid] - t - j["p_bh"])
            score = (wstar[jid] / j["p_bh"]) * math.exp(-slack / denom)
            return (-score, jid)
        return min(queue, key=key), _BIG
    return decider


def make_oracle_edd(due_map):
    def decider(queue, t, rng):
        return min(queue, key=lambda j: (due_map[j["id"]], j["id"])), _BIG
    return decider


# --------------------------------------------------------------------------- #
# Contention: utilization over the arrival window + realized backlog.          #
# --------------------------------------------------------------------------- #
def _contention(inst, rule_sched):
    wos = inst["work_orders"]
    techs = inst["technicians"]
    win = float(inst["meta"]["window_bh"])
    crew = defaultdict(int)
    for tech in techs:
        crew[tech["trade"]] += 1
    proc = defaultdict(float)
    for w in wos:
        proc[w["trade"]] += float(w["p_bh"])
    tot_p = sum(proc.values())
    tot_crew = len(techs)
    pooled = tot_p / (tot_crew * win) if (tot_crew and win > 0) else 0.0
    worst = 0.0
    worst_tr = None
    for tr, p in proc.items():
        k = crew.get(tr, 0)
        if k > 0 and win > 0:
            u = p / (k * win)
            if u > worst:
                worst, worst_tr = u, tr
    # realized backlog: orders completing after the arrival window, and makespan.
    ends = [float(a["end_bh"]) for a in rule_sched["assignments"]]
    makespan = max(ends) if ends else 0.0
    incomplete = sum(1 for e in ends if e > win + 1e-9)
    return dict(util_pooled=pooled, util_worst=worst, worst_trade=worst_tr,
                window_bh=win, makespan=makespan, incomplete_at_window=incomplete,
                n_wos=len(wos))


# --------------------------------------------------------------------------- #
# One instance -> per-(beta, rule) records.                                    #
# --------------------------------------------------------------------------- #
def _process(args):
    campus, size, arrival, path = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]
    rec_due = {w["id"]: float(w["due_bh"]) for w in inst["work_orders"]}

    # RULE schedules are beta-independent (recorded w, d only).
    s_rule = {r: dec.run_rule(DispatchEnv(inst), r, seed=SEED) for r in RULES}
    feas_rule = {r: _validator.validate(inst, s_rule[r])["feasible"] for r in RULES}

    cont = _contention(inst, s_rule["atc"])

    out = []
    for beta in BETAS:
        ovl = _overlay(beta)
        applied = ovl.apply(inst)
        wstar, dstar = _fullshift_maps(inst, applied)
        n_shift = sum(1 for v in applied["shift"].values() if v != 0)

        # ORACLE schedules (rule with the true class); full-class-shift uses d*.
        s_or_atc, _ = DispatchEnv(inst).run_supervised(
            make_oracle_atc(wstar, dstar), supervisor=None,
            method="oracle_atc", seed=SEED)
        s_or_edd, _ = DispatchEnv(inst).run_supervised(
            make_oracle_edd(dstar), supervisor=None,
            method="oracle_edd", seed=SEED)
        # weight-only ORACLE (w*, recorded d) for the ATC rule -- isolates the
        # deadline channel's marginal contribution vs the prior weight-only scan.
        s_or_atc_w, _ = DispatchEnv(inst).run_supervised(
            make_oracle_atc(wstar, rec_due), supervisor=None,
            method="oracle_atc_wonly", seed=SEED)

        feas_or = (_validator.validate(inst, s_or_atc)["feasible"]
                   and _validator.validate(inst, s_or_edd)["feasible"])

        for rule, s_or in (("atc", s_or_atc), ("edd", s_or_edd)):
            twt_rule = score_twt(inst, s_rule[rule], wstar, dstar)
            twt_or = score_twt(inst, s_or, wstar, dstar)
            rec = dict(campus=campus, size=size, arrival=arrival, rule=rule,
                       beta=beta, inst_id=inst_id, twt_rule=twt_rule,
                       twt_oracle=twt_or, n_shift=n_shift,
                       feasible=int(bool(feas_rule[rule] and feas_or)))
            if rule == "atc":
                # weight-only headroom on the weight-only objective (recorded d).
                rec["twt_rule_wonly"] = score_twt(inst, s_rule["atc"], wstar, rec_due)
                rec["twt_oracle_wonly"] = score_twt(inst, s_or_atc_w, wstar, rec_due)
            rec.update(cont)
            out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# Aggregation.                                                                 #
# --------------------------------------------------------------------------- #
def _stats(records, rk="twt_rule", ok="twt_oracle"):
    n = len(records)
    if not n:
        return None
    mr = sum(r[rk] for r in records) / n
    mo = sum(r[ok] for r in records) / n
    gap = mr - mo
    pct = (100.0 * gap / mr) if mr > 1e-9 else 0.0
    w = t = l = 0
    for r in records:
        d = r[rk] - r[ok]                    # positive => oracle better
        if abs(d) <= TIE_TOL:
            t += 1
        elif d > TIE_TOL:
            w += 1
        else:
            l += 1
    return dict(n=n, mean_twt_rule=mr, mean_twt_oracle=mo, abs_gap=gap,
                pct_headroom=pct, wins=w, ties=t, losses=l)


def _cell_contention(records):
    return dict(
        util_pooled=statistics.mean(r["util_pooled"] for r in records),
        util_worst=statistics.mean(r["util_worst"] for r in records),
        window_bh=statistics.mean(r["window_bh"] for r in records),
        makespan=statistics.mean(r["makespan"] for r in records),
        incomplete_at_window=statistics.mean(r["incomplete_at_window"] for r in records),
        n_wos=statistics.mean(r["n_wos"] for r in records),
    )


def main():
    tasks = []
    for campus in CAMPUSES:
        cdir = "c%02d" % campus
        for size in SIZES:
            for arrival in ARRIVALS:
                frag = "_%s_%s_" % (arrival, CREW_TAG)
                files = sorted(glob.glob(os.path.join(
                    _INST, cdir, "storm", str(size), "*%s*.json" % frag)))
                for p in files[:N_PER_CELL]:
                    tasks.append((campus, size, arrival, p))
    print("[storm-arrival] %d instance-tasks (x%d betas x%d rules)"
          % (len(tasks), len(BETAS), len(RULES)))

    records = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as ex:
        done = 0
        for res in ex.map(_process, tasks, chunksize=4):
            records.extend(res)
            done += 1
            if done % 150 == 0:
                print("  [storm-arrival] %d/%d instances (%.0fs)"
                      % (done, len(tasks), time.perf_counter() - t0))
    print("[storm-arrival] processed %d instances in %.1fs -> %d records"
          % (len(tasks), time.perf_counter() - t0, len(records)))

    # per-instance CSV
    pi_cols = ["campus", "size", "arrival", "rule", "beta", "inst_id",
               "twt_rule", "twt_oracle", "twt_rule_wonly", "twt_oracle_wonly",
               "n_shift", "feasible", "util_pooled", "util_worst", "worst_trade",
               "window_bh", "makespan", "incomplete_at_window", "n_wos"]
    with open(os.path.join(_OUT, "per_instance.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=pi_cols, extrasaction="ignore")
        wtr.writeheader()
        for r in records:
            row = dict(r)
            for k in ("twt_rule", "twt_oracle", "twt_rule_wonly",
                      "twt_oracle_wonly"):
                if k in row:
                    row[k] = "%.4f" % row[k]
            for k in ("util_pooled", "util_worst"):
                row[k] = "%.4f" % row[k]
            for k in ("window_bh", "makespan"):
                row[k] = "%.3f" % row[k]
            wtr.writerow(row)

    # aggregate cells: per-campus + pooled-loaded (9+12) + pooled-sign (5+10),
    # per (size, arrival, rule, beta).
    cell_rows = []
    scopes = [("c%d" % c, (c,)) for c in CAMPUSES] + \
             [("9+12", LOADED), ("5+10", SIGN)]
    for cname, cset in scopes:
        for size in SIZES:
            for arrival in ARRIVALS:
                for rule in RULES:
                    for beta in BETAS:
                        sub = [r for r in records if r["campus"] in cset
                               and r["size"] == size and r["arrival"] == arrival
                               and r["rule"] == rule and r["beta"] == beta]
                        st = _stats(sub)
                        if not st:
                            continue
                        row = dict(scope=cname, size=size, arrival=arrival,
                                   rule=rule, beta=beta, **st)
                        row.update(_cell_contention(sub))
                        if rule == "atc":
                            stw = _stats(sub, "twt_rule_wonly", "twt_oracle_wonly")
                            row["pct_headroom_wonly"] = stw["pct_headroom"]
                        else:
                            row["pct_headroom_wonly"] = ""
                        cell_rows.append(row)

    cell_cols = ["scope", "size", "arrival", "rule", "beta", "n",
                 "mean_twt_rule", "mean_twt_oracle", "abs_gap", "pct_headroom",
                 "pct_headroom_wonly", "wins", "ties", "losses",
                 "util_pooled", "util_worst", "window_bh", "makespan",
                 "incomplete_at_window", "n_wos"]
    with open(os.path.join(_OUT, "cells.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=cell_cols, extrasaction="ignore")
        wtr.writeheader()
        for r in cell_rows:
            row = dict(r)
            for k in ("mean_twt_rule", "mean_twt_oracle", "abs_gap"):
                row[k] = "%.4f" % row[k]
            for k in ("pct_headroom", "pct_headroom_wonly"):
                if row[k] != "":
                    row[k] = "%.4f" % row[k]
            for k in ("util_pooled", "util_worst"):
                row[k] = "%.4f" % row[k]
            for k in ("window_bh", "makespan", "incomplete_at_window", "n_wos"):
                row[k] = "%.3f" % row[k]
            wtr.writerow(row)

    summary = {
        "regime": "storm (arrival-multiplier)", "crew": CREW_TAG,
        "betas": list(BETAS), "sizes": list(SIZES), "arrivals": list(ARRIVALS),
        "campuses": list(CAMPUSES), "n_per_cell_target": N_PER_CELL,
        "tie_tol": TIE_TOL, "cells": cell_rows,
    }
    with open(os.path.join(_OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    # console: headline ATC pooled-loaded
    print("\n=== ATC pooled-loaded (9+12) full-class-shift headroom ===")
    print("  size arr   beta  n  rule_twt oracle_twt  head%%   (wonly%%)  W/T/L  uPool uWorst")
    for r in cell_rows:
        if r["scope"] == "9+12" and r["rule"] == "atc":
            print("  %4d %s b%.2f %3d  %8.1f %9.1f  %+6.3f  (%+6.3f)  %d/%d/%d  %.2f %.2f"
                  % (r["size"], r["arrival"], r["beta"], r["n"],
                     r["mean_twt_rule"], r["mean_twt_oracle"], r["pct_headroom"],
                     r["pct_headroom_wonly"], r["wins"], r["ties"], r["losses"],
                     r["util_pooled"], r["util_worst"]))
    print("\nwrote %s" % os.path.join(_OUT, "summary.json"))


if __name__ == "__main__":
    main()
