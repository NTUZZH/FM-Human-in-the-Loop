"""Y3 continuation: FULL-CLASS-SHIFT ORACLE-vs-RULE on PMMIX at size 400.

Operationalization under test (deadline moves with the class):
    s_j   = clip(round(sigma_s * xi_j), -2, +2)      (overlay, F-NL, seed 12345)
    c*_j  = clip(c_j - s_j, 1, 4)
    w*_j  = w(c*_j)          with w = 8/4/2/1  for class 1..4
    d*_j  = r_j + SLA(c*_j)  with SLA = 8/24/80/171.4 bh for class 1..4
    TWT*  = sum_j w*_j * max(0, C_j - d*_j)             (TRUE objective)

Comparison (real headline pipeline, run in the actual dynamic dispatch env):
  * RULE   = ATC / EDD on the RECORDED fields (recorded w, recorded due_bh) --
             the deployed rule that never sees the latent.
  * ORACLE = the SAME rule computed with the TRUE class. For ATC this is the
             supervisor preferred-pick path (deciders.run_oracle_greedy over a
             Supervisor whose true-weight ATC already injects w*), with the
             supervisor's due-date term overridden to d* in this scratch wrapper
             (the locked supervisor.py is NOT edited: we reassign sup.due after
             construction). For EDD, a scratch decider sorting by d*.
  Both schedules scored on TWT* (d*, w*); feasibility from the independent
  validator. run_oracle_greedy (sup path) is asserted byte-identical to a
  standalone true-weight/true-deadline ATC decider before the run launches.

Regime: pmmix (arrival_multiplier = 1.0; crew_multiplier is the stress knob),
size 400, cells pm_share {p20,p50,p80} x crew {c60 (loaded), c100}. Campuses
{9,12} (loaded / headline) + {5,10} (sign-stability). Beta {0.25,0.5,1.0}.
Up to 40 instances per cell (30 available on disk). Plus a plain generator/400
baseline at the DEFAULT crew (crew_multiplier=1.0) for reference.

CONTENTION: per instance we record utilization = sum_p_bh / (crew * window_bh),
pooled and worst-trade (per-trade p / (crew_of_trade * window_bh)), and the
realized backlog = count of RULE-ATC assignments finishing after window_bh
(incomplete at the arrival horizon). window_bh is the instance arrival window
(generator sets it to the last release). This mirrors p4_dyneval's
_storm2_realized_util.

Outputs: results/y3_cont/pmmix-size400/{per_instance.csv, agg.csv, summary.json}

Run: PYTHONPATH=src OMP_NUM_THREADS=1 nice python scripts/y3_cont_pmmix-size400.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import glob
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import validator as _validator            # noqa: E402
from fmwos.env import DispatchEnv                     # noqa: E402
from fmwos.hitl import deciders as dec                # noqa: E402
from fmwos.hitl import overlay as ov                  # noqa: E402
from fmwos.hitl.supervisor import Supervisor          # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_cont", "pmmix-size400")
os.makedirs(_OUT, exist_ok=True)

SEED = 301
MASTER_SEED = 12345
FAMILY = "F-NL"
BETAS = (0.25, 0.5, 1.0)
CAMPUSES = (9, 12, 5, 10)         # {9,12} loaded/headline; {5,10} sign-stability
LOADED = (9, 12)
TIE_TOL = 1.0                     # tie band on the per-instance TWT* diff
ATC_K = 2.0
_BIG = 1e9
N_PMMIX = 40                      # cap; 30 available on disk
N_GEN = 40

# SLA(class 1..4) in business-hours (Appendix B; VERIFIED d_j = r_j + SLA(c_j)).
SLA_OF_CLASS = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}
W_OF_CLASS = dict(ov.W_OF_CLASS)   # 8/4/2/1

PM_LEVELS = ("p20", "p50", "p80")
CREW_LEVELS = ("c60", "c100")

_BREACH_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Overlay cache (process-global)                                              #
# --------------------------------------------------------------------------- #
_OVERLAYS = {}


def _overlay(beta):
    o = _OVERLAYS.get(beta)
    if o is None:
        o = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                        master_seed=MASTER_SEED))
        _OVERLAYS[beta] = o
    return o


# --------------------------------------------------------------------------- #
# Full-class-shift maps: true weight w* and true deadline d* per work order.   #
# --------------------------------------------------------------------------- #
def _fullshift_maps(inst, applied):
    cstar = applied["c_star"]
    wstar = dict(applied["w_star"])
    dstar = {}
    for wo in inst["work_orders"]:
        wid = wo["id"]
        dstar[wid] = float(wo["release_bh"]) + SLA_OF_CLASS[int(cstar[wid])]
    return wstar, dstar


# --------------------------------------------------------------------------- #
# TWT* scorer (adapted from fmwos.hitl.true_objective.score_true; NOT edited). #
# Difference from the locked scorer: due date is TRUE d* = r + SLA(c*), not the #
# recorded due_bh. Feasibility from the independent validator.                  #
# --------------------------------------------------------------------------- #
def score_twt_star(inst, sched, wstar, dstar, check_feasible=False):
    feasible = None
    if check_feasible:
        feasible = _validator.validate(inst, sched)["feasible"]
    wo_by = {w["id"]: w for w in inst.get("work_orders", []) or []}
    twt = 0.0
    n_late = 0
    for a in sched.get("assignments", []) or []:
        wid = a.get("wo")
        end = a.get("end_bh")
        if wid not in wo_by or end is None:
            continue
        end = float(end)
        d = dstar[wid]
        w = wstar[wid]
        if end > d + _BREACH_TOL:
            n_late += 1
        twt += w * max(0.0, end - d)
    return twt, n_late, feasible


# --------------------------------------------------------------------------- #
# ORACLE deciders.                                                            #
#  * ATC : the SUPERVISOR preferred-pick path (true-weight ATC already injects #
#          w*), with the supervisor's due term overridden to d*. Honors the    #
#          task's "use deciders.py / supervisor preferred-pick path" while     #
#          reading d* as well as w*. Locked file untouched (sup.due reassigned).#
#  * EDD : scratch decider sorting by (d*, id).                                 #
# --------------------------------------------------------------------------- #
def _oracle_atc_sup(inst, applied, dstar):
    """ORACLE-ATC schedule via the deciders.run_oracle_greedy supervisor path."""
    sup = Supervisor(_overlay_from_applied(applied), inst, rho=0.0, applied=applied)
    sup.due = dstar                     # override recorded due_bh -> d* (w* already set)
    return dec.run_oracle_greedy(DispatchEnv(inst), sup, seed=SEED)


def _overlay_from_applied(applied):
    # Supervisor only needs an object with .apply when applied is not passed; we
    # always pass applied, so any overlay object works. Reuse the beta-1 overlay
    # (its .apply is never called here because applied is provided).
    return _overlay(1.0)


def make_oracle_atc_decider(wstar, dstar, k=ATC_K):
    """Standalone true-weight/true-deadline ATC decider (equivalence check)."""
    def decider(queue, t, rng):
        pbar = sum(j["p_bh"] for j in queue) / len(queue)
        denom = k * pbar

        def key(j):
            jid = j["id"]
            slack = max(0.0, dstar[jid] - t - j["p_bh"])
            score = (wstar[jid] / j["p_bh"]) * math.exp(-slack / denom)
            return (-score, jid)
        return min(queue, key=key), _BIG
    return decider


def make_oracle_edd_decider(dstar):
    def decider(queue, t, rng):
        return min(queue, key=lambda j: (dstar[j["id"]], j["id"])), _BIG
    return decider


# --------------------------------------------------------------------------- #
# Utilization + backlog.                                                      #
# --------------------------------------------------------------------------- #
def _utilization(inst):
    win = float(inst["meta"]["window_bh"])
    tot = sum(float(w["p_bh"]) for w in inst["work_orders"])
    ncrew = len(inst["technicians"])
    pooled = tot / (ncrew * win) if ncrew * win > 0 else 0.0
    ptrade = defaultdict(float)
    ctrade = Counter(t["trade"] for t in inst["technicians"])
    for w in inst["work_orders"]:
        ptrade[w["trade"]] += float(w["p_bh"])
    worst = 0.0
    worst_trade = None
    for tr, pp in ptrade.items():
        c = ctrade.get(tr, 0)
        if c > 0:
            u = pp / (c * win)
            if u > worst:
                worst, worst_trade = u, tr
    return pooled, worst, worst_trade, win


def _backlog_at_horizon(sched, win):
    """Count of assignments completing after the arrival window (backlog), and
    the makespan (max end_bh)."""
    ends = [float(a["end_bh"]) for a in sched.get("assignments", []) or []]
    inc = sum(1 for e in ends if e > win + _BREACH_TOL)
    mksp = max(ends) if ends else 0.0
    return inc, mksp


# --------------------------------------------------------------------------- #
# Worker: one instance -> per-beta ATC/EDD rule-vs-oracle TWT* + contention.   #
# --------------------------------------------------------------------------- #
def _process_instance(args):
    campus, track, cell, path = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]

    pooled, worst, worst_trade, win = _utilization(inst)

    # RULE schedules are beta-independent (recorded w, d only).
    s_rule_atc = dec.run_rule(DispatchEnv(inst), "atc", seed=SEED)
    s_rule_edd = dec.run_rule(DispatchEnv(inst), "edd", seed=SEED)
    inc_atc, mksp_atc = _backlog_at_horizon(s_rule_atc, win)

    out = []
    for beta in BETAS:
        ovl = _overlay(beta)
        applied = ovl.apply(inst)
        wstar, dstar = _fullshift_maps(inst, applied)

        # ORACLE-ATC via supervisor preferred-pick path (due overridden to d*).
        sup = Supervisor(ovl, inst, rho=0.0, applied=applied)
        sup.due = dstar
        s_or_atc = dec.run_oracle_greedy(DispatchEnv(inst), sup, seed=SEED)
        # ORACLE-EDD via scratch decider (no supervisor EDD path).
        s_or_edd, _ = DispatchEnv(inst).run_supervised(
            make_oracle_edd_decider(dstar), supervisor=None,
            method="oracle_edd", seed=SEED)

        twt_rule_atc, nlate_ra, _ = score_twt_star(inst, s_rule_atc, wstar, dstar)
        twt_or_atc, nlate_oa, _ = score_twt_star(inst, s_or_atc, wstar, dstar)
        twt_rule_edd, _, _ = score_twt_star(inst, s_rule_edd, wstar, dstar)
        twt_or_edd, _, _ = score_twt_star(inst, s_or_edd, wstar, dstar)

        base = dict(campus=campus, track=track, cell=cell, beta=beta,
                    inst_id=inst_id, util_pooled=pooled, util_worst=worst,
                    worst_trade=worst_trade, window_bh=win,
                    incomplete_at_horizon=inc_atc, makespan=mksp_atc,
                    n_wo=len(inst["work_orders"]),
                    n_crew=len(inst["technicians"]))
        out.append(dict(base, rule="atc", twt_rule=twt_rule_atc,
                        twt_oracle=twt_or_atc, nlate_rule=nlate_ra,
                        nlate_oracle=nlate_oa))
        out.append(dict(base, rule="edd", twt_rule=twt_rule_edd,
                        twt_oracle=twt_or_edd, nlate_rule=-1, nlate_oracle=-1))
    return out


# --------------------------------------------------------------------------- #
# Task enumeration.                                                           #
# --------------------------------------------------------------------------- #
def _pmmix_files(campus, pm, crew):
    cdir = "c%02d" % campus
    pat = os.path.join(_INST, cdir, "pmmix", "400",
                       "*_%s_%s_*.json" % (pm, crew))
    return sorted(glob.glob(pat))[:N_PMMIX]


def _gen_files(campus):
    cdir = "c%02d" % campus
    pat = os.path.join(_INST, cdir, "generator", "400", "*.json")
    return sorted(glob.glob(pat))[:N_GEN]


def _all_tasks():
    tasks = []
    for campus in CAMPUSES:
        for pm in PM_LEVELS:
            for crew in CREW_LEVELS:
                cell = "pmmix_%s_%s" % (pm, crew)
                for p in _pmmix_files(campus, pm, crew):
                    tasks.append((campus, "pmmix", cell, p))
        for p in _gen_files(campus):
            tasks.append((campus, "generator", "gen400_default", p))
    return tasks


# --------------------------------------------------------------------------- #
# Aggregation.                                                                #
# --------------------------------------------------------------------------- #
def _stats(records):
    n = len(records)
    if not n:
        return None
    sum_r = sum(r["twt_rule"] for r in records)
    sum_o = sum(r["twt_oracle"] for r in records)
    mean_r = sum_r / n
    mean_o = sum_o / n
    gap = mean_r - mean_o
    pct = (100.0 * gap / mean_r) if mean_r > 1e-9 else 0.0
    w = t = l = 0
    for r in records:
        d = r["twt_rule"] - r["twt_oracle"]        # positive => oracle better
        if abs(d) <= TIE_TOL:
            t += 1
        elif d > TIE_TOL:
            w += 1
        else:
            l += 1
    up = sum(r["util_pooled"] for r in records) / n
    uw = sum(r["util_worst"] for r in records) / n
    inc = sum(r["incomplete_at_horizon"] for r in records) / n
    nlr = [r["nlate_rule"] for r in records if r["nlate_rule"] >= 0]
    mean_nlate = (sum(nlr) / len(nlr)) if nlr else -1
    return dict(n=n, mean_rule=mean_r, mean_oracle=mean_o, abs_gap=gap,
                pct_gap=pct, oracle_wins=w, ties=t, oracle_losses=l,
                util_pooled=up, util_worst=uw, incomplete_at_horizon=inc,
                mean_nlate_rule=mean_nlate)


def main():
    tasks = _all_tasks()
    n_pmmix = sum(1 for tk in tasks if tk[1] == "pmmix")
    n_gen = sum(1 for tk in tasks if tk[1] == "generator")
    print("tasks: %d pmmix + %d generator = %d instances (x%d betas x2 rules)"
          % (n_pmmix, n_gen, len(tasks), len(BETAS)))

    # ---- pre-launch equivalence check: supervisor path == standalone decider ----
    ci, ctrack, ccell, cpath = tasks[0]
    cinst = json.load(open(cpath))
    capplied = _overlay(1.0).apply(cinst)
    cwstar, cdstar = _fullshift_maps(cinst, capplied)
    csup = Supervisor(_overlay(1.0), cinst, rho=0.0, applied=capplied)
    csup.due = cdstar
    s_sup = dec.run_oracle_greedy(DispatchEnv(cinst), csup, seed=SEED)
    s_cust, _ = DispatchEnv(cinst).run_supervised(
        make_oracle_atc_decider(cwstar, cdstar), supervisor=None,
        method="oa", seed=SEED)
    a1 = [(a["wo"], round(a["end_bh"], 6)) for a in s_sup["assignments"]]
    a2 = [(a["wo"], round(a["end_bh"], 6)) for a in s_cust["assignments"]]
    assert a1 == a2, "ORACLE supervisor-path != standalone true-ATC decider"
    # verify recorded due identity d = r + SLA(c) on this instance
    for wo in cinst["work_orders"]:
        d_expect = float(wo["release_bh"]) + SLA_OF_CLASS[int(wo["priority"])]
        assert abs(float(wo["due_bh"]) - d_expect) < 1e-6, "recorded due != r+SLA(c)"
    print("pre-launch checks OK: sup-path ORACLE == standalone decider; "
          "recorded due = r+SLA(c) exact.")

    records = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as ex:
        done = 0
        for res in ex.map(_process_instance, tasks, chunksize=4):
            records.extend(res)
            done += 1
            if done % 100 == 0:
                print("  %d/%d instances (%.0fs)"
                      % (done, len(tasks), time.perf_counter() - t0))
    print("processed %d instances in %.1fs; %d records"
          % (len(tasks), time.perf_counter() - t0, len(records)))

    # ---- per-instance CSV ----
    pi_cols = ["campus", "track", "cell", "rule", "beta", "inst_id", "n_wo",
               "n_crew", "window_bh", "util_pooled", "util_worst", "worst_trade",
               "incomplete_at_horizon", "makespan", "twt_rule", "twt_oracle",
               "nlate_rule", "nlate_oracle"]
    with open(os.path.join(_OUT, "per_instance.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=pi_cols)
        wtr.writeheader()
        for r in records:
            row = {c: r.get(c) for c in pi_cols}
            for k in ("window_bh", "util_pooled", "util_worst", "makespan",
                      "twt_rule", "twt_oracle"):
                row[k] = "%.4f" % float(row[k])
            wtr.writerow(row)

    # ---- aggregate ----
    cells = ["pmmix_%s_%s" % (pm, cr) for pm in PM_LEVELS for cr in CREW_LEVELS]
    cells.append("gen400_default")
    # crew-pooled (over pm) pseudo-cells
    crew_pool = {"c60": ["pmmix_%s_c60" % pm for pm in PM_LEVELS],
                 "c100": ["pmmix_%s_c100" % pm for pm in PM_LEVELS]}

    agg = []

    def _emit(scope, campus_name, cell_name, rule, beta, sub):
        st = _stats(sub)
        if st:
            agg.append(dict(scope=scope, campus=campus_name, cell=cell_name,
                            rule=rule, beta=beta, **st))

    for rule in ("atc", "edd"):
        for beta in BETAS:
            # per campus, per cell
            for campus in CAMPUSES:
                for cell in cells:
                    sub = [r for r in records if r["campus"] == campus
                           and r["cell"] == cell and r["rule"] == rule
                           and r["beta"] == beta]
                    _emit("campus", str(campus), cell, rule, beta, sub)
                # per campus, crew-pooled over pm
                for cr, clist in crew_pool.items():
                    sub = [r for r in records if r["campus"] == campus
                           and r["cell"] in clist and r["rule"] == rule
                           and r["beta"] == beta]
                    _emit("campus", str(campus), "pmmix_%s_poolpm" % cr,
                          rule, beta, sub)
            # pooled loaded {9,12} per cell
            for cell in cells:
                sub = [r for r in records if r["campus"] in LOADED
                       and r["cell"] == cell and r["rule"] == rule
                       and r["beta"] == beta]
                _emit("pooled_loaded", "9+12", cell, rule, beta, sub)
            # pooled loaded {9,12} crew-pooled over pm
            for cr, clist in crew_pool.items():
                sub = [r for r in records if r["campus"] in LOADED
                       and r["cell"] in clist and r["rule"] == rule
                       and r["beta"] == beta]
                _emit("pooled_loaded", "9+12", "pmmix_%s_poolpm" % cr,
                      rule, beta, sub)
            # pooled ALL campuses crew-pooled over pm (sign-stability view)
            for cr, clist in crew_pool.items():
                sub = [r for r in records if r["cell"] in clist
                       and r["rule"] == rule and r["beta"] == beta]
                _emit("pooled_all", "5+9+10+12", "pmmix_%s_poolpm" % cr,
                      rule, beta, sub)

    agg_cols = ["scope", "campus", "cell", "rule", "beta", "n", "mean_rule",
                "mean_oracle", "abs_gap", "pct_gap", "oracle_wins", "ties",
                "oracle_losses", "util_pooled", "util_worst",
                "incomplete_at_horizon", "mean_nlate_rule"]
    with open(os.path.join(_OUT, "agg.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=agg_cols)
        wtr.writeheader()
        for r in agg:
            row = dict(r)
            for k in ("mean_rule", "mean_oracle", "abs_gap", "util_pooled",
                      "util_worst"):
                row[k] = "%.4f" % row[k]
            row["pct_gap"] = "%.4f" % row["pct_gap"]
            row["incomplete_at_horizon"] = "%.2f" % row["incomplete_at_horizon"]
            row["mean_nlate_rule"] = "%.2f" % row["mean_nlate_rule"]
            wtr.writerow({c: row[c] for c in agg_cols})

    with open(os.path.join(_OUT, "summary.json"), "w") as fh:
        json.dump({"regime": "pmmix_size400_full_class_shift",
                   "betas": list(BETAS), "campuses": list(CAMPUSES),
                   "loaded": list(LOADED), "n_pmmix_per_cell_cap": N_PMMIX,
                   "agg": agg}, fh, indent=1)

    # ---- console: headline ATC, pooled loaded, crew-pooled ----
    print("\n=== ATC pooled-loaded {9,12}, crew-pooled over pm ===")
    for r in agg:
        if (r["scope"] == "pooled_loaded" and r["rule"] == "atc"
                and r["cell"].endswith("_poolpm")):
            print("  %-20s b%.2f n=%d up=%.2f uw=%.2f inc@H=%.0f  "
                  "rule=%.1f oracle=%.1f  head=%+.3f%%  W/T/L=%d/%d/%d"
                  % (r["cell"], r["beta"], r["n"], r["util_pooled"],
                     r["util_worst"], r["incomplete_at_horizon"],
                     r["mean_rule"], r["mean_oracle"], r["pct_gap"],
                     r["oracle_wins"], r["ties"], r["oracle_losses"]))
    print("\n=== ATC per-campus c60 (crew-pooled over pm) -- realism view ===")
    for r in agg:
        if (r["scope"] == "campus" and r["rule"] == "atc"
                and r["cell"] == "pmmix_c60_poolpm"):
            print("  c%-3s b%.2f n=%d up=%.2f uw=%.2f inc@H=%.0f  head=%+.3f%%  "
                  "W/T/L=%d/%d/%d"
                  % (r["campus"], r["beta"], r["n"], r["util_pooled"],
                     r["util_worst"], r["incomplete_at_horizon"], r["pct_gap"],
                     r["oracle_wins"], r["ties"], r["oracle_losses"]))
    print("\nwrote %s" % _OUT)


if __name__ == "__main__":
    main()
