"""Y3 diagnostic: FULL-CLASS-SHIFT operationalization of the private-info latent.

PIVOTAL FINDING (notes/decisions.md): the LOCKED weight-only latent moves only
the tardiness WEIGHT and freezes the deadline, and it has ~0 decision leverage on
this benchmark (myopic ORACLE-vs-ATC within +-0.07%; CP-SAT recorded-weight
optimum == true-weight optimum, exact 0.0 gap: the optimal schedule is
weight-invariant).

This script probes the FULL-CLASS-SHIFT fix. The supervisor's private info is the
TRUE priority class c*_j, and (exactly as the RECORDED class sets recorded weight
AND recorded SLA/deadline jointly) the true class sets BOTH the true weight AND
the true operational deadline:

    s_j   = clip(round(sigma_s * xi_j), -2, +2)     (overlay, sigma_s=1.0)
    c*_j  = clip(c_j - s_j, 1, 4)                    (positive shift => more urgent)
    w*_j  = w(c*_j)          with w = 8/4/2/1  for class 1..4
    d*_j  = r_j + SLA(c*_j)  with SLA = 8/24/80/171.4 bh for class 1..4

    TRUE OBJECTIVE   TWT* = sum_j w(c*_j) * max(0, C_j - d*_j)

    ORACLE rule R    = R computed with c*  (uses w* and d*)
    RULE   R         = R computed with recorded c  (uses recorded w and d)

We measure the information ceiling as the TWT* gap between the ORACLE and the
RULE for R in {ATC, EDD}:
  * MYOPIC : one-step-greedy dispatch (ORACLE-ATC / ORACLE-EDD vs recorded-ATC /
             recorded-EDD), on the loaded cells; up to 30 inst/cell.
  * NON-MYOPIC : CP-SAT solved twice on the SAME instance -- once with recorded
             (w,d) in the objective, once with true (w*,d*) -- and BOTH scored on
             TWT*. Static solves at size 150 return OPTIMAL, so the true-weighted
             solve's TWT* is the proven optimum and the gap is the non-myopic
             information ceiling (unlike weight-only, the deadline changes, so the
             optimum is NOT class-invariant).

Constants VERIFIED against src/fmwos/hitl/overlay.py (W_OF_CLASS = 8/4/2/1,
SIGMA_S=1.0, c*=clip(c-s,1,4)) and against a sample instance (d_j = r_j + SLA(c_j)
holds exactly; SLA = 8/24/80/171.4). Latent s_j / c*_j reuse overlay.apply's exact
draw (F-NL family, master_seed=12345), so the shift is byte-identical to the
locked overlay.

Outputs: results/y3_diag/full-class-shift/{myopic.csv, cpsat.csv, summary.json}

Run:  PYTHONPATH=src OMP_NUM_THREADS=1 nice python scripts/y3_diag_full-class-shift.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import copy
import csv
import glob
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import cpsat                              # noqa: E402
from fmwos import validator as _validator           # noqa: E402
from fmwos.env import DispatchEnv                    # noqa: E402
from fmwos.hitl import deciders as dec               # noqa: E402
from fmwos.hitl import overlay as ov                 # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_diag", "full-class-shift")
os.makedirs(_OUT, exist_ok=True)

SEED = 301
MASTER_SEED = 12345
FAMILY = "F-NL"
BETAS = (0.5, 1.0)
CAMPUSES = (9, 12, 5, 10)          # {9,12} loaded; {5,10} for sign-stability
LOADED = (9, 12)
TIE_TOL = 1.0                      # tie band on the per-instance TWT* diff
ATC_K = 2.0
_BIG = 1e9

# SLA(class 1..4) in business-hours (Appendix B; VERIFIED d_j = r_j + SLA(c_j)).
SLA_OF_CLASS = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}
# w(class 1..4) mirrors overlay.W_OF_CLASS (8/4/2/1).
W_OF_CLASS = dict(ov.W_OF_CLASS)

# Loaded myopic cells: (track, size, glob-tail-fragment).
MYOPIC_CELLS = [
    ("storm", "150", "a200_c80"),
    ("storm", "150", "a300_c80"),
    ("storm", "150", "a200_c100"),
    ("storm", "150", "a300_c100"),
    ("pmmix", "150", "p20_c60"),
    ("pmmix", "150", "p50_c60"),
    ("pmmix", "150", "p80_c60"),
]
N_MYOPIC = 30

# Non-myopic CP-SAT cells (tractable at size 150; solve to OPTIMAL).
CPSAT_CELLS = [
    ("storm", "150", "a300_c80"),
    ("pmmix", "150", "p50_c60"),
]
CPSAT_CAMPUSES = (9, 12)
N_CPSAT = 20
NAIVE_BUDGET_S = 20.0     # tardiness-only solves reach OPTIMAL fast
FLOW_BUDGET_S = 10.0      # lexicographic solve: WWT primary optimal quickly (may
                          # stay FEASIBLE while proving the combined optimum)


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
    """Return (wstar, dstar) dicts keyed by work-order id.

    wstar[id] = w(c*)  from the overlay (== W_OF_CLASS[c*]).
    dstar[id] = r + SLA(c*)  -- the TRUE operational deadline set by the true
    class, mirroring the recorded d = r + SLA(c)."""
    cstar = applied["c_star"]
    wstar = dict(applied["w_star"])
    dstar = {}
    for wo in inst["work_orders"]:
        wid = wo["id"]
        dstar[wid] = float(wo["release_bh"]) + SLA_OF_CLASS[int(cstar[wid])]
    return wstar, dstar


# --------------------------------------------------------------------------- #
# TWT* scorer (adapted from fmwos.hitl.true_objective.score_true; NOT edited). #
# Difference from the locked scorer: due date is the TRUE d* = r + SLA(c*),    #
# not the recorded due_bh. Feasibility is taken from the independent validator.#
# --------------------------------------------------------------------------- #
_BREACH_TOL = 1e-9


def score_twt_star(instance, schedule, wstar, dstar, check_feasible=False):
    feasible = None
    if check_feasible:
        feasible = _validator.validate(instance, schedule)["feasible"]
    wo_by_id = {wo["id"]: wo for wo in instance.get("work_orders", []) or []}
    twt = 0.0
    for a in schedule.get("assignments", []) or []:
        wid = a.get("wo")
        wo = wo_by_id.get(wid)
        end = a.get("end_bh")
        if wo is None or end is None:
            continue
        end = float(end)
        d = dstar[wid]        # TRUE deadline r + SLA(c*)
        w = wstar[wid]        # TRUE weight w(c*)
        twt += w * max(0.0, end - d)
    return twt, feasible


# --------------------------------------------------------------------------- #
# ORACLE deciders: the rule computed with the TRUE class (w*, d*).            #
# decider(queue, t, rng) -> (job, margin)                                     #
# --------------------------------------------------------------------------- #
def make_oracle_atc(wstar, dstar, k=ATC_K):
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


def make_oracle_edd(dstar):
    def decider(queue, t, rng):
        return min(queue, key=lambda j: (dstar[j["id"]], j["id"])), _BIG
    return decider


# --------------------------------------------------------------------------- #
# Myopic worker: one instance -> per-beta ATC/EDD rule-vs-oracle TWT*.         #
# --------------------------------------------------------------------------- #
def _myopic_instance(args):
    campus, track, size, level, path = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]

    # RULE schedules are beta-independent (recorded w, d only).
    s_rule_atc = dec.run_rule(DispatchEnv(inst), "atc", seed=SEED)
    s_rule_edd = dec.run_rule(DispatchEnv(inst), "edd", seed=SEED)

    out = []
    for beta in BETAS:
        ovl = _overlay(beta)
        applied = ovl.apply(inst)
        wstar, dstar = _fullshift_maps(inst, applied)

        # ORACLE schedules (rule computed with the true class).
        s_or_atc, _ = DispatchEnv(inst).run_supervised(
            make_oracle_atc(wstar, dstar), supervisor=None,
            method="oracle_atc", seed=SEED)
        s_or_edd, _ = DispatchEnv(inst).run_supervised(
            make_oracle_edd(dstar), supervisor=None,
            method="oracle_edd", seed=SEED)

        for rule, s_rule, s_or in (("atc", s_rule_atc, s_or_atc),
                                   ("edd", s_rule_edd, s_or_edd)):
            twt_rule, _ = score_twt_star(inst, s_rule, wstar, dstar)
            twt_or, _ = score_twt_star(inst, s_or, wstar, dstar)
            out.append({
                "campus": campus, "track": track, "size": size, "level": level,
                "rule": rule, "beta": beta, "inst_id": inst_id,
                "twt_rule": twt_rule, "twt_oracle": twt_or,
            })
    return out


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
        d = r["twt_rule"] - r["twt_oracle"]      # positive => oracle better
        if abs(d) <= TIE_TOL:
            t += 1
        elif d > TIE_TOL:
            w += 1
        else:
            l += 1
    return dict(n=n, mean_rule=mean_r, mean_oracle=mean_o, abs_gap=gap,
                pct_gap=pct, oracle_wins=w, ties=t, oracle_losses=l)


def run_myopic():
    tasks = []
    for campus in CAMPUSES:
        cdir = "c%02d" % campus
        for track, size, frag in MYOPIC_CELLS:
            files = sorted(glob.glob(os.path.join(
                _INST, cdir, track, size, "*%s*.json" % frag)))
            for p in files[:N_MYOPIC]:
                tasks.append((campus, track, size, frag, p))
    print("[myopic] %d instance-tasks (x%d betas x2 rules)" % (len(tasks), len(BETAS)))

    records = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        done = 0
        for res in ex.map(_myopic_instance, tasks, chunksize=4):
            records.extend(res)
            done += 1
            if done % 100 == 0:
                print("  [myopic] %d/%d" % (done, len(tasks)))

    # per-instance CSV
    with open(os.path.join(_OUT, "myopic.csv"), "w", newline="") as fh:
        cols = ["campus", "track", "size", "level", "rule", "beta", "inst_id",
                "twt_rule", "twt_oracle"]
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        for r in records:
            row = dict(r)
            row["twt_rule"] = "%.4f" % row["twt_rule"]
            row["twt_oracle"] = "%.4f" % row["twt_oracle"]
            wtr.writerow(row)

    # aggregate per (campus, cell, rule, beta) and pooled-loaded
    agg = []
    for campus in list(CAMPUSES) + ["9+12"]:
        for track, size, frag in MYOPIC_CELLS:
            for rule in ("atc", "edd"):
                for beta in BETAS:
                    if campus == "9+12":
                        sub = [r for r in records if r["campus"] in LOADED
                               and r["track"] == track and r["level"] == frag
                               and r["rule"] == rule and r["beta"] == beta]
                        cname = "9+12"
                    else:
                        sub = [r for r in records if r["campus"] == campus
                               and r["track"] == track and r["level"] == frag
                               and r["rule"] == rule and r["beta"] == beta]
                        cname = str(campus)
                    st = _stats(sub)
                    if st:
                        agg.append(dict(campus=cname, cell="%s_%s" % (track, frag),
                                        rule=rule, beta=beta, **st))
    return records, agg


# --------------------------------------------------------------------------- #
# Non-myopic CP-SAT worker.                                                   #
# --------------------------------------------------------------------------- #
def _true_instance(inst, wstar, dstar):
    it = copy.deepcopy(inst)
    for wo in it["work_orders"]:
        wid = wo["id"]
        wo["weight"] = float(wstar[wid])
        wo["due_bh"] = float(dstar[wid])
    return it


def _cpsat_instance(args):
    """Solve the SAME instance with the RECORDED class objective (w,d) and the
    TRUE class objective (w*,d*), in TWO modes, and score every schedule on TWT*.

    Modes:
      * naive : minimize weighted tardiness only. WARNING -- the recorded
        objective is FLAT over the timing of hidden-urgent jobs (a class-4 job's
        recorded 171.4 h SLA is never binding), so among recorded-optimal plans
        CP-SAT parks those jobs arbitrarily late; scored on TWT* this looks like
        huge (spurious) information value. This is the degeneracy the locked
        solver's own docstring warns about ("zero-tardiness snapshots have
        degenerate late-start optima ... 2-9x WWT blowups").
      * flow : minimize (weighted tardiness, then total completion) lexicographically
        (cpsat.solve flow_tiebreak=True). Among tardiness-optimal plans it finishes
        ASAP, so neither the recorded nor the true planner exploits a flat
        direction. THIS is the fair, apples-to-apples non-myopic comparison.

    The recorded-objective solves are beta-independent (solve ONCE each mode)."""
    campus, track, size, frag, path = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]
    nwo = len(inst["work_orders"])

    # Recorded-class planners (beta-independent), all scored on TWT*:
    #  * sr_naive : recorded-tardiness optimum, NAIVE (degenerate tie-break) -- kept
    #    ONLY to expose the spurious-headroom artifact.
    #  * sr_flow  : recorded-tardiness optimum with the flow tie-break (finish ASAP
    #    among ties) -- the FAIR non-myopic recorded planner. May be FEASIBLE (the
    #    huge lexicographic coefficient is hard to *prove* optimal), but its primary
    #    WWT equals the naive optimum, so it is a competent recorded plan.
    #  * ATC / EDD recorded dispatch -- deployable myopic recorded planners.
    sr_naive = cpsat.solve(inst, time_limit_s=NAIVE_BUDGET_S, workers=2,
                           flow_tiebreak=False)
    sr_flow = cpsat.solve(inst, time_limit_s=FLOW_BUDGET_S, workers=2,
                          flow_tiebreak=True)
    s_atc_rec = dec.run_rule(DispatchEnv(inst), "atc", seed=SEED)
    s_edd_rec = dec.run_rule(DispatchEnv(inst), "edd", seed=SEED)
    wwt_naive = sr_naive.get("objective_bh")
    wwt_flow = sr_flow.get("objective_bh")

    out = []
    for beta in BETAS:
        ovl = _overlay(beta)
        applied = ovl.apply(inst)
        wstar, dstar = _fullshift_maps(inst, applied)
        it = _true_instance(inst, wstar, dstar)

        # True-class optimum. The naive solve minimizes EXACTLY TWT*, so its TWT*
        # score IS the proven optimum regardless of tie-break (OPTIMAL below).
        st_naive = cpsat.solve(it, time_limit_s=NAIVE_BUDGET_S, workers=2,
                               flow_tiebreak=False)

        true_opt, feas_true = score_twt_star(inst, st_naive, wstar, dstar,
                                             check_feasible=True)
        rec_naive, _ = score_twt_star(inst, sr_naive, wstar, dstar)
        rec_flow, _ = score_twt_star(inst, sr_flow, wstar, dstar)
        rec_atc, _ = score_twt_star(inst, s_atc_rec, wstar, dstar)
        rec_edd, _ = score_twt_star(inst, s_edd_rec, wstar, dstar)
        # Best competent recorded-class planner (the conservative headroom).
        best_recorded = min(rec_flow, rec_atc, rec_edd)

        out.append({
            "campus": campus, "track": track, "size": size,
            "cell": "%s_%s" % (track, frag), "beta": beta, "inst_id": inst_id,
            "nwo": nwo,
            "st_true_optimal": st_naive.get("status") == "OPTIMAL",
            "sr_naive_optimal": sr_naive.get("status") == "OPTIMAL",
            "sr_flow_status": sr_flow.get("status"),
            "wwt_naive": wwt_naive, "wwt_flow": wwt_flow,
            "feas_true": feas_true,
            # true-class proven TWT* optimum
            "twt_true": true_opt,
            # FAIR recorded planners
            "twt_recorded": best_recorded,     # best competent recorded planner
            "twt_rec_flow": rec_flow, "twt_rec_atc": rec_atc, "twt_rec_edd": rec_edd,
            # NAIVE degenerate recorded (artifact only)
            "twt_recorded_naive": rec_naive,
        })
    return out


def _cpsat_agg(records):
    n = len(records)
    if not n:
        return None
    # FAIR headline: best competent recorded planner vs proven true-class optimum.
    mr = sum(r["twt_recorded"] for r in records) / n
    mt = sum(r["twt_true"] for r in records) / n
    gap = mr - mt
    pct = (100.0 * gap / mr) if mr > 1e-9 else 0.0
    # Component recorded planners (all scored on TWT*).
    m_flow = sum(r["twt_rec_flow"] for r in records) / n
    m_atc = sum(r["twt_rec_atc"] for r in records) / n
    pct_flow = (100.0 * (m_flow - mt) / m_flow) if m_flow > 1e-9 else 0.0
    pct_atc = (100.0 * (m_atc - mt) / m_atc) if m_atc > 1e-9 else 0.0
    # NAIVE degenerate recorded CP-SAT vs true optimum -- the spurious artifact.
    mrn = sum(r["twt_recorded_naive"] for r in records) / n
    pct_naive = (100.0 * (mrn - mt) / mrn) if mrn > 1e-9 else 0.0
    w = t = l = 0
    for r in records:
        d = r["twt_recorded"] - r["twt_true"]     # positive => true class helps
        if abs(d) <= TIE_TOL:
            t += 1
        elif d > TIE_TOL:
            w += 1
        else:
            l += 1
    # true optimum is proven (OPTIMAL) on every included instance-beta.
    all_opt = all(r["st_true_optimal"] for r in records)
    return dict(n=n, mean_recorded=mr, mean_true=mt, abs_gap=gap, pct_gap=pct,
                mean_rec_flow=m_flow, mean_rec_atc=m_atc, pct_gap_flow=pct_flow,
                pct_gap_atc=pct_atc, mean_recorded_naive=mrn, pct_gap_naive=pct_naive,
                true_wins=w, ties=t, true_losses=l, all_optimal=all_opt)


def run_cpsat():
    tasks = []
    for campus in CPSAT_CAMPUSES:
        cdir = "c%02d" % campus
        for track, size, frag in CPSAT_CELLS:
            files = sorted(glob.glob(os.path.join(
                _INST, cdir, track, size, "*%s*.json" % frag)))
            for p in files[:N_CPSAT]:
                tasks.append((campus, track, size, frag, p))
    print("[cpsat] %d instance-tasks (x%d betas, recorded solve reused)"
          % (len(tasks), len(BETAS)))

    records = []
    with ProcessPoolExecutor(max_workers=4) as ex:   # 4 procs x 2 workers = 8
        done = 0
        for res in ex.map(_cpsat_instance, tasks):
            records.extend(res)
            done += 1
            if done % 10 == 0:
                print("  [cpsat] %d/%d instances" % (done, len(tasks)))

    with open(os.path.join(_OUT, "cpsat.csv"), "w", newline="") as fh:
        cols = ["campus", "cell", "beta", "inst_id", "nwo", "st_true_optimal",
                "sr_naive_optimal", "sr_flow_status", "wwt_naive", "wwt_flow",
                "feas_true", "twt_true", "twt_recorded", "twt_rec_flow",
                "twt_rec_atc", "twt_rec_edd", "twt_recorded_naive"]
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        for r in records:
            row = dict(r)
            for k in ("twt_true", "twt_recorded", "twt_rec_flow", "twt_rec_atc",
                      "twt_rec_edd", "twt_recorded_naive"):
                row[k] = "%.4f" % row[k]
            wtr.writerow({c: row[c] for c in cols})

    agg = []
    for campus in list(CPSAT_CAMPUSES) + ["9+12"]:
        for track, size, frag in CPSAT_CELLS:
            cell = "%s_%s" % (track, frag)
            for beta in BETAS:
                if campus == "9+12":
                    sub = [r for r in records if r["campus"] in CPSAT_CAMPUSES
                           and r["cell"] == cell and r["beta"] == beta]
                    cname = "9+12"
                else:
                    sub = [r for r in records if r["campus"] == campus
                           and r["cell"] == cell and r["beta"] == beta]
                    cname = str(campus)
                a = _cpsat_agg(sub)
                if a:
                    agg.append(dict(campus=cname, cell=cell, beta=beta, **a))
    return records, agg


def main():
    do_cpsat = "--myopic-only" not in sys.argv
    do_myopic = "--cpsat-only" not in sys.argv

    summary = {"variant": "full-class-shift"}

    if do_myopic:
        t0 = time.perf_counter()
        _, m_agg = run_myopic()
        summary["myopic_agg"] = m_agg
        print("[myopic] done in %.1fs, %d agg rows" % (time.perf_counter() - t0, len(m_agg)))
        print("\n=== MYOPIC (pooled loaded 9+12) ATC ===")
        for row in m_agg:
            if row["campus"] == "9+12" and row["rule"] == "atc":
                print("  %-16s b%.2f n=%d  rule=%.1f oracle=%.1f  %+.3f%%  W/T/L=%d/%d/%d"
                      % (row["cell"], row["beta"], row["n"], row["mean_rule"],
                         row["mean_oracle"], row["pct_gap"], row["oracle_wins"],
                         row["ties"], row["oracle_losses"]))

    if do_cpsat:
        t0 = time.perf_counter()
        _, c_agg = run_cpsat()
        summary["cpsat_agg"] = c_agg
        print("[cpsat] done in %.1fs, %d agg rows" % (time.perf_counter() - t0, len(c_agg)))
        print("\n=== NON-MYOPIC CP-SAT: FAIR (best recorded planner vs true optimum) ===")
        print("    [flow=recorded-CPSAT-lex, atc=recorded-ATC; naive=degenerate artifact]")
        for row in c_agg:
            print("  c%-4s %-16s b%.2f n=%d  bestRec=%.1f trueOPT=%.1f  FAIR%+.2f%%  "
                  "(flow%+.2f%% atc%+.2f%%)  W/T/L=%d/%d/%d  [naive%+.1f%%]  trueOPT=%s"
                  % (row["campus"], row["cell"], row["beta"], row["n"],
                     row["mean_recorded"], row["mean_true"], row["pct_gap"],
                     row["pct_gap_flow"], row["pct_gap_atc"],
                     row["true_wins"], row["ties"], row["true_losses"],
                     row["pct_gap_naive"], row["all_optimal"]))

    with open(os.path.join(_OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\nwrote %s" % os.path.join(_OUT, "summary.json"))


if __name__ == "__main__":
    main()
