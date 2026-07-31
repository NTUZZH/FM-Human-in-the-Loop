"""Y3 diagnostic: DEADLINE-ONLY operationalization of the supervisor's private info.

Prior variants (notes/headroom_scan.md, headroom_scan2.md) let the latent class
shift move BOTH the class weight AND the deadline (via c*), and scored TWT* with
the shifted weight w*(c*) while keeping the recorded due date. That gave ~zero
information headroom: relabelling the cost of the SAME tardy set never reorders
an optimizing planner, and only rarely reorders ATC.

THIS variant is different. The supervisor's private information moves ONLY the
effective DEADLINE; the contractual cost weight stays at the recorded w(c_j).
Story: "urgency = this must be done sooner", without changing what lateness costs.

    true class     c*_j = clip(c_j - s_j, 1, 4)      (overlay.apply, unchanged)
    true deadline  d*_j = r_j + SLA(c*_j)            (r_j = release_bh)
    recorded dead. d_j  = r_j + SLA(c_j) = wo["due_bh"]   (verified: exact)
    TRUE objective TWT* = sum_j w(c_j) * max(0, C_j - d*_j)   (recorded weight!)

  RULE   = a dispatching rule (ATC / EDD) on the RECORDED (w, d).
  ORACLE = the SAME rule on (recorded w, TRUE deadline d*).  Realised by feeding
           the rule a copy of the instance whose due_bh is overwritten with d*
           (weight untouched), so the rule's slack / due-date term sees d*.

Both schedules are scored on TWT* by an INDEPENDENT scorer implemented inline
here (NOT fmwos.hitl.true_objective, which keeps the recorded due date -- wrong
for this variant). Headroom = (TWT*_RULE - TWT*_ORACLE) / TWT*_RULE; positive =
the private deadline information helps.

Two probes, both on the loaded cells that carry real tardiness:
  * MYOPIC     -- ATC and EDD, campuses {5,9,10,12}, size 150, storm (all a/c
                  levels) + pmmix (c60 crew), betas {0.5, 1.0}, all 30 inst/cell.
  * NON-MYOPIC -- CP-SAT solved twice (RULE objective vs ORACLE objective d*) on
                  the tractable stressed cells (storm/pmmix 150 solve OPTIMAL in
                  seconds), campuses {9,12}, betas {0.5, 1.0}, 20 inst/cell. The
                  gap between the two proven optima is the non-myopic information
                  ceiling for THIS operationalization. storm2 is skipped
                  (intractable for CP-SAT at its size).

Run:
  OMP_NUM_THREADS=1 PYTHONPATH=src nice python scripts/y3_diag_deadline-only-shift.py myopic
  OMP_NUM_THREADS=1 PYTHONPATH=src nice python scripts/y3_diag_deadline-only-shift.py probe
  (no arg -> both)
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
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import cpsat, pdrs                        # noqa: E402
from fmwos.hitl import overlay as ov                 # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_diag", "deadline-only-shift")

MASTER_SEED = 12345
FAMILY = "F-NL"
SEED = 301                       # dispatch seed (inert for atc/edd)
BETAS = (0.5, 1.0)
RULES = ("atc", "edd")
TIE_TOL = 1.0                    # weighted unit; per-instance tie band

# SLA(class 1..4) in business-hours (proposal; verified d_j = r_j + SLA(c_j)).
SLA = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}
# Recorded tardiness weight by class (kept fixed in this variant).
W_OF_CLASS = {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}

CAMPUSES = (5, 9, 10, 12)
LOADED = (9, 12)

# Myopic cells: storm all a/c levels + pmmix c60 (the loaded crew), size 150.
STORM_LEVELS = ("a125_c100", "a125_c80", "a150_c100", "a150_c80",
                "a200_c100", "a200_c80", "a300_c100", "a300_c80")
PMMIX_LEVELS = ("p20_c60", "p50_c60", "p80_c60")
# Hard subset used for the pooled structured report.
HARD_CELLS = [("storm", "a300_c80"), ("storm", "a300_c100"),
              ("storm", "a200_c80"), ("storm", "a200_c100"),
              ("pmmix", "p20_c60"), ("pmmix", "p50_c60"), ("pmmix", "p80_c60")]

# Non-myopic CP-SAT cells (tractable, carry tardiness).
PROBE_CELLS = [("storm", "150", "a300_c80"),
               ("storm", "150", "a200_c100"),
               ("pmmix", "150", "p20_c60")]
PROBE_CAMPUSES = (9, 12)
PROBE_N = 20
PROBE_BUDGET_S = 20.0
PROBE_WORKERS = 2

_OVERLAYS = {}


def _overlay(beta):
    o = _OVERLAYS.get(beta)
    if o is None:
        o = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                        master_seed=MASTER_SEED))
        _OVERLAYS[beta] = o
    return o


# --------------------------------------------------------------------------- #
# Deadline-only latent + independent scorer                                    #
# --------------------------------------------------------------------------- #
def dstar_map(inst, applied):
    """d*_j = release_bh + SLA(c*_j) (true class sets the deadline only)."""
    cstar = applied["c_star"]
    out = {}
    for wo in inst["work_orders"]:
        c = int(cstar.get(wo["id"], int(wo["priority"])))
        out[wo["id"]] = float(wo["release_bh"]) + SLA[c]
    return out


def inst_with_dstar(inst, dstar):
    """Copy of the instance with due_bh overwritten by d* (weight untouched)."""
    it = copy.deepcopy(inst)
    for wo in it["work_orders"]:
        wo["due_bh"] = dstar[wo["id"]]
    return it


def score_twt_deadline(inst, sched, dstar):
    """INDEPENDENT deadline-only true objective.

    TWT* = sum_j w(c_j) * max(0, C_j - d*_j).  Recorded weight w(c_j) = wo weight;
    TRUE deadline d*_j; C_j = assignment end_bh.
    """
    wo_by_id = {wo["id"]: wo for wo in inst["work_orders"]}
    twt = 0.0
    for a in sched.get("assignments", []):
        wid = a.get("wo")
        end = a.get("end_bh")
        wo = wo_by_id.get(wid)
        if wo is None or end is None:
            continue
        tard = float(end) - dstar[wid]
        if tard > 0.0:
            twt += float(wo["weight"]) * tard
    return twt


# --------------------------------------------------------------------------- #
# File enumeration                                                             #
# --------------------------------------------------------------------------- #
def _cell_files(campus, track, size, level, cap=None):
    cdir = "c%02d" % campus
    d = os.path.join(_INST, cdir, track, size)
    files = sorted(glob.glob(os.path.join(d, "*%s_*.json" % level)))
    return files if cap is None else files[:cap]


# --------------------------------------------------------------------------- #
# MYOPIC worker                                                                #
# --------------------------------------------------------------------------- #
def _myopic_instance(args):
    campus, track, level, path = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]

    # RULE schedules (recorded due) are beta-independent -> compute once per rule.
    rule_sched = {r: pdrs.dispatch(inst, r, seed=SEED) for r in RULES}

    out = []
    for beta in BETAS:
        applied = _overlay(beta).apply(inst)
        dstar = dstar_map(inst, applied)
        it = inst_with_dstar(inst, dstar)
        for r in RULES:
            twt_rule = score_twt_deadline(inst, rule_sched[r], dstar)
            oracle_sched = pdrs.dispatch(it, r, seed=SEED)
            twt_oracle = score_twt_deadline(inst, oracle_sched, dstar)
            out.append({
                "campus": campus, "track": track, "level": level, "rule": r,
                "beta": beta, "inst_id": inst_id,
                "twt_rule": twt_rule, "twt_oracle": twt_oracle,
            })
    return out


def _stats(records):
    n = len(records)
    if not n:
        return dict(n=0, mean_rule=0.0, mean_oracle=0.0, abs_gap=0.0,
                    pct=0.0, wins=0, ties=0, losses=0)
    sr = sum(r["twt_rule"] for r in records)
    so = sum(r["twt_oracle"] for r in records)
    mr, mo = sr / n, so / n
    gap = mr - mo
    pct = (100.0 * gap / mr) if mr > 1e-9 else 0.0
    w = t = l = 0
    for r in records:
        d = r["twt_rule"] - r["twt_oracle"]     # + => oracle better
        if abs(d) <= TIE_TOL:
            t += 1
        elif d > TIE_TOL:
            w += 1
        else:
            l += 1
    return dict(n=n, mean_rule=mr, mean_oracle=mo, abs_gap=gap, pct=pct,
                wins=w, ties=t, losses=l)


def run_myopic():
    tasks = []
    for campus in CAMPUSES:
        for level in STORM_LEVELS:
            for p in _cell_files(campus, "storm", "150", level):
                tasks.append((campus, "storm", level, p))
        for level in PMMIX_LEVELS:
            for p in _cell_files(campus, "pmmix", "150", level):
                tasks.append((campus, "pmmix", level, p))
    print("[myopic] instance-tasks: %d (x%d betas x%d rules)"
          % (len(tasks), len(BETAS), len(RULES)))

    records = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        done = 0
        for res in ex.map(_myopic_instance, tasks, chunksize=8):
            records.extend(res)
            done += 1
            if done % 500 == 0:
                print("  processed %d/%d" % (done, len(tasks)))

    os.makedirs(_OUT, exist_ok=True)
    # Per-instance rows.
    with open(os.path.join(_OUT, "myopic_perinstance.csv"), "w", newline="") as fh:
        cols = ["campus", "track", "level", "rule", "beta", "inst_id",
                "twt_rule", "twt_oracle"]
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        for r in records:
            row = dict(r)
            row["twt_rule"] = "%.4f" % row["twt_rule"]
            row["twt_oracle"] = "%.4f" % row["twt_oracle"]
            wtr.writerow(row)

    # Aggregate: per (campus, track, level, rule, beta) and pooled-loaded.
    def cell_tag(track, level):
        return "%s/150/%s" % (track, level)

    agg = []
    levels = {"storm": STORM_LEVELS, "pmmix": PMMIX_LEVELS}
    for track, lv_list in levels.items():
        for level in lv_list:
            for rule in RULES:
                for beta in BETAS:
                    for campus in CAMPUSES:
                        sub = [r for r in records if r["campus"] == campus
                               and r["track"] == track and r["level"] == level
                               and r["rule"] == rule and r["beta"] == beta]
                        if sub:
                            agg.append(dict(scope="campus", campus=str(campus),
                                            cell=cell_tag(track, level),
                                            rule=rule, beta=beta, **_stats(sub)))
                    sub = [r for r in records if r["campus"] in LOADED
                           and r["track"] == track and r["level"] == level
                           and r["rule"] == rule and r["beta"] == beta]
                    if sub:
                        agg.append(dict(scope="pooled_loaded", campus="9+12",
                                        cell=cell_tag(track, level),
                                        rule=rule, beta=beta, **_stats(sub)))

    with open(os.path.join(_OUT, "myopic_agg.csv"), "w", newline="") as fh:
        cols = ["scope", "campus", "cell", "rule", "beta", "n", "mean_rule",
                "mean_oracle", "abs_gap", "pct", "wins", "ties", "losses"]
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        for r in agg:
            o = dict(r)
            for k in ("mean_rule", "mean_oracle", "abs_gap"):
                o[k] = "%.4f" % o[k]
            o["pct"] = "%.4f" % o["pct"]
            wtr.writerow({c: o[c] for c in cols})

    # Console: pooled-loaded on hard cells.
    print("\n=== MYOPIC pooled-loaded {9,12} on hard cells ===")
    print("cell                    rule beta  n  meanRULE  meanORA   gap    %hd   W/T/L")
    for track, level in HARD_CELLS:
        for rule in RULES:
            for beta in BETAS:
                sub = [r for r in records if r["campus"] in LOADED
                       and r["track"] == track and r["level"] == level
                       and r["rule"] == rule and r["beta"] == beta]
                if not sub:
                    continue
                s = _stats(sub)
                print("%-22s %-4s %.2f %3d %9.2f %9.2f %7.2f %6.3f %d/%d/%d"
                      % (cell_tag(track, level), rule, beta, s["n"],
                         s["mean_rule"], s["mean_oracle"], s["abs_gap"],
                         s["pct"], s["wins"], s["ties"], s["losses"]))

    import pickle
    with open(os.path.join(_OUT, "_myopic_agg.pkl"), "wb") as fh:
        pickle.dump(agg, fh)
    print("[myopic] wrote %s" % _OUT)
    return agg


# --------------------------------------------------------------------------- #
# NON-MYOPIC CP-SAT probe                                                      #
# --------------------------------------------------------------------------- #
def _probe_instance(args):
    campus, track, size, level, path = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]
    nwo = len(inst["work_orders"])

    # RULE objective solve (recorded w, recorded d) is beta-independent.
    #  * NAIVE: default CP-SAT (no secondary). Among recorded-optimal schedules
    #    CP-SAT procrastinates (late starts), which is systematically bad on the
    #    tighter d* -> inflates the RULE-vs-ORACLE gap. Kept only to quantify the
    #    tie-breaking ARTIFACT.
    #  * FAIR: flow_tiebreak=True breaks ties toward finishing early (minimise
    #    sum C_j, a deadline-BLIND secondary). This is what a sensible d-only
    #    planner does; the residual gap to the oracle is the real information
    #    value that a d*-blind planner cannot close.
    s_rule = cpsat.solve(inst, time_limit_s=PROBE_BUDGET_S, workers=PROBE_WORKERS)
    s_rule_fair = cpsat.solve(inst, time_limit_s=PROBE_BUDGET_S,
                              workers=PROBE_WORKERS, flow_tiebreak=True)
    status_rule = s_rule.get("status")
    status_rule_fair = s_rule_fair.get("status")

    out = []
    for beta in BETAS:
        applied = _overlay(beta).apply(inst)
        dstar = dstar_map(inst, applied)
        it = inst_with_dstar(inst, dstar)
        # ORACLE is optimal on TWT*(d*); its TWT* is tiebreak-invariant, so a
        # plain solve suffices for its true-objective value.
        s_oracle = cpsat.solve(it, time_limit_s=PROBE_BUDGET_S,
                               workers=PROBE_WORKERS)
        status_oracle = s_oracle.get("status")

        twt_rule_naive = score_twt_deadline(inst, s_rule, dstar)
        twt_rule_fair = score_twt_deadline(inst, s_rule_fair, dstar)
        twt_oracle = score_twt_deadline(inst, s_oracle, dstar)
        out.append({
            "campus": campus, "track": track, "size": size, "level": level,
            "beta": beta, "inst_id": inst_id, "nwo": nwo,
            "status_rule": status_rule, "status_rule_fair": status_rule_fair,
            "status_oracle": status_oracle,
            "both_opt": (status_rule == "OPTIMAL" and status_oracle == "OPTIMAL"),
            "both_opt_fair": (status_rule_fair == "OPTIMAL"
                              and status_oracle == "OPTIMAL"),
            "twt_rule": twt_rule_naive,          # naive baseline (artifact)
            "twt_rule_fair": twt_rule_fair,      # defensible baseline
            "twt_oracle": twt_oracle,
        })
    return out


def run_probe():
    tasks = []
    for campus in PROBE_CAMPUSES:
        for track, size, level in PROBE_CELLS:
            for p in _cell_files(campus, track, size, level, cap=PROBE_N):
                tasks.append((campus, track, size, level, p))
    print("[probe] instance-tasks: %d (x%d betas, 2 solves each)"
          % (len(tasks), len(BETAS)))

    records = []
    # workers=2 per solve x pool=4 => <= 8 threads.
    with ProcessPoolExecutor(max_workers=4) as ex:
        done = 0
        for res in ex.map(_probe_instance, tasks):
            records.extend(res)
            done += 1
            if done % 20 == 0:
                print("  processed %d/%d" % (done, len(tasks)))

    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "probe_perinstance.csv"), "w", newline="") as fh:
        cols = ["campus", "track", "size", "level", "beta", "inst_id", "nwo",
                "status_rule", "status_rule_fair", "status_oracle", "both_opt",
                "both_opt_fair", "twt_rule", "twt_rule_fair", "twt_oracle"]
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        for r in records:
            row = dict(r)
            for k in ("twt_rule", "twt_rule_fair", "twt_oracle"):
                row[k] = "%.4f" % row[k]
            wtr.writerow(row)

    # Aggregate per (campus, cell, beta) and pooled-loaded. Report BOTH the naive
    # (artifact) and fair (flow-tiebreak, defensible) RULE baselines.
    def _agg(sub):
        n = len(sub)
        if not n:
            return None
        mr = sum(r["twt_rule"] for r in sub) / n
        mrf = sum(r["twt_rule_fair"] for r in sub) / n
        mo = sum(r["twt_oracle"] for r in sub) / n
        pct = (100.0 * (mr - mo) / mr) if mr > 1e-9 else 0.0
        pctf = (100.0 * (mrf - mo) / mrf) if mrf > 1e-9 else 0.0
        w = t = l = 0                       # W/T/L on the FAIR baseline
        for r in sub:
            d = r["twt_rule_fair"] - r["twt_oracle"]
            if abs(d) <= TIE_TOL:
                t += 1
            elif d > TIE_TOL:
                w += 1
            else:
                l += 1
        return dict(n=n, mean_rule=mr, mean_rule_fair=mrf, mean_oracle=mo,
                    pct_naive=pct, pct_fair=pctf, wins=w, ties=t, losses=l,
                    all_opt=all(r["both_opt"] for r in sub),
                    all_opt_fair=all(r["both_opt_fair"] for r in sub))

    print("\n=== NON-MYOPIC CP-SAT probe (RULE-opt vs ORACLE-opt on d*) ===")
    print("cell                  campus beta  n meanRULEfair meanORA  %hdFAIR  "
          "%hdNAIVE  W/T/L allOptF")
    agg = []
    for track, size, level in PROBE_CELLS:
        cell = "%s/%s/%s" % (track, size, level)
        for beta in BETAS:
            for campus in PROBE_CAMPUSES:
                sub = [r for r in records if r["campus"] == campus
                       and r["track"] == track and r["level"] == level
                       and r["beta"] == beta]
                a = _agg(sub)
                if a:
                    agg.append(dict(scope="campus", campus=str(campus), cell=cell,
                                    beta=beta, **a))
                    print("%-21s %6s %.2f %3d %11.2f %8.2f %8.3f %8.2f %d/%d/%d %s"
                          % (cell, campus, beta, a["n"], a["mean_rule_fair"],
                             a["mean_oracle"], a["pct_fair"], a["pct_naive"],
                             a["wins"], a["ties"], a["losses"], a["all_opt_fair"]))
            sub = [r for r in records if r["campus"] in LOADED
                   and r["track"] == track and r["level"] == level
                   and r["beta"] == beta]
            a = _agg(sub)
            if a:
                agg.append(dict(scope="pooled_loaded", campus="9+12", cell=cell,
                                beta=beta, **a))
                print("%-21s %6s %.2f %3d %11.2f %8.2f %8.3f %8.2f %d/%d/%d %s <pool"
                      % (cell, "9+12", beta, a["n"], a["mean_rule_fair"],
                         a["mean_oracle"], a["pct_fair"], a["pct_naive"],
                         a["wins"], a["ties"], a["losses"], a["all_opt_fair"]))

    with open(os.path.join(_OUT, "probe_agg.csv"), "w", newline="") as fh:
        cols = ["scope", "campus", "cell", "beta", "n", "mean_rule",
                "mean_rule_fair", "mean_oracle", "pct_naive", "pct_fair",
                "wins", "ties", "losses", "all_opt", "all_opt_fair"]
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        for r in agg:
            o = dict(r)
            for k in ("mean_rule", "mean_rule_fair", "mean_oracle"):
                o[k] = "%.4f" % o[k]
            o["pct_naive"] = "%.4f" % o["pct_naive"]
            o["pct_fair"] = "%.4f" % o["pct_fair"]
            wtr.writerow({c: o[c] for c in cols})

    import pickle
    with open(os.path.join(_OUT, "_probe_agg.pkl"), "wb") as fh:
        pickle.dump(agg, fh)
    print("[probe] wrote %s" % _OUT)
    return agg


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("myopic", "both"):
        run_myopic()
    if mode in ("probe", "both"):
        run_probe()


if __name__ == "__main__":
    main()
