"""Y3 diagnostic: weight-only channel STRESS probe (DEAD CONTROL).

Question. The locked latent moves the tardiness WEIGHT only, freezing due dates
(overlay: c*_j = clip(c_j - s_j, 1, 4); w*_j = w(c*_j); d_j unchanged). Two
independent headroom scans + a CP-SAT probe already found ~0 decision leverage
for this weight-only channel (notes/decisions.md PIVOTAL FINDING). This script
asks whether the weight channel can be REVIVED by two knobs, keeping due dates
FROZEN:

  (a) a larger class-shift scale  sigma_s in {1.0, 2.0}   (bigger s_j -> c*_j
      differs from c_j more often and by more)
  (b) a MORE SKEWED weight vector  w in {8/4/2/1 (baseline), 16/4/2/1, 27/9/3/1}
      (amplifies the difference between w(c_j) and w(c*_j) whenever c_j != c*_j)

Objective and deciders (this variant):

    TWT* = sum_j  w(c*_j) * max(0, C_j - d_j)          d_j = recorded due (frozen)
    RULE   = ATC dispatched on the RECORDED weight  w(c_j)   (never sees latent)
    ORACLE = ATC dispatched on the TRUE     weight  w(c*_j)  (myopic-greedy skyline)

Both the RULE and the ORACLE use the SAME weight vector w (the "skew" knob moves
the whole class->weight map); the only difference between them is recorded class
c_j vs true class c*_j. Both are scored on TWT*. This is exactly the P1/P2
ORACLE-GREEDY-vs-ATC framing, generalized over (sigma_s, w).

We reuse the LOCKED overlay's exact xi_j / s_j draw (F-NL, master_seed=12345),
setting OverlayParams.sigma_s to 1.0 or 2.0; c*_j = clip(c_j - s_j, 1, 4) comes
from overlay.apply(). The weight vector is applied in this scratch script only:
recorded weight = W_SKEW[priority] (overwritten in a copy of the instance for the
RULE's ATC), true weight = W_SKEW[c*_j] (injected as the Supervisor's w_star for
the ORACLE, and used in the TWT* scorer copied below).

PHASE 1 (myopic): campuses {9,12} loaded + {5,10} control; size-150 loaded cells
storm a300_c80, storm a200_c80, pmmix p20_c60; 30 inst/cell; w x sigma_s x beta.
PHASE 2 (non-myopic CP-SAT): the strongest knob setting (sigma_s=2.0, w=27/9/3/1,
beta=1.0) on the two tractable loaded cells (solve to OPTIMAL in seconds); does an
OPTIMIZING planner convert the private urgency info into a TWT* gain?

Run:
  conda activate fjsp
  OMP_NUM_THREADS=1 nice python scripts/y3_diag_weight-only-stress.py --phase myopic
  OMP_NUM_THREADS=1 nice python scripts/y3_diag_weight-only-stress.py --phase cpsat
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import copy
import csv
import glob
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.env import DispatchEnv                 # noqa: E402
from fmwos.hitl import deciders as dec            # noqa: E402
from fmwos.hitl import overlay as ov              # noqa: E402
from fmwos.hitl.supervisor import Supervisor      # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_diag", "weight-only-stress")

SEED = 301
MASTER_SEED = 12345
FAMILY = "F-NL"

# --- knobs -------------------------------------------------------------------
# Weight vectors as class(1..4) -> weight. baseline reproduces the locked overlay
# W_OF_CLASS (8/4/2/1); the two skews concentrate weight on class 1 (and 2).
WEIGHTS = {
    "w8421":  {1: 8.0,  2: 4.0, 3: 2.0, 4: 1.0},   # baseline == overlay W_OF_CLASS
    "w16421": {1: 16.0, 2: 4.0, 3: 2.0, 4: 1.0},
    "w27931": {1: 27.0, 2: 9.0, 3: 3.0, 4: 1.0},
}
SIGMAS = (1.0, 2.0)
BETAS = (0.5, 0.75, 1.0)

CAMPUSES = (9, 12, 5, 10)          # {9,12} loaded; {5,10} over-resourced controls
LOADED = (9, 12)
TIE_TOL = 1.0
N_PER_CELL = 30

# (cell_tag, glob-tail) size-150 loaded cells (see notes/headroom_scan2.md).
CELLS = [
    ("storm150_a300_c80", "storm/150/*a300_c80*"),
    ("storm150_a200_c80", "storm/150/*a200_c80*"),
    ("pmmix150_p20_c60",  "pmmix/150/*p20_c60*"),
]

# Overlays are process-global, keyed on (sigma_s, beta); coeffs read once.
_OVERLAYS = {}


def _overlay(sigma_s, beta):
    key = (sigma_s, beta)
    o = _OVERLAYS.get(key)
    if o is None:
        o = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                        master_seed=MASTER_SEED, sigma_s=sigma_s))
        _OVERLAYS[key] = o
    return o


# --- TWT* scorer (copied from fmwos.hitl.true_objective.score_true metric block;
#     access OFF, due dates FROZEN=recorded). Independent of the env/solver. -----
def score_twt_star(inst, schedule, wstar_map):
    wo_by_id = {wo["id"]: wo for wo in inst["work_orders"]}
    total = 0.0
    for a in schedule.get("assignments", []) or []:
        wo = wo_by_id.get(a.get("wo"))
        end = a.get("end_bh")
        if wo is None or end is None:
            continue
        due = float(wo["due_bh"])
        tard = max(0.0, float(end) - due)
        total += float(wstar_map[wo["id"]]) * tard
    return total


def _recorded_weight_instance(inst, wvec):
    """Copy of ``inst`` with each wo['weight'] = wvec[recorded class] (the RULE's
    ATC then dispatches on the recorded skewed weights)."""
    it = copy.deepcopy(inst)
    for wo in it["work_orders"]:
        wo["weight"] = wvec[int(wo["priority"])]
    return it


def _process_instance_myopic(args):
    campus, cell_tag, path = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]

    out = []
    # RULE schedule depends ONLY on the weight vector (never on the latent), so
    # build one per weight vector.
    rule_sched = {}
    for wtag, wvec in WEIGHTS.items():
        rule_sched[wtag] = dec.run_rule(
            DispatchEnv(_recorded_weight_instance(inst, wvec)), "atc", seed=SEED)

    for sigma_s in SIGMAS:
        for beta in BETAS:
            overlay = _overlay(sigma_s, beta)
            applied = overlay.apply(inst)      # shift & c_star (latent, frozen d)
            cstar = applied["c_star"]
            for wtag, wvec in WEIGHTS.items():
                wstar_map = {wid: wvec[int(cstar[wid])] for wid in cstar}

                # RULE, scored on TWT* under this (sigma_s, beta, w).
                twt_rule = score_twt_star(inst, rule_sched[wtag], wstar_map)

                # ORACLE = myopic true-weight ATC. Inject skewed w_star into the
                # Supervisor via a custom `applied`; preferred_pick then ranks by
                # w(c*)/p * exp(-slack/(2 pbar)) (same ATC index the rule uses).
                custom = {"per_order": applied["per_order"],
                          "shift": applied["shift"],
                          "w_star": wstar_map, "c_star": cstar}
                sup = Supervisor(overlay, inst, rho=0.0, applied=custom)
                s_or = dec.run_oracle_greedy(DispatchEnv(inst), sup, seed=SEED)
                twt_or = score_twt_star(inst, s_or, wstar_map)

                out.append({
                    "campus": campus, "cell": cell_tag, "inst_id": inst_id,
                    "sigma_s": sigma_s, "beta": beta, "wtag": wtag,
                    "twt_rule": twt_rule, "twt_oracle": twt_or,
                })
    return out


def _stats(recs):
    n = len(recs)
    if not n:
        return dict(n=0, mean_rule=0.0, mean_oracle=0.0, abs_gap=0.0,
                    pct_gap=0.0, wins=0, ties=0, losses=0)
    sr = sum(r["twt_rule"] for r in recs)
    so = sum(r["twt_oracle"] for r in recs)
    mr = sr / n
    mo = so / n
    gap = mr - mo                       # positive => oracle better (lower TWT*)
    pct = (100.0 * gap / mr) if mr > 1e-9 else 0.0
    w = t = l = 0
    for r in recs:
        d = r["twt_rule"] - r["twt_oracle"]
        if abs(d) <= TIE_TOL:
            t += 1
        elif d > TIE_TOL:
            w += 1
        else:
            l += 1
    return dict(n=n, mean_rule=mr, mean_oracle=mo, abs_gap=gap, pct_gap=pct,
                wins=w, ties=t, losses=l)


def run_myopic():
    tasks = []
    for campus in CAMPUSES:
        cdir = "c%02d" % campus
        for cell_tag, tail in CELLS:
            files = sorted(glob.glob(os.path.join(_INST, cdir, tail + "*.json")))
            for p in files[:N_PER_CELL]:
                tasks.append((campus, cell_tag, p))
    print("myopic tasks: %d instances x %d (sigma,beta,w) configs"
          % (len(tasks), len(SIGMAS) * len(BETAS) * len(WEIGHTS)))

    records = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        done = 0
        for res in ex.map(_process_instance_myopic, tasks, chunksize=2):
            records.extend(res)
            done += 1
            if done % 60 == 0:
                print("  %d/%d instances" % (done, len(tasks)))

    # per-instance CSV
    os.makedirs(_OUT, exist_ok=True)
    inst_csv = os.path.join(_OUT, "myopic_perinstance.csv")
    cols = ["campus", "cell", "inst_id", "sigma_s", "beta", "wtag",
            "twt_rule", "twt_oracle"]
    with open(inst_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in records:
            row = dict(r)
            row["twt_rule"] = "%.4f" % row["twt_rule"]
            row["twt_oracle"] = "%.4f" % row["twt_oracle"]
            wr.writerow({c: row[c] for c in cols})
    print("wrote", inst_csv, "(%d rows)" % len(records))

    # aggregate rows: per (scope, cell, sigma, beta, w)
    agg_rows = []

    def _emit(scope, campus_lbl, subset_campuses):
        for cell_tag, _ in CELLS:
            for sigma_s in SIGMAS:
                for beta in BETAS:
                    for wtag in WEIGHTS:
                        sub = [r for r in records
                               if r["campus"] in subset_campuses
                               and r["cell"] == cell_tag
                               and r["sigma_s"] == sigma_s
                               and r["beta"] == beta and r["wtag"] == wtag]
                        st = _stats(sub)
                        agg_rows.append(dict(scope=scope, campus=campus_lbl,
                                             cell=cell_tag, sigma_s=sigma_s,
                                             beta=beta, wtag=wtag, **st))

    for campus in CAMPUSES:
        _emit("campus", str(campus), (campus,))
    _emit("pooled_loaded", "9+12", LOADED)

    agg_csv = os.path.join(_OUT, "myopic_agg.csv")
    acols = ["scope", "campus", "cell", "sigma_s", "beta", "wtag", "n",
             "mean_rule", "mean_oracle", "abs_gap", "pct_gap",
             "wins", "ties", "losses"]
    with open(agg_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=acols)
        wr.writeheader()
        for row in agg_rows:
            o = dict(row)
            for k in ("mean_rule", "mean_oracle", "abs_gap", "pct_gap"):
                o[k] = "%.4f" % o[k]
            wr.writerow({c: o[c] for c in acols})
    print("wrote", agg_csv, "(%d rows)" % len(agg_rows))

    # console summary: pooled-loaded, per (cell,sigma,w) at beta=1.0
    print("\n=== POOLED LOADED {9,12}  (beta=1.0) ===")
    print("cell                 sigma w        n  meanRULE  meanORAC   %gap   W/T/L")
    for cell_tag, _ in CELLS:
        for sigma_s in SIGMAS:
            for wtag in WEIGHTS:
                r = next(x for x in agg_rows if x["scope"] == "pooled_loaded"
                         and x["cell"] == cell_tag and x["sigma_s"] == sigma_s
                         and x["beta"] == 1.0 and x["wtag"] == wtag)
                print("%-20s %4.1f %-6s %3d %9.2f %9.2f %6.3f  %d/%d/%d"
                      % (cell_tag, sigma_s, wtag, r["n"], r["mean_rule"],
                         r["mean_oracle"], r["pct_gap"], r["wins"], r["ties"],
                         r["losses"]))

    # beta-monotonicity of |gap| (pooled loaded, strongest w & sigma)
    print("\n=== beta-monotonicity of |%gap| (pooled loaded) ===")
    for cell_tag, _ in CELLS:
        for sigma_s in SIGMAS:
            for wtag in WEIGHTS:
                gaps = []
                for beta in BETAS:
                    r = next(x for x in agg_rows if x["scope"] == "pooled_loaded"
                             and x["cell"] == cell_tag and x["sigma_s"] == sigma_s
                             and x["beta"] == beta and x["wtag"] == wtag)
                    gaps.append(r["pct_gap"])
                mono = abs(gaps[0]) <= abs(gaps[1]) + 1e-9 <= abs(gaps[2]) + 1e-9
                if sigma_s == 2.0 and wtag == "w27931":
                    print("  %-20s sigma=%.1f %-6s  |gap|@beta 0.5/0.75/1.0 = "
                          "%.3f/%.3f/%.3f  monotone=%s"
                          % (cell_tag, sigma_s, wtag, abs(gaps[0]), abs(gaps[1]),
                             abs(gaps[2]), mono))


# --------------------------------------------------------------------------- #
# PHASE 2: non-myopic CP-SAT check on the strongest knob setting              #
# --------------------------------------------------------------------------- #
CPSAT_SIGMA = 2.0
CPSAT_WTAG = "w27931"
CPSAT_BETA = 1.0
CPSAT_N = 20
CPSAT_BUDGET_S = 8.0
CPSAT_CELLS = [
    ("storm150_a300_c80", "storm/150/*a300_c80*"),
    ("pmmix150_p20_c60",  "pmmix/150/*p20_c60*"),
]


def _process_instance_cpsat(args):
    from fmwos import cpsat
    campus, cell_tag, path = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]
    wvec = WEIGHTS[CPSAT_WTAG]

    overlay = _overlay(CPSAT_SIGMA, CPSAT_BETA)
    applied = overlay.apply(inst)
    cstar = applied["c_star"]
    wstar_map = {wid: wvec[int(cstar[wid])] for wid in cstar}

    # RECORDED-optimal: CP-SAT minimizing sum w(c_j)*T_j (recorded skewed w).
    it_rec = _recorded_weight_instance(inst, wvec)
    # TRUE-optimal: CP-SAT minimizing sum w(c*_j)*T_j (true skewed w).
    it_true = copy.deepcopy(inst)
    for wo in it_true["work_orders"]:
        wo["weight"] = wstar_map[wo["id"]]

    t0 = time.perf_counter()
    sr = cpsat.solve(it_rec, time_limit_s=CPSAT_BUDGET_S, workers=2)
    st = cpsat.solve(it_true, time_limit_s=CPSAT_BUDGET_S, workers=2)
    wall = time.perf_counter() - t0

    twt_rec_opt = score_twt_star(inst, sr, wstar_map)      # recorded-opt on TWT*
    twt_true_opt = score_twt_star(inst, st, wstar_map)     # true-opt on TWT*
    return {
        "campus": campus, "cell": cell_tag, "inst_id": inst_id,
        "nwo": len(inst["work_orders"]),
        "status_rec": sr.get("status"), "status_true": st.get("status"),
        "twt_rule_opt": twt_rec_opt, "twt_oracle_opt": twt_true_opt,
        "wall_s": wall,
    }


def run_cpsat():
    tasks = []
    for campus in LOADED:
        cdir = "c%02d" % campus
        for cell_tag, tail in CPSAT_CELLS:
            files = sorted(glob.glob(os.path.join(_INST, cdir, tail + "*.json")))
            for p in files[:CPSAT_N]:
                tasks.append((campus, cell_tag, p))
    print("cpsat tasks: %d instances (sigma=%.1f, w=%s, beta=%.1f)"
          % (len(tasks), CPSAT_SIGMA, CPSAT_WTAG, CPSAT_BETA))

    records = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        done = 0
        for r in ex.map(_process_instance_cpsat, tasks):
            records.append(r)
            done += 1
            if done % 20 == 0:
                print("  %d/%d" % (done, len(tasks)))

    os.makedirs(_OUT, exist_ok=True)
    inst_csv = os.path.join(_OUT, "cpsat_perinstance.csv")
    cols = ["campus", "cell", "inst_id", "nwo", "status_rec", "status_true",
            "twt_rule_opt", "twt_oracle_opt", "wall_s"]
    with open(inst_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in records:
            row = dict(r)
            for k in ("twt_rule_opt", "twt_oracle_opt"):
                row[k] = "%.4f" % row[k]
            row["wall_s"] = "%.3f" % row["wall_s"]
            wr.writerow({c: row[c] for c in cols})
    print("wrote", inst_csv, "(%d rows)" % len(records))

    print("\n=== CP-SAT non-myopic (sigma=2.0, w=27/9/3/1, beta=1.0) ===")
    print("campus cell                 n  meanRULEopt meanORACopt   %gap   "
          "W/T/L  allOPT  wall")
    for campus in LOADED:
        for cell_tag, _ in CPSAT_CELLS:
            sub = [r for r in records if r["campus"] == campus
                   and r["cell"] == cell_tag]
            if not sub:
                continue
            n = len(sub)
            mr = sum(r["twt_rule_opt"] for r in sub) / n
            mo = sum(r["twt_oracle_opt"] for r in sub) / n
            gap = mr - mo
            pct = (100.0 * gap / mr) if mr > 1e-9 else 0.0
            w = t = l = 0
            for r in sub:
                d = r["twt_rule_opt"] - r["twt_oracle_opt"]
                if abs(d) <= TIE_TOL:
                    t += 1
                elif d > TIE_TOL:
                    w += 1
                else:
                    l += 1
            all_opt = all(r["status_rec"] == "OPTIMAL"
                          and r["status_true"] == "OPTIMAL" for r in sub)
            mw = sum(r["wall_s"] for r in sub) / n
            print("%6d %-20s %2d %11.2f %11.2f %6.3f  %d/%d/%d  %s  %.2fs"
                  % (campus, cell_tag, n, mr, mo, pct, w, t, l, all_opt, mw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("myopic", "cpsat"), default="myopic")
    a = ap.parse_args()
    if a.phase == "myopic":
        run_myopic()
    else:
        run_cpsat()


if __name__ == "__main__":
    main()
