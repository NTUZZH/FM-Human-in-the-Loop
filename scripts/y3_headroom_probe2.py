"""Y3 P1 headroom scan v2 -- non-myopic CP-SAT probe.

The v2 myopic scan (scripts/y3_headroom_scan2.py) shows ORACLE-GREEDY (myopic
true-weight ATC) extracts ~zero TWT* headroom over recorded ATC on the stress
tracks, and on the big storm2 instances it is net NEGATIVE and grows more
negative with load and beta (the myopic reordering fires but hurts). The open
question the gate needs: would a NON-myopic planner convert the private urgency
information into a real TWT* gain?

This probe answers it by running a CP-SAT planner twice on the SAME instance,
once with the RECORDED weights in its objective, once with the TRUE weights
w*, and scoring both schedules on the true objective TWT*. The gap
(recorded - true) is the information value for that planner. Two planners:

  * static   -- offline CP-SAT over the whole realized horizon (releases known
                up front). At <=400 work orders this solves to OPTIMAL in <3 s,
                so it is the true non-myopic CEILING: a proven optimum for each
                weight vector.
  * rolling  -- Y1's rollcp2 (fmwos.rolling.roll_cpsat, budget_s=2.0, workers=2):
                the same replan-on-arrival policy Y1 dyneval used.

Feeding w* to either planner: deep-copy the instance and overwrite each work
order's ``weight`` with w*(c*_j) (the planner objective is the only thing that
changes). Scoring is always done on the ORIGINAL instance under the overlay, so
due dates and w* are the realized ones.

storm2 note. The high-load storm2 track (u70..u130, ~1500-3574 work orders per
instance, NO size-150 variant) is where the myopic oracle's negative signal
lives, but it is out of reach for this probe: a single rolling run at
budget_s=2.0 exceeds 400 s (every replan hits the 2 s budget because the queued
snapshot never proves optimal), and a static solve returns status UNKNOWN with
no feasible incumbent even at a 30 s budget. So the non-myopic probe is reported
on the tractable highest-load storm cell (size 150, a300_c80) plus two more
tractable stressed cells; storm2 is documented, not run.

Run:  PYTHONPATH=src nice python scripts/y3_headroom_probe2.py
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
import time
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import cpsat, rolling                   # noqa: E402
from fmwos.hitl import overlay as ov               # noqa: E402
from fmwos.hitl.true_objective import score_true   # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
MASTER_SEED = 12345
FAMILY = "F-NL"
BETA = 1.0                      # strongest recoverable-info setting
N_INST = 20
CAMPUSES = (9, 12)
TIE_TOL = 1.0
STATIC_BUDGET_S = 5.0
ROLLING_BUDGET_S = 2.0          # Y1 rollcp2 budget

# (track, size, glob-tail) tractable stressed cells.
CELLS = [
    ("storm", "150", "storm/150/*a300_c80*"),   # highest-load storm at size 150
    ("storm", "400", "storm/400/*a200_c100*"),
    ("pmmix", "400", "pmmix/400/*p20_c100*"),
]


def _overlay():
    return ov.Overlay(ov.OverlayParams(beta=BETA, family=FAMILY,
                                       master_seed=MASTER_SEED))


def _true_weighted(inst, applied):
    wstar = applied["w_star"]
    it = copy.deepcopy(inst)
    for wo in it["work_orders"]:
        wo["weight"] = wstar[wo["id"]]
    return it


def _run_cell(args):
    campus, track, size, path, planners = args
    inst = json.load(open(path))
    ovl = _overlay()
    applied = ovl.apply(inst)
    it = _true_weighted(inst, applied)
    inst_id = inst["meta"]["id"]
    nwo = len(inst["work_orders"])
    out = []
    for planner in planners:
        if planner == "static":
            t0 = time.perf_counter()
            sr = cpsat.solve(inst, time_limit_s=STATIC_BUDGET_S, workers=2)
            st = cpsat.solve(it, time_limit_s=STATIC_BUDGET_S, workers=2)
            wall = time.perf_counter() - t0
            status = "%s/%s" % (sr.get("status"), st.get("status"))
        else:  # rolling
            t0 = time.perf_counter()
            sr = rolling.roll_cpsat(inst, budget_s=ROLLING_BUDGET_S)
            st = rolling.roll_cpsat(it, budget_s=ROLLING_BUDGET_S)
            wall = time.perf_counter() - t0
            status = "rollcp2"
        twt_rec = score_true(inst, sr, ovl, applied=applied)["TWT_true"]
        twt_true = score_true(inst, st, ovl, applied=applied)["TWT_true"]
        out.append({
            "planner": planner, "campus": campus, "track": track, "size": size,
            "inst_id": inst_id, "nwo": nwo, "status": status,
            "twt_recorded": twt_rec, "twt_true": twt_true, "wall_s": wall,
        })
    return out


def _agg(records):
    n = len(records)
    if not n:
        return None
    mr = sum(r["twt_recorded"] for r in records) / n
    mt = sum(r["twt_true"] for r in records) / n
    gap = mr - mt
    pct = (100.0 * gap / mr) if mr > 1e-9 else 0.0
    w = t = l = 0
    for r in records:
        d = r["twt_recorded"] - r["twt_true"]   # positive => true weights help
        if abs(d) <= TIE_TOL:
            t += 1
        elif d > TIE_TOL:
            w += 1
        else:
            l += 1
    mean_wall = sum(r["wall_s"] for r in records) / n
    return dict(n=n, mean_recorded=mr, mean_true=mt, abs_gap=gap, pct_gap=pct,
                true_wins=w, ties=t, true_losses=l, mean_wall_s=mean_wall)


def main():
    planners = ("static", "rolling")
    tasks = []
    for campus in CAMPUSES:
        cdir = "c%02d" % campus
        for track, size, tail in CELLS:
            files = sorted(glob.glob(os.path.join(_INST, cdir, tail + "*.json")))
            # rolling only on the size-150 highest-load cell (tractable); static
            # on all cells.
            use = []
            if size == "150":
                use = ["static", "rolling"]
            else:
                use = ["static"]
            for p in files[:N_INST]:
                tasks.append((campus, track, size, p, tuple(use)))
    print("probe tasks: %d instance-cells (beta=%.2f, n<=%d)"
          % (len(tasks), BETA, N_INST))

    records = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        done = 0
        for res in ex.map(_run_cell, tasks):
            records.extend(res)
            done += 1
            if done % 20 == 0:
                print("  %d/%d instance-cells" % (done, len(tasks)))

    # write per-instance rows
    out_csv = os.path.join(_ROOT, "results", "y3_p1", "headroom_probe2.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = ["planner", "campus", "track", "size", "inst_id", "nwo", "status",
            "twt_recorded", "twt_true", "wall_s"]
    with open(out_csv, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        for r in sorted(records, key=lambda x: (x["planner"], x["campus"],
                                                x["track"], x["size"],
                                                x["inst_id"])):
            row = dict(r)
            for k in ("twt_recorded", "twt_true"):
                row[k] = "%.4f" % row[k]
            row["wall_s"] = "%.3f" % row["wall_s"]
            wtr.writerow({c: row[c] for c in cols})
    print("wrote %s (%d rows)" % (out_csv, len(records)))

    # aggregate table
    print("\n=== PROBE AGGREGATE (per planner, campus, cell) ===")
    print("planner campus track size  n  meanREC   meanTRUE   gap    %gap   "
          "W/T/L    wall")
    agg_rows = []
    for planner in planners:
        for campus in CAMPUSES:
            for track, size, _ in CELLS:
                sub = [r for r in records if r["planner"] == planner
                       and r["campus"] == campus and r["track"] == track
                       and r["size"] == size]
                a = _agg(sub)
                if not a:
                    continue
                agg_rows.append(dict(planner=planner, campus=campus, track=track,
                                     size=size, **a))
                print("%-7s %6d %-6s %-4s %2d %8.2f %8.2f %7.3f %6.3f  "
                      "%d/%d/%d  %.2fs"
                      % (planner, campus, track, size, a["n"], a["mean_recorded"],
                         a["mean_true"], a["abs_gap"], a["pct_gap"],
                         a["true_wins"], a["ties"], a["true_losses"],
                         a["mean_wall_s"]))

    import pickle
    with open(os.path.join(_ROOT, "results", "y3_p1",
                           "_probe2_rows.pkl"), "wb") as fh:
        pickle.dump({"records": records, "agg": agg_rows}, fh)


if __name__ == "__main__":
    main()
