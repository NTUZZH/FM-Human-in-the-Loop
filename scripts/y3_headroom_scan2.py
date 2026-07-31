"""Y3 P1 headroom scan v2: capacity-stress tracks.

Extends scripts/y3_headroom_scan.py. v1 scanned replay/generator instances at
Y1's calibrated capacity and found ~zero information headroom (almost nothing is
tardy, so relabelling the class weight w->w* rarely reorders the ATC queue).

HYPOTHESIS tested here: under CAPACITY STRESS tardiness becomes widespread and
the supervisor's private urgency information gains real value. This scan scores
the same skyline gap on the high-load benchmark tracks:

  (a) ATC on the RECORDED weights   -- rule that never sees the latent
  (b) ORACLE-GREEDY (true-weight ATC) -- myopic-greedy skyline acting on w*

both on the TRUE objective TWT* (fmwos.hitl.true_objective.score_true).

STRESS TRACKS (per campus, under data/processed/instances/cNN/):
  * storm/{150,400}   -- id  cNN_storm_{size}_a{A}_c{C}_####
                         A = arrival_multiplier*100 in {125,150,200,300}
                         C = crew_multiplier*100 in {80,100}. 30 inst / cell.
  * storm2/w80        -- id  cNN_storm2_w80_u{U}_####    (NO 150/400 variant;
                         big instances, ~1500-3500 work orders each)
                         U = u_target*100 in {70,90,100,110,130}; the load
                         knob. arrival_multiplier is a derived meta field
                         (4.19..7.78). 30 inst / level.
  * pmmix/{150,400}   -- id  cNN_pmmix_{size}_p{P}_c{C}_####
                         P = pm_share*100 in {20,50,80}, C in {60,80,100}.
                         arrival_multiplier=1.0 (crew is the stress knob).

NOTE on the task brief: the "storm2 at arrival multipliers 1.25/1.5/2/3"
description actually matches the *storm* track (the a-levels a125/a150/a200/
a300). storm2 is a distinct big-instance track parameterised by a utilisation
target u_target (u70..u130), encoded as u{U} in the id; arrival_multiplier is a
derived field, not the load label. Both are scanned here.

Each (campus, track, size, load-level, beta) cell uses up to 100 instances (all
30 available per cell here). ATC's schedule is beta-independent; ORACLE-GREEDY's
changes with beta. Tie band on the per-instance TWT* diff = 1.0 weighted unit.

Run:  PYTHONPATH=src nice python scripts/y3_headroom_scan2.py
"""

import os

# Single-threaded numeric libs BEFORE numpy import (worker processes).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import glob
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.env import DispatchEnv                 # noqa: E402
from fmwos.hitl import deciders as dec            # noqa: E402
from fmwos.hitl import overlay as ov              # noqa: E402
from fmwos.hitl.supervisor import Supervisor      # noqa: E402
from fmwos.hitl.true_objective import score_true  # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
SEED = 301
MASTER_SEED = 12345
FAMILY = "F-NL"
BETAS = (0.5, 0.75, 1.0)
CAMPUSES = (5, 9, 10, 12)       # {9,12} loaded; {5,10} over-resourced (cheap here)
LOADED = (9, 12)
TIE_TOL = 1.0
N_PER_CELL = 100
MAX_WORKERS = 10

# Regexes to pull the load level out of the instance id.
_RE_STORM = re.compile(r"_storm_\d+_a(\d+)_c(\d+)_")
_RE_STORM2 = re.compile(r"_storm2_w80_u(\d+)_")
_RE_PMMIX = re.compile(r"_pmmix_\d+_p(\d+)_c(\d+)_")

_OVERLAYS = {}


def _overlay(beta):
    ov_ = _OVERLAYS.get(beta)
    if ov_ is None:
        ov_ = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                          master_seed=MASTER_SEED))
        _OVERLAYS[beta] = ov_
    return ov_


def _load_level(track, inst_id):
    """Return (level_tag, level_order) for sorting; None if unparsable."""
    if track == "storm":
        m = _RE_STORM.search(inst_id)
        if m:
            a, c = int(m.group(1)), int(m.group(2))
            # order: arrival dominant, then crew reduction (lower crew = harder)
            return "a%d_c%d" % (a, c), (a, -c)
    elif track == "storm2":
        m = _RE_STORM2.search(inst_id)
        if m:
            u = int(m.group(1))
            return "u%d" % u, (u,)
    elif track == "pmmix":
        m = _RE_PMMIX.search(inst_id)
        if m:
            p, c = int(m.group(1)), int(m.group(2))
            return "p%d_c%d" % (p, c), (-c, p)   # lower crew = harder
    return None, None


def _cell_files(campus, track, size):
    cdir = "c%02d" % campus
    if track == "storm2":
        d = os.path.join(_INST, cdir, track, "w80")
    else:
        d = os.path.join(_INST, cdir, track, str(size))
    return sorted(glob.glob(os.path.join(d, "*.json")))


def _tasks_for(campus):
    """Enumerate (campus, track, size, path); cap N_PER_CELL per load-level."""
    out = []
    specs = [("storm", (150, 400)), ("storm2", ("w80",)), ("pmmix", (150, 400))]
    for track, sizes in specs:
        for size in sizes:
            files = _cell_files(campus, track, size)
            per_level = {}
            for p in files:
                lvl, _ = _load_level(track, os.path.basename(p))
                if lvl is None:
                    continue
                bucket = per_level.setdefault(lvl, [])
                if len(bucket) < N_PER_CELL:
                    bucket.append(p)
            for lvl, paths in per_level.items():
                for p in paths:
                    out.append((campus, track, size, p))
    return out


def _process_instance(args):
    campus, track, size, path = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]
    level, _ = _load_level(track, inst_id)
    arr_mult = float(inst["meta"].get("arrival_multiplier", 0.0))

    # ATC on recorded weights -- schedule is beta-independent.
    s_atc = dec.run_rule(DispatchEnv(inst), "atc", seed=SEED)

    out = []
    for beta in BETAS:
        overlay = _overlay(beta)
        applied = overlay.apply(inst)
        atc_twt = score_true(inst, s_atc, overlay, applied=applied)["TWT_true"]

        sup_pref = Supervisor(overlay, inst, rho=0.0, applied=applied)
        s_or = dec.run_oracle_greedy(DispatchEnv(inst), sup_pref, seed=SEED)
        or_twt = score_true(inst, s_or, overlay, applied=applied)["TWT_true"]

        out.append({
            "campus": campus, "track": track, "size": str(size),
            "level": level, "arr_mult": arr_mult, "beta": beta,
            "inst_id": inst_id, "twt_atc": atc_twt, "twt_oracle": or_twt,
        })
    return out


def _stats(records):
    n = len(records)
    if not n:
        return dict(n=0, mean_atc=0.0, mean_oracle=0.0, abs_gap=0.0,
                    pct_gap=0.0, oracle_wins=0, ties=0, oracle_losses=0)
    sum_atc = sum(r["twt_atc"] for r in records)
    sum_or = sum(r["twt_oracle"] for r in records)
    mean_atc = sum_atc / n
    mean_or = sum_or / n
    abs_gap = mean_atc - mean_or
    pct_gap = (100.0 * abs_gap / mean_atc) if mean_atc > 1e-9 else 0.0
    wins = ties = losses = 0
    for r in records:
        diff = r["twt_atc"] - r["twt_oracle"]   # positive => oracle better
        if abs(diff) <= TIE_TOL:
            ties += 1
        elif diff > TIE_TOL:
            wins += 1
        else:
            losses += 1
    return dict(n=n, mean_atc=mean_atc, mean_oracle=mean_or, abs_gap=abs_gap,
                pct_gap=pct_gap, oracle_wins=wins, ties=ties,
                oracle_losses=losses)


def main():
    tasks = []
    for campus in CAMPUSES:
        tasks.extend(_tasks_for(campus))
    print("total instance-tasks: %d (x%d betas)" % (len(tasks), len(BETAS)))

    records = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        done = 0
        for res in ex.map(_process_instance, tasks, chunksize=4):
            records.extend(res)
            done += 1
            if done % 300 == 0:
                print("  processed %d/%d instances" % (done, len(tasks)))

    # mean arrival multiplier per (track, level) for reference
    arr_by_level = {}
    for r in records:
        arr_by_level.setdefault((r["track"], r["level"]), []).append(r["arr_mult"])
    arr_by_level = {k: (sum(v) / len(v)) for k, v in arr_by_level.items()}

    # distinct (track,size,level) cells, ordered by load
    cells = {}
    for r in records:
        key = (r["track"], r["size"], r["level"])
        cells.setdefault(key, None)
    def _level_order(track, level):
        nums = [int(x) for x in re.findall(r"\d+", level)]
        if track == "storm":       # a, c -> arrival dominant, lower crew harder
            return (nums[0], -nums[1])
        if track == "storm2":      # u_target
            return (nums[0],)
        if track == "pmmix":       # p, c -> lower crew harder, then pm share
            return (-nums[1], nums[0])
        return tuple(nums)
    def _cell_sort(key):
        track, size, level = key
        return (track, size, _level_order(track, level))
    cell_keys = sorted(cells.keys(), key=_cell_sort)

    rows = []

    # per (campus, track, size, level, beta)
    for campus in CAMPUSES:
        for (track, size, level) in cell_keys:
            for beta in BETAS:
                sub = [r for r in records if r["campus"] == campus
                       and r["track"] == track and r["size"] == size
                       and r["level"] == level and r["beta"] == beta]
                if not sub:
                    continue
                st = _stats(sub)
                rows.append(dict(scope="campus", campus=str(campus), track=track,
                                 size=size, level=level,
                                 arr_mult=arr_by_level.get((track, level), 0.0),
                                 beta=beta, **st))

    # pooled over loaded campuses {9,12}, per (track,size,level,beta)
    for (track, size, level) in cell_keys:
        for beta in BETAS:
            sub = [r for r in records if r["campus"] in LOADED
                   and r["track"] == track and r["size"] == size
                   and r["level"] == level and r["beta"] == beta]
            if not sub:
                continue
            st = _stats(sub)
            rows.append(dict(scope="pooled_loaded", campus="9+12", track=track,
                             size=size, level=level,
                             arr_mult=arr_by_level.get((track, level), 0.0),
                             beta=beta, **st))

    out_csv = os.path.join(_ROOT, "results", "y3_p1", "headroom_scan2.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = ["scope", "campus", "track", "size", "level", "arr_mult", "beta",
            "n", "mean_atc", "mean_oracle", "abs_gap", "pct_gap",
            "oracle_wins", "ties", "oracle_losses"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            out = dict(row)
            for k in ("mean_atc", "mean_oracle", "abs_gap"):
                out[k] = "%.4f" % out[k]
            out["pct_gap"] = "%.4f" % out["pct_gap"]
            out["arr_mult"] = "%.3f" % out["arr_mult"]
            w.writerow({c: out[c] for c in cols})
    print("wrote %s (%d rows)" % (out_csv, len(rows)))

    import pickle
    with open(os.path.join(_ROOT, "results", "y3_p1",
                           "_headroom2_rows.pkl"), "wb") as fh:
        pickle.dump(rows, fh)


if __name__ == "__main__":
    main()
