"""Y3 P1 headroom scan.

Headroom = the ceiling on how much a dispatcher can gain from the supervisor's
private urgency information, measured as the true-objective (TWT*) gap between:

  (a) ATC on the RECORDED weights   -- what a rule that never sees the latent does
  (b) ORACLE-GREEDY (true-weight ATC) -- the myopic-greedy skyline that acts on w*

Both are scored on the TRUE objective (fmwos.hitl.true_objective.score_true).
Grid: campus in {5,9,10,12} x size in {150,400} x beta in {0.5,0.75,1.0};
F-NL family, sigma_s=1.0, master_seed=12345, epsilon=0, access OFF. Per
(campus,size) cell we use up to 100 replay + up to 100 generator instances.

ATC's schedule is beta-independent (it never touches the latent); only its TWT*
score changes with beta (via w*). ORACLE-GREEDY's schedule changes with beta.

Run:  PYTHONPATH=src nice python scripts/y3_headroom_scan.py
"""

import os

# Single-threaded numeric libs BEFORE numpy import (10 worker processes).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import glob
import json
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
CAMPUSES = (5, 9, 10, 12)
LOADED = (9, 12)               # capacity-binding campuses (signal-carrying)
SIZES = (150, 400)
TIE_TOL = 1.0                  # tie band on TWT* diff (weighted units)
N_PER_TRACK = 100

# Overlays are process-global (coeffs read once from the recorded file).
_OVERLAYS = {}


def _overlay(beta):
    ov_ = _OVERLAYS.get(beta)
    if ov_ is None:
        ov_ = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                          master_seed=MASTER_SEED))
        _OVERLAYS[beta] = ov_
    return ov_


def _cell_paths(campus, size):
    cdir = "c%02d" % campus
    paths = []
    for track in ("replay", "generator"):
        g = sorted(glob.glob(os.path.join(_INST, cdir, track, str(size), "*.json")))
        for p in g[:N_PER_TRACK]:
            paths.append((track, p))
    return paths


def _process_instance(args):
    campus, size, track, path = args
    inst = json.load(open(path))
    inst_id = inst["meta"]["id"]

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
            "campus": campus, "size": size, "beta": beta,
            "track": track, "inst_id": inst_id,
            "twt_atc": atc_twt, "twt_oracle": or_twt,
        })
    return out


def _stats(records):
    """records: list of per-instance dicts (single scope, single beta)."""
    n = len(records)
    sum_atc = sum(r["twt_atc"] for r in records)
    sum_or = sum(r["twt_oracle"] for r in records)
    mean_atc = sum_atc / n if n else 0.0
    mean_or = sum_or / n if n else 0.0
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
    return {
        "n": n, "mean_atc": mean_atc, "mean_oracle": mean_or,
        "abs_gap": abs_gap, "pct_gap": pct_gap,
        "oracle_wins": wins, "ties": ties, "oracle_losses": losses,
    }


def main():
    tasks = []
    for campus in CAMPUSES:
        for size in SIZES:
            for track, path in _cell_paths(campus, size):
                tasks.append((campus, size, track, path))
    print("total instance-tasks: %d (x%d betas)" % (len(tasks), len(BETAS)))

    records = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        done = 0
        for res in ex.map(_process_instance, tasks, chunksize=4):
            records.extend(res)
            done += 1
            if done % 200 == 0:
                print("  processed %d/%d instances" % (done, len(tasks)))

    # ------- aggregate -------
    rows = []

    # per (campus,size,beta)
    for campus in CAMPUSES:
        for size in SIZES:
            for beta in BETAS:
                sub = [r for r in records if r["campus"] == campus
                       and r["size"] == size and r["beta"] == beta]
                st = _stats(sub)
                rows.append(dict(scope="campus", campus=str(campus),
                                 size=size, beta=beta, **st))

    # pooled over all campuses, per (size,beta)
    for size in SIZES:
        for beta in BETAS:
            sub = [r for r in records if r["size"] == size and r["beta"] == beta]
            st = _stats(sub)
            rows.append(dict(scope="pooled_all", campus="5+9+10+12",
                             size=size, beta=beta, **st))

    # pooled over loaded campuses {9,12}, per (size,beta)
    for size in SIZES:
        for beta in BETAS:
            sub = [r for r in records if r["campus"] in LOADED
                   and r["size"] == size and r["beta"] == beta]
            st = _stats(sub)
            rows.append(dict(scope="pooled_loaded", campus="9+12",
                             size=size, beta=beta, **st))

    # ------- write CSV -------
    out_csv = os.path.join(_ROOT, "results", "y3_p1", "headroom_scan.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = ["scope", "campus", "size", "beta", "n", "mean_atc", "mean_oracle",
            "abs_gap", "pct_gap", "oracle_wins", "ties", "oracle_losses"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            out = dict(row)
            for k in ("mean_atc", "mean_oracle", "abs_gap"):
                out[k] = "%.4f" % out[k]
            out["pct_gap"] = "%.4f" % out["pct_gap"]
            w.writerow({c: out[c] for c in cols})
    print("wrote %s (%d rows)" % (out_csv, len(rows)))

    # stash rows for the markdown writer
    import pickle
    with open(os.path.join(_ROOT, "results", "y3_p1", "_headroom_rows.pkl"), "wb") as fh:
        pickle.dump(rows, fh)


if __name__ == "__main__":
    main()
