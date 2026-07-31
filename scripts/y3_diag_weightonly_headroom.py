"""Y3 diagnostic: independent re-verification of weight-only latent myopic headroom.

Re-runs, from scratch (NOT reading any prior CSV), the ORACLE-GREEDY vs ATC
headroom on the CURRENT true objective for the LOCKED gate cells:

  (a) ATC        -- Apparent Tardiness Cost on the RECORDED weights w(c).
  (b) ORACLE-GREEDY -- ATC computed with the TRUE weights w*(c*) (myopic-greedy
                       skyline; supervisor.preferred_pick, rho=0 so no review).

Both schedules are scored on the TRUE objective, computed INLINE here (my own
scoring, not fmwos.hitl.true_objective) so the check is independent of that file:

  TWT* = sum_j w*(c*_j) * max(0, C_j - d_j),  due dates d_j UNCHANGED, access OFF.

Grid (from the task): campus in {5,9,10,12} x size 150 x beta in {0.75,1.0};
F-NL, sigma_s=1.0, master_seed=12345, epsilon=0, access OFF. Per cell up to 100
replay + 100 generator instances. Tie band = 1.0 weighted unit.

The env, deciders, overlay and supervisor (all LOCKED files) are imported and
reused to GENERATE schedules and the latent; only the scoring is re-implemented.

Run:  OMP_NUM_THREADS=1 PYTHONPATH=src nice python scripts/y3_diag_weightonly_headroom.py
"""

import os

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

_INST = os.path.join(_ROOT, "data", "processed", "instances")
SEED = 301           # dispatch seed (only affects the 'random' rule; inert here)
MASTER_SEED = 12345
FAMILY = "F-NL"
BETAS = (0.75, 1.0)
CAMPUSES = (5, 9, 10, 12)
LOADED = (9, 12)
SIZE = 150
TIE_TOL = 1.0
N_PER_TRACK = 100

_OVERLAYS = {}


def _overlay(beta):
    o = _OVERLAYS.get(beta)
    if o is None:
        o = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                        master_seed=MASTER_SEED))
        _OVERLAYS[beta] = o
    return o


def _cell_paths(campus, size):
    cdir = "c%02d" % campus
    paths = []
    for track in ("replay", "generator"):
        g = sorted(glob.glob(os.path.join(_INST, cdir, track, str(size), "*.json")))
        for p in g[:N_PER_TRACK]:
            paths.append((track, p))
    return paths


def score_twt_star(inst, sched, applied):
    """INDEPENDENT true-objective scorer.

    TWT* = sum_j w*(c*_j) * max(0, C_j - d_j), due dates unchanged, no access.
    Reads w*(c*) from the overlay's per-order map; C_j = assignment end_bh.
    """
    wstar = applied["w_star"]
    wo_by_id = {wo["id"]: wo for wo in inst["work_orders"]}
    twt = 0.0
    for a in sched.get("assignments", []):
        wid = a.get("wo")
        end = a.get("end_bh")
        wo = wo_by_id.get(wid)
        if wo is None or end is None:
            continue
        tard = float(end) - float(wo["due_bh"])
        if tard > 0.0:
            twt += wstar.get(wid, float(wo["weight"])) * tard
    return twt


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
        atc_twt = score_twt_star(inst, s_atc, applied)

        # ORACLE-GREEDY: true-weight ATC. rho=0 => no review, pure preferred pick.
        sup = Supervisor(overlay, inst, rho=0.0, epsilon=0.0, applied=applied)
        s_or = dec.run_oracle_greedy(DispatchEnv(inst), sup, seed=SEED)
        or_twt = score_twt_star(inst, s_or, applied)

        out.append({
            "campus": campus, "size": size, "beta": beta,
            "track": track, "inst_id": inst_id,
            "twt_atc": atc_twt, "twt_oracle": or_twt,
        })
    return out


def _stats(records):
    n = len(records)
    if not n:
        return dict(n=0, mean_atc=0.0, mean_oracle=0.0, abs_gap=0.0,
                    pct_gap=0.0, wins=0, ties=0, losses=0, tie_frac=0.0)
    sum_atc = sum(r["twt_atc"] for r in records)
    sum_or = sum(r["twt_oracle"] for r in records)
    mean_atc = sum_atc / n
    mean_or = sum_or / n
    abs_gap = mean_atc - mean_or                        # +: oracle better (lower TWT*)
    pct_gap = (100.0 * abs_gap / mean_atc) if mean_atc > 1e-9 else 0.0
    wins = ties = losses = 0
    for r in records:
        diff = r["twt_atc"] - r["twt_oracle"]            # +: oracle better
        if abs(diff) <= TIE_TOL:
            ties += 1
        elif diff > TIE_TOL:
            wins += 1
        else:
            losses += 1
    return dict(n=n, mean_atc=mean_atc, mean_oracle=mean_or, abs_gap=abs_gap,
                pct_gap=pct_gap, wins=wins, ties=ties, losses=losses,
                tie_frac=100.0 * ties / n)


def main():
    tasks = []
    for campus in CAMPUSES:
        for track, path in _cell_paths(campus, SIZE):
            tasks.append((campus, SIZE, track, path))
    print("total instance-tasks: %d (x%d betas)" % (len(tasks), len(BETAS)))

    records = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        done = 0
        for res in ex.map(_process_instance, tasks, chunksize=4):
            records.extend(res)
            done += 1
            if done % 200 == 0:
                print("  processed %d/%d instances" % (done, len(tasks)))

    rows = []
    # per (campus, beta)
    for campus in CAMPUSES:
        for beta in BETAS:
            sub = [r for r in records if r["campus"] == campus and r["beta"] == beta]
            rows.append(dict(scope="campus", campus=str(campus), beta=beta,
                             **_stats(sub)))
    # pooled over ALL gate campuses per beta
    for beta in BETAS:
        sub = [r for r in records if r["beta"] == beta]
        rows.append(dict(scope="pooled_all", campus="5+9+10+12", beta=beta,
                         **_stats(sub)))
    # pooled over LOADED {9,12} per beta
    for beta in BETAS:
        sub = [r for r in records if r["campus"] in LOADED and r["beta"] == beta]
        rows.append(dict(scope="pooled_loaded", campus="9+12", beta=beta,
                         **_stats(sub)))
    # grand pool over all campuses x all betas (single headline number)
    rows.append(dict(scope="pooled_grand", campus="5+9+10+12", beta="0.75+1.0",
                     **_stats(records)))
    # grand pool over loaded campuses x all betas
    sub = [r for r in records if r["campus"] in LOADED]
    rows.append(dict(scope="pooled_loaded_grand", campus="9+12", beta="0.75+1.0",
                     **_stats(sub)))

    out_dir = os.path.join(_ROOT, "results", "y3_diag", "weightonly_headroom")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "headroom_size150_beta075_10.csv")
    cols = ["scope", "campus", "beta", "n", "mean_atc", "mean_oracle", "abs_gap",
            "pct_gap", "wins", "ties", "losses", "tie_frac"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            o = dict(row)
            for k in ("mean_atc", "mean_oracle", "abs_gap"):
                o[k] = "%.4f" % o[k]
            o["pct_gap"] = "%.4f" % o["pct_gap"]
            o["tie_frac"] = "%.2f" % o["tie_frac"]
            w.writerow({c: o[c] for c in cols})
    print("wrote %s (%d rows)" % (out_csv, len(rows)))

    # console print
    def fmt(r):
        return ("%-16s c=%-9s beta=%-9s n=%-4d ATC=%10.3f OR=%10.3f "
                "gap=%+8.4f (%+.4f%%) W/T/L=%d/%d/%d tie%%=%.2f"
                % (r["scope"], str(r["campus"]), str(r["beta"]), r["n"],
                   r["mean_atc"], r["mean_oracle"], r["abs_gap"], r["pct_gap"],
                   r["wins"], r["ties"], r["losses"], r["tie_frac"]))
    print("\n=== PER (campus, beta) ===")
    for r in rows:
        if r["scope"] == "campus":
            print(fmt(r))
    print("\n=== POOLED ===")
    for r in rows:
        if r["scope"] != "campus":
            print(fmt(r))


if __name__ == "__main__":
    main()
