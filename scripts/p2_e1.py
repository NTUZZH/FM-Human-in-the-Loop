#!/usr/bin/env python
"""E1 static-benchmark runner: full static evaluation of every method
on ALL test-split instances (replay AND generator tracks).

For each instance in data/processed/instances/index.csv with split=='test' we run
nine methods and validate every schedule with the independent referee
(``fmwos.validator``), then write one row per (instance, method):

    PDRs      : edd, wspt, atc, pfifo, mor, random   (seed 301)
    metaheur. : ga                                    (seed 301, 60 s wall budget)
    exact     : cpsat60, cpsat300                     (workers 2)

CP-SAT protocol optimisation
----------------------------
1. Run the six PDRs; pick the best *feasible* PDR schedule (min validator WWT).
2. Run cpsat60 (workers 2) **warm-started from that best PDR schedule**.
3. If cpsat60 proves OPTIMAL, cpsat300 is recorded as a *copy* of the cpsat60
   row (status OPTIMAL, identical numbers, wall_seconds == cpsat60's) with the
   flag column ``reused_from_60=True`` -- we never waste 300 s re-proving a
   solved instance.  Otherwise cpsat300 (workers 2) is solved, warm-started from
   the same best PDR schedule, with ``reused_from_60=False``.

Metrics come ONLY from the validator (method self-reports are never trusted).
Feasibility of every schedule is asserted: an infeasible schedule is a FATAL
error -- it is logged to results/e1_static/errors.log and counted, but the run
continues and the (feasible=0) row is still written.

Parallelism / restart-safety (shards design)
--------------------------------------------
A multiprocessing Pool runs over INSTANCES (default 10 workers, ``--workers``).
One task == one instance x all nine methods run sequentially; GA and CP-SAT are
single-threaded inside a task except CP-SAT's own ``workers=2``.  Each finished
instance is written atomically to its own shard results/e1_static/shards/<id>.json
(a shard holds all nine method rows).  This needs no lock and is fully restart
safe: on start we scan the shards and skip any instance whose shard already holds
all nine methods.  ``--merge`` (also auto-run at the end) concatenates the shards
into results/e1_static/results.csv and rebuilds errors.log.

Usage
-----
    PYTHONPATH=src python scripts/p2_e1.py [--workers N] [--limit N]
                                           [--campus C[,C...]] [--size S[,S...]]
                                           [--merge]

--limit N       cap the number of not-yet-done instances processed this run.
--campus/--size restrict the target set (smoke runs / chunked resumes).
--merge         only (re)build results.csv + errors.log from existing shards.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Keep BLAS/OpenMP from oversubscribing the shared box: the only threaded work
# we want is CP-SAT's own num_search_workers=2.  Set before numpy is imported
# (fmwos.validator imports numpy).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from fmwos import cpsat, ga, pdrs  # noqa: E402
from fmwos.validator import validate  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
INDEX_CSV = _ROOT / "data" / "processed" / "instances" / "index.csv"
OUT_DIR = _ROOT / "results" / "e1_static"
SHARD_DIR = OUT_DIR / "shards"
OUT_CSV = OUT_DIR / "results.csv"
META_JSON = OUT_DIR / "meta.json"
ERR_LOG = OUT_DIR / "errors.log"

PDR_RULES = ["edd", "wspt", "atc", "pfifo", "mor", "random"]
ALL_METHODS = PDR_RULES + ["ga", "cpsat60", "cpsat300"]  # nine methods, fixed order
_METHOD_ORDER = {m: i for i, m in enumerate(ALL_METHODS)}

SEED = 301
GA_BUDGET_S = 60.0
CPSAT60_S = 60.0
CPSAT300_S = 300.0
CPSAT_WORKERS = 2

# cpsat60 wall (s), workers=2, warm-started -- measured on real test instances.
# cpsat300 is reused on OPTIMAL (Gate A: 100% OPTIMAL at every size), so the
# expected CP-SAT cost per instance is ~cpsat60 only.
_CPSAT_EST_S = {50: 0.05, 150: 0.20, 400: 0.50}

# Measured GA wall (s), uncontended: on this (easy) test set the GA converges to
# its 200-generation stall limit long before the 60 s budget (sizes 50/150/400
# stall in ~0.7/1.9/6.8 s), so the 60 s budget is rarely binding.  These give a
# realistic *lower-bound* estimate alongside the spec's 60 s upper bound.
_GA_STALL_EST_S = {50: 0.7, 150: 1.9, 400: 6.8}


def _ga_stall_est(size):
    try:
        return _GA_STALL_EST_S.get(int(size), 6.8)
    except (TypeError, ValueError):
        return 6.8

FIELDS = [
    "id", "campus", "track", "size", "method", "feasible", "wwt", "makespan",
    "mean_flow", "breach_share", "breach_p1", "breach_p2", "breach_p3",
    "breach_p4", "wall_seconds", "decisions", "status", "objective_bh",
    "best_bound_bh", "reused_from_60",
]


# --------------------------------------------------------------------------- #
# Index reading / filtering
# --------------------------------------------------------------------------- #
def _read_index():
    if not INDEX_CSV.exists():
        raise FileNotFoundError(
            "instance index not found: %s (run the P1 instance builder first)"
            % INDEX_CSV
        )
    with open(INDEX_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _is_test(row):
    return str(row.get("split", "")).strip().lower() == "test"


def _instance_path(row):
    p = row.get("path")
    if p:
        cand = INDEX_CSV.parent / p
        if cand.exists():
            return cand
    campus = int(row["campus"])
    track = row.get("track", "replay")
    size = row.get("size_class", "?")
    iid = row["id"]
    return (INDEX_CSV.parent / ("c%02d" % campus) / str(track)
            / str(size) / (iid + ".json"))


def _cpsat_est(size):
    try:
        return _CPSAT_EST_S.get(int(size), 0.5)
    except (TypeError, ValueError):
        return 0.5


# --------------------------------------------------------------------------- #
# Row building
# --------------------------------------------------------------------------- #
def _row(iid, campus, track, size, method, sched, res, reused_from_60=False):
    """Build one CSV row from a schedule + its validator result. Null-safe."""
    m = res["metrics"]
    pp = m["per_priority_breach_share"]  # {1..4: float|None}
    return {
        "id": iid,
        "campus": campus,
        "track": track,
        "size": size,
        "method": method,
        "feasible": int(bool(res["feasible"])),
        "wwt": m["WWT"],
        "makespan": m["makespan"],
        "mean_flow": m["mean_flow"],
        "breach_share": m["breach_share"],
        "breach_p1": pp.get(1),
        "breach_p2": pp.get(2),
        "breach_p3": pp.get(3),
        "breach_p4": pp.get(4),
        "wall_seconds": sched.get("wall_seconds"),
        "decisions": sched.get("decisions"),
        "status": sched.get("status", ""),
        "objective_bh": sched.get("objective_bh"),
        "best_bound_bh": sched.get("best_bound_bh"),
        "reused_from_60": bool(reused_from_60),
    }


def _write_shard(iid, shard):
    """Atomic shard write (temp + os.replace) so a shard is all-or-nothing."""
    dst = SHARD_DIR / (iid + ".json")
    tmp = SHARD_DIR / (iid + ".json.tmp")
    with open(tmp, "w") as f:
        json.dump(shard, f)
    os.replace(tmp, dst)


# --------------------------------------------------------------------------- #
# One instance x all nine methods (runs in a worker process)
# --------------------------------------------------------------------------- #
def _run_one(row):
    """Process one instance under all nine methods; write its shard.

    Returns a small picklable summary dict for the parent (progress + logging).
    Infeasible schedules are recorded (feasible=0) and reported, not raised.
    A genuine exception is returned as an 'error' summary and NO shard is
    written, so the instance is retried on the next resume.
    """
    iid = row["id"]
    campus = row["campus"]
    track = row.get("track", "")
    size = row.get("size_class", "")
    t0 = time.perf_counter()
    try:
        with open(_instance_path(row)) as f:
            instance = json.load(f)

        out_rows = {}
        infeasible = []  # [{"method":..., "violations":[...]}]

        # ---- PDRs ----------------------------------------------------------
        best_warm = None
        best_wwt = None
        for rule in PDR_RULES:
            sched = pdrs.dispatch(instance, rule, seed=SEED)
            res = validate(instance, sched)
            out_rows[rule] = _row(iid, campus, track, size, rule, sched, res)
            if not res["feasible"]:
                infeasible.append({"method": rule,
                                   "violations": res["violations"][:3]})
            elif best_wwt is None or res["metrics"]["WWT"] < best_wwt:
                best_wwt = res["metrics"]["WWT"]
                best_warm = sched

        # ---- GA (seeds itself from the PDRs; no external warm start) --------
        gsched = ga.solve_ga(instance, budget_s=GA_BUDGET_S, seed=SEED)
        gres = validate(instance, gsched)
        out_rows["ga"] = _row(iid, campus, track, size, "ga", gsched, gres)
        if not gres["feasible"]:
            infeasible.append({"method": "ga",
                               "violations": gres["violations"][:3]})

        # ---- cpsat60 (warm-started from best PDR) --------------------------
        c60 = cpsat.solve(instance, time_limit_s=CPSAT60_S,
                          workers=CPSAT_WORKERS, warm_start=best_warm)
        c60res = validate(instance, c60)
        out_rows["cpsat60"] = _row(iid, campus, track, size, "cpsat60",
                                   c60, c60res, reused_from_60=False)
        if not c60res["feasible"]:
            infeasible.append({"method": "cpsat60",
                               "violations": c60res["violations"][:3]})

        # ---- cpsat300: reuse cpsat60 row if OPTIMAL, else solve ------------
        if c60.get("status") == "OPTIMAL":
            reuse = dict(out_rows["cpsat60"])  # copy every number
            reuse["method"] = "cpsat300"
            reuse["reused_from_60"] = True
            out_rows["cpsat300"] = reuse
            # (feasibility already asserted on cpsat60; same schedule.)
        else:
            c300 = cpsat.solve(instance, time_limit_s=CPSAT300_S,
                              workers=CPSAT_WORKERS, warm_start=best_warm)
            c300res = validate(instance, c300)
            out_rows["cpsat300"] = _row(iid, campus, track, size, "cpsat300",
                                        c300, c300res, reused_from_60=False)
            if not c300res["feasible"]:
                infeasible.append({"method": "cpsat300",
                                   "violations": c300res["violations"][:3]})

        assert set(out_rows) == set(ALL_METHODS), "internal: missing methods"

        shard = {
            "id": iid, "campus": campus, "track": track, "size": size,
            "rows": out_rows, "infeasible": infeasible,
            "wall_seconds_total": time.perf_counter() - t0,
        }
        _write_shard(iid, shard)
        return {"id": iid, "size": size, "ok": True,
                "infeasible": infeasible,
                "wall": shard["wall_seconds_total"]}
    except Exception as e:  # noqa: BLE001 -- report, don't write a shard
        import traceback
        return {"id": iid, "size": size, "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc(),
                "wall": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# Resumability
# --------------------------------------------------------------------------- #
def _finished_ids():
    """Ids whose shard already holds all nine methods."""
    done = set()
    if not SHARD_DIR.exists():
        return done
    for p in SHARD_DIR.glob("*.json"):
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue  # corrupt/partial -> re-run this instance
        rows = d.get("rows", {})
        if isinstance(rows, dict) and set(rows) >= set(ALL_METHODS):
            done.add(d.get("id", p.stem))
    return done


# --------------------------------------------------------------------------- #
# Merge: shards -> results.csv (+ errors.log)
# --------------------------------------------------------------------------- #
def _merge(verbose=True):
    """Build results.csv from all finished shards; rebuild errors.log."""
    all_rows = []
    infeasible = []   # (id, method, violations)
    n_finished = 0
    n_partial = 0
    for p in sorted(SHARD_DIR.glob("*.json")) if SHARD_DIR.exists() else []:
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            n_partial += 1
            continue
        rows = d.get("rows", {})
        if not (isinstance(rows, dict) and set(rows) >= set(ALL_METHODS)):
            n_partial += 1
            continue
        n_finished += 1
        iid = d.get("id", p.stem)
        for m in ALL_METHODS:
            r = rows[m]
            all_rows.append(r)
            if not r.get("feasible"):
                # prefer stored violations for the log
                viol = ""
                for it in d.get("infeasible", []):
                    if it.get("method") == m:
                        viol = " | ".join(it.get("violations", []))
                        break
                infeasible.append((iid, m, viol))

    all_rows.sort(key=lambda r: (r["id"], _METHOD_ORDER[r["method"]]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # Rebuild errors.log deterministically from the shards (restart-safe: even
    # instances finished on a previous run are represented).
    with open(ERR_LOG, "w") as f:
        f.write("# E1 infeasible schedules (FATAL). Rebuilt at merge from shards.\n")
        f.write("# columns: instance_id\tmethod\tviolations\n")
        for iid, m, viol in infeasible:
            f.write("%s\t%s\t%s\n" % (iid, m, viol))

    if verbose:
        print("Merged %d finished shard(s) -> %d rows -> %s"
              % (n_finished, len(all_rows), OUT_CSV))
        if n_partial:
            print("  (%d partial/corrupt shard(s) skipped)" % n_partial)
        print("  infeasible rows (FATAL): %d  -> %s"
              % (len(infeasible), ERR_LOG))
    return {"n_finished": n_finished, "n_rows": len(all_rows),
            "n_infeasible": len(infeasible), "n_partial": n_partial}


def _git_describe():
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=str(_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _fmt_hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return "%dh%02dm%02ds" % (h, m, s)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="E1 static benchmark runner (P2).")
    ap.add_argument("--workers", type=int, default=10,
                    help="parallel instance workers (default 10; box is shared)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap not-yet-done instances processed this run")
    ap.add_argument("--campus", default=None,
                    help="restrict to campus id(s), comma-separated")
    ap.add_argument("--size", default=None,
                    help="restrict to size class(es), comma-separated")
    ap.add_argument("--merge", action="store_true",
                    help="only (re)build results.csv + errors.log from shards")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        _merge(verbose=True)
        return

    # ---- build target set --------------------------------------------------
    rows = _read_index()
    test = [r for r in rows if _is_test(r)]
    if args.campus:
        keep = {c.strip() for c in args.campus.split(",")}
        test = [r for r in test if str(r["campus"]).strip() in keep]
    if args.size:
        keep = {s.strip() for s in args.size.split(",")}
        test = [r for r in test if str(r["size_class"]).strip() in keep]

    done = _finished_ids()
    pending = [r for r in test if r["id"] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]

    n_test = len(test)
    n_done = n_test - len([r for r in test if r["id"] not in done])
    print("E1 static benchmark")
    print("  test target set : %d instance(s)%s%s"
          % (n_test,
             (" campus=%s" % args.campus) if args.campus else "",
             (" size=%s" % args.size) if args.size else ""))
    print("  already finished: %d  ->  pending this run: %d"
          % (n_done, len(pending)))
    print("  methods/instance: %d  (%s)" % (len(ALL_METHODS), ", ".join(ALL_METHODS)))
    print("  workers=%d  cpsat_workers=%d  seed=%d  ga_budget=%.0fs"
          % (args.workers, CPSAT_WORKERS, SEED, GA_BUDGET_S))

    # ---- runtime estimate --------------------------------------------------
    est_inst_s = sum(GA_BUDGET_S + _cpsat_est(r["size_class"]) for r in pending)
    est_wall = est_inst_s / args.workers if args.workers else est_inst_s
    print("  runtime estimate: n_pending * (GA %.0fs + cpsat_est) / workers"
          % GA_BUDGET_S)
    print("                  = %.0f instance-seconds / %d workers"
          % (est_inst_s, args.workers))
    print("                  ~= %s  [UPPER BOUND: cpsat300 reused; GA budget=60s]"
          % _fmt_hms(est_wall))
    emp_inst_s = sum(_ga_stall_est(r["size_class"]) + _cpsat_est(r["size_class"])
                     for r in pending)
    print("                  ~= %s  [empirical: GA stalls ~200 gens, uncontended]"
          % _fmt_hms(emp_inst_s / args.workers if args.workers else emp_inst_s))
    print("                  real runtime lands between these (box is shared).")

    if not pending:
        print("Nothing pending -- merging existing shards.")
        _merge(verbose=True)
        return

    start_iso = _dt.datetime.now().isoformat(timespec="seconds")
    t_start = time.perf_counter()

    # ---- parallel run over instances --------------------------------------
    n_infeasible = 0
    n_errors = 0
    completed = 0
    total = len(pending)
    ctx = mp.get_context("fork")
    err_fh = open(ERR_LOG, "a")  # live append; merge rebuilds it at the end
    with ctx.Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(_run_one, pending):
            completed += 1
            if not res.get("ok"):
                n_errors += 1
                msg = "[ERROR] %s: %s" % (res["id"], res.get("error"))
                print(msg)
                err_fh.write("%s\tEXCEPTION\t%s\n"
                             % (res["id"], res.get("error")))
                err_fh.flush()
            else:
                for it in res.get("infeasible", []):
                    n_infeasible += 1
                    line = ("[FATAL infeasible] id=%s method=%s :: %s"
                            % (res["id"], it["method"],
                               " | ".join(it.get("violations", []))))
                    print(line)
                    err_fh.write("%s\t%s\t%s\n"
                                 % (res["id"], it["method"],
                                    " | ".join(it.get("violations", []))))
                    err_fh.flush()
            if completed % 50 == 0 or completed == total:
                elapsed = time.perf_counter() - t_start
                rate = elapsed / completed
                eta = rate * (total - completed)
                print("  progress %d/%d  elapsed %s  eta %s  "
                      "(%.1f inst/min; %d infeasible, %d errors)"
                      % (completed, total, _fmt_hms(elapsed), _fmt_hms(eta),
                         60.0 * completed / elapsed if elapsed else 0.0,
                         n_infeasible, n_errors))
    err_fh.close()

    elapsed = time.perf_counter() - t_start
    end_iso = _dt.datetime.now().isoformat(timespec="seconds")
    print("Run complete: %d instance(s) in %s (%d infeasible, %d errors)."
          % (completed, _fmt_hms(elapsed), n_infeasible, n_errors))

    # ---- auto-merge + meta.json -------------------------------------------
    merged = _merge(verbose=True)
    meta = {
        "experiment": "e1_static",
        "start_time": start_iso,
        "end_time": end_iso,
        "elapsed_seconds": round(elapsed, 3),
        "workers": args.workers,
        "cpsat_workers": CPSAT_WORKERS,
        "seed": SEED,
        "ga_budget_s": GA_BUDGET_S,
        "methods": ALL_METHODS,
        "n_test_target": n_test,
        "n_pending_this_run": total,
        "n_completed_this_run": completed,
        "n_errors_this_run": n_errors,
        "n_instances_in_results": merged["n_finished"],
        "n_rows": merged["n_rows"],
        "n_infeasible": merged["n_infeasible"],
        "filters": {"campus": args.campus, "size": args.size,
                    "limit": args.limit},
        "git_describe": _git_describe(),
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print("Wrote %s" % META_JSON)


if __name__ == "__main__":
    main()
