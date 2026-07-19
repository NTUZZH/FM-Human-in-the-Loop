#!/usr/bin/env python
"""P4 / E5 sensitivity runner: SLA and capacity robustness sweeps (appendix).

Answers the protocol's calibration-sweep question (docs/protocol.md locked defaults) -- "do the benchmark
conclusions survive a +-50% SLA and +-25% capacity perturbation?" -- by re-running
every online policy on a fixed base set under five conditions and scoring every
schedule with the independent validator.

Base set (sorted, deterministic)
--------------------------------
  campus in {5, 9, 10, 12} x size in {150, 400}: the FIRST 30 replay TEST
  instances in sorted-id order == 4 x 2 x 30 = 240 base instances.

Conditions (5) -- each a pure transform of the base instance
------------------------------------------------------------
  baseline   f=1.00 m=1.00   base instance untouched (id == base id)
  sla0.5     f=0.50 m=1.00   fmwos.sensitivity.scale_sla  (tighter deadlines)
  sla1.5     f=1.50 m=1.00   fmwos.sensitivity.scale_sla  (looser deadlines)
  crew0.75   f=1.00 m=0.75   fmwos.tightness.scale_crew   (-25% capacity)
  crew1.25   f=1.00 m=1.25   fmwos.tightness.scale_crew   (+25% capacity)
  => 5 x 240 = 1200 instance-configurations.

Methods per config
------------------
  PDRs : edd, wspt, atc, pfifo, mor, random   (seed 301)
  RL   : rl301, rl302, rl303  (greedy argmax, results/p3_train/seed<t>/best.pt,
         via the DispatchEnv reset()/step() path on CPU; tag/dir reconfigurable
         with --rl-tag / --rl-dir).
  No rolling CP-SAT -- appendix scope (dynamic-solver robustness is E2/Gate B).

Every schedule is scored ONLY by fmwos.validator.  Output columns:
  id, base_id, campus, size, condition, sla_multiplier, crew_multiplier,
  method, seed, feasible, wwt, makespan, mean_flow, breach_share,
  breach_p1..p4, wall_seconds.

Sharded / resumable (dyneval pattern)
-------------------------------------
One task == one instance-configuration x all its methods.  Each finished config
is written atomically to results/p4_sensitivity/shards/<config_id>.json holding
every method row; a config is skipped on resume once its shard holds all
expected methods (the expected set derives from --rl-tag, so an rl* shard is NOT
reused for e.g. v2rl*).  --merge (auto-run at the end) concatenates the shards
into results/p4_sensitivity/results.csv.

Usage
-----
    PYTHONPATH=src python scripts/p4_sensitivity.py [--workers 8] [--limit N]
        [--rl-tag rl] [--rl-dir results/p3_train] [--out DIR] [--merge]
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

# Keep BLAS/OpenMP/torch from oversubscribing the shared box (dyneval policy).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

# torch / fmwos.policy are imported LAZILY inside the worker (fork-safety).
from fmwos import pdrs, sensitivity, tightness      # noqa: E402
from fmwos.env import DispatchEnv                    # noqa: E402
from fmwos.validator import validate                 # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
INST_ROOT = _ROOT / "data" / "processed" / "instances"
INDEX_CSV = INST_ROOT / "index.csv"
TRAIN_DIR = _ROOT / "results" / "p3_train"
OUT_DIR = _ROOT / "results" / "p4_sensitivity"
SHARD_DIR = OUT_DIR / "shards"
OUT_CSV = OUT_DIR / "results.csv"
META_JSON = OUT_DIR / "meta.json"

CAMPUSES = [5, 9, 10, 12]
SIZES = [150, 400]
N_BASE = 30                                          # first N replay TEST per cell

# (name, sla_multiplier f, crew_multiplier m).  Order fixes the config sort so a
# small --limit still yields baseline + the sla conditions of the same base id.
CONDITIONS = [
    ("baseline", 1.0, 1.0),
    ("sla0.5", 0.5, 1.0),
    ("sla1.5", 1.5, 1.0),
    ("crew0.75", 1.0, 0.75),
    ("crew1.25", 1.0, 1.25),
]
_COND_ORDER = {name: i for i, (name, _f, _m) in enumerate(CONDITIONS)}

PDR_RULES = ["edd", "wspt", "atc", "pfifo", "mor", "random"]
RL_SEEDS = [301, 302, 303]
# RL naming + checkpoint root are reconfigurable (--rl-tag/--rl-dir); rederived
# in the parent before the pool is forked so every worker inherits the set.
RL_TAG = "rl"
RL_DIR = TRAIN_DIR
RL_METHODS = ["%s%d" % (RL_TAG, t) for t in RL_SEEDS]
BASE_METHODS = PDR_RULES + RL_METHODS
_METHOD_ORDER = {m: i for i, m in enumerate(BASE_METHODS)}

SEED = 301
TORCH_THREADS = 2

FIELDS = [
    "id", "base_id", "campus", "size", "condition", "sla_multiplier",
    "crew_multiplier", "method", "seed", "feasible", "wwt", "makespan",
    "mean_flow", "breach_share", "breach_p1", "breach_p2", "breach_p3",
    "breach_p4", "wall_seconds",
]


# --------------------------------------------------------------------------- #
# Reconfiguration (parent, before fork)
# --------------------------------------------------------------------------- #
def _configure_rl(tag, rl_dir):
    global RL_TAG, RL_DIR, RL_METHODS, BASE_METHODS, _METHOD_ORDER
    RL_TAG = str(tag)
    RL_DIR = Path(rl_dir)
    RL_METHODS = ["%s%d" % (RL_TAG, t) for t in RL_SEEDS]
    BASE_METHODS = PDR_RULES + RL_METHODS
    _METHOD_ORDER = {m: i for i, m in enumerate(BASE_METHODS)}


def _configure_out(out_dir):
    global OUT_DIR, SHARD_DIR, OUT_CSV, META_JSON
    OUT_DIR = Path(out_dir)
    SHARD_DIR = OUT_DIR / "shards"
    OUT_CSV = OUT_DIR / "results.csv"
    META_JSON = OUT_DIR / "meta.json"


def _rl_method(seed):
    return "%s%d" % (RL_TAG, seed)


# --------------------------------------------------------------------------- #
# Target-set construction
# --------------------------------------------------------------------------- #
def _read_index():
    with open(INDEX_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _replay_path(row):
    p = row.get("path")
    if p:
        cand = INST_ROOT / p
        if cand.exists():
            return cand
    return (INST_ROOT / ("c%02d" % int(row["campus"])) / "replay"
            / str(row["size_class"]) / (row["id"] + ".json"))


def _base_rows():
    """First N_BASE replay TEST rows (sorted id) per campus x size."""
    idx = _read_index()
    rows = []
    for campus in CAMPUSES:
        for size in SIZES:
            cell = [r for r in idx
                    if str(r.get("split", "")).strip().lower() == "test"
                    and str(r.get("track", "")).strip().lower() == "replay"
                    and int(r["campus"]) == campus
                    and int(r["size_class"]) == size]
            cell.sort(key=lambda r: r["id"])
            rows.extend(cell[:N_BASE])
    return rows


def _config_id(base_id, f, m):
    """Perturbed-instance id, matching the transforms' meta.id suffixing.

    baseline (f=1, m=1) keeps the base id; sla -> ``_sla<f>`` (scale_sla),
    crew -> ``_m<m>`` (scale_crew).  Conditions never combine f!=1 with m!=1,
    so this is unambiguous and collision-free across the 5 conditions.
    """
    if f != 1.0:
        return "%s_sla%s" % (base_id, f)
    if m != 1.0:
        return "%s_m%s" % (base_id, m)
    return base_id


def build_targets():
    """Ordered list of picklable, self-describing config dicts (one per
    instance-configuration)."""
    configs = []
    for row in _base_rows():
        base_id = row["id"]
        campus = int(row["campus"])
        size = int(row["size_class"])
        path = str(_replay_path(row))
        for name, f, m in CONDITIONS:
            configs.append({
                "id": _config_id(base_id, f, m), "base_id": base_id,
                "campus": campus, "size": size, "condition": name,
                "sla_multiplier": float(f), "crew_multiplier": float(m),
                "path": path,
            })
    configs.sort(key=lambda c: (c["campus"], c["size"], c["base_id"],
                                _COND_ORDER[c["condition"]]))
    return configs


# --------------------------------------------------------------------------- #
# Worker: RL policy cache + rollout (dyneval pattern)
# --------------------------------------------------------------------------- #
_POLICY_CACHE = {}


def _get_policy(seed):
    pol = _POLICY_CACHE.get(seed)
    if pol is None:
        import torch  # lazy: only inside the worker
        from fmwos.policy import DispatchPolicy
        torch.set_num_threads(TORCH_THREADS)
        pol = DispatchPolicy.load(str(RL_DIR / ("seed%d" % seed) / "best.pt"),
                                  map_location="cpu")
        pol.eval()
        _POLICY_CACHE[seed] = pol
    return pol


def _rl_rollout(instance, seed):
    pol = _get_policy(seed)
    env = DispatchEnv(instance)
    obs = env.reset()
    done = False
    while not done:
        a, _, _, _ = pol.act(obs, greedy=True, device="cpu")
        obs, _r, done, _info = env.step(a)
    return env.to_schedule(_rl_method(seed), seed=seed)


# --------------------------------------------------------------------------- #
# Row building + shard IO
# --------------------------------------------------------------------------- #
def _row(config, method, seed, sched, res):
    m = res["metrics"]
    pp = m["per_priority_breach_share"]
    return {
        "id": config["id"], "base_id": config["base_id"],
        "campus": config["campus"], "size": config["size"],
        "condition": config["condition"],
        "sla_multiplier": config["sla_multiplier"],
        "crew_multiplier": config["crew_multiplier"],
        "method": method, "seed": seed,
        "feasible": int(bool(res["feasible"])),
        "wwt": m["WWT"], "makespan": m["makespan"], "mean_flow": m["mean_flow"],
        "breach_share": m["breach_share"],
        "breach_p1": pp.get(1), "breach_p2": pp.get(2),
        "breach_p3": pp.get(3), "breach_p4": pp.get(4),
        "wall_seconds": sched.get("wall_seconds"),
    }


def _write_shard(shard_id, shard):
    dst = SHARD_DIR / (shard_id + ".json")
    tmp = SHARD_DIR / (shard_id + ".json.tmp")
    with open(tmp, "w") as f:
        json.dump(shard, f)
    os.replace(tmp, dst)


def _transform(instance, config):
    """Apply the condition's pure transform (baseline == untouched instance)."""
    f = config["sla_multiplier"]
    m = config["crew_multiplier"]
    if f != 1.0:
        return sensitivity.scale_sla(instance, f)
    if m != 1.0:
        return tightness.scale_crew(instance, m)
    return instance


# --------------------------------------------------------------------------- #
# One config x all methods (worker process)
# --------------------------------------------------------------------------- #
def _run_one(config):
    t0 = time.perf_counter()
    try:
        with open(config["path"]) as f:
            base = json.load(f)
        instance = _transform(base, config)
        # The perturbed meta.id must equal the config id the schedule is scored
        # against (validator check (f)); a mismatch is a construction bug.
        assert instance["meta"]["id"] == config["id"], (
            "id mismatch: %r != %r" % (instance["meta"]["id"], config["id"]))

        out_rows = {}
        infeasible = []

        for rule in PDR_RULES:
            sched = pdrs.dispatch(instance, rule, seed=SEED)
            res = validate(instance, sched)
            out_rows[rule] = _row(config, rule, SEED, sched, res)
            if not res["feasible"]:
                infeasible.append({"method": rule,
                                   "violations": res["violations"][:3]})

        for t in RL_SEEDS:
            meth = _rl_method(t)
            sched = _rl_rollout(instance, t)
            res = validate(instance, sched)
            out_rows[meth] = _row(config, meth, t, sched, res)
            if not res["feasible"]:
                infeasible.append({"method": meth,
                                   "violations": res["violations"][:3]})

        assert set(out_rows) == set(BASE_METHODS), "internal: method set mismatch"

        shard = {
            "shard_id": config["id"], "id": config["id"],
            "base_id": config["base_id"], "campus": config["campus"],
            "size": config["size"], "condition": config["condition"],
            "rows": out_rows, "methods_expected": list(BASE_METHODS),
            "infeasible": infeasible,
            "wall_seconds_total": time.perf_counter() - t0,
        }
        _write_shard(config["id"], shard)
        return {"id": config["id"], "condition": config["condition"], "ok": True,
                "infeasible": infeasible, "wall": shard["wall_seconds_total"]}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"id": config["id"], "condition": config["condition"],
                "ok": False, "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc(),
                "wall": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# Resumability + merge
# --------------------------------------------------------------------------- #
def _shard_methods():
    """Map shard id -> set of method rows it holds (corrupt -> absent)."""
    have = {}
    if not SHARD_DIR.exists():
        return have
    for p in SHARD_DIR.glob("*.json"):
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        rows = d.get("rows", {})
        if isinstance(rows, dict):
            have[d.get("id", p.stem)] = set(rows)
    return have


def _merge(verbose=True):
    all_rows = []
    n_finished = n_partial = n_infeasible = 0
    for p in sorted(SHARD_DIR.glob("*.json")) if SHARD_DIR.exists() else []:
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            n_partial += 1
            continue
        rows = d.get("rows", {})
        expected = d.get("methods_expected", BASE_METHODS)
        if not (isinstance(rows, dict) and set(rows) >= set(expected)):
            n_partial += 1
            continue
        n_finished += 1
        for meth in expected:
            r = rows[meth]
            all_rows.append(r)
            if not r.get("feasible"):
                n_infeasible += 1

    all_rows.sort(key=lambda r: (r["campus"], r["size"], r["base_id"],
                                 _COND_ORDER.get(r["condition"], 99),
                                 _METHOD_ORDER.get(r["method"], 99)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    if verbose:
        print("Merged %d finished config(s) -> %d rows -> %s"
              % (n_finished, len(all_rows), OUT_CSV))
        if n_partial:
            print("  (%d partial/corrupt shard(s) skipped)" % n_partial)
        print("  infeasible rows: %d" % n_infeasible)
    return {"n_finished": n_finished, "n_rows": len(all_rows),
            "n_infeasible": n_infeasible, "n_partial": n_partial}


def _git_describe():
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=str(_ROOT), stderr=subprocess.DEVNULL).decode().strip()
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
    ap = argparse.ArgumentParser(description="P4/E5 sensitivity runner.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap not-yet-done configs processed this run")
    ap.add_argument("--rl-tag", default="rl",
                    help="method-column prefix for the RL policies "
                         "(default 'rl' -> rl301..rl303)")
    ap.add_argument("--rl-dir", default=str(TRAIN_DIR),
                    help="checkpoint root; loads <dir>/seed<t>/best.pt for "
                         "t in {301,302,303} (default results/p3_train)")
    ap.add_argument("--out", default=str(OUT_DIR),
                    help="results root for shards/results.csv/meta.json "
                         "(default results/p4_sensitivity)")
    ap.add_argument("--merge", action="store_true",
                    help="only (re)build results.csv from existing shards")
    args = ap.parse_args(argv)

    # Reconfigure BEFORE any dir/worker is created so forked workers inherit it.
    _configure_rl(args.rl_tag, args.rl_dir)
    _configure_out(args.out)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        _merge(verbose=True)
        return

    configs = build_targets()

    print("P4/E5 sensitivity evaluation")
    print("  conditions      : %s" % ", ".join(n for n, _f, _m in CONDITIONS))
    print("  base set        : campus %s x size %s x first %d replay-test"
          % (CAMPUSES, SIZES, N_BASE))
    print("  rl policies     : %s  (from %s)" % (", ".join(RL_METHODS), RL_DIR))
    print("  out root        : %s" % OUT_DIR)

    have = _shard_methods()
    pending = [c for c in configs
               if not (have.get(c["id"], set()) >= set(BASE_METHODS))]
    n_pending_all = len(pending)
    if args.limit is not None:
        pending = pending[:args.limit]

    by_cond = {}
    for c in configs:
        by_cond[c["condition"]] = by_cond.get(c["condition"], 0) + 1
    print("  configs total   : %d  %s" % (len(configs), dict(by_cond)))
    print("  already finished: %d  ->  pending this run: %d"
          % (len(configs) - n_pending_all, len(pending)))
    print("  methods/config  : %d (%s)"
          % (len(BASE_METHODS), ", ".join(BASE_METHODS)))
    print("  workers=%d  torch_threads=%d  seed=%d"
          % (args.workers, TORCH_THREADS, SEED))

    if not pending:
        print("Nothing pending -- merging existing shards.")
        _merge(verbose=True)
        return

    start_iso = _dt.datetime.now().isoformat(timespec="seconds")
    t_start = time.perf_counter()
    n_infeasible = n_errors = completed = 0
    total = len(pending)
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(_run_one, pending):
            completed += 1
            if not res.get("ok"):
                n_errors += 1
                print("[ERROR] %s (%s): %s"
                      % (res["id"], res["condition"], res.get("error")))
            else:
                for it in res.get("infeasible", []):
                    n_infeasible += 1
                    print("[INFEASIBLE] id=%s method=%s :: %s"
                          % (res["id"], it["method"],
                             " | ".join(it.get("violations", []))))
            if completed % 25 == 0 or completed == total:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / completed * (total - completed)
                print("  progress %d/%d  elapsed %s  eta %s  "
                      "(%d infeasible, %d errors)"
                      % (completed, total, _fmt_hms(elapsed), _fmt_hms(eta),
                         n_infeasible, n_errors))

    elapsed = time.perf_counter() - t_start
    end_iso = _dt.datetime.now().isoformat(timespec="seconds")
    print("Run complete: %d config(s) in %s (%d infeasible, %d errors)."
          % (completed, _fmt_hms(elapsed), n_infeasible, n_errors))

    merged = _merge(verbose=True)
    meta = {
        "experiment": "p4_sensitivity", "start_time": start_iso,
        "end_time": end_iso, "elapsed_seconds": round(elapsed, 3),
        "workers": args.workers, "conditions": [list(c) for c in CONDITIONS],
        "campuses": CAMPUSES, "sizes": SIZES, "n_base_per_cell": N_BASE,
        "rl_tag": RL_TAG, "rl_dir": str(RL_DIR), "rl_methods": RL_METHODS,
        "out_dir": str(OUT_DIR), "n_configs": len(configs),
        "n_pending_this_run": total, "n_completed_this_run": completed,
        "n_errors_this_run": n_errors, "n_rows": merged["n_rows"],
        "n_infeasible": merged["n_infeasible"],
        "filters": {"limit": args.limit},
        "git_describe": _git_describe(),
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print("Wrote %s" % META_JSON)


if __name__ == "__main__":
    main()
