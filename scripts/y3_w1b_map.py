#!/usr/bin/env python
"""W1b: the E3 regime map re-run under the DEPLOYABLE stability-routing policy.

Why this exists. W1 replaced the manuscript's review-routing policy. The
published policy (``Supervisor`` with ``mechanism="targeted"``) decides which
decisions the supervisor reviews partly through ``_has_plus2``, which reads the
realized latent shift of the pending queue, so no site can run it. W1 measured
the replacement on the headline cell and an eight-cell contention grid, but it
did NOT re-run the regime map (Figure F3), which is a headline figure and is
still on the old policy. This script closes that gap: it reproduces the map,
cell for cell, under the deployable policy.

Nothing here re-derives the pipeline. ``y3_w1_sweep.evaluate_cell`` drives every
cell, unchanged, with its cache redirected into ``results/y3_w1b/cache``; the
statistics helpers come from ``y3_p4_m0grid``. The published map's own grid
(campuses, utilisation ladder, recoverable shares, seed count, held-out size) is
READ from ``results/y3_p4/e3_map_summary.json`` and asserted against the grid
this script builds, so the two cannot drift apart silently.

Parts
-----
gate  Reproduction gate. Headline cell (c9 storm2 u100, beta 1.0, rho 0.25),
      seeds 301-310, policy=targeted + split_fit=False, which is the published
      protocol. Per-instance TWT* must be bit-identical to the committed
      ``results/y3_p4/cache`` records, and the ten-seed reduction of the
      correction layer over the tuned rule must reproduce \\MzeroGain = 45.4%.
      Nothing else runs until this passes.
map   The deployable regime map. Campuses 9 and 10, storm2 u{70,90,100,110,130},
      beta{0,0.5,1.0}, rho 0.25, seeds 301-303, ten held-out instances per cell,
      policy=stability.
ctrl  Split-protocol control on campus 9 only: the same 15 C9 cells under the
      ORACLE-INFORMED policy with the conformal fold split in force. The
      deployable arm differs from the published map in two coupled ways (the
      routing policy, and the fold split the conformal band needs); this arm
      holds the policy at the published one and turns only the split on, so a
      per-cell difference can be attributed. C10 carries no such control,
      because a C10 cell-seed costs roughly fifteen times a C9 one.

Proof of which policy ran, three ways, because "the map looks like the published
one" is also what running the OLD policy by accident would produce:

  1. ``make_supervisor`` is wrapped so every supervisor constructed anywhere in
     the run (training loop and evaluation alike) has its class asserted against
     the policy the task asked for, and the counts are recorded per cell.
  2. ``Supervisor._has_plus2`` -- the oracle clause itself -- is wrapped with a
     call counter. Under the deployable policy it must be called ZERO times.
     Under the published policy it must be called many times. This is a direct,
     falsifiable check that the undeployable criterion did not run.
  3. The record carries the undetermined rate of the stability test, a quantity
     the old policy does not produce at all (it is NaN there), plus the
     calibrated band and the realised review fraction.

Compute. One numeric thread per process: this pipeline reproduces bit-exactly at
one thread only, and more threads change the floating-point reduction order.
Parallelism comes from separate processes, pinned:

    PYTHONPATH=src OMP_NUM_THREADS=1 taskset -c 0-9 \\
        python scripts/y3_w1b_map.py --part gate --workers 9

No wall-clock figure from this script is a measurement of anything: the machine
is shared and cores 10-23 belong to other agents.
"""

from __future__ import annotations

import os

# ONE numeric thread per process, installed BEFORE numpy/torch are imported and
# before y3_w1_sweep is imported (its module-level ``setdefault`` would otherwise
# install four, which changes the reduction order and moves the headline).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse                                                    # noqa: E402
import json                                                        # noqa: E402
import sys                                                         # noqa: E402
import time                                                        # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed   # noqa: E402

import numpy as np                                                 # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch                                                       # noqa: E402

from fmwos.hitl import routing as R                                 # noqa: E402
from fmwos.hitl.supervisor import Supervisor                        # noqa: E402

import y3_w1_sweep as W1                                            # noqa: E402
import y3_p4_m0grid as P4                                           # noqa: E402

_OUT = os.path.join(_ROOT, "results", "y3_w1b")
_CACHE = os.path.join(_OUT, "cache")
_PUB_SUMMARY = os.path.join(_ROOT, "results", "y3_p4", "e3_map_summary.json")

# Redirect the reused runner's result cache into this work package's directory,
# so no record of W1's is read or overwritten and every cell here is computed by
# this run. ``evaluate_cell`` resolves this global at call time.
W1._CACHE = _CACHE


# --------------------------------------------------------------------------- #
# The published map's grid, READ from the published summary rather than assumed #
# --------------------------------------------------------------------------- #
def published_grid():
    """(campuses, utilisation levels, recoverable shares, rho, n_seeds) of the
    published regime map, parsed from its own summary file."""
    with open(_PUB_SUMMARY) as fh:
        s = json.load(fh)
    campuses, us, betas = set(), set(), set()
    for key in s["cells"]:
        c, u, b = key.split("_")
        campuses.add(int(c[1:]))
        us.add(int(u[1:]))
        betas.add(float(b[1:]))
    n_seeds = {int(v["n_seeds"]) for v in s["cells"].values()}
    assert len(n_seeds) == 1, "published map is not at a single seed count: %r" % n_seeds
    return {"campuses": sorted(campuses), "u_levels": sorted(us),
            "betas": sorted(betas), "rho": float(s["config"]["rho"]),
            "n_seeds": n_seeds.pop(), "n_cells": len(s["cells"]),
            "channel": s["config"]["channel"]}


def published_task(campus, u, beta, rho, seed, n_eval):
    """The committed y3_p4 task for one published map cell-seed (Part B)."""
    return P4._base_task(campus=campus, u=u, beta=beta, rho=rho, seed=seed,
                         n_eval=n_eval, scope="e3", part="B")


def published_record(campus, u, beta, rho, seed, n_eval):
    """The committed y3_p4 cache record for one published map cell-seed."""
    t = published_task(campus, u, beta, rho, seed, n_eval)
    p = os.path.join(_ROOT, "results", "y3_p4", "cache",
                     "%s.json" % P4._cell_sig(t))
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Proof of which policy ran                                                    #
# --------------------------------------------------------------------------- #
_EXPECT_CLASS = {"stability": "StabilityRoutingSupervisor",
                 "margin": "MarginRoutingSupervisor",
                 "targeted": "Supervisor",
                 "random": "Supervisor"}

_PROOF = {"n_make_supervisor": 0, "classes": {}, "policies": {},
          "has_plus2_calls": 0}

_ORIG_MAKE = R.make_supervisor
_ORIG_HAS_PLUS2 = Supervisor._has_plus2
_PATCHED = False


def _make_supervisor_checked(*a, **kw):
    """``routing.make_supervisor`` with the returned object's class asserted
    against the policy the caller asked for. Wraps every construction site: the
    DAgger training loop inside ``run_m0_routed`` and the held-out evaluation
    inside ``evaluate_cell`` both route through here."""
    policy = a[0] if a else kw["policy"]
    sup = _ORIG_MAKE(*a, **kw)
    cls = type(sup).__name__
    want = _EXPECT_CLASS[policy]
    if cls != want:
        raise AssertionError("policy %r built a %s, expected %s"
                             % (policy, cls, want))
    if policy == "stability":
        # The deployable class, its own routing method, and not the parent's.
        if not isinstance(sup, R.StabilityRoutingSupervisor):
            raise AssertionError("stability policy did not build a "
                                 "StabilityRoutingSupervisor")
        if sup.mechanism != "stability":
            raise AssertionError("stability supervisor reports mechanism %r"
                                 % (sup.mechanism,))
        if type(sup)._decide_review is not \
                R.StabilityRoutingSupervisor._decide_review:
            raise AssertionError("stability supervisor's _decide_review is not "
                                 "the stability one")
        if type(sup)._decide_review is Supervisor._decide_review:
            raise AssertionError("stability supervisor inherited the published "
                                 "review criterion")
    _PROOF["n_make_supervisor"] += 1
    _PROOF["classes"][cls] = _PROOF["classes"].get(cls, 0) + 1
    _PROOF["policies"][policy] = _PROOF["policies"].get(policy, 0) + 1
    return sup


def _has_plus2_counted(self, candidates):
    """``Supervisor._has_plus2`` -- the undeployable clause, which reads the
    realized latent shift of the pending queue -- with a call counter. Under the
    deployable policy this must never be reached."""
    _PROOF["has_plus2_calls"] += 1
    return _ORIG_HAS_PLUS2(self, candidates)


def _install_patches():
    global _PATCHED
    if _PATCHED:
        return
    R.make_supervisor = _make_supervisor_checked
    Supervisor._has_plus2 = _has_plus2_counted
    _PATCHED = True


def _reset_proof():
    _PROOF["n_make_supervisor"] = 0
    _PROOF["classes"] = {}
    _PROOF["policies"] = {}
    _PROOF["has_plus2_calls"] = 0


_install_patches()


# --------------------------------------------------------------------------- #
# Worker: one (cell, seed, policy), through the reused W1 code path            #
# --------------------------------------------------------------------------- #
def _worker(task):
    _install_patches()                     # idempotent; survives a spawn start
    torch.set_num_threads(1)
    _reset_proof()
    rec = W1.evaluate_cell(task)
    from_cache = bool(rec.get("cached"))
    if not from_cache:
        proof = {"n_make_supervisor": _PROOF["n_make_supervisor"],
                 "classes": dict(_PROOF["classes"]),
                 "policies": dict(_PROOF["policies"]),
                 "has_plus2_calls": _PROOF["has_plus2_calls"],
                 "from_cache": False}
        _check_proof(task, rec, proof)
        rec["policy_proof"] = proof
        # Persist the proof into the cache record so a resumed run keeps it.
        p = os.path.join(_CACHE, "%s.json" % rec["sig"])
        tmp = p + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(rec, fh)
        os.replace(tmp, p)
    else:
        rec.setdefault("policy_proof", {"from_cache": True})
        rec["policy_proof"]["from_cache"] = True
        _check_proof(task, rec, rec["policy_proof"])
    return rec


def _check_proof(task, rec, proof):
    """Assertions that must hold for the record, whether or not it was cached."""
    pol = task["policy"]
    if rec["policy"] != pol:
        raise AssertionError("record policy %r != task policy %r"
                             % (rec["policy"], pol))
    if rec["run_config"]["policy"] != pol:
        raise AssertionError("run_m0_routed ran policy %r, task asked %r"
                             % (rec["run_config"]["policy"], pol))
    if bool(rec["run_config"]["split_fit"]) != bool(task["split_fit"]):
        raise AssertionError("split_fit drift")
    und = rec["routing"]["m0_sup_undetermined"]
    if pol == "stability":
        # The stability test's own output; the published policy produces none.
        if not np.isfinite(und):
            raise AssertionError("stability cell has no undetermined rate")
        if rec.get("band") is None:
            raise AssertionError("stability cell has no calibrated band")
        if proof.get("from_cache") is False and proof["has_plus2_calls"] != 0:
            raise AssertionError("the oracle clause _has_plus2 was called %d "
                                 "times under the deployable policy"
                                 % proof["has_plus2_calls"])
        if proof.get("from_cache") is False and \
                proof["classes"].get("StabilityRoutingSupervisor", 0) == 0:
            raise AssertionError("no StabilityRoutingSupervisor was built")
    else:
        if np.isfinite(und):
            raise AssertionError("non-stability cell reported an undetermined "
                                 "rate; the wrong policy ran")


# --------------------------------------------------------------------------- #
# Task builders                                                                #
# --------------------------------------------------------------------------- #
# Per cell-seed cost, in single-core seconds, read off the committed y3_p4 cache
# (published policy) and inflated by the ~1.2 the stability test adds. Used ONLY
# to submit the long tasks first so the pool packs; never reported as a timing.
_COST = {(9, 70): 8, (9, 90): 36, (9, 100): 54, (9, 110): 58, (9, 130): 108,
         (10, 70): 40, (10, 90): 300, (10, 100): 570, (10, 110): 1010,
         (10, 130): 2280}


def tasks_gate(seeds=range(301, 311)):
    """Reproduction gate: the published protocol at the headline cell."""
    return [W1._base_task(seed=s, arm="targeted_pub", part="gate",
                          policy="targeted", split_fit=False, **W1.HEAD)
            for s in seeds]


def _map_cells(grid, campuses=None):
    for campus in (campuses or grid["campuses"]):
        for u in grid["u_levels"]:
            for beta in grid["betas"]:
                yield campus, u, beta


def tasks_map(grid, campuses=None, seeds=None, n_eval=10):
    """The published map's grid under the deployable policy."""
    seeds = seeds or range(301, 301 + grid["n_seeds"])
    out = []
    for campus, u, beta in _map_cells(grid, campuses):
        for seed in seeds:
            out.append(W1._base_task(campus=campus, regime="storm2", u=u,
                                     beta=beta, rho=grid["rho"], seed=seed,
                                     n_eval=n_eval, arm="stability", part="map",
                                     policy="stability", split_fit=True))
    return out


def tasks_ctrl(grid, campuses=(9,), seeds=None, n_eval=10):
    """Split-protocol control: published policy, conformal fold split ON."""
    seeds = seeds or range(301, 301 + grid["n_seeds"])
    out = []
    for campus, u, beta in _map_cells(grid, campuses):
        for seed in seeds:
            out.append(W1._base_task(campus=campus, regime="storm2", u=u,
                                     beta=beta, rho=grid["rho"], seed=seed,
                                     n_eval=n_eval, arm="targeted_split",
                                     part="ctrl", policy="targeted",
                                     split_fit=True))
    return out


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
def run_tasks(tasks, csv_path, workers, label, fresh=True):
    """Longest-estimated-first so the pool packs; the C10 overload cells are two
    orders of magnitude more expensive than the C9 slack ones."""
    if fresh and os.path.exists(csv_path):
        os.remove(csv_path)
    tasks = sorted(tasks, key=lambda t: -_COST.get((t["campus"], t["u"]), 100))
    print("[%s] %d cell-seed tasks, %d workers (1 thread each) -> %s"
          % (label, len(tasks), workers, csv_path), flush=True)
    records, done, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(_worker, t): t for t in tasks}
        for f in as_completed(fut):
            t = fut[f]
            rec = f.result()
            records.append((t, rec))
            W1._append_rows(csv_path, t, rec)
            done += 1
            rt = rec["routing"]
            pf = rec.get("policy_proof", {})
            print("  [%s %d/%d] %-14s c%-2d u%-3d b%.2f r%.2f s%d | rule=%.0f "
                  "m0=%.0f rsup=%.0f m0sup=%.0f | rev=%.3f und=%s q=%s "
                  "sup=%s hasplus2=%s %s (%.0fs, wall %.0fs)"
                  % (label, done, len(tasks), t["arm"], rec["campus"], rec["u"],
                     rec["beta"], rec["rho"], rec["seed"],
                     np.mean(rec["per"]["rule"]),
                     np.mean(rec["per"]["m0_alone"]),
                     np.mean(rec["per"]["rule_sup"]),
                     np.mean(rec["per"]["m0_sup"]),
                     rt["m0_sup_revfrac_mean"],
                     ("%.3f" % rt["m0_sup_undetermined"])
                     if np.isfinite(rt["m0_sup_undetermined"]) else "n/a",
                     ("%.3f" % rec["band"]["q"]) if rec.get("band") else "-",
                     pf.get("classes"), pf.get("has_plus2_calls"),
                     "CACHED" if rec.get("cached") else "",
                     rec["elapsed_s"], time.time() - t0), flush=True)
    return records


def dump_records(records, path):
    out = []
    for t, r in records:
        rr = {k: v for k, v in r.items() if k != "per_iter"}
        rr["arm"] = t["arm"]
        rr["part"] = t["part"]
        out.append(rr)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("[dump] wrote %s (%d records)" % (path, len(out)), flush=True)


# --------------------------------------------------------------------------- #
# Reproduction gate                                                            #
# --------------------------------------------------------------------------- #
def run_gate(workers):
    """Reproduce \\MzeroGain through THIS code path before anything else runs."""
    print("=" * 78)
    print("REPRODUCTION GATE -- headline cell, seeds 301-310, published protocol")
    print("=" * 78)
    tasks = tasks_gate()
    recs = run_tasks(tasks, os.path.join(_OUT, "gate.csv"), workers, "gate")
    dump_records(recs, os.path.join(_OUT, "gate_records.json"))

    ok = True
    maxdiff = 0.0
    n_cmp = 0
    for t, r in recs:
        pub = W1.published_record(t["seed"])
        if pub is None:
            print("  !! committed y3_p4 record missing for seed %d" % t["seed"])
            ok = False
            continue
        if list(r["inst_ids"]) != list(pub["inst_ids"]):
            print("  !! held-out instance ids differ at seed %d" % t["seed"])
            ok = False
        for k in W1.DECIDERS:
            a = np.asarray(r["per"][k], float)
            b = np.asarray(pub["per"][k], float)
            same = bool(np.array_equal(a, b))
            maxdiff = max(maxdiff, float(np.abs(a - b).max()))
            n_cmp += 1
            ok = ok and same

    m, sd, S = P4._seed_meanstd([(t, r) for t, r in recs], "m0_alone")
    rm, _rsd, _ = P4._seed_meanstd([(t, r) for t, r in recs], "rule")
    gain = 100.0 * (rm - m) / rm

    with open(os.path.join(_ROOT, "results", "y3_p4",
                           "m0_gate_summary.json")) as fh:
        pubs = json.load(fh)
    pub_gain = pubs["cells"]["c9_storm2_u100_b1.00_r0.25"]["ladder"]["m0_alone"][
        "pct_below_rule"]

    print("-" * 78)
    print("  per-instance TWT* vs the committed y3_p4 records: %d comparisons, "
          "max |diff| = %.3e" % (n_cmp, maxdiff))
    print("  \\MzeroGain published  : %.4f %%  (macro rounds to 45.4%%)" % pub_gain)
    print("  \\MzeroGain reproduced : %.4f %%  (%d seeds, this code path)"
          % (gain, S))
    print("  difference            : %.3e percentage points" % (gain - pub_gain))
    print("  REPRODUCTION GATE: %s" % ("PASS" if (ok and abs(gain - pub_gain)
                                                  < 1e-9) else "FAIL"))
    out = {"published_MzeroGain_pct": pub_gain, "reproduced_MzeroGain_pct": gain,
           "difference_pp": gain - pub_gain, "n_seeds": S,
           "per_instance_comparisons": n_cmp, "max_abs_diff": maxdiff,
           "bit_identical": bool(ok), "rule_twt_mean": rm, "m0_twt_mean": m,
           "macro": "\\MzeroGain", "macro_value": "45.4%"}
    with open(os.path.join(_OUT, "gate_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return bool(ok and abs(gain - pub_gain) < 1e-9)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["gate", "map", "map-c9", "map-c10",
                                       "ctrl", "grid-check"], required=True)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--n-eval", type=int, default=10,
                    help="held-out instances per cell-seed; the published map "
                         "used 10 on BOTH campuses (verified in its own CSV)")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    os.makedirs(_OUT, exist_ok=True)
    os.makedirs(_CACHE, exist_ok=True)

    grid = published_grid()
    print("[grid] published map, read from %s:" % _PUB_SUMMARY, flush=True)
    print("       campuses=%s u=%s beta=%s rho=%s seeds=%d cells=%d channel=%s"
          % (grid["campuses"], grid["u_levels"], grid["betas"], grid["rho"],
             grid["n_seeds"], grid["n_cells"], grid["channel"]), flush=True)

    if args.part == "grid-check":
        t = tasks_map(grid)
        print("[grid] this run would build %d cell-seed tasks over %d cells"
              % (len(t), len(t) // grid["n_seeds"]))
        est = sum(_COST.get((x["campus"], x["u"]), 100) for x in t)
        print("[grid] estimated single-core cost: %.0f s (%.1f core-hours)"
              % (est, est / 3600.0))
        for camp in grid["campuses"]:
            e = sum(_COST.get((x["campus"], x["u"]), 100) for x in t
                    if x["campus"] == camp)
            print("       C%d: %.0f s (%.1f core-hours)" % (camp, e, e / 3600.0))
        return

    if args.part == "gate":
        if not run_gate(args.workers):
            print("reproduction gate FAILED; refusing to run the map")
            sys.exit(1)
        return

    if args.part in ("map", "map-c9", "map-c10"):
        camps = {"map": None, "map-c9": [9], "map-c10": [10]}[args.part]
        tag = args.part.replace("-", "_")
        recs = run_tasks(tasks_map(grid, campuses=camps, n_eval=args.n_eval),
                         os.path.join(_OUT, "%s.csv" % tag), args.workers, tag)
        dump_records(recs, os.path.join(_OUT, "%s_records.json" % tag))
        return

    if args.part == "ctrl":
        recs = run_tasks(tasks_ctrl(grid, n_eval=args.n_eval),
                         os.path.join(_OUT, "ctrl.csv"), args.workers, "ctrl")
        dump_records(recs, os.path.join(_OUT, "ctrl_records.json"))
        return


if __name__ == "__main__":
    main()
