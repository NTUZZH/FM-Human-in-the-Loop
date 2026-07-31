#!/usr/bin/env python
"""y3_verc probe: instance sizes, rollout steps/sec (TASK 4), GA feasibility/timing
(TASK 1b), and the recorded-due invariant d_j == r_j + SLA(c_j)."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import glob, json, sys, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from fmwos.env import DispatchEnv
from fmwos.hitl import deciders as dec
from fmwos import ga as _ga
from fmwos.hitl import overlay as ov
from fmwos.hitl.supervisor import Supervisor
from fmwos import validator as _validator

_INST = os.path.join(_ROOT, "data", "processed", "instances")
SLA = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}


def one(campus, u, idx=0):
    cdir = "c%02d" % campus
    fs = sorted(glob.glob(os.path.join(_INST, cdir, "storm2", "w80",
                "%s_storm2_w80_u%d_*.json" % (cdir, u))))
    p = fs[idx]
    inst = json.load(open(p))
    n = len(inst["work_orders"])
    # recorded-due invariant
    bad = 0
    for w in inst["work_orders"]:
        d_expect = float(w["release_bh"]) + SLA[int(w["priority"])]
        if abs(float(w["due_bh"]) - d_expect) > 1e-6:
            bad += 1
    # ATC rollout timing
    env = DispatchEnv(inst)
    t0 = time.perf_counter()
    sched = dec.run_rule(env, "atc", seed=301)
    dt = time.perf_counter() - t0
    ndec = len(sched["assignments"])
    sps = ndec / dt if dt > 0 else 0.0
    print("c%d u%d [%s] n_wos=%d due_bad=%d | ATC rollout: %.3fs, %d steps, %.0f steps/s"
          % (campus, u, os.path.basename(p), n, bad, dt, ndec, sps), flush=True)
    return inst, n, sps


def ga_probe(inst, budget_s=60.0):
    n = len(inst["work_orders"])
    t0 = time.perf_counter()
    sched = _ga.solve_ga(inst, budget_s=budget_s, seed=301, pop=100)
    dt = time.perf_counter() - t0
    feas = _validator.validate(inst, sched)["feasible"]
    print("  GA n=%d: %.1fs wall, %d generations, %d evals, feasible=%s"
          % (n, dt, sched.get("generations"), sched.get("decisions"), feas), flush=True)
    return sched


if __name__ == "__main__":
    print("=== instance sizes + rollout steps/sec + due invariant ===")
    inst9_100, n9, sps9 = one(9, 100, 0)
    one(9, 90, 0)
    inst10_100, n10, sps10 = one(10, 100, 0)
    one(10, 90, 0)
    # a small-instance reference for TASK 4 scaling (size ~150/400 analog: use u70 small)
    print("=== small reference (for compute scaling) ===")
    # smallest campus for a rough small-n analog
    one(9, 70, 0)
    print("=== GA feasibility/timing on one c09 u100 (n~2300) ===")
    ga_probe(inst9_100, budget_s=60.0)
