#!/usr/bin/env python
"""y3_verc TASK 4 -- compute projection for ONE M1 DAgger run on storm2.

M1 config (scripts/y3_p2_train.py): n_envs=16, steps_per_env=512 -> 8192
env-steps/update; fair-compute total = Y1 budget = 4,915,200 env-steps -> 600
updates over 8 outer iters.  Anchor (notes/decisions.md C-COMPUTE): the full
Tier-1 run at size mix 150/400, 8-thread CPU, is ~17 min/run.

The env-step budget is FIXED, so larger instances = fewer episodes, same #steps.
The network fwd/bwd cost (K=64, size-invariant) is fixed at ~17 min.  The extra
storm2 cost is the size-DEPENDENT per-step env work: obs construction (a slack
sort O(|Q|) every decision) + transition, plus the expert query.  We measure:
  * RL-path env.step() throughput (obs+transition) at 150/400 vs storm2 sizes,
  * ORACLE rollout throughput (adds the supervisor preferred-pick expert cost).
Then per-run wall ~= T_net(fixed ~=17min - tiny 150/400 env) + 4.9M/env_sps.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import glob, json, sys, time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from fmwos.env import DispatchEnv
from fmwos.hitl import deciders as dec
from fmwos.hitl import overlay as ov
from fmwos.hitl.supervisor import Supervisor
from fmwos.train import list_replay_train_files

_INST = os.path.join(_ROOT, "data", "processed", "instances")
BUDGET = 4_915_200
BASELINE_MIN = 17.0     # 150/400 anchor


def env_step_sps(inst, reps=1):
    """RL-path env.step throughput (obs construction + transition), action 0."""
    env = DispatchEnv(inst, reward_mode="shaped")
    t0 = time.perf_counter()
    steps = 0
    for _ in range(reps):
        env.reset()
        done = env._done
        while not done:
            _obs, _r, done, _i = env.step(0)   # action 0 = smallest-slack cand
            steps += 1
    dt = time.perf_counter() - t0
    return steps, steps / dt


def oracle_sps(inst):
    """ORACLE-greedy rollout throughput (supervisor preferred_pick expert cost)."""
    overlay = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL", master_seed=12345))
    applied = overlay.apply(inst)
    cs = applied["c_star"]
    SLA = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}
    dstar = {w["id"]: float(w["release_bh"]) + SLA[cs[w["id"]]] for w in inst["work_orders"]}
    sup = Supervisor(overlay, inst, rho=0.0, applied=applied)
    sup.due = dstar
    t0 = time.perf_counter()
    sched = dec.run_oracle_greedy(DispatchEnv(inst), sup, seed=301)
    dt = time.perf_counter() - t0
    n = len(sched["assignments"])
    return n, n / dt


def load(campus, u, i=0):
    cdir = "c%02d" % campus
    fs = sorted(glob.glob(os.path.join(_INST, cdir, "storm2", "w80",
                "%s_storm2_w80_u%d_*.json" % (cdir, u))))
    return json.load(open(fs[i]))


if __name__ == "__main__":
    # small 150/400 replay baseline
    rep = list_replay_train_files([5, 9, 10, 12], [150, 400])
    small150 = json.load(open([f for f in rep if "/150/" in f][0]))
    small400 = json.load(open([f for f in rep if "/400/" in f][0]))
    c9 = load(9, 100, 0)
    c10 = load(10, 100, 0)

    print("=== RL-path env.step throughput (single thread) ===")
    results = {}
    for name, inst, reps in [("replay150", small150, 20), ("replay400", small400, 8),
                             ("c9_u100", c9, 2), ("c10_u100", c10, 1)]:
        n = len(inst["work_orders"])
        steps, sps = env_step_sps(inst, reps=reps)
        _, osps = oracle_sps(inst)
        results[name] = dict(n_wos=n, env_sps=sps, oracle_sps=osps)
        print(" %-11s n_wos=%5d  env.step=%8.0f steps/s   oracle_rollout=%8.0f steps/s"
              % (name, n, sps, osps))

    # projection: T_net fixed = baseline - 150/400 env time; storm2 = T_net + env time
    # use replay400 as the representative small-mix env speed (mix is 150+400)
    env_small = results["replay400"]["env_sps"]
    t_env_small = BUDGET / env_small
    t_net_fixed_s = BASELINE_MIN * 60.0 - t_env_small     # size-invariant remainder
    print("\n=== projection (fixed env-step budget = %d) ===" % BUDGET)
    print(" baseline 150/400 = %.1f min; est env-only(400) = %.0fs -> T_net(fixed) ~= %.0fs (%.1f min)"
          % (BASELINE_MIN, t_env_small, t_net_fixed_s, t_net_fixed_s / 60.0))
    for name in ("c9_u100", "c10_u100"):
        sps = results[name]["env_sps"]
        t_env = BUDGET / sps
        # env cost is single-thread; the 8-thread run parallelizes env across
        # workers ~ like the baseline did, so scale env overhead by the same
        # factor the baseline env got. Report BOTH a conservative (env serial
        # added) and an 8x-parallel env estimate.
        est_serial_min = (t_net_fixed_s + t_env) / 60.0
        est_par8_min = (t_net_fixed_s + t_env / 8.0) / 60.0
        print(" %-9s  env-only=%.0fs (%.1f min)  ->  per-run est: %.0f-%.0f min "
              "(env-serial upper / 8-thread lower)"
              % (name, t_env, t_env / 60.0, est_par8_min, est_serial_min))

    with open(os.path.join(_ROOT, "results", "y3_verc", "task4_compute.json"), "w") as fh:
        json.dump(dict(budget=BUDGET, baseline_min=BASELINE_MIN,
                       t_net_fixed_s=t_net_fixed_s, results=results), fh, indent=1)
    print("\nwrote task4_compute.json")
