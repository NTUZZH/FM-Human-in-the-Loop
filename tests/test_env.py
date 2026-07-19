"""DispatchEnv tests -- plain python script (no pytest).

Run:  PYTHONPATH=src python tests/test_env.py

(a) PARITY: for 3 replay instances (c01/50, c05/150, c10/400 -- first sorted
    file each) and rules edd + wspt, env.run_policy(rule) reproduces
    pdrs.dispatch assignment-for-assignment (wo->tech, start, end within 1e-9),
    and both schedules pass the independent fmwos.validator.
(b) REWARD TELESCOPING: a full step-path episode under EDD has
    sum(rewards)*100 == -finalWWT within 1e-6 (Phi(s_0)==0), where finalWWT is
    the validator's WWT of the emitted schedule.
(c) LB ADMISSIBILITY: on 20 decision states sampled from an EDD rollout of the
    tight campus c02/150 first file, the admissible bound never exceeds the
    weighted tardiness of continuing EDD from that state with NO further
    arrivals (LB <= remaining WWT within 1e-6).

Prints a report and finally 'ALL ENV TESTS PASSED'.
"""

import json
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import lb, pdrs, validator                       # noqa: E402
from fmwos.env import DispatchEnv                            # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")

PARITY_FILES = [
    ("c01", "50", "edd"),
    ("c01", "50", "wspt"),
    ("c05", "150", "edd"),
    ("c05", "150", "wspt"),
    ("c10", "400", "edd"),
    ("c10", "400", "wspt"),
]
TELESCOPE_FILES = [
    ("c01", "50"),
    ("c05", "150"),
    ("c10", "400"),
    ("c02", "150"),   # tight campus, large non-zero WWT under EDD
]
LB_CAMPUS, LB_SIZE = "c02", "150"
N_LB_STATES = 20
TOL_PARITY = 1e-9
TOL_TELE = 1e-6
TOL_LB = 1e-6


def first_file(campus, size):
    import glob
    files = sorted(glob.glob(os.path.join(_INST, campus, "replay", size, "*.json")))
    return files[0]


def load(path):
    with open(path) as fh:
        return json.load(fh)


def _by_wo(schedule):
    return {a["wo"]: (a["tech"], a["start_bh"], a["end_bh"])
            for a in schedule["assignments"]}


# --------------------------------------------------------------------------- #
# (a) parity                                                                  #
# --------------------------------------------------------------------------- #
def test_parity(failures):
    print("(a) PARITY: env.run_policy vs pdrs.dispatch")
    for campus, size, rule in PARITY_FILES:
        inst = load(first_file(campus, size))
        ref = pdrs.dispatch(inst, rule)
        env = DispatchEnv(inst)
        got = env.run_policy(pdrs.get_rule(rule), method=rule)

        rmap, gmap = _by_wo(ref), _by_wo(got)
        exact = (set(rmap) == set(gmap)) and all(
            rmap[w][0] == gmap[w][0]
            and abs(rmap[w][1] - gmap[w][1]) <= TOL_PARITY
            and abs(rmap[w][2] - gmap[w][2]) <= TOL_PARITY
            for w in rmap)
        vref = validator.validate(inst, ref)
        vgot = validator.validate(inst, got)
        ok = exact and vref["feasible"] and vgot["feasible"] \
            and len(got["assignments"]) == len(inst["work_orders"])
        print("    %-18s %-5s  exact=%s  feasible(pdrs=%s,env=%s)  n=%d"
              % (inst["meta"]["id"], rule, exact, vref["feasible"],
                 vgot["feasible"], len(got["assignments"])))
        if not ok:
            failures.append("PARITY FAIL %s %s (exact=%s feas=%s/%s)"
                            % (inst["meta"]["id"], rule, exact,
                               vref["feasible"], vgot["feasible"]))


# --------------------------------------------------------------------------- #
# (b) reward telescoping                                                      #
# --------------------------------------------------------------------------- #
def test_telescoping(failures):
    print("(b) REWARD TELESCOPING: sum(rewards)*100 == -finalWWT")
    for campus, size in TELESCOPE_FILES:
        inst = load(first_file(campus, size))
        env = DispatchEnv(inst)
        env.reset()
        if env.phi_prev != 0.0:
            failures.append("TELESCOPE FAIL %s Phi(s_0)=%r != 0"
                            % (inst["meta"]["id"], env.phi_prev))
        total_r = 0.0
        done = False
        while not done:
            cands = env._candidates
            # EDD pick among candidates (smallest due, id tiebreak).
            a = min(range(len(cands)),
                    key=lambda i: (cands[i]["due_bh"], cands[i]["id"]))
            _obs, r, done, _info = env.step(a)
            total_r += r
        sched = env.to_schedule("edd")
        wwt = validator.validate(inst, sched)["metrics"]["WWT"]
        lhs = total_r * 100.0
        diff = abs(lhs - (-wwt))
        print("    %-18s  sum*100=%12.6f  -finalWWT=%12.6f  |diff|=%.2e"
              % (inst["meta"]["id"], lhs, -wwt, diff))
        if diff > TOL_TELE:
            failures.append("TELESCOPE FAIL %s |diff|=%.3e > %.1e"
                            % (inst["meta"]["id"], diff, TOL_TELE))


# --------------------------------------------------------------------------- #
# (c) LB admissibility                                                        #
# --------------------------------------------------------------------------- #
def _edd_complete_wwt(queues, tech_free, t):
    """WWT of continuing EDD (no arrivals) from a captured decision state.

    queues    : dict trade -> list of (p, d, w, id) currently queued
    tech_free : dict trade -> list of technician free-times
    t         : current bh time
    Non-delay EDD list schedule per trade: assign EDD-ordered jobs to the
    earliest-available machine (avail_i = max(t, free_i)).  Feasible schedule of
    exactly the queued jobs, so its WWT upper-bounds the optimal remaining WWT.
    """
    import heapq
    total = 0.0
    for trade, jobs in queues.items():
        if not jobs:
            continue
        avail = [max(t, f) for f in tech_free.get(trade, [])]
        if not avail:
            continue
        heapq.heapify(avail)
        for (p, d, w, _id) in sorted(jobs, key=lambda x: (x[1], x[3])):
            f = heapq.heappop(avail)
            c = f + p
            heapq.heappush(avail, c)
            if c > d:
                total += w * (c - d)
    return total


def test_lb_admissibility(failures):
    print("(c) LB ADMISSIBILITY: LB <= EDD-continuation remaining WWT")
    inst = load(first_file(LB_CAMPUS, LB_SIZE))
    env = DispatchEnv(inst)
    env.reset()

    # Snapshot every decision state (queues + tech availability + t) BEFORE the
    # pick mutates the queue, driving the rollout with EDD.
    states = []
    done = False
    while not done:
        t = env._cur_now
        queues = {tr: [(j["p_bh"], j["due_bh"], j["weight"], j["id"]) for j in q]
                  for tr, q in env.queue.items() if q}
        tech_free = {tr: [env.tech_free_at[tid] for tid in env.techs_of[tr]]
                     for tr in queues}
        states.append((t, queues, tech_free))
        cands = env._candidates
        a = min(range(len(cands)),
                key=lambda i: (cands[i]["due_bh"], cands[i]["id"]))
        _obs, _r, done, _info = env.step(a)

    rng = random.Random(1234)
    sample = rng.sample(states, min(N_LB_STATES, len(states)))
    npass = 0
    max_ratio = 0.0
    worst = None
    for (t, queues, tech_free) in sample:
        lb_val = lb.lb_remaining(
            {tr: [(p, d, w) for (p, d, w, _i) in js] for tr, js in queues.items()},
            tech_free, t)
        edd_wwt = _edd_complete_wwt(queues, tech_free, t)
        if lb_val <= edd_wwt + TOL_LB:
            npass += 1
        else:
            failures.append("LB FAIL t=%.4f LB=%.6f > EDDwwt=%.6f" % (t, lb_val, edd_wwt))
        if edd_wwt > 1e-9:
            ratio = lb_val / edd_wwt
            if ratio > max_ratio:
                max_ratio, worst = ratio, (t, lb_val, edd_wwt)
    print("    states sampled=%d  pass=%d/%d  max LB/actual ratio=%.4f"
          % (len(sample), npass, len(sample), max_ratio))
    if worst:
        print("    tightest: t=%.3f LB=%.4f actual=%.4f" % worst)
    if npass != len(sample):
        failures.append("LB ADMISSIBILITY: only %d/%d states passed"
                        % (npass, len(sample)))


# --------------------------------------------------------------------------- #
def main():
    failures = []
    test_parity(failures)
    print()
    test_telescoping(failures)
    print()
    test_lb_admissibility(failures)
    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("ALL ENV TESTS PASSED")


if __name__ == "__main__":
    main()
