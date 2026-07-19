"""Baseline solver tests — plain python script (no pytest).

Run:  PYTHONPATH=src python tests/test_baselines.py

Loads the hand-built fixture (tests/fixtures/tiny_instance.json, optimum derived
in tiny_instance.md), runs every PDR and CP-SAT, checks each schedule with a
MINIMAL inline feasibility checker (deliberately NOT importing fmwos.validator,
so the referee and the test cannot share a bug), and asserts:

  * every schedule is feasible (exactly-once, trade match, release, duration,
    no overlap);
  * CP-SAT proves OPTIMAL and its WWT equals the hand-derived optimum (32);
  * every PDR's WWT is >= CP-SAT's WWT.

Prints a rule -> WWT table and finally 'ALL BASELINE TESTS PASSED'.
"""

import json
import os
import sys

# Make ``fmwos`` importable whether or not PYTHONPATH=src is set.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import cpsat, pdrs  # noqa: E402

FIXTURE = os.path.join(_ROOT, "tests", "fixtures", "tiny_instance.json")
HAND_OPTIMUM = 32.0
RULES = ["edd", "wspt", "atc", "pfifo", "mor", "random"]
SEED = 301
TOL = 1e-6


# --------------------------------------------------------------------------- #
# Minimal, self-contained feasibility check + WWT (re-derived from the spec,
# NOT imported from fmwos.validator).
# --------------------------------------------------------------------------- #
def check_feasible(instance, schedule):
    """Return list of violation strings ([] == feasible)."""
    v = []
    wo_by_id = {w["id"]: w for w in instance["work_orders"]}
    tech_by_id = {t["id"]: t for t in instance["technicians"]}
    assigns = schedule["assignments"]

    # (a) every WO assigned exactly once.
    seen = {}
    for a in assigns:
        seen[a["wo"]] = seen.get(a["wo"], 0) + 1
    for wid in wo_by_id:
        if seen.get(wid, 0) != 1:
            v.append("WO %s assigned %d times (expected 1)" % (wid, seen.get(wid, 0)))
    for wid in seen:
        if wid not in wo_by_id:
            v.append("assignment references unknown WO %s" % wid)

    # (b) eligibility, (c) release, (d) duration.
    for a in assigns:
        wo = wo_by_id.get(a["wo"])
        tech = tech_by_id.get(a["tech"])
        if wo is None or tech is None:
            v.append("assignment %s/%s references missing WO or tech" % (a["wo"], a["tech"]))
            continue
        if tech["trade"] != wo["trade"]:
            v.append("tech %s trade %s ineligible for WO %s trade %s"
                     % (a["tech"], tech["trade"], a["wo"], wo["trade"]))
        if a["start_bh"] < wo["release_bh"] - TOL:
            v.append("WO %s starts %.6f before release %.6f"
                     % (a["wo"], a["start_bh"], wo["release_bh"]))
        if abs((a["end_bh"] - a["start_bh"]) - wo["p_bh"]) > TOL:
            v.append("WO %s duration %.6f != p_bh %.6f"
                     % (a["wo"], a["end_bh"] - a["start_bh"], wo["p_bh"]))

    # (e) no overlap per technician.
    by_tech = {}
    for a in assigns:
        by_tech.setdefault(a["tech"], []).append(a)
    for tid, jobs in by_tech.items():
        jobs = sorted(jobs, key=lambda a: a["start_bh"])
        for prev, cur in zip(jobs, jobs[1:]):
            if cur["start_bh"] < prev["end_bh"] - TOL:
                v.append("tech %s overlap: %s ends %.6f, %s starts %.6f"
                         % (tid, prev["wo"], prev["end_bh"], cur["wo"], cur["start_bh"]))
    return v


def wwt(instance, schedule):
    """Primary objective: sum_j w_j * max(0, end_j - due_j)."""
    wo_by_id = {w["id"]: w for w in instance["work_orders"]}
    total = 0.0
    for a in schedule["assignments"]:
        wo = wo_by_id[a["wo"]]
        total += wo["weight"] * max(0.0, a["end_bh"] - wo["due_bh"])
    return total


def main():
    with open(FIXTURE) as f:
        instance = json.load(f)

    failures = []
    results = {}  # method -> WWT

    # --- PDRs ---------------------------------------------------------------
    for rule in RULES:
        sched = pdrs.dispatch(instance, rule, seed=SEED)
        viol = check_feasible(instance, sched)
        if viol:
            failures.append("[%s] infeasible: %s" % (rule, viol[0]))
        results[rule] = wwt(instance, sched)

    # --- CP-SAT (10 s) ------------------------------------------------------
    csched = cpsat.solve(instance, time_limit_s=10.0, workers=8)
    viol = check_feasible(instance, csched)
    if viol:
        failures.append("[cpsat] infeasible: %s" % viol[0])
    cp_wwt = wwt(instance, csched)
    results[csched["method"]] = cp_wwt

    # --- assertions ---------------------------------------------------------
    if csched["status"] != "OPTIMAL":
        failures.append("cpsat status is %s, expected OPTIMAL" % csched["status"])
    if abs(cp_wwt - HAND_OPTIMUM) > TOL:
        failures.append("cpsat WWT %.6f != hand optimum %.1f" % (cp_wwt, HAND_OPTIMUM))
    if csched.get("objective_bh") is None or abs(csched["objective_bh"] - HAND_OPTIMUM) > TOL:
        failures.append("cpsat objective_bh %s != hand optimum %.1f"
                        % (csched.get("objective_bh"), HAND_OPTIMUM))
    for rule in RULES:
        if results[rule] < cp_wwt - TOL:
            failures.append("PDR %s WWT %.6f < cpsat WWT %.6f"
                            % (rule, results[rule], cp_wwt))

    # --- report -------------------------------------------------------------
    print("fixture: %s  (hand-derived optimum WWT = %.1f)" % (instance["meta"]["id"], HAND_OPTIMUM))
    print("-" * 40)
    print("%-12s %10s" % ("method", "WWT"))
    print("-" * 40)
    for rule in RULES:
        print("%-12s %10.3f" % (rule, results[rule]))
    print("%-12s %10.3f   status=%s  bound=%.3f"
          % (csched["method"], cp_wwt, csched["status"], csched["best_bound_bh"]))
    print("-" * 40)

    if failures:
        print("\nFAILURES:")
        for fmsg in failures:
            print("  - " + fmsg)
        sys.exit(1)

    print("\nALL BASELINE TESTS PASSED")


if __name__ == "__main__":
    main()
