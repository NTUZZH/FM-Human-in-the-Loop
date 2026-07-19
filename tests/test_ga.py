"""Genetic-algorithm baseline tests -- plain python script (no pytest).

Run:  PYTHONPATH=src python tests/test_ga.py

Two checks:

  1. Fixture (tests/fixtures/tiny_instance.json, hand-derived optimum 32, best
     PDR 56): run solve_ga (budget 5 s), assert the schedule passes the
     INDEPENDENT validator (fmwos.validator.validate) and that GA WWT <= the best
     PDR WWT (56).  The known optimum is 32; the GA should reach it.

  2. One real instance (first sorted file under
     data/processed/instances/c05/replay/150/): run solve_ga (budget 15 s), assert
     the schedule is feasible and GA WWT <= the best PDR WWT.

Prints the GA WWTs / generations and finally 'ALL GA TESTS PASSED'.
"""

import glob
import json
import os
import sys

# Make ``fmwos`` importable whether or not PYTHONPATH=src is set.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import ga, pdrs  # noqa: E402
from fmwos.validator import validate  # noqa: E402

FIXTURE = os.path.join(_ROOT, "tests", "fixtures", "tiny_instance.json")
REAL_GLOB = os.path.join(
    _ROOT, "data", "processed", "instances", "c05", "replay", "150", "*.json"
)
PDR_RULES = ["edd", "wspt", "atc", "pfifo", "mor"]
HAND_OPTIMUM = 32.0
SEED = 301
TOL = 1e-6


def best_pdr_wwt(instance):
    """Best (lowest) WWT over the 5 deterministic PDRs, via the validator."""
    best = None
    for rule in PDR_RULES:
        sched = pdrs.dispatch(instance, rule, seed=SEED)
        wwt = validate(instance, sched)["metrics"]["WWT"]
        best = wwt if best is None else min(best, wwt)
    return best


def main():
    failures = []

    # ----------------------------------------------------------------- fixture
    with open(FIXTURE) as f:
        fixture = json.load(f)

    fx_best_pdr = best_pdr_wwt(fixture)
    fx_sched = ga.solve_ga(fixture, budget_s=5, seed=SEED)
    fx_res = validate(fixture, fx_sched)
    fx_wwt = fx_res["metrics"]["WWT"]

    if not fx_res["feasible"]:
        failures.append("fixture: GA schedule INFEASIBLE: %s" % fx_res["violations"][:2])
    if fx_wwt > fx_best_pdr + TOL:
        failures.append(
            "fixture: GA WWT %.3f > best PDR WWT %.3f" % (fx_wwt, fx_best_pdr)
        )
    hit_opt = abs(fx_wwt - HAND_OPTIMUM) <= TOL
    print(
        "fixture %s: GA WWT=%.3f (best PDR=%.3f, optimum=%.1f)  hit_optimum=%s  "
        "generations=%d  evals=%d"
        % (
            fixture["meta"]["id"], fx_wwt, fx_best_pdr, HAND_OPTIMUM,
            hit_opt, fx_sched["generations"], fx_sched["decisions"],
        )
    )

    # ------------------------------------------------------------ real instance
    real_files = sorted(glob.glob(REAL_GLOB))
    if not real_files:
        failures.append("no real instances found under %s" % REAL_GLOB)
    else:
        real_path = real_files[0]
        with open(real_path) as f:
            real = json.load(f)

        r_best_pdr = best_pdr_wwt(real)
        r_sched = ga.solve_ga(real, budget_s=15, seed=SEED)
        r_res = validate(real, r_sched)
        r_wwt = r_res["metrics"]["WWT"]

        if not r_res["feasible"]:
            failures.append(
                "real: GA schedule INFEASIBLE: %s" % r_res["violations"][:2]
            )
        if r_wwt > r_best_pdr + TOL:
            failures.append(
                "real: GA WWT %.3f > best PDR WWT %.3f" % (r_wwt, r_best_pdr)
            )
        print(
            "real %s: GA WWT=%.3f  best PDR WWT=%.3f  generations=%d  evals=%d  "
            "wall=%.2fs"
            % (
                real["meta"]["id"], r_wwt, r_best_pdr,
                r_sched["generations"], r_sched["decisions"], r_sched["wall_seconds"],
            )
        )

    # ------------------------------------------------------------------ verdict
    if failures:
        print("\nFAILURES:")
        for msg in failures:
            print("  - " + msg)
        sys.exit(1)

    print("\nALL GA TESTS PASSED")


if __name__ == "__main__":
    main()
