"""Fair-compute assertion: configured DAgger budget equals Y1's single run."""

import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

# load the runner module directly (it lives under scripts/)
_spec = importlib.util.spec_from_file_location(
    "y3_p2_train", os.path.join(_ROOT, "scripts", "y3_p2_train.py"))
y3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(y3)


def test_budget_equals_y1():
    y1_budget, cfg = y3.y1_single_run_budget()
    expected = int(cfg["updates"]) * int(cfg["n_envs"]) * int(cfg["steps_per_env"])
    assert y1_budget == expected == 4915200, (y1_budget, expected)
    # 8 outer iters, 16 envs, 512 steps: configured must equal Y1's budget exactly
    upi, configured = y3.resolve_schedule(y1_budget, 8, 16, 512, y1_budget, 1.0)
    assert configured == y1_budget, (configured, y1_budget)
    assert upi == 75, upi
    assert upi * 8 * 16 * 512 == y1_budget
    print("  fair-compute: Y1 budget=%d == 8 iters x %d updates x 16 x 512 OK"
          % (y1_budget, upi))


def test_violation_raises():
    y1_budget, _ = y3.y1_single_run_budget()
    # a total_budget that does not match Y1 at full frac must trip the assertion
    raised = False
    try:
        y3.resolve_schedule(y1_budget - 8192, 8, 16, 512, y1_budget, 1.0)
    except AssertionError:
        raised = True
    assert raised, "fair-compute violation was NOT caught"
    print("  fair-compute: mismatched budget at frac=1.0 raises AssertionError OK")


def test_smoke_relaxed():
    y1_budget, _ = y3.y1_single_run_budget()
    total = int(round(y1_budget * 0.05))
    upi, configured = y3.resolve_schedule(total, 2, 16, 512, y1_budget, 0.05)
    # smoke: internal consistency only, ~5% of Y1
    assert configured == upi * 2 * 16 * 512
    assert 0.03 * y1_budget < configured < 0.07 * y1_budget, configured
    print("  fair-compute: smoke frac=0.05 -> %d steps (~%.1f%% of Y1) OK"
          % (configured, 100 * configured / y1_budget))


if __name__ == "__main__":
    test_budget_equals_y1()
    test_violation_raises()
    test_smoke_relaxed()
    print("ALL FAIR-COMPUTE TESTS PASSED")
