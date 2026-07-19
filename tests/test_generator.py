"""Plain-Python (no pytest) tests for the calibrated instance generator.

Run with:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate fjsp && \
    PYTHONPATH=src python tests/test_generator.py

Fast path: fits parameters for campus 5 ONLY (raw is filtered to campus 5
*before* cleaning, so the R7 dedup groupby stays small) and exercises the
generator's schema, determinism and knob behaviour.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmwos import generator, io          # noqa: E402
from fmwos import timeaxis as ta         # noqa: E402

RAW = ROOT / "data" / "raw" / "FMUCD.csv"
CAMPUS = 5


def _fit_campus5() -> dict:
    """Fast fit: filter raw to campus 5 before cleaning, then fit_params."""
    raw = io.load_raw(RAW)
    raw = raw[raw["UniversityID"] == CAMPUS]
    clean, _ = io.clean(raw)
    return generator.fit_params(clean, CAMPUS)


META_KEYS = {"id", "campus", "track", "size_class", "window_start", "window_bh",
             "provenance", "seed"}
WO_KEYS = {"id", "trade", "p_bh", "release_bh", "due_bh", "priority", "weight",
           "building", "is_pm"}


def test_schema_and_size(params):
    inst = generator.generate(params, size=50, seed=1)
    assert set(inst.keys()) == {"meta", "trades", "technicians", "work_orders"}, \
        inst.keys()
    assert META_KEYS.issubset(set(inst["meta"].keys())), inst["meta"].keys()
    # generator-specific meta
    assert inst["meta"]["track"] == "generator"
    assert inst["meta"]["provenance"] == "C"
    assert inst["meta"]["window_start"] == "synthetic"
    assert inst["meta"]["seed"] == 1
    assert inst["meta"]["campus"] == CAMPUS
    for extra in ("crew_multiplier", "pm_share_override", "arrival_multiplier"):
        assert extra in inst["meta"], extra
    assert len(inst["work_orders"]) == 50, len(inst["work_orders"])
    assert inst["meta"]["size_class"] == 50
    for wo in inst["work_orders"]:
        assert set(wo.keys()) == WO_KEYS, wo.keys()
    # every WO's trade must have >= 1 eligible technician
    tech_trades = {t["trade"] for t in inst["technicians"]}
    for wo in inst["work_orders"]:
        assert wo["trade"] in tech_trades, wo["trade"]
    print("PASS test_schema_and_size")


def test_size_seeds(params):
    for seed in (1, 2, 3):
        inst = generator.generate(params, size=50, seed=seed)
        assert len(inst["work_orders"]) == 50, (seed, len(inst["work_orders"]))
    print("PASS test_size_seeds")


def test_monotone_releases(params):
    for seed in (1, 2, 3):
        inst = generator.generate(params, size=50, seed=seed)
        rels = [wo["release_bh"] for wo in inst["work_orders"]]
        for a, b in zip(rels, rels[1:]):
            assert a <= b + 1e-9, (seed, a, b)
        # window_bh is the last kept release (>= 8)
        assert inst["meta"]["window_bh"] >= 8.0 - 1e-9
        assert abs(inst["meta"]["window_bh"] - max(rels)) < 1e-6 or \
            inst["meta"]["window_bh"] == 8.0
    print("PASS test_monotone_releases")


def test_due_equals_release_plus_sla(params):
    for seed in (1, 2, 3):
        inst = generator.generate(params, size=50, seed=seed)
        for wo in inst["work_orders"]:
            expected = wo["release_bh"] + ta.SLA_BH[wo["priority"]]
            assert abs(wo["due_bh"] - expected) < 1e-4, (wo["due_bh"], expected)
            assert wo["weight"] == ta.WEIGHT[wo["priority"]]
    print("PASS test_due_equals_release_plus_sla")


def test_pm_implies_priority4(params):
    saw_pm = False
    for seed in (1, 2, 3, 4, 5):
        inst = generator.generate(params, size=50, seed=seed)
        for wo in inst["work_orders"]:
            if wo["is_pm"]:
                saw_pm = True
                assert wo["priority"] == 4, wo
    assert saw_pm, "no PM work orders drawn across seeds 1..5 (cannot verify R5a)"
    print("PASS test_pm_implies_priority4")


def test_determinism(params):
    a = generator.generate(params, size=50, seed=1)
    b = generator.generate(params, size=50, seed=1)
    sa = json.dumps(a, sort_keys=True)
    sb = json.dumps(b, sort_keys=True)
    assert sa == sb, "same seed produced different JSON"
    # different seed -> (almost surely) different
    c = generator.generate(params, size=50, seed=2)
    assert json.dumps(c, sort_keys=True) != sa, "seed had no effect"
    print("PASS test_determinism")


def test_crew_multiplier(params):
    full = generator.generate(params, size=50, seed=1, crew_multiplier=1.0)
    half = generator.generate(params, size=50, seed=1, crew_multiplier=0.5)
    cap = params["capacity"]

    full_counts = Counter(t["trade"] for t in full["technicians"])
    half_counts = Counter(t["trade"] for t in half["technicians"])

    for trade, crew in cap.items():
        assert full_counts[trade] == max(1, int(round(crew * 1.0))), trade
        assert half_counts[trade] == max(1, int(round(crew * 0.5))), \
            (trade, crew, half_counts[trade])
    # at least one trade actually shrank (sanity that 0.5 did something)
    assert sum(half_counts.values()) < sum(full_counts.values()), \
        "crew_multiplier=0.5 did not reduce total crew"
    print("PASS test_crew_multiplier")


if __name__ == "__main__":
    print("fitting campus 5 (fast path) ...", flush=True)
    params = _fit_campus5()
    print(f"  fitted {len(params['trades'])} trades; "
          f"labor_cap={params['labor_cap']:.2f}; "
          f"span_bh={params['train_span_bh']:.0f}", flush=True)

    test_schema_and_size(params)
    test_size_seeds(params)
    test_monotone_releases(params)
    test_due_equals_release_plus_sla(params)
    test_pm_implies_priority4(params)
    test_determinism(params)
    test_crew_multiplier(params)
    print("ALL GENERATOR TESTS PASSED")
