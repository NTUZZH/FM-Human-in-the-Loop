"""Plain-Python (no pytest) tests for the storm-v2 utilization-sweep track.

Run with:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate fjsp && \
    PYTHONPATH=src python tests/test_storm2.py

Covers the fixed-window generator (``generator.generate_window`` +
``generator.base_utilization``) -- determinism, linear scaling of the drawn
work-order count n with arrival_multiplier, and the realized-utilization
mapping -- plus the PM/priority invariants, then a small end-to-end smoke of
scripts/p4_dyneval.py's storm2 regime (generation + eval to a scratch --out).

Fast path: fits parameters for campus 5 ONLY (raw filtered to campus 5 before
cleaning) as in tests/test_generator.py.  The smoke persists the storm2
instance corpus + index rows under data/processed/instances (by design); only
the scratch results --out dir is cleaned up.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmwos import generator             # noqa: E402
from fmwos import io                    # noqa: E402
from fmwos import timeaxis as ta        # noqa: E402

RAW = ROOT / "data" / "raw" / "FMUCD.csv"
CAMPUS = 5
WINDOW = 80.0

META_KEYS = {"id", "campus", "track", "size_class", "window_start", "window_bh",
             "provenance", "seed", "crew_multiplier", "pm_share_override",
             "arrival_multiplier", "redrawn"}
WO_KEYS = {"id", "trade", "p_bh", "release_bh", "due_bh", "priority", "weight",
           "building", "is_pm"}


def _fit_campus5() -> dict:
    raw = io.load_raw(RAW)
    raw = raw[raw["UniversityID"] == CAMPUS]
    clean, _ = io.clean(raw)
    return generator.fit_params(clean, CAMPUS)


def _realized_util(inst) -> float:
    total_p = sum(float(w["p_bh"]) for w in inst["work_orders"])
    n_crew = len(inst["technicians"])
    win = float(inst["meta"]["window_bh"])
    return total_p / (n_crew * win)


# --------------------------------------------------------------------------- #
# Generator unit tests
# --------------------------------------------------------------------------- #
def test_schema_and_meta(params):
    inst = generator.generate_window(params, window_bh=WINDOW, seed=7,
                                     arrival_multiplier=3.0)
    assert set(inst.keys()) == {"meta", "trades", "technicians", "work_orders"}
    assert META_KEYS.issubset(set(inst["meta"].keys())), inst["meta"].keys()
    m = inst["meta"]
    assert m["track"] == "storm2", m["track"]
    assert m["provenance"] == "C"
    assert m["window_start"] == "synthetic"
    assert abs(float(m["window_bh"]) - WINDOW) < 1e-9, m["window_bh"]
    assert m["seed"] == 7 and m["campus"] == CAMPUS
    assert m["arrival_multiplier"] == 3.0
    assert m["redrawn"] is False
    n = len(inst["work_orders"])
    assert n > 0, "empty draw"
    assert m["size_class"] == n, (m["size_class"], n)
    for wo in inst["work_orders"]:
        assert set(wo.keys()) == WO_KEYS, wo.keys()
    tech_trades = {t["trade"] for t in inst["technicians"]}
    for wo in inst["work_orders"]:
        assert wo["trade"] in tech_trades, wo["trade"]
        assert 0.0 <= wo["release_bh"] <= WINDOW + 1e-6, wo["release_bh"]
    print("PASS test_schema_and_meta (n=%d)" % n)


def test_determinism(params):
    a = generator.generate_window(params, window_bh=WINDOW, seed=11,
                                  arrival_multiplier=3.0)
    b = generator.generate_window(params, window_bh=WINDOW, seed=11,
                                  arrival_multiplier=3.0)
    sa = json.dumps(a, sort_keys=True)
    sb = json.dumps(b, sort_keys=True)
    assert sa == sb, "same seed produced different JSON"
    c = generator.generate_window(params, window_bh=WINDOW, seed=12,
                                  arrival_multiplier=3.0)
    assert json.dumps(c, sort_keys=True) != sa, "seed had no effect"
    print("PASS test_determinism")


def test_n_scales_with_arrival(params):
    """Doubling arrival_multiplier ~doubles n (Poisson mean is linear in it)."""
    ratios = []
    for seed in (1, 2, 3, 4, 5):
        lo = generator.generate_window(params, window_bh=WINDOW, seed=seed,
                                       arrival_multiplier=2.0)
        hi = generator.generate_window(params, window_bh=WINDOW, seed=seed,
                                       arrival_multiplier=4.0)
        n_lo, n_hi = len(lo["work_orders"]), len(hi["work_orders"])
        assert n_lo > 0
        r = n_hi / n_lo
        ratios.append(r)
        assert 1.6 <= r <= 2.4, (seed, n_lo, n_hi, r)
    mean_r = sum(ratios) / len(ratios)
    assert 1.8 <= mean_r <= 2.2, mean_r
    print("PASS test_n_scales_with_arrival (mean n(4x)/n(2x)=%.3f)" % mean_r)


def test_realized_utilization(params):
    """Realized utilization centers on u_target=1.0 over 10 draws (+-0.15).

    arrival_multiplier = u_target / u0 where u0 = base_utilization; the
    clipped-lognormal mean estimator makes E[realized] = u_target, so the
    +-0.15 tolerance the paper trusts is met with wide margin (fitted-pack
    rates do NOT make this infeasible -- no remapping needed).
    """
    u_target = 1.0
    u0 = generator.base_utilization(params, crew_multiplier=1.0)
    assert u0 > 0.0, u0
    am = u_target / u0
    reals = []
    for seed in range(10):
        inst = generator.generate_window(params, window_bh=WINDOW,
                                         seed=1000 + seed, arrival_multiplier=am)
        reals.append(_realized_util(inst))
    mean_u = sum(reals) / len(reals)
    assert abs(mean_u - u_target) <= 0.15, (u0, am, mean_u)
    print("PASS test_realized_utilization "
          "(u0=%.4f, am=%.3f, mean realized=%.3f, target=%.1f)"
          % (u0, am, mean_u, u_target))


def test_pm_and_priority_invariants(params):
    saw_pm = False
    floor = generator.P_BH_FLOOR
    cap = float(params["labor_cap"])
    for seed in (1, 2, 3):
        inst = generator.generate_window(params, window_bh=WINDOW, seed=seed,
                                         arrival_multiplier=3.0)
        for wo in inst["work_orders"]:
            prio = wo["priority"]
            assert prio in (1, 2, 3, 4), prio
            assert abs(wo["due_bh"] - (wo["release_bh"] + ta.SLA_BH[prio])) < 1e-4
            assert wo["weight"] == ta.WEIGHT[prio], wo
            assert floor - 1e-9 <= wo["p_bh"] <= cap + 1e-9, wo["p_bh"]
            if wo["is_pm"]:
                saw_pm = True
                assert prio == 4, wo          # R5a: PM is always class 4
    assert saw_pm, "no PM work orders drawn (cannot verify R5a)"
    print("PASS test_pm_and_priority_invariants")


def test_pm_share_override(params):
    """pm_share_override forces (nearly) all WOs to PM -> priority 4."""
    inst = generator.generate_window(params, window_bh=WINDOW, seed=3,
                                     arrival_multiplier=2.0,
                                     pm_share_override=1.0)
    assert inst["meta"]["pm_share_override"] == 1.0
    pm = [wo for wo in inst["work_orders"] if wo["is_pm"]]
    assert len(pm) == len(inst["work_orders"]), "override=1.0 left non-PM WOs"
    assert all(wo["priority"] == 4 for wo in inst["work_orders"])
    print("PASS test_pm_share_override")


# --------------------------------------------------------------------------- #
# Smoke: scripts/p4_dyneval.py storm2 regime (generation + eval -> scratch out)
# --------------------------------------------------------------------------- #
def test_smoke_dyneval():
    scratch = Path(tempfile.mkdtemp(prefix="storm2_smoke_"))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        sys.executable, str(ROOT / "scripts" / "p4_dyneval.py"),
        "--regime", "storm2", "--with-storm2",
        "--limit", "3", "--workers", "2", "--no-rollcp",
        "--rl-tag", "v2rl", "--rl-dir", str(ROOT / "results" / "p3_train" / "v2"),
        "--out", str(scratch),
    ]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                           text=True, timeout=2400)
        if r.returncode != 0:
            sys.stdout.write(r.stdout[-3000:])
            sys.stderr.write(r.stderr[-3000:])
            raise AssertionError("dyneval storm2 smoke exited %d" % r.returncode)

        res_csv = scratch / "results.csv"
        assert res_csv.exists(), "no results.csv written to scratch out"
        with open(res_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        s2 = [row for row in rows if row.get("regime") == "storm2"]
        assert s2, "no storm2 rows in results.csv"

        # every storm2 row: feasible is an INT flag (0/1) and populated
        # u_target / u_realized columns
        methods = set()
        for row in s2:
            assert row["feasible"] in ("0", "1"), row["feasible"]
            assert int(row["feasible"]) == 1, ("infeasible storm2 row", row["id"])
            assert row["u_target"] not in ("", None), row
            assert row["u_realized"] not in ("", None), row
            ut = float(row["u_target"])
            ur = float(row["u_realized"])
            assert ut in (0.7, 0.9, 1.0, 1.1, 1.3), ut
            assert abs(ur - ut) < 0.3, (row["id"], ut, ur)  # per-instance, loose
            methods.add(row["method"])
        # PDRs + the active v2rl tag ran
        assert {"edd", "wspt", "atc"} <= methods, methods
        assert any(m.startswith("v2rl") for m in methods), methods

        # non-storm2 rows (none expected here) would carry null u_target/u_realized
        n_ids = len({row["id"] for row in s2})
        print("PASS test_smoke_dyneval "
              "(%d storm2 rows over %d instance-config(s); methods=%s)"
              % (len(s2), n_ids, sorted(methods)))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)   # clean ONLY the scratch out


if __name__ == "__main__":
    print("fitting campus 5 (fast path) ...", flush=True)
    params = _fit_campus5()
    u0 = generator.base_utilization(params)
    print("  fitted %d trades; labor_cap=%.2f; u0=%.4f (am@u=1.0 -> %.3f)"
          % (len(params["trades"]), params["labor_cap"], u0, 1.0 / u0),
          flush=True)

    test_schema_and_meta(params)
    test_determinism(params)
    test_n_scales_with_arrival(params)
    test_realized_utilization(params)
    test_pm_and_priority_invariants(params)
    test_pm_share_override(params)
    print("running dyneval storm2 smoke (generates the storm2 corpus; "
          "may take a few minutes on first run) ...", flush=True)
    test_smoke_dyneval()
    print("ALL STORM2 TESTS PASSED")
