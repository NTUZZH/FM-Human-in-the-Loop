#!/usr/bin/env python
"""P5 - rolling-CP-SAT replan diagnostic for Figure 10.

Reproduces the "replan on a clock, not only on arrivals" study on the campus-9,
size-400, crew-multiplier-0.6 (regime replay-tight) cell.  For each selected
instance it runs the rolling CP-SAT baseline TWICE and plain EDD once:

  variant 'arrival-only' : monkeypatch fmwos.rolling.REPLAN_EVERY_BH = inf so a
                           replan fires ONLY on an arrival event (the pre-fix
                           pathology: a single budget-starved big-bang solve is
                           never revisited).
  variant 'periodic'     : the shipped default (REPLAN_EVERY_BH = 4.0 bh), which
                           also replans on a clock while the queue is non-empty.
  EDD reference          : fmwos.pdrs.dispatch(instance, 'edd', seed=301).

Every schedule is scored by the INDEPENDENT validator (fmwos.validator).  The
business-hour timestamps of every actual replan are captured by a LOCAL wrapper
around fmwos.rolling._RollingSim._replan (src/fmwos/rolling.py is NOT edited).

Instances are scaled to crew m=0.6 with fmwos.tightness.scale_crew, exactly as
scripts/p4_dyneval.py does for the replay-tight regime, and rolling runs at
budget_s=2.0 (ROLLCP_BUDGET_S) so the numbers are comparable to results.csv.

Output: results/p4_dyneval/rolling_diag.json  (one record per (id, variant))
  {id, base_id, short, crew_multiplier, kind, variant, budget_s,
   replan_times_bh, n_replans, wwt, edd_wwt, makespan, horizon_bh,
   n_wos, window_bh, wall_seconds}

Run (PYTHONPATH=src, ortools env; ~10-20 min):
  PYTHONPATH=src conda run -n fjsp python scripts/p5_rolling_diag.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fmwos.rolling as R                       # noqa: E402
from fmwos import pdrs, tightness               # noqa: E402
from fmwos.validator import validate            # noqa: E402

INST_ROOT = ROOT / "data" / "processed" / "instances"
OUT = ROOT / "results" / "p4_dyneval" / "rolling_diag.json"

CREW_M = 0.6
BUDGET_S = 2.0

# campus-9 / size-400 replay cell.  0102 & 0105 are the two big-bang pathologies
# (docs/decision_log.md: 0102 3560->402==EDD, 0105 15471->22.6, EDD 0); 0101 & 0107 are
# spread-arrival controls (rolling competitive with EDD).  The figure uses the
# two pathologies + one spread control; the extra spread instance is provenance.
TARGETS = [
    ("c09_replay_400_0102", "0102", "pathological"),
    ("c09_replay_400_0105", "0105", "pathological"),
    ("c09_replay_400_0101", "0101", "spread"),
    ("c09_replay_400_0107", "0107", "spread"),
]

# --------------------------------------------------------------------------- #
# LOCAL wrapper: record the business-hour instant of every ACTUAL replan.
# (n_replans only increments when CP-SAT is actually invoked on a non-empty
# queue, so we key on that increment.)  src/fmwos/rolling.py is untouched.
# --------------------------------------------------------------------------- #
_orig_replan = R._RollingSim._replan


def _replan_rec(self, now):
    n0 = self.n_replans
    _orig_replan(self, now)
    if self.n_replans > n0:
        lst = getattr(self, "_replan_times", None)
        if lst is not None:
            lst.append(float(now))


R._RollingSim._replan = _replan_rec


def _run_rolling(instance, budget_s):
    """roll_cpsat, but keep the sim so we can read replan times + n_replans."""
    sim = R._RollingSim(instance, budget_s=budget_s)
    sim._replan_times = []
    t0 = time.perf_counter()
    sim.run()
    wall = time.perf_counter() - t0
    sched = sim.to_schedule(wall)
    return sched, list(sim._replan_times), sim.n_replans


def _wwt(instance, sched):
    res = validate(instance, sched)
    if not res["feasible"]:
        print("    [WARN] infeasible schedule:", res["violations"][:2], flush=True)
    return float(res["metrics"]["WWT"]), float(res["metrics"]["makespan"])


def _load(base_id):
    size = base_id.split("_")[2]
    campus = base_id.split("_")[0]  # c09
    path = INST_ROOT / campus / "replay" / size / (base_id + ".json")
    with open(path) as f:
        return json.load(f)


def main() -> int:
    records = []
    for base_id, short, kind in TARGETS:
        raw = _load(base_id)
        inst = tightness.scale_crew(raw, CREW_M)          # replay-tight m=0.6
        scaled_id = inst["meta"]["id"]
        n_wos = len(inst["work_orders"])
        window_bh = float(raw["meta"].get("window_bh", 0.0))
        n_tech = len(inst["technicians"])
        print(f"\n=== {scaled_id}  ({kind}, n_wos={n_wos}, "
              f"window_bh={window_bh:g}, crew={n_tech}) ===", flush=True)

        # EDD reference (deterministic) -------------------------------------
        edd_sched = pdrs.dispatch(inst, "edd", seed=301)
        edd_wwt, edd_mk = _wwt(inst, edd_sched)
        print(f"  EDD          wwt={edd_wwt:.4f}  makespan={edd_mk:.2f}", flush=True)

        # rolling: arrival-only then periodic -------------------------------
        for variant, every in (("arrival-only", float("inf")),
                               ("periodic", 4.0)):
            R.REPLAN_EVERY_BH = every
            t0 = time.perf_counter()
            sched, times, n_rep = _run_rolling(inst, BUDGET_S)
            wall = time.perf_counter() - t0
            wwt, mk = _wwt(inst, sched)
            print(f"  {variant:<12} wwt={wwt:.4f}  n_replans={n_rep}  "
                  f"replans@bh={[round(t, 2) for t in times][:8]}"
                  f"{'...' if len(times) > 8 else ''}  makespan={mk:.2f}  "
                  f"wall={wall:.1f}s", flush=True)
            records.append({
                "id": scaled_id, "base_id": base_id, "short": short,
                "crew_multiplier": CREW_M, "kind": kind, "variant": variant,
                "budget_s": BUDGET_S,
                "replan_times_bh": [round(float(t), 4) for t in times],
                "n_replans": int(n_rep),
                "wwt": round(wwt, 4), "edd_wwt": round(edd_wwt, 4),
                "makespan": round(mk, 4), "horizon_bh": round(max(mk, edd_mk), 4),
                "n_wos": n_wos, "window_bh": window_bh,
                "wall_seconds": round(wall, 3),
            })
        R.REPLAN_EVERY_BH = 4.0  # restore default

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n[rolling_diag] wrote {OUT} ({len(records)} records)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
