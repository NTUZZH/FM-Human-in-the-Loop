"""True-objective channel-mode tests (Paper Y3, Phase P1.5) -- plain python.

Run:  PYTHONPATH=src python tests/test_true_objective.py

The P1.5 re-scope gives ``score_true`` a ``deadline_mode``:

(a) WEIGHT_ONLY == OLD  -- deadline_mode="recorded" (the weight_only E6 boundary)
    reproduces the PRE-P1.5 scorer byte-for-byte:
        TWT_true = sum_j w*_j * max(0, C_j - d_recorded_j).
(b) FULL_CLASS_SHIFT     -- deadline_mode="true" (the headline) equals the
    verified y3_verc / y3_cont d* scoring EXACTLY on a storm2 gate cell:
        TWT_true = sum_j w*_j * max(0, C_j - d*_j),  d*_j = r_j + SLA(c*_j).
(c) DEFAULT FROM CHANNEL -- a caller who passes nothing gets "true" under a
    full_class_shift overlay and "recorded" under a weight_only overlay.

The storm2 c9 u100 sample is scored on the plain-ATC and the full-info oracle
schedules (both real dynamic-env schedules) so the check exercises non-trivial
tardiness.
"""

import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.env import DispatchEnv                         # noqa: E402
from fmwos.hitl import deciders as dec                    # noqa: E402
from fmwos.hitl import overlay as ov                      # noqa: E402
from fmwos.hitl.supervisor import Supervisor              # noqa: E402
from fmwos.hitl import true_objective as TO               # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_SLA = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}
SEED = 301


def _cell_files(n=6):
    fs = sorted(glob.glob(os.path.join(
        _INST, "c09", "storm2", "w80", "c09_storm2_w80_u100_*.json")))[:n]
    if not fs:      # fall back to a replay cell if storm2 is absent
        fs = sorted(glob.glob(os.path.join(
            _INST, "c09", "replay", "150", "*.json")))[:n]
    return fs


def _verc_dstar_score(inst, sched, applied):
    """Independent copy of y3_verc/_score_twt_star (reads d*=r+SLA(c*), w*)."""
    wstar = applied["w_star"]
    cs = applied["c_star"]
    dstar = {w["id"]: float(w["release_bh"]) + _SLA[cs[w["id"]]]
             for w in inst["work_orders"]}
    twt = 0.0
    for a in sched.get("assignments", []) or []:
        wid = a.get("wo")
        end = a.get("end_bh")
        if wid is None or end is None or wid not in dstar:
            continue
        twt += wstar[wid] * max(0.0, float(end) - dstar[wid])
    return twt


def _old_recorded_score(inst, sched, applied):
    """Independent copy of the PRE-P1.5 score_true block (w*, recorded due)."""
    wstar = applied["w_star"]
    wo_by = {w["id"]: w for w in inst["work_orders"]}
    twt = 0.0
    for a in sched.get("assignments", []) or []:
        wo = wo_by.get(a.get("wo"))
        end = a.get("end_bh")
        if wo is None or end is None:
            continue
        twt += wstar.get(wo["id"], float(wo["weight"])) * max(
            0.0, float(end) - float(wo["due_bh"]))
    return twt


def _schedules(inst, overlay, applied):
    atc = dec.run_rule(DispatchEnv(inst), "atc", seed=SEED)
    sup = Supervisor(overlay, inst, rho=0.0, applied=applied)   # due=d* (full)
    orc = dec.run_oracle_greedy(DispatchEnv(inst), sup, seed=SEED)
    return {"atc": atc, "oracle": orc}


def test_weight_only_reproduces_old(failures):
    print("(a) weight_only (deadline_mode='recorded') == OLD scorer, byte-exact")
    ov_full = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL", master_seed=12345))
    ov_wo = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL",
                                        master_seed=12345, channel="weight_only"))
    max_err = 0.0
    n = 0
    for p in _cell_files():
        inst = json.load(open(p))
        ap = ov_full.apply(inst)
        for sched in _schedules(inst, ov_full, ap).values():
            # explicit override on the full overlay
            st1 = TO.score_true(inst, sched, ov_full, ap, deadline_mode="recorded")
            # channel default on the weight_only overlay
            st2 = TO.score_true(inst, sched, ov_wo, ov_wo.apply(inst))
            old = _old_recorded_score(inst, sched, ap)
            if st1["deadline_mode"] != "recorded" or st2["deadline_mode"] != "recorded":
                failures.append("weight_only did not resolve deadline_mode='recorded'")
            max_err = max(max_err, abs(st1["TWT_true"] - old),
                          abs(st2["TWT_true"] - old))
            n += 1
    if max_err > 1e-9:
        failures.append("weight_only TWT_true != OLD scorer (max err %.3e)" % max_err)
    print("    %d schedules, max|weight_only - OLD| = %.2e" % (n, max_err))


def test_full_matches_verc(failures):
    print("(b) full_class_shift (deadline_mode='true') == verified d* scoring")
    ov_full = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL", master_seed=12345))
    max_err = 0.0
    n = 0
    for p in _cell_files():
        inst = json.load(open(p))
        ap = ov_full.apply(inst)
        for sched in _schedules(inst, ov_full, ap).values():
            st = TO.score_true(inst, sched, ov_full, ap)      # default -> 'true'
            if st["deadline_mode"] != "true":
                failures.append("full_class_shift did not resolve deadline_mode='true'")
            max_err = max(max_err, abs(st["TWT_true"] - _verc_dstar_score(inst, sched, ap)))
            n += 1
    if max_err > 1e-9:
        failures.append("full_class_shift TWT_true != verc d* scoring (max err %.3e)" % max_err)
    print("    %d schedules, max|full - verc_d*| = %.2e" % (n, max_err))


def test_default_resolution(failures):
    print("(c) default deadline_mode resolves from the overlay channel")
    ov_full = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL", master_seed=12345))
    ov_wo = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL",
                                        master_seed=12345, channel="weight_only"))
    ok = (TO.resolve_deadline_mode(ov_full) == "true"
          and TO.resolve_deadline_mode(ov_wo) == "recorded"
          and TO.resolve_deadline_mode(ov_full, "recorded") == "recorded"
          and TO.resolve_deadline_mode(ov_wo, "true") == "true")
    if not ok:
        failures.append("deadline_mode default/override resolution wrong")
    print("    full->true, weight_only->recorded, explicit override honoured: %s" % ok)


def main():
    failures = []
    test_weight_only_reproduces_old(failures)
    print()
    test_full_matches_verc(failures)
    print()
    test_default_resolution(failures)
    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("ALL TRUE-OBJECTIVE TESTS PASSED")


if __name__ == "__main__":
    main()
