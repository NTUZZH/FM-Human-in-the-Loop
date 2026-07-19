"""Plain-Python (no pytest) tests for the independent feasibility validator.

Run with:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate fjsp && \
    PYTHONPATH=src python tests/test_validator.py

Covers one fully hand-computed feasible fixture (every metric checked against
arithmetic worked out in the comments) plus one deliberately-broken schedule
per violation type (a)-(f).
"""

from fmwos.validator import validate


# ---------------------------------------------------------------------------
# Base fixture: a small, fully feasible instance + schedule.
#
# Technicians:
#   T0 -> trade D20
#   T1 -> trade E10
#
# Work orders (id, trade, p_bh, release_bh, due_bh, priority, weight):
#   WO1  D20  p=4  r=0  due=8   P1  w=8
#   WO2  D20  p=1  r=0  due=4   P2  w=4
#   WO3  E10  p=6  r=0  due=5   P3  w=2
#   WO4  E10  p=1  r=0  due=80  P4  w=1
#
# Schedule (tech, start_bh, end_bh):
#   WO1 on T0: [0, 4]   WO2 on T0: [4, 5]   (T0 back-to-back, no overlap)
#   WO3 on T1: [0, 6]   WO4 on T1: [6, 7]   (T1 back-to-back, no overlap)
# ---------------------------------------------------------------------------

def base_instance():
    return {
        "meta": {"id": "c05_test_0001"},
        "trades": ["D20", "E10"],
        "technicians": [
            {"id": "T0", "trade": "D20"},
            {"id": "T1", "trade": "E10"},
        ],
        "work_orders": [
            {"id": "WO1", "trade": "D20", "p_bh": 4.0, "release_bh": 0.0,
             "due_bh": 8.0, "priority": 1, "weight": 8.0},
            {"id": "WO2", "trade": "D20", "p_bh": 1.0, "release_bh": 0.0,
             "due_bh": 4.0, "priority": 2, "weight": 4.0},
            {"id": "WO3", "trade": "E10", "p_bh": 6.0, "release_bh": 0.0,
             "due_bh": 5.0, "priority": 3, "weight": 2.0},
            {"id": "WO4", "trade": "E10", "p_bh": 1.0, "release_bh": 0.0,
             "due_bh": 80.0, "priority": 4, "weight": 1.0},
        ],
    }


def base_schedule():
    return {
        "instance_id": "c05_test_0001",
        "method": "edd",
        "seed": 301,
        "wall_seconds": 0.012,
        "decisions": 4,
        "assignments": [
            {"wo": "WO1", "tech": "T0", "start_bh": 0.0, "end_bh": 4.0},
            {"wo": "WO2", "tech": "T0", "start_bh": 4.0, "end_bh": 5.0},
            {"wo": "WO3", "tech": "T1", "start_bh": 0.0, "end_bh": 6.0},
            {"wo": "WO4", "tech": "T1", "start_bh": 6.0, "end_bh": 7.0},
        ],
    }


def approx(a, b, tol=1e-12):
    return abs(a - b) <= tol


def has_violation(result, marker):
    """True iff some violation string carries the given ``(x)`` type marker."""
    return any(v.startswith(marker) for v in result["violations"])


# ---------------------------------------------------------------------------
# Test 1: feasible fixture -- verify feasibility AND every metric value.
# ---------------------------------------------------------------------------

def test_feasible_and_metrics():
    result = validate(base_instance(), base_schedule())

    assert result["feasible"] is True, result["violations"]
    assert result["violations"] == [], result["violations"]

    m = result["metrics"]

    # WWT = sum_j w_j * max(0, end_j - due_j)
    #   WO1: 8 * max(0, 4 - 8)  = 8 * 0 = 0
    #   WO2: 4 * max(0, 5 - 4)  = 4 * 1 = 4
    #   WO3: 2 * max(0, 6 - 5)  = 2 * 1 = 2
    #   WO4: 1 * max(0, 7 - 80) = 1 * 0 = 0
    #   total = 0 + 4 + 2 + 0 = 6.0
    assert approx(m["WWT"], 6.0), m["WWT"]

    # makespan = max end_bh = max(4, 5, 6, 7) = 7.0
    assert approx(m["makespan"], 7.0), m["makespan"]

    # mean_flow = mean(end - release)
    #   flows = (4-0), (5-0), (6-0), (7-0) = 4, 5, 6, 7
    #   mean = (4 + 5 + 6 + 7) / 4 = 22 / 4 = 5.5
    assert approx(m["mean_flow"], 5.5), m["mean_flow"]

    # breach_share = share with end > due + 1e-9
    #   breached: WO2 (5>4), WO3 (6>5)  -> 2 of 4 -> 0.5
    assert approx(m["breach_share"], 0.5), m["breach_share"]

    # per-priority breach shares:
    #   P1: 1 WO (WO1), 0 breached -> 0.0
    #   P2: 1 WO (WO2), 1 breached -> 1.0
    #   P3: 1 WO (WO3), 1 breached -> 1.0
    #   P4: 1 WO (WO4), 0 breached -> 0.0
    pp = m["per_priority_breach_share"]
    assert approx(pp[1], 0.0), pp
    assert approx(pp[2], 1.0), pp
    assert approx(pp[3], 1.0), pp
    assert approx(pp[4], 0.0), pp

    # pass-through fields copied verbatim from the schedule.
    assert m["wall_seconds"] == 0.012, m["wall_seconds"]
    assert m["decisions"] == 4, m["decisions"]

    print("PASS test_feasible_and_metrics")


# ---------------------------------------------------------------------------
# Test 2 (a): work order missing / duplicated.
# ---------------------------------------------------------------------------

def test_a_missing():
    inst = base_instance()
    sched = base_schedule()
    # Drop WO4's assignment entirely -> WO4 is never assigned.
    sched["assignments"] = [a for a in sched["assignments"] if a["wo"] != "WO4"]
    result = validate(inst, sched)
    assert result["feasible"] is False
    assert has_violation(result, "(a)"), result["violations"]
    assert any("WO4" in v and "missing" in v for v in result["violations"]), \
        result["violations"]
    print("PASS test_a_missing")


def test_a_duplicated():
    inst = base_instance()
    sched = base_schedule()
    # Assign WO1 a second time (on the other tech, at a non-overlapping slot on
    # T1 that would otherwise be fine) -> WO1 assigned twice.
    sched["assignments"].append(
        {"wo": "WO1", "tech": "T1", "start_bh": 7.0, "end_bh": 11.0}
    )
    result = validate(inst, sched)
    assert result["feasible"] is False
    assert has_violation(result, "(a)"), result["violations"]
    assert any("WO1" in v and "duplicated" in v for v in result["violations"]), \
        result["violations"]
    print("PASS test_a_duplicated")


# ---------------------------------------------------------------------------
# Test 3 (b): eligibility -- assigned tech trade must match WO trade.
# ---------------------------------------------------------------------------

def test_b_trade_mismatch():
    inst = base_instance()
    sched = base_schedule()
    # WO1 is trade D20 but we assign it to T1 (trade E10). Move it to a free
    # slot on T1 so ONLY the trade rule is violated (no overlap on T1).
    for a in sched["assignments"]:
        if a["wo"] == "WO1":
            a["tech"] = "T1"
            a["start_bh"] = 7.0
            a["end_bh"] = 11.0
    result = validate(inst, sched)
    assert result["feasible"] is False
    assert has_violation(result, "(b)"), result["violations"]
    print("PASS test_b_trade_mismatch")


def test_b_unknown_tech():
    inst = base_instance()
    sched = base_schedule()
    for a in sched["assignments"]:
        if a["wo"] == "WO1":
            a["tech"] = "T999"  # no such technician
    result = validate(inst, sched)
    assert result["feasible"] is False
    assert has_violation(result, "(b)"), result["violations"]
    print("PASS test_b_unknown_tech")


# ---------------------------------------------------------------------------
# Test 4 (c): start_bh >= release_bh.
# ---------------------------------------------------------------------------

def test_c_release_violation():
    inst = base_instance()
    sched = base_schedule()
    # Give WO4 a release of 6.0, then start it at 5.0 (< 6.0 - 1e-9).
    # T0 is free from 5.0 onward (WO1 [0,4], WO2 [4,5]), so placing WO4 at
    # [5,6] on T0 isolates the release violation: duration stays 1.0 == p_bh
    # and there is no overlap.
    for wo in inst["work_orders"]:
        if wo["id"] == "WO4":
            wo["release_bh"] = 6.0
    for a in sched["assignments"]:
        if a["wo"] == "WO4":
            a["tech"] = "T0"
            a["start_bh"] = 5.0   # 5.0 < 6.0 - 1e-9 -> release violation
            a["end_bh"] = 6.0     # duration 1.0 == p_bh, no overlap on T0
    result = validate(inst, sched)
    assert result["feasible"] is False
    assert has_violation(result, "(c)"), result["violations"]
    # Make sure we isolated (c): no overlap/duration/eligibility noise.
    assert not has_violation(result, "(d)"), result["violations"]
    assert not has_violation(result, "(e)"), result["violations"]
    print("PASS test_c_release_violation")


# ---------------------------------------------------------------------------
# Test 5 (d): end_bh - start_bh == p_bh within 1e-6.
# ---------------------------------------------------------------------------

def test_d_duration_violation():
    inst = base_instance()
    sched = base_schedule()
    # WO1 has p_bh 4.0; make its slot last 4.5 -> duration mismatch of 0.5.
    for a in sched["assignments"]:
        if a["wo"] == "WO1":
            a["end_bh"] = 4.5   # 4.5 - 0.0 = 4.5 != 4.0
    result = validate(inst, sched)
    assert result["feasible"] is False
    assert has_violation(result, "(d)"), result["violations"]
    print("PASS test_d_duration_violation")


def test_d_within_tolerance_is_ok():
    inst = base_instance()
    sched = base_schedule()
    # A sub-1e-6 wobble must NOT trip (d).
    for a in sched["assignments"]:
        if a["wo"] == "WO1":
            a["end_bh"] = 4.0 + 5e-7
    result = validate(inst, sched)
    assert not has_violation(result, "(d)"), result["violations"]
    print("PASS test_d_within_tolerance_is_ok")


# ---------------------------------------------------------------------------
# Test 6 (e): per-technician overlap.
# ---------------------------------------------------------------------------

def test_e_overlap_violation():
    inst = base_instance()
    sched = base_schedule()
    # Make WO2 start before WO1 ends on T0: WO1 [0,4], WO2 [3,4] -> overlap.
    for a in sched["assignments"]:
        if a["wo"] == "WO2":
            a["start_bh"] = 3.0
            a["end_bh"] = 4.0   # duration 1.0 == p_bh so only (e) trips
    result = validate(inst, sched)
    assert result["feasible"] is False
    assert has_violation(result, "(e)"), result["violations"]
    print("PASS test_e_overlap_violation")


# ---------------------------------------------------------------------------
# Test 7 (f): schedule.instance_id must equal instance.meta.id.
# ---------------------------------------------------------------------------

def test_f_instance_id_mismatch():
    inst = base_instance()
    sched = base_schedule()
    sched["instance_id"] = "c05_test_WRONG"
    result = validate(inst, sched)
    assert result["feasible"] is False
    assert has_violation(result, "(f)"), result["violations"]
    print("PASS test_f_instance_id_mismatch")


# ---------------------------------------------------------------------------
# Bonus: metrics are computed even when infeasible.
# ---------------------------------------------------------------------------

def test_metrics_present_when_infeasible():
    inst = base_instance()
    sched = base_schedule()
    sched["instance_id"] = "c05_test_WRONG"  # forces infeasible via (f)
    result = validate(inst, sched)
    assert result["feasible"] is False
    # metrics still fully populated from the assignments given.
    assert approx(result["metrics"]["WWT"], 6.0), result["metrics"]
    assert approx(result["metrics"]["makespan"], 7.0), result["metrics"]
    print("PASS test_metrics_present_when_infeasible")


def test_absent_priority_class_is_null():
    # Instance with only P1 work orders -> P2/P3/P4 shares must be None.
    inst = {
        "meta": {"id": "c05_solo_0001"},
        "trades": ["D20"],
        "technicians": [{"id": "T0", "trade": "D20"}],
        "work_orders": [
            {"id": "WO1", "trade": "D20", "p_bh": 2.0, "release_bh": 0.0,
             "due_bh": 8.0, "priority": 1, "weight": 8.0},
        ],
    }
    sched = {
        "instance_id": "c05_solo_0001",
        "method": "edd", "seed": 1, "wall_seconds": 0.001, "decisions": 1,
        "assignments": [
            {"wo": "WO1", "tech": "T0", "start_bh": 0.0, "end_bh": 2.0},
        ],
    }
    result = validate(inst, sched)
    assert result["feasible"] is True, result["violations"]
    pp = result["metrics"]["per_priority_breach_share"]
    assert approx(pp[1], 0.0), pp     # 1 WO, not breached (2 <= 8)
    assert pp[2] is None, pp
    assert pp[3] is None, pp
    assert pp[4] is None, pp
    print("PASS test_absent_priority_class_is_null")


if __name__ == "__main__":
    test_feasible_and_metrics()
    test_a_missing()
    test_a_duplicated()
    test_b_trade_mismatch()
    test_b_unknown_tech()
    test_c_release_violation()
    test_d_duration_violation()
    test_d_within_tolerance_is_ok()
    test_e_overlap_violation()
    test_f_instance_id_mismatch()
    test_metrics_present_when_infeasible()
    test_absent_priority_class_is_null()
    print("ALL VALIDATOR TESTS PASSED")
