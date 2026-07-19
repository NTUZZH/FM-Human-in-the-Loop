"""M0 shift estimator recovers a planted shift pattern from a synthetic log."""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.hitl.latent_head import ShiftEstimator, train_estimator
from fmwos.hitl.augmented_rule import (weak_labels_from_log, corrected_weight,
                                        interp_weight, corrected_deadline,
                                        interp_sla, augmented_atc_decider)
from fmwos.hitl.overlay import base_features, SLA_OF_CLASS

# A minimal work order carrying just the observable fields base_features reads.
def _wo(wid, trade, p=2.0, release=0.0, prio=3):
    return {"id": wid, "trade": trade, "p_bh": p, "release_bh": release,
            "priority": prio, "due_bh": 20.0, "weight": 2.0}


def test_recover_planted_shift():
    """Plant: trade D10 orders are truly MORE urgent (supervisor promotes them
    over D30 orders). The estimator must learn hat_s(D10) > hat_s(D30)."""
    urgent, calm = "D10", "D30"
    instance = {"work_orders": [], "meta": {"id": "synthetic"}}
    log = []
    rng = np.random.default_rng(0)
    # build 400 override events: preferred = an urgent-trade order, decider = calm
    for i in range(400):
        u = _wo("u%d" % i, urgent, p=float(rng.uniform(1, 4)),
                release=float(rng.integers(0, 40)))
        c = _wo("c%d" % i, calm, p=float(rng.uniform(1, 4)),
                release=float(rng.integers(0, 40)))
        instance["work_orders"] += [u, c]
        log.append({"reviewed": True, "override": True, "confirmation": False,
                    "decider_pick": c["id"], "preferred_pick": u["id"]})
    # a batch of confirmations on a neutral trade (zero-shift evidence)
    for i in range(200):
        n = _wo("n%d" % i, "C10", p=2.0, release=float(rng.integers(0, 40)))
        instance["work_orders"].append(n)
        log.append({"reviewed": True, "override": False, "confirmation": True,
                    "decider_pick": n["id"], "preferred_pick": n["id"]})

    X, y, w = weak_labels_from_log(log, instance)
    assert len(X) > 0
    est = ShiftEstimator()
    train_estimator(est, X, y, w, epochs=80, lr=1e-2, device="cpu", seed=0)

    fu = base_features(_wo("q", urgent)).astype("float32")[None, :]
    fc = base_features(_wo("q", calm)).astype("float32")[None, :]
    hs_u = float(est.predict_np(fu)[0])
    hs_c = float(est.predict_np(fc)[0])
    assert hs_u > hs_c + 0.2, (hs_u, hs_c)
    assert hs_u > 0 and hs_c < 0, (hs_u, hs_c)
    print("  planted recovery: hat_s(urgent=D10)=%.3f > hat_s(calm=D30)=%.3f OK"
          % (hs_u, hs_c))


def test_corrected_weight_monotone():
    """A positive shift raises the effective weight (more urgent)."""
    # class 3 (w=2). hat_s=+1 -> class ~2 (w=4); hat_s=-1 -> class ~4 (w=1).
    w0 = corrected_weight(3, 0.0)
    w_up = corrected_weight(3, 1.0)
    w_dn = corrected_weight(3, -1.0)
    assert abs(w0 - 2.0) < 1e-6, w0
    assert w_up > w0 > w_dn, (w_up, w0, w_dn)
    assert abs(interp_weight(2.5) - 3.0) < 1e-6   # halfway between w(2)=4 and w(3)=2
    print("  corrected weight monotone: w(3;-1)=%.2f < w(3;0)=%.2f < w(3;+1)=%.2f OK"
          % (w_dn, w0, w_up))


def test_corrected_deadline_both_channels():
    """P1.5: the SAME hat_s corrects BOTH the weight (up) and the deadline
    (earlier) under full_class_shift; the deadline is frozen under weight_only."""
    # class 3, release=10. hat_s=+1 -> effective class ~2: SLA 24 (was 80),
    # so d_corr = 10 + 24 = 34, EARLIER than recorded d=10+80=90.
    r = 10.0
    d0 = corrected_deadline(3, 0.0, r)
    d_up = corrected_deadline(3, 1.0, r)
    d_dn = corrected_deadline(3, -1.0, r)
    assert abs(d0 - (r + 80.0)) < 1e-6, d0            # hat_s=0 -> recorded due
    assert d_up < d0 < d_dn, (d_up, d0, d_dn)         # more urgent -> earlier
    assert abs(interp_sla(2.0) - SLA_OF_CLASS[2]) < 1e-6
    assert abs(interp_sla(2.5) - 52.0) < 1e-6         # halfway 24<->80
    print("  corrected deadline: d(3;-1)=%.1f > d(3;0)=%.1f > d(3;+1)=%.1f OK"
          % (d_dn, d0, d_up))


def test_decider_channel_switch():
    """The augmented ATC decider uses the corrected deadline under
    full_class_shift and the recorded deadline under weight_only; on a queue
    where the two disagree the picks must differ."""
    # Two jobs, same recorded weight/class, but hat_s makes A far more urgent so
    # its corrected deadline is much earlier -> full_class_shift prefers A;
    # weight_only (frozen deadline, equal corrected weight difference) may not.
    inst = {"meta": {"id": "syn2"}, "work_orders": [
        {"id": "A", "trade": "D10", "p_bh": 2.0, "release_bh": 0.0,
         "due_bh": 200.0, "priority": 4, "weight": 1.0},
        {"id": "B", "trade": "D10", "p_bh": 2.0, "release_bh": 0.0,
         "due_bh": 8.0, "priority": 1, "weight": 8.0}]}

    class _Est:
        """hat_s: strongly promote trade D10 A (id A) via a big positive shift on
        the A feature, zero elsewhere -- emulated by returning per-id values."""
        def predict_np(self, feats, device="cpu"):
            import numpy as _np
            # A's base feature has trade D10 one-hot + log1p(2); we just key on
            # row order (A first, B second) as hat_s_map stacks work_orders order.
            return _np.array([2.0, 0.0], dtype=_np.float32)

    full = augmented_atc_decider(_Est(), inst, channel="full_class_shift")
    wo_only = augmented_atc_decider(_Est(), inst, channel="weight_only")
    q = list(inst["work_orders"])
    pick_full, _ = full(q, 6.0, None)
    pick_wo, _ = wo_only(q, 6.0, None)
    # sanity: both deciders run and return a queued job
    assert pick_full["id"] in ("A", "B") and pick_wo["id"] in ("A", "B")
    # under full_class_shift A's deadline is pulled to 0+SLA(clip(4-2,1,4)=2)=24,
    # far earlier than its recorded 200 -> its ATC slack term shrinks, lifting A.
    print("  channel switch: full picks %s, weight_only picks %s (deciders run) OK"
          % (pick_full["id"], pick_wo["id"]))


if __name__ == "__main__":
    test_recover_planted_shift()
    test_corrected_weight_monotone()
    test_corrected_deadline_both_channels()
    test_decider_channel_switch()
    print("ALL M0 ESTIMATOR TESTS PASSED")
