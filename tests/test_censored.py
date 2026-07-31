"""W3 tests for fmwos.hitl.censored.

The five that carry the manuscript's claims:
  * the censored fit REDUCES to the shipped weighted-squared-error fit,
    bit-for-bit, when nothing is censored -- so any difference the package
    reports is the likelihood and nothing else;
  * the weak labels are the shipped weak labels, byte for byte, plus the
    recorded class;
  * the censoring rule censors exactly the labels the priority scale cannot
    express, and nothing else;
  * the plug-in deployment path is the shipped decider, decision for decision;
  * the estimator can only ever see OBSERVABLE fields (enforced by recording
    every key the feature path touches, not by promise), and the parameter
    count matches the incumbent exactly.
"""

import glob
import json
import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.env import DispatchEnv                                 # noqa: E402
from fmwos.hitl import augmented_rule as AR                       # noqa: E402
from fmwos.hitl import censored as CN                             # noqa: E402
from fmwos.hitl import deciders as dec                            # noqa: E402
from fmwos.hitl import overlay as ov                              # noqa: E402
from fmwos.hitl.latent_head import (LAT_DIM, ShiftEstimator,      # noqa: E402
                                    train_estimator)
from fmwos.hitl.supervisor import Supervisor                      # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")


def _an_instance():
    pats = [os.path.join(_INST, "c09", "replay", "150", "*.json"),
            os.path.join(_INST, "c09", "storm2", "w80", "*u100_*.json")]
    for p in pats:
        f = sorted(glob.glob(p))
        if f:
            with open(f[0]) as fh:
                return json.load(fh)
    pytest.skip("no instance files available")


def _overlay(beta=1.0):
    return ov.Overlay(ov.OverlayParams(beta=beta, family="F-NL",
                                       master_seed=12345,
                                       channel="full_class_shift"))


def _a_log(inst, overlay, seed=301):
    sup = Supervisor(overlay, inst, rho=0.25, epsilon=0.0, theta=1.0,
                     mechanism="targeted", seed=seed, applied=overlay.apply(inst))
    _sched, log = dec.run_rule_sup(DispatchEnv(inst), "atc", sup, seed=seed)
    return log


# --------------------------------------------------------------------------- #
# 1. Censoring bounds and codes                                               #
# --------------------------------------------------------------------------- #
def test_censoring_bounds_are_the_expressible_range_of_the_effective_shift():
    for c in (1, 2, 3, 4):
        lo, hi = CN.censoring_bounds(c)
        # brute force: t = c - clip(c - s, 1, 4) over every reachable shift
        ts = [c - min(4, max(1, c - s)) for s in range(-6, 7)]
        assert float(lo) == min(ts)
        assert float(hi) == max(ts)


def test_strict_mode_censors_exactly_the_boundary_labels():
    c = np.array([1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.])
    y = np.array([1., 0., -1., 1., 0., -1., 1., 0., -1., 1., 0., -1.])
    code, lo, hi = CN.censor_codes(y, c, mode="strict")
    # class 1: U = 0, so +1 and 0 are right-censored, -1 is a point label.
    # class 4: L = 0, so -1 and 0 are left-censored, +1 is a point label.
    # class 2: U = +1 so a promotion is right-censored; class 3: L = -1 so a
    # demotion is left-censored.
    assert list(code) == [+1, +1, 0, +1, 0, 0, 0, 0, -1, 0, -1, -1]
    assert list(lo) == [-3, -3, -3, -2, -2, -2, -1, -1, -1, 0, 0, 0]
    assert list(hi) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_impossible_mode_censors_only_the_two_impossible_labels():
    c = np.array([1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.])
    y = np.array([1., 0., -1., 1., 0., -1., 1., 0., -1., 1., 0., -1.])
    code, _lo, _hi = CN.censor_codes(y, c, mode="impossible")
    # only (c=1, y=+1) "more urgent than the most urgent class" and
    # (c=4, y=-1) "less urgent than the least urgent class" are impossible.
    assert list(code) == [+1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]


def test_none_mode_censors_nothing():
    c = np.array([1., 2., 3., 4.])
    y = np.array([1., 1., -1., -1.])
    code, _lo, _hi = CN.censor_codes(y, c, mode="none")
    assert not code.any()


# --------------------------------------------------------------------------- #
# 2. Weak labels: the shipped labels plus the recorded class                  #
# --------------------------------------------------------------------------- #
def test_weak_labels_are_bit_identical_to_the_shipped_constructor():
    inst, overlay = _an_instance(), _overlay()
    log = _a_log(inst, overlay)
    X0, y0, w0 = AR.weak_labels_from_log(log, inst)
    X1, y1, w1, c1 = CN.weak_labels_with_class(log, inst)
    assert np.array_equal(X0, X1)
    assert np.array_equal(y0, y1)
    assert np.array_equal(w0, w1)
    assert c1.shape == y1.shape
    assert set(np.unique(c1)) <= {1.0, 2.0, 3.0, 4.0}


def test_weak_label_class_column_is_the_recorded_priority():
    inst, overlay = _an_instance(), _overlay()
    log = _a_log(inst, overlay)
    prio = {w["id"]: float(w["priority"]) for w in inst["work_orders"]}
    _X, y, _w, c = CN.weak_labels_with_class(log, inst)
    # rebuild the expected class stream in the same order the constructor walks
    exp = []
    for e in log:
        if not e.get("reviewed"):
            continue
        di = e["decider_pick"]
        pos = e.get("executed_pick", e.get("preferred_pick"))
        if e["override"]:
            if pos is not None:
                exp.append(prio[pos])
            exp.append(prio[di])
        elif e.get("confirmation"):
            exp.append(prio[di])
    assert list(c) == exp
    assert len(exp) == len(y)


# --------------------------------------------------------------------------- #
# 3. The reduction to the incumbent (the controlled-comparison licence)       #
# --------------------------------------------------------------------------- #
def test_censored_objective_equals_the_shipped_squared_error_when_uncensored():
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(256, generator=g)
    y = torch.randint(-1, 2, (256,), generator=g).float()
    code = torch.zeros(256, dtype=torch.int64)
    lo = torch.full((256,), -3.0)
    hi = torch.full((256,), 3.0)
    per = CN.censored_terms(mu, y, code, lo, hi, torch.tensor(1.0))
    assert torch.equal(per, (mu - y) ** 2)


def test_none_mode_training_reproduces_the_shipped_fit_bit_for_bit():
    inst, overlay = _an_instance(), _overlay()
    log = _a_log(inst, overlay)
    X, y, w, c = CN.weak_labels_with_class(log, inst)
    assert len(X) > 0

    torch.manual_seed(7)
    ref = ShiftEstimator(hidden=32)
    torch.manual_seed(7)
    mine = CN.CensoredShiftEstimator(hidden=32, sigma=1.0, learn_sigma=False)
    # same seed => the imported core initialises to the incumbent's weights
    for a, b in zip(ref.state_dict().values(), mine.core.state_dict().values()):
        assert torch.equal(a, b)

    l0 = train_estimator(ref, X, y, w, seed=11)
    l1 = CN.train_censored_estimator(mine, X, y, w, c, mode="none", seed=11)
    assert l0 == l1
    for k in ref.state_dict():
        assert torch.equal(ref.state_dict()[k], mine.core.state_dict()[k]), k


def test_censored_gradients_are_finite_in_every_mode():
    inst, overlay = _an_instance(), _overlay()
    log = _a_log(inst, overlay)
    X, y, w, c = CN.weak_labels_with_class(log, inst)
    for mode in CN.CENSOR_MODES:
        for learn in (False, True):
            torch.manual_seed(3)
            m = CN.CensoredShiftEstimator(hidden=32, learn_sigma=learn)
            loss = CN.train_censored_estimator(m, X, y, w, c, mode=mode,
                                               epochs=2, seed=5)
            assert np.isfinite(loss)
            for p in m.parameters():
                assert torch.isfinite(p).all()


# --------------------------------------------------------------------------- #
# 4. Capacity and inputs                                                      #
# --------------------------------------------------------------------------- #
def test_parameter_count_matches_the_incumbent():
    ref = sum(p.numel() for p in ShiftEstimator(hidden=32).parameters())
    fixed = CN.CensoredShiftEstimator(hidden=32, learn_sigma=False)
    fitted = CN.CensoredShiftEstimator(hidden=32, learn_sigma=True)
    assert ref == 1761
    assert fixed.n_params_estimator() == ref
    assert fixed.n_params() == ref                 # sigma is a buffer, not a parameter
    assert fitted.n_params_estimator() == ref
    assert fitted.n_params() == ref + 1            # the one fitted scale


class _SpyDict(dict):
    """A dict that records every key read from it."""

    def __init__(self, d, seen):
        super().__init__(d)
        self._seen = seen

    def __getitem__(self, k):
        self._seen.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self._seen.add(k)
        return super().get(k, default)


def test_feature_path_reads_only_observable_fields():
    inst = _an_instance()
    seen = set()
    spied = {"meta": inst["meta"],
             "work_orders": [_SpyDict(w, seen) for w in inst["work_orders"]],
             "technicians": inst["technicians"]}
    torch.manual_seed(0)
    m = CN.CensoredShiftEstimator(hidden=32)
    CN.deployed_hat_s_map(m, spied, deploy="expected")
    allowed = {"id", "trade", "p_bh", "release_bh", "priority"}
    assert seen <= allowed, "feature path read non-observable fields: %r" % (seen - allowed)
    latent = {"s", "shift", "c_star", "w_star", "d_star", "xi", "f", "latent"}
    assert not (seen & latent)


# --------------------------------------------------------------------------- #
# 5. Deployment                                                               #
# --------------------------------------------------------------------------- #
def test_plugin_decider_is_the_shipped_decider():
    inst, overlay = _an_instance(), _overlay()
    torch.manual_seed(21)
    m = CN.CensoredShiftEstimator(hidden=32)
    with torch.no_grad():                       # give it a non-trivial output
        m.core.out.bias.fill_(0.3)
    ship = AR.augmented_atc_decider(m.core, inst, channel="full_class_shift")
    mine = CN.censored_atc_decider(m, inst, deploy="plugin",
                                   channel="full_class_shift")
    s0, _ = DispatchEnv(inst).run_supervised(ship, supervisor=None, method="m0", seed=301)
    s1, _ = DispatchEnv(inst).run_supervised(mine, supervisor=None, method="m0", seed=301)
    assert s0["assignments"] == s1["assignments"]


def test_expected_shift_matches_monte_carlo_and_the_zero_noise_limit():
    rng = np.random.default_rng(0)
    mu = rng.normal(0.0, 1.0, size=200)
    for c in (1.0, 2.0, 3.0, 4.0):
        lo, hi = CN.censoring_bounds(np.full(200, c))
        for sigma in (0.25, 1.0, 2.0):
            got = CN.expected_effective_shift(mu, sigma, lo, hi)
            draws = mu[:, None] + sigma * rng.standard_normal((200, 40000))
            mc = np.clip(draws, lo[:, None], hi[:, None]).mean(axis=1)
            assert np.max(np.abs(got - mc)) < 0.02
        tiny = CN.expected_effective_shift(mu, 1e-6, lo, hi)
        assert np.max(np.abs(tiny - np.clip(mu, lo, hi))) < 1e-4


def test_expected_shift_never_leaves_the_expressible_range():
    mu = np.linspace(-6.0, 6.0, 101)
    for c in (1.0, 2.0, 3.0, 4.0):
        lo, hi = CN.censoring_bounds(np.full(101, c))
        t = CN.expected_effective_shift(mu, 1.0, lo, hi)
        assert np.all(t >= lo - 1e-9) and np.all(t <= hi + 1e-9)


def test_applied_shift_is_what_the_corrected_class_receives():
    inst = _an_instance()
    hs = {w["id"]: 0.9 for w in inst["work_orders"]}
    ids, ap = CN.applied_shift(hs, inst)
    prio = {w["id"]: int(w["priority"]) for w in inst["work_orders"]}
    for wid, a in zip(ids, ap):
        c = prio[wid]
        # the corrected class the decider computes, back-solved for the shift
        assert abs((c - min(4.0, max(1.0, c - 0.9))) - a) < 1e-12
