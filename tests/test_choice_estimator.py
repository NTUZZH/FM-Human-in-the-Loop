"""W2 tests for fmwos.hitl.choice_estimator.

The four that matter for the manuscript's claims:
  * the queue-logging supervisor changes NOTHING about the supervisor's
    behaviour, it only widens the log;
  * the Deep-Sets queue encoder is PERMUTATION-INVARIANT;
  * the queue encoder can only ever see OBSERVABLE fields (enforced by
    recording every key the feature constructor touches, not by promise);
  * the choice utilities and the queue-conditioned decider are the SAME
    deployed ATC index the shipped augmented rule uses, so a likelihood fitted
    on them is a likelihood on the thing the dispatcher runs.
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

from fmwos import pdrs                                            # noqa: E402
from fmwos.env import DispatchEnv                                 # noqa: E402
from fmwos.hitl import augmented_rule as AR                       # noqa: E402
from fmwos.hitl import choice_estimator as CE                     # noqa: E402
from fmwos.hitl import deciders as dec                            # noqa: E402
from fmwos.hitl import overlay as ov                              # noqa: E402
from fmwos.hitl.latent_head import LAT_DIM, ShiftEstimator        # noqa: E402
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


# --------------------------------------------------------------------------- #
# 1. The widened log: behaviour identical, feasible set now recoverable       #
# --------------------------------------------------------------------------- #
def test_queue_logging_supervisor_is_behaviourally_identical():
    inst, overlay = _an_instance(), _overlay()
    applied = overlay.apply(inst)
    out = {}
    for name, cls in (("base", Supervisor), ("wrapped", CE.QueueLoggingSupervisor)):
        sup = cls(overlay, inst, rho=0.25, epsilon=0.1, theta=1.0,
                  mechanism="targeted", seed=301, applied=applied)
        sched, log = dec.run_rule_sup(DispatchEnv(inst), "atc", sup, seed=301)
        out[name] = (sched, log, sup.summary())
    base_sched, base_log, base_sum = out["base"]
    wrap_sched, wrap_log, wrap_sum = out["wrapped"]
    assert base_sum == wrap_sum
    assert base_sched["assignments"] == wrap_sched["assignments"]
    assert len(base_log) == len(wrap_log)
    extra = {"cand_ids", "now"}
    for a, b in zip(base_log, wrap_log):
        assert set(b) - set(a) <= extra
        for k in a:
            assert a[k] == b[k], k


def test_feasible_set_is_recoverable_only_with_the_wrapper():
    inst, overlay = _an_instance(), _overlay()
    sup = Supervisor(overlay, inst, rho=0.25, theta=1.0, seed=301)
    _s, log = dec.run_rule_sup(DispatchEnv(inst), "atc", sup, seed=301)
    reviewed = [e for e in log if e["reviewed"]]
    assert reviewed, "expected at least one reviewed decision"
    assert "cand_ids" not in reviewed[0], (
        "shipped log unexpectedly carries the feasible set")
    ds = CE.ChoiceDataset()
    with pytest.raises(KeyError):
        ds.add_log(log, inst)

    sup2 = CE.QueueLoggingSupervisor(overlay, inst, rho=0.25, theta=1.0, seed=301)
    _s2, log2 = dec.run_rule_sup(DispatchEnv(inst), "atc", sup2, seed=301)
    ds2 = CE.ChoiceDataset()
    ds2.add_log(log2, inst)
    c = ds2.counts()
    assert c["n_decisions"] > 0 and c["mean_queue"] >= 2.0


# --------------------------------------------------------------------------- #
# 2. Permutation invariance of the queue encoder                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [2, 7, 33, 64])
def test_queue_encoder_is_permutation_invariant(k):
    torch.manual_seed(0)
    est = CE.QueueShiftEstimator(use_queue=True)
    feats = torch.randn(3, k, LAT_DIM)
    mask = torch.ones(3, k, dtype=torch.bool)
    with torch.no_grad():
        ref = est(feats, mask)
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(20):
        perm = torch.as_tensor(rng.permutation(k))
        with torch.no_grad():
            got = est(feats[:, perm], mask[:, perm])
        worst = max(worst, float((got - ref[:, perm]).abs().max()))
    assert worst < 1e-5, "queue encoder is not permutation-invariant (%g)" % worst


def test_queue_encoder_is_permutation_invariant_with_padding():
    torch.manual_seed(0)
    est = CE.QueueShiftEstimator(use_queue=True)
    k, kk = 40, 25                       # 25 real candidates padded to 40
    feats = torch.randn(2, k, LAT_DIM)
    mask = torch.zeros(2, k, dtype=torch.bool)
    mask[:, :kk] = True
    feats[:, kk:] = 0.0
    with torch.no_grad():
        ref = est(feats, mask)
    rng = np.random.default_rng(3)
    for _ in range(20):
        perm = np.concatenate([rng.permutation(kk), np.arange(kk, k)])
        p = torch.as_tensor(perm)
        with torch.no_grad():
            got = est(feats[:, p], mask[:, p])
        assert torch.allclose(got[:, :kk], ref[:, perm[:kk]], atol=1e-5)


def test_per_order_estimator_ignores_the_queue():
    torch.manual_seed(0)
    est = CE.QueueShiftEstimator(use_queue=False)
    feats = torch.randn(2, 9, LAT_DIM)
    with torch.no_grad():
        a = est(feats, torch.ones(2, 9, dtype=torch.bool))
        b = est(feats, torch.zeros(2, 9, dtype=torch.bool))
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# 3. Parameter counts (a variant that silently grew is not a controlled test) #
# --------------------------------------------------------------------------- #
def test_parameter_counts_are_exact():
    shipped = ShiftEstimator(lat_dim=LAT_DIM, hidden=32)
    n_shipped = sum(p.numel() for p in shipped.parameters())
    assert n_shipped == 1761                       # 20*32+32 + 32*32+32 + 32+1

    per_order = CE.ChoiceModel(use_queue=False)
    queued = CE.ChoiceModel(use_queue=True)
    assert per_order.n_params_estimator() == 1761
    assert per_order.n_params() == 1762            # + log_tau
    assert queued.n_params_estimator() == 3041     # 60*32+32 + 32*32+32 + 32+1
    assert queued.n_params() == 3042
    # the ONLY growth is the first layer's input width; depth/width unchanged.
    assert queued.est.core.fc2.weight.shape == per_order.est.core.fc2.weight.shape
    assert queued.est.core.out.weight.shape == per_order.est.core.out.weight.shape
    assert (queued.n_params_estimator() - per_order.n_params_estimator()
            == 2 * LAT_DIM * 32)


def test_per_order_variant_is_the_shipped_architecture():
    torch.manual_seed(11)
    a = ShiftEstimator(lat_dim=LAT_DIM, hidden=32)
    torch.manual_seed(11)
    b = CE.ChoiceModel(use_queue=False)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(),
                                  b.est.core.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)


# --------------------------------------------------------------------------- #
# 4. No latent quantity can reach the estimator                              #
# --------------------------------------------------------------------------- #
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


def test_instance_tables_read_only_observable_fields():
    inst = _an_instance()
    seen = set()
    spied = {"meta": _SpyDict(inst["meta"], seen),
             "work_orders": [_SpyDict(w, seen) for w in inst["work_orders"]],
             "technicians": inst["technicians"]}
    tab = CE.instance_tables(spied)
    allowed = {"id", "trade", "p_bh", "release_bh", "priority", "due_bh", "weight"}
    assert seen <= allowed, "feature path read non-observable fields: %r" % (seen - allowed)
    latent = {"s", "shift", "c_star", "w_star", "d_star", "xi", "f", "latent"}
    assert not (seen & latent)
    assert tab.feats.shape == (len(inst["work_orders"]), LAT_DIM)


def test_estimator_features_equal_the_shipped_feature_constructor():
    inst = _an_instance()
    tab = CE.instance_tables(inst)
    ref = np.stack([ov.base_features(w) for w in inst["work_orders"]]).astype(np.float32)
    assert np.array_equal(tab.feats, ref)


# --------------------------------------------------------------------------- #
# 5. The choice utilities ARE the deployed index                             #
# --------------------------------------------------------------------------- #
def test_torch_class_curves_match_the_shipped_numpy_curves():
    cs = np.linspace(0.0, 5.0, 501)
    t = torch.as_tensor(cs, dtype=torch.float64)
    gw = CE.interp_weight_t(t).numpy()
    gs = CE.interp_sla_t(t).numpy()
    for c, w, s in zip(cs, gw, gs):
        assert abs(w - AR.interp_weight(c)) < 1e-9
        assert abs(s - AR.interp_sla(c)) < 1e-9


def test_corrected_utilities_reproduce_the_deployed_atc_index():
    inst = _an_instance()
    tab = CE.instance_tables(inst)
    rng = np.random.default_rng(0)
    rows = rng.choice(len(tab.ids), size=12, replace=False).astype(np.int32)
    now = 17.0
    hat = rng.normal(0.0, 0.8, size=12).astype(np.float32)

    T = torch.as_tensor
    u = CE.corrected_utilities(
        T(hat).unsqueeze(0), T(tab.prio[rows]).unsqueeze(0),
        T(tab.p[rows]).unsqueeze(0), T(tab.rel[rows]).unsqueeze(0),
        T(tab.due_rec[rows]).unsqueeze(0), T(np.float32(now)).reshape(1),
        torch.ones(1, 12, dtype=torch.bool))[0].numpy()

    p = tab.p[rows].astype(np.float64)
    pbar = p.mean()
    ref = []
    for i, r in enumerate(rows):
        c_hat = float(np.clip(tab.prio[r] - hat[i], 1.0, 4.0))
        w = AR.interp_weight(c_hat)
        d = tab.rel[r] + AR.interp_sla(c_hat)
        slack = max(0.0, d - now - p[i])
        ref.append(np.log((w / p[i]) * np.exp(-slack / (2.0 * pbar))))
    assert np.allclose(u, np.asarray(ref), atol=1e-4)


def test_queue_conditioned_decider_matches_the_shipped_decider_without_queue():
    """A per-order model deployed through the W2 decider must pick exactly what
    the shipped augmented_atc_decider picks: the deployment path is not new."""
    inst = _an_instance()
    torch.manual_seed(5)
    model = CE.ChoiceModel(use_queue=False)
    with torch.no_grad():                       # spread hat_s so ties are rare
        model.est.core.out.weight.mul_(6.0); model.est.core.out.bias.add_(0.3)
    shipped = AR.augmented_atc_decider(model.est.core, inst, channel="full_class_shift")
    mine = CE.queue_conditioned_atc_decider(model, inst, channel="full_class_shift")

    import random
    rng = random.Random(0)
    env = DispatchEnv(inst)
    picks_a, picks_b = [], []

    def _both(queue, t, r):
        a, ma = shipped(queue, t, r)
        b, mb = mine(queue, t, r)
        picks_a.append(a["id"]); picks_b.append(b["id"])
        assert abs(ma - mb) < 1e-6 * max(1.0, abs(ma))
        return a, ma

    env.run_supervised(_both, supervisor=None, method="x", seed=0)
    assert picks_a and picks_a == picks_b
    del rng


# --------------------------------------------------------------------------- #
# 6. Dataset hygiene                                                          #
# --------------------------------------------------------------------------- #
def test_split_by_instance_is_leak_free():
    inst, overlay = _an_instance(), _overlay()
    ds = CE.ChoiceDataset()
    sup = CE.QueueLoggingSupervisor(overlay, inst, rho=0.25, theta=1.0, seed=301)
    _s, log = dec.run_rule_sup(DispatchEnv(inst), "atc", sup, seed=301)
    ds.add_log(log, inst)
    tr, va = ds.split_by_instance({inst["meta"]["id"]})
    assert len(tr) == 0 and len(va) == len(ds)
    tr2, va2 = ds.split_by_instance(set())
    assert len(tr2) == len(ds) and len(va2) == 0


def test_choice_set_truncation_keeps_both_picks_and_ignores_order():
    inst, overlay = _an_instance(), _overlay()
    ds_full, ds_cap = CE.ChoiceDataset(), CE.ChoiceDataset()
    sup = CE.QueueLoggingSupervisor(overlay, inst, rho=0.5, theta=1.0, seed=301)
    _s, log = dec.run_rule_sup(DispatchEnv(inst), "atc", sup, seed=301)
    ds_full.add_log(log, inst, k_max=CE.K_MAX)
    ds_cap.add_log(log, inst, k_max=8)
    assert len(ds_full) == len(ds_cap)
    for i in range(len(ds_cap)):
        assert len(ds_cap.cand[i]) <= 8
        assert 0 <= ds_cap.chosen[i] < len(ds_cap.cand[i])
        assert 0 <= ds_cap.decider[i] < len(ds_cap.cand[i])
        tab = ds_cap.tables[ds_cap.inst[i]]
        kept = set(ds_cap.cand[i].tolist())
        assert ds_full.cand[i][ds_full.chosen[i]] in kept
        assert ds_full.cand[i][ds_full.decider[i]] in kept
        del tab


def test_truncation_keeps_both_picks_when_both_fall_outside_the_cap():
    """Regression: with BOTH picks outside the top-k, an implementation that
    writes each required pick into the tail of the kept list loses the first.
    It fired on ~1 in 6,000 reviewed decisions of the headline cell."""
    inst = _an_instance()
    tab = CE.instance_tables(inst)
    n = len(tab.ids)
    rows = np.arange(min(40, n), dtype=np.int32)
    now = 0.0
    p = tab.p[rows]
    pbar = float(p.mean())
    slack = np.maximum(0.0, tab.due_rec[rows] - now - p)
    sc = (tab.w_rec[rows] / p) * np.exp(-slack / max(2.0 * pbar, 1e-6))
    worst = np.lexsort((rows, -sc))[-2:]            # the two LOWEST-index rivals
    ci, di = int(worst[0]), int(worst[1])
    new, nci, ndi = CE._truncate_choice_set(tab, rows, ci, di, now, 5)
    assert len(new) == 5
    assert new[nci] == rows[ci] and new[ndi] == rows[di]
    assert nci != ndi
    assert len(set(new.tolist())) == len(new)


def test_choice_logprob_is_a_proper_log_probability():
    inst, overlay = _an_instance(), _overlay()
    ds = CE.ChoiceDataset()
    sup = CE.QueueLoggingSupervisor(overlay, inst, rho=0.25, theta=1.0, seed=301)
    _s, log = dec.run_rule_sup(DispatchEnv(inst), "atc", sup, seed=301)
    ds.add_log(log, inst)
    torch.manual_seed(0)
    model = CE.ChoiceModel(use_queue=False)
    CE.init_temperature(model, ds)
    for b in ds.batches(batch_size=64):
        lp = CE.choice_logprob(model, b)
        assert torch.all(lp <= 1e-6) and torch.all(torch.isfinite(lp))
        floor = -torch.log(b["mask"].sum(1).to(lp.dtype))
        assert torch.all(lp >= floor * 60.0)      # never absurdly below uniform
        break
