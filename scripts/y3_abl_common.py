#!/usr/bin/env python
"""Shared M0 machinery for the Paper Y3 Phase-5 CHEAP ablations (E5/E4).

Everything here REUSES the locked hitl package building blocks verbatim
(overlay, supervisor, augmented_rule, true_objective, deciders); it only adds
two capabilities the locked ``augmented_rule.run_m0`` does not expose, without
editing any locked file:

  1. an independent DECIDER-CORRECTION channel for the augmented rule, so M0's
     recovered class-shift can be applied to the weight only, the deadline only,
     or both (full_class_shift). The overlay / supervisor / true objective stay
     full_class_shift throughout, so the private information and the scoring
     objective are unchanged; only how M0 USES its estimate varies (E5-channel).

  2. a per-iteration held-out evaluation callback, so the M0 true-TWT* and the
     estimator accuracy can be logged against the cumulative override count
     across the 8 DAgger iterations (E4 recovery curve).

``replicate_m0`` mirrors ``augmented_rule.run_m0`` step for step (same estimator
construction, same permutation RNG, same never-reset aggregate, same
``train_estimator(seed=seed+it)``, same ``probe_shift_accuracy``). With
decider_channel="full_class_shift" and mechanism="targeted" it reproduces
``run_m0`` bit-for-bit (verified against the results/y3_p4 cache); the per-iter
eval callback only runs eval-mode forward passes, which consume no RNG.
"""

from __future__ import annotations

import math
import os

import numpy as np

import torch

from fmwos.env import DispatchEnv
from fmwos.hitl import deciders as dec
from fmwos.hitl import overlay as ov
from fmwos.hitl.supervisor import Supervisor
from fmwos.hitl import augmented_rule as AR
from fmwos.hitl import true_objective as TO
from fmwos.hitl.latent_head import ShiftEstimator, train_estimator, LAT_DIM
from fmwos import pdrs

try:
    from scipy.stats import wilcoxon as _wilcoxon
    _HAVE_SCIPY = True
except Exception:                                                # pragma: no cover
    _HAVE_SCIPY = False

_TOL = 1e-9
DECIDER_CHANNELS = ("full_class_shift", "weight_only", "deadline_only")


# --------------------------------------------------------------------------- #
# Decider factory: apply hat_s to weight only / deadline only / both           #
# --------------------------------------------------------------------------- #
def _deadline_only_decider(estimator, instance, device="cpu", k=AR._ATC_K):
    """Augmented ATC that corrects the DEADLINE only (weight frozen at the
    recorded class weight). Mirrors ``augmented_rule.augmented_atc_decider`` but
    scores with the RECORDED weight w(c) and the corrected deadline
    r + SLA(clip(c - hat_s, 1, 4)). Isolates the deadline half of the lever."""
    hs = AR.hat_s_map(estimator, instance, device=device)
    prio = {w["id"]: int(w["priority"]) for w in instance["work_orders"]}
    p_of = {w["id"]: float(w["p_bh"]) for w in instance["work_orders"]}
    rel = {w["id"]: float(w["release_bh"]) for w in instance["work_orders"]}
    wrec = {w["id"]: float(w["weight"]) for w in instance["work_orders"]}

    def _decider(queue, t, rng):
        pbar = sum(p_of[j["id"]] for j in queue) / len(queue)
        denom = k * pbar
        scores = []
        for j in queue:
            wid = j["id"]
            dcorr = AR.corrected_deadline(prio[wid], hs.get(wid, 0.0), rel[wid])
            slack = max(0.0, dcorr - t - p_of[wid])
            s = (wrec[wid] / p_of[wid]) * math.exp(-slack / denom)
            scores.append((s, wid, j))
        scores.sort(key=lambda x: (-x[0], x[1]))
        best = scores[0][2]
        margin = (scores[0][0] - scores[1][0]) if len(scores) >= 2 else pdrs._BIG_MARGIN
        return best, float(margin)

    return _decider


def make_decider(estimator, instance, decider_channel, device="cpu"):
    if decider_channel in ("full_class_shift", "weight_only"):
        return AR.augmented_atc_decider(estimator, instance, device=device,
                                        channel=decider_channel)
    if decider_channel == "deadline_only":
        return _deadline_only_decider(estimator, instance, device=device)
    raise ValueError("decider_channel must be one of %r" % (DECIDER_CHANNELS,))


# --------------------------------------------------------------------------- #
# M0 pipeline replica (adds decider_channel + per-iter held-out eval)          #
# --------------------------------------------------------------------------- #
def replicate_m0(train_instances, probe_instances, overlay, *, beta, rho, eps,
                 decider_channel="full_class_shift", outer_iters=8,
                 mechanism="targeted", theta=1.0, override_weight=5.0,
                 confirm_weight=1.0, est_hidden=32, seed=0, device="cpu",
                 eval_insts=None):
    """Run the M0 DAgger pipeline; return {estimator, per_iter}.

    per_iter rows carry the run_m0 fields PLUS cum_overrides and, when
    eval_insts is given, the held-out M0-alone true-TWT* (m0_twt_mean / _per).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    estimator = ShiftEstimator(hidden=est_hidden)
    Xagg = np.zeros((0, LAT_DIM), np.float32)
    yagg = np.zeros((0,), np.float32)
    wagg = np.zeros((0,), np.float32)
    per_iter = []
    rng = np.random.default_rng(seed)
    n_ep = len(train_instances)
    cum_over = 0

    for it in range(outer_iters):
        order = rng.permutation(len(train_instances))[:n_ep]
        n_over = n_rev = n_conf = 0
        for k in order:
            inst = train_instances[int(k)]
            applied = overlay.apply(inst)
            sup = Supervisor(overlay, inst, rho=rho, epsilon=eps, theta=theta,
                             mechanism=mechanism, seed=seed, applied=applied)
            decider = make_decider(estimator, inst, decider_channel, device=device)
            _sched, log = DispatchEnv(inst).run_supervised(
                decider, supervisor=sup, method="m0_atc", seed=seed)
            X, y, w = AR.weak_labels_from_log(log, inst, override_weight, confirm_weight)
            if len(X):
                Xagg = np.concatenate([Xagg, X]); yagg = np.concatenate([yagg, y])
                wagg = np.concatenate([wagg, w])
            s = sup.summary()
            n_over += s["n_overrides"]; n_rev += s["n_reviews"]
            n_conf += s["n_confirmations"]
        loss = train_estimator(estimator, Xagg, yagg, wagg, device=device,
                               seed=seed + it)
        acc = AR.probe_shift_accuracy(estimator, probe_instances, overlay, device=device)
        orr = (n_over / n_rev) if n_rev else 0.0
        cum_over += n_over
        row = {"iter": it, "n_reviews": n_rev, "n_overrides": n_over,
               "cum_overrides": cum_over, "n_confirmations": n_conf,
               "override_rate": orr, "n_examples_agg": int(len(Xagg)),
               "est_loss": loss, **acc}
        if eval_insts is not None:
            m0_twts = []
            for inst in eval_insts:
                applied = overlay.apply(inst)
                d = make_decider(estimator, inst, decider_channel, device=device)
                sched, _ = DispatchEnv(inst).run_supervised(
                    d, supervisor=None, method="m0", seed=seed)
                m0_twts.append(TO.score_true(inst, sched, overlay, applied)["TWT_true"])
            row["m0_twt_mean"] = float(np.mean(m0_twts))
            row["m0_twt_per"] = [float(x) for x in m0_twts]
        per_iter.append(row)
    return {"estimator": estimator, "per_iter": per_iter}


# --------------------------------------------------------------------------- #
# Held-out ladder scoring (RULE / M0 / M0+SUP / RULE+SUP / ORACLE)             #
# --------------------------------------------------------------------------- #
def eval_ladder(estimator, eval_insts, overlay, *, rho, eps, theta, mechanism,
                decider_channel, seed, device="cpu"):
    """Per-instance TWT*(w*,d*) for the cheap ladder on the held-out set.

    RULE and ORACLE are estimator- and mechanism-independent; M0/M0+SUP use the
    given decider_channel; RULE+SUP/M0+SUP use the given review mechanism.
    Returns a dict of per-instance lists + realized review fractions.
    """
    per = {k: [] for k in ["rule", "m0_alone", "m0_sup", "rule_sup", "oracle"]}
    inst_ids = []
    rsup_rf, rsup_orr, m0sup_rf, m0sup_orr = [], [], [], []
    for inst in eval_insts:
        applied = overlay.apply(inst)
        inst_ids.append(inst["meta"]["id"])

        def sc(sched):
            return TO.score_true(inst, sched, overlay, applied)["TWT_true"]

        per["rule"].append(sc(dec.run_rule(DispatchEnv(inst), "atc", seed=seed)))

        m0d = make_decider(estimator, inst, decider_channel, device=device)
        m0_sched, _ = DispatchEnv(inst).run_supervised(
            m0d, supervisor=None, method="m0", seed=seed)
        per["m0_alone"].append(sc(m0_sched))

        osup = Supervisor(overlay, inst, rho=0.0, applied=applied)
        per["oracle"].append(sc(dec.run_oracle_greedy(DispatchEnv(inst), osup, seed=seed)))

        rsup = Supervisor(overlay, inst, rho=rho, epsilon=eps, theta=theta,
                          mechanism=mechanism, seed=seed, applied=applied)
        rsched, _ = dec.run_rule_sup(DispatchEnv(inst), "atc", rsup, seed=seed)
        per["rule_sup"].append(sc(rsched))
        rs = rsup.summary()
        rsup_rf.append(rs["reviewed_fraction"]); rsup_orr.append(rs["override_rate_of_reviews"])

        m0d2 = make_decider(estimator, inst, decider_channel, device=device)
        m0sup = Supervisor(overlay, inst, rho=rho, epsilon=eps, theta=theta,
                           mechanism=mechanism, seed=seed, applied=applied)
        m0s_sched, _ = DispatchEnv(inst).run_supervised(
            m0d2, supervisor=m0sup, method="m0_sup", seed=seed)
        per["m0_sup"].append(sc(m0s_sched))
        ms = m0sup.summary()
        m0sup_rf.append(ms["reviewed_fraction"]); m0sup_orr.append(ms["override_rate_of_reviews"])

    return {
        "inst_ids": inst_ids,
        "per": {k: [float(x) for x in v] for k, v in per.items()},
        "rule_sup_revfrac": [float(x) for x in rsup_rf],
        "rule_sup_orr": [float(x) for x in rsup_orr],
        "m0_sup_revfrac": [float(x) for x in m0sup_rf],
        "m0_sup_orr": [float(x) for x in m0sup_orr],
        "n_wos": len(eval_insts[0]["work_orders"]),
    }


# --------------------------------------------------------------------------- #
# Statistics (identical settings to scripts/y3_p4_m0grid.py)                    #
# --------------------------------------------------------------------------- #
def paired_wilcoxon(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.allclose(a - b, 0.0):
        return 1.0
    if not _HAVE_SCIPY:
        return float("nan")
    try:
        return float(_wilcoxon(a, b, zero_method="pratt").pvalue)
    except Exception:
        return float("nan")


def win_tie_loss(a, b, tol=_TOL):
    """W = test (a) strictly LOWER TWT* than comparator (b) = better."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    w = int(np.sum(a < b - tol)); l = int(np.sum(a > b + tol))
    return {"W": w, "T": int(len(a) - w - l), "L": l}


def seed_avg_contrast(test_mat, comp_mat):
    """Seed-averaged per-instance paired contrast. Inputs are (S seeds x n
    instances) matrices aligned by instance. Returns pct gain of test vs comp
    (comp better baseline => positive = test lower TWT*), W/T/L, Wilcoxon p."""
    a = np.asarray(test_mat, float).mean(axis=0)
    b = np.asarray(comp_mat, float).mean(axis=0)
    am, bm = float(a.mean()), float(b.mean())
    return {"test_mean": am, "comparator_mean": bm,
            "pct_gain": (100.0 * (bm - am) / bm) if abs(bm) > 1e-12 else 0.0,
            "wtl": win_tie_loss(a, b), "wilcoxon_p": paired_wilcoxon(a, b),
            "n_instances": int(a.size)}


# --------------------------------------------------------------------------- #
# Instance pools                                                              #
# --------------------------------------------------------------------------- #
import glob
import json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INST = os.path.join(_ROOT, "data", "processed", "instances")


def locate_storm2(campus, u, w="w80"):
    cdir = "c%02d" % campus
    return sorted(glob.glob(os.path.join(
        _INST, cdir, "storm2", w, "%s_storm2_%s_u%d_*.json" % (cdir, w, u))))


def load_json(p):
    with open(p) as fh:
        return json.load(fh)


def load_pools(campus, u, n_train=16, n_probe=4, n_eval=10):
    files = locate_storm2(campus, u)
    need = n_train + n_probe + n_eval
    if len(files) < need:
        raise RuntimeError("only %d files at c%d u%d (need %d)" % (len(files), campus, u, need))
    train = [load_json(p) for p in files[:n_train]]
    probe = [load_json(p) for p in files[n_train:n_train + n_probe]]
    eval_files = files[n_train + n_probe:n_train + n_probe + n_eval]
    eval_insts = [load_json(p) for p in eval_files]
    assert not (set(eval_files) & set(files[:n_train + n_probe])), "eval overlaps train"
    return train, probe, eval_insts, [os.path.basename(f) for f in eval_files]


def make_overlay(beta, family="F-NL", master_seed=12345, channel="full_class_shift"):
    o = ov.Overlay(ov.OverlayParams(beta=beta, family=family,
                                    master_seed=master_seed, channel=channel))
    assert o.params.channel == channel
    return o
