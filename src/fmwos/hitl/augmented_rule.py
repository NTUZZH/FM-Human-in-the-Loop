"""M0: the augmented rule (Paper Y3, P2, method-ladder rung 0).

M0 is the "just make the rule learn" baseline, the simplest correction layer.
It trains the SAME shift estimator as M1 (same architecture, same weak-label
training) but from the override log of the RULE(ATC)+SUP system, and then
re-scores ATC with corrected weights ``w(clip(c - hat_s, 1, 4))``. No RL, no
policy network anywhere in the loop.

Symmetric protocol with M1: the same 8 outer DAgger iterations, the same
per-iteration episode count, the same aggregation (the override examples are
never reset; the estimator is retrained on the full aggregate each iteration).
Per iteration the decider is the CURRENT augmented ATC (hat_s from the last fit),
so as the estimator recovers the latent the override rate falls, exactly as for
M1.

The corrected weight interpolates the class->weight table (w = 8/4/2/1 on classes
1..4) at the continuous corrected class ``clip(c - hat_s, 1, 4)``, so a graded
``hat_s`` produces a graded weight.
"""

from __future__ import annotations

import math

import numpy as np

from .. import pdrs
from ..env import DispatchEnv
from .overlay import base_features, W_OF_CLASS, SLA_OF_CLASS
from .latent_head import ShiftEstimator, train_estimator, LAT_DIM
from .supervisor import Supervisor, _ATC_K
from . import true_objective

# Class -> weight and class -> SLA lookups as interpolable curves on [1, 4].
_CLASS_GRID = np.array([1.0, 2.0, 3.0, 4.0])
_W_GRID = np.array([W_OF_CLASS[1], W_OF_CLASS[2], W_OF_CLASS[3], W_OF_CLASS[4]])
_SLA_GRID = np.array([SLA_OF_CLASS[1], SLA_OF_CLASS[2], SLA_OF_CLASS[3], SLA_OF_CLASS[4]])


def interp_weight(c_eff) -> float:
    """Weight at a (possibly non-integer) corrected class, clipped to [1, 4]."""
    c = float(min(4.0, max(1.0, c_eff)))
    return float(np.interp(c, _CLASS_GRID, _W_GRID))


def interp_sla(c_eff) -> float:
    """SLA (business-hours) at a (possibly non-integer) corrected class, clipped
    to [1, 4] -- the deadline half of the full-class-shift correction."""
    c = float(min(4.0, max(1.0, c_eff)))
    return float(np.interp(c, _CLASS_GRID, _SLA_GRID))


def corrected_weight(c: int, hat_s: float) -> float:
    """w(clip(c - hat_s, 1, 4)) with the interpolated class->weight curve."""
    return interp_weight(float(c) - float(hat_s))


def corrected_deadline(c: int, hat_s: float, release_bh: float) -> float:
    """r + SLA(clip(c - hat_s, 1, 4)) with the interpolated class->SLA curve.

    The deadline half of the P1.5 full-class-shift correction: a positive
    hat_s (more urgent than recorded) lifts the effective class, shortening the
    SLA and so pulling the effective deadline EARLIER, exactly mirroring the
    weight lift in ``corrected_weight``.
    """
    return float(release_bh) + interp_sla(float(c) - float(hat_s))


# --------------------------------------------------------------------------- #
# Weak labels from a RULE+SUP override log (features from instance data only)  #
# --------------------------------------------------------------------------- #
def weak_labels_from_log(log, instance, override_weight=5.0, confirm_weight=1.0,
                         label_source="executed"):
    """(X, y, w) weak-supervision examples from one episode's override log.

    Uses ONLY the log (decider / executed / preferred pick ids and flags) and the
    instance's observable fields (via ``base_features``); no latent quantity is
    read.

    ``label_source`` selects which pick carries the positive (+1) label on an
    OVERRIDE. The negative (-1) label on the decider pick and the zero-labelled
    confirmations are the same for both settings.

    * ``"executed"`` (DEFAULT): the pick the supervisor ACTUALLY started
      (``executed_pick``) -- the only thing a deployed logger can observe. Under
      the Appendix-D.4 noise model the random-override branch starts a random
      eligible order, so at eps>0 the positive label is genuinely corrupted; this
      is the honest weak-supervision signal.
    * ``"preferred"``: the supervisor's noise-free preferred pick
      (``preferred_pick``). At eps=0 an honest override executes exactly the
      preferred pick, so this is BIT-IDENTICAL to ``"executed"``. At eps>0 it is
      an UPPER BOUND that assumes the log records the supervisor's intent rather
      than the action it actually took -- information a deployed system does not
      have. Kept only as an explicit leakage/oracle-label variant.
    """
    if label_source not in ("executed", "preferred"):
        raise ValueError("label_source must be 'executed' or 'preferred'")
    wo_by_id = {w["id"]: w for w in instance["work_orders"]}

    def feat(wid):
        return base_features(wo_by_id[wid]).astype(np.float32)

    X, y, wt = [], [], []
    for e in log:
        if not e.get("reviewed"):
            continue
        di = e["decider_pick"]
        if label_source == "executed":
            # executed_pick is what a deployed logger records and is always
            # present in real supervisor logs; fall back to preferred_pick only
            # for legacy / synthetic logs that predate the executed field (there
            # executed==preferred is the intended meaning).
            pos = e.get("executed_pick", e.get("preferred_pick"))
        else:
            pos = e.get("preferred_pick")
        if e["override"]:
            if pos is not None:
                X.append(feat(pos)); y.append(1.0); wt.append(override_weight)
            X.append(feat(di)); y.append(-1.0); wt.append(override_weight)
        elif e.get("confirmation"):
            X.append(feat(di)); y.append(0.0); wt.append(confirm_weight)
    if not X:
        return (np.zeros((0, LAT_DIM), np.float32),
                np.zeros((0,), np.float32), np.zeros((0,), np.float32))
    return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(wt, np.float32)


# --------------------------------------------------------------------------- #
# Augmented ATC decider (corrected weights from hat_s)                        #
# --------------------------------------------------------------------------- #
def hat_s_map(estimator: ShiftEstimator, instance, device="cpu") -> dict:
    """hat_s for every work order in the instance (wo_id -> float)."""
    wos = instance["work_orders"]
    feats = np.stack([base_features(w) for w in wos]).astype(np.float32)
    vals = estimator.predict_np(feats, device=device)
    return {w["id"]: float(v) for w, v in zip(wos, vals)}


def augmented_atc_decider(estimator, instance, device="cpu", k=_ATC_K,
                          channel="full_class_shift"):
    """A (job, margin) decider: ATC re-scored with the P1.5 corrected class.

    The corrected class is ``c_hat = clip(c_j - hat_s_j, 1, 4)``. It moves:
      * the weight    ``w_corr = w(c_hat)``               (both channels), and
      * the deadline  ``d_corr = r_j + SLA(c_hat)``       (full_class_shift only).

        ``score(j) = (w_corr / p_j) * exp(-max(0, due_corr - t - p_j) / (k*pbar))``

    with ``due_corr = d_corr`` under full_class_shift (the headline: hat_s moves
    BOTH the weight and the deadline) and ``due_corr = recorded d_j`` under the
    weight_only ablation (hat_s moves only the weight, deadline frozen). Margin =
    top1-top2 corrected score.
    """
    if channel not in ("full_class_shift", "weight_only"):
        raise ValueError("channel must be full_class_shift or weight_only")
    hs = hat_s_map(estimator, instance, device=device)
    prio = {w["id"]: int(w["priority"]) for w in instance["work_orders"]}
    p_of = {w["id"]: float(w["p_bh"]) for w in instance["work_orders"]}
    rel = {w["id"]: float(w["release_bh"]) for w in instance["work_orders"]}
    due_rec = {w["id"]: float(w["due_bh"]) for w in instance["work_orders"]}
    full = (channel == "full_class_shift")

    def wcorr(wid):
        return corrected_weight(prio[wid], hs.get(wid, 0.0))

    def dcorr(wid):
        if full:
            return corrected_deadline(prio[wid], hs.get(wid, 0.0), rel[wid])
        return due_rec[wid]

    def _decider(queue, t, rng):
        pbar = sum(p_of[j["id"]] for j in queue) / len(queue)
        denom = k * pbar
        scores = []
        for j in queue:
            wid = j["id"]
            slack = max(0.0, dcorr(wid) - t - p_of[wid])
            s = (wcorr(wid) / p_of[wid]) * math.exp(-slack / denom)
            scores.append((s, wid, j))
        scores.sort(key=lambda x: (-x[0], x[1]))
        best = scores[0][2]
        margin = (scores[0][0] - scores[1][0]) if len(scores) >= 2 else pdrs._BIG_MARGIN
        return best, float(margin)

    return _decider


# --------------------------------------------------------------------------- #
# Probe accuracy (EVAL ONLY -- quarantined use of the overlay latent)         #
# --------------------------------------------------------------------------- #
def probe_shift_accuracy(estimator, probe_instances, overlay, device="cpu"):
    """hat_s recovery vs the TRUE shift s. EVALUATION ONLY.

    This is the SINGLE place M0 touches the overlay latent, and only to score
    recovery for the metrics table; the estimator is never trained on s. Returns
    a dict: sign accuracy on s!=0 orders, exact class-shift accuracy, Pearson r,
    and the always-zero baseline accuracy (fraction with s==0).
    """
    hs_all, s_all = [], []
    for inst in probe_instances:
        applied = overlay.apply(inst)               # EVAL-ONLY latent read
        shift = applied["shift"]
        feats = np.stack([base_features(w) for w in inst["work_orders"]]).astype(np.float32)
        hs = estimator.predict_np(feats, device=device)
        for w, h in zip(inst["work_orders"], hs):
            hs_all.append(float(h)); s_all.append(int(shift[w["id"]]))
    hs_all = np.asarray(hs_all); s_all = np.asarray(s_all)
    nz = s_all != 0
    sign_acc = float(np.mean(np.sign(hs_all[nz]) == np.sign(s_all[nz]))) if nz.any() else float("nan")
    pred_class = np.clip(np.round(hs_all), -2, 2).astype(int)
    exact_acc = float(np.mean(pred_class == s_all))
    zero_baseline = float(np.mean(s_all == 0))
    if hs_all.std() > 1e-9 and s_all.std() > 1e-9:
        r = float(np.corrcoef(hs_all, s_all)[0, 1])
    else:
        r = float("nan")
    return {"sign_acc_nonzero": sign_acc, "exact_class_acc": exact_acc,
            "pearson_r": r, "zero_baseline_acc": zero_baseline,
            "n_orders": int(s_all.size)}


# --------------------------------------------------------------------------- #
# M0 pipeline: symmetric 8-iteration aggregate + estimator training           #
# --------------------------------------------------------------------------- #
def run_m0(train_instances, probe_instances, overlay, *, beta_rho_eps,
           outer_iters=8, episodes_per_iter=None, mechanism="targeted",
           theta=1.0, override_weight=5.0, confirm_weight=1.0,
           est_hidden=32, seed=0, device="cpu", verbose=True):
    """Run the full M0 pipeline for one supervisor cell.

    beta_rho_eps : (beta, rho, epsilon) -- the cell (overlay carries beta).
    Returns a dict with per-iteration accuracy and the trained estimator.
    """
    beta, rho, eps = beta_rho_eps
    channel = getattr(overlay.params, "channel", "full_class_shift")
    estimator = ShiftEstimator(hidden=est_hidden)
    # aggregated weak-label examples (NEVER reset).
    Xagg = np.zeros((0, LAT_DIM), np.float32)
    yagg = np.zeros((0,), np.float32)
    wagg = np.zeros((0,), np.float32)
    per_iter = []
    rng = np.random.default_rng(seed)
    n_ep = episodes_per_iter or len(train_instances)

    for it in range(outer_iters):
        # rotate through the training instances so each iteration sees a batch
        order = rng.permutation(len(train_instances))[:n_ep]
        n_over = n_rev = n_conf = 0
        for k in order:
            inst = train_instances[int(k)]
            applied = overlay.apply(inst)
            sup = Supervisor(overlay, inst, rho=rho, epsilon=eps, theta=theta,
                             mechanism=mechanism, seed=seed, applied=applied)
            decider = augmented_atc_decider(estimator, inst, device=device,
                                            channel=channel)
            _sched, log = DispatchEnv(inst).run_supervised(
                decider, supervisor=sup, method="m0_atc", seed=seed)
            X, y, w = weak_labels_from_log(log, inst, override_weight, confirm_weight)
            if len(X):
                Xagg = np.concatenate([Xagg, X]); yagg = np.concatenate([yagg, y])
                wagg = np.concatenate([wagg, w])
            s = sup.summary()
            n_over += s["n_overrides"]; n_rev += s["n_reviews"]
            n_conf += s["n_confirmations"]
        # retrain the estimator on the FULL aggregate (symmetric with M1)
        loss = train_estimator(estimator, Xagg, yagg, wagg, device=device,
                               seed=seed + it)
        acc = probe_shift_accuracy(estimator, probe_instances, overlay, device=device)
        orr = (n_over / n_rev) if n_rev else 0.0
        row = {"iter": it, "n_reviews": n_rev, "n_overrides": n_over,
               "n_confirmations": n_conf, "override_rate": orr,
               "n_examples_agg": int(len(Xagg)), "est_loss": loss, **acc}
        per_iter.append(row)
        if verbose:
            print("[M0 it%d] reviews=%d over=%d orr=%.3f  sign_acc=%.3f "
                  "exact_acc=%.3f (zero=%.3f) r=%.3f n_agg=%d"
                  % (it, n_rev, n_over, orr, acc["sign_acc_nonzero"],
                     acc["exact_class_acc"], acc["zero_baseline_acc"],
                     acc["pearson_r"], len(Xagg)))
    return {"estimator": estimator, "per_iter": per_iter}


def evaluate_m0_vs_atc(estimator, eval_instances, overlay, seed=0, device="cpu"):
    """Held-out true-TWT* of the augmented rule (M0) vs plain ATC.

    Both schedules are scored on the TRUE objective under the overlay's active
    channel (full_class_shift => TWT*(w*,d*); weight_only => TWT*(w*,recorded d),
    the E6 boundary). Plain ATC is the deployed recorded-field rule; M0 re-scores
    ATC with the corrected class (weight AND deadline under full_class_shift).
    Returns pooled sums and per-instance win/tie/loss (M0 < ATC on TWT* = a win).
    """
    channel = getattr(overlay.params, "channel", "full_class_shift")
    m0_sum = atc_sum = 0.0
    wins = ties = losses = 0
    for inst in eval_instances:
        applied = overlay.apply(inst)
        # plain ATC
        sched_atc = DispatchEnv(inst).run_policy(pdrs.get_rule("atc"), method="atc")
        # augmented ATC (M0)
        dec = augmented_atc_decider(estimator, inst, device=device, channel=channel)
        sched_m0, _ = DispatchEnv(inst).run_supervised(dec, supervisor=None,
                                                       method="m0", seed=seed)
        t_atc = true_objective.score_true(inst, sched_atc, overlay, applied)["TWT_true"]
        t_m0 = true_objective.score_true(inst, sched_m0, overlay, applied)["TWT_true"]
        atc_sum += t_atc; m0_sum += t_m0
        if t_m0 < t_atc - 1e-9:
            wins += 1
        elif t_m0 > t_atc + 1e-9:
            losses += 1
        else:
            ties += 1
    return {"m0_true_twt": m0_sum, "atc_true_twt": atc_sum,
            "n": len(eval_instances), "wins": wins, "ties": ties, "losses": losses}
