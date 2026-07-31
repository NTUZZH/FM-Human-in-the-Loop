"""W3: censoring-aware ordinal shift estimation (Paper Y3).

The true priority class is ``c* = clip(c - s, 1, 4)``, so the scale itself
censors the evidence at both ends. An order recorded in the most urgent class
(c = 1) can only be revealed as LESS urgent, and one recorded in the least
urgent class (c = 4) only as MORE urgent. The shipped estimator
(``augmented_rule.weak_labels_from_log`` + ``latent_head.train_estimator``)
fits a weighted squared error to point labels in {+1, -1, 0}, so at a boundary
class it fits a target the scale cannot express: a -1 on a recorded class-4
order asserts a true class of 5, which does not exist.

This module keeps the shipped network EXACTLY as it is and changes only the
likelihood.

THE MODEL
    latent shift        s_j = mu(x_j) + sigma * eps,   eps ~ N(0, 1)
    effective shift     t_j = clip(s_j, L_j, U_j),     L_j = c_j - 4, U_j = c_j - 1

``t_j`` is not a modelling flourish: it is exactly the shift the deployed
decider applies, because ``c_hat = clip(c - s_hat, 1, 4)`` means the correction
actually delivered to the ATC index is ``clip(s_hat, L, U)``. The weak label
``y_j`` is a proxy observation of ``t_j``.

THE LIKELIHOOD (Tobit / two-limit censored regression)
    L_j <  y_j <  U_j   uncensored:      Gaussian density of s at y_j
         y_j <= L_j     left-censored:   log Phi((L_j - mu)/sigma)   "at most this urgent"
         y_j >= U_j     right-censored:  log Phi((mu - U_j)/sigma)   "at least this urgent"

Written as a MINIMISED objective and scaled by 2 so that the uncensored term is
byte-for-byte the shipped squared error at sigma = 1:

    uncensored     (mu - y)^2 / sigma^2   [ + 2 log sigma, only when sigma is fitted ]
    left-censored  -2 log Phi((L - mu)/sigma)
    right-censored -2 log Phi((mu - U)/sigma)

With ``mode="none"`` no example is censored and the objective, the optimiser,
the batch stream and the epoch count reduce to ``latent_head.train_estimator``
exactly; ``tests/test_censored.py`` pins that reduction bit-for-bit, which is
what licenses reading any difference as the likelihood and nothing else.

WHY TOBIT AND NOT AN ORDINAL LOGIT WITH COLLAPSED BOUNDARY CATEGORIES
Both were admissible. Tobit was chosen for three reasons. (1) It keeps the
output semantics the deployed decider already consumes -- one continuous shift
per order -- whereas an ordinal logit outputs a distribution over classes and
needs a decoder before the ATC index can use it, which would add a second
difference to the comparison. (2) It nests the incumbent: at sigma = 1 with no
censored example the two objectives are the same arithmetic, so the comparison
is controlled by construction rather than by argument. (3) It keeps the network
at the incumbent's parameter count exactly (1761 with sigma fixed; 1762 with
sigma fitted, the one extra scalar being the scale).

THE LEVEL IS ANCHORED (the W2 lesson, deliberately not repeated)
W2 replaced the squared error with a ranking (conditional-logit) objective and
it failed on deployment, because a ranking objective is invariant to the LEVEL
of the shift and confirmations were what pinned that level. Every term above is
a probability of a half-line, or a density at a point, on the SHIFT SCALE
ITSELF: no term is invariant to adding a constant to ``mu``. Uncensored
observations (every confirmation away from a boundary class, every override
label inside the expressible range) contribute an unbounded quadratic pull
toward a fixed point, and the censored terms are proper one-sided probabilities
on the same scale. The location is therefore identified, and ``mode="impossible"``
is provided as the maximally anchor-preserving variant: it censors ONLY the two
structurally impossible labels and leaves every attainable point label alone.

DEPLOYMENT: TWO MAPPINGS, REPORTED SEPARATELY
``plugin``   feeds ``mu(x)`` to the shipped ``augmented_rule.augmented_atc_decider``
             verbatim, so the deployment path is byte-identical to the incumbent's
             and only the fitted weights differ.
``expected`` feeds ``E[t | x, c] = E[clip(s, L, U)]`` (closed form for a Gaussian),
             the decision-theoretically correct corrected-class input, which differs
             from ``clip(E[s], L, U)`` exactly at the boundary classes -- the same
             place the censoring bites. Kept separate because it changes the
             DECISION RULE, not the likelihood, and the two claims must be able to
             succeed or fail on their own.

GUARDRAIL (structural). Every feature this module sees is built by
``overlay.base_features``, which reads only ``trade`` / ``p_bh`` /
``release_bh``. The recorded class ``c`` enters ONLY as the censoring bound and
the deployment clip, never as a network input, so the estimator remains the
incumbent's function of the incumbent's features. No latent quantity (s, xi, f,
c*, w*, d*) is read anywhere in this file.

Nothing here edits or forks a shipped file: the estimator core, the
class->weight / class->SLA curves and the ATC constant are IMPORTED.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .. import pdrs
from .augmented_rule import _CLASS_GRID, _SLA_GRID, _W_GRID
from .latent_head import LAT_DIM, ShiftEstimator
from .overlay import base_features
from .supervisor import _ATC_K

# The recorded priority scale. c* = clip(c - s, CLASS_MIN, CLASS_MAX).
CLASS_MIN = 1.0
CLASS_MAX = 4.0

# Censoring conventions.
#   "none"        no example is censored -> reduces to the shipped squared error.
#   "impossible"  censor ONLY labels the scale cannot express at all (a -1 on a
#                 recorded class-4 order, a +1 on a recorded class-1 order).
#                 Every attainable point label, including every confirmation,
#                 keeps its point anchor. The minimal fix, and the most
#                 conservative with respect to the W2 level-anchoring lesson.
#   "strict"      the textbook two-limit Tobit convention: an observation AT the
#                 censoring point is censored too, so a confirmation on a
#                 boundary class contributes "t = L" as the one-sided statement
#                 "s <= L" rather than the point "s = 0".
CENSOR_MODES = ("none", "impossible", "strict")

DEPLOY_MODES = ("plugin", "expected")

_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# 1. Censoring bounds and codes                                               #
# --------------------------------------------------------------------------- #
def censoring_bounds(c):
    """Expressible range [L, U] of the effective shift t = c - c*.

    ``c*`` is confined to [CLASS_MIN, CLASS_MAX], so t = c - c* is confined to
    [c - CLASS_MAX, c - CLASS_MIN]. Accepts a scalar or an array.
    """
    c = np.asarray(c, dtype=np.float64)
    return c - CLASS_MAX, c - CLASS_MIN


def censor_codes(y, c, mode="strict"):
    """Per-example censoring code: -1 left, 0 uncensored, +1 right.

    ``y`` weak label, ``c`` RECORDED priority class. Under ``"strict"`` an
    observation at or beyond a limit is censored (the Tobit convention); under
    ``"impossible"`` only observations strictly beyond a limit are, which for
    labels in {+1, -1, 0} means exactly the two structurally impossible ones.
    """
    if mode not in CENSOR_MODES:
        raise ValueError("mode must be one of %r" % (CENSOR_MODES,))
    y = np.asarray(y, dtype=np.float64)
    lo, hi = censoring_bounds(c)
    code = np.zeros(y.shape, dtype=np.int64)
    if mode == "none":
        return code, lo, hi
    if mode == "strict":
        code[y <= lo] = -1
        code[y >= hi] = +1
    else:                                   # "impossible"
        code[y < lo] = -1
        code[y > hi] = +1
    return code, lo, hi


# --------------------------------------------------------------------------- #
# 2. Weak labels, carrying the recorded class                                 #
# --------------------------------------------------------------------------- #
def weak_labels_with_class(log, instance, override_weight=5.0, confirm_weight=1.0,
                           label_source="executed"):
    """(X, y, w, c) weak-supervision examples from one episode's override log.

    (X, y, w) is BIT-IDENTICAL to ``augmented_rule.weak_labels_from_log`` with
    the same arguments -- the construction below is the same one, in the same
    order, with the same feature constructor; the only addition is ``c``, the
    RECORDED priority class of the labelled order, which the censoring bounds
    need and which the shipped constructor does not return.
    ``tests/test_censored.py`` pins the identity.
    """
    if label_source not in ("executed", "preferred"):
        raise ValueError("label_source must be 'executed' or 'preferred'")
    wo_by_id = {w["id"]: w for w in instance["work_orders"]}

    def feat(wid):
        return base_features(wo_by_id[wid]).astype(np.float32)

    def cls(wid):
        return float(wo_by_id[wid]["priority"])

    X, y, wt, cc = [], [], [], []
    for e in log:
        if not e.get("reviewed"):
            continue
        di = e["decider_pick"]
        if label_source == "executed":
            pos = e.get("executed_pick", e.get("preferred_pick"))
        else:
            pos = e.get("preferred_pick")
        if e["override"]:
            if pos is not None:
                X.append(feat(pos)); y.append(1.0); wt.append(override_weight)
                cc.append(cls(pos))
            X.append(feat(di)); y.append(-1.0); wt.append(override_weight)
            cc.append(cls(di))
        elif e.get("confirmation"):
            X.append(feat(di)); y.append(0.0); wt.append(confirm_weight)
            cc.append(cls(di))
    if not X:
        z = np.zeros((0,), np.float32)
        return np.zeros((0, LAT_DIM), np.float32), z, z, z
    return (np.asarray(X, np.float32), np.asarray(y, np.float32),
            np.asarray(wt, np.float32), np.asarray(cc, np.float32))


# --------------------------------------------------------------------------- #
# 3. The model: the shipped estimator plus one scale scalar                   #
# --------------------------------------------------------------------------- #
class CensoredShiftEstimator(nn.Module):
    """The SHIPPED ``ShiftEstimator`` plus the censored model's scale sigma.

    ``core`` is imported, not re-implemented, so the architecture, the parameter
    names and (given the same torch seed) the initial weights are the
    incumbent's exactly. ``sigma`` is a fixed buffer by default, which leaves the
    parameter count at the incumbent's 1761; ``learn_sigma=True`` promotes it to
    a parameter (1762) and is run only as a robustness row.
    """

    def __init__(self, lat_dim: int = LAT_DIM, hidden: int = 32,
                 sigma: float = 1.0, learn_sigma: bool = False):
        super().__init__()
        self.core = ShiftEstimator(lat_dim=lat_dim, hidden=hidden)
        self.learn_sigma = bool(learn_sigma)
        v = torch.tensor(math.log(float(sigma)))
        if learn_sigma:
            self.log_sigma = nn.Parameter(v)
        else:
            self.register_buffer("log_sigma", v)

    def forward(self, latfeat):
        return self.core(latfeat)

    def sigma(self):
        return self.log_sigma.exp().clamp_min(1e-3)

    def sigma_value(self) -> float:
        with torch.no_grad():
            return float(self.sigma())

    def predict_np(self, feats: np.ndarray, device="cpu") -> np.ndarray:
        return self.core.predict_np(feats, device=device)

    def n_params_estimator(self) -> int:
        """Parameters of the shift network alone -- must equal the incumbent's."""
        return sum(p.numel() for p in self.core.parameters())

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# 4. The censored objective                                                   #
# --------------------------------------------------------------------------- #
def censored_terms(mu, y, code, lo, hi, sigma, learn_sigma=False):
    """Per-example minimised objective (2 x negative log-likelihood, constants
    dropped). Returns a tensor shaped like ``mu``.

    At sigma = 1 with every code 0 this is exactly ``(mu - y)**2``, the shipped
    squared error, evaluated with the same arithmetic.
    """
    sq = (mu - y) ** 2 / (sigma ** 2)
    if learn_sigma:
        sq = sq + 2.0 * torch.log(sigma)
    # One safe standardised argument per example: the discarded branch must stay
    # finite or ``torch.where`` would back-propagate a NaN through it.
    z_left = (lo - mu) / sigma
    z_right = (mu - hi) / sigma
    zero = torch.zeros_like(mu)
    z = torch.where(code < 0, z_left, torch.where(code > 0, z_right, zero))
    cens = -2.0 * torch.special.log_ndtr(z)
    return torch.where(code == 0, sq, cens)


def train_censored_estimator(model, X, y, w, c, *, mode="strict", epochs: int = 40,
                             lr: float = 1e-2, batch_size: int = 512,
                             device="cpu", seed: int = 0):
    """Censored fit of ``model`` to weak labels. Returns final loss.

    Deliberately a line-for-line mirror of ``latent_head.train_estimator``: the
    same optimiser (Adam at lr 1e-2), the same batch size, the same 40 epochs,
    the same ``np.random.default_rng(seed).permutation`` batch stream and the
    same weighted-mean reduction. The ONLY difference is the per-example term.
    With ``mode="none"`` the two routines are bit-identical.
    """
    if len(X) == 0:
        return float("nan")
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    code_np, lo_np, hi_np = censor_codes(y, c, mode=mode)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    wt = torch.as_tensor(w, dtype=torch.float32, device=device)
    ct = torch.as_tensor(code_np, dtype=torch.int64, device=device)
    lot = torch.as_tensor(lo_np, dtype=torch.float32, device=device)
    hit = torch.as_tensor(hi_np, dtype=torch.float32, device=device)
    n = Xt.shape[0]
    rng = np.random.default_rng(seed)
    last = float("nan")
    for _ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            bt = torch.as_tensor(b, device=device)
            pred = model(Xt[bt])
            per = censored_terms(pred, yt[bt], ct[bt], lot[bt], hit[bt],
                                 model.sigma(), learn_sigma=model.learn_sigma)
            loss = (wt[bt] * per).sum() / wt[bt].sum().clamp_min(1e-8)
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.item())
    return last


def censor_summary(y, c, mode="strict"):
    """How many examples the censoring rule actually touches, by class."""
    code, _lo, _hi = censor_codes(y, c, mode=mode)
    out = {"n": int(code.size), "n_left": int((code < 0).sum()),
           "n_right": int((code > 0).sum()),
           "n_uncensored": int((code == 0).sum()), "mode": mode}
    for k in (1, 2, 3, 4):
        m = np.asarray(c) == float(k)
        out["class_%d" % k] = {
            "n": int(m.sum()), "n_left": int((code[m] < 0).sum()),
            "n_right": int((code[m] > 0).sum())}
    return out


# --------------------------------------------------------------------------- #
# 5. Deployment mapping                                                       #
# --------------------------------------------------------------------------- #
def expected_effective_shift(mu, sigma, lo, hi):
    """E[clip(s, lo, hi)] for s ~ N(mu, sigma^2), in closed form.

    This is the corrected class's decision-theoretic input: the deployed
    correction applies ``clip(s_hat, lo, hi)``, and E[clip(s)] differs from
    clip(E[s]) exactly at the boundary classes, which is where the censoring
    bites. As sigma -> 0 it converges to clip(mu, lo, hi).
    """
    mu = np.asarray(mu, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    s = float(sigma)
    a = (lo - mu) / s
    b = (hi - mu) / s
    Fa = 0.5 * (1.0 + np.vectorize(math.erf)(a / math.sqrt(2.0)))
    Fb = 0.5 * (1.0 + np.vectorize(math.erf)(b / math.sqrt(2.0)))
    fa = np.exp(-0.5 * a * a) / _SQRT_TWO_PI
    fb = np.exp(-0.5 * b * b) / _SQRT_TWO_PI
    return lo * Fa + hi * (1.0 - Fb) + mu * (Fb - Fa) + s * (fa - fb)


def deployed_hat_s_map(model, instance, deploy="plugin", device="cpu") -> dict:
    """wo_id -> the shift the decider will actually be handed.

    ``plugin``   mu(x), i.e. exactly ``augmented_rule.hat_s_map``.
    ``expected`` E[clip(s, L, U) | x, c], the censored model's posterior-mean
                 effective shift, which the clip inside the decider then leaves
                 unchanged because it already lies inside [L, U].
    """
    if deploy not in DEPLOY_MODES:
        raise ValueError("deploy must be one of %r" % (DEPLOY_MODES,))
    wos = instance["work_orders"]
    feats = np.stack([base_features(w) for w in wos]).astype(np.float32)
    mu = np.asarray(model.predict_np(feats, device=device), dtype=np.float64)
    if deploy == "plugin":
        return {w["id"]: float(v) for w, v in zip(wos, mu)}
    c = np.asarray([float(w["priority"]) for w in wos], dtype=np.float64)
    lo, hi = censoring_bounds(c)
    t = expected_effective_shift(mu, model.sigma_value(), lo, hi)
    return {w["id"]: float(v) for w, v in zip(wos, t)}


def hat_s_map_atc_decider(hs, instance, k=_ATC_K, channel="full_class_shift"):
    """The shipped augmented-ATC decider, driven by a PRE-COMPUTED hat_s map.

    Byte-for-byte the body of ``augmented_rule.augmented_atc_decider`` (same
    corrected weight, same corrected deadline, same score, same deterministic
    ``(-score, id)`` tie-break, same top1-top2 margin); the only change is that
    the hat_s map is supplied rather than computed from a ``ShiftEstimator``, so
    a deployment mapping that needs the recorded class can be used.
    ``tests/test_censored.py`` asserts that with the plug-in map this decider is
    identical to the shipped one, decision for decision.
    """
    if channel not in ("full_class_shift", "weight_only"):
        raise ValueError("channel must be full_class_shift or weight_only")
    prio = {w["id"]: int(w["priority"]) for w in instance["work_orders"]}
    p_of = {w["id"]: float(w["p_bh"]) for w in instance["work_orders"]}
    rel = {w["id"]: float(w["release_bh"]) for w in instance["work_orders"]}
    due_rec = {w["id"]: float(w["due_bh"]) for w in instance["work_orders"]}
    full = (channel == "full_class_shift")

    def _decider(queue, t, rng):
        pbar = sum(p_of[j["id"]] for j in queue) / len(queue)
        denom = k * pbar
        scores = []
        for j in queue:
            wid = j["id"]
            c_eff = min(4.0, max(1.0, float(prio[wid]) - hs.get(wid, 0.0)))
            w_corr = float(np.interp(c_eff, _CLASS_GRID, _W_GRID))
            if full:
                d_corr = rel[wid] + float(np.interp(c_eff, _CLASS_GRID, _SLA_GRID))
            else:
                d_corr = due_rec[wid]
            slack = max(0.0, d_corr - t - p_of[wid])
            s = (w_corr / p_of[wid]) * math.exp(-slack / denom)
            scores.append((s, wid, j))
        scores.sort(key=lambda x: (-x[0], x[1]))
        best = scores[0][2]
        margin = (scores[0][0] - scores[1][0]) if len(scores) >= 2 else pdrs._BIG_MARGIN
        return best, float(margin)

    return _decider


def censored_atc_decider(model, instance, deploy="plugin", device="cpu",
                         k=_ATC_K, channel="full_class_shift"):
    """The augmented-ATC decider for a censored model under a deployment mapping."""
    hs = deployed_hat_s_map(model, instance, deploy=deploy, device=device)
    return hat_s_map_atc_decider(hs, instance, k=k, channel=channel)


# --------------------------------------------------------------------------- #
# 6. What the schedule actually receives                                      #
# --------------------------------------------------------------------------- #
def applied_shift(hs_map, instance):
    """The shift the corrected class really applies: clip(hat_s, L, U) per order.

    The deployed correction is ``c_hat = clip(c - hat_s, 1, 4)``, so the shift
    that reaches the ATC index is ``clip(hat_s, c - 4, c - 1)``, not ``hat_s``.
    Reporting both separates "the estimator's fitted level" from "the correction
    the schedule sees", which are not the same number at the boundary classes.
    """
    ids, out = [], []
    for w in instance["work_orders"]:
        c = float(w["priority"])
        lo, hi = c - CLASS_MAX, c - CLASS_MIN
        ids.append(w["id"])
        out.append(float(min(hi, max(lo, hs_map.get(w["id"], 0.0)))))
    return ids, np.asarray(out, dtype=np.float64)
