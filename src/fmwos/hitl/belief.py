"""M2: the Bayesian-belief / active-elicitation rung (Paper Y3, P5 ABLATION).

M2 is the third rung of the M0/M1/M2 method ladder. It is an
ABLATION: the headline (M0, the deployable correction layer) does not depend on
it. M2 asks two questions the point-estimator M0 cannot:

  1. Does a *Bayesian* estimator of the latent class shift match M0's point
     estimator at the same review budget?  (parity check)
  2. Given a posterior *variance* over the belief, can we place a fixed review
     budget better -- reviewing decisions that are both consequential AND
     uncertain -- and so recover the latent from FEWER / better-placed overrides?
     (active elicitation)

M2 differs from M0 in EXACTLY ONE component: the estimator. Everything else is
reused verbatim from ``augmented_rule`` (M0):

  * the override-log weak labels (``weak_labels_from_log``): an override that
    promotes an order is +1 evidence (its true class is more urgent than
    recorded), the demoted order is -1, and a CONFIRMATION is a censored
    "within-theta" observation at y=0 -- documented below;
  * the augmented-ATC decider (``augmented_atc_decider``): ATC re-scored with the
    corrected class ``clip(c - hat_s, 1, 4)``, correcting BOTH the weight and the
    deadline under full_class_shift;
  * the held-out TWT*(w*,d*) scoring and probe accuracy.

In place of M0's MLP ``ShiftEstimator`` (fit by weighted-MSE SGD), M2 uses a
conjugate Bayesian *linear* regression on the SAME campus-agnostic features x
(``overlay.base_features``: trade one-hot, log1p processing time, release
day-of-week; NEVER a latent quantity). Because the estimator exposes the same
``predict_np`` interface, it is a drop-in replacement everywhere M0 consumes an
estimator -- the parity question is therefore a clean "MLP vs Bayesian-linear"
swap on identical logs.

The belief additionally exposes a POSTERIOR VARIANCE per order (the epistemic
parameter uncertainty projected onto the order's features, ``x^T Sigma x``),
which the active-review supervisor (``ActiveReviewSupervisor``) uses to steer a
fixed review budget toward under-observed feature regions.

Design choice: the CENSORED confirmation update
-----------------------------------------------
A confirmation certifies the decider's pick was within ``theta`` of the
supervisor's preference, i.e. the realized class shift for that order is SMALL in
magnitude. We model this exactly as M0's weak label does: a confirmation is an
observation ``y = 0`` at the lighter ``confirm_weight`` (so it pulls the belief
toward zero shift in that order's feature region, but with less force than a
decisive override). Under the conjugate Gaussian model an observation's weight IS
its precision, so a confirmation is a low-precision "shift approximately 0"
datum -- a reasonable, transparent censored update. (A fuller Tobit-style
left/right-censored likelihood is possible; we keep the y=0-at-low-precision form
because it makes M2 differ from M0 in the estimator ALONE, which is the point of
the ablation.)

Conjugacy note: batch == sequential
-----------------------------------
With a Gaussian prior and Gaussian (per-observation-precision) likelihood the
posterior is Gaussian and the update is order-independent: fitting the whole
aggregated override log at once gives byte-identical posterior mean and
covariance to folding the overrides/confirmations in one at a time. The M2
pipeline therefore fits the belief on the never-reset aggregate each DAgger
iteration (mirroring M0's symmetric protocol), and ``update`` is provided for the
incremental view.
"""

from __future__ import annotations

import numpy as np

from .overlay import base_features, _BASE_DIM
from .supervisor import Supervisor


# --------------------------------------------------------------------------- #
# Conjugate Bayesian linear regression over the class shift                    #
# --------------------------------------------------------------------------- #
class BayesianLinearShift:
    """Conjugate Bayesian linear regression predicting the class shift hat_s.

    Model (observation noise variance folded to 1, so an observation's WEIGHT is
    its precision):

        s = x^T w + e ,   e ~ N(0, 1 / weight)
        prior:      w ~ N(0, alpha^{-1} I)
        posterior:  w ~ N(m, Sigma)  with
                    A     = alpha I + X^T diag(weight) X       (posterior precision)
                    Sigma = A^{-1}
                    m     = Sigma (X^T diag(weight) y)         (prior mean 0)

    ``alpha`` is the prior precision (ridge strength); larger = stronger pull to a
    zero shift, smaller = let the override log speak. The posterior mean is the
    ridge estimate with lambda = alpha, so with alpha small it tracks the linear
    least-squares fit of the +1/-1/0 weak labels; the posterior covariance
    ``Sigma`` shrinks in feature directions the override log has visited, which is
    exactly the signal the active-review supervisor exploits.

    Interface: ``predict_np(feats, device=...)`` returns the posterior MEAN
    (drop-in for ``latent_head.ShiftEstimator``); ``predict_var_np(feats)``
    returns the per-row epistemic variance ``x^T Sigma x``.
    """

    def __init__(self, dim: int = _BASE_DIM, alpha: float = 1.0):
        self.dim = int(dim)
        self.alpha = float(alpha)
        # Sufficient statistics of the posterior precision / information vector.
        self._A = self.alpha * np.eye(self.dim)          # posterior precision
        self._b = np.zeros(self.dim)                     # X^T diag(w) y
        self._n_obs = 0
        self._refresh()

    # ------------------------------------------------------------------ #
    def _refresh(self):
        """Recompute posterior mean and covariance from (A, b)."""
        self._cov = np.linalg.inv(self._A)
        self._mean = self._cov @ self._b

    def _reset_prior(self):
        self._A = self.alpha * np.eye(self.dim)
        self._b = np.zeros(self.dim)
        self._n_obs = 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_XYW(X, y, w):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if w is None:
            w = np.ones_like(y)
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        return X, y, w

    def update(self, X, y, w=None):
        """Fold observations into the posterior (incremental; conjugate)."""
        X, y, w = self._as_XYW(X, y, w)
        if X.shape[0] == 0:
            return self
        Xw = X * w[:, None]
        self._A += Xw.T @ X
        self._b += Xw.T @ y
        self._n_obs += X.shape[0]
        self._refresh()
        return self

    def fit(self, X, y, w=None):
        """Batch posterior from the PRIOR over the full observation set.

        Equal (to machine precision) to ``reset``+repeated ``update`` because the
        conjugate Gaussian update is order-independent. Used by the M2 pipeline on
        the never-reset aggregate each DAgger iteration (symmetric with M0).
        """
        self._reset_prior()
        return self.update(X, y, w)

    # ------------------------------------------------------------------ #
    def predict_np(self, feats, device=None) -> np.ndarray:
        """Posterior MEAN hat_s for a [N, dim] feature matrix.

        Signature (``device`` kwarg) matches ``ShiftEstimator.predict_np`` so the
        belief drops into ``augmented_rule``'s decider / probe / hat_s_map
        unchanged. ``device`` is ignored (pure numpy).
        """
        feats = np.asarray(feats, dtype=np.float64)
        if feats.ndim == 1:
            feats = feats[None, :]
        return feats @ self._mean

    def predict_var_np(self, feats) -> np.ndarray:
        """Per-order epistemic (parameter) variance x^T Sigma x for [N, dim].

        This is the belief uncertainty used to target reviews: it is large in
        feature directions the override log has NOT yet pinned down and shrinks as
        overrides/confirmations accumulate there. Observation noise (the +1) is
        omitted deliberately -- targeting cares about what MORE data would
        resolve, not the irreducible label noise.
        """
        feats = np.asarray(feats, dtype=np.float64)
        if feats.ndim == 1:
            feats = feats[None, :]
        v = np.einsum("ni,ij,nj->n", feats, self._cov, feats)
        return np.maximum(v, 0.0)

    # ------------------------------------------------------------------ #
    @property
    def posterior_mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def posterior_cov(self) -> np.ndarray:
        return self._cov.copy()

    @property
    def n_obs(self) -> int:
        return self._n_obs


# --------------------------------------------------------------------------- #
# Per-instance belief maps (mirror augmented_rule.hat_s_map)                    #
# --------------------------------------------------------------------------- #
def belief_mean_map(belief: BayesianLinearShift, instance) -> dict:
    """Posterior-mean hat_s for every work order (wo_id -> float)."""
    wos = instance["work_orders"]
    feats = np.stack([base_features(w) for w in wos]).astype(np.float64)
    vals = belief.predict_np(feats)
    return {w["id"]: float(v) for w, v in zip(wos, vals)}


def belief_variance_map(belief: BayesianLinearShift, instance) -> dict:
    """Posterior variance x^T Sigma x for every work order (wo_id -> float)."""
    wos = instance["work_orders"]
    feats = np.stack([base_features(w) for w in wos]).astype(np.float64)
    vals = belief.predict_var_np(feats)
    return {w["id"]: float(v) for w, v in zip(wos, vals)}


# --------------------------------------------------------------------------- #
# Active-elicitation supervisor: budget targeted at consequential AND uncertain #
# --------------------------------------------------------------------------- #
class ActiveReviewSupervisor(Supervisor):
    """Supervisor whose fixed review budget is steered by the belief's variance.

    The override protocol, the myopic-greedy preferred pick, the noise model and
    the logging are INHERITED UNCHANGED from ``Supervisor``. Only the review
    TARGETING changes: instead of reviewing the ``rho``-fraction of reviewable
    decisions with the smallest decider margin (consequential), it reviews the
    ``rho``-fraction with the highest COMBINED priority

        combined = z(-margin) + var_weight * z(max_candidate_belief_variance)

    i.e. decisions that are jointly the most consequential (small margin) AND the
    most uncertain (a candidate lies in an under-observed feature region). Both
    signals are standardized against the CURRENT rolling window and the combined
    score is compared to the window's (1 - rho)-quantile, then passed through the
    SAME online budget controller, so the realized reviewed fraction still tracks
    ``rho`` (matched budget). The s=+2 forced-review rule and the >=2-candidate
    gate are inherited exactly.

    With ``var_weight = 0`` the combined score is a monotone (affine) transform of
    -margin over the window, so the review decisions reduce BIT-FOR-BIT to the
    base TARGETED supervisor's ``margin <= rho-quantile`` rule (a clean control;
    verified in tests).

    ``variance_map`` (wo_id -> posterior variance under the CURRENT belief) is
    fixed for the episode, exactly as the augmented-ATC decider's hat_s is fixed
    for the episode; both are refreshed between DAgger iterations.
    """

    def __init__(self, *args, variance_map=None, var_weight: float = 1.0,
                 prior_variance: float = 0.0, **kw):
        super().__init__(*args, **kw)
        self.variance_map = variance_map or {}
        self.var_weight = float(var_weight)
        self.prior_variance = float(prior_variance)
        self._lm_hist: list[float] = []     # -margin (consequential signal)
        self._uv_hist: list[float] = []     # max candidate belief variance

    def _cand_uncertainty(self, candidates) -> float:
        vs = [self.variance_map.get(c["id"], self.prior_variance) for c in candidates]
        return max(vs) if vs else self.prior_variance

    @staticmethod
    def _combined(lm, uv, lm_arr, uv_arr, var_weight):
        """z(-margin) + var_weight * z(variance), both standardized against the
        current window ``(lm_arr, uv_arr)``. Vectorized: ``lm``/``uv`` may be
        scalars (current decision) or arrays (the window itself)."""
        def z(v, arr):
            mu = float(arr.mean())
            sd = float(arr.std())
            return (v - mu) / (sd if sd > 1e-9 else 1.0)
        return z(lm, lm_arr) + var_weight * z(uv, uv_arr)

    def _decide_review(self, margin, candidates):
        # var_weight == 0 IS the base TARGETED supervisor: delegate so the control
        # reduces to it bit-for-bit (no float-boundary drift from the z-transform).
        if self.var_weight == 0.0:
            return super()._decide_review(margin, candidates)
        if self.rho <= 0.0:
            return False
        if self.mechanism == "random":                 # inherited path (unused here)
            return bool(self.rng.random() < self.rho)
        if len(candidates) < 2:
            return False
        self.n_reviewable += 1

        lm = -float(margin)
        uv = float(self._cand_uncertainty(candidates))
        if len(self._lm_hist) >= 8:
            lm_arr = np.asarray(self._lm_hist)
            uv_arr = np.asarray(self._uv_hist)
            combined_cur = self._combined(lm, uv, lm_arr, uv_arr, self.var_weight)
            combined_hist = self._combined(lm_arr, uv_arr, lm_arr, uv_arr, self.var_weight)
            thr = float(np.percentile(combined_hist, 100.0 * (1.0 - self.rho)))
            consequential = (combined_cur >= thr)
        else:
            consequential = True                       # warmup; budget cap still bounds

        if self._has_plus2(candidates):
            consequential = True

        self._lm_hist.append(lm)
        self._uv_hist.append(uv)
        for h in (self._lm_hist, self._uv_hist):
            if len(h) > self.window:
                h.pop(0)

        if not consequential:
            return False
        return (self.n_reviews < self.rho * self.n_reviewable) or (self.n_reviews == 0)
