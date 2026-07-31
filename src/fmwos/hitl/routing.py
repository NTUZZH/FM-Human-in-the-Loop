"""W1: deployable review routing via a calibrated uncertainty band.

The headline review policy of the current manuscript (``Supervisor`` with
``mechanism="targeted"``) is NOT deployable: its consequential test contains
``_has_plus2``, which reads the realized latent shift of the pending queue. No
real site can run that protocol. This module replaces it with a routing rule
computable from OBSERVABLES ALONE, and keeps the oracle-informed policy in one
role only, a labelled upper reference.

The three pieces, in the order they compose:

1. **The corrected ATC index and its monotonicity.** The augmented rule scores a
   candidate at the corrected class ``c_hat = clip(c - s, 1, 4)``, which moves
   the weight up and the deadline earlier as ``s`` grows. Both factors push the
   same way, so the index is non-decreasing in ``s``
   (``corrected_atc_index``; proved numerically by ``tests/test_routing.py``).
   This is what makes the stability test below exact rather than a heuristic.

2. **A per-order uncertainty band from split conformal prediction**
   (``calibrate_band``). The estimator is fitted on a proper-training split of
   the override-derived weak-label examples; absolute residuals on a DISJOINT
   calibration split give the ``(1 - alpha)`` empirical quantile ``q``, and the
   band is ``[s_hat - q, s_hat + q]`` (clipped to the protocol's shift range).

   **The correctness point of this package.** The only labels a deployed system
   has are the override-derived weak labels, so the band targets coverage of the
   WEAK labels, not of the latent. The simulator's true shift is used for
   exactly one thing: evaluating empirical coverage as a reported result. That
   is enforced structurally, not promised:

   * ``calibrate_band`` and ``fit_band_from_examples`` have NO parameter that
     could carry an overlay, an applied-overlay dict, or any latent quantity;
   * ``_forbid_latent`` rejects, at run time, any overlay-like object or any
     mapping carrying a latent key that reaches them anyway;
   * the calibration labels are asserted to lie in the weak-label alphabet
     ``{-1, 0, +1}``, which the latent shift (range ``{-2..+2}``) violates;
   * ``tests/test_routing.py`` greps the source of every calibration and routing
     function for latent tokens and fails on a hit.

   The coverage EVALUATION lives outside this module, in
   ``scripts/y3_w1_band_coverage.py``, so nothing in the calibration path can
   reach the latent even by accident.

3. **The decision-stability test** (``stability_verdict``). At a dispatch event
   with feasible set ``Q``, let ``p`` be the top pick under the corrected index.
   The pick is STABLE if no admissible shift vector inside the per-order bands
   changes the argmax; otherwise the decision is UNDETERMINED. Because each
   order's index depends only on its own shift and is monotone in it, and the
   bands form a box, the exact test is pairwise: evaluate ``p`` at its LEAST
   urgent band end and every rival at its MOST urgent band end. The INSTABILITY
   MARGIN is the smallest such pairwise gap, so decisions can be ranked by how
   close they are to flipping.

The test then does two jobs. ``StabilityRoutingSupervisor`` spends the review
budget ``rho`` on undetermined decisions in order of instability margin, which
is the deployable headline policy; and the same verdict is the per-decision
automate/refer output the paper ships.

``MarginRoutingSupervisor`` is the third policy in the comparison: the
observable half of TARGETED (the margin quantile, with the latent ``_has_plus2``
clause deleted). It answers the obvious reviewer question of whether the
stability test earns its keep against the trivial observable proxy.

Nothing here forks the existing code. The index reuses
``augmented_rule.interp_weight`` / ``corrected_deadline``; the supervisors
subclass ``Supervisor`` and override only ``_decide_review``; ``run_m0_routed``
drives the same ``augmented_atc_decider`` / ``weak_labels_from_log`` /
``train_estimator`` / ``probe_shift_accuracy`` calls as ``augmented_rule.run_m0``
in the same order, and reproduces it bit-for-bit at ``policy="targeted",
split_fit=False`` (asserted by ``tests/test_routing.py``).
"""

from __future__ import annotations

import math

import numpy as np

from ..env import DispatchEnv
from .augmented_rule import (augmented_atc_decider, hat_s_map, interp_weight,
                             corrected_deadline, probe_shift_accuracy,
                             weak_labels_from_log)
from .latent_head import LAT_DIM, ShiftEstimator, train_estimator
from .supervisor import Supervisor, _ATC_K

# The class-shift scale is capped at +/- 2 by the overlay protocol (a design
# constant of the priority scale, stated in the paper, not a latent read), so an
# admissible shift never leaves this interval and the band is clipped to it.
SHIFT_LO, SHIFT_HI = -2.0, 2.0

# The weak-label alphabet produced by ``weak_labels_from_log``: +1 promoted,
# -1 demoted, 0 confirmation. The latent shift is NOT confined to this set.
WEAK_LABEL_ALPHABET = (-1.0, 0.0, 1.0)

ROUTING_POLICIES = ("stability", "margin", "targeted", "random")

# Keys/attributes that would smuggle a latent quantity into a calibration call.
_LATENT_KEYS = ("shift", "w_star", "d_star", "c_star", "xi", "per_order",
                "s_true", "applied", "overlay")


# --------------------------------------------------------------------------- #
# 0. Structural guard: nothing latent may reach the calibration path           #
# --------------------------------------------------------------------------- #
def _forbid_latent(*objs):
    """Raise if any argument is an overlay, an applied-overlay dict, or a
    mapping carrying a latent key. Called at the head of every calibration
    entry point, so the "calibration never sees the latent" claim is checked at
    run time rather than asserted in a comment."""
    for o in objs:
        if o is None or isinstance(o, (np.ndarray, int, float, str, bool)):
            continue
        if hasattr(o, "apply") and hasattr(o, "params"):
            raise TypeError("routing: an overlay-like object reached a "
                            "calibration function; the band may only be fitted "
                            "on override-derived weak labels")
        if isinstance(o, dict):
            bad = sorted(set(o) & set(_LATENT_KEYS))
            if bad:
                raise ValueError("routing: latent key(s) %r reached a "
                                 "calibration function" % (bad,))


def _assert_weak_labels(y):
    """Assert every calibration label is an override-derived weak label.

    The weak labels take values in {-1, 0, +1}. The simulator's latent shift
    takes values in {-2, ..., +2}, so a latent label leaks through this check
    whenever any order carries a +/-2 shift, which is the common case on the
    headline cells. It is a tripwire, not a proof; the source-level test in
    ``tests/test_routing.py`` is the proof.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return
    ok = np.isin(y, np.asarray(WEAK_LABEL_ALPHABET))
    if not bool(ok.all()):
        raise ValueError("routing: calibration labels outside the weak-label "
                         "alphabet %r (saw %r); the band must be calibrated on "
                         "override-derived labels only"
                         % (WEAK_LABEL_ALPHABET,
                            sorted(set(np.asarray(y)[~ok].tolist()))[:5]))


# --------------------------------------------------------------------------- #
# 1. The corrected ATC index (identical to the augmented rule's own score)      #
# --------------------------------------------------------------------------- #
# Scalar fast paths for the two interpolated curves. The grids are the unit grid
# [1,2,3,4], so linear interpolation reduces to one multiply-add, which is
# BIT-IDENTICAL to ``np.interp`` on this grid (checked over 4e5 points, including
# the knots and their neighbours, by tests/test_routing.py). Worth the duplicate
# because the stability test evaluates the curves a few hundred million times
# across a sweep.
_W_V = (8.0, 4.0, 2.0, 1.0)
_S_V = (8.0, 24.0, 80.0, 171.4)

# Band-end index (0 = least urgent end, 1 = point estimate, 2 = most urgent end)
# into a plain ``(s_hat, s_lo, s_hi)`` band tuple.
_END2TUP = (1, 0, 2)


def _interp_unit(c_eff, V):
    c = c_eff
    if c < 1.0:
        c = 1.0
    elif c > 4.0:
        c = 4.0
    i = int(c) - 1
    if i > 2:
        i = 2
    return V[i] + (c - (i + 1.0)) * (V[i + 1] - V[i])


def index_terms(c, p, release_bh, s, channel="full_class_shift",
                due_recorded=None):
    """The shift-dependent half of the index, ``(w/p, due)``, at one shift.

    Precomputing this per order and per band end turns the per-decision work
    into one ``exp`` and three flops. Identical values to
    ``interp_weight`` / ``corrected_deadline``.
    """
    c_eff = float(c) - float(s)
    w = _interp_unit(c_eff, _W_V)
    if channel == "full_class_shift":
        due = float(release_bh) + _interp_unit(c_eff, _S_V)
    elif channel == "weight_only":
        due = float(due_recorded)
    else:
        raise ValueError("channel must be full_class_shift or weight_only")
    return w / float(p), due


def corrected_atc_index(c, p, release_bh, s, t, denom, channel="full_class_shift",
                        due_recorded=None) -> float:
    """The augmented rule's per-candidate score at shift ``s``.

    Bit-for-bit the expression inside ``augmented_rule.augmented_atc_decider``:

        w_corr = w(clip(c - s, 1, 4))
        d_corr = r + SLA(clip(c - s, 1, 4))        [full_class_shift]
        score  = (w_corr / p) * exp(-max(0, d_corr - t - p) / denom)

    ``denom = k * pbar`` is supplied by the caller because ``pbar`` is the mean
    processing time over the candidate list and does NOT depend on any shift,
    which is what keeps the index separable across orders.
    """
    w = interp_weight(float(c) - float(s))
    if channel == "full_class_shift":
        due = corrected_deadline(c, s, release_bh)
    elif channel == "weight_only":
        due = float(due_recorded)
    else:
        raise ValueError("channel must be full_class_shift or weight_only")
    slack = max(0.0, due - t - p)
    return (w / p) * math.exp(-slack / denom)


def index_slope_check(c, p, release_bh, t, denom, shifts,
                      channel="full_class_shift", due_recorded=None):
    """Return the vector of index values over ``shifts`` (ascending), for the
    monotonicity test. A separate helper so the test never re-derives the
    formula."""
    return np.asarray([corrected_atc_index(c, p, release_bh, s, t, denom,
                                           channel=channel,
                                           due_recorded=due_recorded)
                       for s in shifts], dtype=np.float64)


# --------------------------------------------------------------------------- #
# 2. Split-conformal band on the override-derived weak labels                  #
# --------------------------------------------------------------------------- #
def conformal_quantile(residuals, alpha=0.1) -> float:
    """Finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))-th
    smallest absolute residual (``+inf`` when n is too small for the level)."""
    r = np.sort(np.asarray(residuals, dtype=np.float64))
    n = r.size
    if n == 0:
        return float("inf")
    k = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    if k > n:
        return float("inf")            # not enough calibration points at alpha
    return float(r[k - 1])


def _trade_group(X):
    """Group index per example = argmax of the trade one-hot block.

    ``overlay.base_features`` lays the vector out as [14 trade one-hot |
    log1p(p) | 5 day one-hot], so the first 14 columns identify the trade. Used
    only by the locally-adaptive band; reads a FEATURE, never a latent."""
    X = np.asarray(X)
    return np.argmax(X[:, :14], axis=1).astype(int)


class ConformalBand:
    """A calibrated half-width for the estimated shift.

    ``mode="global"``: one half-width ``q`` for every order (plain split
    conformal). ``mode="normalized"``: locally adaptive, ``q * sigma_g`` with a
    per-trade scale ``sigma_g`` fitted on the proper-training residuals, which
    is the cheap normalised variant of split conformal.
    """

    def __init__(self, q, alpha, mode="global", scale=None, default_scale=1.0,
                 n_cal=0, n_prop=0):
        self.q = float(q)
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.scale = dict(scale or {})
        self.default_scale = float(default_scale)
        self.n_cal = int(n_cal)
        self.n_prop = int(n_prop)

    # -- per-example half-width -------------------------------------------- #
    def half_width(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.size == 0:
            return np.zeros((0,), dtype=np.float64)
        if self.mode == "global":
            return np.full(X.shape[0], self.q, dtype=np.float64)
        g = _trade_group(X)
        sig = np.asarray([self.scale.get(int(k), self.default_scale) for k in g],
                         dtype=np.float64)
        return self.q * sig

    def as_dict(self):
        return {"q": self.q, "alpha": self.alpha, "mode": self.mode,
                "n_cal": self.n_cal, "n_prop": self.n_prop,
                "default_scale": self.default_scale,
                "scale": {str(k): float(v) for k, v in self.scale.items()}}

    def __repr__(self):                                    # pragma: no cover
        return "ConformalBand(q=%.4f, alpha=%.2f, mode=%s, n_cal=%d)" % (
            self.q, self.alpha, self.mode, self.n_cal)


def calibrate_band(estimator, X_cal, y_cal, alpha=0.1, device="cpu",
                   X_prop=None, y_prop=None, mode="global",
                   scale_floor=0.25) -> ConformalBand:
    """Split-conformal half-width from override-derived weak labels ONLY.

    Parameters are deliberately restricted to (a fitted estimator, a feature
    matrix, a weak-label vector). There is no parameter through which an
    overlay, an applied-overlay dict, an instance, or any latent quantity could
    enter, and ``_forbid_latent`` rejects one that is passed positionally
    anyway.

    ``X_cal`` / ``y_cal`` must come from the CALIBRATION fold, disjoint from
    every example the estimator was ever fitted on. ``X_prop`` / ``y_prop`` (the
    proper-training fold) are needed only by ``mode="normalized"``, to fit the
    per-trade residual scale without touching the calibration fold.
    """
    _forbid_latent(estimator, X_cal, y_cal, X_prop, y_prop)
    _assert_weak_labels(y_cal)
    X_cal = np.asarray(X_cal, dtype=np.float32)
    y_cal = np.asarray(y_cal, dtype=np.float64)
    if X_cal.shape[0] == 0:
        return ConformalBand(float("inf"), alpha, mode=mode, n_cal=0)

    res = np.abs(estimator.predict_np(X_cal, device=device).astype(np.float64) - y_cal)

    if mode == "global":
        q = conformal_quantile(res, alpha)
        return ConformalBand(q, alpha, mode="global", n_cal=int(res.size),
                             n_prop=0 if X_prop is None else int(len(X_prop)))

    if mode != "normalized":
        raise ValueError("mode must be 'global' or 'normalized'")

    # Locally-adaptive: per-trade mean absolute residual on the PROPER-TRAINING
    # fold is the scale; the conformal quantile is taken on the normalised
    # calibration residuals, which keeps the split-conformal guarantee.
    if X_prop is None or len(X_prop) == 0:
        raise ValueError("mode='normalized' needs the proper-training fold")
    _assert_weak_labels(y_prop)
    X_prop = np.asarray(X_prop, dtype=np.float32)
    y_prop = np.asarray(y_prop, dtype=np.float64)
    rp = np.abs(estimator.predict_np(X_prop, device=device).astype(np.float64) - y_prop)
    gp = _trade_group(X_prop)
    default = float(max(rp.mean(), scale_floor)) if rp.size else 1.0
    scale = {}
    for k in np.unique(gp):
        m = (gp == k)
        if int(m.sum()) >= 8:
            scale[int(k)] = float(max(rp[m].mean(), scale_floor))
    sig_cal = np.asarray([scale.get(int(k), default) for k in _trade_group(X_cal)],
                         dtype=np.float64)
    q = conformal_quantile(res / np.maximum(sig_cal, 1e-9), alpha)
    return ConformalBand(q, alpha, mode="normalized", scale=scale,
                         default_scale=default, n_cal=int(res.size),
                         n_prop=int(rp.size))


def fit_band_from_examples(estimator, X, y, folds, alpha=0.1, device="cpu",
                           mode="global") -> ConformalBand:
    """``calibrate_band`` on a pre-assigned fold vector (0 = proper training,
    1 = calibration). Same latent-free contract."""
    _forbid_latent(estimator, X, y, folds)
    folds = np.asarray(folds).astype(int)
    cal = folds == 1
    prop = folds == 0
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    return calibrate_band(estimator, X[cal], y[cal], alpha=alpha, device=device,
                          X_prop=X[prop], y_prop=y[prop], mode=mode)


class OrderBands:
    """Per-order shift band for one instance, with the index terms precomputed.

    ``bands[wo_id]`` gives ``(s_hat, s_lo, s_hi)``; ``terms[wo_id]`` gives the
    six numbers ``(wp_lo, d_lo, wp_hat, d_hat, wp_hi, d_hi)`` the stability test
    needs, so a dispatch event costs one ``exp`` per candidate.
    """

    def __init__(self, bands, terms):
        self.bands = bands
        self.terms = terms

    @classmethod
    def from_band_map(cls, work_orders, band_map, channel="full_class_shift"):
        """Precompute the index terms for an explicit ``(s_hat, s_lo, s_hi)``
        map. Used by the tests to check the precomputed and on-the-fly paths
        against each other."""
        bands, terms = {}, {}
        for w in work_orders:
            wid = w["id"]
            s_hat, lo, hi = band_map[wid]
            c, p, r, dr = int(w["priority"]), float(w["p_bh"]), \
                float(w["release_bh"]), float(w["due_bh"])
            bands[wid] = (s_hat, lo, hi)
            terms[wid] = (index_terms(c, p, r, lo, channel, dr)
                          + index_terms(c, p, r, s_hat, channel, dr)
                          + index_terms(c, p, r, hi, channel, dr))
        return cls(bands, terms)

    def __contains__(self, k):
        return k in self.bands

    def get(self, k, default=None):
        return self.bands.get(k, default)

    def __getitem__(self, k):
        return self.bands[k]


def band_for_instance(estimator, instance, band, device="cpu",
                      channel="full_class_shift") -> OrderBands:
    """Per-order ``(s_hat, s_lo, s_hi)`` for one instance, from observables.

    ``s_hat`` is the estimator's prediction on ``overlay.base_features`` (trade,
    log processing time, release weekday), exactly the vector the augmented
    decider uses; the half-width comes from the calibrated band. The band is
    clipped to the protocol's shift range, a stated design constant.
    """
    from .overlay import base_features                     # observable features
    wos = instance["work_orders"]
    feats = np.stack([base_features(w) for w in wos]).astype(np.float32)
    s_hat = estimator.predict_np(feats, device=device).astype(np.float64)
    if band is None:
        hw = np.full(s_hat.shape, float("inf"))
    else:
        hw = band.half_width(feats)
    lo = np.clip(s_hat - hw, SHIFT_LO, SHIFT_HI)
    hi = np.clip(s_hat + hw, SHIFT_LO, SHIFT_HI)
    bands, terms = {}, {}
    for i, w in enumerate(wos):
        wid = w["id"]
        c, p, r, dr = int(w["priority"]), float(w["p_bh"]), \
            float(w["release_bh"]), float(w["due_bh"])
        bands[wid] = (float(s_hat[i]), float(lo[i]), float(hi[i]))
        terms[wid] = (index_terms(c, p, r, lo[i], channel, dr)
                      + index_terms(c, p, r, s_hat[i], channel, dr)
                      + index_terms(c, p, r, hi[i], channel, dr))
    return OrderBands(bands, terms)


# --------------------------------------------------------------------------- #
# 3. The decision-stability test                                               #
# --------------------------------------------------------------------------- #
def stability_verdict(candidates, t, band_map, channel="full_class_shift",
                      pick_id=None, k=_ATC_K, need_top=False):
    """Exact stability test for one dispatch event.

    ``candidates``  the feasible set, as work-order dicts (observable fields
                    only: id, priority, p_bh, release_bh, due_bh).
    ``band_map``    wo_id -> (s_hat, s_lo, s_hi).
    ``pick_id``     the decision under test; defaults to the top pick under the
                    corrected index at ``s_hat`` (which is exactly what the
                    augmented decider dispatches).

    Returns a dict with the top pick, the pick under test, ``stable``, the
    ``margin`` (the smallest pairwise gap; positive == stable, and ``+inf`` for
    a forced pick), and the id of the rival that comes closest to overturning
    the pick.

    Exactness. Each order's corrected index depends only on its own shift and is
    non-decreasing in it (see ``corrected_atc_index``), and the admissible set
    is a box, so the worst case for the comparison ``p`` versus ``j`` is
    ``s_p = s_lo(p)`` together with ``s_j = s_hi(j)``, which is itself
    admissible. Checking those corners for every rival is therefore not a
    relaxation but the exact condition. Exact ties count as UNDETERMINED, which
    is conservative: the id tie-break could go either way under an arbitrarily
    small perturbation.
    """
    n = len(candidates)
    if n == 0:
        raise ValueError("empty feasible set")
    pbar = sum(float(c["p_bh"]) for c in candidates) / n
    denom = k * pbar
    fast = getattr(band_map, "terms", None)

    def terms(c, end):
        """``(w/p, due)`` at band end 0=lo, 1=hat, 2=hi for candidate ``c``.

        ``OrderBands.terms`` stores the three ends in that order; a plain
        ``band_map`` dict stores ``(s_hat, s_lo, s_hi)``, hence ``_END2TUP``.
        """
        if fast is not None:
            tt = fast.get(c["id"])
            if tt is not None:
                return tt[2 * end], tt[2 * end + 1]
        s = band_map.get(c["id"], (0.0, 0.0, 0.0))[_END2TUP[end]]
        return index_terms(int(c["priority"]), float(c["p_bh"]),
                           float(c["release_bh"]), s, channel,
                           float(c["due_bh"]))

    def idx(c, end):
        wp, due = terms(c, end)
        slack = due - t - float(c["p_bh"])
        if slack <= 0.0:
            return wp
        return wp * math.exp(-slack / denom)

    if n == 1:
        return {"top": candidates[0]["id"], "pick": candidates[0]["id"],
                "stable": True, "margin": float("inf"), "rival": None,
                "n_cand": 1, "forced": True}

    if pick_id is None or need_top:
        # Top pick under the corrected index at s_hat; id tie-break, matching
        # ``augmented_atc_decider``'s ``sort(key=(-score, id))``.
        best_key = None
        for c in candidates:
            key = (-idx(c, 1), c["id"])
            if best_key is None or key < best_key:
                best_key, top_id = key, c["id"]
    else:
        top_id = None

    pid = top_id if pick_id is None else pick_id
    pc = next((c for c in candidates if c["id"] == pid), None)
    if pc is None:
        raise ValueError("pick %r is not in the feasible set" % (pid,))

    lo_p = idx(pc, 0)                              # p at its LEAST urgent end
    worst = float("inf")
    rival = None
    for c in candidates:
        if c["id"] == pid:
            continue
        gap = lo_p - idx(c, 2)                     # rival at its MOST urgent end
        if gap < worst:
            worst, rival = gap, c["id"]
    return {"top": top_id, "pick": pid, "stable": bool(worst > 0.0),
            "margin": float(worst), "rival": rival, "n_cand": n,
            "forced": False}


def brute_force_stable(candidates, t, band_map, channel="full_class_shift",
                       n_grid=7, k=_ATC_K, pick_id=None):
    """Reference implementation for the unit test: does the argmax ever change
    over a dense grid of admissible shift vectors? O(n_grid ** |Q|), so only for
    tiny synthetic queues."""
    import itertools
    n = len(candidates)
    pbar = sum(float(c["p_bh"]) for c in candidates) / n
    denom = k * pbar

    def idx(c, s):
        return corrected_atc_index(int(c["priority"]), float(c["p_bh"]),
                                   float(c["release_bh"]), s, t, denom,
                                   channel=channel,
                                   due_recorded=float(c["due_bh"]))

    if pick_id is None:
        pick_id = stability_verdict(candidates, t, band_map, channel=channel,
                                    k=k)["top"]
    grids = []
    for c in candidates:
        _s, lo, hi = band_map[c["id"]]
        grids.append(np.linspace(lo, hi, n_grid))
    for combo in itertools.product(*grids):
        scores = [(-idx(c, s), c["id"]) for c, s in zip(candidates, combo)]
        win = min(scores)[1]
        if win != pick_id:
            return False
    return True


# --------------------------------------------------------------------------- #
# 4. Routing supervisors                                                       #
# --------------------------------------------------------------------------- #
class StabilityRoutingSupervisor(Supervisor):
    """The DEPLOYABLE review policy: refer undetermined decisions, worst first.

    Every input to the routing decision is an observable: the feasible set's
    recorded fields, the estimator's ``s_hat`` (fitted on override-derived weak
    labels), and the conformal half-width (calibrated on the same labels). The
    inherited override behaviour is unchanged -- that is the SIMULATED HUMAN,
    which of course knows the truth; the routing logic below never touches
    ``self.shift``, ``self.wstar``, ``self.applied`` or ``self.overlay``.

    Placement. A forced pick (one feasible candidate) is never reviewed and
    never counts against the budget, exactly as in TARGETED. A decision is
    consequential when it is UNDETERMINED and its instability margin sits in the
    worst ``rho`` fraction of a rolling window of margins, so the budget lands
    on the decisions closest to flipping. The same online budget controller as
    TARGETED then caps the realised reviewed fraction at ``rho``. When the
    undetermined rate is below ``rho`` the policy spends LESS than its budget,
    which is the automation coverage the routing curve reports.

    Cold start. Before any calibration data exists (the first DAgger iteration)
    ``band`` is ``None``; there is then no certified stable decision, and the
    policy falls back to the observable margin quantile, which is TARGETED
    minus its latent clause.
    """

    def __init__(self, overlay, instance, rho, epsilon=0.0, theta=1.0,
                 seed=0, window=64, applied=None, band_map=None,
                 channel="full_class_shift", record_verdicts=False,
                 max_records=0):
        super().__init__(overlay, instance, rho=rho, epsilon=epsilon,
                         theta=theta, mechanism="targeted", seed=seed,
                         window=window, applied=applied)
        self.mechanism = "stability"
        self.routing_channel = channel
        self.band_map = band_map          # wo_id -> (s_hat, s_lo, s_hi); None = cold
        self._now = 0.0
        self._pick = None
        self._imargins = []
        self.n_undetermined = 0
        self.n_stable = 0
        self.n_forced = 0
        self.n_pick_ne_top = 0
        self.record_verdicts = bool(record_verdicts)
        self.max_records = int(max_records)
        self.verdicts = []

    # -- routing (OBSERVABLES ONLY) ---------------------------------------- #
    def _decide_review(self, margin, candidates):
        if self.rho <= 0.0:
            return False
        if len(candidates) < 2:
            self.n_forced += 1
            return False
        self.n_reviewable += 1

        if self.band_map is None:
            # Cold start: no band yet, so nothing is certified; rank by the
            # observable top1-top2 margin (TARGETED minus its latent clause).
            imargin = float(margin)
            undetermined = True
        else:
            want_top = self.record_verdicts and len(self.verdicts) < self.max_records
            v = stability_verdict(candidates, self._now, self.band_map,
                                  channel=self.routing_channel,
                                  pick_id=self._pick, need_top=want_top)
            imargin = v["margin"]
            undetermined = not v["stable"]
            if want_top and v["top"] != v["pick"]:
                self.n_pick_ne_top += 1
            if self.record_verdicts and len(self.verdicts) < self.max_records:
                self.verdicts.append({"t": self._now, "margin": imargin,
                                      "stable": bool(v["stable"]),
                                      "pick": v["pick"], "top": v["top"],
                                      "rival": v["rival"],
                                      "n_cand": v["n_cand"]})
        if undetermined:
            self.n_undetermined += 1
        else:
            self.n_stable += 1

        # Rolling-window rank: the worst rho fraction of margins competes for the
        # budget. Undetermined decisions carry the smallest margins, so the
        # quantile picks them first; the explicit conjunct below guarantees a
        # STABLE decision is never reviewed even if the window is degenerate.
        if len(self._imargins) >= 8:
            finite = [m for m in self._imargins if math.isfinite(m)]
            thr = float(np.percentile(finite, 100.0 * self.rho)) if finite else float("inf")
            prioritized = (imargin <= thr)
        else:
            prioritized = True                 # warmup; the budget cap still bounds
        self._imargins.append(imargin)
        if len(self._imargins) > self.window:
            self._imargins.pop(0)

        if not (undetermined and prioritized):
            return False
        return (self.n_reviews < self.rho * self.n_reviewable) or (self.n_reviews == 0)

    # The env passes ``now`` and the decider's pick to ``review``; capture both
    # so ``_decide_review`` can evaluate the index at the right clock, for the
    # decision actually about to be executed, without changing the parent's API.
    def review(self, decider_pick, candidates, now, margin):
        self._now = float(now)
        self._pick = decider_pick["id"]
        return super().review(decider_pick, candidates, now, margin)

    def routing_summary(self):
        n = self.n_undetermined + self.n_stable
        s = self.summary()
        s.update({
            "n_stable": self.n_stable, "n_undetermined": self.n_undetermined,
            "n_forced": self.n_forced,
            "undetermined_rate": (self.n_undetermined / n) if n else 0.0,
            "automation_coverage_all": (1.0 - self.n_reviews / self.n_decisions)
            if self.n_decisions else 1.0,
            "automation_coverage_reviewable": (1.0 - self.n_reviews / self.n_reviewable)
            if self.n_reviewable else 1.0,
            "policy": "stability", "cold_start": self.band_map is None,
        })
        return s


class MarginRoutingSupervisor(Supervisor):
    """Observable control: TARGETED with the latent ``_has_plus2`` clause removed.

    Deployable, but blind to how far a decision is from flipping under the
    estimator's own uncertainty. Included so the headline claim is tested
    against the trivial observable proxy, not only against RANDOM.
    """

    def __init__(self, overlay, instance, rho, epsilon=0.0, theta=1.0, seed=0,
                 window=64, applied=None):
        super().__init__(overlay, instance, rho=rho, epsilon=epsilon,
                         theta=theta, mechanism="targeted", seed=seed,
                         window=window, applied=applied)
        self.mechanism = "margin"

    def _decide_review(self, margin, candidates):
        if self.rho <= 0.0:
            return False
        if len(candidates) < 2:
            return False
        self.n_reviewable += 1
        if len(self._margins) >= 8:
            thr = float(np.percentile(self._margins, 100.0 * self.rho))
            consequential = (margin <= thr)
        else:
            consequential = True
        self._margins.append(float(margin))
        if len(self._margins) > self.window:
            self._margins.pop(0)
        if not consequential:
            return False
        return (self.n_reviews < self.rho * self.n_reviewable) or (self.n_reviews == 0)

    def routing_summary(self):
        s = self.summary()
        s.update({"policy": "margin",
                  "automation_coverage_all": (1.0 - self.n_reviews / self.n_decisions)
                  if self.n_decisions else 1.0,
                  "automation_coverage_reviewable":
                      (1.0 - self.n_reviews / self.n_reviewable)
                      if self.n_reviewable else 1.0})
        return s


def make_supervisor(policy, overlay, instance, rho, *, epsilon=0.0, theta=1.0,
                    seed=0, applied=None, band_map=None,
                    channel="full_class_shift", record_verdicts=False,
                    max_records=0):
    """One constructor for the four review policies compared in W1.

    ``stability`` deployable headline; ``margin`` observable control;
    ``targeted`` the ORACLE-INFORMED UPPER REFERENCE (its consequential test
    reads the realized latent through ``_has_plus2``, so it is retained for
    reference only, never as a deployable protocol); ``random`` lower control.
    """
    if policy == "stability":
        return StabilityRoutingSupervisor(
            overlay, instance, rho, epsilon=epsilon, theta=theta, seed=seed,
            applied=applied, band_map=band_map, channel=channel,
            record_verdicts=record_verdicts, max_records=max_records)
    if policy == "margin":
        return MarginRoutingSupervisor(overlay, instance, rho, epsilon=epsilon,
                                       theta=theta, seed=seed, applied=applied)
    if policy in ("targeted", "random"):
        return Supervisor(overlay, instance, rho=rho, epsilon=epsilon,
                          theta=theta, mechanism=policy, seed=seed,
                          applied=applied)
    raise ValueError("policy must be one of %r" % (ROUTING_POLICIES,))


def routing_summary_of(sup):
    """``routing_summary`` for the routed supervisors, a compatible dict for the
    two stock mechanisms."""
    if hasattr(sup, "routing_summary"):
        return sup.routing_summary()
    s = sup.summary()
    s.update({"policy": sup.mechanism,
              "automation_coverage_all": (1.0 - sup.n_reviews / sup.n_decisions)
              if sup.n_decisions else 1.0,
              "automation_coverage_reviewable":
                  (1.0 - sup.n_reviews / sup.n_reviewable)
                  if sup.n_reviewable else 1.0})
    return s


# --------------------------------------------------------------------------- #
# 5. The M0 pipeline under a routing policy                                    #
# --------------------------------------------------------------------------- #
def _assign_folds(n, cal_frac, rng):
    """Permanent fold assignment for freshly created weak-label examples.

    An example assigned to the calibration fold is NEVER used to fit the
    estimator, in this or any later DAgger iteration, so the conformal residuals
    are computed on examples the estimator has never seen. Assigning the fold at
    creation (rather than re-splitting the aggregate each iteration) is what
    makes that true under the warm-started, aggregate-retrained protocol the
    published pipeline uses.
    """
    if n == 0:
        return np.zeros((0,), dtype=int)
    return (rng.random(n) < float(cal_frac)).astype(int)


def run_m0_routed(train_instances, probe_instances, overlay, *, beta_rho_eps,
                  outer_iters=8, episodes_per_iter=None, policy="targeted",
                  theta=1.0, override_weight=5.0, confirm_weight=1.0,
                  est_hidden=32, seed=0, device="cpu", verbose=False,
                  split_fit=True, cal_frac=0.3, alpha=0.1, band_mode="global",
                  probe=True):
    """``augmented_rule.run_m0`` with the review policy swapped out.

    Same 8 outer DAgger iterations, same episode rotation, same never-reset
    aggregate, same warm-started estimator, same ``train_estimator`` call, in the
    same order, so the ONLY difference against the published pipeline is the
    review policy (plus the conformal fold split when ``split_fit=True``).

    ``split_fit=False`` reproduces ``run_m0`` bit-for-bit at
    ``policy="targeted"`` and returns no band.

    Returns ``{"estimator", "band", "per_iter", "config"}``.
    """
    if policy not in ROUTING_POLICIES:
        raise ValueError("policy must be one of %r" % (ROUTING_POLICIES,))
    beta, rho, eps = beta_rho_eps
    channel = getattr(overlay.params, "channel", "full_class_shift")
    estimator = ShiftEstimator(hidden=est_hidden)

    Xagg = np.zeros((0, LAT_DIM), np.float32)
    yagg = np.zeros((0,), np.float32)
    wagg = np.zeros((0,), np.float32)
    fagg = np.zeros((0,), int)                     # 0 proper-training, 1 calibration
    per_iter = []
    rng = np.random.default_rng(seed)
    fold_rng = np.random.default_rng(int(seed) + 90210)
    n_ep = episodes_per_iter or len(train_instances)
    band = None

    for it in range(outer_iters):
        order = rng.permutation(len(train_instances))[:n_ep]
        n_over = n_rev = n_conf = 0
        n_und = n_stab = 0
        for kk in order:
            inst = train_instances[int(kk)]
            applied = overlay.apply(inst)          # the SIMULATED HUMAN's own truth
            band_map = None
            if policy == "stability" and band is not None:
                band_map = band_for_instance(estimator, inst, band, device=device)
            sup = make_supervisor(policy, overlay, inst, rho, epsilon=eps,
                                  theta=theta, seed=seed, applied=applied,
                                  band_map=band_map, channel=channel)
            decider = augmented_atc_decider(estimator, inst, device=device,
                                            channel=channel)
            _sched, log = DispatchEnv(inst).run_supervised(
                decider, supervisor=sup, method="m0_atc", seed=seed)
            X, y, w = weak_labels_from_log(log, inst, override_weight, confirm_weight)
            if len(X):
                f = _assign_folds(len(X), cal_frac, fold_rng) if split_fit \
                    else np.zeros(len(X), int)
                Xagg = np.concatenate([Xagg, X]); yagg = np.concatenate([yagg, y])
                wagg = np.concatenate([wagg, w]); fagg = np.concatenate([fagg, f])
            s = routing_summary_of(sup)
            n_over += s["n_overrides"]; n_rev += s["n_reviews"]
            n_conf += s["n_confirmations"]
            n_und += s.get("n_undetermined", 0); n_stab += s.get("n_stable", 0)

        # Fit on the proper-training fold only (all of it when split_fit=False).
        m = (fagg == 0)
        loss = train_estimator(estimator, Xagg[m], yagg[m], wagg[m],
                               device=device, seed=seed + it)
        if split_fit:
            band = fit_band_from_examples(estimator, Xagg, yagg, fagg,
                                          alpha=alpha, device=device,
                                          mode=band_mode)
        acc = probe_shift_accuracy(estimator, probe_instances, overlay,
                                   device=device) if probe else {}
        row = {"iter": it, "n_reviews": n_rev, "n_overrides": n_over,
               "n_confirmations": n_conf,
               "override_rate": (n_over / n_rev) if n_rev else 0.0,
               "n_examples_agg": int(len(Xagg)),
               "n_examples_fit": int(m.sum()),
               "n_examples_cal": int((~m).sum()),
               "undetermined_rate_train": (n_und / (n_und + n_stab))
               if (n_und + n_stab) else float("nan"),
               "band_q": (band.q if band is not None else float("nan")),
               "est_loss": loss, **acc}
        per_iter.append(row)
        if verbose:
            print("[m0/%s it%d] rev=%d over=%d orr=%.3f fit=%d cal=%d q=%s "
                  "sign=%.3f r=%.3f"
                  % (policy, it, n_rev, n_over, row["override_rate"],
                     row["n_examples_fit"], row["n_examples_cal"],
                     ("%.3f" % band.q) if band is not None else "-",
                     acc.get("sign_acc_nonzero", float("nan")),
                     acc.get("pearson_r", float("nan"))), flush=True)

    return {"estimator": estimator, "band": band, "per_iter": per_iter,
            "config": {"policy": policy, "split_fit": bool(split_fit),
                       "cal_frac": float(cal_frac), "alpha": float(alpha),
                       "band_mode": band_mode, "outer_iters": int(outer_iters),
                       "channel": channel, "beta": beta, "rho": rho, "eps": eps,
                       "theta": theta, "seed": int(seed)}}


# --------------------------------------------------------------------------- #
# 6. Per-decision automate/refer verdicts (the shipped output)                 #
# --------------------------------------------------------------------------- #
def verdict_stream(estimator, instance, band, channel="full_class_shift",
                   device="cpu", seed=0, max_records=None):
    """Run the augmented rule over one instance with no supervisor and record the
    automate/refer verdict at every dispatch event.

    This is the deliverable's per-decision output: for each event, the pick, its
    corrected class and interval, the verdict, and the instability margin. No
    latent quantity is read anywhere in this function.
    """
    band_map = band_for_instance(estimator, instance, band, device=device)
    by_id = {w["id"]: w for w in instance["work_orders"]}
    recs = []
    counts = {"automate": 0, "refer": 0, "forced": 0}

    def _entry(pick, margin, verdict):
        return {"reviewed": False, "override": False, "confirmation": False,
                "decider_pick": pick, "executed_pick": pick,
                "margin": float(margin), "verdict": verdict}

    class _Recorder:
        """A pass-through 'supervisor' that never overrides: it only observes."""
        def __init__(self):
            self.log = []

        def review(self, decider_pick, candidates, now, margin):
            pick = decider_pick["id"]
            if len(candidates) < 2:
                counts["forced"] += 1
                counts["automate"] += 1
                return decider_pick, _entry(pick, margin, "automate")
            v = stability_verdict(candidates, float(now), band_map,
                                  channel=channel, pick_id=pick, need_top=True)
            verdict = "automate" if v["stable"] else "refer"
            counts[verdict] += 1
            if max_records is None or len(recs) < max_records:
                s_hat, lo, hi = band_map[pick]
                c = int(by_id[pick]["priority"])
                recs.append({
                    "t_bh": float(now), "wo": pick,
                    "n_cand": v["n_cand"], "recorded_class": c,
                    "s_hat": s_hat, "s_lo": lo, "s_hi": hi,
                    "c_hat": float(min(4.0, max(1.0, c - s_hat))),
                    "c_hat_lo": float(min(4.0, max(1.0, c - hi))),
                    "c_hat_hi": float(min(4.0, max(1.0, c - lo))),
                    "margin": v["margin"], "rival": v["rival"],
                    "verdict": verdict})
            return decider_pick, _entry(pick, margin, verdict)

    dec_fn = augmented_atc_decider(estimator, instance, device=device,
                                   channel=channel)
    sched, _log = DispatchEnv(instance).run_supervised(
        dec_fn, supervisor=_Recorder(), method="m0_verdict", seed=seed)
    n = counts["automate"] + counts["refer"]
    return {"schedule": sched, "records": recs, "counts": counts,
            "automation_coverage": (counts["automate"] / n) if n else 1.0,
            "n_decisions": n}
