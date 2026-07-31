"""Simulated private-information supervisor (Paper Y3, Appendix D).

The supervisor is an oracle bound to one instance's overlay. It reviews a
budget-limited fraction of dispatch decisions and may override the decider's
pick toward the myopic-greedy true-objective optimum. It is DECIDER-AGNOSTIC:
it consumes only the decider's proposed pick and a scalar confidence margin.

Review mechanisms (both decider-agnostic, Appendix D.3):
* TARGETED  -- a decision is "consequential" if the decider's top-1/top-2 score
  margin is below the running 25th percentile of its own margins OR the pending
  queue holds an order with realized shift s = +2. Consequential decisions are
  reviewed subject to an online budget controller tuned so the episode's
  reviewed fraction tracks ``rho`` (an online stand-in for "top-k consequential
  per rolling window with k set so reviewed fraction = rho").
* RANDOM    -- iid Bernoulli(rho) per decision (attribution control, E5).

Override rule (Appendix D.4). On a reviewed decision the preferred pick is the
feasible candidate minimizing the ONE-STEP true-objective increment under the
realized latent. Override iff the improvement over the decider's pick exceeds
``theta``; EXCEPT with prob epsilon/2 the supervisor fails to override when it
should, and with prob epsilon/2 it overrides to a uniformly random eligible
pick. Every decision is logged; a reviewed-but-not-overridden decision is a
confirmation.

Myopic-greedy preferred pick. The standard one-step-greedy heuristic for
weighted tardiness is the Apparent Tardiness Cost (ATC) dispatcher (Vepsalainen
& Morton), which the recorded-field rule ``atc`` already uses on the RECORDED
weights ``w`` and recorded deadline ``d``. The supervisor's edge is that it sees
the realized TRUE class, which (under the full_class_shift channel, P1.5) moves
BOTH the true weight ``w*(c*)`` AND the true deadline ``d*=r+SLA(c*)``. Its
preferred pick is the ATC dispatcher computed with the TRUE quantities:

    score(j) = (w*_j / p_j) * exp(-max(0, due_j - now - p_j) / (k * pbar))

where ``due_j = d*_j`` under full_class_shift and ``due_j = d_j`` (recorded,
frozen) under the weight_only E6 boundary. ``self.due`` holds the active
deadline (set in ``__init__`` from the channel), so ``preferred_pick`` and the
one-step ``_pair_improvement`` both use d* automatically under full_class_shift.
k = 2 and pbar = mean processing time over the candidates. This is
myopic-greedy on the true objective; it is NOT a true-objective upper bound, so
ORACLE-GREEDY can occasionally invert against a rule -- flagged, per the
proposal, not hidden. The override IMPROVEMENT over the decider's pick is the
one-step pairwise-swap true-objective increment: the extra true weighted
tardiness of serving the decider's pick before the preferred pick versus the
reverse, on the freed technician (completion = now + p for the first, now + p +
p' for the second). Override fires iff this improvement exceeds ``theta``.
"""

from __future__ import annotations

import math

import numpy as np

from .overlay import stable_seed

_ATC_K = 2.0

REVIEW_MECHANISMS = ("targeted", "random")


class Supervisor:
    """Per-instance supervisor oracle. All randomness is deterministically
    seeded from (seed, instance id)."""

    def __init__(self, overlay, instance, rho, epsilon=0.0, theta=1.0,
                 mechanism="targeted", seed=0, window=64, applied=None):
        if mechanism not in REVIEW_MECHANISMS:
            raise ValueError("mechanism must be one of %r" % (REVIEW_MECHANISMS,))
        self.overlay = overlay
        self.instance_id = instance["meta"]["id"]
        self.rho = float(rho)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.mechanism = mechanism
        self.window = int(window)

        self.applied = applied if applied is not None else overlay.apply(instance)
        self.shift = self.applied["shift"]
        self.wstar = self.applied["w_star"]

        # Active private-information channel (P1.5). full_class_shift => the
        # preferred pick optimises the TRUE objective under the TRUE deadline d*
        # (the slack / urgency term uses d*, not only w*); weight_only freezes
        # the deadline at the recorded due (the E6 boundary control). The channel
        # is read from the overlay when present; with no overlay (hand-built unit
        # tests) or no d_star map, it falls back to the recorded deadline.
        self.channel = getattr(getattr(overlay, "params", None), "channel",
                               "full_class_shift")
        dstar = self.applied.get("d_star") if isinstance(self.applied, dict) else None

        # per-order p_bh / due for the preferred-pick + increment computation.
        # ``self.due`` is the DEADLINE the oracle optimises against.
        self.p_bh = {w["id"]: float(w["p_bh"]) for w in instance["work_orders"]}
        if self.channel == "full_class_shift" and dstar is not None:
            self.due = {w["id"]: float(dstar[w["id"]]) for w in instance["work_orders"]}
        else:
            self.due = {w["id"]: float(w["due_bh"]) for w in instance["work_orders"]}

        self.rng = np.random.default_rng(stable_seed("sup", seed, self.instance_id))

        self._margins = []                 # rolling window of decider margins
        self.n_decisions = 0
        self.n_reviewable = 0              # decisions with >= 2 candidates
        self.n_reviews = 0
        self.n_overrides = 0
        self.n_confirmations = 0
        self.log = []

    # ------------------------------------------------------------------ #
    # Myopic-greedy preferred pick (true-weight ATC)                      #
    # ------------------------------------------------------------------ #
    def _true_atc_score(self, cand, pbar, now):
        cid = cand["id"]
        p = self.p_bh[cid]
        slack = max(0.0, self.due[cid] - now - p)
        return (self.wstar[cid] / p) * math.exp(-slack / (_ATC_K * pbar))

    def preferred_pick(self, candidates, now):
        """Feasible candidate maximizing the true-weight ATC score (myopic-greedy
        on the true objective; tie: work-order id)."""
        pbar = sum(self.p_bh[c["id"]] for c in candidates) / len(candidates)
        best = None
        best_key = None
        for c in candidates:
            # max score == min (-score); id tiebreak matches the pdrs rules.
            key = (-self._true_atc_score(c, pbar, now), c["id"])
            if best_key is None or key < best_key:
                best_key, best = key, c
        return best

    def _pair_improvement(self, decider_pick, preferred, now):
        """One-step pairwise-swap true-objective increment: cost(serve decider
        first) - cost(serve preferred first) on the freed technician."""
        if decider_pick["id"] == preferred["id"]:
            return 0.0

        def cost(first, second):
            f, s = first["id"], second["id"]
            c1 = now + self.p_bh[f]
            c2 = c1 + self.p_bh[s]
            return (self.wstar[f] * max(0.0, c1 - self.due[f])
                    + self.wstar[s] * max(0.0, c2 - self.due[s]))

        return cost(decider_pick, preferred) - cost(preferred, decider_pick)

    # ------------------------------------------------------------------ #
    # Review decision                                                     #
    # ------------------------------------------------------------------ #
    def _has_plus2(self, candidates):
        return any(self.shift.get(c["id"], 0) == 2 for c in candidates)

    def _decide_review(self, margin, candidates):
        if self.rho <= 0.0:
            return False
        if self.mechanism == "random":
            return bool(self.rng.random() < self.rho)
        # TARGETED. A forced pick (single candidate) carries no choice to
        # review; it is never reviewed and never counts against the budget.
        if len(candidates) < 2:
            return False
        self.n_reviewable += 1
        # Consequential: margin below the running rho-quantile of its own
        # margins (Appendix D's 25th-percentile is the rho=0.25 case; using the
        # rho-quantile makes the reviewed fraction track rho across the grid)
        # OR the pending queue holds an order with realized shift s = +2.
        if len(self._margins) >= 8:
            thr = float(np.percentile(self._margins, 100.0 * self.rho))
            consequential = (margin <= thr)
        else:
            consequential = True            # warmup; the budget cap still bounds
        if self._has_plus2(candidates):
            consequential = True
        self._margins.append(float(margin))
        if len(self._margins) > self.window:
            self._margins.pop(0)
        if not consequential:
            return False
        # Online budget controller (top-k consequential per window, k set so the
        # reviewed fraction over reviewable decisions tracks rho).
        return (self.n_reviews < self.rho * self.n_reviewable) or (self.n_reviews == 0)

    # ------------------------------------------------------------------ #
    # Public: review one decision                                        #
    # ------------------------------------------------------------------ #
    def review(self, decider_pick, candidates, now, margin):
        """Return ``(executed_pick, log_entry)`` for one dispatch decision."""
        self.n_decisions += 1
        reviewed = self._decide_review(margin, candidates)

        preferred = None
        override = False
        noise = "none"
        improvement = 0.0
        executed = decider_pick

        if reviewed:
            self.n_reviews += 1
            preferred = self.preferred_pick(candidates, now)
            improvement = self._pair_improvement(decider_pick, preferred, now)
            should = improvement > self.theta
            u = float(self.rng.random())
            half = self.epsilon / 2.0
            if u < half:                     # miss: fail to override
                executed, override, noise = decider_pick, False, "miss"
            elif u < self.epsilon:           # override to a random eligible pick
                executed = candidates[int(self.rng.integers(len(candidates)))]
                override = (executed["id"] != decider_pick["id"])
                noise = "random_override"
            else:                            # honest
                if should:
                    executed, override = preferred, True
                else:
                    executed, override = decider_pick, False
            if override:
                self.n_overrides += 1
            else:
                self.n_confirmations += 1

        entry = {
            "decider_pick": decider_pick["id"],
            "reviewed": reviewed,
            "preferred_pick": preferred["id"] if preferred is not None else None,
            "executed_pick": executed["id"],
            "override": override,
            "confirmation": bool(reviewed and not override),
            "margin": float(margin),
            "improvement": float(improvement),
            "noise": noise,
            "executed_shift": self.shift.get(executed["id"], 0),
        }
        self.log.append(entry)
        return executed, entry

    # ------------------------------------------------------------------ #
    def summary(self):
        # Primary reviewed fraction is over reviewable (>=2 candidate) decisions,
        # which is what rho budgets; the all-decisions fraction is also reported.
        rf = (self.n_reviews / self.n_reviewable) if self.n_reviewable else 0.0
        rf_all = (self.n_reviews / self.n_decisions) if self.n_decisions else 0.0
        orr = (self.n_overrides / self.n_reviews) if self.n_reviews else 0.0
        return {
            "n_decisions": self.n_decisions,
            "n_reviewable": self.n_reviewable,
            "n_reviews": self.n_reviews,
            "reviewed_fraction": rf,
            "reviewed_fraction_all": rf_all,
            "n_overrides": self.n_overrides,
            "override_rate_of_reviews": orr,
            "n_confirmations": self.n_confirmations,
        }
