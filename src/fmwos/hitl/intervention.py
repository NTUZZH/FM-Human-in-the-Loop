"""DAgger intervention buffer D_int + the intervention-weighted loss (P2).

``InterventionBuffer`` aggregates every REVIEWED decision across all outer DAgger
iterations. It is NEVER reset (the DAgger "aggregate" step); to keep memory
bounded on long runs it holds a reservoir sample of a fixed capacity, so the
retained set stays an unbiased draw over all iterations rather than the last one.

Each entry records exactly what §5.2 asks for:
  state features at the decision (cand / mask / ctx, plus the campus-agnostic
  latent features latfeat used by the M1 head), the candidate set, the decider's
  pick, the supervisor's preferred pick, the override flag, the confirmation
  flag, and the decider margin.

The loss combines:
  * the standard Y1 PPO loss on the OBSERVABLE shaped reward (built by the
    caller from its rollout; this module does not touch the reward), and
  * a supervised imitation term on D_int: cross-entropy of the policy's masked
    candidate distribution toward the supervisor's PREFERRED pick, with
    OVERRIDDEN (corrected) decisions up-weighted relative to CONFIRMATIONS
    (locked P2 defaults: override weight 5.0, confirmation weight 1.0; both are
    config knobs). This is the HG-DAgger / IWR instinct: learn hardest where the
    supervisor bothered to intervene.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# Locked P2 defaults (logged in notes/decisions.md; overridable via config).
OVERRIDE_WEIGHT = 5.0
CONFIRM_WEIGHT = 1.0
DEFAULT_CAPACITY = 60000


class InterventionBuffer:
    """Aggregated, reservoir-bounded buffer of reviewed decisions (D_int)."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY, seed: int = 0):
        self.capacity = int(capacity)
        self.entries: list[dict] = []
        self._seen = 0                     # total ever offered (reservoir count)
        self.rng = np.random.default_rng(seed)
        # running tallies over ALL entries ever added (not just the retained set)
        self.total_reviewed = 0
        self.total_overrides = 0
        self.total_confirmations = 0

    def __len__(self):
        return len(self.entries)

    def add(self, entry: dict):
        """Offer one reviewed decision to the reservoir. ``entry`` must carry
        cand/mask/ctx/latfeat arrays, decider_idx, preferred_idx (int or None),
        override (bool), confirmation (bool), margin (float)."""
        self.total_reviewed += 1
        if entry["override"]:
            self.total_overrides += 1
        elif entry["confirmation"]:
            self.total_confirmations += 1
        self._seen += 1
        if len(self.entries) < self.capacity:
            self.entries.append(entry)
        else:
            j = int(self.rng.integers(self._seen))
            if j < self.capacity:
                self.entries[j] = entry

    def add_many(self, entries):
        for e in entries:
            self.add(e)

    # ------------------------------------------------------------------ #
    def sample_batch(self, n: int, device="cpu", label_source="preferred"):
        """Draw ``n`` entries and stack them into padded tensors for the loss.

        Returns a dict with cand/mask/ctx/latfeat tensors, target [n] (the
        imitation target index), and weight [n] (override_weight vs confirm_weight
        applied by the loss).

        ``label_source`` selects the imitation target on OVERRIDE entries only,
        exactly as ``augmented_rule.weak_labels_from_log`` does for M0:
          * ``"preferred"`` (DEFAULT): the supervisor's noise-free preferred pick
            (``preferred_idx``) -- the committed M1 behaviour.
          * ``"executed"``: the pick the supervisor ACTUALLY started
            (``executed_idx``), the only thing a deployed logger observes. On an
            override at eps=0 the executed pick IS the preferred pick, so this is
            BIT-IDENTICAL to ``"preferred"`` there; at eps>0 the random-override
            noise branch corrupts it, giving the honest weak-supervision target.
        Confirmation entries (and legacy entries without ``executed_idx``) are
        unchanged in either setting, so eps=0 is a strict no-op.
        """
        if not self.entries:
            return None
        m = min(n, len(self.entries))
        idx = self.rng.integers(0, len(self.entries), size=m)
        ents = [self.entries[i] for i in idx]
        cand = torch.as_tensor(np.stack([e["cand"] for e in ents]),
                               dtype=torch.float32, device=device)
        mask = torch.as_tensor(np.stack([e["mask"] for e in ents]),
                               dtype=torch.bool, device=device)
        ctx = torch.as_tensor(np.stack([e["ctx"] for e in ents]),
                              dtype=torch.float32, device=device)
        latfeat = torch.as_tensor(np.stack([e["latfeat"] for e in ents]),
                                  dtype=torch.float32, device=device)
        targets, over = [], []
        for e in ents:
            t = e["preferred_idx"] if e["preferred_idx"] is not None else e["decider_idx"]
            if (label_source == "executed" and e["override"]
                    and e.get("executed_idx") is not None):
                t = e["executed_idx"]
            targets.append(int(t))
            over.append(bool(e["override"]))
        target = torch.as_tensor(targets, dtype=torch.long, device=device)
        is_over = torch.as_tensor(over, dtype=torch.bool, device=device)
        return {"cand": cand, "mask": mask, "ctx": ctx, "latfeat": latfeat,
                "target": target, "is_override": is_over}

    # ------------------------------------------------------------------ #
    def stats(self):
        orr = (self.total_overrides / self.total_reviewed
               if self.total_reviewed else 0.0)
        return {"retained": len(self.entries), "total_reviewed": self.total_reviewed,
                "total_overrides": self.total_overrides,
                "total_confirmations": self.total_confirmations,
                "override_rate_of_reviews": orr}


# --------------------------------------------------------------------------- #
# Intervention-weighted imitation loss                                        #
# --------------------------------------------------------------------------- #
def imitation_loss(policy, batch, *, override_weight: float = OVERRIDE_WEIGHT,
                   confirm_weight: float = CONFIRM_WEIGHT, use_latent: bool = True):
    """Weighted cross-entropy toward the supervisor's preferred pick.

    ``policy`` exposes ``forward(cand, mask, ctx[, latfeat]) -> (logits, value)``.
    Corrected (overridden) decisions get ``override_weight``; confirmations get
    ``confirm_weight``. Returns a scalar loss (mean over the batch, weighted).
    """
    if batch is None:
        return None
    latfeat = batch["latfeat"] if use_latent else None
    try:
        logits, _v = policy.forward(batch["cand"], batch["mask"], batch["ctx"], latfeat)
    except TypeError:                       # a plain Y1 policy without latfeat arg
        logits, _v = policy.forward(batch["cand"], batch["mask"], batch["ctx"])
    logp = F.log_softmax(logits, dim=-1)
    ce = -logp.gather(-1, batch["target"].unsqueeze(-1)).squeeze(-1)   # [n]
    w = torch.where(batch["is_override"],
                    torch.full_like(ce, override_weight),
                    torch.full_like(ce, confirm_weight))
    return (w * ce).sum() / w.sum().clamp_min(1e-8)
