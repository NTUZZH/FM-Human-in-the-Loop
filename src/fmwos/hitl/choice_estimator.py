"""W2: queue-conditioned choice-model shift estimator (Paper Y3).

An override is a CHOICE from the feasible queue, not a pointwise measurement of
one order's urgency. The shipped pipeline (``augmented_rule.weak_labels_from_log``
+ ``latent_head.train_estimator``) reduces each reviewed decision to per-order
labels in {+1, -1, 0} fitted by weighted squared error, which discards the
choice-set structure. This module replaces that objective with a conditional
logit (Plackett-Luce, top-1) likelihood over the reviewed queue, and optionally
conditions the estimator on the queue itself.

Two SEPARABLE claims, each able to succeed or fail on its own:

  A. LIKELIHOOD.  Same inputs (per-order observable features), new objective:
     for a reviewed decision with feasible set Q, the corrected ATC index of
     order j is scored at ``c_hat_j = clip(c_j - s_hat_j, 1, 4)``; an override
     toward A contributes ``log P(A | Q)`` under a softmax over the corrected
     log-indices, a confirmation contributes ``log P(decider's pick | Q)``.
     The shipped 5x up-weighting of overrides is kept.
  B. CONDITIONING.  ``s_hat(x_j, Q) = head(concat(x_j, pool(Q)))`` with ``pool``
     a parameter-free Deep-Sets mean/max encoder over the feasible set and
     ``head`` the SHIPPED ``ShiftEstimator`` 2x32 shape (imported, not forked).

Recorded design decision (flagged for the manuscript). A confirmation certifies
only that the decider's pick was within the override tolerance ``theta`` of the
supervisor's preference; treating it as a full choice of that pick slightly
OVER-READS it. ``confirm_mode="choice"`` (the plan's default) does exactly that;
``confirm_mode="tolerance"`` is the robustness variant, in which a confirmation
instead contributes ``sum_{j != pick} log sigma((u_pick - u_j + delta)/tau)``,
i.e. "no rival was better than the pick by more than the tolerance ``delta``".

GUARDRAIL (structural, not promised). Every feature this module ever sees is
built by ``latent_head.candidate_latent_features`` -> ``overlay.base_features``,
which reads ONLY ``trade`` / ``p_bh`` / ``release_bh``. There is exactly ONE
feature constructor here (``instance_tables``); the estimator API accepts
nothing else, so the queue encoder cannot see a latent quantity (s, xi, f, c*,
w*, d*) even by accident. ``tests/test_choice_estimator.py`` pins this with a
key-access-recording instance dict.

Nothing in this module edits or forks a shipped file: the supervisor is
SUBCLASSED to widen its log, the estimator core, the class->weight / class->SLA
curves, the ATC constant, the probe metric and the environment are IMPORTED.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .augmented_rule import (_CLASS_GRID, _SLA_GRID, _W_GRID, interp_sla,
                             interp_weight)
from .latent_head import LAT_DIM, ShiftEstimator, candidate_latent_features
from .supervisor import _ATC_K, Supervisor

# Maximum choice-set width for the conditional logit.
#
# The plan specifies "the K <= 64 candidates", but the measured feasible set at
# a reviewed decision of the headline cell has mean 50, median 31 and MAX 176,
# with 30% of reviewed decisions wider than 64 (results/y3_w2/probe.json). Two
# reasons make the default NO truncation (512 is above every observed queue):
#   * the Deep-Sets encoder is parameter-free and size-agnostic, so 64 is not an
#     architectural bound -- nothing about the encoder needs it;
#   * the deployed decider pools over the WHOLE queue, so truncating during
#     training only would introduce a train/deploy mismatch on 30% of decisions.
# ``K_MAX_PLAN = 64`` is retained and run as an explicit robustness row. When a
# cap binds, the kept candidates are the highest RECORDED-ATC-index ones -- an
# observable, estimator-independent, order-independent selection -- with the
# supervisor's pick and the decider's pick always retained.
K_MAX = 512
K_MAX_PLAN = 64

_NEG_BIG = -1.0e30


# --------------------------------------------------------------------------- #
# 1. Widening the supervisor log so the feasible set Q is recoverable          #
# --------------------------------------------------------------------------- #
class QueueLoggingSupervisor(Supervisor):
    """``Supervisor`` whose log entry also records the feasible set and the clock.

    The shipped ``Supervisor.review`` logs the decider's pick, the preferred
    pick, the executed pick and the flags, but NOT the candidate list it chose
    from and NOT the decision time -- so the choice model cannot be built from a
    shipped log. This subclass adds ``cand_ids`` (the feasible set Q, in the
    env's queue order) and ``now`` (the dispatch clock) to every REVIEWED entry.
    It changes no behaviour: ``super().review`` runs first and its return value,
    RNG draws and counters are untouched; the entry dict appended to
    ``self.log`` is the same object the caller receives, so widening it in place
    widens both views.
    """

    def review(self, decider_pick, candidates, now, margin):
        executed, entry = super().review(decider_pick, candidates, now, margin)
        if entry["reviewed"]:
            entry["cand_ids"] = [c["id"] for c in candidates]
            entry["now"] = float(now)
        return executed, entry


# --------------------------------------------------------------------------- #
# 2. Observable per-instance tables (the ONLY feature constructor here)        #
# --------------------------------------------------------------------------- #
@dataclass
class InstanceTable:
    """Observable per-order tables for one instance. No latent quantity."""
    inst_id: str
    ids: list
    pos: dict
    feats: np.ndarray        # [n, LAT_DIM] float32, from base_features
    p: np.ndarray            # [n] float32  processing time (bh)
    prio: np.ndarray         # [n] float32  RECORDED priority class c_j
    rel: np.ndarray          # [n] float32  release_bh
    due_rec: np.ndarray      # [n] float32  recorded due_bh
    w_rec: np.ndarray        # [n] float32  recorded tardiness weight


def instance_tables(instance) -> InstanceTable:
    """Build the observable tables for one instance.

    The feature block comes from ``latent_head.candidate_latent_features``
    (= ``overlay.base_features``), which reads only trade / p_bh / release_bh.
    The scalar columns are the recorded CMMS fields the deployed ATC rule
    already consumes (p_bh, priority, release_bh, due_bh, weight). Nothing here
    touches the overlay latent.
    """
    wos = instance["work_orders"]
    ids = [w["id"] for w in wos]
    feats = np.stack([candidate_latent_features(w) for w in wos]).astype(np.float32)
    return InstanceTable(
        inst_id=instance["meta"]["id"], ids=ids,
        pos={wid: i for i, wid in enumerate(ids)}, feats=feats,
        p=np.asarray([float(w["p_bh"]) for w in wos], np.float32),
        prio=np.asarray([float(w["priority"]) for w in wos], np.float32),
        rel=np.asarray([float(w["release_bh"]) for w in wos], np.float32),
        due_rec=np.asarray([float(w["due_bh"]) for w in wos], np.float32),
        w_rec=np.asarray([float(w["weight"]) for w in wos], np.float32))


# --------------------------------------------------------------------------- #
# 3. Differentiable class -> weight / class -> SLA curves                      #
# --------------------------------------------------------------------------- #
def _interp_curve_t(c, grid_y):
    """Piecewise-linear interpolation of ``grid_y`` on classes 1..4, in torch.

    Mirrors ``augmented_rule.interp_weight`` / ``interp_sla`` (which use
    ``np.interp`` on the same knots) and is differentiable in ``c`` with the
    clip's zero gradient outside [1, 4] -- the boundary censoring W3 addresses.
    """
    c = torch.clamp(c, 1.0, 4.0)
    y = torch.full_like(c, float(grid_y[0]))
    for i in range(len(grid_y) - 1):
        slope = float(grid_y[i + 1] - grid_y[i])
        y = y + slope * torch.clamp(c - float(_CLASS_GRID[i]), 0.0, 1.0)
    return y


def interp_weight_t(c):
    """Torch twin of ``augmented_rule.interp_weight``."""
    return _interp_curve_t(c, _W_GRID)


def interp_sla_t(c):
    """Torch twin of ``augmented_rule.interp_sla``."""
    return _interp_curve_t(c, _SLA_GRID)


def corrected_utilities(hat_s, prio, p, rel, due_rec, now, mask,
                        channel="full_class_shift", k=_ATC_K):
    """Log of the corrected ATC index for every candidate: the choice utility.

    ``score(j) = (w(c_hat_j) / p_j) * exp(-max(0, d_corr_j - t - p_j)/(k*pbar))``
    with ``c_hat_j = clip(c_j - s_hat_j, 1, 4)``, ``d_corr_j = r_j + SLA(c_hat_j)``
    under full_class_shift and ``d_corr_j = d_j`` (recorded) under weight_only --
    the same index ``augmented_rule.augmented_atc_decider`` deploys. Taking logs
    is monotone, so the argmax is unchanged and the softmax is a conditional
    logit on the deployed index.

    Shapes: everything [B, K] except ``now`` [B]. ``pbar`` is the masked mean
    processing time over the feasible set, matching the deployed decider.
    Padded entries are returned at ``_NEG_BIG`` so they cannot win a softmax.
    """
    m = mask.to(p.dtype)
    pbar = (p * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)
    denom = (k * pbar).clamp_min(1e-6)
    c_hat = torch.clamp(prio - hat_s, 1.0, 4.0)
    w_corr = interp_weight_t(c_hat)
    if channel == "full_class_shift":
        d_corr = rel + interp_sla_t(c_hat)
    elif channel == "weight_only":
        d_corr = due_rec
    else:
        raise ValueError("channel must be full_class_shift or weight_only")
    slack = torch.clamp(d_corr - now.unsqueeze(1) - p, min=0.0)
    u = torch.log(w_corr.clamp_min(1e-9)) - torch.log(p.clamp_min(1e-9)) - slack / denom
    return torch.where(mask, u, torch.full_like(u, _NEG_BIG))


# --------------------------------------------------------------------------- #
# 4. Permutation-invariant queue encoder + the estimator                      #
# --------------------------------------------------------------------------- #
def pool_queue(feats, mask):
    """Deep-Sets mean/max pool over the feasible set. Zero parameters.

    ``feats`` [B, K, D], ``mask`` [B, K] bool -> [B, 2D]. Permutation-invariant
    by construction (mean and max are symmetric functions of the set).
    """
    m = mask.unsqueeze(-1).to(feats.dtype)
    cnt = m.sum(1).clamp_min(1.0)
    mean = (feats * m).sum(1) / cnt
    mx = torch.where(mask.unsqueeze(-1), feats,
                     torch.full_like(feats, _NEG_BIG)).max(1).values
    mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=-1)


class QueueShiftEstimator(nn.Module):
    """``s_hat(x_j, Q) = head(concat(x_j, pool(Q)))``.

    ``head`` is the SHIPPED ``ShiftEstimator`` (imported from ``latent_head``,
    not re-implemented), so with ``use_queue=False`` this is bit-for-bit the
    incumbent architecture -- same layer shapes, same parameter names, same
    initialisation given the same torch seed. With ``use_queue=True`` only the
    first layer widens (LAT_DIM -> 3*LAT_DIM input); depth and width are
    unchanged, per the plan's "no deeper than that".
    """

    def __init__(self, lat_dim: int = LAT_DIM, hidden: int = 32,
                 use_queue: bool = False):
        super().__init__()
        self.lat_dim = int(lat_dim)
        self.hidden = int(hidden)
        self.use_queue = bool(use_queue)
        in_dim = lat_dim * 3 if use_queue else lat_dim
        self.core = ShiftEstimator(lat_dim=in_dim, hidden=hidden)

    def forward(self, feats, mask=None):
        """feats [B, K, LAT_DIM] (+ mask [B, K]) -> hat_s [B, K].

        Also accepts a plain [N, LAT_DIM] matrix when ``use_queue=False``.
        """
        if not self.use_queue:
            return self.core(feats)
        if mask is None:
            mask = torch.ones(feats.shape[:2], dtype=torch.bool,
                              device=feats.device)
        pooled = pool_queue(feats, mask)                       # [B, 2D]
        x = torch.cat([feats, pooled.unsqueeze(1).expand(-1, feats.shape[1], -1)],
                      dim=-1)
        return self.core(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ChoiceModel(nn.Module):
    """A ``QueueShiftEstimator`` plus the conditional logit's scale.

    ``log_tau`` is the single extra parameter of the choice likelihood (the
    Gumbel scale of the discrete-choice model). It is symmetric across the two
    choice variants, so the (ii) -> (iii) comparison isolates the queue
    conditioning exactly.
    """

    def __init__(self, use_queue: bool = False, lat_dim: int = LAT_DIM,
                 hidden: int = 32, log_tau: float = 0.0):
        super().__init__()
        self.est = QueueShiftEstimator(lat_dim=lat_dim, hidden=hidden,
                                       use_queue=use_queue)
        self.log_tau = nn.Parameter(torch.tensor(float(log_tau)))

    @property
    def use_queue(self):
        return self.est.use_queue

    def tau(self):
        return self.log_tau.exp().clamp_min(1e-2)

    def tau_value(self) -> float:
        with torch.no_grad():
            return float(self.tau())

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_params_estimator(self) -> int:
        return self.est.n_params()


# --------------------------------------------------------------------------- #
# 5. The reviewed-decision dataset                                            #
# --------------------------------------------------------------------------- #
@dataclass
class ChoiceDataset:
    """Reviewed decisions in choice form. Observable columns only."""
    tables: list = field(default_factory=list)     # list[InstanceTable]
    tab_of: dict = field(default_factory=dict)     # inst_id -> table index
    inst: list = field(default_factory=list)       # per decision: table index
    cand: list = field(default_factory=list)       # per decision: int32 [k] rows
    now: list = field(default_factory=list)
    chosen: list = field(default_factory=list)     # index INTO cand
    decider: list = field(default_factory=list)    # index INTO cand
    override: list = field(default_factory=list)
    weight: list = field(default_factory=list)
    n_truncated: int = 0
    n_skipped: int = 0

    def __len__(self):
        return len(self.inst)

    # ------------------------------------------------------------------ #
    def add_table(self, instance) -> int:
        iid = instance["meta"]["id"]
        if iid not in self.tab_of:
            self.tab_of[iid] = len(self.tables)
            self.tables.append(instance_tables(instance))
        return self.tab_of[iid]

    def add_log(self, log, instance, override_weight=5.0, confirm_weight=1.0,
                k_max=K_MAX, label_source="executed"):
        """Append every REVIEWED entry of one episode's widened log.

        ``label_source="executed"`` uses the pick the supervisor actually
        STARTED -- the only thing a deployed logger observes, and the shipped
        M0 default (``augmented_rule.weak_labels_from_log``). At eps=0 it is
        bit-identical to the noise-free ``preferred``.
        """
        ti = self.add_table(instance)
        tab = self.tables[ti]
        for e in log:
            if not e.get("reviewed"):
                continue
            if "cand_ids" not in e:
                raise KeyError(
                    "log entry has no 'cand_ids': the choice model needs the "
                    "feasible set Q. Run the episode with "
                    "QueueLoggingSupervisor, not the shipped Supervisor.")
            cids = e["cand_ids"]
            if len(cids) < 2:
                self.n_skipped += 1
                continue
            di = cids.index(e["decider_pick"])
            if label_source == "executed":
                pick_id = e.get("executed_pick", e.get("preferred_pick"))
            else:
                pick_id = e.get("preferred_pick", e.get("executed_pick"))
            if e["override"]:
                if pick_id is None or pick_id not in cids:
                    self.n_skipped += 1
                    continue
                ci = cids.index(pick_id)
                wgt = float(override_weight)
            elif e.get("confirmation"):
                ci = di
                wgt = float(confirm_weight)
            else:
                self.n_skipped += 1
                continue
            rows = np.asarray([tab.pos[c] for c in cids], dtype=np.int32)
            now = float(e["now"])
            if rows.size > k_max:
                rows, ci, di = _truncate_choice_set(tab, rows, ci, di, now, k_max)
                self.n_truncated += 1
            self.inst.append(ti); self.cand.append(rows); self.now.append(now)
            self.chosen.append(int(ci)); self.decider.append(int(di))
            self.override.append(bool(e["override"])); self.weight.append(wgt)

    # ------------------------------------------------------------------ #
    def subset(self, idx):
        d = ChoiceDataset(tables=self.tables, tab_of=self.tab_of)
        for i in idx:
            d.inst.append(self.inst[i]); d.cand.append(self.cand[i])
            d.now.append(self.now[i]); d.chosen.append(self.chosen[i])
            d.decider.append(self.decider[i]); d.override.append(self.override[i])
            d.weight.append(self.weight[i])
        return d

    def split_by_instance(self, val_inst_ids):
        """Leak-free split: ALL decisions of a validation instance go to val.

        Splitting by decision would leak, because the same work order recurs in
        many decisions of the same instance (and across DAgger iterations), and
        the estimator's input is a per-order feature vector.
        """
        val_ti = {self.tab_of[i] for i in val_inst_ids if i in self.tab_of}
        tr = [i for i in range(len(self)) if self.inst[i] not in val_ti]
        va = [i for i in range(len(self)) if self.inst[i] in val_ti]
        return self.subset(tr), self.subset(va)

    def counts(self):
        ov = int(sum(self.override))
        return {"n_decisions": len(self), "n_overrides": ov,
                "n_confirmations": len(self) - ov,
                "n_truncated": self.n_truncated, "n_skipped": self.n_skipped,
                "mean_queue": float(np.mean([len(c) for c in self.cand]))
                if len(self) else float("nan")}

    # ------------------------------------------------------------------ #
    def batches(self, batch_size=128, shuffle=False, rng=None, sort_by_size=True):
        """Yield padded batches. Sorting by |Q| keeps padding waste small; the
        BATCH ORDER is shuffled each epoch so SGD still sees a random stream."""
        n = len(self)
        if n == 0:
            return
        order = np.arange(n)
        if sort_by_size:
            order = order[np.argsort([len(self.cand[i]) for i in order],
                                     kind="stable")]
        chunks = [order[s:s + batch_size] for s in range(0, n, batch_size)]
        if shuffle:
            rng = rng or np.random.default_rng(0)
            rng.shuffle(chunks)
        for ch in chunks:
            yield self._pack(ch)

    def _pack(self, idx):
        k = max(len(self.cand[i]) for i in idx)
        b = len(idx)
        d = self.tables[0].feats.shape[1]
        feats = np.zeros((b, k, d), np.float32)
        p = np.ones((b, k), np.float32)
        prio = np.full((b, k), 4.0, np.float32)
        rel = np.zeros((b, k), np.float32)
        due = np.zeros((b, k), np.float32)
        mask = np.zeros((b, k), bool)
        now = np.zeros((b,), np.float32)
        chosen = np.zeros((b,), np.int64)
        decider = np.zeros((b,), np.int64)
        wgt = np.zeros((b,), np.float32)
        isov = np.zeros((b,), bool)
        for r, i in enumerate(idx):
            t = self.tables[self.inst[i]]
            rows = self.cand[i]
            kk = len(rows)
            feats[r, :kk] = t.feats[rows]
            p[r, :kk] = t.p[rows]
            prio[r, :kk] = t.prio[rows]
            rel[r, :kk] = t.rel[rows]
            due[r, :kk] = t.due_rec[rows]
            mask[r, :kk] = True
            now[r] = self.now[i]
            chosen[r] = self.chosen[i]
            decider[r] = self.decider[i]
            wgt[r] = self.weight[i]
            isov[r] = self.override[i]
        T = torch.as_tensor
        return {"feats": T(feats), "p": T(p), "prio": T(prio), "rel": T(rel),
                "due_rec": T(due), "mask": T(mask), "now": T(now),
                "chosen": T(chosen), "decider": T(decider), "w": T(wgt),
                "override": T(isov)}


def _truncate_choice_set(tab, rows, ci, di, now, k_max):
    """Keep the ``k_max`` highest RECORDED-ATC-index candidates, always keeping
    the supervisor's pick and the decider's pick.

    The selection uses only observable recorded fields, so it is independent of
    the estimator being fitted (no circularity) and identical for every variant.
    Ties break on row index, so the selected SET does not depend on queue order.
    """
    p = tab.p[rows]
    pbar = float(p.mean())
    slack = np.maximum(0.0, tab.due_rec[rows] - now - p)
    sc = (tab.w_rec[rows] / p) * np.exp(-slack / max(_ATC_K * pbar, 1e-6))
    order = np.lexsort((rows, -sc))                # -score asc, then row id
    must = {int(ci), int(di)}
    # Reserve a slot for each required pick FIRST, then fill the remainder with
    # the highest-index rivals. Writing the required picks into the tail of the
    # kept list instead would let the second overwrite the first whenever BOTH
    # fall outside the top k_max (a real case: it fired on ~1 in 6,000 reviewed
    # decisions of the headline cell).
    room = max(0, k_max - len(must))
    keep = [int(o) for o in order if int(o) not in must][:room]
    keep = sorted(set(keep) | must)
    newrows = rows[keep]
    remap = {int(o): n for n, o in enumerate(keep)}
    return newrows, remap[int(ci)], remap[int(di)]


# --------------------------------------------------------------------------- #
# 6. Losses                                                                    #
# --------------------------------------------------------------------------- #
def _hat_s(model, batch):
    if model.use_queue:
        return model.est(batch["feats"], batch["mask"])
    return model.est(batch["feats"])


def choice_logprob(model, batch, channel="full_class_shift"):
    """log P(chosen | Q) per decision under the conditional logit. [B]"""
    hs = _hat_s(model, batch)
    u = corrected_utilities(hs, batch["prio"], batch["p"], batch["rel"],
                            batch["due_rec"], batch["now"], batch["mask"],
                            channel=channel)
    logp = F.log_softmax(u / model.tau(), dim=1)
    return logp.gather(1, batch["chosen"].unsqueeze(1)).squeeze(1)


def tolerance_logprob(model, batch, delta=1.0, channel="full_class_shift"):
    """Tolerance-aware per-decision log-likelihood (the robustness variant).

    A confirmation certifies only that NO rival beat the decider's pick by more
    than the override tolerance, so it contributes
    ``sum_{j != pick} log sigma((u_pick - u_j + delta)/tau)``. An override toward
    A additionally certifies that A DID beat it by more than the tolerance, so it
    contributes ``log P(A | Q) + log sigma((u_A - u_pick - delta)/tau)``.
    """
    hs = _hat_s(model, batch)
    u = corrected_utilities(hs, batch["prio"], batch["p"], batch["rel"],
                            batch["due_rec"], batch["now"], batch["mask"],
                            channel=channel)
    tau = model.tau()
    b = u.shape[0]
    ar = torch.arange(b, device=u.device)
    u_pick = u[ar, batch["decider"]]
    u_ch = u[ar, batch["chosen"]]
    # confirmations: no rival exceeded the pick by more than delta.
    gap = (u_pick.unsqueeze(1) - u + delta) / tau
    ok = batch["mask"].clone()
    ok[ar, batch["decider"]] = False
    conf_ll = (F.logsigmoid(gap) * ok.to(gap.dtype)).sum(1)
    # overrides: the executed pick beat the decider's pick by more than delta.
    logp = F.log_softmax(u / tau, dim=1)
    ov_ll = logp.gather(1, batch["chosen"].unsqueeze(1)).squeeze(1) \
        + F.logsigmoid((u_ch - u_pick - delta) / tau)
    return torch.where(batch["override"], ov_ll, conf_ll)


def mse_loss_terms(model, batch):
    """Weighted-squared-error terms in the SHIPPED reduction, decision-wise.

    Reproduces ``augmented_rule.weak_labels_from_log`` exactly: an override
    contributes (+1 on the executed pick, -1 on the decider's pick), a
    confirmation contributes (0 on the decider's pick). Returned per decision as
    (sum of squared errors, number of label terms) so a decision-level train /
    validation split can weight them identically to the shipped loss.
    """
    hs = _hat_s(model, batch)
    b = hs.shape[0]
    ar = torch.arange(b, device=hs.device)
    s_dec = hs[ar, batch["decider"]]
    s_ch = hs[ar, batch["chosen"]]
    ov = batch["override"]
    se_ov = (s_ch - 1.0) ** 2 + (s_dec + 1.0) ** 2
    se_cf = s_dec ** 2
    se = torch.where(ov, se_ov, se_cf)
    cnt = torch.where(ov, torch.full_like(se, 2.0), torch.ones_like(se))
    return se, cnt


# --------------------------------------------------------------------------- #
# 7. Fitting, with early stopping on held-out reviewed decisions               #
# --------------------------------------------------------------------------- #
def _epoch_objective(model, ds, objective, channel, delta, batch_size=512):
    """Weighted mean objective over a dataset (no grad). Lower is better."""
    if len(ds) == 0:
        return float("nan")
    model.eval()
    tot = wtot = 0.0
    with torch.no_grad():
        for b in ds.batches(batch_size=batch_size):
            w = b["w"]
            if objective == "mse":
                se, cnt = mse_loss_terms(model, b)
                tot += float((w * se).sum()); wtot += float((w * cnt).sum())
            else:
                ll = (choice_logprob(model, b, channel) if objective == "choice"
                      else tolerance_logprob(model, b, delta, channel))
                tot += float((-w * ll).sum()); wtot += float(w.sum())
    return tot / max(wtot, 1e-8)


def fit_estimator(model, ds_tr, ds_va, *, objective="choice",
                  channel="full_class_shift", delta=1.0, epochs=60, lr=1e-2,
                  batch_size=512, patience=8, seed=0, min_epochs=1):
    """Fit ``model`` with early stopping on the held-out reviewed decisions.

    ``objective`` in {"choice", "tolerance", "mse"}. The validation criterion is
    the SAME weighted objective as training (so early stopping is not measuring
    a different thing from the loss); the unweighted held-out choice
    log-likelihood is reported separately by ``choice_loglik``.
    Returns a dict of fit diagnostics; the model is left holding the BEST
    validation state, not the last.
    """
    if len(ds_tr) == 0:
        return {"epochs_run": 0, "best_epoch": -1, "best_val": float("nan"),
                "train_obj": float("nan")}
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, best_ep, bad = float("inf"), None, -1, 0
    ep = 0
    for ep in range(epochs):
        model.train()
        for b in ds_tr.batches(batch_size=batch_size, shuffle=True, rng=rng):
            w = b["w"]
            if objective == "mse":
                se, cnt = mse_loss_terms(model, b)
                loss = (w * se).sum() / (w * cnt).sum().clamp_min(1e-8)
            else:
                ll = (choice_logprob(model, b, channel) if objective == "choice"
                      else tolerance_logprob(model, b, delta, channel))
                loss = (-w * ll).sum() / w.sum().clamp_min(1e-8)
            opt.zero_grad(); loss.backward(); opt.step()
        val = _epoch_objective(model, ds_va, objective, channel, delta)
        if not np.isfinite(val):
            val = _epoch_objective(model, ds_tr, objective, channel, delta)
        if val < best_val - 1e-6:
            best_val, best_ep, bad = val, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience and ep + 1 >= min_epochs:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"epochs_run": ep + 1, "best_epoch": best_ep, "best_val": best_val,
            "train_obj": _epoch_objective(model, ds_tr, objective, channel, delta),
            "tau": model.tau_value()}


def init_temperature(model, ds, channel="full_class_shift", batch_size=512):
    """Scale-initialise ``log_tau`` to the spread of the corrected utilities.

    The corrected ATC log-index spans tens of nats (the slack term divides by
    ``k*pbar``), so a tau of 1 makes every softmax degenerate and the first
    gradients enormous. Setting tau to the within-queue standard deviation of
    the utilities at init puts the likelihood on a workable scale; it is one
    scalar and it is re-fitted by the optimiser afterwards.
    """
    if len(ds) == 0:
        return
    devs = []
    with torch.no_grad():
        for b in ds.batches(batch_size=batch_size):
            hs = _hat_s(model, b)
            u = corrected_utilities(hs, b["prio"], b["p"], b["rel"],
                                    b["due_rec"], b["now"], b["mask"],
                                    channel=channel)
            m = b["mask"]
            uu = torch.where(m, u, torch.zeros_like(u))
            cnt = m.sum(1, keepdim=True).clamp_min(1.0)
            mu = uu.sum(1, keepdim=True) / cnt
            var = (((uu - mu) ** 2) * m).sum(1, keepdim=True) / cnt
            devs.append(var.sqrt().squeeze(1))
    sd = float(torch.cat(devs).median())
    model.log_tau.data.fill_(math.log(max(sd, 1e-2)))


# --------------------------------------------------------------------------- #
# 8. Deployment: the queue-conditioned augmented-ATC decider                   #
# --------------------------------------------------------------------------- #
def queue_conditioned_atc_decider(model, instance, channel="full_class_shift",
                                  k=_ATC_K, table=None, stats=None):
    """A (job, margin) decider: ATC re-scored with a QUEUE-CONDITIONED hat_s.

    Same index as ``augmented_rule.augmented_atc_decider``; the difference is
    that ``hat_s`` is recomputed at EVERY decision from the current feasible
    set, because a queue-conditioned estimator has no per-instance static map.
    ``stats`` (optional dict) accumulates decision count and elapsed seconds so
    the per-decision deployment cost can be reported.
    """
    tab = table if table is not None else instance_tables(instance)
    model.eval()

    def _decider(queue, t, rng):
        import time as _time
        t0 = _time.perf_counter()
        rows = np.asarray([tab.pos[j["id"]] for j in queue], dtype=np.int32)
        feats = torch.as_tensor(tab.feats[rows]).unsqueeze(0)
        mask = torch.ones(feats.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            hs = (model.est(feats, mask) if model.use_queue
                  else model.est(feats))[0].numpy()
        p = tab.p[rows]
        pbar = float(p.mean())
        denom = max(k * pbar, 1e-12)
        c_hat = np.clip(tab.prio[rows] - hs, 1.0, 4.0)
        w_corr = np.interp(c_hat, _CLASS_GRID, _W_GRID)
        if channel == "full_class_shift":
            d_corr = tab.rel[rows] + np.interp(c_hat, _CLASS_GRID, _SLA_GRID)
        else:
            d_corr = tab.due_rec[rows]
        slack = np.maximum(0.0, d_corr - t - p)
        sc = (w_corr / p) * np.exp(-slack / denom)
        ids = [j["id"] for j in queue]
        order = np.lexsort((np.asarray(ids), -sc))
        best = queue[int(order[0])]
        margin = float(sc[order[0]] - sc[order[1]]) if len(queue) >= 2 else 1e9
        if stats is not None:
            stats["n"] = stats.get("n", 0) + 1
            stats["s"] = stats.get("s", 0.0) + (_time.perf_counter() - t0)
        return best, margin

    return _decider


def static_hat_s_map(model, instance, table=None):
    """Per-order hat_s with NO queue context (use_queue=False models only)."""
    if model.use_queue:
        raise ValueError("queue-conditioned model has no static hat_s map")
    tab = table if table is not None else instance_tables(instance)
    with torch.no_grad():
        hs = model.est(torch.as_tensor(tab.feats)).numpy()
    return {wid: float(v) for wid, v in zip(tab.ids, hs)}


# --------------------------------------------------------------------------- #
# 9. Evaluation metrics                                                        #
# --------------------------------------------------------------------------- #
def choice_loglik(model, ds, channel="full_class_shift", tau_override=None,
                  batch_size=512):
    """Held-out choice-model log-likelihood under the COMMON functional.

    Every variant -- including the squared-error incumbent, which has no
    likelihood of its own -- is scored by the SAME conditional logit on the
    corrected index, so the numbers are comparable. Returns overall, override-
    only and confirmation-only means of ``log P(chosen | Q)``, plus the uniform
    -log|Q| floor.
    """
    if len(ds) == 0:
        return {k: float("nan") for k in
                ("ll", "ll_overrides", "ll_confirmations", "ll_uniform",
                 "acc_top1")} | {"n": 0, "n_overrides": 0}
    old = None
    if tau_override is not None:
        old = float(model.log_tau.data)
        model.log_tau.data.fill_(math.log(float(tau_override)))
    tot = totO = totC = unif = acc = 0.0
    n = nO = 0
    model.eval()
    with torch.no_grad():
        for b in ds.batches(batch_size=batch_size):
            hs = _hat_s(model, b)
            u = corrected_utilities(hs, b["prio"], b["p"], b["rel"],
                                    b["due_rec"], b["now"], b["mask"],
                                    channel=channel)
            lp = F.log_softmax(u / model.tau(), dim=1)
            ll = lp.gather(1, b["chosen"].unsqueeze(1)).squeeze(1)
            ov = b["override"]
            tot += float(ll.sum()); totO += float(ll[ov].sum())
            totC += float(ll[~ov].sum())
            unif += float(-torch.log(b["mask"].sum(1).to(ll.dtype)).sum())
            acc += float((u.argmax(1) == b["chosen"]).sum())
            n += ll.numel(); nO += int(ov.sum())
    if old is not None:
        model.log_tau.data.fill_(old)
    return {"ll": tot / n, "ll_overrides": totO / max(nO, 1),
            "ll_confirmations": totC / max(n - nO, 1), "ll_uniform": unif / n,
            "acc_top1": acc / n, "n": n, "n_overrides": nO}


def fit_temperature(model, ds, channel="full_class_shift",
                    grid=None):
    """Calibrate the choice scale on a held-out set, by grid search on log tau.

    Needed so the squared-error incumbent -- which never had a temperature --
    is scored at ITS best scale rather than an arbitrary one; otherwise the
    likelihood comparison would be a comparison of scales. The grid is searched
    on the VALIDATION reviewed decisions only, never on the test set.
    """
    if len(ds) == 0:
        return float(model.tau())
    grid = grid if grid is not None else np.exp(np.linspace(-4.0, 6.0, 81))
    best, best_ll = float(model.tau()), -np.inf
    for t in grid:
        ll = choice_loglik(model, ds, channel=channel, tau_override=float(t))["ll"]
        if np.isfinite(ll) and ll > best_ll:
            best_ll, best = ll, float(t)
    return best


def kendall_at_decisions(score_fn, decisions, true_scores):
    """Mean Kendall tau-b between a candidate ranking and the TRUE ranking.

    ``decisions`` is the COMMON reference trajectory (see
    ``collect_decision_points``): every variant is scored at the same decision
    points, so the metric compares rankings, not trajectories. ``score_fn(d) ->
    np.ndarray`` gives the variant's index over that decision's feasible set.
    """
    from scipy.stats import kendalltau
    taus = []
    for d, ts in zip(decisions, true_scores):
        if len(ts) < 2:
            continue
        sc = np.asarray(score_fn(d), dtype=np.float64)
        t = kendalltau(sc, np.asarray(ts, dtype=np.float64)).correlation
        if np.isfinite(t):
            taus.append(float(t))
    if not taus:
        return {"kendall_tau": float("nan"), "n_decisions": 0}
    return {"kendall_tau": float(np.mean(taus)), "n_decisions": len(taus),
            "kendall_tau_sd": float(np.std(taus))}
