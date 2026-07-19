"""M1 latent-shift auxiliary head + the shared shift estimator (Paper Y3, P2).

The estimator predicts the class shift ``hat_s_j`` for a work order from its
OBSERVABLE, campus-agnostic features (the same feature list the overlay uses to
build the latent: trade one-hot, log1p processing time, release day-of-week).
Crucially the features are built from INSTANCE DATA ONLY (``overlay.base_features``
reads ``trade`` / ``p_bh`` / ``release_bh`` and nothing else); the latent
internals (s, xi, f, c*, w*) never enter here. The estimator is trained on the
override stream as weak supervision (see ``weak_labels_from_entries``): a pick the
supervisor promoted carries evidence its true class is more urgent than recorded
(positive shift), the demoted pick carries the opposite, and a confirmation is
censored "within-theta" evidence of zero shift at base weight.

Two consumers share this one estimator architecture and training:
* M0 (``augmented_rule``): a STANDALONE ``ShiftEstimator`` trained from the
  RULE+SUP override log; ATC is then re-scored with corrected weights.
* M1 (``LatentDispatchPolicy``): the SAME estimator as an auxiliary head on the
  Y1 MLP scorer; ``hat_s`` folds ADDITIVELY into the per-candidate score through a
  scalar gate. gate == 0 reproduces the Y1 forward pass BIT-EXACTLY (anchor E0);
  the head is otherwise detached from the value path.

Checkpoint format stays Y1-compatible: the backbone parameter names are
unchanged, so a Y1 ``best.pt`` state_dict loads into the backbone with
``strict=False`` and the head starts at init with gate 0.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..policy import DispatchPolicy, _NEG_INF
from ..env import F_CTX, F_JOB, K_CAND
from .overlay import base_features, _BASE_DIM

# Latent-feature dimension: the campus-agnostic overlay feature width
# (14 trade one-hot + log1p(p) + 5 day-of-week one-hot = 20). Built from
# instance data only; never from overlay latent internals.
LAT_DIM = _BASE_DIM

# ATC tuning constant, matching the recorded-field ATC rule and the supervisor's
# true-weight ATC (fmwos.pdrs / fmwos.hitl.supervisor use k = 2.0). Used only by
# the optional in-network deadline head to shape its correction like an ATC slack
# term; defined here to avoid importing the supervisor (no cycle risk).
_ATC_K = 2.0

# Extra observable per-candidate features the deadline head reads on TOP of the
# campus-agnostic latfeat: recorded slack_days (cand col 1), wait_days = (t-r)/8
# (cand col 9), and the recorded-class one-hot (cand cols 4:8). All are OBSERVABLE
# candidate-feature columns already produced by ``DispatchEnv._fill_job_features``;
# none is a latent quantity.
_DL_EXTRA = 6


# --------------------------------------------------------------------------- #
# Feature extraction (instance data only)                                     #
# --------------------------------------------------------------------------- #
def candidate_latent_features(job: dict) -> np.ndarray:
    """Campus-agnostic latent feature vector for one work order (float32).

    Delegates to ``overlay.base_features`` which reads ONLY trade / p_bh /
    release_bh. No latent quantity (s, xi, f, c*, w*) is ever touched here.
    """
    return base_features(job).astype(np.float32)


def latfeat_for_candidates(candidates, feat_cache=None) -> np.ndarray:
    """Stack the [K_CAND, LAT_DIM] latent-feature matrix for a candidate list.

    Padded rows (beyond ``len(candidates)``) are left zero, exactly like the
    Y1 candidate padding. ``feat_cache`` (wo_id -> vector) avoids recomputation
    across the many decisions of one episode.
    """
    out = np.zeros((K_CAND, LAT_DIM), dtype=np.float32)
    for i, job in enumerate(candidates[:K_CAND]):
        if feat_cache is not None:
            v = feat_cache.get(job["id"])
            if v is None:
                v = candidate_latent_features(job)
                feat_cache[job["id"]] = v
        else:
            v = candidate_latent_features(job)
        out[i] = v
    return out


# --------------------------------------------------------------------------- #
# The shared shift estimator                                                  #
# --------------------------------------------------------------------------- #
class ShiftEstimator(nn.Module):
    """MLP [LAT_DIM -> hidden -> hidden -> 1] predicting the class shift hat_s.

    Used both as M0's standalone estimator and as M1's auxiliary head.
    """

    def __init__(self, lat_dim: int = LAT_DIM, hidden: int = 32):
        super().__init__()
        self.lat_dim = lat_dim
        self.hidden = hidden
        self.fc1 = nn.Linear(lat_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, latfeat):
        """latfeat [..., LAT_DIM] -> hat_s [...] (last dim squeezed)."""
        h = F.relu(self.fc1(latfeat))
        h = F.relu(self.fc2(h))
        return self.out(h).squeeze(-1)

    def predict_np(self, feats: np.ndarray, device="cpu") -> np.ndarray:
        """hat_s for a [N, LAT_DIM] numpy matrix (eval convenience)."""
        self.eval()
        with torch.no_grad():
            t = torch.as_tensor(feats, dtype=torch.float32, device=device)
            return self.forward(t).cpu().numpy()

    def save(self, path):
        torch.save({"state_dict": self.state_dict(),
                    "config": {"lat_dim": self.lat_dim, "hidden": self.hidden},
                    "kind": "shift_estimator"}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt["config"]
        m = cls(lat_dim=cfg["lat_dim"], hidden=cfg["hidden"])
        m.load_state_dict(ckpt["state_dict"])
        return m


# --------------------------------------------------------------------------- #
# Weak-label construction from the override stream                            #
# --------------------------------------------------------------------------- #
def weak_labels_from_entries(entries, override_weight: float = 5.0,
                             confirm_weight: float = 1.0,
                             label_source: str = "preferred"):
    """Build (feature, label, weight) weak-supervision examples from D_int-style
    entries (each carries ``latfeat`` [K,LAT_DIM], ``decider_idx``,
    ``preferred_idx``, ``executed_idx``, ``override``, ``confirmation``).

    * override    -> promoted order: label +1; demoted (decider) order: label -1;
                     both at ``override_weight``.
    * confirmation -> decider order: label 0 (censored within-theta), at
                     ``confirm_weight``.

    ``label_source`` selects which pick carries the +1 label on an OVERRIDE,
    mirroring ``augmented_rule.weak_labels_from_log`` for the M0 estimator:
      * ``"preferred"`` (DEFAULT): the noise-free ``preferred_idx`` -- the
        committed M1 behaviour.
      * ``"executed"``: the actually-started ``executed_idx``. At eps=0 an honest
        override starts the preferred pick, so this is BIT-IDENTICAL to
        ``"preferred"``; at eps>0 it is the honest (noise-corrupted) label a
        deployed logger records. The -1 decider label and the 0 confirmations are
        the same for both settings.

    Returns (X [M, LAT_DIM] float32, y [M] float32, w [M] float32).
    """
    X, y, w = [], [], []
    for e in entries:
        lat = e["latfeat"]
        di = e["decider_idx"]
        pi = e["preferred_idx"]
        if label_source == "executed" and e.get("executed_idx") is not None:
            pos = e["executed_idx"]
        else:
            pos = pi
        if e["override"]:
            if pos is not None:
                X.append(lat[pos]); y.append(1.0); w.append(override_weight)
            X.append(lat[di]); y.append(-1.0); w.append(override_weight)
        elif e["confirmation"]:
            X.append(lat[di]); y.append(0.0); w.append(confirm_weight)
    if not X:
        return (np.zeros((0, LAT_DIM), np.float32),
                np.zeros((0,), np.float32), np.zeros((0,), np.float32))
    return (np.asarray(X, np.float32), np.asarray(y, np.float32),
            np.asarray(w, np.float32))


def train_estimator(estimator: ShiftEstimator, X, y, w, *, epochs: int = 40,
                    lr: float = 1e-2, batch_size: int = 512, device="cpu",
                    seed: int = 0):
    """Weighted-MSE fit of ``estimator`` to weak labels. Returns final loss.

    This is the SINGLE training routine shared by M0 and M1's head so the two
    rungs use the same estimator architecture AND training (symmetric protocol).
    """
    if len(X) == 0:
        return float("nan")
    estimator.to(device).train()
    opt = torch.optim.Adam(estimator.parameters(), lr=lr)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    wt = torch.as_tensor(w, dtype=torch.float32, device=device)
    n = Xt.shape[0]
    rng = np.random.default_rng(seed)
    last = float("nan")
    for _ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            bt = torch.as_tensor(b, device=device)
            pred = estimator(Xt[bt])
            loss = (wt[bt] * (pred - yt[bt]) ** 2).sum() / wt[bt].sum().clamp_min(1e-8)
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.item())
    return last


# --------------------------------------------------------------------------- #
# M1 policy: Y1 scorer + gated latent-shift head                              #
# --------------------------------------------------------------------------- #
class LatentDispatchPolicy(DispatchPolicy):
    """Y1 MLP scorer with an additive, gated latent-shift head (M1).

    forward(cand, mask, ctx, latfeat=None) returns (logits, value). When
    ``latfeat`` is None OR ``gate == 0`` the head contributes nothing and the
    forward pass is BIT-IDENTICAL to ``DispatchPolicy.forward`` (the base score
    path, value path and masking are copied verbatim). Otherwise
    ``logits += gate * hat_s(latfeat)`` is added to the per-candidate score
    before masking. The head never feeds the value head.

    P1.5 deadline channel. Under full_class_shift the private true class moves
    BOTH the cost weight w* and the deadline d*. In M1 the deadline reaches the
    learner through the DAgger IMITATION TARGET: the supervisor's preferred pick
    and executed override (Supervisor.preferred_pick, whose ``self.due`` is d*)
    now optimise TWT*(w*,d*), so the imitation cross-entropy in ``intervention.
    imitation_loss`` pulls the policy toward the d*-aware ordering, and the
    executed-override rollout stores those d*-aware actions for PPO. The additive
    ``gate * hat_s`` head carries the recovered class shift itself (a positive
    shift = more urgent than recorded, i.e. costlier AND due sooner), the same
    scalar M0 feeds into BOTH its corrected weight and corrected deadline. The
    additive form is kept because the observable candidate vector does not expose
    release_bh, so an in-network d_corr cannot be reconstructed without a leak or
    an architecture change that would break the gate=0 E0 anchor; the deadline
    influence is instead carried faithfully by the imitation target. ``correction_
    mode`` records which channel the recovered shift corresponds to
    (full_class_shift headline vs weight_only ablation) and travels with the
    checkpoint; the weight_only ablation is selected by pairing the policy with a
    weight_only overlay / supervisor (recorded deadline in the preferred pick).

    FAIR-M1 deadline head (flag ``deadline_head=True``; Paper Y3 P3 M1-fairness).
    The additive-``hat_s`` term above is a per-candidate LOGIT BUMP: it cannot
    express the deadline half of the full-class-shift correction, which in an ATC
    index is a SLACK term (nonlinear, saturating at the deadline), not a flat
    offset. The optional ``deadline_head`` gives M1 that missing channel IN
    NETWORK, E0-preserving. It is a small MLP reading OBSERVABLE candidate
    features only -- the campus-agnostic ``latfeat`` plus the recorded slack, the
    wait-time ``(t - r)`` (which exposes the order's release time relative to now),
    and the recorded-class one-hot -- and NEVER a latent quantity (s / c* / w* /
    d*). Its output is a signed per-candidate deadline shift ``delta_d`` (bh) whose
    FINAL layer is ZERO-INITIALIZED, so ``delta_d == 0`` and the head contributes
    EXACTLY 0 at init. ``delta_d`` modulates the effective due date inside an
    ATC-slack term, added to the pair score in log space:

        slack_rec  = d_rec - t - p                 (recorded slack, from cand)
        slack_eff  = slack_rec - delta_d           (delta_d>0 => earlier deadline)
        adjust     = -(relu(slack_eff) - relu(slack_rec)) / (k * pbar)
        logits    += gate * adjust

    This is exactly the deadline half of log(ATC_corrected / ATC_recorded): linear
    in ``delta_d`` while the order still has slack, and saturating (relu kink) once
    it is already late, i.e. ATC-slack-shaped. The head is trained END-TO-END by
    the same PPO + intervention-weighted imitation signal (the imitation target is
    the supervisor's d*-aware preferred pick), so the deadline-aware overrides now
    have an in-network pathway to shape the ordering, not only the flat hat_s bump.
    With ``gate == 0`` the whole block is skipped, so E0 (Y1 bit-exactness) holds
    unchanged. ``deadline_head=False`` (default) reproduces the OLD M1 exactly.
    """

    def __init__(self, f_job: int = F_JOB, f_ctx: int = F_CTX,
                 hidden: int = 64, k_cand: int = K_CAND,
                 lat_dim: int = LAT_DIM, lat_hidden: int = 32,
                 gate: float = 0.0, correction_mode: str = "full_class_shift",
                 deadline_head: bool = False, dl_hidden: int = 32):
        super().__init__(f_job=f_job, f_ctx=f_ctx, hidden=hidden, k_cand=k_cand)
        if correction_mode not in ("full_class_shift", "weight_only"):
            raise ValueError("correction_mode must be full_class_shift or weight_only")
        self.lat_dim = lat_dim
        self.lat_hidden = lat_hidden
        self.correction_mode = correction_mode
        self.shift_head = ShiftEstimator(lat_dim=lat_dim, hidden=lat_hidden)
        # Optional in-network deadline head (fair M1). Additive, gated, and
        # zero-initialized at the output so it contributes exactly 0 at init. It
        # is a plain backbone-side head (name does NOT start with "shift_head"),
        # so it is trained by PPO + imitation, unlike the weak-label shift head.
        self.use_deadline_head = bool(deadline_head)
        self.dl_hidden = int(dl_hidden)
        if self.use_deadline_head:
            self.deadline_head = nn.Sequential(
                nn.Linear(lat_dim + _DL_EXTRA, dl_hidden), nn.ReLU(),
                nn.Linear(dl_hidden, dl_hidden), nn.ReLU(),
                nn.Linear(dl_hidden, 1))
            nn.init.zeros_(self.deadline_head[-1].weight)
            nn.init.zeros_(self.deadline_head[-1].bias)
        # gate as a registered buffer (not a parameter): fixed schedule, never
        # receives a gradient, and travels with the checkpoint.
        self.register_buffer("gate", torch.tensor(float(gate)))

    def set_gate(self, value: float):
        self.gate.fill_(float(value))

    # ------------------------------------------------------------------ #
    def forward(self, cand, mask, ctx, latfeat=None):
        b, k, _ = cand.shape
        ctx_b = ctx.unsqueeze(1).expand(b, k, self.f_ctx)
        x = torch.cat([cand, ctx_b], dim=-1)
        h = F.relu(self.enc1(x))
        emb = F.relu(self.enc2(h))                     # [B, K, hidden]
        logits = self.score(emb).squeeze(-1)           # [B, K]

        # Latent-shift head: additive, gated. Skipped entirely when off so the
        # base path is bit-exact with Y1 (anchor E0). hat_s is DETACHED: the head
        # is trained only by weak-label regression (``train_estimator``), while
        # PPO and the imitation term train the backbone around the recovered
        # shift. Detaching does not change the forward values (E0 unaffected).
        if latfeat is not None and float(self.gate) != 0.0:
            hat_s = self.shift_head(latfeat).detach()  # [B, K]
            logits = logits + self.gate * hat_s
            # Fair-M1 deadline channel (flag-gated). NOT detached: trained
            # end-to-end by PPO + imitation. Zero-init output => 0 at init.
            if self.use_deadline_head:
                logits = logits + self.gate * self._deadline_adjust(cand, mask, latfeat)

        m = mask.to(logits.dtype)
        logits = torch.where(mask, logits, torch.full_like(logits, _NEG_INF))

        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (emb * m.unsqueeze(-1)).sum(dim=1) / denom
        vh = F.relu(self.val1(torch.cat([pooled, ctx], dim=-1)))
        value = self.val2(vh).squeeze(-1)
        return logits, value

    def hat_s_of(self, latfeat):
        """hat_s [B,K] from a latent-feature batch (diagnostics / M1 recovery)."""
        return self.shift_head(latfeat)

    # ------------------------------------------------------------------ #
    def _deadline_adjust(self, cand, mask, latfeat):
        """ATC-slack-shaped, per-candidate logit adjustment from the deadline head.

        Reads OBSERVABLE candidate features only: the campus-agnostic ``latfeat``,
        the recorded slack_days (cand col 1), the wait_days ``(t - r)/8`` (cand
        col 9, which exposes the release time relative to now), and the recorded-
        class one-hot (cand cols 4:8). The head outputs a signed deadline shift
        ``delta_d`` (bh); the effective due date d_rec - delta_d feeds an ATC slack
        term. Zero-init output => returns all-zeros at init (E0-safe). Returns
        [B, K]; padded rows are masked out by the caller.
        """
        slack_days = cand[..., 1:2]                 # recorded (d_rec - t - p)/8
        wait_days = cand[..., 9:10]                 # (t - r)/8  (release-relative)
        prio = cand[..., 4:8]                        # recorded-class one-hot
        head_in = torch.cat([latfeat, slack_days, wait_days, prio], dim=-1)
        delta_d = self.deadline_head(head_in).squeeze(-1)          # [B, K] bh
        # Reconstruct bh-scale quantities from the observable candidate columns.
        p_bh = torch.expm1(cand[..., 0])                           # log1p(p) -> p
        slack_rec = cand[..., 1] * 8.0                             # days -> bh
        m = mask.to(p_bh.dtype)
        pbar = (p_bh * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0)
        denom = (_ATC_K * pbar).clamp_min(1e-6)                    # [B, 1]
        slack_eff = slack_rec - delta_d
        return -(F.relu(slack_eff) - F.relu(slack_rec)) / denom    # [B, K]

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act_with_margin(self, obs, latfeat=None, greedy=False, device=None):
        """Pick an action AND report a top1-top2 softmax-prob margin.

        Returns (action:int, logprob:float, value:float, margin:float). ``obs``
        is the Y1 obs dict; ``latfeat`` is the optional [K,LAT_DIM] matrix.
        """
        device = device or next(self.parameters()).device
        cand = torch.as_tensor(obs["cand"], dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.as_tensor(obs["mask"], dtype=torch.bool, device=device).unsqueeze(0)
        ctx = torch.as_tensor(obs["ctx"], dtype=torch.float32, device=device).unsqueeze(0)
        lat = None
        if latfeat is not None:
            lat = torch.as_tensor(latfeat, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(cand, mask, ctx, lat)
        logp, probs, _ent = self._masked_dist(logits, mask)
        if greedy:
            action = torch.argmax(logits, dim=-1)
        else:
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        a = int(action.item())
        # margin = top1 - top2 of the masked softmax probabilities.
        p = probs[0]
        top = torch.topk(p, k=min(2, int(mask[0].sum().item()) or 1)).values
        margin = float(top[0] - top[1]) if top.numel() >= 2 else 1e9
        return a, float(logp[0, a].item()), float(value.item()), margin

    def evaluate(self, cand, mask, ctx, actions, latfeat=None):
        """Batched re-scoring for PPO (optionally with the latent head)."""
        logits, value = self.forward(cand, mask, ctx, latfeat)
        logp, _probs, entropy = self._masked_dist(logits, mask)
        logprobs = logp.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
        return logprobs, entropy, value

    # ------------------------------------------------------------------ #
    def save(self, path):
        torch.save({"state_dict": self.state_dict(),
                    "config": {"f_job": self.f_job, "f_ctx": self.f_ctx,
                               "hidden": self.hidden, "k_cand": self.k_cand},
                    "latent_config": {"lat_dim": self.lat_dim,
                                      "lat_hidden": self.lat_hidden,
                                      "gate": float(self.gate),
                                      "correction_mode": self.correction_mode,
                                      "deadline_head": self.use_deadline_head,
                                      "dl_hidden": self.dl_hidden},
                    "arch": "latent_mlp"}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt["config"]
        lc = ckpt.get("latent_config", {})
        m = cls(f_job=cfg["f_job"], f_ctx=cfg["f_ctx"], hidden=cfg["hidden"],
                k_cand=cfg["k_cand"], lat_dim=lc.get("lat_dim", LAT_DIM),
                lat_hidden=lc.get("lat_hidden", 32), gate=lc.get("gate", 0.0),
                correction_mode=lc.get("correction_mode", "full_class_shift"),
                deadline_head=lc.get("deadline_head", False),
                dl_hidden=lc.get("dl_hidden", 32))
        m.load_state_dict(ckpt["state_dict"])
        return m

    @classmethod
    def from_y1_checkpoint(cls, path, gate: float = 0.0, map_location="cpu",
                           correction_mode: str = "full_class_shift",
                           deadline_head: bool = False, dl_hidden: int = 32):
        """Build an M1 policy whose BACKBONE is a loaded Y1 checkpoint.

        The backbone weights load exactly (same parameter names); the auxiliary
        head(s) stay at init and ``gate`` defaults to 0, so the forward pass is
        bit-identical to the Y1 policy until the gate is opened (E0). This holds
        with OR without the fair-M1 deadline head, whose output is zero-init and
        whose whole block is skipped at gate=0.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt["config"]
        m = cls(f_job=cfg["f_job"], f_ctx=cfg["f_ctx"], hidden=cfg["hidden"],
                k_cand=cfg["k_cand"], gate=gate, correction_mode=correction_mode,
                deadline_head=deadline_head, dl_hidden=dl_hidden)
        missing, unexpected = m.load_state_dict(ckpt["state_dict"], strict=False)
        # Only the auxiliary head params may be "missing" from a Y1 checkpoint.
        assert all(k.startswith("shift_head") or k.startswith("deadline_head")
                   or k == "gate" for k in missing), missing
        assert not unexpected, unexpected
        return m
