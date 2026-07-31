"""Supervisor overlay: seeded, bit-for-bit reproducible latent-urgency generator.

Paper Y3, Phase P1. Implements the variance-preserving latent of proposal
Sec.4.2 / Appendix D.1:

    xi_j = sqrt(beta) * f(x_j) + sqrt(1 - beta) * z_j
    s_j  = clip(round(sigma_s * xi_j), -2, +2)        (sigma_s = 1.0)
    c*_j = clip(c_j - s_j, 1, 4)                       (positive s = more urgent)
    w*_j = w(c*_j)                                     (w = 8/4/2/1)

Design choices (recorded in notes/decisions.md):

* Features ``x`` are CAMPUS-AGNOSTIC only: trade one-hot (fixed 14-trade vocab),
  log1p(p_bh), release day-of-week one-hot (bh-axis day index mod 5). The FMUCD
  instance schema carries NO free-text description field (work orders only hold
  id/trade/p_bh/release_bh/due_bh/priority/weight/building/is_pm), so the
  optional "text-derived proxy" of Sec.4.2 is not available and is omitted; this
  is noted, not silently dropped. The recorded class ``c_j`` is EXCLUDED from
  ``x`` by construction (the latent is exactly what the recorded class missed),
  as are all campus-specific fields (building, campus id).
* ``f`` coefficients are drawn ONCE per (family, master_seed) and standardized to
  zero mean / unit variance over the TRAINING-campus order population (campuses
  5, 9, 10, 12, split=train). They are recorded in an overlay coefficient file
  and SHARED between train and test instances. Per-order noise ``z_j`` is fresh
  per instance, seeded deterministically from (instance id, master seed).
* Two locked ``f`` families: F-LIN (linear in standardized x) and F-NL (F-LIN
  PLUS 4 sparse two-way interactions: 2 of type trade x duration-bucket, 2 of
  type trade x day). F-NL reuses F-LIN's linear coefficients so it is literally
  "F-LIN + interactions".

Nothing here imports the environment, the scheduler, or the policy: an overlay
is a pure function of instance content and overlay parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Locked constants (Appendix B)                                               #
# --------------------------------------------------------------------------- #
# Global trade vocabulary: the union of trades over every released instance
# (verified across all 9666 instance files). Fixed and campus-agnostic so the
# one-hot is identical for train and test.
TRADE_VOCAB = ("B20", "B30", "C10", "C30", "D10", "D20", "D30", "D40",
               "D50", "D90", "E10", "E20", "MISC", "UNK")
_TRADE_IDX = {t: i for i, t in enumerate(TRADE_VOCAB)}
N_TRADES = len(TRADE_VOCAB)

N_DAYS = 5                      # weekday slots on the business-hour axis
_DAY_BH = 8.0                   # one business day = 8 bh
BUCKET_EDGES = (1.0, 2.0, 4.0)  # duration buckets: <=1, (1,2], (2,4], >4  -> 4 buckets
N_BUCKETS = len(BUCKET_EDGES) + 1

# Recorded tardiness weights by class (Appendix B): w = 8/4/2/1 for P1..P4.
W_OF_CLASS = {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}

# SLA (contractual lead time) by class in business-hours (Appendix B). The
# recorded due date is d_j = r_j + SLA(c_j); this holds EXACTLY on every released
# instance (verified: 0 / 36306 sampled orders mismatched, all 4 classes). The
# SAME table maps the TRUE class c* to the TRUE deadline d*_j = r_j + SLA(c*_j),
# which is the deadline half of the full-class-shift channel (Paper Y3 P1.5).
SLA_OF_CLASS = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}

# Private-information channel (Paper Y3 P1.5). The latent draw (xi / s / c*) is
# byte-identical across channels; the channel only decides what the true class
# moves:
#   * "full_class_shift" (HEADLINE): the true class moves BOTH the cost weight
#     w*=w(c*) AND the deadline d*=r+SLA(c*). The private urgency reaches a
#     quantity the dispatcher controls (the effective deadline), so it can pay.
#   * "weight_only" (E6 BOUNDARY control): the deadline is frozen at the recorded
#     due date; only w* moves. This is the P1/P2 behaviour, kept selectable.
CHANNELS = ("full_class_shift", "weight_only")
DEFAULT_CHANNEL = "full_class_shift"

SIGMA_S = 1.0                   # class-shift scale (Appendix B)
FAMILIES = ("F-LIN", "F-NL")

# Access channel (secondary; OFF by default). Fixed per-violation penalty =
# weight-8 x 8 bh equivalent (Appendix B).
ACCESS_PENALTY = 8.0 * 8.0

# Base feature layout width: trade one-hot | log1p(p) | day one-hot.
_BASE_DIM = N_TRADES + 1 + N_DAYS

# --------------------------------------------------------------------------- #
# Repo paths                                                                  #
# --------------------------------------------------------------------------- #
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))   # .../fm-hitl
_INSTANCE_ROOT = os.path.join(_REPO, "data", "processed", "instances")
_INDEX_CSV = os.path.join(_INSTANCE_ROOT, "index.csv")
_COEFF_DIR = os.path.join(_REPO, "results", "y3_p1", "overlay_coeffs")

TRAIN_CAMPUSES = (5, 9, 10, 12)


# --------------------------------------------------------------------------- #
# Deterministic seeding                                                       #
# --------------------------------------------------------------------------- #
def stable_seed(*parts) -> int:
    """A machine-independent 63-bit seed from the given parts.

    Uses SHA-256 (not Python's salted ``hash``) so overlays regenerate
    bit-for-bit across processes and machines.
    """
    s = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big") >> 1


# --------------------------------------------------------------------------- #
# Feature extraction (campus-agnostic)                                        #
# --------------------------------------------------------------------------- #
def _day_index(release_bh: float) -> int:
    return int(math.floor(release_bh / _DAY_BH)) % N_DAYS


def _bucket_index(p_bh: float) -> int:
    for i, edge in enumerate(BUCKET_EDGES):
        if p_bh <= edge:
            return i
    return N_BUCKETS - 1


def base_features(wo: dict) -> np.ndarray:
    """Base campus-agnostic feature vector (length ``_BASE_DIM``), float64."""
    x = np.zeros(_BASE_DIM, dtype=np.float64)
    ti = _TRADE_IDX.get(wo["trade"], _TRADE_IDX["UNK"])
    x[ti] = 1.0
    x[N_TRADES] = math.log1p(float(wo["p_bh"]))
    x[N_TRADES + 1 + _day_index(float(wo["release_bh"]))] = 1.0
    return x


def _interaction_indicator(wo: dict, kind: str, trade_idx: int, other_idx: int) -> float:
    """0/1 indicator for a two-way interaction cell."""
    if _TRADE_IDX.get(wo["trade"], _TRADE_IDX["UNK"]) != trade_idx:
        return 0.0
    if kind == "trade_bucket":
        return 1.0 if _bucket_index(float(wo["p_bh"])) == other_idx else 0.0
    return 1.0 if _day_index(float(wo["release_bh"])) == other_idx else 0.0


# --------------------------------------------------------------------------- #
# Training-campus order population (for standardization + coefficient draw)    #
# --------------------------------------------------------------------------- #
_POP_CACHE: dict | None = None


def _iter_index_rows():
    with open(_INDEX_CSV) as fh:
        header = fh.readline().rstrip("\n").split(",")
        col = {name: i for i, name in enumerate(header)}
        for line in fh:
            row = line.rstrip("\n").split(",")
            yield {k: row[col[k]] for k in col}


def load_training_population():
    """Load base features + duration bucket / day of every training-campus order.

    Population = campuses 5/9/10/12, split=train (all replay track). Returns a
    dict with a stacked ``base`` matrix [N, _BASE_DIM] and, per order, its trade
    index, bucket index and day index (needed to build interaction indicators).
    Cached module-side; the enumeration order is deterministic (index.csv order
    then work-order-id sort) so every derived statistic is reproducible.
    """
    global _POP_CACHE
    if _POP_CACHE is not None:
        return _POP_CACHE

    paths = []
    for r in _iter_index_rows():
        if int(r["campus"]) in TRAIN_CAMPUSES and r["split"] == "train":
            paths.append(r["path"])
    paths.sort()

    base_rows = []
    trade_idx = []
    bucket_idx = []
    day_idx = []
    for rel in paths:
        with open(os.path.join(_INSTANCE_ROOT, rel)) as fh:
            inst = json.load(fh)
        for wo in sorted(inst["work_orders"], key=lambda w: w["id"]):
            base_rows.append(base_features(wo))
            trade_idx.append(_TRADE_IDX.get(wo["trade"], _TRADE_IDX["UNK"]))
            bucket_idx.append(_bucket_index(float(wo["p_bh"])))
            day_idx.append(_day_index(float(wo["release_bh"])))

    _POP_CACHE = {
        "base": np.asarray(base_rows, dtype=np.float64),
        "trade_idx": np.asarray(trade_idx, dtype=np.int64),
        "bucket_idx": np.asarray(bucket_idx, dtype=np.int64),
        "day_idx": np.asarray(day_idx, dtype=np.int64),
        "n_instances": len(paths),
    }
    return _POP_CACHE


# --------------------------------------------------------------------------- #
# Coefficient set (drawn once per family+seed, recorded, shared train/test)    #
# --------------------------------------------------------------------------- #
def _standardize_cols(mat: np.ndarray):
    """Column mean/std over a population matrix; zero-variance cols -> std 1."""
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def _draw_interactions(master_seed: int, pop) -> list[dict]:
    """Draw 4 sparse two-way interactions (2 trade x bucket, 2 trade x day).

    Trades are drawn weighted by their population frequency so each interaction
    cell is actually populated; bucket/day are uniform. Coefficients b ~ N(0,1).
    """
    rng = np.random.default_rng(stable_seed("nl", master_seed))
    trade_counts = np.bincount(pop["trade_idx"], minlength=N_TRADES).astype(np.float64)
    trade_p = trade_counts / trade_counts.sum()
    kinds = ["trade_bucket", "trade_bucket", "trade_day", "trade_day"]
    inters = []
    for kind in kinds:
        trade_idx = int(rng.choice(N_TRADES, p=trade_p))
        other_n = N_BUCKETS if kind == "trade_bucket" else N_DAYS
        other_idx = int(rng.integers(other_n))
        b = float(rng.standard_normal())
        inters.append({"type": kind, "trade_idx": trade_idx,
                       "other_idx": other_idx, "b": b})
    return inters


def _population_interaction_matrix(pop, inters: list[dict]) -> np.ndarray:
    """[N, len(inters)] indicator matrix over the training population."""
    n = pop["base"].shape[0]
    g = np.zeros((n, len(inters)), dtype=np.float64)
    for m, it in enumerate(inters):
        trade_mask = (pop["trade_idx"] == it["trade_idx"])
        if it["type"] == "trade_bucket":
            other_mask = (pop["bucket_idx"] == it["other_idx"])
        else:
            other_mask = (pop["day_idx"] == it["other_idx"])
        g[:, m] = (trade_mask & other_mask).astype(np.float64)
    return g


def build_coeffs(family: str, master_seed: int) -> dict:
    """Draw + standardize the coefficient set for (family, master_seed).

    Deterministic: same inputs -> byte-identical dict. Linear coefficients depend
    only on ``master_seed`` (so F-NL = F-LIN + interactions).
    """
    if family not in FAMILIES:
        raise ValueError("family must be one of %r" % (FAMILIES,))
    pop = load_training_population()
    base = pop["base"]

    feat_mean, feat_std = _standardize_cols(base)
    base_std = (base - feat_mean) / feat_std

    a = np.random.default_rng(stable_seed("lin", master_seed)).standard_normal(_BASE_DIM)
    f_raw = base_std @ a

    inters = []
    if family == "F-NL":
        inters = _draw_interactions(master_seed, pop)
        g = _population_interaction_matrix(pop, inters)
        g_mean, g_std = _standardize_cols(g)
        g_std_mat = (g - g_mean) / g_std
        b = np.asarray([it["b"] for it in inters], dtype=np.float64)
        f_raw = f_raw + g_std_mat @ b
        for m, it in enumerate(inters):
            it["g_mean"] = float(g_mean[m])
            it["g_std"] = float(g_std[m])

    f_mean = float(f_raw.mean())
    f_std = float(f_raw.std())
    if f_std < 1e-12:
        f_std = 1.0

    return {
        "family": family,
        "master_seed": int(master_seed),
        "trade_vocab": list(TRADE_VOCAB),
        "n_days": N_DAYS,
        "bucket_edges": list(BUCKET_EDGES),
        "base_dim": _BASE_DIM,
        "feat_mean": [float(v) for v in feat_mean],
        "feat_std": [float(v) for v in feat_std],
        "a": [float(v) for v in a],
        "interactions": inters,
        "f_mean": f_mean,
        "f_std": f_std,
        "population": {"n_orders": int(base.shape[0]),
                       "n_instances": int(pop["n_instances"]),
                       "campuses": list(TRAIN_CAMPUSES), "split": "train"},
    }


def get_coeffs(family: str, master_seed: int, cache: bool = True) -> dict:
    """Load recorded coefficients or build + record them (bit-for-bit stable)."""
    path = os.path.join(_COEFF_DIR, "%s_seed%d.json" % (family, master_seed))
    if cache and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    coeffs = build_coeffs(family, master_seed)
    if cache:
        os.makedirs(_COEFF_DIR, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(coeffs, fh, indent=1, sort_keys=True)
    return coeffs


# --------------------------------------------------------------------------- #
# f evaluation                                                                #
# --------------------------------------------------------------------------- #
def eval_f_matrix(coeffs: dict, work_orders: list[dict]) -> np.ndarray:
    """Standardized latent function f(x) for a list of work orders."""
    feat_mean = np.asarray(coeffs["feat_mean"])
    feat_std = np.asarray(coeffs["feat_std"])
    a = np.asarray(coeffs["a"])
    base = np.stack([base_features(w) for w in work_orders])
    base_std = (base - feat_mean) / feat_std
    f_raw = base_std @ a
    for it in coeffs["interactions"]:
        g = np.asarray([_interaction_indicator(w, it["type"], it["trade_idx"],
                                                it["other_idx"]) for w in work_orders])
        f_raw = f_raw + it["b"] * (g - it["g_mean"]) / it["g_std"]
    return (f_raw - coeffs["f_mean"]) / coeffs["f_std"]


# --------------------------------------------------------------------------- #
# Overlay: bind parameters, apply to an instance                              #
# --------------------------------------------------------------------------- #
@dataclass
class OverlayParams:
    beta: float                      # recoverable-information share
    family: str = "F-NL"             # F-LIN | F-NL (headline: F-NL)
    master_seed: int = 12345         # overlay construction seed (Appendix B)
    sigma_s: float = SIGMA_S
    access_alpha: float = 0.0        # secondary access channel; 0 = OFF (headline)
    channel: str = DEFAULT_CHANNEL   # full_class_shift (headline) | weight_only (E6)

    def __post_init__(self):
        if self.channel not in CHANNELS:
            raise ValueError("channel must be one of %r" % (CHANNELS,))


@dataclass
class Overlay:
    """A supervisor overlay bound to fixed parameters.

    ``apply(instance)`` returns per-order latent quantities. Deterministic in
    (instance content, params, master seed).
    """
    params: OverlayParams
    coeffs: dict = field(default=None, repr=False)

    def __post_init__(self):
        if self.coeffs is None:
            self.coeffs = get_coeffs(self.params.family, self.params.master_seed)

    # -- per-instance latent ------------------------------------------------ #
    def apply(self, instance: dict) -> dict:
        """Return {'per_order': {wo_id: {...}}, plus vector maps}.

        Per order: xi, s (class shift), c_recorded, c_star, w_recorded, w_star,
        d_recorded (the recorded due date) and d_star (the TRUE deadline
        r_j + SLA(c*_j)). The d_star map is additive and channel-independent: it
        is ALWAYS computed as r+SLA(c*); consumers decide whether to USE it
        (full_class_shift) or the recorded due (weight_only), via
        ``params.channel``. Computing d_star touches no RNG draw, so xi / s /
        c* / w* stay byte-identical to the pre-P1.5 overlay.

        Latent noise z is fresh per (instance, master_seed), drawn in wo-id sort
        order so it is reproducible regardless of work-order file ordering.
        """
        p = self.params
        wos = instance["work_orders"]
        wos_sorted = sorted(wos, key=lambda w: w["id"])
        f = eval_f_matrix(self.coeffs, wos_sorted)

        z_seed = stable_seed("z", p.master_seed, instance["meta"]["id"])
        z = np.random.default_rng(z_seed).standard_normal(len(wos_sorted))

        beta = float(p.beta)
        xi = math.sqrt(beta) * f + math.sqrt(1.0 - beta) * z
        s = np.clip(np.round(p.sigma_s * xi), -2, 2).astype(int)

        per_order = {}
        shift_map = {}
        wstar_map = {}
        cstar_map = {}
        dstar_map = {}
        for i, wo in enumerate(wos_sorted):
            c = int(wo["priority"])
            si = int(s[i])
            cstar = int(min(4, max(1, c - si)))
            wstar = W_OF_CLASS[cstar]
            r = float(wo["release_bh"])
            dstar = r + SLA_OF_CLASS[cstar]          # true deadline r + SLA(c*)
            d_recorded = float(wo["due_bh"])
            wid = wo["id"]
            per_order[wid] = {
                "xi": float(xi[i]), "s": si,
                "c_recorded": c, "c_star": cstar,
                "w_recorded": float(wo["weight"]), "w_star": wstar,
                "d_recorded": d_recorded, "d_star": dstar,
            }
            shift_map[wid] = si
            wstar_map[wid] = wstar
            cstar_map[wid] = cstar
            dstar_map[wid] = dstar

        return {"per_order": per_order, "shift": shift_map,
                "w_star": wstar_map, "c_star": cstar_map, "d_star": dstar_map}

    # -- secondary access channel ------------------------------------------ #
    def restricted_buildings(self, instance: dict) -> set:
        """Buildings with a latent access window, density = access_alpha.

        A restricted building is enterable only on weekday mornings (bh-of-day
        < 4). Empty when access_alpha == 0 (headline default).
        """
        alpha = float(self.params.access_alpha)
        if alpha <= 0.0:
            return set()
        buildings = sorted({wo.get("building") for wo in instance["work_orders"]
                            if wo.get("building") is not None})
        restricted = set()
        for b in buildings:
            r = np.random.default_rng(
                stable_seed("access", self.params.master_seed,
                            instance["meta"]["id"], b))
            if float(r.random()) < alpha:
                restricted.add(b)
        return restricted

    @staticmethod
    def _violates_window(start_bh: float) -> bool:
        return (start_bh % _DAY_BH) >= 4.0

    def access_penalty(self, instance: dict, schedule: dict) -> float:
        """Total fixed penalty for access-window violations (0 when channel off)."""
        restricted = self.restricted_buildings(instance)
        if not restricted:
            return 0.0
        b_of = {wo["id"]: wo.get("building") for wo in instance["work_orders"]}
        total = 0.0
        for a in schedule.get("assignments", []):
            if b_of.get(a.get("wo")) in restricted and \
                    self._violates_window(float(a.get("start_bh", 0.0))):
                total += ACCESS_PENALTY
        return total
