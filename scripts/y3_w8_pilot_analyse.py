"""Paper Y3 -- W8: pre-registered analysis of the practitioner urgency-pairs pilot.

One command, from returned response sheets to the numbers the manuscript quotes:

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c 20-23 \
        python scripts/y3_w8_pilot_analyse.py

Reads ``pilot/y3_w8_manifest.csv`` and every ``*.csv`` in ``pilot/responses/``.
Writes ``results/y3_w8/pilot_analysis.json`` (everything) and
``results/y3_w8/pilot_analysis.md`` (a summary with a manuscript-ready block).

The hypotheses, the statistics, the sample size, and the reading of every
possible outcome are fixed in ``results/y3_w8/PREREGISTRATION.md`` BEFORE any
response exists. This file implements that document and nothing beyond it; any
statistic added later must be labelled exploratory in the output.

WHAT THE PILOT ESTABLISHES
--------------------------
1. Practitioners agree with each other about relative urgency more than chance,
   which is what makes a hidden but shared urgency a coherent notion and which
   bounds the override noise level.
2. Their judgements are partly predictable, out of sample, from the observable
   order attributes the estimator reads, which is the real-data analogue of
   beta > 0.

WHAT IT CANNOT CLAIM
--------------------
The pilot measures how practitioners rank pairs of real work orders; it does not
put a practitioner in the dispatch loop. It therefore grounds two premises of the
supervisor model and nothing else: it does not validate the correction loop, the
dispatching results, or any reported reduction in true weighted tardiness, all of
which rest on the simulated supervisor.

GRACEFUL DEGRADATION
--------------------
Every statistic carries ``computable`` and, when false, the reason. With two
completed responses the script runs and reports honestly, saying which quantities
the sample size cannot support rather than printing a meaningless number. With
one response only the within-rater statistics exist. With none it prints the
design and stops.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "pilot"
RES = ROOT / "results" / "y3_w8"

# ---- pre-registered analysis constants ------------------------------------ #
SEED = 20260801
N_BOOT = 10000              # bootstrap resamples over pairs
N_PERM_ALPHA = 10000        # permutation resamples for the agreement null
N_PERM_AUC = 500            # permutation resamples for the predictability null
CV_FOLDS = 5
CV_REPEATS = 20             # repeats of stratified K-fold for the reported model
CV_REPEATS_PERM = 5         # repeats inside the permutation null (cost control)
RIDGE_LAMBDA = 1.0          # L2 penalty on standardised features, intercept free
CLASS_SETTLES_NULL = 0.90   # H3 tests the class-agreement rate against this
SUPERMAJORITY = 2.0 / 3.0   # "decisive" majority on class-silent items
MC_DRAWS = 400000           # Monte-Carlo draws for the beta -> AUC map
PAPER_BETA_BAND = (0.75, 1.00)
PAPER_EPS_GRID = (0.0, 0.10, 0.25)

STRATA = ("S1_equal_class", "S2_one_apart", "S3_class_vs_attributes", "S4_far_apart")


# =========================================================================== #
# Small statistics, all implemented here so the run has no fitted-library      #
# version sensitivity and every formula is visible next to its use.            #
# =========================================================================== #
def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + conf / 2.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial interval."""
    if n == 0:
        return (float("nan"), float("nan"))
    a = (1.0 - conf) / 2.0
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return (lo, hi)


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve by the rank (Mann-Whitney) identity, ties at 0.5."""
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(np.asarray(s, dtype=float))
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def krippendorff_alpha_nominal(units: list[list[int]]) -> float:
    """Krippendorff's alpha with the nominal difference function.

    ``units`` is one list of observed values per unit (item); units rated fewer
    than twice contribute nothing, which is how alpha absorbs missing responses.
    Chosen over Fleiss' kappa because the design is fully crossed but ratings may
    be missing, the number of raters is small and may differ across items, and
    alpha is defined for any number of raters without dropping incomplete items.
    """
    vals = sorted({v for u in units for v in u})
    if len(vals) < 2:
        return float("nan")
    vi = {v: i for i, v in enumerate(vals)}
    k = len(vals)
    o = np.zeros((k, k), dtype=float)
    for u in units:
        m = len(u)
        if m < 2:
            continue
        cnt = np.zeros(k)
        for v in u:
            cnt[vi[v]] += 1
        for c in range(k):
            for d in range(k):
                pairs = cnt[c] * (cnt[c] - 1) if c == d else cnt[c] * cnt[d]
                o[c, d] += pairs / (m - 1)
    n = o.sum()
    if n <= 1:
        return float("nan")
    nc = o.sum(axis=1)
    delta = 1.0 - np.eye(k)
    d_obs = float((o * delta).sum()) / n
    d_exp = float((np.outer(nc, nc) * delta).sum()) / (n * (n - 1))
    if d_exp == 0:
        return float("nan")
    return 1.0 - d_obs / d_exp


def fleiss_kappa(units: list[list[int]]) -> float:
    """Fleiss' kappa on the units rated by the full complement of raters."""
    m = Counter(len(u) for u in units)
    if not m:
        return float("nan")
    mode = max(m.items(), key=lambda t: (t[1], t[0]))[0]
    use = [u for u in units if len(u) == mode]
    if mode < 2 or len(use) < 2:
        return float("nan")
    vals = sorted({v for u in use for v in u})
    vi = {v: i for i, v in enumerate(vals)}
    n, k = len(use), len(vals)
    tab = np.zeros((n, k))
    for r, u in enumerate(use):
        for v in u:
            tab[r, vi[v]] += 1
    p_j = tab.sum(axis=0) / (n * mode)
    p_i = (np.square(tab).sum(axis=1) - mode) / (mode * (mode - 1))
    pbar, pebar = float(p_i.mean()), float(np.square(p_j).sum())
    if pebar >= 1.0:
        return float("nan")
    return (pbar - pebar) / (1.0 - pebar)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two aligned nominal series."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return float("nan")
    vals = sorted(set(a.tolist()) | set(b.tolist()))
    if len(vals) < 2:
        return float("nan")
    vi = {v: i for i, v in enumerate(vals)}
    k = len(vals)
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[vi[x], vi[y]] += 1
    n = m.sum()
    po = float(np.trace(m)) / n
    pe = float((m.sum(axis=0) * m.sum(axis=1)).sum()) / (n * n)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def boot_ci(fn, n_units: int, rng: np.random.Generator, n_boot: int = N_BOOT,
            conf: float = 0.95) -> tuple[float, float, list]:
    """Percentile bootstrap over units (items), the sample the design draws."""
    if n_units < 2:
        return (float("nan"), float("nan"), [])
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        v = fn(idx)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            draws.append(float(v))
    if len(draws) < max(50, n_boot // 20):
        return (float("nan"), float("nan"), draws)
    a = (1.0 - conf) / 2.0
    return (float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a)), draws)


# --------------------------------------------------------------------------- #
# Ridge logistic regression (IRLS), 20 lines, deterministic, no library fit    #
# --------------------------------------------------------------------------- #
def fit_ridge_logistic(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA,
                       iters: int = 50, tol: float = 1e-9) -> np.ndarray:
    """Newton (IRLS) fit of an L2-penalised logistic model; intercept unpenalised.

    X already carries a leading column of ones. A fixed penalty is pre-registered
    rather than tuned, because at fifty pairs an inner tuning loop is noisier than
    the choice it makes.
    """
    n, p = X.shape
    w = np.zeros(p)
    pen = np.full(p, lam)
    pen[0] = 0.0
    for _ in range(iters):
        eta = X @ w
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        s = np.clip(mu * (1 - mu), 1e-8, None)
        g = X.T @ (y - mu) - pen * w
        H = (X.T * s) @ X + np.diag(pen)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w = w + step
        if float(np.max(np.abs(step))) < tol:
            break
    return w


def predict_ridge_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -35, 35)))


def cv_out_of_fold(F: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                   repeats: int = CV_REPEATS, folds: int = CV_FOLDS) -> np.ndarray:
    """Repeated stratified K-fold out-of-fold probabilities, averaged over repeats.

    Standardisation is fitted inside each training fold, never on the full sample.
    A fold whose training half holds one class only falls back to that fold's
    training base rate, which is the honest degenerate answer rather than a crash.
    """
    n = len(y)
    if F.shape[1] == 0:                       # the constant null model
        acc = np.zeros(n)
        for _ in range(repeats):
            for tr, te in _strat_folds(y, folds, rng):
                acc[te] += float(y[tr].mean()) if len(tr) else 0.5
        return acc / repeats
    acc = np.zeros(n)
    for _ in range(repeats):
        for tr, te in _strat_folds(y, folds, rng):
            if len(tr) < 4 or len(np.unique(y[tr])) < 2:
                acc[te] += float(y[tr].mean()) if len(tr) else 0.5
                continue
            mu, sd = F[tr].mean(axis=0), F[tr].std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            Xtr = np.column_stack([np.ones(len(tr)), (F[tr] - mu) / sd])
            Xte = np.column_stack([np.ones(len(te)), (F[te] - mu) / sd])
            w = fit_ridge_logistic(Xtr, y[tr].astype(float))
            acc[te] += predict_ridge_logistic(Xte, w)
    return acc / repeats


def _strat_folds(y: np.ndarray, folds: int, rng: np.random.Generator):
    """Stratified fold assignment; degrades to plain folds when a class is tiny."""
    n = len(y)
    k = min(folds, n)
    if k < 2:
        return []
    assign = np.empty(n, dtype=int)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        idx = idx[rng.permutation(len(idx))]
        assign[idx] = np.arange(len(idx)) % k
    return [(np.flatnonzero(assign != f), np.flatnonzero(assign == f))
            for f in range(k) if np.any(assign == f) and np.any(assign != f)]


# --------------------------------------------------------------------------- #
# beta -> AUC map for the translation (seeded Monte Carlo, inverted by interp) #
# --------------------------------------------------------------------------- #
def beta_auc_map(seed: int = SEED, draws: int = MC_DRAWS) -> tuple[np.ndarray, np.ndarray]:
    """AUC an oracle scoring by f(x) attains against sign(xi), as a function of beta.

    Under the paper's latent, xi = sqrt(beta) u + sqrt(1-beta) v with u = f(x) and
    v independent standard normal. A pair's consensus ordering is sign(xi_A - xi_B),
    whose distribution is the same as sign of a single standard normal once the
    difference is rescaled, so the map is computed on that reduced form.
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(draws)
    v = rng.standard_normal(draws)
    betas = np.linspace(0.0, 1.0, 101)
    aucs = np.empty_like(betas)
    for i, b in enumerate(betas):
        xi = math.sqrt(b) * u + math.sqrt(1.0 - b) * v
        aucs[i] = auc_score((xi > 0).astype(int), u)
    aucs = np.maximum.accumulate(aucs)         # monotone by construction; enforce
    return betas, aucs


def beta_from_auc(auc: float, betas: np.ndarray, aucs: np.ndarray) -> float:
    if not np.isfinite(auc):
        return float("nan")
    if auc <= aucs[0]:
        return 0.0
    if auc >= aucs[-1]:
        return 1.0
    return float(np.interp(auc, aucs, betas))


# =========================================================================== #
# Loading                                                                      #
# =========================================================================== #
def read_manifest(path: Path) -> list[dict]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("presentation", "is_repeat", "campus", "a_on_left",
                  "cls_a", "cls_b", "d_cls", "is_pm_a", "is_pm_b"):
            r[k] = int(float(r[k]))
        for k in ("anchor_bh", "chat_a", "chat_b", "d_chat", "trade_prior_a",
                  "trade_prior_b", "labor_h_a", "labor_h_b", "log1p_labor_a",
                  "log1p_labor_b", "wait_days_a", "wait_days_b"):
            r[k] = float(r[k])
    return rows


def read_responses(resp_dir: Path, man: dict[str, dict]) -> tuple[dict, dict]:
    """Return ({rater: {item_id: chosen_order_id}}, load report)."""
    files = sorted(p for p in glob.glob(str(resp_dir / "*.csv")))
    out: dict[str, dict[str, str]] = defaultdict(dict)
    meta: dict[str, dict] = {}
    rep = dict(files=[], rows_read=0, rows_blank=0, rows_unknown_item=0,
               rows_side_conflict=0, rows_bad_choice=0, duplicates_overwritten=0)
    for f in files:
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        n_used = 0
        for r in rows:
            rep["rows_read"] += 1
            rid = (r.get("rater_id") or "").strip()
            iid = (r.get("item_id") or "").strip()
            if not iid or iid not in man:
                if iid:
                    rep["rows_unknown_item"] += 1
                continue
            side = (r.get("choice_side") or "").strip().upper()[:1]
            chosen = (r.get("chosen_order_id") or "").strip()
            m = man[iid]
            resolved = ""
            if side in ("L", "R"):
                resolved = m["left_order"] if side == "L" else m["right_order"]
            if chosen:
                if chosen not in (m["left_order"], m["right_order"]):
                    rep["rows_bad_choice"] += 1
                    continue
                if resolved and resolved != chosen:
                    rep["rows_side_conflict"] += 1
                    continue
                resolved = chosen
            if not resolved:
                rep["rows_blank"] += 1
                continue
            if not rid:
                rid = Path(f).stem
            if iid in out[rid]:
                rep["duplicates_overwritten"] += 1
            out[rid][iid] = resolved
            meta.setdefault(rid, dict(role=(r.get("rater_role") or "").strip(),
                                      years=(r.get("rater_years_fm") or "").strip(),
                                      source=os.path.basename(f)))
            n_used += 1
        rep["files"].append(dict(file=os.path.basename(f), rows=len(rows), used=n_used))
    out = {k: v for k, v in out.items() if v}
    rep["raters"] = sorted(out)
    rep["rater_meta"] = {k: meta.get(k, {}) for k in sorted(out)}
    rep["answers_per_rater"] = {k: len(v) for k, v in sorted(out.items())}
    return out, rep


# =========================================================================== #
# Analysis blocks                                                              #
# =========================================================================== #
def na(reason: str) -> dict:
    return dict(computable=False, reason=reason)


def block_agreement(first: list[dict], resp: dict, rng) -> dict:
    """Inter-rater agreement on the unique pairs, overall and per stratum."""
    raters = sorted(resp)
    R = len(raters)
    if R < 2:
        return na(f"inter-rater agreement needs at least 2 completed responses; "
                  f"{R} present")

    def units_for(rows):
        us, keep = [], []
        for m in rows:
            u = [1 if resp[r].get(m["item_id"]) == m["order_a"] else 0
                 for r in raters if m["item_id"] in resp[r]]
            if len(u) >= 2:
                us.append(u)
                keep.append(m)
        return us, keep

    units, used = units_for(first)
    if len(units) < 2:
        return na(f"only {len(units)} pair(s) carry two or more ratings")

    alpha = krippendorff_alpha_nominal(units)
    lo, hi, _ = boot_ci(lambda idx: krippendorff_alpha_nominal([units[i] for i in idx]),
                        len(units), rng)

    # pairwise agreement, the quantity the noise translation consumes
    pair_agree, pair_kappa = [], []
    for a in range(R):
        for b in range(a + 1, R):
            ra, rb = raters[a], raters[b]
            common = [m for m in used if m["item_id"] in resp[ra] and m["item_id"] in resp[rb]]
            if len(common) < 2:
                continue
            va = np.array([1 if resp[ra][m["item_id"]] == m["order_a"] else 0 for m in common])
            vb = np.array([1 if resp[rb][m["item_id"]] == m["order_a"] else 0 for m in common])
            pair_agree.append(dict(rater_a=ra, rater_b=rb, n=len(common),
                                   percent_agreement=float((va == vb).mean()),
                                   cohen_kappa=cohen_kappa(va, vb)))
            pair_kappa.append(cohen_kappa(va, vb))
    mean_pa = float(np.mean([p["percent_agreement"] for p in pair_agree])) if pair_agree else float("nan")

    def mean_pa_boot(idx):
        vals = []
        for a in range(R):
            for b in range(a + 1, R):
                ra, rb = raters[a], raters[b]
                acc = []
                for i in idx:
                    m = used[i]
                    if m["item_id"] in resp[ra] and m["item_id"] in resp[rb]:
                        acc.append((resp[ra][m["item_id"]] == m["order_a"]) ==
                                   (resp[rb][m["item_id"]] == m["order_a"]))
                if acc:
                    vals.append(float(np.mean(acc)))
        return float(np.mean(vals)) if vals else float("nan")

    pa_lo, pa_hi, _ = boot_ci(mean_pa_boot, len(used), rng, n_boot=2000)

    # permutation null: each rater's answers shuffled across pairs independently
    obs = alpha
    ge = 0
    # 1 = chose the canonical first order, -1 = chose the other, 0 = no answer.
    # A missing answer must not be encoded as a choice, or the null would credit
    # every skipped item to one side; shuffling a rater's row carries their
    # missingness with it, which preserves each rater's answer count.
    arr = np.array([[(1 if resp[r][m["item_id"]] == m["order_a"] else -1)
                     if m["item_id"] in resp[r] else 0
                     for m in used] for r in raters])
    for _ in range(N_PERM_ALPHA):
        perm = np.array([row[rng.permutation(len(used))] for row in arr])
        us = []
        for c in range(len(used)):
            col = [int(v) for v in perm[:, c] if v != 0]
            if len(col) >= 2:
                us.append(col)
        a = krippendorff_alpha_nominal(us)
        if np.isfinite(a) and a >= obs:
            ge += 1
    p_perm = (1.0 + ge) / (1.0 + N_PERM_ALPHA)

    per_stratum = {}
    for st in STRATA:
        rows = [m for m in first if m["stratum"] == st]
        us, _u = units_for(rows)
        if len(us) < 2:
            per_stratum[st] = na(f"{len(us)} pair(s) with two or more ratings")
            continue
        a = krippendorff_alpha_nominal(us)
        s_lo, s_hi, _ = boot_ci(lambda idx: krippendorff_alpha_nominal([us[i] for i in idx]),
                                len(us), rng, n_boot=4000)
        per_stratum[st] = dict(computable=True, n_pairs=len(us),
                               krippendorff_alpha=a, ci95=[s_lo, s_hi],
                               note="a stratum holds few pairs; the interval is wide "
                                    "by design and is reported, not smoothed")

    return dict(
        computable=True, n_raters=R, n_pairs=len(units),
        krippendorff_alpha=alpha, ci95=[lo, hi],
        permutation_p=p_perm, permutation_resamples=N_PERM_ALPHA,
        fleiss_kappa=fleiss_kappa(units),
        mean_pairwise_cohen_kappa=float(np.nanmean(pair_kappa)) if pair_kappa else float("nan"),
        mean_pairwise_percent_agreement=mean_pa,
        mean_pairwise_percent_agreement_ci95=[pa_lo, pa_hi],
        pairwise=pair_agree,
        per_stratum=per_stratum,
        statistic_choice=("Krippendorff's alpha, nominal difference function. The "
                          "design is fully crossed but responses may be missing, the "
                          "rater count is small and may vary by item, and alpha is "
                          "defined for any number of raters without discarding "
                          "incomplete items; Fleiss' kappa and mean pairwise Cohen's "
                          "kappa are reported alongside as conventional references. "
                          "The interval is a percentile bootstrap over pairs, which "
                          "are the units the sampling rule drew; raters are a small "
                          "convenience sample and resampling three of them would "
                          "produce an interval with no usable coverage."),
    )


def block_within_rater(man_rows: list[dict], resp: dict) -> dict:
    """Within-rater consistency from the repeated pairs."""
    rep_pairs = sorted({m["pair_id"] for m in man_rows if m["is_repeat"] == 1})
    if not rep_pairs:
        return na("the instrument carries no repeated pairs")
    by_pair = defaultdict(dict)
    for m in man_rows:
        by_pair[m["pair_id"]][m["presentation"]] = m

    per_rater, hits, trials = {}, 0, 0
    first_v, second_v = [], []
    for r in sorted(resp):
        k = n = 0
        for pid in rep_pairs:
            p1, p2 = by_pair[pid].get(1), by_pair[pid].get(2)
            if not p1 or not p2:
                continue
            a1, a2 = resp[r].get(p1["item_id"]), resp[r].get(p2["item_id"])
            if a1 is None or a2 is None:
                continue
            n += 1
            same = (a1 == a2)
            k += int(same)
            first_v.append(1 if a1 == p1["order_a"] else 0)
            second_v.append(1 if a2 == p2["order_a"] else 0)
        per_rater[r] = dict(consistent=k, repeats_answered=n,
                            rate=(k / n) if n else float("nan"))
        hits += k
        trials += n
    if trials == 0:
        return na("no rater answered both presentations of any repeated pair")
    lo, hi = clopper_pearson(hits, trials)
    p_gt_half = float(stats.binomtest(hits, trials, 0.5, alternative="greater").pvalue)
    return dict(
        computable=True, n_repeat_pairs=len(rep_pairs),
        pooled_consistent=hits, pooled_trials=trials,
        pooled_rate=hits / trials, ci95_exact=[lo, hi],
        binomial_p_greater_than_chance=p_gt_half,
        cohen_kappa_first_vs_second=cohen_kappa(np.array(first_v), np.array(second_v)),
        per_rater=per_rater,
        note=(f"{len(rep_pairs)} repeats per rater: a single rater's rate moves in "
              f"steps of {1.0 / max(len(rep_pairs), 1):.2f} and cannot separate, say, "
              f"0.75 from 0.95. The pooled rate over {trials} rater-repeat trials is "
              f"the reportable quantity; the per-rater column is descriptive."),
    )


def block_class(first: list[dict], resp: dict) -> dict:
    """Does the recorded class settle which job goes first?"""
    raters = sorted(resp)
    R = len(raters)
    if R < 1:
        return na("no responses")

    maj = {}
    for m in first:
        votes = [resp[r][m["item_id"]] for r in raters if m["item_id"] in resp[r]]
        if not votes:
            continue
        c = Counter(votes)
        top, n_top = c.most_common(1)[0]
        tied = (len(c) > 1 and c.most_common(2)[1][1] == n_top)
        maj[m["item_id"]] = dict(label=None if tied else top, n_votes=len(votes),
                                 n_top=n_top, tied=tied,
                                 share=n_top / len(votes))

    diff = [m for m in first if m["d_cls"] != 0]
    scored = [m for m in diff if maj.get(m["item_id"], {}).get("label")]
    n_tied = len(diff) - len(scored)
    if not scored:
        return na("no class-differing pair carries an untied majority")

    def class_pick(m):
        return m["order_a"] if m["cls_a"] < m["cls_b"] else m["order_b"]

    agree = [int(maj[m["item_id"]]["label"] == class_pick(m)) for m in scored]
    k, n = int(sum(agree)), len(agree)
    lo, hi = wilson_ci(k, n)
    p_vs_chance = float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue)
    p_vs_settles = float(stats.binomtest(k, n, CLASS_SETTLES_NULL, alternative="less").pvalue)

    per_stratum = {}
    for st in ("S2_one_apart", "S3_class_vs_attributes", "S4_far_apart"):
        rows = [m for m in scored if m["stratum"] == st]
        if not rows:
            per_stratum[st] = na("no untied class-differing pair in this stratum")
            continue
        kk = int(sum(maj[m["item_id"]]["label"] == class_pick(m) for m in rows))
        slo, shi = wilson_ci(kk, len(rows))
        per_stratum[st] = dict(computable=True, n=len(rows), agree=kk,
                               rate=kk / len(rows), ci95_wilson=[slo, shi])

    # class-silent pairs: the class makes no prediction, so ask whether the
    # practitioners still order them decisively
    silent = [m for m in first if m["d_cls"] == 0 and m["item_id"] in maj]
    dec = [m for m in silent if not maj[m["item_id"]]["tied"]
           and maj[m["item_id"]]["share"] >= SUPERMAJORITY]
    silent_block = dict(computable=bool(silent), n_pairs=len(silent),
                        n_decisive=len(dec),
                        decisive_rate=(len(dec) / len(silent)) if silent else float("nan"),
                        supermajority_threshold=SUPERMAJORITY,
                        note=("on these pairs the recorded class is silent, so a "
                              "decisive majority cannot come from the class"))

    per_rater = {}
    for r in raters:
        rows = [m for m in diff if m["item_id"] in resp[r]]
        if not rows:
            continue
        kk = int(sum(resp[r][m["item_id"]] == class_pick(m) for m in rows))
        per_rater[r] = dict(n=len(rows), agree=kk, rate=kk / len(rows))

    return dict(
        computable=True, n_raters=R,
        n_class_differing=len(diff), n_scored=n, n_dropped_tied_majority=n_tied,
        majority_agrees_with_class=k, rate=k / n, ci95_wilson=[lo, hi],
        binomial_p_vs_chance=p_vs_chance,
        settles_null=CLASS_SETTLES_NULL,
        binomial_p_vs_settles_null=p_vs_settles,
        per_stratum=per_stratum, class_silent_pairs=silent_block,
        per_rater=per_rater,
        caveat=("with an even number of raters a pair can split evenly and is then "
                "dropped; at two raters only unanimous pairs survive, so the rate is "
                "conditioned on the easy pairs and reads high"
                if R % 2 == 0 else ""),
    )


def block_predictability(first: list[dict], resp: dict, rng) -> dict:
    """Out-of-sample predictability of the majority from observable attributes."""
    raters = sorted(resp)
    R = len(raters)
    rows, y = [], []
    n_unanswered, n_tied = 0, 0
    for m in first:
        votes = [resp[r][m["item_id"]] for r in raters if m["item_id"] in resp[r]]
        if not votes:
            n_unanswered += 1
            continue
        c = Counter(votes)
        top, n_top = c.most_common(1)[0]
        if len(c) > 1 and c.most_common(2)[1][1] == n_top:
            n_tied += 1                    # tied majority: no label
            continue
        rows.append(m)
        y.append(1 if top == m["order_a"] else 0)
    y = np.array(y, dtype=int)
    if len(y) < 8 or len(np.unique(y)) < 2:
        return na(f"{len(y)} labelled pair(s), {len(np.unique(y))} distinct "
                  f"outcome(s): fewer than the 8 pairs and both outcomes a "
                  f"cross-validated fit needs")

    d_trade = np.array([m["trade_prior_a"] - m["trade_prior_b"] for m in rows])
    d_logp = np.array([m["log1p_labor_a"] - m["log1p_labor_b"] for m in rows])
    d_wait = np.array([m["wait_days_a"] - m["wait_days_b"] for m in rows])
    d_cls = np.array([float(m["d_cls"]) for m in rows])

    feats = {
        "M0_constant": np.zeros((len(y), 0)),
        "M1_recorded_class": np.column_stack([d_cls]),
        "M2_attributes": np.column_stack([d_trade, d_logp, d_wait]),
        "M3_class_plus_attributes": np.column_stack([d_cls, d_trade, d_logp, d_wait]),
    }

    out = {}
    # Seeds are derived from the model's position, never from hash(name):
    # Python salts string hashing per process, so a hash-derived seed would make
    # the run irreproducible across invocations.
    for slot, (name, F) in enumerate(feats.items()):
        seed_rng = np.random.default_rng(SEED + 101 * (slot + 1))
        p = cv_out_of_fold(F, y, seed_rng)
        acc = float(((p >= 0.5).astype(int) == y).mean())
        brier = float(np.mean((p - y) ** 2))
        if F.shape[1] == 0:
            # A constant predictor has no discrimination, so its area is 0.5 by
            # definition. The cross-validated base rate is not quite constant,
            # because a held-out positive is missing from its own training fold,
            # and that leave-one-out anti-correlation drags the empirical area
            # below 0.5. The null is fixed at 0.5 rather than reporting the
            # artefact; the fold-wise base rate still supplies Brier and accuracy,
            # which is where a null model is genuinely informative.
            out[name] = dict(computable=True, n=len(y), auc=0.5,
                             auc_ci95=[0.5, 0.5], accuracy=acc, brier=brier,
                             features=0,
                             note=("null model: area fixed at 0.5 by definition; "
                                   "Brier and accuracy are cross-validated"))
            continue
        a = auc_score(y, p)
        b_rng = np.random.default_rng(SEED + 7)
        lo, hi, _ = boot_ci(lambda idx: auc_score(y[idx], p[idx]), len(y), b_rng)
        out[name] = dict(computable=True, n=len(y), auc=a, auc_ci95=[lo, hi],
                         accuracy=acc, brier=brier,
                         features=int(F.shape[1]))

    # permutation null for the pre-registered primary model
    F2 = feats["M2_attributes"]
    obs = out["M2_attributes"]["auc"]
    ge = 0
    for i in range(N_PERM_AUC):
        yp = y[rng.permutation(len(y))]
        if len(np.unique(yp)) < 2:
            continue
        pr = cv_out_of_fold(F2, yp, np.random.default_rng(SEED + 1000 + i),
                            repeats=CV_REPEATS_PERM)
        ap = auc_score(yp, pr)
        if np.isfinite(ap) and ap >= obs:
            ge += 1
    out["M2_attributes"]["permutation_p"] = (1.0 + ge) / (1.0 + N_PERM_AUC)
    out["M2_attributes"]["permutation_resamples"] = N_PERM_AUC

    return dict(
        computable=True, n_labelled_pairs=int(len(y)),
        n_dropped_tied_majority=int(n_tied),
        n_dropped_unanswered=int(n_unanswered),
        base_rate_chose_A=float(y.mean()),
        models=out,
        protocol=(f"{CV_REPEATS} repeats of stratified {CV_FOLDS}-fold "
                  f"cross-validation; L2-penalised logistic regression with a "
                  f"pre-registered penalty of {RIDGE_LAMBDA} on features "
                  f"standardised inside each training fold; out-of-fold "
                  f"probabilities averaged over repeats; interval is a percentile "
                  f"bootstrap over pairs."),
        features=("trade or system as a corpus-level urgency prior for the merged "
                  "trade, job size as log1p of estimated labour hours, and days "
                  "waited, each entered as the A-minus-B difference. These are the "
                  "three observable inputs the estimator reads. A scalar trade "
                  "encoding computed from the corpus, not from the responses, keeps "
                  "the model at three parameters, which fifty pairs can support."),
        caveat=("at two raters the majority label exists only where the two agree, "
                "so this estimate is conditioned on unanimous pairs and reads high"
                if R == 2 else ("the target is a single rater's judgement, not a "
                                "consensus" if R == 1 else "")),
    )


def block_translation(agreement: dict, predict: dict, n_raters: int) -> dict:
    """Translate the measured quantities into ranges for epsilon and beta.

    Stated as a translation under assumptions, not a measurement of the model's
    parameters. Both assumptions are named in the output.
    """
    out = dict(computable=True, disclaimer=(
        "This is a translation under stated assumptions, not a measurement of the "
        "model's parameters. The pilot measures how practitioners rank pairs of real "
        "work orders; epsilon and beta are properties of a simulated supervisor "
        "acting inside a dispatch loop. The two are analogues, and the arithmetic "
        "below only says which parameter values a reader should regard as plausible."))

    # ---- epsilon ---------------------------------------------------------- #
    if not agreement.get("computable"):
        out["epsilon"] = na("needs inter-rater agreement, which needs 2+ responses")
    else:
        pa = agreement["mean_pairwise_percent_agreement"]
        lo_pa, hi_pa = agreement["mean_pairwise_percent_agreement_ci95"]

        def q_of(p):
            if not np.isfinite(p) or p < 0.5:
                return float("nan")
            return (1.0 - math.sqrt(max(0.0, 2.0 * p - 1.0))) / 2.0

        q, q_lo, q_hi = q_of(pa), q_of(hi_pa), q_of(lo_pa)   # q decreases in p
        if not np.isfinite(q):
            out["epsilon"] = na("pairwise agreement is at or below chance, so the "
                                "single-consensus model does not identify an error rate")
        else:
            band = [q, min(1.0, 2.0 * q)]
            covered = [e for e in PAPER_EPS_GRID if band[0] <= e <= band[1]]
            out["epsilon"] = dict(
                computable=True,
                per_rater_error_rate_q=q, q_ci95=[q_lo, q_hi],
                implied_epsilon_range=band,
                paper_epsilon_grid=list(PAPER_EPS_GRID),
                paper_grid_values_inside_range=covered,
                assumptions=(
                    "Assume one consensus ordering per pair and that each "
                    "practitioner reports it independently with a constant error "
                    "rate q. Then two practitioners agree with probability "
                    "(1-q)^2 + q^2, so the observed pairwise agreement identifies "
                    "q = (1 - sqrt(2*P_agree - 1)) / 2. The model's epsilon is the "
                    "probability that a reviewed decision yields a corrupted "
                    "correction; on a two-alternative item a corrupted correction is "
                    "wrong with probability 1/2 under the random-pick branch and with "
                    "probability 1 in the worst case, which brackets epsilon in "
                    "[q, 2q]."),
                conservatism=(
                    "Genuine differences of professional judgement between "
                    "practitioners are counted as error by the single-consensus "
                    "model, so q over-states any one practitioner's error against "
                    "their own standard, and the translated epsilon band is an "
                    "upper reading rather than a point estimate."))

    # ---- beta ------------------------------------------------------------- #
    if not predict.get("computable"):
        out["beta"] = na("needs the out-of-sample predictability block")
    else:
        m2 = predict["models"]["M2_attributes"]
        betas, aucs = beta_auc_map()
        raw = beta_from_auc(m2["auc"], betas, aucs)
        raw_ci = [beta_from_auc(m2["auc_ci95"][0], betas, aucs),
                  beta_from_auc(m2["auc_ci95"][1], betas, aucs)]

        e_maj, deatt_auc, deatt = float("nan"), float("nan"), float("nan")
        eps = out.get("epsilon", {})
        if eps.get("computable") and n_raters >= 2:
            q = eps["per_rater_error_rate_q"]
            R = n_raters
            ks = np.arange(R + 1)
            pmf = stats.binom.pmf(ks, R, q)
            wrong = float(pmf[ks > R / 2.0].sum())
            if R % 2 == 0:
                tie = float(pmf[ks == R // 2].sum())
                e_maj = wrong / (1.0 - tie) if tie < 1.0 else float("nan")
            else:
                e_maj = wrong
            if np.isfinite(e_maj) and e_maj < 0.5:
                deatt_auc = 0.5 + (m2["auc"] - 0.5) / (1.0 - 2.0 * e_maj)
                deatt = beta_from_auc(min(deatt_auc, 1.0), betas, aucs)

        out["beta"] = dict(
            computable=True,
            measured_auc=m2["auc"], measured_auc_ci95=m2["auc_ci95"],
            implied_beta_low=raw, implied_beta_from_auc_ci=raw_ci,
            majority_label_error_rate=e_maj,
            deattenuated_auc=deatt_auc, implied_beta_deattenuated=deatt,
            implied_beta_range=[raw, deatt] if np.isfinite(deatt) else [raw, float("nan")],
            paper_beta_band=list(PAPER_BETA_BAND),
            assumptions=(
                "Under the paper's latent, xi = sqrt(beta) f(x) + sqrt(1-beta) z, an "
                "oracle scoring pairs by f(x) attains a known area under the ROC "
                "curve at each beta; that map is computed by seeded Monte Carlo and "
                "inverted here. Two corrections act in opposite directions. The "
                "pilot's model is a three-feature linear proxy for a function no one "
                "has written down, fitted on about fifty pairs, so it under-fits and "
                "pushes the reading down. Symmetric error in the majority label "
                "attenuates the area by the exact factor (1 - 2e), where e is the "
                "probability that a majority of practitioners errs, and dividing it "
                "out pushes the reading up."),
            reading=("Report the range, never either endpoint alone, and call it a "
                     "range of plausible values rather than an interval or a bound. "
                     "The lower endpoint credits the proxy with capturing all of the "
                     "practitioners' urgency function, which it will not; the upper "
                     "endpoint removes the label noise but not the proxy's "
                     "under-fitting, so where the proxy happens to be well specified "
                     "it can sit above the true recoverable share."))
    return out


# =========================================================================== #
# Reporting                                                                    #
# =========================================================================== #
def fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.{nd}f}"
    return str(v)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment over the pre-registered primary family."""
    items = [(k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    if not items:
        return {}
    items.sort(key=lambda t: t[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def write_markdown(path: Path, res: dict) -> None:
    L = []
    A = L.append
    A("# W8 practitioner pilot: results")
    A("")
    A(f"Generated by `scripts/y3_w8_pilot_analyse.py` from "
      f"`{res['inputs']['manifest']}` and {len(res['inputs']['load']['files'])} "
      f"response file(s). Analysis pre-registered in `PREREGISTRATION.md`.")
    A("")
    A("## What this pilot cannot claim")
    A("")
    A(res["boundary_statement"])
    A("")
    A("## Sample")
    A("")
    A(f"- Completed responses: **{res['n_raters']}**")
    A(f"- Unique pairs: {res['design']['n_unique_pairs']}; presented items: "
      f"{res['design']['n_items']}; repeated pairs: {res['design']['n_repeat_pairs']}")
    A(f"- Answers per response: {res['inputs']['load']['answers_per_rater']}")
    A("")

    ag = res["agreement"]
    A("## H1 Inter-rater agreement")
    A("")
    if ag.get("computable"):
        A(f"- Krippendorff's alpha (nominal) = **{fmt(ag['krippendorff_alpha'])}**, "
          f"95% bootstrap CI [{fmt(ag['ci95'][0])}, {fmt(ag['ci95'][1])}], "
          f"permutation p = {fmt(ag['permutation_p'], 4)}")
        A(f"- Fleiss' kappa = {fmt(ag['fleiss_kappa'])}; mean pairwise Cohen's kappa "
          f"= {fmt(ag['mean_pairwise_cohen_kappa'])}; mean pairwise agreement = "
          f"{fmt(ag['mean_pairwise_percent_agreement'])}")
        A("")
        A("| stratum | pairs | alpha | 95% CI |")
        A("|---|---|---|---|")
        for st in STRATA:
            s = ag["per_stratum"][st]
            if s.get("computable"):
                A(f"| {st} | {s['n_pairs']} | {fmt(s['krippendorff_alpha'])} | "
                  f"[{fmt(s['ci95'][0])}, {fmt(s['ci95'][1])}] |")
            else:
                A(f"| {st} | - | not computable | {s['reason']} |")
    else:
        A(f"Not computable: {ag['reason']}")
    A("")

    wr = res["within_rater"]
    A("## Instrument check: within-rater consistency")
    A("")
    if wr.get("computable"):
        A(f"- Pooled {wr['pooled_consistent']}/{wr['pooled_trials']} repeats answered "
          f"identically = **{fmt(wr['pooled_rate'])}**, exact 95% CI "
          f"[{fmt(wr['ci95_exact'][0])}, {fmt(wr['ci95_exact'][1])}], "
          f"p = {fmt(wr['binomial_p_greater_than_chance'], 4)} against chance")
        A(f"- {wr['note']}")
        g = res.get("instrument_check_H4", {})
        A(f"- Pre-registered gate (rate at or above {fmt(g.get('floor'), 2)} and the "
          f"interval clear of chance): **{'passed' if g.get('passed') else 'not passed'}"
          f"**. {g.get('consequence')}")
    else:
        A(f"Not computable: {wr['reason']}")
    A("")

    cl = res["class_test"]
    A("## H3 Does the recorded class settle the order of service?")
    A("")
    if cl.get("computable"):
        A(f"- On {cl['n_scored']} class-differing pairs with an untied majority, the "
          f"majority follows the recorded class in **{cl['majority_agrees_with_class']}"
          f"** of them, rate {fmt(cl['rate'])}, Wilson 95% CI "
          f"[{fmt(cl['ci95_wilson'][0])}, {fmt(cl['ci95_wilson'][1])}]")
        A(f"- Against the null that the class settles it "
          f"({cl['settles_null']}): p = {fmt(cl['binomial_p_vs_settles_null'], 4)}; "
          f"against chance: p = {fmt(cl['binomial_p_vs_chance'], 4)}")
        s = cl["class_silent_pairs"]
        A(f"- On the {s['n_pairs']} pairs whose recorded classes are equal, "
          f"{s['n_decisive']} carry a majority of at least two thirds "
          f"({fmt(s['decisive_rate'])})")
        if cl.get("caveat"):
            A(f"- Caveat: {cl['caveat']}")
    else:
        A(f"Not computable: {cl['reason']}")
    A("")

    pr = res["predictability"]
    A("## H2 Out-of-sample predictability from observable attributes")
    A("")
    if pr.get("computable"):
        A(f"Labelled pairs: {pr['n_labelled_pairs']} "
          f"(dropped for a tied majority: {pr['n_dropped_tied_majority']}).")
        A("")
        A("| model | features | AUC | 95% CI | accuracy | Brier |")
        A("|---|---|---|---|---|---|")
        for k, v in pr["models"].items():
            A(f"| {k} | {v['features']} | {fmt(v['auc'])} | "
              f"[{fmt(v['auc_ci95'][0])}, {fmt(v['auc_ci95'][1])}] | "
              f"{fmt(v['accuracy'])} | {fmt(v['brier'])} |")
        A("")
        A(f"Permutation p for the attribute model: "
          f"{fmt(pr['models']['M2_attributes'].get('permutation_p'), 4)}")
        if pr.get("caveat"):
            A(f"Caveat: {pr['caveat']}")
    else:
        A(f"Not computable: {pr['reason']}")
    A("")

    tr = res["translation"]
    A("## Translation into the model's parameters")
    A("")
    A(tr.get("disclaimer", f"Not computable: {tr.get('reason')}"))
    A("")
    e, b = tr.get("epsilon", {}), tr.get("beta", {})
    if e.get("computable"):
        A(f"- Per-practitioner error rate q = {fmt(e['per_rater_error_rate_q'])} "
          f"(95% CI [{fmt(e['q_ci95'][0])}, {fmt(e['q_ci95'][1])}]) implies "
          f"epsilon in [{fmt(e['implied_epsilon_range'][0])}, "
          f"{fmt(e['implied_epsilon_range'][1])}]; swept values inside that range: "
          f"{e['paper_grid_values_inside_range']}")
    else:
        A(f"- epsilon: not computable ({e.get('reason')})")
    if b.get("computable"):
        A(f"- Attribute-model AUC {fmt(b['measured_auc'])} places the recoverable "
          f"share in the range {fmt(b['implied_beta_low'])} to "
          f"{fmt(b['implied_beta_deattenuated'])}, the upper end correcting for a "
          f"majority-label error rate of {fmt(b['majority_label_error_rate'])}. "
          f"Paper's headline band: {b['paper_beta_band']}. {b['reading']}")
    else:
        A(f"- beta: not computable ({b.get('reason')})")
    A("")

    A("## Pre-registered primary family (Holm-adjusted)")
    A("")
    A("| hypothesis | raw p | Holm p | verdict |")
    A("|---|---|---|---|")
    for k, v in res["primary_family"].items():
        A(f"| {k} | {fmt(v['raw_p'], 4)} | {fmt(v['holm_p'], 4)} | {v['verdict']} |")
    A("")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


BOUNDARY = (
    "The pilot asks practitioners to rank pairs of real work orders drawn from the "
    "same corpus the study models; it does not place a practitioner inside the "
    "dispatch loop. It therefore grounds two premises of the supervisor model, that "
    "an urgency ordering exists which practitioners share beyond the recorded "
    "priority class and that it is partly predictable from the observable order "
    "attributes the estimator reads, and it grounds nothing else. It does not "
    "validate the correction loop, the dispatching results, or any reported "
    "reduction in true weighted tardiness, every one of which is measured against a "
    "simulated supervisor.")


H4_GATE_RATE = 0.60         # pre-registered floor on pooled repeat consistency


def _h4_gate(within: dict) -> dict:
    """Pre-registered instrument check, reported outside the multiplicity family.

    The repeats measure whether a practitioner gives the same answer to the same
    pair twice. They are a check on the instrument, so the verdict is stated as a
    gate on the estimate and its interval rather than as a hypothesis test: the
    manuscript may quote H1 to H3 without qualification only if the pooled rate
    clears the floor and its exact interval excludes chance.
    """
    if not within.get("computable"):
        return dict(passed=False, reason=within.get("reason"),
                    consequence=("report H1 to H3 with an explicit note that "
                                 "within-rater consistency could not be measured"))
    rate = within["pooled_rate"]
    lo = within["ci95_exact"][0]
    ok = bool(rate >= H4_GATE_RATE and lo > 0.5)
    return dict(
        passed=ok, pooled_rate=rate, ci95_exact=within["ci95_exact"],
        floor=H4_GATE_RATE,
        binomial_p_greater_than_chance=within["binomial_p_greater_than_chance"],
        consequence=("The instrument elicited a stable judgement, so H1 to H3 "
                     "may be quoted without a consistency caveat."
                     if ok else
                     "the manuscript must state that the instrument did not "
                     "demonstrate stable within-rater judgement at this sample "
                     "size, and must attach that caveat wherever H1 to H3 are "
                     "quoted"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, default=PILOT / "y3_w8_manifest.csv")
    ap.add_argument("--responses", type=Path, default=PILOT / "responses")
    ap.add_argument("--out-dir", type=Path, default=RES)
    ap.add_argument("--tag", default="pilot_analysis",
                    help="output basename; the self-test uses its own")
    args = ap.parse_args()

    if not args.manifest.exists():
        sys.exit(f"manifest not found: {args.manifest}\n"
                 f"run scripts/y3_w8_pilot_build.py first")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    man_rows = read_manifest(args.manifest)
    man = {m["item_id"]: m for m in man_rows}
    first = [m for m in man_rows if m["presentation"] == 1]
    design = dict(
        n_items=len(man_rows), n_unique_pairs=len(first),
        n_repeat_pairs=len({m["pair_id"] for m in man_rows if m["is_repeat"] == 1}),
        strata={st: sum(1 for m in first if m["stratum"] == st) for st in STRATA},
        campuses=sorted({m["campus"] for m in first}),
    )

    resp, load = read_responses(args.responses, man)
    R = len(resp)

    res = dict(
        boundary_statement=BOUNDARY,
        preregistration=str((args.out_dir / "PREREGISTRATION.md")),
        inputs=dict(manifest=str(args.manifest), responses_dir=str(args.responses),
                    load=load),
        design=design, n_raters=R,
    )

    if R == 0:
        res.update(agreement=na("no responses yet"),
                   within_rater=na("no responses yet"),
                   class_test=na("no responses yet"),
                   predictability=na("no responses yet"),
                   translation=dict(computable=False, reason="no responses yet"),
                   primary_family={},
                   instrument_check_H4=dict(passed=False, reason="no responses yet"))
        print(f"No responses found in {args.responses}.")
        print(f"Design ready: {design['n_unique_pairs']} pairs "
              f"({design['strata']}), {design['n_items']} presented items.")
    else:
        agreement = block_agreement(first, resp, rng)
        within = block_within_rater(man_rows, resp)
        class_test = block_class(first, resp)
        predict = block_predictability(first, resp, rng)
        translation = block_translation(agreement, predict, R)
        res.update(agreement=agreement, within_rater=within, class_test=class_test,
                   predictability=predict, translation=translation)

        # The pre-registered primary family holds the three claims that reach the
        # manuscript. Within-rater consistency is an instrument check, not a
        # claim, and is deliberately kept out of the family: including a
        # deliberately small, predictably underpowered test would tax the three
        # real claims through the multiplicity correction for nothing.
        raw = {
            "H1_raters_agree_above_chance":
                agreement.get("permutation_p") if agreement.get("computable") else None,
            "H2_attributes_predict_majority":
                predict["models"]["M2_attributes"].get("permutation_p")
                if predict.get("computable") else None,
            "H3_class_does_not_settle_order":
                class_test.get("binomial_p_vs_settles_null")
                if class_test.get("computable") else None,
        }
        adj = holm({k: v for k, v in raw.items() if v is not None})
        fam = {}
        for k, v in raw.items():
            if v is None:
                fam[k] = dict(raw_p=None, holm_p=None,
                              verdict="not computable at this sample size")
            else:
                fam[k] = dict(raw_p=float(v), holm_p=float(adj[k]),
                              verdict=("supported" if adj[k] < 0.05
                                       else "not supported at alpha = 0.05"))
        res["primary_family"] = fam
        res["instrument_check_H4"] = _h4_gate(within)

    jpath = args.out_dir / f"{args.tag}.json"
    mpath = args.out_dir / f"{args.tag}.md"
    with open(jpath, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    write_markdown(mpath, res)

    # ---- console ---------------------------------------------------------- #
    print("\n=== W8 practitioner pilot ===")
    print(f"responses: {R}  | pairs: {design['n_unique_pairs']}  | "
          f"items: {design['n_items']}  | strata: {design['strata']}")
    if R:
        ag = res["agreement"]
        print("H1 agreement            :",
              (f"alpha={fmt(ag['krippendorff_alpha'])} "
               f"CI[{fmt(ag['ci95'][0])},{fmt(ag['ci95'][1])}] "
               f"perm p={fmt(ag['permutation_p'], 4)}")
              if ag.get("computable") else f"n/a ({ag['reason']})")
        wr = res["within_rater"]
        g = res.get("instrument_check_H4", {})
        print("   within-rater check   :",
              (f"{wr['pooled_consistent']}/{wr['pooled_trials']} = "
               f"{fmt(wr['pooled_rate'])} CI[{fmt(wr['ci95_exact'][0])},"
               f"{fmt(wr['ci95_exact'][1])}]  gate "
               f"{'PASSED' if g.get('passed') else 'NOT PASSED'}")
              if wr.get("computable") else f"n/a ({wr['reason']})")
        cl = res["class_test"]
        print("H3 majority vs class    :",
              (f"{cl['majority_agrees_with_class']}/{cl['n_scored']} = "
               f"{fmt(cl['rate'])} CI[{fmt(cl['ci95_wilson'][0])},"
               f"{fmt(cl['ci95_wilson'][1])}]")
              if cl.get("computable") else f"n/a ({cl['reason']})")
        pr = res["predictability"]
        if pr.get("computable"):
            for k, v in pr["models"].items():
                print(f"H2 {k:26s}: AUC={fmt(v['auc'])} "
                      f"CI[{fmt(v['auc_ci95'][0])},{fmt(v['auc_ci95'][1])}] "
                      f"acc={fmt(v['accuracy'])}")
        else:
            print("H2 predictability       :", f"n/a ({pr['reason']})")
        tr = res["translation"]
        e, b = tr.get("epsilon", {}), tr.get("beta", {})
        print("translation epsilon     :",
              (f"q={fmt(e['per_rater_error_rate_q'])} -> eps in "
               f"[{fmt(e['implied_epsilon_range'][0])},"
               f"{fmt(e['implied_epsilon_range'][1])}]")
              if e.get("computable") else f"n/a ({e.get('reason')})")
        print("translation beta        :",
              (f"AUC={fmt(b['measured_auc'])} -> beta in "
               f"[{fmt(b['implied_beta_low'])}, "
               f"{fmt(b['implied_beta_deattenuated'])}]")
              if b.get("computable") else f"n/a ({b.get('reason')})")
        print("\nprimary family (Holm):")
        for k, v in res["primary_family"].items():
            print(f"  {k:34s} raw p={fmt(v['raw_p'], 4):>8s} "
                  f"holm={fmt(v['holm_p'], 4):>8s}  {v['verdict']}")
    print(f"\nwrote {jpath}\n      {mpath}")


if __name__ == "__main__":
    main()
