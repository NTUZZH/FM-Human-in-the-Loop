"""Paper Y3 -- Phase P6 real-data anchor for the beta>0 premise.

Motivation: real-data evidence that a latent, partially feature-predictable
urgency structure exists BEYOND the recorded priority class (the paper's beta>0
assumption, where the latent shift is xi = sqrt(beta) f(x) + sqrt(1-beta) z and
x is the campus-agnostic feature set {trade, log1p(p_bh), release day-of-week}).

WHAT THIS SCRIPT CAN AND CANNOT USE
-----------------------------------
The released replay instances carry, per order:
    trade, p_bh (= LaborHours), release_bh (= to_bh(WOStartDate)),
    due_bh, priority (recorded class), weight, building, is_pm.
Two of these are EXACT deterministic functions of the recorded class and so
carry zero information beyond it (verified in overlay.py, 0/36306 mismatch):
    weight  = w(class)              {1:8, 2:4, 3:2, 4:1}
    due_bh  = release_bh + SLA(class){1:8, 2:24, 3:80, 4:171.4}
The realized completion duration that the class was actually calibrated on
(dur_days = WOEndDate - WOStartDate) lives only in the raw corpus, which is not
shipped; only per-class / per-raw-value medians survive (priority_reliability.csv,
priority_mapping.csv). So the preferred proxy (realized completion tardiness) is
NOT available per order.

PROXY USED (fallback #2: per-order duration-vs-class residual)
--------------------------------------------------------------
Target y = log1p(p_bh), the realized labour content of the order. This is the
one per-order realized magnitude that is NOT a function of the recorded class
(the class was calibrated on calendar completion duration and PM/keyword rules,
not on labour hours). We ask, per campus and OUT OF SAMPLE (fit on the split=train
orders, score on split=test), how much of its variance the recorded class alone
explains versus the campus-agnostic observable features on top of the class:
    (a) class-only     y ~ class
    (b) class+features y ~ class + trade + release-day-of-week
    (c) features-only  y ~ trade + release-day-of-week
increment = R2(class+features) - R2(class-only).
log1p(p_bh) is the paper's third latent feature, so it is EXCLUDED from the
feature block here (it is the target); the two remaining paper features (trade,
day-of-week) are exactly the campus-agnostic block the overlay uses.

LIMITATION (stated, not hidden): labour hours is a job-size / work-content
proxy, not a direct urgency measure; a completion-time urgency measure would be
confounded by crew load and job size and is unavailable per order here. This is
therefore SUGGESTIVE evidence that observable structure beyond the recorded
class exists and is partly feature-predictable, not a causal urgency claim.

Additive only. Writes results/y3_p6/residual_structure.{csv,json}. Raw data only.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# Reuse the paper's EXACT campus-agnostic feature definitions.
from fmwos.hitl.overlay import TRADE_VOCAB, _TRADE_IDX, _day_index, N_TRADES, N_DAYS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
INST = os.path.join(ROOT, "data", "processed", "instances")
INDEX = os.path.join(INST, "index.csv")
OUT_DIR = os.path.join(ROOT, "results", "y3_p6")
CAMPUSES = [1, 2, 5, 9, 10, 12]
CLASSES = [1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# Load deduplicated per-order table (replay track only = real-corpus replays) #
# --------------------------------------------------------------------------- #
def load_orders():
    """Return {campus: {wid: rec}}. One row per unique work order.

    A work order recurs across overlapping first-N windows; trade/p_bh/priority/
    is_pm are anchor-invariant, so we dedup by id and keep the first occurrence.
    release_bh (hence the day-of-week index) IS anchor-relative, so we take it
    from the first occurrence in index order and flag mismatches for honesty.
    split is a temporal campus-span partition and is constant for a given id.
    """
    with open(INDEX) as fh:
        idx = list(csv.DictReader(fh))
    per = {c: {} for c in CAMPUSES}
    dow_mismatch = defaultdict(int)   # same id, different anchor-relative day
    field_mismatch = 0
    for r in idx:
        if r["track"] != "replay":
            continue
        camp = int(r["campus"])
        with open(os.path.join(INST, r["path"])) as fh:
            inst = json.load(fh)
        for wo in inst["work_orders"]:
            wid = wo["id"]
            dow = _day_index(float(wo["release_bh"]))
            rec = per[camp].get(wid)
            if rec is None:
                per[camp][wid] = dict(
                    trade=wo["trade"], p=float(wo["p_bh"]), pri=int(wo["priority"]),
                    pm=bool(wo["is_pm"]), split=r["split"], dow=dow,
                )
            else:
                if (rec["trade"] != wo["trade"] or rec["pri"] != int(wo["priority"])
                        or abs(rec["p"] - float(wo["p_bh"])) > 1e-9):
                    field_mismatch += 1
                if rec["dow"] != dow:
                    dow_mismatch[camp] += 1
    return per, dict(dow_mismatch), field_mismatch


# --------------------------------------------------------------------------- #
# Drop-first one-hot design (full rank; train defines the column set)         #
# --------------------------------------------------------------------------- #
class Design:
    """Builds an intercept + drop-first one-hot design matrix.

    Column levels are fixed from the TRAIN rows so train and test share columns;
    a test level unseen in train maps to the (all-zero) base cell.
    """

    def __init__(self, use_class, use_trade, use_dow, train_recs):
        self.use_class, self.use_trade, self.use_dow = use_class, use_trade, use_dow
        self.class_cols = self._levels([r["pri"] for r in train_recs], CLASSES) if use_class else []
        self.trade_cols = self._levels([r["trade"] for r in train_recs], list(TRADE_VOCAB)) if use_trade else []
        self.dow_cols = self._levels([r["dow"] for r in train_recs], list(range(N_DAYS))) if use_dow else []

    @staticmethod
    def _levels(vals, order):
        present = [lv for lv in order if lv in set(vals)]
        return present[1:]           # drop first present level (its the base)

    def matrix(self, recs):
        n = len(recs)
        blocks = [np.ones((n, 1))]   # intercept
        if self.use_class:
            blocks.append(np.array([[1.0 if r["pri"] == lv else 0.0 for lv in self.class_cols] for r in recs]))
        if self.use_trade:
            blocks.append(np.array([[1.0 if r["trade"] == lv else 0.0 for lv in self.trade_cols] for r in recs]))
        if self.use_dow:
            blocks.append(np.array([[1.0 if r["dow"] == lv else 0.0 for lv in self.dow_cols] for r in recs]))
        blocks = [b.reshape(n, -1) for b in blocks if b.size or b.shape == (n, 0)]
        return np.hstack(blocks) if blocks else np.ones((n, 1))

    @property
    def n_params(self):
        return 1 + len(self.class_cols) + len(self.trade_cols) + len(self.dow_cols)


def _r2_rho(y_te, pred):
    ss_res = float(np.sum((y_te - pred) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rho = float(spearmanr(pred, y_te).statistic) if np.std(pred) > 0 else float("nan")
    return r2, rho


def _predict(train_recs, test_recs, use_class, use_trade, use_dow):
    y_tr = np.array([math.log1p(r["p"]) for r in train_recs])
    des = Design(use_class, use_trade, use_dow, train_recs)
    beta, *_ = np.linalg.lstsq(des.matrix(train_recs), y_tr, rcond=None)
    return des.matrix(test_recs) @ beta, des.n_params


def fit_eval(train_recs, test_recs, use_class, use_trade, use_dow):
    """OLS fit on train, out-of-sample R2 and Spearman rho on test (one split)."""
    pred, npar = _predict(train_recs, test_recs, use_class, use_trade, use_dow)
    r2, rho = _r2_rho(np.array([math.log1p(r["p"]) for r in test_recs]), pred)
    return r2, rho, npar


def cv_eval(recs, use_class, use_trade, use_dow, k=5, seed=301):
    """K-fold CV: pool held-out predictions, then R2/rho over the pool.

    Balanced folds make this a campus-symmetric existence test for recoverable
    structure, robust to the uneven temporal train/test proportions.
    """
    rng = np.random.default_rng(seed)
    n = len(recs)
    fold = rng.integers(0, k, size=n)
    pred = np.empty(n)
    npar = 0
    for f in range(k):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        p, npar = _predict([recs[i] for i in tr], [recs[i] for i in te],
                           use_class, use_trade, use_dow)
        pred[te] = p
    y = np.array([math.log1p(r["p"]) for r in recs])
    r2, rho = _r2_rho(y, pred)
    return r2, rho, npar


# --------------------------------------------------------------------------- #
# Corroborating aggregate evidence: class is a lossy encoder of realized       #
# completion duration (the quantity the class was actually calibrated on).     #
# --------------------------------------------------------------------------- #
def load_reliability():
    path = os.path.join(OUT_DIR, "priority_reliability.csv")
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[int(r["campus"])] = r
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    per, dow_mismatch, field_mismatch = load_orders()
    rel = load_reliability()

    rows = []
    detail = {}
    for c in CAMPUSES:
        recs = list(per[c].values())
        train = [r for r in recs if r["split"] == "train"]
        test = [r for r in recs if r["split"] == "test"]
        # PRIMARY: balanced 5-fold CV (campus-symmetric existence test).
        r2_c, rho_c, np_c = cv_eval(recs, True, False, False)
        r2_cf, rho_cf, np_cf = cv_eval(recs, True, True, True)
        r2_f, rho_f, np_f = cv_eval(recs, False, True, True)
        # ROBUSTNESS: the paper's temporal train->test split.
        t_c = fit_eval(train, test, True, False, False)[0]
        t_cf = fit_eval(train, test, True, True, True)[0]
        t_f = fit_eval(train, test, False, True, True)[0]
        rr = rel.get(c, {})
        row = dict(
            campus=c, proxy="log1p(p_bh) [realized labour hours]",
            n_total=len(recs), n_train=len(train), n_test=len(test),
            r2_class_only=round(r2_c, 4), r2_class_feat=round(r2_cf, 4),
            r2_feat_only=round(r2_f, 4), r2_increment=round(r2_cf - r2_c, 4),
            rho_class_only=round(rho_c, 4), rho_class_feat=round(rho_cf, 4),
            rho_feat_only=round(rho_f, 4),
            # corroborating class-vs-realized-completion-duration reliability:
            class_completion_spearman=rr.get("spearman_rho_rank_vs_completion", "n/a"),
            class_duration_monotone=rr.get("class_duration_monotone", "n/a"),
        )
        rows.append(row)
        detail[str(c)] = dict(
            row, n_params_class=np_c, n_params_classfeat=np_cf, n_params_feat=np_f,
            dow_first_occ_mismatch=dow_mismatch.get(c, 0),
            temporal_r2_class_only=round(t_c, 4), temporal_r2_class_feat=round(t_cf, 4),
            temporal_r2_feat_only=round(t_f, 4), temporal_r2_increment=round(t_cf - t_c, 4),
        )

    fields = ["campus", "proxy", "n_total", "n_train", "n_test",
              "r2_class_only", "r2_class_feat", "r2_feat_only", "r2_increment",
              "rho_class_only", "rho_class_feat", "rho_feat_only",
              "class_completion_spearman", "class_duration_monotone"]
    with open(os.path.join(OUT_DIR, "residual_structure.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    incs = [r["r2_increment"] for r in rows]
    n_tot = sum(r["n_total"] for r in rows)
    wavg_inc = sum(r["r2_increment"] * r["n_total"] for r in rows) / n_tot
    summary = dict(
        proxy="log1p(p_bh) = realized labour content (LaborHours) per order",
        why_this_proxy=("weight and due_bh are exact functions of the recorded class "
                        "(w(c); r+SLA(c)); the calendar completion duration the class was "
                        "calibrated on is only in the unshipped raw corpus. Labour hours is "
                        "the one per-order realized magnitude not built from the class."),
        features="campus-agnostic {trade one-hot (14-vocab), release day-of-week one-hot}; "
                 "log1p(p_bh) is the target so it is excluded from features",
        oos_scheme="primary: seeded 5-fold CV, pooled held-out predictions (balanced). "
                   "robustness (per_campus.temporal_*): paper split=train -> split=test.",
        temporal_note="campus 2 has only 1401 train orders (5% of its orders, an early window); "
                      "its temporal-split R2 is negative and unreliable, so the 5-fold CV is the "
                      "primary read for it. All other campuses agree in sign under both schemes.",
        limitation=("labour hours is a job-size/work-content proxy, not urgency; a completion-time "
                    "urgency measure would be confounded by crew load and job size and is unavailable "
                    "per order. Suggestive of recoverable observable structure beyond the recorded "
                    "class, not a causal urgency measure."),
        increment_min=round(min(incs), 4), increment_max=round(max(incs), 4),
        increment_median=round(float(np.median(incs)), 4),
        increment_n_weighted_mean=round(wavg_inc, 4),
        n_orders_total=n_tot, field_mismatch_across_dupes=field_mismatch,
        per_campus=detail,
    )
    with open(os.path.join(OUT_DIR, "residual_structure.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("proxy: log1p(p_bh) (realized labour hours); primary = 5-fold CV, per campus")
    print("camp   n_tot   R2_class  R2_cls+ft  R2_ft   incr   rho_ft  classMono  temporal_incr")
    for r in rows:
        print("%4d %7d   %7.3f  %8.3f  %6.3f  %6.3f  %6.3f   %-6s     %+.3f"
              % (r["campus"], r["n_total"],
                 r["r2_class_only"], r["r2_class_feat"], r["r2_feat_only"],
                 r["r2_increment"], r["rho_feat_only"], r["class_duration_monotone"],
                 detail[str(r["campus"])]["temporal_r2_increment"]))
    print("\nincrement: median %.3f  n-weighted mean %.3f  range [%.3f, %.3f]"
          % (summary["increment_median"], summary["increment_n_weighted_mean"],
             summary["increment_min"], summary["increment_max"]))
    print("field mismatch across duplicate ids:", field_mismatch,
          "| dow first-occ mismatches:", dow_mismatch)
    print("wrote", os.path.join(OUT_DIR, "residual_structure.csv"),
          "and residual_structure.json")


if __name__ == "__main__":
    main()
