#!/usr/bin/env python
"""Paper Y3 -- real-data calibration of the supervisor model against FMUCD.

WHAT QUESTION THIS ANSWERS
--------------------------
The manuscript models a facility supervisor who holds a hidden urgency, reviews a
share of dispatch decisions, and overrides the dispatcher when the true urgency
disagrees with the recorded priority class. Two parameters govern that model and
are currently swept as free axes with no real-data anchor:

  beta     the share of the hidden urgency that is a recoverable function of
           observable order features (formulation.tex, xi = sqrt(beta) f(x) +
           sqrt(1-beta) z);
  epsilon  the supervisor's override noise.

FMUCD records no review, no reassignment, no reopen and no re-prioritisation, so
neither parameter can be point-identified from it. What the corpus does record is
the sequence in which work orders were closed. This script asks two questions of
that sequence, per campus:

  A1  How often was an order served while a strictly more urgent open order was
      still waiting? Call that an OBSERVED DEVIATION. It is the observational
      analogue of an override, and it is an UPPER BOUND on how often the recorded
      priority failed to determine the order of service, not a measured override
      rate. Trade eligibility, crew availability, travel, batching, parts lead
      time and access windows produce the same footprint and are not recorded.

  A2  Are those deviations systematic or random? Out of sample, per campus, can
      observable order features (system code, log size, day of week raised,
      planned vs unplanned) predict a deviation BEYOND what the recorded priority
      class alone predicts? If yes, a real dispatcher's departures from the
      recorded priority carry structure an estimator could learn, which is the
      paper's premise, and the recovered share of the class-residual variation is
      the empirical analogue of beta. If no, the corpus supports beta ~ 0 and the
      paper must say so.

  A3  Secondary: does the free text of WODescription add predictive signal over
      the coded features? That is, is supervisor knowledge already written down
      but unread by the priority rule?

EPSILON. The corpus contains no record of a supervisor reviewing a dispatch, so
epsilon is not identifiable. The one quantity the corpus does bound is the share
of the deviation signal that NO observable feature explains, which is an upper
bound on the noise share of the departure process. It is reported as such and is
not epsilon.

INHERITED FILTERS (not re-derived; see the module docstrings named below)
------------------------------------------------------------------------
fmwos.io.load_raw / fmwos.io.clean supply R1 typed date parsing with the explicit
"%Y-%m-%d %H:%M:%S" format, R2 drop rows with no WOID / UniversityID /
WOStartDate, R3 drop non-positive LaborHours, R7 aggregate duplicate labour lines
per (UniversityID, WOID), R4 cap LaborHours at the global post-aggregation p99.5,
R6 trade = UNIFORMAT SystemCode. fmwos.calib.build_priority_mapping and
fmwos.calib.priority_class_series supply the priority mapping v2 (R5a preventive
-> class 4, R5b keyword -> class, R5c campus numeric scale with its direction set
by the Spearman sign against median realised corrective completion duration, R5d
rare/missing -> class 3) and the campus set {1,2,5,9,10,12}, which is the set of
timestamp-complete campuses. Every one of those is reproduced here by calling the
committed helpers, and the reconstruction is asserted against the committed
results/p1_calib/priority_mapping.csv and results/y3_p6/priority_reliability.csv.

Run:  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c 20-23 \
        python scripts/y3_calib_overrides.py
Writes results/y3_calib/. Reads the raw corpus only. Additive; touches nothing else.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fmwos import calib, io  # noqa: E402

RAW_DEFAULT = "/home/ziheng/PaperY-FMScheduling/data/raw/FMUCD.csv"
OUT_DIR = os.path.join(ROOT, "results", "y3_calib")
REF_MAPPING = os.path.join(ROOT, "results", "p1_calib", "priority_mapping.csv")
REF_RELIABILITY = os.path.join(ROOT, "results", "y3_p6", "priority_reliability.csv")
REF_OVERVIEW = os.path.join(ROOT, "results", "p0_profile", "overview.json")

SEED = 301
CLASSES = (1, 2, 3, 4)
DAY_S = 86400
WINDOW_DAYS = 30            # "recent actionable backlog" window
N_FOLDS = 5
TEXT_MAX_EVENTS = 200_000   # per campus cap for the free-text analysis
TEXT_MAX_FEATURES = 30_000

# Expected values from the committed pipeline (data-accuracy gates D2-D4).
EXP_ROWS_RAW = 3_731_442            # results/p0_profile/overview.json rows_raw
EXP_ROWS_POST_R2R3 = 1_906_865      # docs/decision_log.md 2026-07-04
EXP_WORK_ORDERS = 1_454_039         # docs/decision_log.md 2026-07-04
EXP_LABOR_CAP = 90.86               # docs/decision_log.md 2026-07-04 (2 dp)

# Candidate-set restrictions, loosest to tightest. `keys` name the columns that
# must match between the served order and a candidate; `window` caps how long an
# order stays in the comparison backlog after it was raised.
VARIANTS = [
    ("all", (), None),
    ("trade", ("trade",), None),
    ("bldg", ("bldg",), None),
    ("trade_bldg", ("trade", "bldg"), None),
    ("trade_w30", ("trade",), WINDOW_DAYS),
    ("trade_bldg_w30", ("trade", "bldg"), WINDOW_DAYS),
    ("trade_size_w30", ("trade", "sizeq"), WINDOW_DAYS),
]
BUILDING_VARIANTS = {"bldg", "trade_bldg", "trade_bldg_w30"}

# Labels carried into the predictability analysis (pre-registered, not chosen
# after seeing A1): the tightest restriction available on every campus, plus the
# loosest defensible one, on both populations.
PRED_LABELS = [("cm", "trade_w30"), ("cm", "trade"), ("cm", "trade_bldg_w30"),
               ("all", "trade_w30"), ("all", "trade")]
PRED_PRIMARY = ("cm", "trade_w30")


# ==========================================================================
# 1. Load and clean, reusing the committed helpers, and gate on the artefacts
# ==========================================================================
def load_clean(raw_path: str, chunksize: int = 500_000):
    """Chunked typed load + the committed cleaning rules.

    R2 and R3 are row-local, so applying them per chunk before calling
    ``io.clean`` on the concatenation is byte-identical to calling ``io.clean``
    on the whole frame, and roughly halves peak memory. Everything after that
    (R7 aggregation, the global R4 cap, R6) is done by ``io.clean`` itself.
    """
    usecols = list(io.USECOLS) + ["WODescription"]
    dtypes = dict(io.DTYPES)
    dtypes["WODescription"] = "string"
    parts, rows_raw = [], 0
    bad_start = bad_end = 0
    for ch in pd.read_csv(raw_path, usecols=usecols, dtype=dtypes,
                          chunksize=chunksize):
        rows_raw += len(ch)
        for c in io.DATE_COLS:
            s = ch[c]
            ch[c] = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
            n_bad = int((s.notna() & ch[c].isna()).sum())
            if c == "WOStartDate":
                bad_start += n_bad
            else:
                bad_end += n_bad
        ch = ch[ch["WOID"].notna() & ch["UniversityID"].notna()
                & ch["WOStartDate"].notna()]
        ch = ch[ch["LaborHours"].notna() & (ch["LaborHours"] > 0)]
        parts.append(ch)
    raw = pd.concat(parts, axis=0)
    del parts
    rows_post = len(raw)
    clean, audit = io.clean(raw)
    del raw
    meta = dict(rows_raw=rows_raw, rows_post_r2_r3=rows_post,
                unparseable_wostartdate=bad_start, unparseable_woenddate=bad_end,
                **{k: (float(v) if isinstance(v, float) else int(v))
                   for k, v in audit.items()})
    return clean, meta


def data_checks(clean: pd.DataFrame, meta: dict, mapping: pd.DataFrame,
                raw_path: str) -> dict:
    """D1-D8. Every check either passes or is recorded with its discrepancy."""
    checks: dict = {}
    sha = io.sha256_of(raw_path)
    checks["D1_raw_sha256"] = dict(value=sha, expected=io.RAW_SHA256,
                                   passed=sha == io.RAW_SHA256)
    with open(REF_OVERVIEW) as fh:
        exp_raw = int(json.load(fh)["rows_raw"])
    checks["D2_rows_raw"] = dict(value=meta["rows_raw"], expected=exp_raw,
                                 passed=meta["rows_raw"] == exp_raw == EXP_ROWS_RAW)
    checks["D3_rows_post_r2_r3"] = dict(
        value=meta["rows_post_r2_r3"], expected=EXP_ROWS_POST_R2R3,
        passed=meta["rows_post_r2_r3"] == EXP_ROWS_POST_R2R3)
    checks["D3b_work_orders_after_r7"] = dict(
        value=meta["R7_work_orders_after_dedup"], expected=EXP_WORK_ORDERS,
        passed=meta["R7_work_orders_after_dedup"] == EXP_WORK_ORDERS)
    cap = meta["R4_labor_cap_hours"]
    checks["D4_labor_cap_p995_hours"] = dict(
        value=round(cap, 4), expected=EXP_LABOR_CAP,
        passed=abs(round(cap, 2) - EXP_LABOR_CAP) < 1e-9)

    # D5: reconstructed mapping vs the committed one.
    ref = pd.read_csv(REF_MAPPING)
    a, b = mapping.copy(), ref.copy()
    a["raw_value"] = a["raw_value"].astype(str)
    b["raw_value"] = b["raw_value"].astype(str)
    key = ["campus", "raw_value", "is_pm_split"]
    m = a.merge(b, on=key, how="outer", suffixes=("_mine", "_ref"), indicator=True)
    unmatched = int((m["_merge"] != "both").sum())
    cls_diff = int((m["mapped_class_mine"].astype("Int64")
                    != m["mapped_class_ref"].astype("Int64")).sum())
    rule_diff = int((m["rule_mine"].astype(str) != m["rule_ref"].astype(str)).sum())
    row_delta = (m["rows_mine"].astype("Int64") - m["rows_ref"].astype("Int64"))
    row_diff_rows = int((row_delta != 0).sum())
    checks["D5_priority_mapping_vs_committed"] = dict(
        n_rows=int(len(m)), unmatched_keys=unmatched,
        mapped_class_mismatches=cls_diff, rule_mismatches=rule_diff,
        row_count_mismatches=row_diff_rows,
        max_abs_row_count_delta=int(row_delta.abs().max()),
        passed=(unmatched == 0 and cls_diff == 0 and rule_diff == 0
                and int(row_delta.abs().max()) <= 1),
        note=("R7 picks the dominant labour line by max LaborHours with a "
              "non-stable sort, so a single campus-9 work order with tied hours "
              "falls on the other side of the pm/cm split than in the committed "
              "run. No mapped class, rule or campus total changes."))

    # D6: per-campus cleaned counts vs the committed reliability descriptor.
    rel = pd.read_csv(REF_RELIABILITY)
    camp = clean["UniversityID"].astype("int64")
    per = {int(c): int((camp == c).sum()) for c in calib.CAMPUSES}
    exp = {int(r["campus"]): int(r["n_rows"]) for _, r in rel.iterrows()}
    checks["D6_per_campus_rows"] = dict(value=per, expected=exp,
                                        passed=per == exp)

    # D7: date-parsing and timeline sanity, reported not guessed.
    neg = (clean["WOEndDate"] - clean["WOStartDate"]) < pd.Timedelta(0)
    checks["D7_date_parsing"] = dict(
        unparseable_wostartdate=meta["unparseable_wostartdate"],
        unparseable_woenddate=meta["unparseable_woenddate"],
        work_orders_with_end_before_start=int(neg.sum()),
        note=("every unparseable WOStartDate is a date-only value on campuses "
              "3,4,6,7,8,11; those campuses are dropped by inherited rule R2, "
              "which is why the pipeline's campus set has six members."))

    # D8: campuses present after cleaning but outside the inherited set.
    outside = sorted(int(c) for c in set(camp.unique()) - set(calib.CAMPUSES))
    checks["D8_campus_set"] = dict(
        analysed_set=list(calib.CAMPUSES), present_but_outside=outside,
        rows_outside=int((~camp.isin(calib.CAMPUSES)).sum()),
        note=("the inherited set is docs/decision_log.md 2026-07-04 'Campus "
              "set': the six timestamp-complete campuses. The residue outside "
              "it is the handful of rows on other campuses whose start "
              "timestamp happened to carry a time component."))
    return checks


# ==========================================================================
# 2. Per-campus population and disposition
# ==========================================================================
def campus_population(clean: pd.DataFrame, campus: int) -> pd.DataFrame:
    s = clean[clean["UniversityID"].astype("int64") == campus].copy()
    s["campus"] = campus
    s["bldg"] = s["BuildingID"].astype("string")
    s["desc"] = s["WODescription"].astype("string").fillna("").astype(str)
    # is_pm is nullable (PPM/UPM can be missing); calib.build_priority_mapping
    # resolves that with fillna(False), so use exactly the same convention.
    s["is_pm"] = s["is_pm"].fillna(False).astype(bool)
    s["raise_t"] = (s["WOStartDate"].to_numpy().astype("datetime64[s]")
                    .astype("int64"))
    # NaT casts to the int64 sentinel, not to NaN, so mask it explicitly.
    close = (s["WOEndDate"].to_numpy().astype("datetime64[s]")
             .astype("int64").astype("float64"))
    close[s["WOEndDate"].isna().to_numpy()] = np.nan
    s["close_t"] = close
    return s


def disposition(pop: pd.DataFrame, campus: int) -> dict:
    n = len(pop)
    end_ok = np.isfinite(pop["close_t"].to_numpy())
    dur = pop["close_t"].to_numpy() - pop["raise_t"].to_numpy()
    neg = int(np.sum(dur < 0))          # NaN compares False, so this is exact
    cm = pop[~pop["is_pm"].astype(bool)]
    cm_cls = cm["priority"].value_counts().to_dict()
    shares = np.array([cm_cls.get(c, 0) for c in CLASSES], dtype=float)
    shares = shares / shares.sum() if shares.sum() else shares
    ent = float(-np.sum([p * np.log(p) for p in shares if p > 0]))
    # Could a missing WOEndDate be recovered from the recorded WODuration?
    # Only if that field is actually populated there and is not identically zero.
    no_end = ~end_ok
    dur_field = pop["WODuration"].to_numpy(dtype="float64", na_value=np.nan)
    have_dur = no_end & np.isfinite(dur_field)
    dur_vals = dur_field[have_dur]
    d = dict(
        campus=campus, n_work_orders=n,
        end_timestamp_coverage=round(float(end_ok.mean()), 4),
        n_end_before_start=int(neg),
        n_woduration_without_end=int(have_dur.sum()),
        woduration_without_end_max=(round(float(dur_vals.max()), 2)
                                    if dur_vals.size else ""),
        woduration_fallback_usable=int(dur_vals.size > 0 and float(dur_vals.max()) > 0),
        n_analysable=int(np.sum(end_ok & (dur >= 0))),
        n_corrective=int(len(cm)),
        cm_class_counts=json.dumps({int(k): int(v) for k, v in
                                    sorted(cm_cls.items())}),
        cm_class_entropy_nats=round(ent, 4),
        n_cm_classes_present=int(sum(1 for c in CLASSES if cm_cls.get(c, 0) > 0)),
        building_coverage=round(float(pop["bldg"].notna().mean()), 4),
        description_coverage=round(float((pop["desc"].str.len() > 0).mean()), 4),
        n_unique_descriptions=int(pop.loc[pop["desc"].str.len() > 0,
                                          "desc"].nunique()),
    )
    reasons = []
    if d["end_timestamp_coverage"] < 0.90:
        r = ("completion timestamp on only %.1f%% of orders"
             % (100 * d["end_timestamp_coverage"]))
        if d["n_woduration_without_end"] > 0 and not d["woduration_fallback_usable"]:
            r += (", and the WODuration field is identically zero on exactly "
                  "those %d orders, so no completion time can be recovered from it"
                  % d["n_woduration_without_end"])
        elif d["n_woduration_without_end"] == 0:
            r += ", with no WODuration value to fall back on either"
        reasons.append(r)
    if d["n_cm_classes_present"] < 2:
        reasons.append("recorded class constant across corrective orders "
                       "(no deviation can exist by construction)")
    if not reasons:
        d["status"], d["status_reason"] = "headline", ""
    elif d["end_timestamp_coverage"] >= 0.40 and d["n_cm_classes_present"] >= 2:
        d["status"] = "flagged"
        d["status_reason"] = ("; ".join(reasons)
                              + "; backlog reconstructed from a subset, so the "
                                "deviation rate is biased downward")
    else:
        d["status"], d["status_reason"] = "excluded", "; ".join(reasons)
    return d


# ==========================================================================
# 3. A1 -- observed deviations
# ==========================================================================
def open_counts_by_class(enter, exit_, gcode, cls, q_time, q_gcode):
    """counts[i, c-1] = number of orders k with

        gcode[k] == q_gcode[i],  cls[k] == c,  enter[k] <= q_time[i],
        exit_[k] > q_time[i]

    i.e. the orders of class c that were already raised and still open in the
    served order's comparison group at the moment it closed. The served order
    itself has exit == q_time and so is excluded automatically.

    Vectorised: for each (group, class) cell the candidate enter and exit times
    are sorted once and every query in that group is answered with two
    ``searchsorted`` calls, so the cost is O(n log n) rather than O(n^2).
    """
    n_q = len(q_time)
    out = np.zeros((n_q, len(CLASSES)), dtype=np.int64)
    cand_order = np.lexsort((cls, gcode))
    g_sorted, c_sorted = gcode[cand_order], cls[cand_order]
    ent_sorted, ex_sorted = enter[cand_order], exit_[cand_order]
    # start of every (g, c) run in the lexsorted candidate array
    keys = g_sorted.astype(np.int64) * 8 + c_sorted.astype(np.int64)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    cell = {(int(g_sorted[s]), int(c_sorted[s])): (s, e)
            for s, e in zip(starts, ends)}

    q_order = np.argsort(q_gcode, kind="stable")
    gq = q_gcode[q_order]
    q_starts = np.flatnonzero(np.r_[True, gq[1:] != gq[:-1]])
    q_ends = np.r_[q_starts[1:], len(gq)]
    for qs, qe in zip(q_starts, q_ends):
        g = int(gq[qs])
        idx = q_order[qs:qe]
        t = q_time[idx]
        for ci, c in enumerate(CLASSES):
            se = cell.get((g, c))
            if se is None:
                continue
            s, e = se
            a = np.searchsorted(np.sort(ent_sorted[s:e]), t, side="right")
            b = np.searchsorted(np.sort(ex_sorted[s:e]), t, side="right")
            out[idx, ci] = a - b
    return out


def _brute_open_counts(enter, exit_, gcode, cls, t, g):
    m = (gcode == g) & (enter <= t) & (exit_ > t)
    return np.array([int(np.sum(m & (cls == c))) for c in CLASSES], dtype=np.int64)


def deviation_table(pop: pd.DataFrame, campus: int, rng: np.random.Generator,
                    verify: bool = True) -> tuple[list[dict], list[dict],
                                                  dict[tuple[str, str], pd.DataFrame]]:
    """Deviation statistics for every (population, candidate restriction) cell."""
    rows, by_class, by_trade, labels = [], [], [], {}
    base = pop[np.isfinite(pop["close_t"].to_numpy())].copy()
    base = base[(base["close_t"].to_numpy() - base["raise_t"].to_numpy()) >= 0]
    try:
        base["sizeq"] = pd.qcut(base["LaborHours"], 4, labels=False,
                                duplicates="drop").astype("int64")
    except ValueError:                       # a campus with a constant size
        base["sizeq"] = 0
    for population in ("cm", "all"):
        sub = base if population == "all" else base[~base["is_pm"].astype(bool)]
        for vname, keys, window in VARIANTS:
            s = sub
            if "bldg" in keys:
                s = s[s["bldg"].notna()]
            n_pop = len(s)
            if n_pop < 100 or s["priority"].nunique() < 2:
                rows.append(dict(campus=campus, population=population,
                                 variant=vname, n_pop=n_pop, n_events=0,
                                 note="insufficient population or constant class"))
                continue
            enter = s["raise_t"].to_numpy()
            close = s["close_t"].to_numpy().astype(np.int64)
            exit_ = close if window is None else np.minimum(
                close, enter + window * DAY_S)
            cls = s["priority"].to_numpy().astype(np.int64)
            if keys:
                key = s[keys[0]].astype(str)
                for k in keys[1:]:
                    key = key + "\x1f" + s[k].astype(str)
                gcode = pd.factorize(key)[0].astype(np.int64)
            else:
                gcode = np.zeros(len(s), dtype=np.int64)
            cnt = open_counts_by_class(enter, exit_, gcode, cls, close, gcode)

            if verify and len(s) > 0:
                pick = rng.choice(len(s), size=min(60, len(s)), replace=False)
                for i in pick:
                    ref = _brute_open_counts(enter, exit_, gcode, cls,
                                             close[i], gcode[i])
                    if not np.array_equal(ref, cnt[i]):
                        raise AssertionError(
                            "open-count mismatch c%d %s/%s row %d: %s vs %s"
                            % (campus, population, vname, i, ref, cnt[i]))

            n_cand = cnt.sum(axis=1)
            better = np.zeros(len(s), dtype=np.int64)
            best_open = np.full(len(s), np.nan)
            for ci, c in enumerate(CLASSES):
                better += np.where(cls > c, cnt[:, ci], 0)
            for ci, c in reversed(list(enumerate(CLASSES))):
                best_open = np.where(cnt[:, ci] > 0, float(c), best_open)
            has = n_cand > 0
            dev = (better > 0)
            gap = np.where(dev, cls - np.nan_to_num(best_open, nan=0.0), 0.0)
            frac = better / np.maximum(n_cand, 1)

            ev = has
            n_ev = int(ev.sum())
            r = dict(campus=campus, population=population, variant=vname,
                     n_pop=n_pop, n_events=n_ev,
                     n_events_no_candidate=int((~ev).sum()),
                     dev_rate=round(float(dev[ev].mean()), 4) if n_ev else np.nan,
                     n_deviations=int(dev[ev].sum()),
                     frac_more_urgent_mean=round(float(frac[ev].mean()), 4) if n_ev else np.nan,
                     frac_more_urgent_median=round(float(np.median(frac[ev])), 4) if n_ev else np.nan,
                     frac_more_urgent_p90=round(float(np.percentile(frac[ev], 90)), 4) if n_ev else np.nan,
                     class_gap_mean_on_dev=round(float(gap[ev & dev].mean()), 4) if dev[ev].sum() else np.nan,
                     n_candidates_median=int(np.median(n_cand[ev])) if n_ev else 0,
                     n_candidates_p90=int(np.percentile(n_cand[ev], 90)) if n_ev else 0,
                     note="")
            for k in (1, 2, 3):
                r["class_gap_%d_share" % k] = (
                    round(float(np.mean(gap[ev & dev] == k)), 4)
                    if dev[ev].sum() else np.nan)
            rows.append(r)

            for c in CLASSES:
                m = ev & (cls == c)
                by_class.append(dict(
                    campus=campus, population=population, variant=vname,
                    own_class=c, n_events=int(m.sum()),
                    dev_rate=round(float(dev[m].mean()), 4) if m.sum() else np.nan,
                    frac_more_urgent_mean=round(float(frac[m].mean()), 4) if m.sum() else np.nan))

            if (population, vname) in PRED_LABELS:
                lab = s.loc[ev, ["campus", "priority", "trade", "LaborHours",
                                 "is_pm", "WOStartDate", "desc"]].copy()
                lab["dev"] = dev[ev].astype(int)
                lab["close_t"] = close[ev]
                lab["n_cand"] = n_cand[ev]
                labels[(population, vname)] = lab
                # Per-system-code rates, so whatever trade-level structure the
                # model exploits is readable in interpretable units and not only
                # through a regression coefficient.
                tr = s["trade"].to_numpy()[ev]
                for t in sorted(set(tr.tolist())):
                    m = tr == t
                    by_trade.append(dict(
                        campus=campus, population=population, variant=vname,
                        trade=t, n_events=int(m.sum()),
                        dev_rate=round(float(dev[ev][m].mean()), 4),
                        share_class1=round(float(np.mean(cls[ev][m] == 1)), 4),
                        frac_more_urgent_mean=round(float(frac[ev][m].mean()), 4)))
    return rows, by_class, by_trade, labels


# ==========================================================================
# 4. A2 -- is a deviation predictable from observable features?
# ==========================================================================
def design_blocks(lab: pd.DataFrame, population: str):
    """Return {block name: dense design matrix}. Drop-first one-hots + intercept."""
    n = len(lab)
    def onehot(vals):
        codes, uniq = pd.factorize(vals)
        if len(uniq) < 2:
            return np.zeros((n, 0))
        m = np.zeros((n, len(uniq) - 1))
        keep = codes > 0
        m[np.flatnonzero(keep), codes[keep] - 1] = 1.0
        return m
    cls_b = onehot(lab["priority"].astype(int))
    trade_b = onehot(lab["trade"].astype(str))
    dow_b = onehot(lab["WOStartDate"].dt.dayofweek.astype(int))
    size_b = np.log1p(lab["LaborHours"].to_numpy(dtype=float)).reshape(-1, 1)
    parts_nosize = [trade_b, dow_b]
    parts_full = [trade_b, dow_b, size_b]
    if population == "all":
        pm_b = lab["is_pm"].astype(bool).to_numpy(dtype=float).reshape(-1, 1)
        parts_nosize.append(pm_b)
        parts_full.append(pm_b)
    coded = np.hstack(parts_full)
    coded_ns = np.hstack(parts_nosize)
    non_trade = [dow_b, size_b] + ([pm_b] if population == "all" else [])
    out = {
        "class_only": cls_b,
        "class_coded": np.hstack([cls_b, coded]),
        "class_coded_nosize": np.hstack([cls_b, coded_ns]),
        "coded_only": coded,
        # single-feature ablations: which observable actually carries the signal
        "class_trade": np.hstack([cls_b, trade_b]),
        "class_dow": np.hstack([cls_b, dow_b]),
        "class_size": np.hstack([cls_b, size_b]),
        "class_nontrade": np.hstack([cls_b] + non_trade),
    }
    if population == "all":
        out["class_pm"] = np.hstack([cls_b, pm_b])
    return out


def temporal_folds(t: np.ndarray, k: int) -> np.ndarray:
    """k contiguous blocks of the campus's closure-event timeline."""
    order = np.argsort(t, kind="stable")
    fold = np.empty(len(t), dtype=int)
    fold[order] = (np.arange(len(t)) * k) // len(t)
    return fold


def cv_scores(X: np.ndarray, y: np.ndarray, fold: np.ndarray) -> dict:
    """Pooled out-of-fold probability, AUC, balanced accuracy, Brier-R2."""
    p = np.full(len(y), np.nan)
    fold_auc = []
    tp = tn = fp = fn = 0
    for f in np.unique(fold):
        te, tr = fold == f, fold != f
        ytr = y[tr]
        if ytr.min() == ytr.max() or te.sum() == 0:
            p[te] = float(ytr.mean()) if len(ytr) else 0.5
            continue
        if X.shape[1] == 0:
            p[te] = float(ytr.mean())
        else:
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1.0
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit((X[tr] - mu) / sd, ytr)
            p[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
        yte = y[te]
        if yte.min() != yte.max():
            fold_auc.append(float(roc_auc_score(yte, p[te])))
        thr = float(ytr.mean())
        pred = p[te] >= thr
        tp += int(np.sum(pred & (yte == 1)))
        tn += int(np.sum(~pred & (yte == 0)))
        fp += int(np.sum(pred & (yte == 0)))
        fn += int(np.sum(~pred & (yte == 1)))
    auc = float(roc_auc_score(y, p)) if y.min() != y.max() else float("nan")
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(auc=auc, auc_fold_mean=float(np.mean(fold_auc)) if fold_auc else float("nan"),
                auc_fold_sd=float(np.std(fold_auc)) if fold_auc else float("nan"),
                bal_acc=float((sens + spec) / 2),
                ss_res=ss_res, ss_tot=ss_tot,
                brier_r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def predictability(labels_by_campus: dict, rng: np.random.Generator) -> list[dict]:
    rows = []
    for (campus, population, variant), lab in sorted(labels_by_campus.items()):
        y = lab["dev"].to_numpy().astype(int)
        if len(y) < 500 or y.min() == y.max():
            rows.append(dict(campus=campus, population=population, variant=variant,
                             n_events=len(y), note="degenerate label"))
            continue
        X = design_blocks(lab, population)
        t = lab["close_t"].to_numpy()
        f_temporal = temporal_folds(t, N_FOLDS)
        f_random = rng.integers(0, N_FOLDS, size=len(y))
        sc = {b: cv_scores(X[b], y, f_temporal) for b in X}
        sc_rand = {b: cv_scores(X[b], y, f_random)
                   for b in ("class_only", "class_coded")}
        base = sc["class_only"]["ss_res"]
        r = dict(campus=campus, population=population, variant=variant,
                 n_events=len(y), prevalence=round(float(y.mean()), 4),
                 auc_null_prevalence=0.5)
        for b in ("class_only", "class_coded", "class_coded_nosize", "coded_only",
                  "class_trade", "class_dow", "class_size", "class_nontrade",
                  "class_pm"):
            if b not in sc:
                continue
            r["auc_" + b] = round(sc[b]["auc"], 4)
            r["aucsd_" + b] = round(sc[b]["auc_fold_sd"], 4)
            r["balacc_" + b] = round(sc[b]["bal_acc"], 4)
            r["brierr2_" + b] = round(sc[b]["brier_r2"], 4)
            if b != "class_only" and b != "coded_only":
                r["beta_" + b] = round(1.0 - sc[b]["ss_res"]
                                       / sc["class_only"]["ss_res"], 4)
        r["auc_increment_over_class"] = round(sc["class_coded"]["auc"]
                                              - sc["class_only"]["auc"], 4)
        r["auc_increment_nosize"] = round(sc["class_coded_nosize"]["auc"]
                                          - sc["class_only"]["auc"], 4)
        # beta analogue: share of the CLASS-RESIDUAL variation of the deviation
        # indicator that observable features recover, out of sample.
        r["beta_hat"] = round(1.0 - sc["class_coded"]["ss_res"] / base, 4)
        r["beta_hat_nosize"] = round(1.0 - sc["class_coded_nosize"]["ss_res"] / base, 4)
        r["unexplained_share_upper"] = round(
            min(1.0, max(0.0, 1.0 - r["beta_hat"])), 4)
        r["auc_class_only_randomcv"] = round(sc_rand["class_only"]["auc"], 4)
        r["auc_class_coded_randomcv"] = round(sc_rand["class_coded"]["auc"], 4)
        # contamination check: with the label permuted the same pipeline must
        # score at chance. Anything above ~0.52 means the CV is leaking.
        y_perm = rng.permutation(y)
        r["auc_class_coded_permuted"] = round(
            cv_scores(X["class_coded"], y_perm, f_temporal)["auc"], 4)
        r["note"] = ""
        rows.append(r)
    return rows


# ==========================================================================
# 5. A3 -- does the free text add anything a linear model can read?
# ==========================================================================
def text_analysis(labels_by_campus: dict, rng: np.random.Generator):
    rows, terms = [], []
    for (campus, population, variant), lab in sorted(labels_by_campus.items()):
        if (population, variant) != PRED_PRIMARY:
            continue
        lab = lab[lab["desc"].str.len() > 0]
        y = lab["dev"].to_numpy().astype(int)
        if len(y) < 1000 or y.min() == y.max():
            continue
        if len(lab) > TEXT_MAX_EVENTS:
            idx = np.sort(rng.choice(len(lab), TEXT_MAX_EVENTS, replace=False))
            lab = lab.iloc[idx]
            y = lab["dev"].to_numpy().astype(int)
        n_uniq = int(lab["desc"].nunique())
        X = design_blocks(lab, population)
        t = lab["close_t"].to_numpy()
        fold = temporal_folds(t, N_FOLDS)
        docs = lab["desc"].astype(str).to_numpy()

        coded = X["class_coded"]
        sc_coded = cv_scores(coded, y, fold)
        p = np.full(len(y), np.nan)
        fold_auc = []
        for f in range(N_FOLDS):
            te, tr = fold == f, fold != f
            if y[tr].min() == y[tr].max() or te.sum() == 0:
                p[te] = float(y[tr].mean())
                continue
            vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=10,
                                  max_features=TEXT_MAX_FEATURES,
                                  sublinear_tf=True, strip_accents="unicode")
            Ttr = vec.fit_transform(docs[tr])
            Tte = vec.transform(docs[te])
            mu, sd = coded[tr].mean(0), coded[tr].std(0)
            sd[sd == 0] = 1.0
            Atr = sparse.hstack([sparse.csr_matrix((coded[tr] - mu) / sd), Ttr],
                                format="csr")
            Ate = sparse.hstack([sparse.csr_matrix((coded[te] - mu) / sd), Tte],
                                format="csr")
            clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=500)
            clf.fit(Atr, y[tr])
            p[te] = clf.predict_proba(Ate)[:, 1]
            if y[te].min() != y[te].max():
                fold_auc.append(float(roc_auc_score(y[te], p[te])))
        auc_text = float(roc_auc_score(y, p)) if y.min() != y.max() else float("nan")
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        rows.append(dict(campus=campus, population=population, variant=variant,
                         n_events=len(y), n_unique_descriptions=n_uniq,
                         prevalence=round(float(y.mean()), 4),
                         auc_class_coded=round(sc_coded["auc"], 4),
                         auc_class_coded_text=round(auc_text, 4),
                         auc_increment_text=round(auc_text - sc_coded["auc"], 4),
                         aucsd_text=round(float(np.std(fold_auc)), 4) if fold_auc else np.nan,
                         brierr2_class_coded=round(sc_coded["brier_r2"], 4),
                         brierr2_class_coded_text=round(
                             1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan, 4),
                         beta_hat_text=round(
                             1.0 - ss_res / cv_scores(X["class_only"], y, fold)["ss_res"], 4)))

        # interpretive, in-sample: which terms carry the signal
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=10,
                              max_features=TEXT_MAX_FEATURES, sublinear_tf=True,
                              strip_accents="unicode")
        T = vec.fit_transform(docs)
        mu, sd = coded.mean(0), coded.std(0)
        sd[sd == 0] = 1.0
        A = sparse.hstack([sparse.csr_matrix((coded - mu) / sd), T], format="csr")
        clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=500)
        clf.fit(A, y)
        w = clf.coef_[0][coded.shape[1]:]
        names = np.array(vec.get_feature_names_out())
        for sign, order in (("promotes_deviation", np.argsort(-w)),
                            ("suppresses_deviation", np.argsort(w))):
            for rank, j in enumerate(order[:15], 1):
                terms.append(dict(campus=campus, direction=sign, rank=rank,
                                  term=names[j], coefficient=round(float(w[j]), 4)))
    return rows, terms


# ==========================================================================
# 6. Macro snippet
# ==========================================================================
def macro_snippet(dev: pd.DataFrame, pred: pd.DataFrame, txt: pd.DataFrame,
                  disp: pd.DataFrame, trd: pd.DataFrame) -> str:
    head = [c["campus"] for _, c in disp.iterrows() if c["status"] == "headline"]
    flag = [c["campus"] for _, c in disp.iterrows() if c["status"] == "flagged"]
    L = []
    A = L.append
    A("% ---------------------------------------------------------------------------")
    A("% Real-data calibration of the supervisor model against FMUCD (Y3 calib run).")
    A("% Source dir: results/y3_calib/. Script: scripts/y3_calib_overrides.py.")
    A("% Every value below is read from the file named in its trailing comment.")
    A("% OBSERVATIONAL, NOT CAUSAL: a deviation is an upper bound on how often the")
    A("% recorded priority failed to determine service order, not an override rate.")
    A("% ---------------------------------------------------------------------------")

    def num(name, val, comment):
        A("\\newcommand{\\%s}{%s}%s%% %s" % (name, val, " " * max(1, 34 - len(name) - len(str(val))), comment))

    def f(v, dp=2):
        """Format a float for LaTeX text mode: a real minus sign, not a hyphen."""
        v = float(v)
        s = "%.*f" % (dp, abs(v))
        return ("$-$" + s) if (v < 0 and float(s) != 0.0) else s

    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    num("calibNcampAll", words.get(len(disp), len(disp)),
        "campuses in the inherited campus set; campus_disposition.csv rows")
    num("calibNcampHead", words.get(len(head), len(head)),
        "campuses with usable completion timestamps and a non-constant class; "
        "campus_disposition.csv status=headline (campuses %s)"
        % ", ".join(str(c) for c in head))
    num("calibNcampFlag", words.get(len(flag), len(flag)),
        "campus analysed but flagged; campus_disposition.csv status=flagged "
        "(campus %s)" % ", ".join(str(c) for c in flag))

    for c in head + flag:
        d = disp[disp["campus"] == c].iloc[0]
        num("calibEndCov%s" % c, "%.0f\\%%" % (100 * d["end_timestamp_coverage"]),
            "campus %d completion-timestamp coverage; campus_disposition.csv "
            "end_timestamp_coverage" % c)

    def cell(pop, var, c):
        m = dev[(dev["population"] == pop) & (dev["variant"] == var)
                & (dev["campus"] == c)]
        return m.iloc[0] if len(m) else None

    for var, tag in (("all", "Unres"), ("trade", "Trade"), ("trade_w30", "TradeW"),
                     ("trade_size_w30", "TradeS"), ("trade_bldg_w30", "Tight")):
        vals = [cell("cm", var, c) for c in head]
        vals = [v for v in vals if v is not None and np.isfinite(v["dev_rate"])]
        if not vals:
            continue
        if var == "trade_bldg_w30":
            num("calibNcampTight", words.get(len(vals), len(vals)),
                "campuses that record a BuildingID and so admit the tightest "
                "candidate set (same trade, same building, 30-day backlog); "
                "deviation_rates.csv variant=trade_bldg_w30 with n_events>0")
        rates = [float(v["dev_rate"]) for v in vals]
        num("calibDev%sLo" % tag, f(min(rates)),
            "lowest headline-campus deviation rate, corrective work, %s "
            "candidate set; deviation_rates.csv dev_rate" % var)
        num("calibDev%sHi" % tag, f(max(rates)),
            "highest headline-campus deviation rate, corrective work, %s "
            "candidate set; deviation_rates.csv dev_rate" % var)
        num("calibDev%sMed" % tag, f(np.median(rates)),
            "median headline-campus deviation rate, corrective work, %s "
            "candidate set; deviation_rates.csv dev_rate" % var)
        num("calibDev%sEvents" % tag, "{:,}".format(
            int(sum(int(v["n_events"]) for v in vals))).replace(",", "{,}"),
            "start events behind the %s corrective-work rates; "
            "deviation_rates.csv n_events summed over headline campuses" % var)

    for c in head:
        for var, tag in (("trade", "Trade"), ("trade_w30", "TradeW")):
            v = cell("cm", var, c)
            if v is None or not np.isfinite(v["dev_rate"]):
                continue
            num("calibDev%sC%s" % (tag, c), f(v["dev_rate"]),
                "campus %d corrective deviation rate, %s candidate set, "
                "n_events %d; deviation_rates.csv" % (c, var, int(v["n_events"])))

    # How far down the recorded-priority ranking the served order actually sat.
    depth = [cell("cm", "trade_w30", c) for c in head]
    depth = [v for v in depth if v is not None and np.isfinite(v["dev_rate"])]
    if depth:
        num("calibFracUrgentMed", f(np.median([float(v["frac_more_urgent_mean"]) for v in depth])),
            "median over headline campuses of the mean share of the visible "
            "same-trade backlog that outranked the served order; "
            "deviation_rates.csv frac_more_urgent_mean (cm, trade_w30)")
        num("calibFracUrgentMedianMed", f(np.median([float(v["frac_more_urgent_median"]) for v in depth])),
            "median over headline campuses of the MEDIAN such share, i.e. the "
            "typical event; deviation_rates.csv frac_more_urgent_median")
        num("calibClassGapMed", f(np.median([float(v["class_gap_mean_on_dev"]) for v in depth]), 1),
            "median over headline campuses of the mean class gap between the "
            "served order and the most urgent order left open, counted only on "
            "deviating events; deviation_rates.csv class_gap_mean_on_dev")

    pp = pred[(pred["population"] == PRED_PRIMARY[0])
              & (pred["variant"] == PRED_PRIMARY[1])
              & pred["campus"].isin(head)]
    if len(pp) and "auc_class_only" in pp and pp["auc_class_only"].notna().any():
        pp = pp[pp["auc_class_only"].notna()]
        num("calibAucClassLo", f(pp["auc_class_only"].min()),
            "lowest headline-campus class-only null AUC; predictability.csv "
            "auc_class_only (population %s, variant %s)" % PRED_PRIMARY)
        num("calibAucClassHi", f(pp["auc_class_only"].max()),
            "highest headline-campus class-only null AUC; predictability.csv "
            "auc_class_only")
        num("calibAucFeatLo", f(pp["auc_class_coded"].min()),
            "lowest headline-campus class+features AUC; predictability.csv "
            "auc_class_coded")
        num("calibAucFeatHi", f(pp["auc_class_coded"].max()),
            "highest headline-campus class+features AUC; predictability.csv "
            "auc_class_coded")
        num("calibAucFeatMed", f(pp["auc_class_coded"].median()),
            "median headline-campus class+features AUC; predictability.csv "
            "auc_class_coded")
        num("calibAucIncLo", f(pp["auc_increment_over_class"].min(), 3),
            "smallest headline-campus AUC increment over the class-only null; "
            "predictability.csv auc_increment_over_class")
        num("calibAucIncHi", f(pp["auc_increment_over_class"].max(), 3),
            "largest headline-campus AUC increment over the class-only null; "
            "predictability.csv auc_increment_over_class")
        num("calibBetaLo", f(pp["beta_hat"].min()),
            "lowest headline-campus recoverable share of the class-residual "
            "deviation variation; predictability.csv beta_hat")
        num("calibBetaHi", f(pp["beta_hat"].max()),
            "highest headline-campus recoverable share; predictability.csv beta_hat")
        num("calibBetaMed", f(pp["beta_hat"].median()),
            "median headline-campus recoverable share; predictability.csv beta_hat")
        num("calibBetaNosizeMed", f(pp["beta_hat_nosize"].median()),
            "median recoverable share with log labour content ablated out; "
            "predictability.csv beta_hat_nosize")
        num("calibEpsUpperMed", f(pp["unexplained_share_upper"].median()),
            "median unexplained share of the class-residual deviation variation, "
            "an UPPER bound on the noise share, NOT epsilon; predictability.csv "
            "unexplained_share_upper")
        num("calibPredEvents", "{:,}".format(int(pp["n_events"].sum())).replace(",", "{,}"),
            "events behind the predictability numbers; predictability.csv "
            "n_events summed over headline campuses")
        if "auc_class_trade" in pp:
            num("calibAucTradeMed", f(pp["auc_class_trade"].median()),
                "median headline-campus AUC from the recorded class plus the "
                "system code alone; predictability.csv auc_class_trade")
            num("calibAucNontradeMed", f(pp["auc_class_nontrade"].median()),
                "median headline-campus AUC from the recorded class plus every "
                "observable EXCEPT the system code; predictability.csv "
                "auc_class_nontrade")
        if "auc_class_coded_permuted" in pp:
            num("calibAucPermMax", f(pp["auc_class_coded_permuted"].max()),
                "worst-case AUC of the identical pipeline on a permuted label "
                "(chance is 0.50); predictability.csv auc_class_coded_permuted")
        tt = trd[(trd["population"] == PRED_PRIMARY[0])
                 & (trd["variant"] == PRED_PRIMARY[1])
                 & trd["campus"].isin(head) & (trd["n_events"] >= 200)]
        if len(tt):
            num("calibTradeRateLo", f(tt["dev_rate"].min()),
                "lowest per-system-code deviation rate over headline campuses "
                "(system codes with at least 200 events); deviation_by_trade.csv "
                "dev_rate")
            num("calibTradeRateHi", f(tt["dev_rate"].max()),
                "highest per-system-code deviation rate over headline campuses; "
                "deviation_by_trade.csv dev_rate")
            num("calibNtradeCells", str(int(len(tt))),
                "system-code cells behind that spread; deviation_by_trade.csv "
                "rows with n_events>=200 on headline campuses")
        num("calibAucRandomCVMed", f(pp["auc_class_coded_randomcv"].median()),
            "same model scored under random 5-fold instead of contiguous "
            "temporal blocks, i.e. the optimism a leaky split would have bought; "
            "predictability.csv auc_class_coded_randomcv")
    if txt is not None and len(txt):
        tt = txt[txt["campus"].isin(head)]
        if len(tt):
            num("calibTextIncLo", f(tt["auc_increment_text"].min(), 3),
                "smallest headline-campus AUC increment from free text over "
                "class+coded features; text_features.csv auc_increment_text")
            num("calibTextIncHi", f(tt["auc_increment_text"].max(), 3),
                "largest headline-campus AUC increment from free text; "
                "text_features.csv auc_increment_text")
            num("calibTextIncMed", f(tt["auc_increment_text"].median(), 3),
                "median headline-campus AUC increment from free text; "
                "text_features.csv auc_increment_text")
            num("calibTextAucMed", f(tt["auc_class_coded_text"].median()),
                "median headline-campus AUC with free text added; "
                "text_features.csv auc_class_coded_text")
    return "\n".join(L) + "\n"


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--skip-text", action="store_true")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the brute-force check of the backlog sweep")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("[1/6] loading and cleaning the raw corpus ...", flush=True)
    clean, meta = load_clean(args.raw)
    mapping = calib.build_priority_mapping(clean)
    mapping = mapping[mapping["campus"].isin(calib.CAMPUSES)].reset_index(drop=True)
    clean["priority"] = calib.priority_class_series(clean, mapping).to_numpy()
    print("      %d work orders, %.0fs" % (len(clean), time.time() - t0), flush=True)

    print("[2/6] data-accuracy checks ...", flush=True)
    checks = data_checks(clean, meta, mapping, args.raw)
    for k, v in checks.items():
        if isinstance(v, dict) and "passed" in v:
            print("      %-34s %s" % (k, "PASS" if v["passed"] else "FAIL"))
            if not v["passed"]:
                raise SystemExit("data-accuracy check failed: %s -> %s" % (k, v))
    with open(os.path.join(OUT_DIR, "data_checks.json"), "w") as fh:
        json.dump(checks, fh, indent=2, default=str)

    print("[3/6] campus disposition ...", flush=True)
    pops, disp_rows = {}, []
    for c in calib.CAMPUSES:
        pop = campus_population(clean, c)
        pops[c] = pop
        d = disposition(pop, c)
        disp_rows.append(d)
        print("      c%-3d %-8s n=%-7d end=%.3f cm_classes=%d %s"
              % (c, d["status"], d["n_work_orders"], d["end_timestamp_coverage"],
                 d["n_cm_classes_present"], d["status_reason"]), flush=True)
    disp = pd.DataFrame(disp_rows)
    disp.to_csv(os.path.join(OUT_DIR, "campus_disposition.csv"), index=False)
    analysed = [d["campus"] for d in disp_rows if d["status"] != "excluded"]

    print("[4/6] A1 observed deviations ...", flush=True)
    dev_rows, cls_rows, trade_rows, labels = [], [], [], {}
    for c in analysed:
        r, b, tr, lab = deviation_table(pops[c], c, rng, verify=not args.no_verify)
        dev_rows += r
        cls_rows += b
        trade_rows += tr
        for k, v in lab.items():
            labels[(c,) + k] = v
        print("      c%-3d done  %.0fs" % (c, time.time() - t0), flush=True)
    dev = pd.DataFrame(dev_rows)
    dev.to_csv(os.path.join(OUT_DIR, "deviation_rates.csv"), index=False)
    pd.DataFrame(cls_rows).to_csv(
        os.path.join(OUT_DIR, "deviation_by_class.csv"), index=False)
    pd.DataFrame(trade_rows).to_csv(
        os.path.join(OUT_DIR, "deviation_by_trade.csv"), index=False)

    print("[5/6] A2 predictability ...", flush=True)
    pred = pd.DataFrame(predictability(labels, np.random.default_rng(SEED)))
    pred.to_csv(os.path.join(OUT_DIR, "predictability.csv"), index=False)
    head = list(disp.loc[disp["status"] == "headline", "campus"])
    pp = pred[(pred["population"] == PRED_PRIMARY[0])
              & (pred["variant"] == PRED_PRIMARY[1]) & pred["campus"].isin(head)]
    summ = []
    for col in ("prevalence", "auc_class_only", "auc_class_coded",
                "auc_class_coded_nosize", "auc_coded_only",
                "auc_class_trade", "auc_class_dow", "auc_class_size",
                "auc_class_nontrade", "auc_class_coded_permuted",
                "auc_class_coded_randomcv",
                "auc_increment_over_class", "auc_increment_nosize",
                "beta_class_trade", "beta_class_dow", "beta_class_size",
                "beta_class_nontrade",
                "beta_hat", "beta_hat_nosize", "unexplained_share_upper"):
        if col in pp:
            summ.append(dict(metric=col, n_campuses=len(pp),
                             min=round(float(pp[col].min()), 4),
                             median=round(float(pp[col].median()), 4),
                             max=round(float(pp[col].max()), 4)))
    pd.DataFrame(summ).to_csv(
        os.path.join(OUT_DIR, "predictability_summary.csv"), index=False)
    show = [c for c in ("campus", "n_events", "prevalence", "auc_class_only",
                        "auc_class_coded", "auc_increment_over_class",
                        "beta_hat", "beta_hat_nosize") if c in pp]
    if len(pp):
        print(pp[show].to_string(index=False), flush=True)

    txt = None
    if not args.skip_text:
        print("[6/6] A3 free text ...", flush=True)
        trows, tterms = text_analysis(labels, np.random.default_rng(SEED))
        txt = pd.DataFrame(trows)
        txt.to_csv(os.path.join(OUT_DIR, "text_features.csv"), index=False)
        pd.DataFrame(tterms).to_csv(
            os.path.join(OUT_DIR, "text_top_terms.csv"), index=False)
        if len(txt):
            print(txt.to_string(index=False), flush=True)

    with open(os.path.join(OUT_DIR, "macros_snippet.tex"), "w") as fh:
        fh.write(macro_snippet(dev, pred, txt, disp, pd.DataFrame(trade_rows)))

    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        script="scripts/y3_calib_overrides.py",
        raw_path=os.path.abspath(args.raw), raw_sha256=io.RAW_SHA256,
        seed=SEED, window_days=WINDOW_DAYS, n_folds=N_FOLDS,
        cleaning=meta, data_checks=checks,
        inherited_filters=(
            "fmwos.io.clean R1/R2/R3/R4/R6/R7 and fmwos.calib priority mapping "
            "v2 R5a-R5d, campus set {1,2,5,9,10,12}; reconstruction asserted "
            "against results/p1_calib/priority_mapping.csv and "
            "results/y3_p6/priority_reliability.csv"),
        candidate_set_variants={n: dict(match_on=list(k),
                                        backlog_window_days=w) for n, k, w in VARIANTS},
        predictability_labels=[list(x) for x in PRED_LABELS],
        primary_label=list(PRED_PRIMARY),
        cv="5 contiguous blocks of the campus closure-event timeline; random "
           "5-fold reported as a robustness column only",
        beta_hat_definition=(
            "1 - SS_res(class + observable features) / SS_res(class only), both "
            "out of sample on pooled held-out probabilities: the share of the "
            "class-residual variation of the deviation indicator that observable "
            "features recover. It is the empirical analogue of the paper's beta, "
            "not a measurement of it. It is a LOWER bound on the recoverable "
            "share of the underlying propensity for two reasons: the outcome is "
            "binary, so its residual sum of squares retains irreducible "
            "Bernoulli noise that no model can remove, and f is restricted to a "
            "linear logistic model over about thirty coded features."),
        epsilon_note=(
            "FMUCD records no review, reassignment, reopen or re-prioritisation "
            "event, so the supervisor's override noise epsilon is not "
            "identifiable. unexplained_share_upper = 1 - beta_hat is an upper "
            "bound on the noise share of the departure process."),
        honest_framing=(
            "A departure from the recorded priority order is not evidence of "
            "hidden urgency. It is consistent with hidden urgency and equally "
            "consistent with trade eligibility, crew availability, travel, "
            "batching, parts lead time and access windows, none of which this "
            "corpus records. Every rate here is an upper bound on how often the "
            "recorded priority failed to determine the order of service."),
        runtime_seconds=round(time.time() - t0, 1),
    )
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print("\nwrote %s  (%.0fs)" % (OUT_DIR, time.time() - t0))


if __name__ == "__main__":
    main()
