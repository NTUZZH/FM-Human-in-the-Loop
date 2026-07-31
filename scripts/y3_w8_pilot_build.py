"""Paper Y3 -- W8: build the practitioner urgency-pairs pilot instrument.

WHAT THIS SCRIPT PRODUCES (and nothing else)
--------------------------------------------
An *instrument*. It draws ~50 pairs of real work orders from FMUCD, renders them
as a self-contained offline HTML questionnaire plus a CSV response template, and
writes the manifest that ``scripts/y3_w8_pilot_analyse.py`` needs to score the
returned responses. It creates no responses, no participants, and no results.

WHAT THE PILOT IS FOR
---------------------
Two premises of the supervisor model in Section "The private-information
supervisor" are grounded here with real people rather than assumed:

  P1  a hidden urgency that practitioners *share*: they agree with each other,
      above chance, on which of two queued orders should be served first;
  P2  that shared urgency is partly a function of the observable order
      attributes the estimator reads (trade/system, job size, waiting time),
      which is the real-data analogue of beta > 0.

The pilot does NOT validate the correction loop, the dispatching results, or any
reported reduction in true weighted tardiness. See PREREGISTRATION.md.

INHERITED CORPUS TREATMENT (same population the paper models)
-------------------------------------------------------------
Read from ``scripts/y3_fmucd_residual.py`` and the pipeline it sits on:

  * campus selection  ``fmwos.calib.CAMPUSES == fmwos.instances.CAMPUS_SET``
                      = {1, 2, 5, 9, 10, 12}  (the six campuses the paper uses)
  * cleaning          ``fmwos.io.load_raw`` (R1 typed parse) then
                      ``fmwos.io.clean``:
                        R2 drop rows with no WOID / UniversityID / WOStartDate
                        R3 drop LaborHours <= 0
                        R7 aggregate duplicate (campus, WOID) labour lines into
                           one order (hours summed, start = min, end = max, all
                           other fields from the max-LaborHours row)
                        R4 cap LaborHours at the global p99.5 after R7
                        R6 trade = UNIFORMAT top-level SystemCode, blank -> UNK
  * priority class    ``fmwos.calib.build_priority_mapping`` (v2: R5a PM -> 4,
                      R5b keyword anchors, R5c campus numeric scale flipped by
                      the value-rank / median-completion-duration Spearman,
                      R5d rare or missing -> 3), the mapping table restricted to
                      the six campuses exactly as ``calib.write_calibration``
                      does, then ``calib.priority_class_series``
  * merged trade      ``calib.trade_merge_map`` / ``apply_trade_merge``: a trade
                      with < 1000 rows in a campus becomes MISC for that campus
  * time axis         ``fmwos.timeaxis.abs_bh_series``: 8 business hours per
                      weekday, off-shift stamps roll forward to the next shift
  * class semantics   ``fmwos.timeaxis.SLA_BH`` = {1: 8, 2: 24, 3: 80, 4: 171.4}
                      business hours, shown to the rater as the target response
                      time so the recorded class means what it means in the paper

PILOT-SPECIFIC FILTERS (stated here, not inherited)
---------------------------------------------------
An order is eligible for an item only if it additionally has a readable free-text
description (>= 20 characters and >= 4 words after whitespace normalisation), a
recorded component, and a recorded system; and no two selected orders share a
normalised description, so no rater sees the same text twice. A building
descriptor is NOT required, for the reason given under WHAT AN ITEM SHOWS.

WHAT AN ITEM SHOWS
------------------
Only fields a supervisor holds at the moment of dispatch: the free-text
description, the system and component, a building descriptor, the recorded
priority class with its target response time, and how long the order has been
waiting. Realised duration, cost, and close-out date are never shown; they are
outcomes. The estimated labour content IS shown, because job size is one of the
three observable attributes the estimator reads and the simulation treats the
processing time as known at dispatch; FMUCD records it at close-out, so the item
labels it an estimate and PREREGISTRATION.md records the substitution.

The building descriptor is the one field the corpus does not carry everywhere.
FMUCD records a building Type for campus 1 only among the paper's six, and a
BuildingName for campuses 1 and 12; campuses 2, 5, 9 and 10 carry neither.
Requiring it would shrink the instrument to one campus and lose exactly the
cross-campus spread the paper's own boundary results turn on. So the descriptor
is shown when recorded (as a building type where the corpus gives one, otherwise
as the building's name) and the line is omitted when not, under a hard constraint
that BOTH orders of a pair carry the same fields: an item never contrasts a
well-documented order with a thinly documented one, which would be a presentation
artefact rather than an urgency signal.

SAMPLING RULE (reproducible, seeded)
------------------------------------
1. Per campus, every weekday-08:00 anchor in the campus's WOStartDate span is a
   candidate dispatch moment (the anchor scheme of ``scripts/p1_instances.py``),
   shuffled once by the master RNG.
2. At an anchor t0 the *backlog* is every eligible order released in
   [t0 - 120 bh, t0] (15 business days). Anchors with fewer than 10 backlog
   orders are skipped. Waiting time of an order = t0 - release, in business
   hours, so both members of a pair are queued at the same site at the same
   moment and their waits are real.
3. Up to 25 unordered pairs are drawn per accepted anchor, giving the candidate
   pool.
4. Each candidate pair is assigned to exactly one stratum (see below) and the
   per-stratum target is filled greedily from the shuffled pool under three
   constraints: an order appears in at most one pair, a normalised description
   appears in at most one pair, and the campus with the fewest pairs so far is
   preferred at every draw.

STRATA (assignment priority S3 > S1 > S2 > S4)
-----------------------------------------------
  S3  class-vs-attributes disagreement: the recorded classes differ AND the
      attribute-implied class ordering runs the other way by at least 0.25 class
      units. The attribute-implied class chat_j is an ordinary least-squares fit
      of the recorded class on the paper's own campus-agnostic features (merged
      trade one-hot, log1p(labour hours), release day-of-week one-hot) with a
      per-campus intercept, fitted on the whole cleaned six-campus pool. It uses
      no pilot response and no latent quantity.
  S1  equal recorded classes: the record gives no guidance, so any agreement
      between raters must come from somewhere else.
  S2  one class apart, attribute ordering agreeing with the record.
  S4  two or more classes apart, agreeing: the easy control where the record
      should settle it.

REPEATS AND COUNTERBALANCING
----------------------------
Six pairs, chosen deterministically and spread across the strata, are shown twice,
the second time with the two orders swapped left for right and at least 15 items
later, so within-rater consistency is not inflated by position memory. Six rather
than four because the simulation in ``results/y3_w8/selftest/`` showed that four
repeats give a consistency estimate whose interval spans almost the whole unit
range at three to five raters; six raises the pooled trial count by half for the
cost of two more items.

Two side balances have to hold at once and they are not independent, because
"canonical order A on the left" coincides with "the more urgent recorded class on
the left" when c_A < c_B and is its opposite when c_A > c_B. The class-differing
pairs are therefore split into those two groups and each is balanced separately,
with the odd remainders sent to opposite groups so they cancel. Both counts then
land on half by construction, with no repair pass: across the 50 unique items the
canonical first order sits on the left in exactly 25, and among the class-differing
items the more urgent recorded class sits on the left in exactly half. Neither the
pair's internal labelling nor the recorded class can be read off the side.

OUTPUTS
-------
  pilot/y3_w8_pilot.html            self-contained offline questionnaire
  pilot/y3_w8_manifest.csv          one row per presented item (the answer key)
  pilot/y3_w8_items.json            machine-readable item content + build meta
  pilot/y3_w8_response_template.csv blank response sheet in presentation order
  pilot/responses/                  where completed sheets are dropped
  results/y3_w8/build_report.json   filter counts, realised strata, balance checks

Run (CPU-light; cores 20-23 per the run discipline):
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c 20-23 \
      python scripts/y3_w8_pilot_build.py
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmwos import calib, instances, io  # noqa: E402
from fmwos import timeaxis as ta  # noqa: E402

DEFAULT_RAW = Path("/home/ziheng/PaperY-FMScheduling/data/raw/FMUCD.csv")
OUT_DIR = ROOT / "pilot"
RES_DIR = ROOT / "results" / "y3_w8"
CACHE_DIR = RES_DIR / "cache"

# ---- pilot design constants (every one of these is reported) --------------- #
SEED = 908
BACKLOG_BH = 120.0          # 15 business days of visible backlog at an anchor
MIN_BACKLOG = 10            # anchors with a thinner queue are skipped
ANCHORS_PER_CAMPUS = 600    # candidate dispatch moments tried per campus
PAIRS_PER_ANCHOR = 25       # unordered pairs drawn per accepted anchor
DISAGREE_MARGIN = 0.25      # class units, for the S3 disagreement test
MIN_DESC_CHARS = 20
MIN_DESC_WORDS = 4
DESC_MAX_CHARS = 240        # display truncation, at a word boundary
N_REPEATS = 6               # see the note under REPEATS AND COUNTERBALANCING
REPEAT_MIN_GAP = 15         # items between a pair's two presentations

STRATA = ("S1_equal_class", "S2_one_apart", "S3_class_vs_attributes", "S4_far_apart")
STRATUM_TARGET = {
    "S1_equal_class": 18,
    "S2_one_apart": 14,
    "S3_class_vs_attributes": 12,
    "S4_far_apart": 6,
}
STRATUM_FILL_ORDER = ("S3_class_vs_attributes", "S4_far_apart",
                      "S1_equal_class", "S2_one_apart")
STRATUM_LABEL = {
    "S1_equal_class": "recorded classes equal",
    "S2_one_apart": "one class apart, record and attributes agree",
    "S3_class_vs_attributes": "record and attributes disagree",
    "S4_far_apart": "two or more classes apart, record and attributes agree",
}

CLASS_NAME = {1: "P1 Emergency", 2: "P2 Urgent", 3: "P3 Routine", 4: "P4 Planned"}
# Target response time implied by the paper's SLA table, in business days (8 bh).
CLASS_SLA_DAYS = {c: ta.SLA_BH[c] / 8.0 for c in (1, 2, 3, 4)}

# Campus id -> anonymous site letter shown to the rater.
SITE_LETTER = {1: "A", 2: "B", 5: "C", 9: "D", 10: "E", 12: "F"}

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-().\s]{7,}\d)(?!\d)")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Text normalisation                                                          #
# --------------------------------------------------------------------------- #
def norm_text(v) -> str:
    """Whitespace-normalise a corpus text field and strip spreadsheet artefacts.

    FMUCD carries Excel's ``_x000D_`` carriage-return escape inside descriptions.
    Email addresses and long digit runs that look like phone numbers are replaced
    by a placeholder: the corpus is already public and de-identified, and this
    only avoids re-publishing a contact detail on a questionnaire.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = str(v)
    if s.strip().lower() in ("nan", "none", "<na>"):
        return ""
    s = s.replace("_x000D_", " ").replace("\r", " ").replace("\n", " ")
    s = _EMAIL_RE.sub("[contact removed]", s)
    s = _PHONE_RE.sub("[number removed]", s)
    return _WS_RE.sub(" ", s).strip()


def truncate_words(s: str, limit: int = DESC_MAX_CHARS) -> str:
    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit(" ", 1)[0]
    return cut + " ..."


# --------------------------------------------------------------------------- #
# Corpus load -> cleaned, class-mapped, six-campus eligible pool               #
# --------------------------------------------------------------------------- #
TEXT_COLS = ["WODescription", "SubsystemDescription", "ComponentDescription",
             "BuildingName"]


def load_pool(raw_path: Path, nrows: int | None, cache: Path,
              rebuild: bool) -> tuple[pd.DataFrame, dict]:
    """Return (eligible order pool, audit dict), cached as parquet."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache.with_suffix(".meta.json")
    if cache.exists() and meta_path.exists() and not rebuild:
        pool = pd.read_parquet(cache)
        with open(meta_path) as fh:
            audit = json.load(fh)
        print(f"[cache] loaded {len(pool):,} eligible orders from {cache}")
        return pool, audit

    t0 = time.time()
    usecols = sorted(set(io.USECOLS) | set(TEXT_COLS))
    dtypes = dict(io.DTYPES)
    for c in TEXT_COLS:
        dtypes.setdefault(c, "string")
    print(f"loading {raw_path} ({len(usecols)} columns) ...", flush=True)
    raw = pd.read_csv(raw_path, usecols=usecols, dtype=dtypes, nrows=nrows)
    for c in io.DATE_COLS:
        raw[c] = pd.to_datetime(raw[c], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    print(f"  raw rows: {len(raw):,} ({time.time() - t0:.0f}s)", flush=True)

    clean, audit = io.clean(raw)
    del raw
    audit = {k: (float(v) if isinstance(v, (int, float)) else v)
             for k, v in audit.items()}
    print(f"  cleaned work orders: {len(clean):,} ({time.time() - t0:.0f}s)",
          flush=True)

    # Inherited priority mapping v2, restricted to the six campuses exactly as
    # calib.write_calibration does before the class series is merged on.
    mapping = calib.build_priority_mapping(clean)
    mapping = mapping[mapping["campus"].isin(calib.CAMPUSES)].reset_index(drop=True)
    priority = calib.priority_class_series(clean, mapping)
    tmap = calib.trade_merge_map(clean)
    trade_m = calib.apply_trade_merge(clean, tmap)

    keep = clean["UniversityID"].astype("int64").isin(instances.CAMPUS_SET).to_numpy()
    audit["campus_filter_kept"] = int(keep.sum())
    sub = clean[keep]

    pool = pd.DataFrame({
        "campus": sub["UniversityID"].astype("int64").to_numpy(),
        "wo_id": sub["WOID"].astype("object").to_numpy(),
        "trade": trade_m[keep].to_numpy(),
        "trade_raw": sub["trade"].astype("object").to_numpy(),
        "labor_h": sub["LaborHours"].to_numpy(dtype="float64"),
        "cls": priority[keep].to_numpy().astype("int64"),
        "is_pm": sub["is_pm"].fillna(False).astype(bool).to_numpy(),
        "abs_bh": ta.abs_bh_series(sub["WOStartDate"]),
        "start": sub["WOStartDate"].to_numpy(),
        "building": sub["BuildingID"].astype("object").to_numpy(),
        "btype": [norm_text(v) for v in sub["Type"].astype("object").to_numpy()],
        "bname": [norm_text(v) for v in sub["BuildingName"].astype("object").to_numpy()],
        "system": [norm_text(v) for v in sub["SystemDescription"].astype("object").to_numpy()],
        "subsystem": [norm_text(v) for v in sub["SubsystemDescription"].astype("object").to_numpy()],
        "component": [norm_text(v) for v in sub["ComponentDescription"].astype("object").to_numpy()],
        "desc": [norm_text(v) for v in sub["WODescription"].astype("object").to_numpy()],
    })
    del clean, sub

    # Building descriptor: the recorded type where the corpus gives one, else the
    # recorded building name, else nothing. Coverage is campus-specific, so this
    # is a display field, never an eligibility requirement (see the module note).
    bctx, bkind = [], []
    for t, n in zip(pool["btype"].to_numpy(), pool["bname"].to_numpy()):
        if t:
            bctx.append(t)
            bkind.append("type")
        elif n:
            bctx.append(n)
            bkind.append("name")
        else:
            bctx.append("")
            bkind.append("")
    pool["bctx"] = bctx
    pool["bctx_kind"] = bkind

    # ---- pilot-specific eligibility (stated, not inherited) --------------- #
    n0 = len(pool)
    words = pool["desc"].str.split().str.len().fillna(0)
    m_desc = (pool["desc"].str.len() >= MIN_DESC_CHARS) & (words >= MIN_DESC_WORDS)
    m_comp = pool["component"].str.len() > 0
    m_sys = pool["system"].str.len() > 0
    audit["pilot_pool_before_eligibility"] = int(n0)
    audit["pilot_drop_short_or_missing_description"] = int((~m_desc).sum())
    audit["pilot_drop_missing_component"] = int((~m_comp).sum())
    audit["pilot_drop_missing_system"] = int((~m_sys).sum())
    pool = pool[m_desc & m_comp & m_sys].reset_index(drop=True)
    audit["pilot_pool_eligible"] = int(len(pool))
    audit["pilot_building_descriptor_coverage"] = {
        str(c): round(float((pool.loc[pool["campus"] == c, "bctx"].str.len() > 0).mean()), 4)
        for c in sorted(pool["campus"].unique().tolist())}
    print(f"  eligible for pilot items: {len(pool):,}", flush=True)
    print(f"  building-descriptor coverage per campus: "
          f"{audit['pilot_building_descriptor_coverage']}", flush=True)

    pool.to_parquet(cache, index=False)
    with open(meta_path, "w") as fh:
        json.dump(audit, fh, indent=2)
    return pool, audit


# --------------------------------------------------------------------------- #
# Attribute-implied class (S3 definition) and the trade urgency prior          #
# --------------------------------------------------------------------------- #
def day_index(abs_bh: np.ndarray) -> np.ndarray:
    """Weekday slot on the business-hour axis, matching overlay._day_index."""
    return (np.floor(abs_bh / 8.0).astype("int64")) % 5


def fit_attribute_class(pool: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """OLS of the recorded class on the paper's campus-agnostic features.

    Design: per-campus intercept | merged-trade one-hot (drop first) |
    log1p(labour hours) | release day-of-week one-hot (drop first). Fitted on the
    entire cleaned six-campus pool. Returns (chat per order, coefficient record).

    This is the "what the observable attributes would suggest" reference used to
    define the S3 stratum. It reads only observable fields, no pilot response and
    no latent quantity, so the stratum is reproducible from the corpus alone.
    """
    campuses = sorted(pool["campus"].unique().tolist())
    trades = sorted(pool["trade"].unique().tolist())
    days = list(range(5))
    di = day_index(pool["abs_bh"].to_numpy())

    blocks, names = [], []
    for c in campuses:
        blocks.append((pool["campus"].to_numpy() == c).astype("float64"))
        names.append(f"campus[{c}]")
    for t in trades[1:]:
        blocks.append((pool["trade"].to_numpy() == t).astype("float64"))
        names.append(f"trade[{t}]")
    blocks.append(np.log1p(pool["labor_h"].to_numpy(dtype="float64")))
    names.append("log1p_labor_h")
    for d in days[1:]:
        blocks.append((di == d).astype("float64"))
        names.append(f"dow[{d}]")

    X = np.column_stack(blocks)
    y = pool["cls"].to_numpy(dtype="float64")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    chat = X @ coef
    resid = y - chat
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rec = dict(
        n=int(len(y)), n_params=int(X.shape[1]),
        r2=float(1.0 - float(np.sum(resid ** 2)) / ss_tot) if ss_tot > 0 else float("nan"),
        coefficients={n: round(float(v), 6) for n, v in zip(names, coef)},
    )
    return chat, rec


def trade_urgency_prior(pool: pd.DataFrame) -> dict[str, float]:
    """Mean recorded urgency (5 - class) per merged trade over the whole pool.

    A label-free, corpus-level scalar encoding of the trade, pre-registered as
    the analysis's trade feature so the out-of-sample model stays low-dimensional
    at n ~ 50 pairs. Computed here, stored in the manifest, never refitted.
    """
    g = (5.0 - pool["cls"].astype("float64")).groupby(pool["trade"]).mean()
    return {str(k): round(float(v), 6) for k, v in g.items()}


# --------------------------------------------------------------------------- #
# Candidate pair generation                                                   #
# --------------------------------------------------------------------------- #
def campus_anchors(starts: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Weekday-08:00 anchors across the campus span, on the absolute bh axis."""
    first, last = starts.min(), starts.max()
    days = pd.date_range(first.normalize(), last.normalize(), freq="D")
    days = days[days.weekday < 5]
    if len(days) == 0:
        return np.empty(0, dtype="float64")
    anchors = pd.Series(days + pd.Timedelta(hours=8))
    a = ta.abs_bh_series(anchors)
    idx = np.arange(len(a))
    rng.shuffle(idx)
    return a[idx]


def build_candidate_pairs(pool: pd.DataFrame, chat: np.ndarray,
                          rng: np.random.Generator) -> pd.DataFrame:
    """Sample (anchor, backlog) snapshots and draw unordered pairs from each."""
    rows = []
    for campus in instances.CAMPUS_SET:
        m = (pool["campus"] == campus).to_numpy()
        sub = pool[m]
        if len(sub) < MIN_BACKLOG:
            continue
        order = np.argsort(sub["abs_bh"].to_numpy(), kind="stable")
        rel = sub["abs_bh"].to_numpy()[order]
        gidx = np.flatnonzero(m)[order]              # row ids back into `pool`
        anchors = campus_anchors(pd.Series(sub["start"].to_numpy()), rng)
        tried = 0
        for t0 in anchors:
            if tried >= ANCHORS_PER_CAMPUS:
                break
            tried += 1
            lo = np.searchsorted(rel, t0 - BACKLOG_BH, side="left")
            hi = np.searchsorted(rel, t0, side="right")
            n = hi - lo
            if n < MIN_BACKLOG:
                continue
            for _ in range(PAIRS_PER_ANCHOR):
                i, j = rng.choice(n, size=2, replace=False)
                gi, gj = int(gidx[lo + i]), int(gidx[lo + j])
                # canonical order within the pair: smaller work-order id is "A"
                a, b = (gi, gj) if str(pool.at[gi, "wo_id"]) < str(pool.at[gj, "wo_id"]) \
                    else (gj, gi)
                rows.append((campus, float(t0), a, b))
    df = pd.DataFrame(rows, columns=["campus", "anchor_bh", "row_a", "row_b"])
    df = df.drop_duplicates(subset=["row_a", "row_b"]).reset_index(drop=True)

    a, b = df["row_a"].to_numpy(), df["row_b"].to_numpy()
    df["cls_a"] = pool["cls"].to_numpy()[a]
    df["cls_b"] = pool["cls"].to_numpy()[b]
    df["chat_a"] = chat[a]
    df["chat_b"] = chat[b]
    df["wait_a"] = df["anchor_bh"] - pool["abs_bh"].to_numpy()[a]
    df["wait_b"] = df["anchor_bh"] - pool["abs_bh"].to_numpy()[b]
    df["d_cls"] = df["cls_a"] - df["cls_b"]
    df["d_chat"] = df["chat_a"] - df["chat_b"]
    df["stratum"] = assign_strata(df)
    # a pair whose two descriptions are identical text carries no contrast
    da = pool["desc"].to_numpy()[a]
    db = pool["desc"].to_numpy()[b]
    # both orders of a pair must carry the same fields, so a better-documented
    # order never looks more urgent for a presentational reason
    has = (pool["bctx"].str.len() > 0).to_numpy()
    df = df[(da != db) & (has[a] == has[b])].reset_index(drop=True)
    return df


def assign_strata(df: pd.DataFrame) -> np.ndarray:
    """Assign one stratum per pair, priority S3 > S1 > S2 > S4."""
    d_cls = df["d_cls"].to_numpy()
    d_chat = df["d_chat"].to_numpy()
    out = np.array([""] * len(df), dtype=object)
    disagree = (d_cls != 0) & (np.sign(d_chat) == -np.sign(d_cls)) \
        & (np.abs(d_chat) >= DISAGREE_MARGIN)
    out[disagree] = "S3_class_vs_attributes"
    rest = out == ""
    out[rest & (d_cls == 0)] = "S1_equal_class"
    out[rest & (np.abs(d_cls) == 1) & (out == "")] = "S2_one_apart"
    out[out == ""] = "S4_far_apart"
    # np.abs(d_cls) == 0 already claimed by S1; guard the remaining cells
    out[(out == "S2_one_apart") & (np.abs(d_cls) != 1)] = "S4_far_apart"
    out[(out == "S4_far_apart") & (d_cls == 0)] = "S1_equal_class"
    return out


def select_pairs(cand: pd.DataFrame, pool: pd.DataFrame,
                 rng: np.random.Generator) -> tuple[list[dict], dict]:
    """Greedy per-stratum fill under the uniqueness and campus-balance rules.

    An order may appear in at most one selected pair, a normalised description in
    at most one selected pair, and at every draw the campus with the fewest pairs
    so far is preferred, which spreads the instrument over the six campuses
    without a hard quota that a thin stratum could not meet.
    """
    row_a = cand["row_a"].to_numpy()
    row_b = cand["row_b"].to_numpy()
    camp = cand["campus"].to_numpy()
    strat = cand["stratum"].to_numpy()
    desc = pool["desc"].to_numpy()

    idx = np.arange(len(cand))
    rng.shuffle(idx)
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for i in idx:
        by_stratum[str(strat[i])].append(int(i))

    used_orders: set[int] = set()
    used_desc: set[str] = set()
    per_campus: Counter = Counter()
    chosen: list[dict] = []
    realised: dict[str, int] = {}

    for st in STRATUM_FILL_ORDER:
        target = STRATUM_TARGET[st]
        avail = by_stratum.get(st, [])
        taken = 0
        while taken < target and avail:
            best, best_key, keep = None, None, []
            for i in avail:
                a, b = int(row_a[i]), int(row_b[i])
                if a in used_orders or b in used_orders:
                    continue                      # order already spent
                if desc[a] in used_desc or desc[b] in used_desc:
                    continue                      # text already shown
                keep.append(i)
                key = (per_campus[int(camp[i])], i)
                if best_key is None or key < best_key:
                    best_key, best = key, i
            avail = keep
            if best is None:
                break
            a, b = int(row_a[best]), int(row_b[best])
            used_orders.update((a, b))
            used_desc.update((desc[a], desc[b]))
            per_campus[int(camp[best])] += 1
            chosen.append(dict(cand_row=int(best), stratum=st))
            avail = [x for x in avail if x != best]
            taken += 1
        realised[st] = taken
    return chosen, dict(realised=realised, per_campus=dict(per_campus))


# --------------------------------------------------------------------------- #
# Presentation: sides, repeats, item order                                    #
# --------------------------------------------------------------------------- #
def assign_sides(chosen: list[dict], cand: pd.DataFrame,
                 rng: np.random.Generator) -> dict:
    """Decide, per pair, whether the canonical order A is shown on the left.

    Two balances have to hold at once, and they are not independent. Among the
    items whose recorded classes differ, "A on the left" coincides with "the more
    urgent recorded class on the left" when c_A < c_B and is its opposite when
    c_A > c_B. So the class-differing items are split into those two groups and
    each is balanced separately: half of the c_A < c_B group gets A on the left,
    and half of the c_A > c_B group does. That makes both counts land on half by
    construction, with the odd remainders sent to opposite groups so they cancel.
    Items with equal recorded classes carry no class-side question and only need
    the A/B balance.
    """
    cls_a = cand["cls_a"].to_numpy()
    cls_b = cand["cls_b"].to_numpy()
    g_lt, g_gt, g_eq = [], [], []
    for k, ch in enumerate(chosen):
        i = ch["cand_row"]
        if int(cls_a[i]) < int(cls_b[i]):
            g_lt.append(k)
        elif int(cls_a[i]) > int(cls_b[i]):
            g_gt.append(k)
        else:
            g_eq.append(k)

    side_a_left: dict[int, bool] = {}
    for group, n_left in ((g_lt, len(g_lt) // 2),                 # floor
                          (g_gt, len(g_gt) - len(g_gt) // 2),     # ceil: remainders cancel
                          (g_eq, len(g_eq) // 2)):
        perm = list(rng.permutation(len(group)))
        for rank, p in enumerate(perm):
            side_a_left[group[int(p)]] = rank < n_left
    return side_a_left


def urgent_side_counts(chosen, cand, side_a_left) -> tuple[int, int]:
    """(#items with the lower recorded class on the left, #class-differing items)."""
    cls_a = cand["cls_a"].to_numpy()
    cls_b = cand["cls_b"].to_numpy()
    n_left, n_diff = 0, 0
    for k, ch in enumerate(chosen):
        i = ch["cand_row"]
        ca, cb = int(cls_a[i]), int(cls_b[i])
        if ca == cb:
            continue
        n_diff += 1
        left_cls, right_cls = (ca, cb) if side_a_left[k] else (cb, ca)
        if left_cls < right_cls:
            n_left += 1
    return n_left, n_diff


def choose_repeats(chosen, cand, rng) -> list[int]:
    """Spread the repeats across strata, taking mid-difficulty pairs.

    Within a stratum, pairs are ordered by how close their waiting-time gap is to
    the stratum's median gap, so the repeats are neither the most nor the least
    lopsided pairs: either extreme would make within-rater consistency look
    artificially good or bad. Strata are then visited largest first, round robin,
    until the repeat quota is met, which keeps the check from concentrating on the
    thin strata.
    """
    wa, wb = cand["wait_a"].to_numpy(), cand["wait_b"].to_numpy()
    by_st: dict[str, list[int]] = defaultdict(list)
    for k, ch in enumerate(chosen):
        by_st[ch["stratum"]].append(k)

    queues: dict[str, list[int]] = {}
    for st, ks in by_st.items():
        gaps = {k: abs(float(wa[chosen[k]["cand_row"]]) - float(wb[chosen[k]["cand_row"]]))
                for k in ks}
        med = float(np.median(list(gaps.values())))
        queues[st] = [k for k in sorted(ks, key=lambda k: (abs(gaps[k] - med), k))]

    order = sorted(queues, key=lambda st: (-len(by_st[st]), st))
    picks: list[int] = []
    while len(picks) < N_REPEATS and any(queues[st] for st in order):
        for st in order:
            if len(picks) >= N_REPEATS:
                break
            if queues[st]:
                picks.append(queues[st].pop(0))
    return picks[:N_REPEATS]


def sequence_items(n_unique: int, repeats: list[int], rng: np.random.Generator):
    """Presentation sequence as a list of (pair index, presentation number).

    The unique items are shuffled, then each repeated pair's second presentation
    is inserted at a random position at least ``REPEAT_MIN_GAP`` items after its
    first. Insertions only push later items further back, so a gap once satisfied
    stays satisfied.
    """
    limit = n_unique - REPEAT_MIN_GAP
    if limit <= 0:
        raise RuntimeError("too few pairs for the requested repeat gap")
    for _ in range(5000):
        order = [int(k) for k in rng.permutation(n_unique)]
        pos = {k: p for p, k in enumerate(order)}
        if any(pos[k] > limit for k in repeats):
            continue
        seq = [(k, 1) for k in order]
        for k in sorted(repeats, key=lambda r: pos[r]):
            first = next(p for p, (kk, r) in enumerate(seq) if kk == k and r == 1)
            at = int(rng.integers(first + REPEAT_MIN_GAP, len(seq) + 1))
            seq.insert(at, (k, 2))
        return seq
    raise RuntimeError("could not sequence items under the repeat-gap constraint")


# --------------------------------------------------------------------------- #
# Item rendering                                                              #
# --------------------------------------------------------------------------- #
def order_card(pool: pd.DataFrame, row: int, wait_bh: float) -> dict:
    r = pool.loc[row]
    sysline = r["system"]
    if r["subsystem"] and r["subsystem"].lower() != sysline.lower():
        sysline = f"{sysline} / {r['subsystem']}"
    return dict(
        order_id=str(r["wo_id"]),
        description=truncate_words(str(r["desc"])),
        system=sysline,
        component=str(r["component"]),
        building=str(r["bctx"]),
        building_label=("Building type" if r["bctx_kind"] == "type" else "Building"),
        recorded_class=int(r["cls"]),
        recorded_class_label=CLASS_NAME[int(r["cls"])],
        target_response_days=round(CLASS_SLA_DAYS[int(r["cls"])], 1),
        waiting_days=round(float(wait_bh) / 8.0, 1),
        estimated_labour_hours=round(float(r["labor_h"]), 1),
        trade=str(r["trade"]),
    )


def build_items(chosen, cand, pool, side_a_left, sequence) -> list[dict]:
    items = []
    for n, (k, presentation) in enumerate(sequence, start=1):
        ch = chosen[k]
        i = ch["cand_row"]
        a_row, b_row = int(cand.at[i, "row_a"]), int(cand.at[i, "row_b"])
        a_left = side_a_left[k] if presentation == 1 else (not side_a_left[k])
        wa, wb = float(cand.at[i, "wait_a"]), float(cand.at[i, "wait_b"])
        card_a = order_card(pool, a_row, wa)
        card_b = order_card(pool, b_row, wb)
        left, right = (card_a, card_b) if a_left else (card_b, card_a)
        items.append(dict(
            item_id=f"I{n:03d}",
            pair_id=f"P{k + 1:03d}",
            presentation=presentation,
            stratum=ch["stratum"],
            campus=int(cand.at[i, "campus"]),
            site=SITE_LETTER[int(cand.at[i, "campus"])],
            anchor_bh=round(float(cand.at[i, "anchor_bh"]), 4),
            order_a=card_a["order_id"], order_b=card_b["order_id"],
            a_on_left=bool(a_left),
            left=left, right=right,
        ))
    return items


# --------------------------------------------------------------------------- #
# HTML questionnaire                                                          #
# --------------------------------------------------------------------------- #
HTML_CSS = """
:root { --ink:#111; --rule:#c8c8c8; --soft:#f6f5f2; --accent:#33506e; }
* { box-sizing: border-box; }
body { font-family: "Times New Roman", Times, "Liberation Serif", serif;
       font-size: 16px; line-height: 1.5; color: var(--ink); background: #fff;
       margin: 0; padding: 0 16px 80px; }
.wrap { max-width: 940px; margin: 0 auto; }
h1 { font-size: 26px; margin: 28px 0 4px; }
h2 { font-size: 20px; margin: 26px 0 8px; border-bottom: 1px solid var(--rule);
     padding-bottom: 4px; }
h3 { font-size: 17px; margin: 18px 0 6px; }
p, li { font-size: 16px; }
.lead { color: var(--ink); }
.note { background: var(--soft); border: 1px solid var(--rule); padding: 12px 16px;
        margin: 14px 0; }
table.legend { border-collapse: collapse; margin: 10px 0 4px; width: 100%; }
table.legend th, table.legend td { border: 1px solid var(--rule); padding: 5px 8px;
        text-align: left; font-size: 15px; }
table.legend th { background: var(--soft); font-weight: bold; }
.item { border: 1px solid var(--rule); margin: 22px 0; padding: 0 0 12px; }
.item > .hdr { background: var(--soft); border-bottom: 1px solid var(--rule);
        padding: 8px 14px; font-weight: bold; }
.item .sub { font-weight: normal; font-size: 14px; }
.cards { display: flex; flex-wrap: wrap; gap: 14px; padding: 14px; }
.card { flex: 1 1 380px; min-width: 300px; border: 1px solid var(--rule);
        padding: 10px 12px; }
.card .side { font-weight: bold; text-transform: uppercase; letter-spacing: .06em;
        font-size: 13px; color: var(--accent); margin-bottom: 6px; }
.card .desc { font-size: 16px; margin: 0 0 10px; }
.card dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 10px;
        margin: 0; font-size: 15px; }
.card dt { font-weight: bold; }
.card dd { margin: 0; }
.answer { padding: 0 14px; }
.answer fieldset { border: 1px solid var(--rule); margin: 0 0 10px; padding: 8px 12px; }
.answer legend { font-weight: bold; font-size: 15px; padding: 0 6px; }
label.opt { display: inline-block; margin-right: 18px; font-size: 15px; }
textarea { width: 100%; font-family: inherit; font-size: 15px; padding: 6px;
        border: 1px solid var(--rule); }
input[type=text] { font-family: inherit; font-size: 15px; padding: 5px;
        border: 1px solid var(--rule); }
.bar { position: sticky; bottom: 0; background: #fff; border-top: 2px solid var(--accent);
        padding: 10px 0; margin-top: 28px; }
button { font-family: inherit; font-size: 16px; padding: 8px 16px;
        border: 1px solid var(--accent); background: var(--accent); color: #fff;
        cursor: pointer; }
button.secondary { background: #fff; color: var(--accent); }
#status { font-size: 15px; margin-left: 12px; }
#fallback { width: 100%; height: 180px; display: none; margin-top: 10px; }
@media print { .bar, button, #fallback { display: none; } .item { page-break-inside: avoid; } }
@media (prefers-color-scheme: dark) {
  body { background: #fff; color: var(--ink); }
}
"""

HTML_JS = r"""
(function () {
  var ITEMS = __ITEMS__;
  var t0 = {}, seen = {};
  ITEMS.forEach(function (it) {
    var name = "q_" + it.id;
    document.querySelectorAll('input[name="' + name + '"]').forEach(function (el) {
      el.addEventListener('change', function () {
        if (!seen[it.id]) { seen[it.id] = true; }
        if (!t0[it.id]) { t0[it.id] = Date.now(); }
      });
    });
  });
  var pageOpen = Date.now();
  function val(name) {
    var el = document.querySelector('input[name="' + name + '"]:checked');
    return el ? el.value : "";
  }
  function txt(id) { var el = document.getElementById(id); return el ? el.value : ""; }
  function esc(s) {
    s = (s === null || s === undefined) ? "" : String(s);
    if (/[",\n\r]/.test(s)) { return '"' + s.replace(/"/g, '""') + '"'; }
    return s;
  }
  function buildCSV() {
    var head = ["rater_id", "rater_role", "rater_years_fm", "item_id", "pair_id",
                "presented_left_order", "presented_right_order", "choice_side",
                "chosen_order_id", "confidence", "reason",
                "seconds_from_start"];
    var rid = txt("rater_id").trim();
    var role = txt("rater_role").trim();
    var yrs = txt("rater_years").trim();
    var lines = [head.join(",")];
    var missing = 0;
    ITEMS.forEach(function (it) {
      var side = val("q_" + it.id);
      var conf = val("c_" + it.id);
      var reason = txt("r_" + it.id);
      if (!side) { missing += 1; }
      var chosen = side === "L" ? it.left : (side === "R" ? it.right : "");
      // elapsed seconds from opening the page to first answering this item;
      // differences between consecutive items give the time each one took
      var secs = t0[it.id] ? Math.round((t0[it.id] - pageOpen) / 1000) : "";
      lines.push([rid, role, yrs, it.id, it.pair, it.left, it.right,
                  side, chosen, conf, reason, secs].map(esc).join(","));
    });
    return { csv: lines.join("\n") + "\n", missing: missing, rid: rid };
  }
  function report(msg) { document.getElementById("status").textContent = msg; }
  document.getElementById("btn-save").addEventListener("click", function () {
    var r = buildCSV();
    if (!r.rid) { report("Please type your participant code at the top first."); return; }
    var name = "y3_w8_responses_" + r.rid.replace(/[^A-Za-z0-9_-]/g, "_") + ".csv";
    var blob = new Blob([r.csv], { type: "text/csv;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url; a.download = name; document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(url);
    report(r.missing ? ("Saved " + name + " -- " + r.missing + " item(s) still blank.")
                     : ("Saved " + name + " -- all items answered. Thank you."));
  });
  document.getElementById("btn-show").addEventListener("click", function () {
    var r = buildCSV();
    var fb = document.getElementById("fallback");
    fb.style.display = "block"; fb.value = r.csv; fb.select();
    report("Text shown below: select all, copy, and paste it into an email or a file.");
  });
})();
"""


def render_html(items: list[dict], meta: dict) -> str:
    e = html.escape
    legend_rows = "".join(
        f"<tr><td>{e(CLASS_NAME[c])}</td><td>{CLASS_SLA_DAYS[c]:.0f} business "
        f"day{'s' if CLASS_SLA_DAYS[c] >= 2 else ''}</td></tr>"
        for c in (1, 2, 3, 4))

    def card_html(side: str, c: dict) -> str:
        bld = (f"<dt>{e(c['building_label'])}</dt><dd>{e(c['building'])}</dd>"
               if c["building"] else "")
        return f"""
        <div class="card">
          <div class="side">{e(side)}</div>
          <p class="desc">{e(c['description'])}</p>
          <dl>
            <dt>System</dt><dd>{e(c['system'])}</dd>
            <dt>Component</dt><dd>{e(c['component'])}</dd>
            {bld}
            <dt>Recorded priority</dt><dd>{e(c['recorded_class_label'])}
                (target response {c['target_response_days']:.0f} business days)</dd>
            <dt>Waiting so far</dt><dd>{c['waiting_days']:.1f} business days</dd>
            <dt>Estimated labour</dt><dd>{c['estimated_labour_hours']:.1f} hours</dd>
          </dl>
        </div>"""

    blocks = []
    total = len(items)
    for n, it in enumerate(items, start=1):
        q = it["item_id"]
        blocks.append(f"""
    <div class="item" id="item-{e(q)}">
      <div class="hdr">Item {n} of {total}
        <span class="sub">&nbsp;&nbsp;Site {e(it['site'])} &middot; both jobs are
        waiting in the same queue at the same moment</span></div>
      <div class="cards">
        {card_html('Job on the left', it['left'])}
        {card_html('Job on the right', it['right'])}
      </div>
      <div class="answer">
        <fieldset>
          <legend>Which job should be started first?</legend>
          <label class="opt"><input type="radio" name="q_{e(q)}" value="L"> The job on the left</label>
          <label class="opt"><input type="radio" name="q_{e(q)}" value="R"> The job on the right</label>
        </fieldset>
        <fieldset>
          <legend>How confident are you?</legend>
          <label class="opt"><input type="radio" name="c_{e(q)}" value="1"> Not confident</label>
          <label class="opt"><input type="radio" name="c_{e(q)}" value="2"> Fairly confident</label>
          <label class="opt"><input type="radio" name="c_{e(q)}" value="3"> Very confident</label>
        </fieldset>
        <fieldset>
          <legend>Why? (optional, one line)</legend>
          <textarea id="r_{e(q)}" rows="2" placeholder="e.g. a leak spreads, the other can wait"></textarea>
        </fieldset>
      </div>
    </div>""")

    js_items = json.dumps([
        dict(id=it["item_id"], pair=it["pair_id"],
             left=it["left"]["order_id"], right=it["right"]["order_id"])
        for it in items], separators=(",", ":"))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Work-order urgency pilot: which job goes first?</title>
<style>{HTML_CSS}</style>
</head>
<body>
<div class="wrap">

<h1>Which job should go first?</h1>
<p class="lead">A short study on how facility supervisors rank maintenance work
orders. You will see {total} pairs of real work orders. For each pair, say which
of the two you would start first.</p>

<div class="note">
<p><b>How long it takes.</b> About 35 to 45 minutes. You can stop and come back:
the page keeps your answers as long as you do not close the tab.</p>
<p><b>Anonymous, aggregate use.</b> We do not record your name, your employer, or
anything that identifies you. Answers are reported only as group statistics
(how often practitioners agree with each other) in a research paper on automating
work-order dispatch. There are no right answers and nothing is being tested about
you. You may stop at any time and your partial answers can simply be discarded.</p>
<p><b>Where the jobs come from.</b> Every work order shown is a real, already
completed record from a public research dataset of North American university
campuses (FMUCD, Mendeley Data, CC BY-NC). The text is shown as it was written.</p>
</div>

<h2>Before you start</h2>
<p>Please fill these in. The participant code is whatever the person who sent you
this file asked you to use (for example <i>R1</i>); it is how your sheet is
labelled, not who you are.</p>
<p>
  <label>Participant code &nbsp;<input type="text" id="rater_id" size="10"></label>
  &nbsp;&nbsp;
  <label>Your role &nbsp;<input type="text" id="rater_role" size="28"
     placeholder="e.g. FM supervisor, planner, technician"></label>
  &nbsp;&nbsp;
  <label>Years in facilities &nbsp;<input type="text" id="rater_years" size="5"></label>
</p>

<h2>Instructions</h2>
<ol>
  <li>Each item shows two work orders that are sitting in the same site's queue at
      the same moment. Assume a team member is free now and can start either one.</li>
  <li>Pick the one you would start first, on operational urgency: what happens if
      it waits another day, who or what is affected, and how far it has already
      slipped.</li>
  <li>You may disagree with the recorded priority. It is shown because a
      supervisor sees it, not because it is correct.</li>
  <li>Answer from your own judgement and go with your first instinct. Around 40
      seconds per item is normal.</li>
  <li>The optional one-line reason is genuinely useful to us, but skip it whenever
      you are not sure what to write.</li>
  <li>Some campuses record a building descriptor and some do not, so a few items
      carry one line fewer. The two jobs within an item always show the same
      fields, so neither is ever better documented than the other.</li>
</ol>

<h3>What the recorded priority means</h3>
<table class="legend">
  <tr><th>Recorded priority</th><th>Target response time</th></tr>
  {legend_rows}
</table>
<p style="font-size:15px;">Fields you will <i>not</i> see, because a supervisor
does not have them when the job is dispatched: how long the job actually took,
what it eventually cost, and when it was closed out. "Estimated labour" is the
work content a planner would estimate up front.</p>

<h2>The items</h2>
{''.join(blocks)}

<h2>Finishing</h2>
<p>When you are done, press <b>Save my answers</b>. Your browser will download one
small CSV file. Email that file back. If nothing downloads, press
<b>Show answers as text</b> and paste the text into your reply instead.</p>

<div class="bar">
  <button id="btn-save" type="button">Save my answers (CSV)</button>
  <button id="btn-show" type="button" class="secondary">Show answers as text</button>
  <span id="status"></span>
  <textarea id="fallback" readonly></textarea>
</div>

<p style="font-size:14px; color:#333; margin-top:24px;">
Instrument build {e(meta['built'])} &middot; seed {meta['seed']} &middot;
{meta['n_pairs']} pairs, {total} items &middot; manifest
{e(meta['manifest_name'])}. Questions: {e(meta['contact'])}.
</p>

</div>
<script>{HTML_JS.replace('__ITEMS__', js_items)}</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "fmucd_pilot_pool.parquet")
    ap.add_argument("--nrows", type=int, default=None, help="debug: truncate the raw read")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--contact", default="ziheng.zhang@singaporetech.edu.sg")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "responses").mkdir(exist_ok=True)
    (out / "responses" / ".gitkeep").touch()
    RES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pool, audit = load_pool(args.raw, args.nrows, args.cache, args.rebuild_cache)

    chat, chat_rec = fit_attribute_class(pool)
    tprior = trade_urgency_prior(pool)
    print(f"attribute-implied class model: R2 = {chat_rec['r2']:.4f} "
          f"on {chat_rec['n']:,} orders, {chat_rec['n_params']} parameters")

    cand = build_candidate_pairs(pool, chat, rng)
    print(f"candidate pairs: {len(cand):,}")
    print("  by stratum:", dict(Counter(cand['stratum'].tolist())))

    chosen, sel_info = select_pairs(cand, pool, rng)
    print(f"selected pairs: {len(chosen)}  realised strata: {sel_info['realised']}")
    if len(chosen) < sum(STRATUM_TARGET.values()):
        print("  WARNING: a stratum could not be filled; realised counts above.")

    side_a_left = assign_sides(chosen, cand, rng)
    n_a_left = sum(1 for v in side_a_left.values() if v)
    n_urg_left, n_diff = urgent_side_counts(chosen, cand, side_a_left)

    repeats = choose_repeats(chosen, cand, rng)
    sequence = sequence_items(len(chosen), repeats, rng)
    items = build_items(chosen, cand, pool, side_a_left, sequence)

    # ---- manifest --------------------------------------------------------- #
    man_rows = []
    for it in items:
        i = chosen[int(it["pair_id"][1:]) - 1]["cand_row"]
        a_row, b_row = int(cand.at[i, "row_a"]), int(cand.at[i, "row_b"])
        ra, rb = pool.loc[a_row], pool.loc[b_row]
        man_rows.append(dict(
            item_id=it["item_id"], pair_id=it["pair_id"],
            presentation=it["presentation"],
            is_repeat=int(it["presentation"] == 2),
            stratum=it["stratum"], stratum_label=STRATUM_LABEL[it["stratum"]],
            campus=it["campus"], site=it["site"], anchor_bh=it["anchor_bh"],
            order_a=it["order_a"], order_b=it["order_b"],
            left_order=it["left"]["order_id"], right_order=it["right"]["order_id"],
            a_on_left=int(it["a_on_left"]),
            cls_a=int(ra["cls"]), cls_b=int(rb["cls"]),
            d_cls=int(ra["cls"]) - int(rb["cls"]),
            chat_a=round(float(chat[a_row]), 6), chat_b=round(float(chat[b_row]), 6),
            d_chat=round(float(chat[a_row] - chat[b_row]), 6),
            trade_a=str(ra["trade"]), trade_b=str(rb["trade"]),
            trade_prior_a=tprior[str(ra["trade"])], trade_prior_b=tprior[str(rb["trade"])],
            labor_h_a=round(float(ra["labor_h"]), 4), labor_h_b=round(float(rb["labor_h"]), 4),
            log1p_labor_a=round(float(np.log1p(ra["labor_h"])), 6),
            log1p_labor_b=round(float(np.log1p(rb["labor_h"])), 6),
            wait_days_a=round(float(cand.at[i, "wait_a"]) / 8.0, 4),
            wait_days_b=round(float(cand.at[i, "wait_b"]) / 8.0, 4),
            is_pm_a=int(bool(ra["is_pm"])), is_pm_b=int(bool(rb["is_pm"])),
            system_a=str(ra["system"]), system_b=str(rb["system"]),
            has_building_line=int(bool(str(ra["bctx"]))),
        ))
    man = pd.DataFrame(man_rows)
    man_path = out / "y3_w8_manifest.csv"
    man.to_csv(man_path, index=False)

    # ---- response template ------------------------------------------------ #
    tmpl_path = out / "y3_w8_response_template.csv"
    with open(tmpl_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rater_id", "rater_role", "rater_years_fm", "item_id", "pair_id",
                    "presented_left_order", "presented_right_order", "choice_side",
                    "chosen_order_id", "confidence", "reason",
                    "seconds_from_start"])
        for it in items:
            w.writerow(["", "", "", it["item_id"], it["pair_id"],
                        it["left"]["order_id"], it["right"]["order_id"],
                        "", "", "", "", ""])

    # ---- items json + build meta ------------------------------------------ #
    meta = dict(
        built=time.strftime("%Y-%m-%d"), seed=int(args.seed),
        n_pairs=len(chosen), n_items=len(items), n_repeats=len(repeats),
        manifest_name=man_path.name, contact=args.contact,
        raw_path=str(args.raw),
        campuses=list(instances.CAMPUS_SET),
        backlog_bh=BACKLOG_BH, min_backlog=MIN_BACKLOG,
        anchors_per_campus=ANCHORS_PER_CAMPUS, pairs_per_anchor=PAIRS_PER_ANCHOR,
        disagree_margin=DISAGREE_MARGIN,
        stratum_target=STRATUM_TARGET, stratum_realised=sel_info["realised"],
        per_campus_pairs=sel_info["per_campus"],
        repeat_pair_ids=[f"P{k + 1:03d}" for k in repeats],
        repeat_min_gap=REPEAT_MIN_GAP,
        a_on_left_count=int(n_a_left), a_on_left_of=len(chosen),
        urgent_class_left_count=int(n_urg_left),
        urgent_class_left_of=int(n_diff),
        class_sla_business_days={str(k): round(v, 2) for k, v in CLASS_SLA_DAYS.items()},
        attribute_class_model=chat_rec,
        trade_urgency_prior=tprior,
        corpus_audit=audit,
    )
    with open(out / "y3_w8_items.json", "w") as fh:
        json.dump(dict(meta=meta, items=items), fh, indent=1)

    # ---- questionnaire ---------------------------------------------------- #
    html_path = out / "y3_w8_pilot.html"
    with open(html_path, "w") as fh:
        fh.write(render_html(items, meta))

    # ---- build report ----------------------------------------------------- #
    strat_counts = Counter(ch["stratum"] for ch in chosen)
    report = dict(
        meta=meta,
        realised_strata={s: int(strat_counts.get(s, 0)) for s in STRATA},
        realised_per_campus={str(k): int(v) for k, v in sel_info["per_campus"].items()},
        candidate_pairs_by_stratum={k: int(v) for k, v in
                                    Counter(cand["stratum"].tolist()).items()},
        balance=dict(a_on_left=int(n_a_left), of=len(chosen),
                     urgent_recorded_class_on_left=int(n_urg_left),
                     of_class_differing=int(n_diff)),
        waiting_days=dict(
            min=round(float(min(min(it["left"]["waiting_days"], it["right"]["waiting_days"])
                                for it in items)), 2),
            max=round(float(max(max(it["left"]["waiting_days"], it["right"]["waiting_days"])
                                for it in items)), 2)),
        items_with_building_line=int(sum(1 for it in items if it["left"]["building"])),
        recorded_class_counts={
            str(c): int(sum((it["left"]["recorded_class"] == c) +
                            (it["right"]["recorded_class"] == c) for it in items))
            for c in (1, 2, 3, 4)},
        outputs=[str(html_path), str(man_path), str(tmpl_path),
                 str(out / "y3_w8_items.json")],
    )
    with open(RES_DIR / "build_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== realised design ===")
    for s in STRATA:
        print(f"  {s:26s} target {STRATUM_TARGET[s]:2d}   realised "
              f"{strat_counts.get(s, 0):2d}   ({STRATUM_LABEL[s]})")
    print(f"  pairs per campus: {dict(sorted(sel_info['per_campus'].items()))}")
    print(f"  canonical A on the left: {n_a_left}/{len(chosen)}")
    print(f"  more-urgent recorded class on the left: {n_urg_left}/{n_diff}")
    print(f"  repeats: {[f'P{k + 1:03d}' for k in repeats]} (each shown twice, "
          f"sides swapped, >= {REPEAT_MIN_GAP} items apart)")
    print(f"  items presented: {len(items)}")
    print(f"\nwrote {html_path}\n      {man_path}\n      {tmpl_path}"
          f"\n      {out / 'y3_w8_items.json'}\n      {RES_DIR / 'build_report.json'}")


if __name__ == "__main__":
    main()
