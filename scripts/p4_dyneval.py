#!/usr/bin/env python
"""P4 dynamic-evaluation runner (Gate B + E2/E3): every online policy on the
dynamic replay/contended/storm regimes, scored by the independent validator.

Regimes (docs/decision_log.md 2026-07-05, Gate B protocol amendment)
------------------------------------------------------------------
  replay-default : ALL replay TEST instances, sizes {150,400}, all 6 campuses,
                   crew_multiplier m = 1.0 (capacity-adequate reference).
  replay-tight   : the SAME replay instances at m in {0.6, 0.8}
                   (fmwos.tightness.scale_crew -> contended).
  storm          : generator cells arrival_multiplier {1.25,1.5} x
                   crew_multiplier {0.8,1.0} x sizes {150,400} x train campuses
                   {5,9,10,12}, 30 instances/cell, seeds 40000+i, generated on
                   the fly from results/p2_generator/params_c*.json and saved
                   under data/processed/instances/c<k>/storm/ (track='storm').
                   The arrival grid is parameterizable via --storm-arrivals
                   (docs/decision_log.md 2026-07-05 "Storm grid too mild"; see
                   Extensions for the seed scheme of non-default values).

Methods per instance
--------------------
  PDRs : edd, wspt, atc, pfifo, mor, random          (seed 301)
  RL   : rl301, rl302, rl303  (greedy argmax, results/p3_train/seed<t>/best.pt,
         run through the DispatchEnv reset()/step() path on CPU)
  roll : rollcp2  -- ONLY on a subsample: the first --rollcp-per-cell (default 8)
         instances per (regime cell, campus, size) in sorted-id order.

Every schedule is scored ONLY by fmwos.validator.  Output rows carry the base
metric columns plus regime, crew_multiplier, arrival_multiplier, method, seed
and the per-decision latency stat mean_ms_per_decision (plus mean_replan_s for
rollcp2).

Sharded / resumable (copied from scripts/p2_e1.py)
--------------------------------------------------
One task == one instance-configuration x all its methods.  Each finished config
is written atomically to results/p4_dyneval/shards/<shard_id>.json holding every
method row; a config is skipped on resume once its shard holds all expected
methods.  --merge (auto-run at the end) concatenates the shards into
results/p4_dyneval/results.csv.

Usage
-----
    PYTHONPATH=src python scripts/p4_dyneval.py [--workers 8] [--limit N]
        [--regime replay-default,replay-tight,storm] [--no-rollcp]
        [--rollcp-per-cell N] [--campus C[,C...]] [--size S[,S...]] [--merge]
        [--with-pmmix] [--rl-tag TAG] [--rl-dir DIR] [--out DIR]
        [--storm-arrivals CSV]

Extensions
----------
  --storm-arrivals CSV : the storm arrival_multiplier grid (default "1.25,1.5"
        reproduces the current cells byte-identically).  Values IN the default
        set always keep their existing seeds 40000+i and on-disk instances
        (untouched, idempotent).  Every NEW value generates fresh instances
        with the documented seed scheme
            seed = 60000 + cell_index*1000 + i          (i in 0..29)
            cell_index = int(round(am*100))*16
                         + STORM_CAMPUSES.index(campus)*4
                         + STORM_SIZES.index(size)*2
                         + STORM_CREW.index(cm)
        i.e. every (am, campus, size, cm) cell owns a disjoint 1000-seed block
        that is a pure function of the cell (stable across invocations,
        independent of which other arrivals are requested) and disjoint from
        the legacy storm 40000+ and pmmix 50000+ ranges.  Index rows are
        UPSERTED by id, so a partial-grid run never drops other cells' rows,
        and shard finished-checks are per config id (arrival is part of the
        id): adding arrivals adds pending configs and never invalidates
        finished ones.
  --rl-tag / --rl-dir : the three online policies are loaded from
        ``DIR/seed<t>/best.pt`` (t in {301,302,303}) and their method columns
        are ``f'{TAG}{t}'``.  Defaults (tag 'rl', dir results/p3_train)
        reproduce rl301..rl303.  The per-config expected-method set derives from
        the active tag, so a shard finished under rl* is NOT considered finished
        for e.g. v2rl* (it is re-run and atomically replaced).
  --with-pmmix        : add the 'pmmix' regime -- generator cells with
        pm_share_override in {0.2,0.5,0.8} x crew_multiplier in {0.6,0.8,1.0} x
        arrival 1.0 x sizes {150,400} x campuses {5,9,10,12}, 30 instances/cell,
        FRESH seeds 50000+i (docs/decision_log.md 2026-07-05 v2 disclosure).  Instances
        are persisted under data/processed/instances/c<k>/pmmix/<size>/ and
        indexed (track='pmmix', split='test') idempotently.  pmmix is selectable
        via --regime only when --with-pmmix is also passed.
  --with-storm2       : add the 'storm2' regime -- a fixed-window (80 bh)
        UTILIZATION sweep.  For each campus {5,9,10,12} and utilization target
        u in {0.7,0.9,1.0,1.1,1.3} (crew 1.0, 30 instances/cell) the workload is
        drawn over a FIXED 80 bh window at arrival_multiplier = u / u0, where
        u0 = generator.base_utilization(pack) is the offered-load/capacity ratio
        at multiplier 1 (clipped-lognormal mean p_bh).  Unlike storm (first-N
        sampling -> fixed total work -> flat arrival axis), storm2's work-order count n GROWS with u, so it
        actually creates sustained overload -- the E2 intensity curve / E3
        overload frontier.  Seeds 70000 + cell_index*1000 + i (cell_index =
        campus_idx*5 + u_idx; disjoint 1000-block per cell, disjoint from all
        prior ranges).  Instances persist under
        data/processed/instances/c<k>/storm2/w80/ (track='storm2', split='test',
        size_class = the variable n) and are idempotently upserted into
        index.csv.  Two extra result columns u_target / u_realized (null for
        every other regime) record targeted vs realized utilization
        (realized = sum p_bh / (crew * window)).  n varies per draw, so --size
        never filters storm2 rows and rollcp2's first-8-per-cell subsample keys
        on (campus, u) not size.  Selectable via --regime only with --with-storm2.
  --out DIR           : redirect the results root (shards, results.csv,
        meta.json) so v2/pmmix runs write to fresh dirs (default
        results/p4_dyneval).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Keep BLAS/OpenMP/torch from oversubscribing the shared 24-core box: the only
# threaded work we allow is CP-SAT's workers=2 and torch's 2 inference threads.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

# NOTE: torch / fmwos.policy are imported LAZILY inside the worker so the parent
# never initialises torch before fork() (fork-safety on the shared box).
from fmwos import generator, pdrs, rolling, tightness  # noqa: E402
from fmwos.env import DispatchEnv                       # noqa: E402
from fmwos.validator import validate                    # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
INST_ROOT = _ROOT / "data" / "processed" / "instances"
INDEX_CSV = INST_ROOT / "index.csv"
PARAMS_DIR = _ROOT / "results" / "p2_generator"
TRAIN_DIR = _ROOT / "results" / "p3_train"
OUT_DIR = _ROOT / "results" / "p4_dyneval"
SHARD_DIR = OUT_DIR / "shards"
OUT_CSV = OUT_DIR / "results.csv"
META_JSON = OUT_DIR / "meta.json"

PDR_RULES = ["edd", "wspt", "atc", "pfifo", "mor", "random"]
RL_SEEDS = [301, 302, 303]
# RL method naming + checkpoint root are reconfigurable (--rl-tag/--rl-dir).
# _configure_rl() rederives RL_METHODS/BASE_METHODS/_METHOD_ORDER in the parent
# BEFORE the worker pool is forked, so every forked worker inherits the tag set.
RL_TAG = "rl"                                          # method-name prefix
RL_DIR = TRAIN_DIR                                     # checkpoint root
RL_ARCH = "mlp"                                        # policy class to load
RL_METHODS = ["%s%d" % (RL_TAG, t) for t in RL_SEEDS]
BASE_METHODS = PDR_RULES + RL_METHODS                 # every config runs these
ROLLCP_METHOD = "rollcp2"
_METHOD_ORDER = {m: i for i, m in enumerate(BASE_METHODS + [ROLLCP_METHOD])}

SEED = 301
ROLLCP_BUDGET_S = 2.0
CPSAT_WORKERS = 2
TORCH_THREADS = 2

REGIMES = ["replay-default", "replay-tight", "storm"]
PMMIX_REGIME = "pmmix"                                 # opt-in via --with-pmmix
STORM2_REGIME = "storm2"                               # opt-in via --with-storm2
ALL_REGIMES = REGIMES + [PMMIX_REGIME, STORM2_REGIME]
_REGIME_ORDER = {r: i for i, r in enumerate(ALL_REGIMES)}

REPLAY_SIZES = [150, 400]
TIGHT_MULTS = [0.6, 0.8]

STORM_CAMPUSES = [5, 9, 10, 12]
STORM_SIZES = [150, 400]
STORM_ARRIVAL = [1.25, 1.5]          # DEFAULT arrival grid (seeds 40000+i)
STORM_CREW = [0.8, 1.0]
STORM_N = 30
STORM_SEED_BASE = 40000              # default-arrival cells (legacy scheme)
STORM_NEW_SEED_BASE = 60000          # non-default arrival cells (see below)
STORM_CELL_SEED_STRIDE = 1000        # disjoint per-cell seed blocks

# pmmix: PM/CM-ratio sweep at contended crew (fresh seeds per v2 disclosure).
PMMIX_CAMPUSES = [5, 9, 10, 12]
PMMIX_SIZES = [150, 400]
PMMIX_PM_SHARE = [0.2, 0.5, 0.8]
PMMIX_CREW = [0.6, 0.8, 1.0]
PMMIX_ARRIVAL = 1.0
PMMIX_N = 30
PMMIX_SEED_BASE = 50000

# storm2: fixed-window UTILIZATION sweep (the E2 intensity curve / E3 overload
# frontier).  Unlike storm (first-N sampling -> fixed total work -> flat arrival
# axis), storm2 draws a rate-scaled
# Poisson workload over a FIXED 80 bh window, so the offered load AND the
# work-order count n grow with the targeted utilization u.  Per campus the
# arrival_multiplier = u / u0 where u0 = generator.base_utilization(pack) is the
# offered-load/capacity ratio at multiplier 1 (clipped-lognormal mean p_bh), so
# the DRAWN workload targets utilization u (realized u recorded per instance).
STORM2_CAMPUSES = [5, 9, 10, 12]
STORM2_WINDOW_BH = 80.0
STORM2_UTIL = [0.7, 0.9, 1.0, 1.1, 1.3]
STORM2_CREW = 1.0
STORM2_N = 30
STORM2_SEED_BASE = 70000             # disjoint from 20000/40000/50000/60000+
STORM2_CELL_SEED_STRIDE = 1000       # per-(campus,u) disjoint 1000-seed block

DEFAULT_ROLLCP_PER_CELL = 8

FIELDS = [
    "id", "campus", "track", "split", "size", "regime", "crew_multiplier",
    "arrival_multiplier", "pm_share_override", "method", "seed", "feasible",
    "wwt", "makespan", "mean_flow", "breach_share", "breach_p1", "breach_p2",
    "breach_p3", "breach_p4", "wall_seconds", "decisions",
    "mean_ms_per_decision", "mean_replan_s",
    "u_target", "u_realized",
]


# --------------------------------------------------------------------------- #
# Reconfiguration (called in the parent, before the worker pool is forked)
# --------------------------------------------------------------------------- #
def _discover_seeds(rl_dir):
    """Integer seeds discovered as ``<rl_dir>/seed<N>/best.pt`` checkpoint dirs.

    Generalizes the old hard-coded {301,302,303}: any number of seed dirs under
    the checkpoint root are picked up (e.g. the 10-seed MLP pool in
    results/p3_train/v2), sorted numerically."""
    seeds = []
    p = Path(rl_dir)
    if p.exists():
        for d in sorted(p.glob("seed*")):
            m = re.match(r"seed(\d+)$", d.name)
            if m and d.is_dir() and (d / "best.pt").exists():
                seeds.append(int(m.group(1)))
    return sorted(set(seeds))


def _configure_rl(tag, rl_dir, seeds=None, arch="mlp"):
    """Set the RL method tag + checkpoint root/arch and rederive the name lists.

    ``seeds`` overrides the seed set explicitly; when None the seeds are
    auto-discovered from ``rl_dir`` (falling back to the legacy {301,302,303}
    if none are found, so an empty/undertrained dir never silently drops RL).
    Forked workers inherit these module globals, so the tagged BASE_METHODS /
    _METHOD_ORDER / RL_DIR / RL_ARCH are visible in ``_run_one`` without
    threading them through every config dict."""
    global RL_TAG, RL_DIR, RL_ARCH, RL_SEEDS, RL_METHODS, BASE_METHODS, _METHOD_ORDER
    RL_TAG = str(tag)
    RL_DIR = Path(rl_dir)
    RL_ARCH = str(arch)
    if seeds is not None:
        RL_SEEDS = [int(s) for s in seeds]
    else:
        disc = _discover_seeds(rl_dir)
        RL_SEEDS = disc if disc else [301, 302, 303]
    RL_METHODS = ["%s%d" % (RL_TAG, t) for t in RL_SEEDS]
    BASE_METHODS = PDR_RULES + RL_METHODS
    _METHOD_ORDER = {m: i for i, m in enumerate(BASE_METHODS + [ROLLCP_METHOD])}


def _configure_out(out_dir):
    """Point the results root (shards / results.csv / meta.json) at ``out_dir``."""
    global OUT_DIR, SHARD_DIR, OUT_CSV, META_JSON
    OUT_DIR = Path(out_dir)
    SHARD_DIR = OUT_DIR / "shards"
    OUT_CSV = OUT_DIR / "results.csv"
    META_JSON = OUT_DIR / "meta.json"


def _rl_method(seed):
    return "%s%d" % (RL_TAG, seed)


# --------------------------------------------------------------------------- #
# Storm-instance generation (idempotent; run in the parent before the pool)
# --------------------------------------------------------------------------- #
def _storm_id(campus, size, am, cm, i):
    return "c%02d_storm_%d_a%d_c%d_%04d" % (
        campus, size, int(round(am * 100)), int(round(cm * 100)), i)


def _storm_seed(campus, size, am, cm, i):
    """Deterministic storm seed (documented scheme).

    * am in STORM_ARRIVAL (default grid {1.25, 1.5}): seed = 40000 + i -- the
      original scheme; existing on-disk instances stay byte-identical.
    * NEW am values (--storm-arrivals extension): seed = 60000 +
      cell_index*1000 + i where
          cell_index = int(round(am*100))*16 + campus_idx*4 + size_idx*2
                       + crew_idx
      (indices into STORM_CAMPUSES / STORM_SIZES / STORM_CREW; 16 = 4*2*2
      cells per arrival value).  cell_index is a pure function of the cell,
      so seeds are stable across invocations regardless of which other
      arrival values are requested, every cell owns a disjoint 1000-seed
      block, and (for any am >= 0.01) the range is disjoint from the legacy
      storm 40000..40029 and pmmix 50000..50029 seeds.
    """
    if am in STORM_ARRIVAL:
        return STORM_SEED_BASE + i
    cell_index = (int(round(am * 100)) * 16
                  + STORM_CAMPUSES.index(campus) * 4
                  + STORM_SIZES.index(size) * 2
                  + STORM_CREW.index(cm))
    return STORM_NEW_SEED_BASE + cell_index * STORM_CELL_SEED_STRIDE + i


def _load_params(campus):
    with open(PARAMS_DIR / ("params_c%d.json" % campus)) as f:
        return json.load(f)


def _ensure_storm_instance(params, campus, size, am, cm, i):
    """Generate + save one storm instance if absent.

    Returns ``(abs_path, id, index_row)`` where ``index_row`` matches the
    data/processed/instances/index.csv columns (track='storm', split='test').
    """
    iid = _storm_id(campus, size, am, cm, i)
    out_dir = INST_ROOT / ("c%02d" % campus) / "storm" / str(size)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (iid + ".json")
    if not path.exists():
        seed = _storm_seed(campus, size, am, cm, i)
        inst = generator.generate(params, size=size, seed=seed,
                                  crew_multiplier=cm, arrival_multiplier=am)
        inst["meta"]["id"] = iid
        inst["meta"]["track"] = "storm"
        inst["meta"]["split"] = "test"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(inst, f, separators=(",", ":"))
        os.replace(tmp, path)
    else:
        with open(path) as f:
            inst = json.load(f)
    row = {
        "id": iid, "campus": campus, "track": "storm", "size_class": size,
        "split": "test", "n_wos": len(inst["work_orders"]),
        "window_start": "synthetic", "window_bh": inst["meta"]["window_bh"],
        "path": "c%02d/storm/%d/%s.json" % (campus, size, iid),
    }
    return path, iid, row


def _upsert_index_rows(rows):
    """Upsert instance rows into index.csv, keyed by ``id`` (idempotent).

    Existing rows with a matching id are replaced IN PLACE (their position and
    every other row are preserved byte-for-byte -- rebuilding an unchanged
    instance rewrites an identical line); genuinely new ids are appended at
    the end.  A partial-grid run (e.g. --storm-arrivals 2.0 alone) therefore
    never drops other cells' rows.  The write is atomic (tmp + os.replace),
    so a concurrent reader never sees a partial file.
    """
    new = {r["id"]: r for r in rows}
    with open(INDEX_CSV, newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames)
        existing = list(reader)
    seen = set()
    tmp = INDEX_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in existing:
            rid = r.get("id")
            if rid in new:
                w.writerow({c: new[rid].get(c, "") for c in cols})
                seen.add(rid)
            else:
                w.writerow(r)
        for r in rows:
            if r["id"] not in seen:
                w.writerow({c: r.get(c, "") for c in cols})
                seen.add(r["id"])
    os.replace(tmp, INDEX_CSV)
    return len(rows)


# --------------------------------------------------------------------------- #
# pmmix-instance generation (idempotent; run in the parent before the pool)
# --------------------------------------------------------------------------- #
def _pmmix_id(campus, size, pm, cm, i):
    return "c%02d_pmmix_%d_p%d_c%d_%04d" % (
        campus, size, int(round(pm * 100)), int(round(cm * 100)), i)


def _ensure_pmmix_instance(params, campus, size, pm, cm, i):
    """Generate + save one pmmix instance if absent (fresh seed 50000+i).

    Returns ``(abs_path, id, index_row)`` where ``index_row`` matches the
    data/processed/instances/index.csv columns (track='pmmix', split='test').
    """
    iid = _pmmix_id(campus, size, pm, cm, i)
    out_dir = INST_ROOT / ("c%02d" % campus) / "pmmix" / str(size)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (iid + ".json")
    if not path.exists():
        seed = PMMIX_SEED_BASE + i
        inst = generator.generate(params, size=size, seed=seed,
                                  crew_multiplier=cm, pm_share_override=pm,
                                  arrival_multiplier=PMMIX_ARRIVAL)
        inst["meta"]["id"] = iid
        inst["meta"]["track"] = "pmmix"
        inst["meta"]["split"] = "test"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(inst, f, separators=(",", ":"))
        os.replace(tmp, path)
    else:
        with open(path) as f:
            inst = json.load(f)
    row = {
        "id": iid, "campus": campus, "track": "pmmix", "size_class": size,
        "split": "test", "n_wos": len(inst["work_orders"]),
        "window_start": "synthetic", "window_bh": inst["meta"]["window_bh"],
        "path": "c%02d/pmmix/%d/%s.json" % (campus, size, iid),
    }
    return path, iid, row


# --------------------------------------------------------------------------- #
# storm2-instance generation (idempotent; run in the parent before the pool)
# --------------------------------------------------------------------------- #
def _storm2_id(campus, u, i):
    return "c%02d_storm2_w%d_u%d_%04d" % (
        campus, int(round(STORM2_WINDOW_BH)), int(round(u * 100)), i)


def _storm2_cell_index(campus, u):
    """Pure function of the (campus, u) cell -> its seed-block index (0..19).

    20 cells (4 campuses x 5 utilization targets); the block index is stable
    across invocations and independent of which other cells are requested."""
    return (STORM2_CAMPUSES.index(campus) * len(STORM2_UTIL)
            + STORM2_UTIL.index(u))


def _storm2_seed(campus, u, i):
    """Deterministic storm2 seed = 70000 + cell_index*1000 + i (i in 0..29).

    cell_index = STORM2_CAMPUSES.index(campus)*5 + STORM2_UTIL.index(u), so every
    (campus, u) cell owns a disjoint 1000-seed block in [70000, 89029] -- a pure
    function of the cell, and disjoint from every prior range: generator
    20000+i, storm-legacy 40000+i, pmmix 50000+i, and storm-new
    60000+cell*1000+i (whose cell_index for any arrival_multiplier >= 1 is
    int(round(am*100))*16 + ... >= 1600, i.e. seeds >= 1.66M).
    """
    return (STORM2_SEED_BASE
            + _storm2_cell_index(campus, u) * STORM2_CELL_SEED_STRIDE + i)


def _storm2_realized_util(inst):
    """Realized utilization of a storm2 instance = sum p_bh / (crew * window)."""
    total_p = sum(float(w["p_bh"]) for w in inst["work_orders"])
    n_crew = len(inst["technicians"])
    win = float(inst["meta"]["window_bh"])
    denom = n_crew * win
    return float(total_p / denom) if denom > 0 else 0.0


def _ensure_storm2_instance(params, campus, u, arrival_multiplier, i):
    """Generate + save one storm2 instance if absent (fixed 80 bh window).

    Returns ``(abs_path, id, index_row, n, u_realized)``.  ``n`` (the realized
    work-order count) and ``u_realized`` are read back from the on-disk instance
    so a resume reports the same size/utilization it was generated with.  The
    index row uses track='storm2', split='test', size_class = actual n.
    """
    iid = _storm2_id(campus, u, i)
    wtag = "w%d" % int(round(STORM2_WINDOW_BH))
    out_dir = INST_ROOT / ("c%02d" % campus) / "storm2" / wtag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (iid + ".json")
    if not path.exists():
        seed = _storm2_seed(campus, u, i)
        inst = generator.generate_window(
            params, window_bh=STORM2_WINDOW_BH, seed=seed,
            crew_multiplier=STORM2_CREW, arrival_multiplier=arrival_multiplier)
        inst["meta"]["id"] = iid
        inst["meta"]["track"] = "storm2"
        inst["meta"]["split"] = "test"
        inst["meta"]["u_target"] = float(u)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(inst, f, separators=(",", ":"))
        os.replace(tmp, path)
    else:
        with open(path) as f:
            inst = json.load(f)
    n = len(inst["work_orders"])
    u_real = _storm2_realized_util(inst)
    row = {
        "id": iid, "campus": campus, "track": "storm2", "size_class": n,
        "split": "test", "n_wos": n,
        "window_start": "synthetic", "window_bh": inst["meta"]["window_bh"],
        "path": "c%02d/storm2/%s/%s.json" % (campus, wtag, iid),
    }
    return path, iid, row, n, u_real


# --------------------------------------------------------------------------- #
# Target-set construction
# --------------------------------------------------------------------------- #
def _read_index():
    with open(INDEX_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _replay_test_rows():
    rows = []
    for r in _read_index():
        if (str(r.get("split", "")).strip().lower() == "test"
                and str(r.get("track", "")).strip().lower() == "replay"
                and int(r["size_class"]) in REPLAY_SIZES):
            rows.append(r)
    return rows


def _replay_path(row):
    p = row.get("path")
    if p:
        cand = INST_ROOT / p
        if cand.exists():
            return cand
    return (INST_ROOT / ("c%02d" % int(row["campus"])) / "replay"
            / str(row["size_class"]) / (row["id"] + ".json"))


def build_targets(regimes, gen_storm=True, gen_pmmix=True, gen_storm2=True,
                  storm_arrivals=None):
    """Return the full ordered list of config dicts (one per instance-config).

    Each config is picklable and self-describing; the worker materialises the
    instance from ``path`` (+ scale_crew for replay-tight).
    ``storm_arrivals`` (None -> STORM_ARRIVAL) is the arrival grid for the
    storm regime; see _storm_seed for the per-value seed scheme.
    """
    if storm_arrivals is None:
        storm_arrivals = list(STORM_ARRIVAL)
    configs = []

    replay_rows = _replay_test_rows()

    # (a) replay-default -----------------------------------------------------
    if "replay-default" in regimes:
        for row in replay_rows:
            configs.append({
                "id": row["id"], "campus": int(row["campus"]),
                "track": "replay", "split": "test",
                "size": int(row["size_class"]), "regime": "replay-default",
                "crew_multiplier": 1.0, "arrival_multiplier": 1.0,
                "kind": "replay", "path": str(_replay_path(row)), "m": 1.0,
            })

    # (b) replay-tight -------------------------------------------------------
    if "replay-tight" in regimes:
        for m in TIGHT_MULTS:
            for row in replay_rows:
                configs.append({
                    "id": "%s_m%s" % (row["id"], m), "campus": int(row["campus"]),
                    "track": "replay", "split": "test",
                    "size": int(row["size_class"]), "regime": "replay-tight",
                    "crew_multiplier": float(m), "arrival_multiplier": 1.0,
                    "kind": "replay", "path": str(_replay_path(row)), "m": float(m),
                })

    # (c) storm --------------------------------------------------------------
    if "storm" in regimes:
        storm_index_rows = []
        for campus in STORM_CAMPUSES:
            params = _load_params(campus) if gen_storm else None
            for size in STORM_SIZES:
                for am in storm_arrivals:
                    for cm in STORM_CREW:
                        for i in range(STORM_N):
                            if gen_storm:
                                path, iid, irow = _ensure_storm_instance(
                                    params, campus, size, am, cm, i)
                                path = str(path)
                                storm_index_rows.append(irow)
                            else:
                                iid = _storm_id(campus, size, am, cm, i)
                                path = str(INST_ROOT / ("c%02d" % campus)
                                           / "storm" / str(size) / (iid + ".json"))
                            configs.append({
                                "id": iid, "campus": campus, "track": "storm",
                                "split": "test", "size": size, "regime": "storm",
                                "crew_multiplier": float(cm),
                                "arrival_multiplier": float(am),
                                "kind": "storm", "path": path, "m": 1.0,
                            })
        if gen_storm and storm_index_rows:
            n = _upsert_index_rows(storm_index_rows)
            print("  storm set ready : %d instance(s) upserted into %s "
                  "(track='storm', split='test', arrivals=%s)"
                  % (n, INDEX_CSV, ",".join(str(a) for a in storm_arrivals)))

    # (d) pmmix --------------------------------------------------------------
    if PMMIX_REGIME in regimes:
        pmmix_index_rows = []
        for campus in PMMIX_CAMPUSES:
            params = _load_params(campus) if gen_pmmix else None
            for size in PMMIX_SIZES:
                for pm in PMMIX_PM_SHARE:
                    for cm in PMMIX_CREW:
                        for i in range(PMMIX_N):
                            if gen_pmmix:
                                path, iid, irow = _ensure_pmmix_instance(
                                    params, campus, size, pm, cm, i)
                                path = str(path)
                                pmmix_index_rows.append(irow)
                            else:
                                iid = _pmmix_id(campus, size, pm, cm, i)
                                path = str(INST_ROOT / ("c%02d" % campus)
                                           / "pmmix" / str(size) / (iid + ".json"))
                            configs.append({
                                "id": iid, "campus": campus, "track": "pmmix",
                                "split": "test", "size": size,
                                "regime": PMMIX_REGIME,
                                "crew_multiplier": float(cm),
                                "arrival_multiplier": float(PMMIX_ARRIVAL),
                                "pm_share_override": float(pm),
                                "kind": "pmmix", "path": path, "m": 1.0,
                            })
        if gen_pmmix and pmmix_index_rows:
            n = _upsert_index_rows(pmmix_index_rows)
            print("  pmmix set ready : %d instance(s) upserted into %s "
                  "(track='pmmix', split='test')" % (n, INDEX_CSV))

    # (e) storm2 (fixed-window utilization sweep) ----------------------------
    if STORM2_REGIME in regimes:
        storm2_index_rows = []
        for campus in STORM2_CAMPUSES:
            params = _load_params(campus)
            u0 = generator.base_utilization(params, crew_multiplier=STORM2_CREW)
            for u in STORM2_UTIL:
                am = float(u / u0) if u0 > 0 else 1.0
                for i in range(STORM2_N):
                    path, iid, irow, n, u_real = _ensure_storm2_instance(
                        params, campus, u, am, i)
                    if gen_storm2:
                        storm2_index_rows.append(irow)
                    configs.append({
                        "id": iid, "campus": campus, "track": "storm2",
                        "split": "test", "size": int(n),
                        "regime": STORM2_REGIME,
                        "crew_multiplier": float(STORM2_CREW),
                        "arrival_multiplier": am,
                        "u_target": float(u), "u_realized": float(u_real),
                        "kind": "storm2", "path": str(path), "m": 1.0,
                    })
        if gen_storm2 and storm2_index_rows:
            n_up = _upsert_index_rows(storm2_index_rows)
            print("  storm2 set ready: %d instance(s) upserted into %s "
                  "(track='storm2', split='test', window=%g bh, u in %s; "
                  "size_class = variable n)"
                  % (n_up, INDEX_CSV, STORM2_WINDOW_BH, STORM2_UTIL))

    configs.sort(key=lambda c: (_REGIME_ORDER[c["regime"]], c["campus"],
                                c["size"], c["crew_multiplier"],
                                c["arrival_multiplier"], c["id"]))
    return configs


def assign_rollcp(configs, per_cell, enabled):
    """Mark ``rollcp=True`` on the first ``per_cell`` configs of every
    (regime, campus, size, crew_multiplier, arrival_multiplier) cell."""
    from collections import defaultdict
    cells = defaultdict(list)
    for c in configs:
        # storm2's per-instance size varies within a (campus, u) cell, so it is
        # dropped from the cell key (arrival_multiplier = u/u0 already
        # identifies the utilization target); every other regime keys on its
        # fixed size class, so first-8-per-cell is unchanged for them.
        size_key = None if c["regime"] == STORM2_REGIME else c["size"]
        key = (c["regime"], c["campus"], size_key, c["crew_multiplier"],
               c["arrival_multiplier"], c.get("pm_share_override"))
        cells[key].append(c)
    for key, group in cells.items():
        group.sort(key=lambda c: c["id"])
        for j, c in enumerate(group):
            c["rollcp"] = bool(enabled and j < per_cell)
    return configs


def _expected_methods(config):
    return BASE_METHODS + ([ROLLCP_METHOD] if config.get("rollcp") else [])


# --------------------------------------------------------------------------- #
# Worker: RL policy cache + rollout
# --------------------------------------------------------------------------- #
_POLICY_CACHE = {}


def _get_policy(seed):
    pol = _POLICY_CACHE.get(seed)
    if pol is None:
        import torch  # lazy: only inside the worker
        torch.set_num_threads(TORCH_THREADS)
        ckpt = str(RL_DIR / ("seed%d" % seed) / "best.pt")
        if RL_ARCH == "attn":
            from fmwos.policy_attn import AttnDispatchPolicy
            pol = AttnDispatchPolicy.load(ckpt, map_location="cpu")
        else:
            from fmwos.policy import DispatchPolicy
            pol = DispatchPolicy.load(ckpt, map_location="cpu")
        pol.eval()
        _POLICY_CACHE[seed] = pol
    return pol


def _rl_rollout(instance, seed):
    pol = _get_policy(seed)
    env = DispatchEnv(instance)
    obs = env.reset()
    done = False
    while not done:
        a, _, _, _ = pol.act(obs, greedy=True, device="cpu")
        obs, _r, done, _info = env.step(a)
    return env.to_schedule(_rl_method(seed), seed=seed)


# --------------------------------------------------------------------------- #
# Row building
# --------------------------------------------------------------------------- #
def _row(config, method, seed, sched, res):
    m = res["metrics"]
    pp = m["per_priority_breach_share"]
    decisions = sched.get("decisions")
    wall = sched.get("wall_seconds")
    mean_ms = None
    if decisions and wall is not None and decisions > 0:
        mean_ms = 1000.0 * float(wall) / float(decisions)
    return {
        "id": config["id"], "campus": config["campus"], "track": config["track"],
        "split": config["split"], "size": config["size"],
        "regime": config["regime"],
        "crew_multiplier": config["crew_multiplier"],
        "arrival_multiplier": config["arrival_multiplier"],
        "pm_share_override": config.get("pm_share_override"),
        "method": method, "seed": seed,
        "feasible": int(bool(res["feasible"])),
        "wwt": m["WWT"], "makespan": m["makespan"], "mean_flow": m["mean_flow"],
        "breach_share": m["breach_share"],
        "breach_p1": pp.get(1), "breach_p2": pp.get(2),
        "breach_p3": pp.get(3), "breach_p4": pp.get(4),
        "wall_seconds": wall, "decisions": decisions,
        "mean_ms_per_decision": mean_ms,
        "mean_replan_s": sched.get("mean_replan_s"),
        "u_target": config.get("u_target"),
        "u_realized": config.get("u_realized"),
    }


def _write_shard(shard_id, shard):
    dst = SHARD_DIR / (shard_id + ".json")
    tmp = SHARD_DIR / (shard_id + ".json.tmp")
    with open(tmp, "w") as f:
        json.dump(shard, f)
    os.replace(tmp, dst)


# --------------------------------------------------------------------------- #
# One instance-config x all its methods (runs in a worker process)
# --------------------------------------------------------------------------- #
def _run_one(config):
    t0 = time.perf_counter()
    try:
        # INCREMENTAL semantics: a re-run under a new tag set must PRESERVE
        # rows computed under earlier tags (e.g. v1 rl301-303, rollcp2) and
        # compute only the methods missing from the existing shard. A shard
        # is a union of everything ever computed for this config; the
        # finished-check upstream tests expected \subseteq rows.
        dst = SHARD_DIR / (config["id"] + ".json")
        old_rows, old_expected = {}, []
        if dst.exists():
            try:
                with open(dst) as f:
                    _old = json.load(f)
                old_rows = _old.get("rows", {}) or {}
                old_expected = list(_old.get("methods_expected", []) or [])
            except Exception:
                old_rows, old_expected = {}, []

        with open(config["path"]) as f:
            instance = json.load(f)
        if config["kind"] == "replay" and config["m"] != 1.0:
            instance = tightness.scale_crew(instance, config["m"])

        out_rows = {}
        infeasible = []
        expected = _expected_methods(config)
        todo = {m for m in expected if m not in old_rows}

        # PDRs (seed 301) ----------------------------------------------------
        for rule in PDR_RULES:
            if rule not in todo:
                continue
            sched = pdrs.dispatch(instance, rule, seed=SEED)
            res = validate(instance, sched)
            out_rows[rule] = _row(config, rule, SEED, sched, res)
            if not res["feasible"]:
                infeasible.append({"method": rule,
                                   "violations": res["violations"][:3]})

        # RL greedy (seeds 301/302/303, tagged by --rl-tag) -----------------
        for t in RL_SEEDS:
            meth = _rl_method(t)
            if meth not in todo:
                continue
            sched = _rl_rollout(instance, t)
            res = validate(instance, sched)
            out_rows[meth] = _row(config, meth, t, sched, res)
            if not res["feasible"]:
                infeasible.append({"method": meth,
                                   "violations": res["violations"][:3]})

        # rollcp2 (subsample) -----------------------------------------------
        if config.get("rollcp") and ROLLCP_METHOD in todo:
            sched = rolling.roll_cpsat(instance, budget_s=ROLLCP_BUDGET_S)
            res = validate(instance, sched)
            out_rows[ROLLCP_METHOD] = _row(config, ROLLCP_METHOD, 0, sched, res)
            if not res["feasible"]:
                infeasible.append({"method": ROLLCP_METHOD,
                                   "violations": res["violations"][:3]})

        out_rows = {**old_rows, **out_rows}
        assert set(expected).issubset(out_rows), "internal: method set mismatch"
        expected_union = sorted(set(expected) | set(old_expected))

        shard = {
            "shard_id": config["id"], "id": config["id"],
            "campus": config["campus"], "regime": config["regime"],
            "size": config["size"], "rows": out_rows,
            "methods_expected": expected_union, "infeasible": infeasible,
            "wall_seconds_total": time.perf_counter() - t0,
        }
        _write_shard(config["id"], shard)
        return {"id": config["id"], "regime": config["regime"], "ok": True,
                "infeasible": infeasible, "wall": shard["wall_seconds_total"]}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"id": config["id"], "regime": config["regime"], "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc(),
                "wall": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# Resumability + merge
# --------------------------------------------------------------------------- #
def _shard_methods():
    """Map shard id -> set of method rows it already holds (corrupt -> absent).

    Compared against the CURRENT run's expected methods per config, so e.g. a
    config first run without rollcp2 and later marked for it is re-run (the new
    shard atomically replaces the old one)."""
    have = {}
    if not SHARD_DIR.exists():
        return have
    for p in SHARD_DIR.glob("*.json"):
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        rows = d.get("rows", {})
        if isinstance(rows, dict):
            have[d.get("id", p.stem)] = set(rows)
    return have


def _merge(verbose=True):
    all_rows = []
    n_finished = n_partial = n_infeasible = 0
    for p in sorted(SHARD_DIR.glob("*.json")) if SHARD_DIR.exists() else []:
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            n_partial += 1
            continue
        rows = d.get("rows", {})
        expected = d.get("methods_expected", BASE_METHODS)
        if not (isinstance(rows, dict) and set(rows) >= set(expected)):
            n_partial += 1
            continue
        n_finished += 1
        for meth in expected:
            r = rows[meth]
            all_rows.append(r)
            if not r.get("feasible"):
                n_infeasible += 1

    all_rows.sort(key=lambda r: (r["regime"], r["campus"], r["size"],
                                 r["crew_multiplier"], r["arrival_multiplier"],
                                 r["id"], _METHOD_ORDER.get(r["method"], 99)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    if verbose:
        print("Merged %d finished config(s) -> %d rows -> %s"
              % (n_finished, len(all_rows), OUT_CSV))
        if n_partial:
            print("  (%d partial/corrupt shard(s) skipped)" % n_partial)
        print("  infeasible rows: %d" % n_infeasible)
    return {"n_finished": n_finished, "n_rows": len(all_rows),
            "n_infeasible": n_infeasible, "n_partial": n_partial}


def _git_describe():
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=str(_ROOT), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _fmt_hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return "%dh%02dm%02ds" % (h, m, s)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="P4 dynamic-evaluation runner.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap not-yet-done configs processed this run")
    ap.add_argument("--regime", default=None,
                    help="comma-separated subset of %s "
                         "(pmmix selectable only with --with-pmmix)"
                         % ",".join(REGIMES))
    ap.add_argument("--with-pmmix", action="store_true",
                    help="also run the pmmix regime (pm_share_override in "
                         "{0.2,0.5,0.8} x crew {0.6,0.8,1.0}; fresh seeds 50000+)")
    ap.add_argument("--with-storm2", action="store_true",
                    help="also run the storm2 regime: fixed-window (80 bh) "
                         "UTILIZATION sweep u in {0.7,0.9,1.0,1.1,1.3} x campuses "
                         "{5,9,10,12}, crew 1.0, 30 inst/cell, seeds 70000+ "
                         "(n varies per draw; the E2 intensity curve)")
    ap.add_argument("--rl-tag", default="rl",
                    help="method-column prefix for the RL policies "
                         "(default 'rl' -> rl301..rl303)")
    ap.add_argument("--rl-dir", default=str(TRAIN_DIR),
                    help="checkpoint root; loads <dir>/seed<t>/best.pt "
                         "(seeds auto-discovered from seed* dirs; "
                         "default results/p3_train)")
    ap.add_argument("--rl-seeds", default=None,
                    help="comma-separated seed list (default: auto-discover "
                         "seed* checkpoint dirs under --rl-dir)")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "attn"],
                    help="RL policy architecture to load (default mlp; "
                         "attn -> fmwos.policy_attn.AttnDispatchPolicy)")
    ap.add_argument("--storm-arrivals", default=",".join(
                        str(a) for a in STORM_ARRIVAL),
                    help="comma-separated storm arrival_multiplier grid "
                         "(default '1.25,1.5' = current cells, seeds 40000+i; "
                         "NEW values generate fresh instances with seeds "
                         "60000 + cell_index*1000 + i, see module docstring)")
    ap.add_argument("--no-rollcp", action="store_true",
                    help="skip the rollcp2 subsample entirely")
    ap.add_argument("--rollcp-per-cell", type=int, default=DEFAULT_ROLLCP_PER_CELL,
                    help="rollcp2 subsample size per cell (default 8)")
    ap.add_argument("--campus", default=None, help="restrict campus id(s)")
    ap.add_argument("--size", default=None, help="restrict size class(es)")
    ap.add_argument("--out", default=str(OUT_DIR),
                    help="results root for shards/results.csv/meta.json "
                         "(default results/p4_dyneval)")
    ap.add_argument("--merge", action="store_true",
                    help="only (re)build results.csv from existing shards")
    args = ap.parse_args(argv)

    # Reconfigure BEFORE any dir is created or any worker is forked, so the
    # forked workers inherit the tagged method set / checkpoint root / out root.
    rl_seeds = None
    if args.rl_seeds:
        rl_seeds = [int(s) for s in args.rl_seeds.split(",") if s.strip()]
    _configure_rl(args.rl_tag, args.rl_dir, seeds=rl_seeds, arch=args.arch)
    _configure_out(args.out)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        _merge(verbose=True)
        return

    valid_regimes = (REGIMES + ([PMMIX_REGIME] if args.with_pmmix else [])
                     + ([STORM2_REGIME] if args.with_storm2 else []))
    if args.regime:
        regimes = [r.strip() for r in args.regime.split(",")
                   if r.strip() in valid_regimes]
    else:
        regimes = list(REGIMES)          # default: the 3 base regimes (unchanged)
    if args.with_pmmix and PMMIX_REGIME not in regimes:
        regimes.append(PMMIX_REGIME)     # additive: "also run pmmix"
    if args.with_storm2 and STORM2_REGIME not in regimes:
        regimes.append(STORM2_REGIME)    # additive: "also run storm2"
    if not regimes:
        print("no valid regime selected (choices: %s)" % ",".join(valid_regimes))
        return

    try:
        storm_arrivals = sorted({float(a) for a in
                                 args.storm_arrivals.split(",") if a.strip()})
    except ValueError:
        sys.exit("--storm-arrivals must be a comma-separated list of floats "
                 "(got %r)" % args.storm_arrivals)
    if not storm_arrivals or any(a <= 0 for a in storm_arrivals):
        sys.exit("--storm-arrivals needs at least one positive value "
                 "(got %r)" % args.storm_arrivals)

    print("P4 dynamic evaluation")
    print("  regimes         : %s" % ", ".join(regimes))
    print("  rl policies     : %s  (from %s)" % (", ".join(RL_METHODS), RL_DIR))
    print("  out root        : %s" % OUT_DIR)
    if "storm" in regimes:
        new_arr = [a for a in storm_arrivals if a not in STORM_ARRIVAL]
        print("  storm arrivals  : %s%s"
              % (",".join(str(a) for a in storm_arrivals),
                 ("  (new, seeds 60000+cell*1000: %s)"
                  % ",".join(str(a) for a in new_arr)) if new_arr else ""))
        print("  generating storm instances (idempotent) ...", flush=True)
    if PMMIX_REGIME in regimes:
        print("  generating pmmix instances (idempotent) ...", flush=True)
    if STORM2_REGIME in regimes:
        print("  generating storm2 instances (fixed 80 bh window, "
              "utilization sweep; idempotent) ...", flush=True)
    configs = build_targets(regimes, gen_storm=("storm" in regimes),
                            gen_pmmix=(PMMIX_REGIME in regimes),
                            gen_storm2=(STORM2_REGIME in regimes),
                            storm_arrivals=storm_arrivals)

    if args.campus:
        keep = {c.strip() for c in args.campus.split(",")}
        configs = [c for c in configs if str(c["campus"]) in keep]
    if args.size:
        keep = {s.strip() for s in args.size.split(",")}
        # --size applies only to regimes with FIXED size classes; storm2's size
        # is the (variable) realized work-order count, so it is never excluded.
        configs = [c for c in configs
                   if c["regime"] == STORM2_REGIME or str(c["size"]) in keep]

    configs = assign_rollcp(configs, args.rollcp_per_cell, not args.no_rollcp)
    n_rollcp = sum(1 for c in configs if c.get("rollcp"))

    have = _shard_methods()
    pending = [c for c in configs
               if not (have.get(c["id"], set()) >= set(_expected_methods(c)))]
    if args.limit is not None:
        pending = pending[:args.limit]

    by_regime = {}
    for c in configs:
        by_regime[c["regime"]] = by_regime.get(c["regime"], 0) + 1
    print("  configs total   : %d  %s" % (len(configs), dict(by_regime)))
    print("  rollcp2 subset  : %d config(s) (per_cell=%d%s)"
          % (n_rollcp, args.rollcp_per_cell,
             ", DISABLED" if args.no_rollcp else ""))
    n_pending_all = sum(
        1 for c in configs
        if not (have.get(c["id"], set()) >= set(_expected_methods(c))))
    print("  already finished: %d  ->  pending this run: %d"
          % (len(configs) - n_pending_all, len(pending)))
    print("  methods/config  : %d base (%s) + rollcp2 on subset"
          % (len(BASE_METHODS), ", ".join(BASE_METHODS)))
    print("  workers=%d  cpsat_workers=%d  torch_threads=%d  seed=%d"
          % (args.workers, CPSAT_WORKERS, TORCH_THREADS, SEED))

    if not pending:
        print("Nothing pending -- merging existing shards.")
        _merge(verbose=True)
        return

    start_iso = _dt.datetime.now().isoformat(timespec="seconds")
    t_start = time.perf_counter()
    n_infeasible = n_errors = completed = 0
    total = len(pending)
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(_run_one, pending):
            completed += 1
            if not res.get("ok"):
                n_errors += 1
                print("[ERROR] %s (%s): %s"
                      % (res["id"], res["regime"], res.get("error")))
            else:
                for it in res.get("infeasible", []):
                    n_infeasible += 1
                    print("[INFEASIBLE] id=%s method=%s :: %s"
                          % (res["id"], it["method"],
                             " | ".join(it.get("violations", []))))
            if completed % 25 == 0 or completed == total:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / completed * (total - completed)
                print("  progress %d/%d  elapsed %s  eta %s  "
                      "(%d infeasible, %d errors)"
                      % (completed, total, _fmt_hms(elapsed), _fmt_hms(eta),
                         n_infeasible, n_errors))

    elapsed = time.perf_counter() - t_start
    end_iso = _dt.datetime.now().isoformat(timespec="seconds")
    print("Run complete: %d config(s) in %s (%d infeasible, %d errors)."
          % (completed, _fmt_hms(elapsed), n_infeasible, n_errors))

    merged = _merge(verbose=True)
    meta = {
        "experiment": "p4_dyneval", "start_time": start_iso, "end_time": end_iso,
        "elapsed_seconds": round(elapsed, 3), "workers": args.workers,
        "regimes": regimes, "rl_tag": RL_TAG, "rl_dir": str(RL_DIR),
        "rl_arch": RL_ARCH, "rl_seeds": list(RL_SEEDS),
        "rl_methods": RL_METHODS, "with_pmmix": bool(args.with_pmmix),
        "with_storm2": bool(args.with_storm2),
        "storm_arrivals": storm_arrivals,
        "out_dir": str(OUT_DIR), "rollcp_per_cell": args.rollcp_per_cell,
        "no_rollcp": bool(args.no_rollcp), "n_configs": len(configs),
        "n_rollcp": n_rollcp, "n_pending_this_run": total,
        "n_completed_this_run": completed, "n_errors_this_run": n_errors,
        "n_rows": merged["n_rows"], "n_infeasible": merged["n_infeasible"],
        "filters": {"campus": args.campus, "size": args.size, "limit": args.limit},
        "git_describe": _git_describe(),
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print("Wrote %s" % META_JSON)


if __name__ == "__main__":
    main()
