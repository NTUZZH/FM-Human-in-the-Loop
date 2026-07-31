#!/usr/bin/env python
"""P9: corpus-anchored recoverable-share cells (Paper Y3).

The manuscript's headline reduction is measured at a recoverable share of
beta = 1.00, where the supervisor's hidden urgency is by construction a
deterministic function of the observable order features the estimator receives.
The companion corpus analysis (results/y3_calib/, from
scripts/y3_calib_overrides.py) puts the real-data analogue of that share at
0.1056 / 0.2038 / 0.2727 on the three headline campuses, median 0.2038, stated as
a lower bound. The published regime map has rows only at beta in {0, 0.5, 1}, so
no measured cell sits near the value the corpus supports. This script measures
three cells that do.

Cells (all at campus 9, storm2 w80, u = 100 saturation, rho = 0.25, targeted
review, the same ten held-out instances the paper uses):

    A  beta = 0.20, eps = 0.00     the corpus-anchored point (C9's own estimate)
    B  beta = 0.20, eps = 0.25     the same point under a supervisor that errs
    C  beta = 0.25, eps = 0.00     a second anchor, so a range can be stated

Deciders: RULE, RULE+SUP, M0 (correction layer), M0+SUP, ORACLE (myopic
full-information reference). Nothing is trained beyond the per-cell M0 shift
estimator, which is the correction layer itself; the end-to-end learner is out
of scope.

The whole evaluation is scripts/y3_p4_m0grid.evaluate_cell, called verbatim, with
its module-level cache redirected into results/y3_p9/cache so the published cache
is neither read nor written. Because the same function computes the reproduction
gate and the new cells, a bit-exact reproduction of \\MzeroGain is evidence about
the code path the new numbers come out of.

REPRODUCTION DISCIPLINE. This pipeline reproduces bit-exactly only with one
numeric thread per process; with more, the estimator refits with a different
floating-point reduction order and the headline moves by percentage points. The
thread caps below are set before numpy/torch import, every worker asserts
torch.get_num_threads() == 1, and parallelism comes from separate processes.
No wall-clock timing is measured or reported: the machine is shared.

Run (four workers pinned to the four cores this agent owns):

    cd <repo> && PYTHONPATH=src taskset -c 20-23 \\
        python scripts/y3_realistic_cell.py --part all --workers 4
"""

from __future__ import annotations

import os

# Single-threaded numeric libs BEFORE numpy / torch import (parent + workers).
# Hard-set, not setdefault: an inherited OMP_NUM_THREADS=8 would silently move
# the headline number by several percentage points.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import csv
import glob
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch                                                        # noqa: E402

import y3_p4_m0grid as M0G                                          # noqa: E402

_OUT = os.path.join(_ROOT, "results", "y3_p9")
_CACHE = os.path.join(_OUT, "cache")
_P4CACHE = os.path.join(_ROOT, "results", "y3_p4", "cache")
_HARVEST = os.path.join(_ROOT, "results", "y3_p5", "harvest",
                        "primary_multiseed_summary.json")
_E3 = os.path.join(_ROOT, "results", "y3_p4", "e3_map_summary.json")
_CALIB_PRED = os.path.join(_ROOT, "results", "y3_calib", "predictability.csv")
_CALIB_DISP = os.path.join(_ROOT, "results", "y3_calib", "campus_disposition.csv")
_CALIB_SUM = os.path.join(_ROOT, "results", "y3_calib", "summary.json")
_COEFFS = os.path.join(_ROOT, "results", "y3_p1", "overlay_coeffs",
                       "F-NL_seed12345.json")
_INST_DIR = os.path.join(_ROOT, "data", "processed", "instances", "c09",
                         "storm2", "w80")

_EXACT_TOL = 0.0            # the reproduction gate is bit-exact, not approximate

# Locked cell constants, copied from the published harness (y3_p4_m0grid).
FAMILY = "F-NL"
MASTER_SEED = 12345
THETA = 1.0
MECH = "targeted"
CHANNEL = "full_class_shift"
N_TRAIN, N_PROBE, N_EVAL = 16, 4, 10
M0_ITERS = 8

DECIDERS = ["rule", "rule_sup", "m0_alone", "m0_sup", "oracle"]
LABEL = {
    "rule":     "Tuned rule (RULE)",
    "rule_sup": "Rule + supervisor (RULE+SUP)",
    "m0_alone": "Correction layer (M0)",
    "m0_sup":   "Correction layer + supervisor (M0+SUP)",
    "oracle":   "Myopic full-information reference (ORACLE)",
}

SEEDS = list(range(301, 311))          # primary, declared in RUN_PLAN.md
SEEDS_FIVE = list(range(301, 306))     # secondary consistency check only

# The published headline cell: the reproduction target, and the baseline every
# new cell's resolved configuration is diffed against.
CELL_HEAD = {"key": "c9_storm2_u100_b1.00_r0.25", "campus": 9, "regime": "storm2",
             "u": 100, "beta": 1.00, "rho": 0.25, "eps": 0.0,
             "label": "Published headline (beta 1.00, eps 0)"}

CELLS = [
    {"key": "c9_storm2_u100_b0.20_r0.25_eps0.00", "campus": 9, "regime": "storm2",
     "u": 100, "beta": 0.20, "rho": 0.25, "eps": 0.00, "tag": "A",
     "label": "Corpus-anchored (beta 0.20, eps 0)"},
    {"key": "c9_storm2_u100_b0.20_r0.25_eps0.25", "campus": 9, "regime": "storm2",
     "u": 100, "beta": 0.20, "rho": 0.25, "eps": 0.25, "tag": "B",
     "label": "Corpus-anchored, supervisor errs (beta 0.20, eps 0.25)"},
    {"key": "c9_storm2_u100_b0.25_r0.25_eps0.00", "campus": 9, "regime": "storm2",
     "u": 100, "beta": 0.25, "rho": 0.25, "eps": 0.00, "tag": "C",
     "label": "Upper corpus anchor (beta 0.25, eps 0)"},
]


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def cell_task(cell, seed):
    """The RESOLVED configuration handed to the published evaluate_cell.

    Every field the published harness resolves is written out explicitly here, so
    the configuration diff compares resolved values and not command lines.
    """
    return {"campus": cell["campus"], "regime": cell["regime"], "u": cell["u"],
            "size": None, "beta": cell["beta"], "rho": cell["rho"],
            "eps": cell["eps"], "theta": THETA, "mech": MECH, "channel": CHANNEL,
            "family": FAMILY, "master_seed": MASTER_SEED, "seed": seed,
            "n_train": N_TRAIN, "n_probe": N_PROBE, "n_eval": N_EVAL,
            "n_eval_full": N_EVAL, "m0_iters": M0_ITERS,
            "part": "P9", "scope": cell.get("tag", "repro")}


# --------------------------------------------------------------------------- #
# Worker: the published evaluate_cell, with the cache redirected               #
# --------------------------------------------------------------------------- #
def worker(task):
    """Call scripts/y3_p4_m0grid.evaluate_cell verbatim.

    The only change is where its cache lives: redirecting the module global keeps
    the published results/y3_p4/cache read-only, and forces the reproduction gate
    to actually recompute instead of reading the published record back.
    """
    torch.set_num_threads(1)
    assert torch.get_num_threads() == 1, "worker is not single-threaded"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ.get(var) == "1", "%s is not 1 in the worker" % var
    os.makedirs(_CACHE, exist_ok=True)
    M0G._CACHE = _CACHE
    return M0G.evaluate_cell(task)


def run_tasks(tasks, workers, label):
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(worker, t): t for t in tasks}
        for i, f in enumerate(as_completed(fut), 1):
            t = fut[f]
            rec = f.result()
            out.append((t, rec))
            print("  [%s %d/%d] b=%.2f eps=%.2f seed=%d  TWT* rule=%.0f m0=%.0f "
                  "m0+sup=%.0f rule+sup=%.0f oracle=%.0f | r=%.3f %s"
                  % (label, i, len(tasks), t["beta"], t["eps"], t["seed"],
                     np.mean(rec["per"]["rule"]), np.mean(rec["per"]["m0_alone"]),
                     np.mean(rec["per"]["m0_sup"]), np.mean(rec["per"]["rule_sup"]),
                     np.mean(rec["per"]["oracle"]),
                     rec["m0_final"]["pearson_r"],
                     "CACHED" if rec.get("cached") else ""), flush=True)
    out.sort(key=lambda tr: tr[0]["seed"])
    return out


# --------------------------------------------------------------------------- #
# Part 1: data-accuracy checks                                                 #
# --------------------------------------------------------------------------- #
def part_checks():
    files = sorted(glob.glob(os.path.join(_INST_DIR, "c09_storm2_w80_u100_*.json")))
    assert len(files) == 30, "instance pool is %d files, expected 30" % len(files)
    eval_files = files[N_TRAIN + N_PROBE:N_TRAIN + N_PROBE + N_EVAL]
    train_files = files[:N_TRAIN + N_PROBE]
    assert not (set(eval_files) & set(train_files)), "held-out set overlaps train/probe"

    with open(_HARVEST) as fh:
        harvest = json.load(fh)
    pub_ids = harvest["eval_inst_ids"]
    mine_ids = [os.path.basename(p)[:-len(".json")] for p in eval_files]
    assert mine_ids == pub_ids, "held-out ids differ from the published set"

    # corpus anchor, re-read rather than retyped
    with open(_CALIB_SUM) as fh:
        calib = json.load(fh)
    prim = tuple(calib["primary_label"])
    headline = []
    with open(_CALIB_DISP) as fh:
        for row in csv.DictReader(fh):
            if row["status"] == "headline":
                headline.append(int(row["campus"]))
    betas = {}
    with open(_CALIB_PRED) as fh:
        for row in csv.DictReader(fh):
            if (row["population"], row["variant"]) == prim and int(row["campus"]) in headline:
                betas[int(row["campus"])] = float(row["beta_hat"])
    vals = sorted(betas.values())
    anchor = {"primary_label": list(prim), "headline_campuses": headline,
              "beta_hat_by_campus": betas, "min": vals[0],
              "median": float(np.median(vals)), "max": vals[-1],
              "campus9_beta_hat": betas.get(9)}

    out = {
        "instance_pool_dir": _INST_DIR,
        "n_files": len(files),
        "train_slice": "files[0:16]", "probe_slice": "files[16:20]",
        "eval_slice": "files[20:30]",
        "eval_inst_ids": mine_ids,
        "eval_inst_ids_match_published": True,
        "eval_file_sha256": {os.path.basename(p): _sha256(p) for p in eval_files},
        "overlay_coeff_file": os.path.relpath(_COEFFS, _ROOT),
        "overlay_coeff_sha256": _sha256(_COEFFS),
        "corpus_anchor": anchor,
        "threads": {v: os.environ.get(v) for v in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "torch_num_threads_parent": int(torch.get_num_threads()),
    }
    _write_json(os.path.join(_OUT, "data_checks.json"), out)
    print("[checks] pool=%d files; held-out ids match the published set; "
          "corpus beta_hat min/med/max = %.4f / %.4f / %.4f (C9 = %.4f)"
          % (len(files), anchor["min"], anchor["median"], anchor["max"],
             anchor["campus9_beta_hat"]), flush=True)
    return out


# --------------------------------------------------------------------------- #
# Part 2: reproduction gate                                                    #
# --------------------------------------------------------------------------- #
def load_p4_cache(cell, seed):
    """The committed results/y3_p4/cache record for (cell, seed), or None."""
    want = dict(campus=cell["campus"], regime=cell["regime"], u=cell["u"],
                beta=cell["beta"], rho=cell["rho"], channel=CHANNEL, seed=seed)
    for p in sorted(glob.glob(os.path.join(_P4CACHE, "*.json"))):
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if all(d.get(k) == v for k, v in want.items()) and d.get("eps", 0.0) == cell["eps"]:
            return d
    return None


def part_repro(workers):
    with open(_HARVEST) as fh:
        harvest = json.load(fh)
    pub = harvest["cell"]
    for k, v in [("campus", 9), ("regime", "storm2"), ("u", 100), ("beta", 1.0),
                 ("rho", 0.25), ("eps", 0.0), ("theta", 1.0), ("mechanism", MECH),
                 ("family", FAMILY), ("master_seed", MASTER_SEED),
                 ("channel", CHANNEL)]:
        assert pub[k] == v, "headline cell drift on %s: %r != %r" % (k, pub[k], v)
    assert harvest["seeds"] == SEEDS, "published seed set is not 301-310"
    pub_ids = harvest["eval_inst_ids"]

    tasks = [cell_task(CELL_HEAD, s) for s in SEEDS]
    recs = run_tasks(tasks, workers, "repro")

    diffs, n_cmp = [], 0
    per_seed = {}
    for _t, r in recs:
        seed = r["seed"]
        assert r["inst_ids"] == pub_ids, "held-out ids differ at seed %d" % seed
        cache = load_p4_cache(CELL_HEAD, seed)
        assert cache is not None, "no committed cache record for seed %d" % seed
        assert cache["inst_ids"] == pub_ids
        per_seed[seed] = {}
        for k in DECIDERS:
            mine, theirs = list(r["per"][k]), list(cache["per"][k])
            assert len(mine) == len(theirs) == N_EVAL
            for a, b in zip(mine, theirs):
                diffs.append(abs(a - b))
                n_cmp += 1
            per_seed[seed][k] = float(np.mean(mine))

    max_abs = max(diffs) if diffs else 0.0
    n_exact = sum(1 for d in diffs if d == 0.0)
    ladder = {k: float(np.mean([per_seed[s][k] for s in SEEDS])) for k in DECIDERS}
    gain = 100.0 * (ladder["rule"] - ladder["m0_alone"]) / ladder["rule"]
    pub_gain = 100.0 * (harvest["ladder"]["rule"]["twt_mean"]
                        - harvest["ladder"]["m0_alone"]["twt_mean"]) \
        / harvest["ladder"]["rule"]["twt_mean"]

    out = {
        "gate": "reproduction of \\MzeroGain through scripts/y3_realistic_cell.py",
        "cell": CELL_HEAD["key"], "seeds": SEEDS, "eval_inst_ids": pub_ids,
        "cache_dir_used": _CACHE,
        "published_cache_compared": _P4CACHE,
        "n_per_instance_values_compared": n_cmp,
        "n_bit_exact": n_exact,
        "max_abs_diff_vs_committed_cache": max_abs,
        "published_MzeroGain_macro": "45.4%",
        "published_MzeroGain_value": pub_gain,
        "recomputed_MzeroGain_value": gain,
        "difference_pct_points": gain - pub_gain,
        "ladder_recomputed": ladder,
        "ladder_published": {k: harvest["ladder"][k]["twt_mean"] for k in DECIDERS},
        "PASS": bool(max_abs <= _EXACT_TOL and abs(gain - pub_gain) <= 1e-9),
    }
    _write_json(os.path.join(_OUT, "repro_check.json"), out)
    print("[repro] %d per-instance TWT* compared, %d bit-exact, max|diff| = %r"
          % (n_cmp, n_exact, max_abs), flush=True)
    print("[repro] published MzeroGain = %.10f%%  recomputed = %.10f%%  diff = %.2e pp"
          % (pub_gain, gain, gain - pub_gain), flush=True)
    print("[repro] PASS = %s" % out["PASS"], flush=True)
    return out


# --------------------------------------------------------------------------- #
# Part 3: configuration diff                                                   #
# --------------------------------------------------------------------------- #
def part_configdiff():
    """Diff each new cell's resolved configuration against the headline's.

    Aborts unless the difference set is exactly {beta} (cells A, C) or
    {beta, eps} (cell B). Seed is excluded: it varies within every cell and the
    seed SET is compared separately.
    """
    def resolved(cell):
        t = cell_task(cell, seed=None)
        t.pop("seed"); t.pop("part"); t.pop("scope")
        t["seeds"] = SEEDS
        t["eval_inst_ids"] = ["c09_storm2_w80_u100_%04d" % i for i in range(20, 30)]
        t["overlay_coeff_sha256"] = _sha256(_COEFFS)
        t["decider_set"] = DECIDERS
        t["scoring"] = "TWT*(w*,d*) full_class_shift, independent validator"
        return t

    base = resolved(CELL_HEAD)
    out = {"baseline_cell": CELL_HEAD["key"], "baseline_resolved_config": base,
           "cells": {}, "PASS": True}
    for cell in CELLS:
        r = resolved(cell)
        keys = sorted(set(base) | set(r))
        diff = {k: {"headline": base.get(k, "<absent>"), "this_cell": r.get(k, "<absent>")}
                for k in keys if base.get(k, "<absent>") != r.get(k, "<absent>")}
        expected = {"beta"} if cell["eps"] == CELL_HEAD["eps"] else {"beta", "eps"}
        ok = set(diff) == expected
        out["cells"][cell["key"]] = {"resolved_config": r, "diff_vs_headline": diff,
                                     "diff_keys": sorted(diff),
                                     "expected_diff_keys": sorted(expected),
                                     "PASS": ok}
        out["PASS"] = out["PASS"] and ok
        print("[cfgdiff] %-38s differs in %s (expected %s) %s"
              % (cell["key"], sorted(diff), sorted(expected),
                 "OK" if ok else "FAIL"), flush=True)
    _write_json(os.path.join(_OUT, "config_diff.json"), out)
    assert out["PASS"], "configuration diff carries an unintended difference"
    return out


# --------------------------------------------------------------------------- #
# Part 4: the three new cells                                                  #
# --------------------------------------------------------------------------- #
CSV_COLS = ["cell", "tag", "campus", "u", "beta", "rho", "eps", "seed",
            "inst_id"] + DECIDERS + [
            "rule_sup_revfrac", "rule_sup_orr", "m0_sup_revfrac", "m0_sup_orr"]


def part_cells(workers):
    all_rows = []
    store = {}
    for cell in CELLS:
        print("[cells] %s" % cell["label"], flush=True)
        tasks = [cell_task(cell, s) for s in SEEDS]
        recs = run_tasks(tasks, workers, cell["tag"])
        assert len(recs) == len(SEEDS)
        store[cell["key"]] = recs
        for t, r in recs:
            assert r["inst_ids"] == ["c09_storm2_w80_u100_%04d" % i for i in range(20, 30)]
            for i, iid in enumerate(r["inst_ids"]):
                row = {"cell": cell["key"], "tag": cell["tag"], "campus": r["campus"],
                       "u": r["u"], "beta": r["beta"], "rho": r["rho"], "eps": r["eps"],
                       "seed": r["seed"], "inst_id": iid}
                for k in DECIDERS:
                    row[k] = "%.6f" % r["per"][k][i]
                for k in ("rule_sup_revfrac", "rule_sup_orr",
                          "m0_sup_revfrac", "m0_sup_orr"):
                    row[k] = "%.6f" % r[k][i]
                all_rows.append(row)

    # Cross-cell consistency: RULE and ORACLE do not depend on eps, so cells A
    # and B (same beta, different eps) must agree bit-for-bit on both.
    a = store["c9_storm2_u100_b0.20_r0.25_eps0.00"]
    b = store["c9_storm2_u100_b0.20_r0.25_eps0.25"]
    for (_ta, ra), (_tb, rb) in zip(a, b):
        assert ra["seed"] == rb["seed"]
        for k in ("rule", "oracle"):
            assert ra["per"][k] == rb["per"][k], \
                "%s changed with eps at seed %d: eps must not touch it" % (k, ra["seed"])
    print("[cells] eps-invariance of RULE and ORACLE asserted across cells A and B",
          flush=True)

    path = os.path.join(_OUT, "cells.csv")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    os.replace(tmp, path)
    n_expect = len(CELLS) * len(SEEDS) * N_EVAL
    assert len(all_rows) == n_expect, "wrote %d rows, expected %d" % (len(all_rows), n_expect)
    print("[cells] wrote %s (%d rows)" % (path, len(all_rows)), flush=True)
    return store


# --------------------------------------------------------------------------- #
# Part 5: aggregation                                                          #
# --------------------------------------------------------------------------- #
def _stack(recs, decider, seeds):
    """(S x n) per-instance matrix for one decider, rows in seed order."""
    by_seed = {r["seed"]: r for _t, r in recs}
    ids = by_seed[seeds[0]]["inst_ids"]
    mat = []
    for s in seeds:
        r = by_seed[s]
        assert r["inst_ids"] == ids
        mat.append(list(r["per"][decider]))
    return np.asarray(mat, float)


def _contrast(recs, test, comp, seeds):
    """Seed-averaged per-instance paired contrast, the manuscript's convention."""
    a = _stack(recs, test, seeds).mean(axis=0)
    b = _stack(recs, comp, seeds).mean(axis=0)
    am, bm = float(a.mean()), float(b.mean())
    return {"test": test, "comparator": comp, "test_mean": am, "comparator_mean": bm,
            "pct_vs_comparator": (100.0 * (bm - am) / bm) if abs(bm) > 1e-12 else 0.0,
            "wtl": M0G.win_tie_loss(a, b),
            "wilcoxon_p": M0G.paired_wilcoxon(a, b),
            "n_instances": int(a.size), "n_seeds": len(seeds)}


CONTRASTS = [("m0_alone", "rule"), ("m0_sup", "rule"), ("m0_sup", "rule_sup"),
             ("rule_sup", "rule"), ("oracle", "rule"), ("m0_sup", "oracle"),
             ("m0_alone", "oracle")]


def summarize_cell(cell, recs, seeds):
    ladder = {}
    for k in DECIDERS:
        mat = _stack(recs, k, seeds)
        sm = mat.mean(axis=1)
        ladder[k] = {"twt_mean": float(sm.mean()),
                     "twt_std_pop": float(sm.std(ddof=0)),
                     "twt_std_sample": float(sm.std(ddof=1)) if len(sm) > 1 else 0.0,
                     "per_seed_mean": [float(x) for x in sm],
                     "n_seeds": len(seeds)}
    rule_m = ladder["rule"]["twt_mean"]
    for k in DECIDERS:
        ladder[k]["pct_below_rule"] = 100.0 * (rule_m - ladder[k]["twt_mean"]) / rule_m

    gap = rule_m - ladder["oracle"]["twt_mean"]
    gap_closed = {k: (100.0 * (rule_m - ladder[k]["twt_mean"]) / gap) if abs(gap) > 1e-12
                  else float("nan") for k in ("m0_alone", "m0_sup", "rule_sup")}

    contrasts = {"%s_vs_%s" % (t, c): _contrast(recs, t, c, seeds)
                 for t, c in CONTRASTS}
    # the ladder percentage and the contrast percentage are the same quantity
    for k in ("m0_alone", "m0_sup"):
        assert abs(contrasts["%s_vs_rule" % k]["pct_vs_comparator"]
                   - ladder[k]["pct_below_rule"]) < 1e-9

    rec_q = {}
    for f in ("pearson_r", "sign_acc_nonzero", "zero_baseline_acc", "override_rate"):
        vals = [r["m0_final"][f] for _t, r in recs if r["seed"] in seeds]
        rec_q[f + "_mean"] = float(np.mean(vals))
        rec_q[f + "_std_sample"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rec_q[f + "_per_seed"] = [float(v) for v in vals]

    def _mean_field(f):
        return float(np.mean([np.mean(r[f]) for _t, r in recs if r["seed"] in seeds]))

    return {"cell": cell["key"], "tag": cell.get("tag"), "label": cell["label"],
            "campus": cell["campus"], "u": cell["u"], "beta": cell["beta"],
            "rho": cell["rho"], "eps": cell["eps"], "seeds": seeds,
            "n_instances": N_EVAL, "ladder": ladder,
            "gap_closed_pct": gap_closed, "contrasts": contrasts,
            "recovery_quality": rec_q,
            "supervisor_budget": {"rule_sup_reviewed_fraction": _mean_field("rule_sup_revfrac"),
                                  "rule_sup_override_rate": _mean_field("rule_sup_orr"),
                                  "m0_sup_reviewed_fraction": _mean_field("m0_sup_revfrac"),
                                  "m0_sup_override_rate": _mean_field("m0_sup_orr")}}


def part_tables(store, repro, checks, cfg):
    with open(_E3) as fh:
        e3 = json.load(fh)
    with open(_HARVEST) as fh:
        harvest = json.load(fh)

    summ = {"config": {"campus": 9, "regime": "storm2", "u": 100, "rho": 0.25,
                       "mechanism": MECH, "theta": THETA, "channel": CHANNEL,
                       "family": FAMILY, "master_seed": MASTER_SEED,
                       "n_train": N_TRAIN, "n_probe": N_PROBE, "n_eval": N_EVAL,
                       "m0_iters": M0_ITERS, "seeds_primary": SEEDS,
                       "seeds_secondary": SEEDS_FIVE,
                       "scoring": "TWT*(w*,d*) full_class_shift, independent validator",
                       "contrast_method": "seed-averaged per-instance paired Wilcoxon "
                                          "(pratt), W = test strictly lower TWT*",
                       "trained": "per-cell M0 shift estimator only; no policy trained"},
            "repro_gate": {"PASS": repro["PASS"],
                           "published_MzeroGain_value": repro["published_MzeroGain_value"],
                           "recomputed_MzeroGain_value": repro["recomputed_MzeroGain_value"],
                           "difference_pct_points": repro["difference_pct_points"],
                           "n_bit_exact": repro["n_bit_exact"],
                           "n_compared": repro["n_per_instance_values_compared"]},
            "config_diff_PASS": cfg["PASS"],
            "corpus_anchor": checks["corpus_anchor"],
            "cells": {}, "cells_five_seed": {},
            "published_context": {
                "headline_b1.00_m0_alone_pct_below_rule":
                    100.0 * (harvest["ladder"]["rule"]["twt_mean"]
                             - harvest["ladder"]["m0_alone"]["twt_mean"])
                    / harvest["ladder"]["rule"]["twt_mean"],
                "e3_c9_u100_b0.00_m0_over_rule_pct": e3["cells"]["c9_u100_b0.00"]["m0_over_rule_pct"],
                "e3_c9_u100_b0.50_m0_over_rule_pct": e3["cells"]["c9_u100_b0.50"]["m0_over_rule_pct"],
                "e3_c9_u100_b1.00_m0_over_rule_pct": e3["cells"]["c9_u100_b1.00"]["m0_over_rule_pct"],
                "e3_n_seeds": 3}}

    for cell in CELLS:
        recs = store[cell["key"]]
        summ["cells"][cell["key"]] = summarize_cell(cell, recs, SEEDS)
        summ["cells_five_seed"][cell["key"]] = summarize_cell(cell, recs, SEEDS_FIVE)

    _write_json(os.path.join(_OUT, "cell_summary.json"), summ)
    _write_text(os.path.join(_OUT, "results_table.md"), _render_table(summ))
    _write_text(os.path.join(_OUT, "macros_snippet.tex"), _render_macros(summ, repro))
    print("[tables] wrote cell_summary.json, results_table.md, macros_snippet.tex",
          flush=True)
    return summ


def _p(x):
    return "%.4f" % x


def _render_table(summ):
    L = []
    A = L.append
    A("# P9 results: corpus-anchored recoverable-share cells\n")
    A("Campus 9, storm2 w80, utilisation 1.00, review budget 0.25, targeted review,")
    A("ten held-out instances (`c09_storm2_w80_u100_0020` ... `_0029`), seeds 301-310.")
    A("Reductions are in true weighted tardiness TWT*(w*,d*); positive = lower is better.")
    A("Tests are seed-averaged per-instance two-sided paired Wilcoxon signed-rank")
    A("(pratt), W/T/L counted as the test being strictly lower.\n")
    A("Reproduction gate: published MzeroGain = %.10f%%, recomputed = %.10f%%, "
      "difference = %.2e pp, %d/%d per-instance values bit-exact.\n"
      % (summ["repro_gate"]["published_MzeroGain_value"],
         summ["repro_gate"]["recomputed_MzeroGain_value"],
         summ["repro_gate"]["difference_pct_points"],
         summ["repro_gate"]["n_bit_exact"], summ["repro_gate"]["n_compared"]))

    A("## Main table (10 seeds)\n")
    A("| Cell | beta | eps | M0 vs RULE | p (W/T/L) | M0+SUP vs RULE | p (W/T/L) | "
      "M0+SUP vs RULE+SUP | p (W/T/L) | RULE+SUP vs RULE | ORACLE vs RULE | "
      "gap closed M0 | gap closed M0+SUP |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in CELLS:
        c = summ["cells"][cell["key"]]
        k = c["contrasts"]
        def wt(n):
            w = k[n]["wtl"]
            return "%.4f (%d/%d/%d)" % (k[n]["wilcoxon_p"], w["W"], w["T"], w["L"])
        A("| %s | %.2f | %.2f | %+.2f%% | %s | %+.2f%% | %s | %+.2f%% | %s | %+.2f%% | "
          "%+.2f%% | %.1f%% | %.1f%% |"
          % (cell["tag"], c["beta"], c["eps"],
             k["m0_alone_vs_rule"]["pct_vs_comparator"], wt("m0_alone_vs_rule"),
             k["m0_sup_vs_rule"]["pct_vs_comparator"], wt("m0_sup_vs_rule"),
             k["m0_sup_vs_rule_sup"]["pct_vs_comparator"], wt("m0_sup_vs_rule_sup"),
             k["rule_sup_vs_rule"]["pct_vs_comparator"],
             k["oracle_vs_rule"]["pct_vs_comparator"],
             c["gap_closed_pct"]["m0_alone"], c["gap_closed_pct"]["m0_sup"]))

    A("\n## Absolute ladder, mean TWT* over seeds (10 seeds)\n")
    A("| Cell | " + " | ".join(LABEL[k] for k in DECIDERS) + " |")
    A("|---|" + "---|" * len(DECIDERS))
    for cell in CELLS:
        c = summ["cells"][cell["key"]]
        A("| %s (b=%.2f, eps=%.2f) | " % (cell["tag"], c["beta"], c["eps"])
          + " | ".join("%.1f" % c["ladder"][k]["twt_mean"] for k in DECIDERS) + " |")

    A("\n## Recovery quality of the fitted estimator (final DAgger iteration)\n")
    A("| Cell | beta | eps | Pearson r | sign accuracy | zero-baseline accuracy | "
      "training override rate |")
    A("|---|---|---|---|---|---|---|")
    for cell in CELLS:
        c = summ["cells"][cell["key"]]
        q = c["recovery_quality"]
        A("| %s | %.2f | %.2f | %.4f | %.4f | %.4f | %.4f |"
          % (cell["tag"], c["beta"], c["eps"], q["pearson_r_mean"],
             q["sign_acc_nonzero_mean"], q["zero_baseline_acc_mean"],
             q["override_rate_mean"]))

    A("\n## Five-seed subset (301-305), consistency check only\n")
    A("| Cell | M0 vs RULE | M0+SUP vs RULE+SUP |")
    A("|---|---|---|")
    for cell in CELLS:
        c5 = summ["cells_five_seed"][cell["key"]]
        A("| %s | %+.2f%% | %+.2f%% |"
          % (cell["tag"], c5["contrasts"]["m0_alone_vs_rule"]["pct_vs_comparator"],
             c5["contrasts"]["m0_sup_vs_rule_sup"]["pct_vs_comparator"]))

    A("\n## Published neighbours on the same campus and load (context)\n")
    p = summ["published_context"]
    A("| beta | M0 alone vs RULE | seeds | source |")
    A("|---|---|---|---|")
    A("| 0.00 | %+.2f%% | 3 | results/y3_p4/e3_map_summary.json |" % p["e3_c9_u100_b0.00_m0_over_rule_pct"])
    for cell in CELLS:
        c = summ["cells"][cell["key"]]
        A("| %.2f (eps %.2f) | %+.2f%% | 10 | this run |"
          % (c["beta"], c["eps"], c["contrasts"]["m0_alone_vs_rule"]["pct_vs_comparator"]))
    A("| 0.50 | %+.2f%% | 3 | results/y3_p4/e3_map_summary.json |" % p["e3_c9_u100_b0.50_m0_over_rule_pct"])
    A("| 1.00 | %+.2f%% | 10 | results/y3_p5/harvest/primary_multiseed_summary.json |"
      % p["headline_b1.00_m0_alone_pct_below_rule"])
    return "\n".join(L) + "\n"


_TAG2MACRO = {"A": "RealBeta", "B": "RealBetaEps", "C": "RealBetaHi"}


def _render_macros(summ, repro):
    L = []
    A = L.append
    A("% ===========================================================================")
    A("% P9 CORPUS-ANCHORED RECOVERABLE-SHARE CELLS. Generated by")
    A("% scripts/y3_realistic_cell.py --part all. Campus 9, storm2 w80,")
    A("% utilisation 1.00, review budget 0.25, targeted review, ten held-out")
    A("% instances, seeds 301-310. Every number is a fresh run of the published")
    A("% evaluate_cell path (scripts/y3_p4_m0grid.evaluate_cell) with only beta")
    A("% and epsilon changed, gated on a bit-exact reproduction of \\MzeroGain")
    A("%% (results/y3_p9/repro_check.json: %d/%d per-instance TWT* values equal to"
      % (repro["n_bit_exact"], repro["n_per_instance_values_compared"]))
    A("%% results/y3_p4/cache, max|diff| = %r, recomputed gain %.10f%%)."
      % (repro["max_abs_diff_vs_committed_cache"], repro["recomputed_MzeroGain_value"]))
    A("% CONVENTION: Gain macros expand WITH a trailing percent sign. A macro whose")
    A("% value is negative carries an explicit LaTeX minus sign.")
    A("% ===========================================================================")
    A("")
    A("% ---- corpus anchor (results/y3_calib/predictability.csv, population=cm,")
    A("%      variant=trade_w30, campuses with status=headline) ------------------")
    an = summ["corpus_anchor"]
    A("\\newcommand{\\betareal}{%.2f}       %% corpus-anchored recoverable share; "
      "predictability.csv beta_hat median over headline campuses = %.4f"
      % (0.20, an["median"]))
    A("\\newcommand{\\betarealhi}{%.2f}     %% upper corpus anchor actually measured "
      "here; headline-campus beta_hat max = %.4f" % (0.25, an["max"]))
    bc = an["beta_hat_by_campus"]
    c9b = float(bc.get(9, bc.get("9")))     # JSON round-trip turns the key into a string
    A("\\newcommand{\\betarealCnine}{%.2f}  %% campus 9's own estimate; "
      "predictability.csv campus=9 beta_hat = %.4f" % (0.20, c9b))
    A("\\newcommand{\\epsreal}{0.25}        % supervisor error rate at the realistic "
      "operating point (half missed overrides, half random overrides)")
    A("")

    for cell in CELLS:
        c = summ["cells"][cell["key"]]
        m = _TAG2MACRO[cell["tag"]]
        k = c["contrasts"]
        src = "results/y3_p9/cell_summary.json:cells.%s" % cell["key"]
        A("%% ---- cell %s: %s -------------------------------" % (cell["tag"], c["label"]))
        A("%% source %s (10 seeds, 10 held-out instances)" % src)

        def sgn(v):
            return ("$-$%.1f\\%%" % abs(v)) if v < 0 else ("%.1f\\%%" % v)

        A("\\newcommand{\\%sGain}{%s}        %% contrasts.m0_alone_vs_rule."
          "pct_vs_comparator %.4f" % (m, sgn(k["m0_alone_vs_rule"]["pct_vs_comparator"]),
                                      k["m0_alone_vs_rule"]["pct_vs_comparator"]))
        A("\\newcommand{\\%sGainP}{%s}       %% contrasts.m0_alone_vs_rule.wilcoxon_p %.6f"
          % (m, _fmt_p(k["m0_alone_vs_rule"]["wilcoxon_p"]), k["m0_alone_vs_rule"]["wilcoxon_p"]))
        A("\\newcommand{\\%sGainWTL}{%d/%d/%d} %% contrasts.m0_alone_vs_rule.wtl"
          % (m, k["m0_alone_vs_rule"]["wtl"]["W"], k["m0_alone_vs_rule"]["wtl"]["T"],
             k["m0_alone_vs_rule"]["wtl"]["L"]))
        A("\\newcommand{\\%sSupGain}{%s}     %% contrasts.m0_sup_vs_rule."
          "pct_vs_comparator %.4f" % (m, sgn(k["m0_sup_vs_rule"]["pct_vs_comparator"]),
                                      k["m0_sup_vs_rule"]["pct_vs_comparator"]))
        A("\\newcommand{\\%sSupVsSupGain}{%s} %% contrasts.m0_sup_vs_rule_sup."
          "pct_vs_comparator %.4f" % (m, sgn(k["m0_sup_vs_rule_sup"]["pct_vs_comparator"]),
                                      k["m0_sup_vs_rule_sup"]["pct_vs_comparator"]))
        A("\\newcommand{\\%sSupVsSupP}{%s}   %% contrasts.m0_sup_vs_rule_sup.wilcoxon_p %.6f"
          % (m, _fmt_p(k["m0_sup_vs_rule_sup"]["wilcoxon_p"]),
             k["m0_sup_vs_rule_sup"]["wilcoxon_p"]))
        A("\\newcommand{\\%sSupVsSupWTL}{%d/%d/%d} %% contrasts.m0_sup_vs_rule_sup.wtl"
          % (m, k["m0_sup_vs_rule_sup"]["wtl"]["W"], k["m0_sup_vs_rule_sup"]["wtl"]["T"],
             k["m0_sup_vs_rule_sup"]["wtl"]["L"]))
        A("\\newcommand{\\%sSupOnly}{%s}     %% contrasts.rule_sup_vs_rule."
          "pct_vs_comparator %.4f" % (m, sgn(k["rule_sup_vs_rule"]["pct_vs_comparator"]),
                                      k["rule_sup_vs_rule"]["pct_vs_comparator"]))
        A("\\newcommand{\\%sOracle}{%s}      %% contrasts.oracle_vs_rule."
          "pct_vs_comparator %.4f" % (m, sgn(k["oracle_vs_rule"]["pct_vs_comparator"]),
                                      k["oracle_vs_rule"]["pct_vs_comparator"]))
        A("\\newcommand{\\%sGapClosed}{%s}   %% gap_closed_pct.m0_alone %.4f"
          % (m, sgn(c["gap_closed_pct"]["m0_alone"]), c["gap_closed_pct"]["m0_alone"]))
        A("\\newcommand{\\%sSupGapClosed}{%s} %% gap_closed_pct.m0_sup %.4f"
          % (m, sgn(c["gap_closed_pct"]["m0_sup"]), c["gap_closed_pct"]["m0_sup"]))
        A("\\newcommand{\\%sTwtRule}{%s}     %% ladder.rule.twt_mean %.4f"
          % (m, _fmt_num(c["ladder"]["rule"]["twt_mean"]), c["ladder"]["rule"]["twt_mean"]))
        A("\\newcommand{\\%sTwtMzero}{%s}    %% ladder.m0_alone.twt_mean %.4f"
          % (m, _fmt_num(c["ladder"]["m0_alone"]["twt_mean"]),
             c["ladder"]["m0_alone"]["twt_mean"]))
        A("\\newcommand{\\%sRecoveryR}{%.2f} %% recovery_quality.pearson_r_mean %.4f"
          % (m, c["recovery_quality"]["pearson_r_mean"],
             c["recovery_quality"]["pearson_r_mean"]))
        A("\\newcommand{\\%sSignAcc}{%.1f\\%%} %% recovery_quality."
          "sign_acc_nonzero_mean %.4f (zero-baseline %.4f)"
          % (m, 100.0 * c["recovery_quality"]["sign_acc_nonzero_mean"],
             c["recovery_quality"]["sign_acc_nonzero_mean"],
             c["recovery_quality"]["zero_baseline_acc_mean"]))
        A("")

    A("% ---- shared protocol facts -------------------------------------------")
    A("\\newcommand{\\RealBetaSeeds}{10}      % seeds 301-310 (cell_summary.json:"
      "config.seeds_primary)")
    A("\\newcommand{\\RealBetaNinst}{10}      % held-out instances per cell")
    A("\\newcommand{\\ReproMzeroGainPnine}{45.4\\%%}  %% repro_check.json "
      "recomputed_MzeroGain_value %.8f == published" % repro["recomputed_MzeroGain_value"])
    A("\\newcommand{\\ReproNExactPnine}{%d/%d} %% repro_check.json n_bit_exact / "
      "n_per_instance_values_compared"
      % (repro["n_bit_exact"], repro["n_per_instance_values_compared"]))
    return "\n".join(L) + "\n"


def _fmt_p(p):
    if p != p:
        return "n/a"
    if p < 0.001:
        return "$<$0.001"
    return "%.3f" % p


def _fmt_num(x):
    s = "%.1f" % x
    whole, frac = s.split(".")
    if len(whole) > 3:
        whole = whole[:-3] + "{,}" + whole[-3:]
    return whole + "." + frac


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["checks", "repro", "cfgdiff", "cells",
                                       "tables", "all"], default="all")
    ap.add_argument("--workers", type=int, default=4,
                    help="at most 4: this run owns cores 20-23 only")
    args = ap.parse_args(argv)
    assert args.workers <= 4, "at most four workers (cores 20-23)"
    torch.set_num_threads(1)
    os.makedirs(_OUT, exist_ok=True)
    os.makedirs(_CACHE, exist_ok=True)

    checks = repro = cfg = store = None
    if args.part in ("checks", "all"):
        checks = part_checks()
    if args.part in ("cfgdiff", "all"):
        cfg = part_configdiff()
    if args.part in ("repro", "all"):
        repro = part_repro(args.workers)
        assert repro["PASS"], ("the reproduction gate did not pass; everything "
                               "downstream would be meaningless")
    if args.part in ("cells", "all"):
        store = part_cells(args.workers)
    if args.part in ("tables", "all"):
        if store is None:                       # rebuild from the private cache
            store = {}
            for cell in CELLS:
                recs = []
                for s in SEEDS:
                    t = cell_task(cell, s)
                    p = os.path.join(_CACHE, "%s.json" % M0G._cell_sig(t))
                    assert os.path.exists(p), "missing cached record: %s" % p
                    with open(p) as fh:
                        recs.append((t, json.load(fh)))
                store[cell["key"]] = sorted(recs, key=lambda tr: tr[1]["seed"])
        def _load_out(name):
            with open(os.path.join(_OUT, name)) as fh:
                return json.load(fh)

        if checks is None:
            checks = _load_out("data_checks.json")
        if repro is None:
            repro = _load_out("repro_check.json")
        if cfg is None:
            cfg = _load_out("config_diff.json")
        assert repro["PASS"] and cfg["PASS"], "gates did not pass"
        part_tables(store, repro, checks, cfg)

    print("[y3_p9] part=%s complete." % args.part, flush=True)


if __name__ == "__main__":
    main()
