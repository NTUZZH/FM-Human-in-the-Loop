#!/usr/bin/env python
"""P9b: the corpus-anchored recoverable-share cells under the DEPLOYABLE policy.

The abstract quotes two headline numbers side by side. The corpus-anchored one
comes from results/y3_p9/, measured under the published review policy
(``Supervisor`` with ``mechanism="targeted"``), whose consequential test reads
the realized latent shift of the pending queue and which no site can run. The
full-recoverability one now comes from results/y3_w1/, measured under the
deployable policy: a decision-stability test under a split-conformal band
calibrated only on override-derived weak labels. This script re-measures exactly
the three y3_p9 cells under the deployable policy, so the two figures come from
one protocol.

Cells (campus 9, storm2 w80, u = 100 saturation, rho = 0.25, the same ten
held-out instances y3_p9 used, seeds 301-310):

    A'  beta = 0.20, eps = 0.00     the corpus-anchored point (C9's own estimate)
    B'  beta = 0.20, eps = 0.25     the same point under a supervisor that errs
    C'  beta = 0.25, eps = 0.00     the second anchor, so a range can be stated

NOTHING IS RE-DERIVED. The evaluation is scripts/y3_w1_sweep.evaluate_cell,
called verbatim with its module-level cache redirected into results/y3_p9b/cache
so neither published cache is read or written. The statistics are
scripts/y3_realistic_cell.summarize_cell, imported and called, so a P9b number
and a P9 number are computed by the same code.

Arms:
    stability     policy="stability", split_fit=True   the deployable protocol
    targeted_split  policy="targeted", split_fit=True  labelled upper reference
                    under the SAME fold split; a diagnostic that decomposes the
                    difference against y3_p9 into "routing rule" and "fold
                    split". Not quoted in the manuscript.
    targeted_pub  policy="targeted", split_fit=False   the published protocol,
                    run at the headline cell (the \\MzeroGain reproduction gate)
                    and at the three cells (the per-cell bit-exact mirror of
                    results/y3_p9/cache, which is the empirical configuration
                    diff).

REPRODUCTION DISCIPLINE. This pipeline reproduces bit-exactly only with one
numeric thread per process. The caps below are HARD-SET before numpy/torch
import, before y3_w1_sweep gets a chance to ``setdefault`` them to 4; every
worker asserts them and asserts torch.get_num_threads() == 1; parallelism comes
from separate processes. No wall-clock timing is measured or reported: the
machine is shared with three other agents.

Run (eight workers pinned to the ten cores this agent owns):

    cd <repo> && PYTHONPATH=src taskset -c 10-19 \\
        python scripts/y3_p9b_deployable_cells.py --part all --workers 8
"""

from __future__ import annotations

import os

# Single-threaded numeric libs BEFORE numpy / torch import (parent + workers).
# HARD-set, not setdefault: y3_w1_sweep setdefaults these to 4, and four threads
# would move the headline by percentage points and break bit-exactness.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import collections
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

from fmwos.hitl import routing as R                                 # noqa: E402

import y3_w1_sweep as W1                                            # noqa: E402
import y3_realistic_cell as P9                                      # noqa: E402
import y3_p4_m0grid as M0G                                          # noqa: E402

_OUT = os.path.join(_ROOT, "results", "y3_p9b")
_CACHE = os.path.join(_OUT, "cache")
_ASSERT = os.path.join(_OUT, "assert")
_P4CACHE = os.path.join(_ROOT, "results", "y3_p4", "cache")
_P9DIR = os.path.join(_ROOT, "results", "y3_p9")
_P9CACHE = os.path.join(_P9DIR, "cache")
_HARVEST = os.path.join(_ROOT, "results", "y3_p5", "harvest",
                        "primary_multiseed_summary.json")
_COEFFS = os.path.join(_ROOT, "results", "y3_p1", "overlay_coeffs",
                       "F-NL_seed12345.json")
_INST_DIR = os.path.join(_ROOT, "data", "processed", "instances", "c09",
                         "storm2", "w80")

_EXACT_TOL = 0.0            # every reproduction check here is bit-exact

SEEDS = list(P9.SEEDS)                     # 301-310, the primary set of y3_p9
SEEDS_FIVE = list(P9.SEEDS_FIVE)           # 301-305, consistency check only
DECIDERS = list(P9.DECIDERS)               # rule, rule_sup, m0_alone, m0_sup, oracle

# The deployable protocol's band parameters, taken from the W1 headline arm
# (scripts/y3_w1_sweep.ARMS "stability" + _base_task defaults). Not tuned here.
DEPLOY = dict(policy="stability", split_fit=True, cal_frac=0.3, alpha=0.1,
              band_mode="global")
REFSPLIT = dict(policy="targeted", split_fit=True, cal_frac=0.3, alpha=0.1,
                band_mode="global")
PUBPROT = dict(policy="targeted", split_fit=False, cal_frac=0.3, alpha=0.1,
               band_mode="global")

CELL_HEAD = dict(P9.CELL_HEAD)
CELLS = [dict(c) for c in P9.CELLS]

ARM_LABEL = {"stability": "deployable stability routing (split protocol)",
             "targeted_split": "oracle-informed reference, same fold split",
             "targeted_pub": "published protocol (oracle-informed, full aggregate)"}
ARM_KW = {"stability": DEPLOY, "targeted_split": REFSPLIT,
          "targeted_pub": PUBPROT}


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


def cell_task(cell, seed, arm):
    """The RESOLVED configuration handed to y3_w1_sweep.evaluate_cell.

    Every field the W1 harness resolves is written out explicitly, so the
    configuration diff compares resolved values and not command lines. The field
    names are W1's; the y3_p9 counterparts are mapped in part_configdiff.
    """
    t = W1._base_task(campus=cell["campus"], regime=cell["regime"], u=cell["u"],
                      beta=cell["beta"], rho=cell["rho"], eps=cell["eps"],
                      seed=seed, n_train=P9.N_TRAIN, n_probe=P9.N_PROBE,
                      n_eval=P9.N_EVAL, m0_iters=P9.M0_ITERS,
                      **ARM_KW[arm])
    t["arm"] = arm
    t["part"] = "P9b"
    t["scope"] = cell.get("tag", "repro")
    # W1's _base_task already fixes theta, channel, family, master_seed to the
    # same locked constants y3_p9 uses; assert rather than trust.
    assert t["theta"] == P9.THETA and t["channel"] == P9.CHANNEL
    assert t["family"] == P9.FAMILY and t["master_seed"] == P9.MASTER_SEED
    return t


# --------------------------------------------------------------------------- #
# Worker: the W1 evaluate_cell, cache redirected, policy assertion installed    #
# --------------------------------------------------------------------------- #
_SUP_LOG = collections.Counter()


def _install_policy_assertion(expected_policy):
    """Wrap routing.make_supervisor in a pass-through that PROVES which review
    policy object was constructed.

    Both call sites go through the module attribute
    (``y3_w1_sweep`` calls ``R.make_supervisor``; ``routing.run_m0_routed`` calls
    the module global), so one patch covers training and evaluation. The wrapper
    changes no numeric behaviour: it calls the original and returns its result.
    """
    orig = getattr(R, "_p9b_orig_make_supervisor", None) or R.make_supervisor
    R._p9b_orig_make_supervisor = orig

    def checked(policy, overlay, instance, rho, **kw):
        sup = orig(policy, overlay, instance, rho, **kw)
        _SUP_LOG["cls:" + type(sup).__name__] += 1
        _SUP_LOG["policy:" + str(policy)] += 1
        _SUP_LOG["mechanism:" + str(getattr(sup, "mechanism", "?"))] += 1
        if policy == "stability":
            assert isinstance(sup, R.StabilityRoutingSupervisor), (
                "policy='stability' did not build a StabilityRoutingSupervisor "
                "(got %s)" % type(sup).__name__)
            assert sup.mechanism == "stability", (
                "StabilityRoutingSupervisor.mechanism is %r" % sup.mechanism)
            _SUP_LOG["band_map:" + ("set" if sup.band_map is not None
                                    else "cold")] += 1
        else:
            assert not isinstance(sup, R.StabilityRoutingSupervisor), (
                "policy=%r built a StabilityRoutingSupervisor" % policy)
        assert policy == expected_policy, (
            "worker asked for policy %r but the task's policy is %r"
            % (policy, expected_policy))
        return sup

    R.make_supervisor = checked


def worker(task):
    """Call scripts/y3_w1_sweep.evaluate_cell verbatim.

    The only changes are (a) where its cache lives, so the published caches are
    neither read nor written and every reproduction check genuinely recomputes,
    and (b) the pass-through assertion on make_supervisor.
    """
    torch.set_num_threads(1)
    assert torch.get_num_threads() == 1, "worker is not single-threaded"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert os.environ.get(var) == "1", "%s is not 1 in the worker" % var
    os.makedirs(_CACHE, exist_ok=True)
    os.makedirs(_ASSERT, exist_ok=True)
    W1._CACHE = _CACHE

    _SUP_LOG.clear()
    _install_policy_assertion(task["policy"])
    rec = W1.evaluate_cell(task)

    sig = W1._cell_sig(task)
    apath = os.path.join(_ASSERT, "%s.json" % sig)
    if not rec.get("cached"):
        proof = {
            "sig": sig, "arm": task["arm"], "policy": task["policy"],
            "split_fit": task["split_fit"], "beta": task["beta"],
            "eps": task["eps"], "seed": task["seed"],
            "supervisor_constructions": dict(_SUP_LOG),
            "assert_stability_class": task["policy"] == "stability",
        }
        tmp = apath + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(proof, fh, indent=1, sort_keys=True)
        os.replace(tmp, apath)
    else:
        with open(apath) as fh:
            proof = json.load(fh)
    rec["policy_proof"] = proof
    return rec


def run_tasks(tasks, workers, label):
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(worker, t): t for t in tasks}
        for i, f in enumerate(as_completed(fut), 1):
            t = fut[f]
            rec = f.result()
            out.append((t, rec))
            rt = rec["routing"]
            print("  [%s %d/%d] %-14s b=%.2f eps=%.2f seed=%d | rule=%.0f "
                  "m0=%.0f m0+sup=%.0f rule+sup=%.0f oracle=%.0f | gain=%.2f%% "
                  "| rev=%.3f und=%s q=%s %s"
                  % (label, i, len(tasks), t["arm"], t["beta"], t["eps"],
                     t["seed"],
                     np.mean(rec["per"]["rule"]), np.mean(rec["per"]["m0_alone"]),
                     np.mean(rec["per"]["m0_sup"]), np.mean(rec["per"]["rule_sup"]),
                     np.mean(rec["per"]["oracle"]),
                     100.0 * (np.mean(rec["per"]["rule"])
                              - np.mean(rec["per"]["m0_alone"]))
                     / np.mean(rec["per"]["rule"]),
                     rt["m0_sup_revfrac_mean"],
                     ("%.3f" % rt["m0_sup_undetermined"])
                     if rt["m0_sup_undetermined"] == rt["m0_sup_undetermined"]
                     else "n/a",
                     ("%.3f" % rec["band"]["q"]) if rec["band"] else "-",
                     "CACHED" if rec.get("cached") else ""), flush=True)
    out.sort(key=lambda tr: (tr[0]["arm"], tr[0]["beta"], tr[0]["eps"],
                             tr[0]["seed"]))
    return out


# --------------------------------------------------------------------------- #
# Part 1: data-accuracy checks (asserted equal to y3_p9's own record)          #
# --------------------------------------------------------------------------- #
def part_checks():
    files = sorted(glob.glob(os.path.join(_INST_DIR, "c09_storm2_w80_u100_*.json")))
    assert len(files) == 30, "instance pool is %d files, expected 30" % len(files)
    eval_files = files[P9.N_TRAIN + P9.N_PROBE:
                       P9.N_TRAIN + P9.N_PROBE + P9.N_EVAL]
    train_files = files[:P9.N_TRAIN + P9.N_PROBE]
    assert not (set(eval_files) & set(train_files)), "held-out set overlaps train/probe"

    with open(_HARVEST) as fh:
        pub_ids = json.load(fh)["eval_inst_ids"]
    mine_ids = [os.path.basename(p)[:-len(".json")] for p in eval_files]
    assert mine_ids == pub_ids, "held-out ids differ from the published set"

    sha = {os.path.basename(p): _sha256(p) for p in eval_files}
    coeff_sha = _sha256(_COEFFS)

    with open(os.path.join(_P9DIR, "data_checks.json")) as fh:
        p9 = json.load(fh)
    assert p9["eval_inst_ids"] == mine_ids, "y3_p9 used different held-out ids"
    assert p9["eval_file_sha256"] == sha, "held-out instance files changed since y3_p9"
    assert p9["overlay_coeff_sha256"] == coeff_sha, "overlay coefficients changed"
    assert (p9["train_slice"], p9["probe_slice"], p9["eval_slice"]) == \
        ("files[0:16]", "files[16:20]", "files[20:30]"), "y3_p9 used a different split"

    out = {
        "instance_pool_dir": _INST_DIR, "n_files": len(files),
        "train_slice": "files[0:16]", "probe_slice": "files[16:20]",
        "eval_slice": "files[20:30]",
        "eval_inst_ids": mine_ids,
        "eval_inst_ids_match_published": True,
        "eval_file_sha256": sha,
        "eval_file_sha256_match_p9": True,
        "overlay_coeff_file": os.path.relpath(_COEFFS, _ROOT),
        "overlay_coeff_sha256": coeff_sha,
        "overlay_coeff_sha256_match_p9": True,
        "corpus_anchor": p9["corpus_anchor"],
        "threads": {v: os.environ.get(v) for v in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "torch_num_threads_parent": int(torch.get_num_threads()),
    }
    _write_json(os.path.join(_OUT, "data_checks.json"), out)
    print("[checks] pool=%d files; held-out ids, file digests and overlay digest "
          "all identical to results/y3_p9/data_checks.json" % len(files), flush=True)
    return out


# --------------------------------------------------------------------------- #
# Part 2: configuration diff, y3_p9b cell vs its y3_p9 counterpart             #
# --------------------------------------------------------------------------- #
# The fields that describe the review policy. Everything else must be identical.
_POLICY_FIELDS_P9 = ("mech",)
_POLICY_FIELDS_P9B = ("policy", "split_fit", "cal_frac", "alpha", "band_mode")

# W1 task field -> y3_p9 task field, for the fields that must be identical.
_SHARED = ["campus", "regime", "u", "size", "beta", "rho", "eps", "theta",
           "channel", "family", "master_seed", "n_train", "n_probe", "n_eval",
           "m0_iters"]


def part_configdiff():
    """Field-by-field diff of each cell's resolved configuration against the
    resolved configuration y3_p9 used for the same cell.

    Aborts unless every shared field is identical and the difference is exactly
    the review-policy block.
    """
    out = {"shared_fields": _SHARED, "cells": {}, "PASS": True,
           "intended_difference": {
               "y3_p9": {"mech": P9.MECH,
                         "estimator_fit": "whole weak-label aggregate"},
               "y3_p9b": {"policy": DEPLOY["policy"],
                          "split_fit": DEPLOY["split_fit"],
                          "cal_frac": DEPLOY["cal_frac"],
                          "alpha": DEPLOY["alpha"],
                          "band_mode": DEPLOY["band_mode"],
                          "estimator_fit": "proper-training fold only; the "
                                           "calibration fold is never fitted on"},
               "note": "The conformal band the stability test needs must be "
                       "calibrated on examples the estimator has never seen, so "
                       "the fold split is part of the deployable protocol, not "
                       "an independent knob. The targeted_split arm isolates it."}}

    for cell in CELLS:
        p9t = P9.cell_task(cell, seed=None)
        mine = cell_task(cell, seed=None, arm="stability")
        diff = {}
        for k in _SHARED:
            a, b = p9t.get(k, "<absent>"), mine.get(k, "<absent>")
            if a != b:
                diff[k] = {"y3_p9": a, "y3_p9b": b}
        pol_p9 = {k: p9t.get(k) for k in _POLICY_FIELDS_P9}
        pol_mine = {k: mine.get(k) for k in _POLICY_FIELDS_P9B}
        ok = (not diff) and pol_p9 == {"mech": P9.MECH} and pol_mine == DEPLOY
        key = cell["key"]
        out["cells"][key] = {
            "shared_field_diff": diff,
            "review_policy_y3_p9": pol_p9,
            "review_policy_y3_p9b": pol_mine,
            "seeds_y3_p9": SEEDS, "seeds_y3_p9b": SEEDS,
            "eval_inst_ids": ["c09_storm2_w80_u100_%04d" % i for i in range(20, 30)],
            "overlay_coeff_sha256": _sha256(_COEFFS),
            "decider_set": DECIDERS,
            "scoring": "TWT*(w*,d*) full_class_shift, independent validator",
            "PASS": ok}
        out["PASS"] = out["PASS"] and ok
        print("[cfgdiff] %-40s shared fields differ in %s; review policy "
              "%s -> %s  %s"
              % (key, sorted(diff) or "NOTHING", pol_p9, pol_mine,
                 "OK" if ok else "FAIL"), flush=True)
    _write_json(os.path.join(_OUT, "config_diff.json"), out)
    assert out["PASS"], "configuration diff carries an unintended difference"
    return out


# --------------------------------------------------------------------------- #
# Part 3: reproduction gate (\MzeroGain at the published headline cell)        #
# --------------------------------------------------------------------------- #
def _p4_record(cell, seed):
    """The committed results/y3_p4/cache record for (cell, seed), or None."""
    want = dict(campus=cell["campus"], regime=cell["regime"], u=cell["u"],
                beta=cell["beta"], rho=cell["rho"], channel=P9.CHANNEL, seed=seed)
    for p in sorted(glob.glob(os.path.join(_P4CACHE, "*.json"))):
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if all(d.get(k) == v for k, v in want.items()) and d.get("eps", 0.0) == cell["eps"]:
            return d
    return None


def _p9_record(cell, seed):
    """The results/y3_p9/cache record for (cell, seed), or None."""
    want = dict(campus=cell["campus"], regime=cell["regime"], u=cell["u"],
                beta=cell["beta"], rho=cell["rho"], eps=cell["eps"], seed=seed)
    for p in sorted(glob.glob(os.path.join(_P9CACHE, "*.json"))):
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if all(d.get(k) == v for k, v in want.items()):
            return d
    return None


def part_repro(workers):
    with open(_HARVEST) as fh:
        harvest = json.load(fh)
    pub = harvest["cell"]
    for k, v in [("campus", 9), ("regime", "storm2"), ("u", 100), ("beta", 1.0),
                 ("rho", 0.25), ("eps", 0.0), ("theta", 1.0),
                 ("mechanism", P9.MECH), ("family", P9.FAMILY),
                 ("master_seed", P9.MASTER_SEED), ("channel", P9.CHANNEL)]:
        assert pub[k] == v, "headline cell drift on %s: %r != %r" % (k, pub[k], v)
    assert harvest["seeds"] == SEEDS, "published seed set is not 301-310"
    pub_ids = harvest["eval_inst_ids"]

    tasks = [cell_task(CELL_HEAD, s, "targeted_pub") for s in SEEDS]
    recs = run_tasks(tasks, workers, "repro")

    diffs, n_cmp = [], 0
    per_seed = {}
    for _t, r in recs:
        seed = r["seed"]
        assert r["inst_ids"] == pub_ids, "held-out ids differ at seed %d" % seed
        cache = _p4_record(CELL_HEAD, seed)
        assert cache is not None, "no committed y3_p4 cache record for seed %d" % seed
        assert cache["inst_ids"] == pub_ids
        per_seed[seed] = {}
        for k in DECIDERS:
            mine, theirs = list(r["per"][k]), list(cache["per"][k])
            assert len(mine) == len(theirs) == P9.N_EVAL
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
        "gate": "reproduction of \\MzeroGain through "
                "scripts/y3_p9b_deployable_cells.py -> y3_w1_sweep.evaluate_cell",
        "cell": CELL_HEAD["key"], "arm": "targeted_pub", "seeds": SEEDS,
        "eval_inst_ids": pub_ids, "cache_dir_used": _CACHE,
        "published_cache_compared": _P4CACHE,
        "n_per_instance_values_compared": n_cmp, "n_bit_exact": n_exact,
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
    print("[repro] published MzeroGain = %.10f%%  recomputed = %.10f%%  "
          "diff = %.2e pp" % (pub_gain, gain, gain - pub_gain), flush=True)
    print("[repro] PASS = %s" % out["PASS"], flush=True)
    return out


# --------------------------------------------------------------------------- #
# Part 4: per-cell mirror of results/y3_p9/cache (the empirical config diff)   #
# --------------------------------------------------------------------------- #
def part_mirror(workers):
    """Run the three cells through THIS code path with the review policy set back
    to the published one. Every per-instance TWT* must equal results/y3_p9/cache
    to the bit; that is the empirical proof that nothing but the review policy
    differs between the two sets."""
    tasks = [cell_task(c, s, "targeted_pub") for c in CELLS for s in SEEDS]
    recs = run_tasks(tasks, workers, "mirror")
    by = {}
    for t, r in recs:
        by.setdefault((t["beta"], t["eps"]), {})[t["seed"]] = r

    out = {"gate": "per-cell bit-exact mirror of results/y3_p9/cache under the "
                   "published review policy", "cells": {}, "PASS": True}
    for cell in CELLS:
        diffs, n_cmp, per_dec = [], 0, {}
        for seed in SEEDS:
            mine = by[(cell["beta"], cell["eps"])][seed]
            theirs = _p9_record(cell, seed)
            assert theirs is not None, "no y3_p9 cache record for %s seed %d" % (
                cell["key"], seed)
            assert mine["inst_ids"] == theirs["inst_ids"], "held-out ids differ"
            for k in DECIDERS:
                a, b = list(mine["per"][k]), list(theirs["per"][k])
                assert len(a) == len(b) == P9.N_EVAL
                d = [abs(x - y) for x, y in zip(a, b)]
                per_dec.setdefault(k, []).extend(d)
                diffs.extend(d)
                n_cmp += len(d)
        max_abs = max(diffs) if diffs else 0.0
        ok = max_abs <= _EXACT_TOL
        out["cells"][cell["key"]] = {
            "n_compared": n_cmp,
            "n_bit_exact": sum(1 for d in diffs if d == 0.0),
            "max_abs_diff": max_abs,
            "max_abs_diff_per_decider": {k: max(v) for k, v in per_dec.items()},
            "PASS": ok}
        out["PASS"] = out["PASS"] and ok
        print("[mirror] %-40s %d/%d per-instance TWT* bit-exact vs y3_p9, "
              "max|diff| = %r  %s"
              % (cell["key"], out["cells"][cell["key"]]["n_bit_exact"], n_cmp,
                 max_abs, "OK" if ok else "FAIL"), flush=True)
    _write_json(os.path.join(_OUT, "mirror_check.json"), out)
    assert out["PASS"], ("the published protocol does not reproduce y3_p9 through "
                         "this code path; the comparison would not be apples to "
                         "apples")
    return out


# --------------------------------------------------------------------------- #
# Part 5: the three cells under the deployable policy (+ the split reference)  #
# --------------------------------------------------------------------------- #
CSV_COLS = ["cell", "tag", "arm", "policy", "split_fit", "alpha", "band_mode",
            "campus", "u", "beta", "rho", "eps", "seed", "inst_id"] + DECIDERS + [
            "rule_sup_revfrac", "rule_sup_orr", "m0_sup_revfrac", "m0_sup_orr"]


def part_cells(workers):
    store = {}
    all_rows = []
    for arm in ("stability", "targeted_split"):
        tasks = [cell_task(c, s, arm) for c in CELLS for s in SEEDS]
        print("[cells] arm %s -- %s" % (arm, ARM_LABEL[arm]), flush=True)
        recs = run_tasks(tasks, workers, arm)
        by = {}
        for t, r in recs:
            by.setdefault((t["beta"], t["eps"]), []).append((t, r))
        for cell in CELLS:
            rs = sorted(by[(cell["beta"], cell["eps"])], key=lambda tr: tr[1]["seed"])
            assert len(rs) == len(SEEDS)
            store[(arm, cell["key"])] = rs
            for t, r in rs:
                assert r["inst_ids"] == ["c09_storm2_w80_u100_%04d" % i
                                         for i in range(20, 30)]
                for i, iid in enumerate(r["inst_ids"]):
                    row = {"cell": cell["key"], "tag": cell["tag"], "arm": arm,
                           "policy": r["policy"], "split_fit": int(r["split_fit"]),
                           "alpha": r["alpha"], "band_mode": r["band_mode"],
                           "campus": r["campus"], "u": r["u"], "beta": r["beta"],
                           "rho": r["rho"], "eps": r["eps"], "seed": r["seed"],
                           "inst_id": iid}
                    for k in DECIDERS:
                        row[k] = "%.6f" % r["per"][k][i]
                    for k in ("rule_sup_revfrac", "rule_sup_orr",
                              "m0_sup_revfrac", "m0_sup_orr"):
                        v = r.get(k) or []
                        row[k] = ("%.6f" % v[i]) if i < len(v) else ""
                    all_rows.append(row)

    # ---- Cross-cell consistency, exactly the checks y3_p9 made -------------- #
    # RULE never sees a supervisor and ORACLE uses a zero-budget one, so neither
    # depends on eps OR on the review policy. Both must be bit-identical across
    # cells A' and B', and bit-identical to the y3_p9 records at the same beta.
    for arm in ("stability", "targeted_split"):
        a = store[(arm, "c9_storm2_u100_b0.20_r0.25_eps0.00")]
        b = store[(arm, "c9_storm2_u100_b0.20_r0.25_eps0.25")]
        for (_ta, ra), (_tb, rb) in zip(a, b):
            assert ra["seed"] == rb["seed"]
            for k in ("rule", "oracle"):
                assert ra["per"][k] == rb["per"][k], \
                    "%s changed with eps at seed %d in arm %s" % (k, ra["seed"], arm)
    print("[cells] eps-invariance of RULE and ORACLE asserted in both arms",
          flush=True)

    pol_inv = {}
    for cell in CELLS:
        for seed in SEEDS:
            p9r = _p9_record(cell, seed)
            for arm in ("stability", "targeted_split"):
                rec = [r for _t, r in store[(arm, cell["key"])]
                       if r["seed"] == seed][0]
                for k in ("rule", "oracle"):
                    d = max(abs(x - y) for x, y in zip(rec["per"][k], p9r["per"][k]))
                    pol_inv.setdefault("%s:%s" % (arm, k), []).append(d)
                    assert d == 0.0, (
                        "%s at %s seed %d differs from y3_p9 by %r; it must not "
                        "depend on the review policy" % (k, cell["key"], seed, d))
    print("[cells] RULE and ORACLE bit-identical to results/y3_p9/cache in both "
          "arms (max|diff| = %r)"
          % max(max(v) for v in pol_inv.values()), flush=True)

    path = os.path.join(_OUT, "cells.csv")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    os.replace(tmp, path)
    n_expect = 2 * len(CELLS) * len(SEEDS) * P9.N_EVAL
    assert len(all_rows) == n_expect, "wrote %d rows, expected %d" % (
        len(all_rows), n_expect)
    print("[cells] wrote %s (%d rows)" % (path, len(all_rows)), flush=True)
    return store


# --------------------------------------------------------------------------- #
# Part 6: policy proof                                                         #
# --------------------------------------------------------------------------- #
def part_policy_proof(store):
    """Three independent pieces of evidence that the deployable policy ran.

    1. The wrapped constructor asserted the class of every supervisor object.
    2. The record's routing telemetry carries an UNDETERMINED share, which only
       the stability test produces; the published policy leaves the field nan.
    3. A calibrated conformal band exists, with a finite half-width and a
       non-empty calibration fold; the published protocol fits on the whole
       aggregate and returns no band.
    """
    out = {"arms": {}}
    for arm in ("stability", "targeted_split"):
        cls = collections.Counter()
        band_state = collections.Counter()
        rows = []
        for cell in CELLS:
            revfrac, revfrac_rule, und, und_rule, q, ncal = [], [], [], [], [], []
            for _t, r in store[(arm, cell["key"])]:
                for k, v in r["policy_proof"]["supervisor_constructions"].items():
                    cls[k] += v
                    if k.startswith("band_map:"):
                        band_state[k] += v
                rt = r["routing"]
                revfrac.append(rt["m0_sup_revfrac_mean"])
                revfrac_rule.append(rt["rule_sup_revfrac_mean"])
                und.append(rt["m0_sup_undetermined"])
                und_rule.append(rt["rule_sup_undetermined"])
                q.append(r["band"]["q"] if r["band"] else float("nan"))
                ncal.append(r["band"]["n_cal"] if r["band"] else 0)
                assert r["policy"] == ARM_KW[arm]["policy"]
                assert r["run_config"]["policy"] == ARM_KW[arm]["policy"]
                assert bool(r["split_fit"]) is True
            rows.append({
                "cell": cell["key"], "tag": cell["tag"], "beta": cell["beta"],
                "eps": cell["eps"],
                "m0_sup_reviewed_fraction": float(np.mean(revfrac)),
                "rule_sup_reviewed_fraction": float(np.mean(revfrac_rule)),
                "m0_sup_undetermined_share": float(np.mean(und)),
                "rule_sup_undetermined_share": float(np.mean(und_rule)),
                "band_q_mean": float(np.mean(q)),
                "band_n_cal_mean": float(np.mean(ncal))})
        stability_cls = cls.get("cls:StabilityRoutingSupervisor", 0)
        plain_cls = cls.get("cls:Supervisor", 0)
        out["arms"][arm] = {
            "label": ARM_LABEL[arm],
            "expected_policy": ARM_KW[arm]["policy"],
            "supervisor_constructions": dict(cls),
            "n_StabilityRoutingSupervisor": stability_cls,
            "n_plain_Supervisor": plain_cls,
            "band_map_state": dict(band_state),
            "per_cell": rows,
            "PASS": (stability_cls > 0 and plain_cls == 0) if arm == "stability"
                    else (stability_cls == 0 and plain_cls > 0)}
    out["published_policy_reference"] = {
        "note": "results/y3_p9 records carry no routing telemetry at all: they "
                "come from y3_p4_m0grid.evaluate_cell, whose Supervisor has no "
                "stability test and therefore no undetermined share. The "
                "presence of a finite undetermined share is the proof.",
        "p9_m0_sup_reviewed_fraction": {
            c["key"]: _p9_summary()["cells"][c["key"]]["supervisor_budget"]
            ["m0_sup_reviewed_fraction"] for c in CELLS},
        "p9_undetermined_share": "absent (the published policy produces none)"}
    out["PASS"] = all(v["PASS"] for v in out["arms"].values())
    _write_json(os.path.join(_OUT, "policy_proof.json"), out)
    for arm, v in out["arms"].items():
        print("[proof] %-14s supervisors: %s  PASS=%s"
              % (arm, dict(v["supervisor_constructions"]), v["PASS"]), flush=True)
        for r in v["per_cell"]:
            print("         %-40s reviewed=%.4f undetermined=%s band_q=%.4f"
                  % (r["cell"], r["m0_sup_reviewed_fraction"],
                     ("%.4f" % r["m0_sup_undetermined_share"])
                     if r["m0_sup_undetermined_share"] == r["m0_sup_undetermined_share"]
                     else "n/a (none produced)", r["band_q_mean"]), flush=True)
    assert out["PASS"], "the policy proof failed"
    return out


_P9SUM = {}


def _p9_summary():
    if not _P9SUM:
        with open(os.path.join(_P9DIR, "cell_summary.json")) as fh:
            _P9SUM.update(json.load(fh))
    return _P9SUM


def _p9_recs(cell):
    """The ten results/y3_p9/cache records for one cell, as (task, record) pairs
    so y3_realistic_cell's own _stack accepts them."""
    return [(None, _p9_record(cell, s)) for s in SEEDS]


def _policy_contrast(cell, store, decider):
    """Direct paired comparison of the SAME decider under the two review
    policies, on the same ten held-out instances and the same ten seeds.

    Seed-average each policy's per-instance TWT*, then the manuscript's paired
    Wilcoxon. ``pct_lower`` is positive when the DEPLOYABLE policy gives lower
    true weighted tardiness, and ``wtl`` counts the deployable policy as the
    test arm (a win is the deployable policy strictly lower).
    """
    a = P9._stack(store[("stability", cell["key"])], decider, SEEDS).mean(axis=0)
    b = P9._stack(_p9_recs(cell), decider, SEEDS).mean(axis=0)
    c = P9._stack(store[("targeted_split", cell["key"])], decider,
                  SEEDS).mean(axis=0)
    am, bm, cm = float(a.mean()), float(b.mean()), float(c.mean())
    return {"decider": decider,
            "deployable_twt_mean": am, "published_twt_mean": bm,
            "reference_split_twt_mean": cm,
            "pct_lower_than_published": 100.0 * (bm - am) / bm,
            "pct_lower_than_reference_split": 100.0 * (cm - am) / cm,
            "wtl_vs_published": M0G.win_tie_loss(a, b),
            "wilcoxon_p_vs_published": M0G.paired_wilcoxon(a, b),
            "wtl_vs_reference_split": M0G.win_tie_loss(a, c),
            "wilcoxon_p_vs_reference_split": M0G.paired_wilcoxon(a, c),
            "n_instances": int(a.size), "n_seeds": len(SEEDS)}


# --------------------------------------------------------------------------- #
# Part 7: aggregation, side by side with y3_p9                                 #
# --------------------------------------------------------------------------- #
def part_tables(store, repro, mirror, checks, cfg, proof):
    p9 = _p9_summary()
    summ = {
        "config": {"campus": 9, "regime": "storm2", "u": 100, "rho": 0.25,
                   "theta": P9.THETA, "channel": P9.CHANNEL, "family": P9.FAMILY,
                   "master_seed": P9.MASTER_SEED, "n_train": P9.N_TRAIN,
                   "n_probe": P9.N_PROBE, "n_eval": P9.N_EVAL,
                   "m0_iters": P9.M0_ITERS, "seeds_primary": SEEDS,
                   "seeds_secondary": SEEDS_FIVE,
                   "review_policy": "stability (deployable): decision-stability "
                                    "test under a split-conformal band calibrated "
                                    "on override-derived weak labels only",
                   "band": {k: DEPLOY[k] for k in
                            ("split_fit", "cal_frac", "alpha", "band_mode")},
                   "scoring": "TWT*(w*,d*) full_class_shift, independent validator",
                   "contrast_method": "seed-averaged per-instance paired Wilcoxon "
                                      "(pratt), W = test strictly lower TWT*; "
                                      "computed by y3_realistic_cell.summarize_cell",
                   "trained": "per-cell M0 shift estimator only; no policy trained"},
        "gates": {"repro_PASS": repro["PASS"],
                  "published_MzeroGain_value": repro["published_MzeroGain_value"],
                  "recomputed_MzeroGain_value": repro["recomputed_MzeroGain_value"],
                  "difference_pct_points": repro["difference_pct_points"],
                  "n_bit_exact": repro["n_bit_exact"],
                  "n_compared": repro["n_per_instance_values_compared"],
                  "mirror_PASS": mirror["PASS"],
                  "config_diff_PASS": cfg["PASS"],
                  "policy_proof_PASS": proof["PASS"]},
        "corpus_anchor": checks["corpus_anchor"],
        "cells": {}, "cells_five_seed": {}, "cells_reference_split": {},
        "comparison_vs_p9": {}}

    for cell in CELLS:
        recs = store[("stability", cell["key"])]
        summ["cells"][cell["key"]] = P9.summarize_cell(cell, recs, SEEDS)
        summ["cells_five_seed"][cell["key"]] = P9.summarize_cell(cell, recs,
                                                                 SEEDS_FIVE)
        summ["cells_reference_split"][cell["key"]] = P9.summarize_cell(
            cell, store[("targeted_split", cell["key"])], SEEDS)

    for cell in CELLS:
        new = summ["cells"][cell["key"]]
        old = p9["cells"][cell["key"]]
        ref = summ["cells_reference_split"][cell["key"]]
        comp = {"tag": cell["tag"], "beta": cell["beta"], "eps": cell["eps"]}
        for name in ("m0_alone_vs_rule", "m0_sup_vs_rule_sup", "m0_sup_vs_rule",
                     "rule_sup_vs_rule", "oracle_vs_rule"):
            a, b, c = (new["contrasts"][name], old["contrasts"][name],
                       ref["contrasts"][name])
            comp[name] = {
                "deployable_pct": a["pct_vs_comparator"],
                "published_pct": b["pct_vs_comparator"],
                "difference_pct_points": a["pct_vs_comparator"] - b["pct_vs_comparator"],
                "reference_split_pct": c["pct_vs_comparator"],
                "deployable_p": a["wilcoxon_p"], "published_p": b["wilcoxon_p"],
                "deployable_wtl": a["wtl"], "published_wtl": b["wtl"]}
        for k in ("m0_alone", "m0_sup", "rule_sup"):
            comp.setdefault("gap_closed", {})[k] = {
                "deployable_pct": new["gap_closed_pct"][k],
                "published_pct": old["gap_closed_pct"][k],
                "difference_pct_points": new["gap_closed_pct"][k]
                - old["gap_closed_pct"][k],
                "reference_split_pct": ref["gap_closed_pct"][k]}
        comp["ladder"] = {k: {"deployable": new["ladder"][k]["twt_mean"],
                              "published": old["ladder"][k]["twt_mean"],
                              "reference_split": ref["ladder"][k]["twt_mean"]}
                          for k in DECIDERS}
        comp["recovery_quality"] = {
            "deployable_pearson_r": new["recovery_quality"]["pearson_r_mean"],
            "published_pearson_r": old["recovery_quality"]["pearson_r_mean"],
            "deployable_sign_acc": new["recovery_quality"]["sign_acc_nonzero_mean"],
            "published_sign_acc": old["recovery_quality"]["sign_acc_nonzero_mean"],
            "zero_baseline_acc": new["recovery_quality"]["zero_baseline_acc_mean"]}
        comp["routing"] = [r for r in proof["arms"]["stability"]["per_cell"]
                           if r["cell"] == cell["key"]][0]
        # Direct paired test of the two policies on the same instances: the
        # per-cell analogue of W1's price of deployability.
        comp["policy_contrast"] = {d: _policy_contrast(cell, store, d)
                                   for d in ("m0_alone", "m0_sup", "rule_sup")}
        summ["comparison_vs_p9"][cell["key"]] = comp

    # Holm across the three cells, as EXTRA information. results/y3_p9/ reported
    # raw per-cell p-values and did not correct the three cells as a family, so
    # the raw values remain the like-for-like comparison.
    summ["holm_across_the_three_cells"] = {
        "note": "extra information; the like-for-like comparison with "
                "results/y3_p9/ is the raw per-cell p-value",
        "families": {}}
    for name in ("m0_alone_vs_rule", "m0_sup_vs_rule_sup"):
        keys = [c["key"] for c in CELLS]
        raw = [summ["cells"][k]["contrasts"][name]["wilcoxon_p"] for k in keys]
        adj = M0G.holm(raw)
        summ["holm_across_the_three_cells"]["families"][name] = {
            k: {"raw_p": r, "holm_p": float(a)}
            for k, r, a in zip(keys, raw, adj)}

    summ["published_context"] = {
        "headline_cell_published_policy_MzeroGain":
            repro["recomputed_MzeroGain_value"],
        "headline_cell_deployable_policy_RoutedGain": _w1_routed_gain(),
        "source_deployable_headline": "results/y3_w1/head_summary.json:"
                                      "arms.stability.ladder.m0_alone.pct_below_rule"}

    _write_json(os.path.join(_OUT, "cell_summary.json"), summ)
    _write_text(os.path.join(_OUT, "comparison_table.md"), _render_table(summ, p9))
    _write_text(os.path.join(_OUT, "macros_snippet.tex"),
                _render_macros(summ, repro, mirror))
    print("[tables] wrote cell_summary.json, comparison_table.md, "
          "macros_snippet.tex", flush=True)
    return summ


def _w1_grid_price_range():
    """Per-cell range of W1's price of deployability for the layer alone, over
    the eight-cell contention grid (results/y3_w1/grid_summary.json). Positive
    means the deployable policy is worse."""
    p = os.path.join(_ROOT, "results", "y3_w1", "grid_summary.json")
    with open(p) as fh:
        g = json.load(fh)
    v = [c["m0_alone"]["price_pp"] for c in g["price_per_cell"].values()]
    return min(v), max(v)


def _w1_routed_gain():
    p = os.path.join(_ROOT, "results", "y3_w1", "head_summary.json")
    with open(p) as fh:
        s = json.load(fh)
    return s["arms"]["stability"]["ladder"]["m0_alone"]["pct_below_rule"]


def _wt(d, key):
    w = d[key]["deployable_wtl"] if key.endswith("wtl") else None
    return w


def _render_table(summ, p9):
    L = []
    A = L.append
    A("# P9b: the corpus-anchored cells under the DEPLOYABLE review policy\n")
    A("Campus 9, storm2 w80, utilisation 1.00, review budget 0.25, ten held-out")
    A("instances (`c09_storm2_w80_u100_0020` ... `_0029`), seeds 301-310.")
    A("Reductions are in true weighted tardiness TWT*(w*,d*); positive = lower is")
    A("better. Tests are seed-averaged per-instance two-sided paired Wilcoxon")
    A("signed-rank (pratt), W/T/L counted as the test being strictly lower, and")
    A("are computed by `y3_realistic_cell.summarize_cell`, the same function that")
    A("produced `results/y3_p9/cell_summary.json`.\n")
    g = summ["gates"]
    A("Gates. Reproduction: published MzeroGain = %.10f%%, recomputed = %.10f%%, "
      "difference = %.2e pp, %d/%d per-instance values bit-exact against "
      "results/y3_p4/cache. Per-cell mirror against results/y3_p9/cache: %s. "
      "Configuration diff: %s. Policy proof: %s.\n"
      % (g["published_MzeroGain_value"], g["recomputed_MzeroGain_value"],
         g["difference_pct_points"], g["n_bit_exact"], g["n_compared"],
         "PASS" if g["mirror_PASS"] else "FAIL",
         "PASS" if g["config_diff_PASS"] else "FAIL",
         "PASS" if g["policy_proof_PASS"] else "FAIL"))

    A("## Side by side: published review policy vs deployable review policy\n")
    A("`published` = results/y3_p9 (oracle-informed `targeted` routing, whole")
    A("weak-label aggregate). `deployable` = this run (stability routing under a")
    A("split-conformal band, proper-training fold only). Difference is")
    A("deployable minus published, in percentage points.\n")
    A("| Cell | beta | eps | contrast | published | p (W/T/L) | deployable | "
      "p (W/T/L) | difference |")
    A("|---|---|---|---|---|---|---|---|---|")
    names = [("m0_alone_vs_rule", "M0 vs RULE"),
             ("m0_sup_vs_rule_sup", "M0+SUP vs RULE+SUP"),
             ("m0_sup_vs_rule", "M0+SUP vs RULE"),
             ("rule_sup_vs_rule", "RULE+SUP vs RULE"),
             ("oracle_vs_rule", "ORACLE vs RULE")]
    for cell in CELLS:
        c = summ["comparison_vs_p9"][cell["key"]]
        for nm, lab in names:
            d = c[nm]
            A("| %s | %.2f | %.2f | %s | %+.2f%% | %.4f (%d/%d/%d) | %+.2f%% | "
              "%.4f (%d/%d/%d) | %+.2f pp |"
              % (cell["tag"], c["beta"], c["eps"], lab, d["published_pct"],
                 d["published_p"], d["published_wtl"]["W"], d["published_wtl"]["T"],
                 d["published_wtl"]["L"], d["deployable_pct"], d["deployable_p"],
                 d["deployable_wtl"]["W"], d["deployable_wtl"]["T"],
                 d["deployable_wtl"]["L"], d["difference_pct_points"]))

    A("\n## Direct paired test of the two review policies, same instances\n")
    A("The two arms score the same ten held-out instances under the same ten")
    A("seeds, so the two policies can be compared directly rather than through")
    A("their separate contrasts against the rule. `deployable lower` is positive")
    A("when the deployable policy gives lower true weighted tardiness; W/T/L")
    A("counts the deployable policy as the test arm. Ten paired instances put the")
    A("two-sided floor at p = 0.001953.\n")
    A("| Cell | beta | eps | decider | published TWT* | deployable TWT* | "
      "deployable lower | W/T/L | p |")
    A("|---|---|---|---|---|---|---|---|---|")
    for cell in CELLS:
        c = summ["comparison_vs_p9"][cell["key"]]
        for d in ("m0_alone", "m0_sup", "rule_sup"):
            pcx = c["policy_contrast"][d]
            w = pcx["wtl_vs_published"]
            A("| %s | %.2f | %.2f | %s | %.1f | %.1f | %+.2f%% | %d/%d/%d | %.4f |"
              % (cell["tag"], c["beta"], c["eps"], P9.LABEL[d],
                 pcx["published_twt_mean"], pcx["deployable_twt_mean"],
                 pcx["pct_lower_than_published"], w["W"], w["T"], w["L"],
                 pcx["wilcoxon_p_vs_published"]))

    A("\n## Share of the rule-to-reference gap closed\n")
    A("| Cell | beta | eps | published M0 | deployable M0 | difference | "
      "published M0+SUP | deployable M0+SUP | difference |")
    A("|---|---|---|---|---|---|---|---|---|")
    for cell in CELLS:
        c = summ["comparison_vs_p9"][cell["key"]]
        a, b = c["gap_closed"]["m0_alone"], c["gap_closed"]["m0_sup"]
        A("| %s | %.2f | %.2f | %.1f%% | %.1f%% | %+.1f pp | %.1f%% | %.1f%% | "
          "%+.1f pp |" % (cell["tag"], c["beta"], c["eps"], a["published_pct"],
                          a["deployable_pct"], a["difference_pct_points"],
                          b["published_pct"], b["deployable_pct"],
                          b["difference_pct_points"]))

    A("\n## Absolute ladder, mean TWT* over seeds\n")
    A("| Cell | policy | " + " | ".join(P9.LABEL[k] for k in DECIDERS) + " |")
    A("|---|---|" + "---|" * len(DECIDERS))
    for cell in CELLS:
        c = summ["comparison_vs_p9"][cell["key"]]
        for which, lab in (("published", "published"), ("deployable", "deployable"),
                           ("reference_split", "reference (split)")):
            A("| %s (b=%.2f, eps=%.2f) | %s | " % (cell["tag"], c["beta"],
                                                   c["eps"], lab)
              + " | ".join("%.1f" % c["ladder"][k][which] for k in DECIDERS) + " |")

    A("\n## Which policy ran: routing telemetry\n")
    A("The published policy produces no undetermined share at all, because it has")
    A("no stability test; the field's presence is the proof that the deployable")
    A("policy ran.\n")
    A("| Cell | beta | eps | reviewed fraction (M0+SUP) | reviewed fraction "
      "(RULE+SUP) | undetermined share (M0+SUP) | band half-width q | "
      "calibration examples |")
    A("|---|---|---|---|---|---|---|---|")
    for cell in CELLS:
        r = summ["comparison_vs_p9"][cell["key"]]["routing"]
        A("| %s | %.2f | %.2f | %.4f | %.4f | %.4f | %.4f | %.0f |"
          % (cell["tag"], r["beta"], r["eps"], r["m0_sup_reviewed_fraction"],
             r["rule_sup_reviewed_fraction"], r["m0_sup_undetermined_share"],
             r["band_q_mean"], r["band_n_cal_mean"]))
    A("")
    A("For comparison, the published-policy reviewed fractions at the same cells "
      "(results/y3_p9/cell_summary.json:supervisor_budget):")
    A("")
    A("| Cell | reviewed fraction (M0+SUP) | reviewed fraction (RULE+SUP) | "
      "undetermined share |")
    A("|---|---|---|---|")
    for cell in CELLS:
        sb = p9["cells"][cell["key"]]["supervisor_budget"]
        A("| %s | %.4f | %.4f | absent |"
          % (cell["tag"], sb["m0_sup_reviewed_fraction"],
             sb["rule_sup_reviewed_fraction"]))

    A("\n## Decomposition: routing rule vs conformal fold split (diagnostic)\n")
    A("The deployable policy needs a calibration fold the estimator is never")
    A("fitted on, which costs it about 30% of its training labels. The")
    A("`reference (split)` arm is the oracle-informed policy under the SAME fold")
    A("split, so the difference against the published column is the fold split")
    A("and the difference between the deployable and reference columns is the")
    A("routing rule. This is a diagnostic; it is not quoted in the manuscript.\n")
    A("| Cell | contrast | published | reference (split) | deployable | "
      "fold-split effect | routing-rule effect |")
    A("|---|---|---|---|---|---|---|")
    for cell in CELLS:
        c = summ["comparison_vs_p9"][cell["key"]]
        for nm, lab in names[:2]:
            d = c[nm]
            A("| %s | %s | %+.2f%% | %+.2f%% | %+.2f%% | %+.2f pp | %+.2f pp |"
              % (cell["tag"], lab, d["published_pct"], d["reference_split_pct"],
                 d["deployable_pct"],
                 d["reference_split_pct"] - d["published_pct"],
                 d["deployable_pct"] - d["reference_split_pct"]))
    lo, hi = _w1_grid_price_range()
    A("")
    A("The routing-rule effect is the negative of W1's price of deployability")
    A("(oracle-informed minus deployable, positive meaning the deployable policy")
    A("is worse). Here that price is " + ", ".join(
        "%+.2f" % (summ["comparison_vs_p9"][c["key"]]["m0_alone_vs_rule"]
                   ["reference_split_pct"]
                   - summ["comparison_vs_p9"][c["key"]]["m0_alone_vs_rule"]
                   ["deployable_pct"]) for c in CELLS) + " points for the layer")
    A("alone. The eight-cell contention grid of results/y3_w1 reported the same")
    A("quantity between %+.2f and %+.2f points, but every cell in that grid sits"
      % (lo, hi))
    A("at a recoverable share of 0.75 or 1.00. The price of deployability is")
    A("therefore larger at these low recoverable shares than anywhere the grid")
    A("measured it, which the grid could not have shown.")

    A("\n## Recovery quality of the fitted estimator (final DAgger iteration)\n")
    A("| Cell | beta | eps | Pearson r (published) | Pearson r (deployable) | "
      "sign acc (published) | sign acc (deployable) | zero-baseline acc |")
    A("|---|---|---|---|---|---|---|---|")
    for cell in CELLS:
        c = summ["comparison_vs_p9"][cell["key"]]
        q = c["recovery_quality"]
        A("| %s | %.2f | %.2f | %.4f | %.4f | %.4f | %.4f | %.4f |"
          % (cell["tag"], c["beta"], c["eps"], q["published_pearson_r"],
             q["deployable_pearson_r"], q["published_sign_acc"],
             q["deployable_sign_acc"], q["zero_baseline_acc"]))

    A("\n## Five-seed subset (301-305), consistency check only\n")
    A("| Cell | M0 vs RULE | M0+SUP vs RULE+SUP |")
    A("|---|---|---|")
    for cell in CELLS:
        c5 = summ["cells_five_seed"][cell["key"]]
        A("| %s | %+.2f%% | %+.2f%% |"
          % (cell["tag"], c5["contrasts"]["m0_alone_vs_rule"]["pct_vs_comparator"],
             c5["contrasts"]["m0_sup_vs_rule_sup"]["pct_vs_comparator"]))

    A("\n## Where the abstract's two numbers now stand, under one protocol\n")
    pc = summ["published_context"]
    A("| Recoverable share | published review policy | deployable review policy |")
    A("|---|---|---|")
    for cell in CELLS:
        c = summ["comparison_vs_p9"][cell["key"]]
        A("| beta = %.2f, eps = %.2f | %+.2f%% | %+.2f%% |"
          % (c["beta"], c["eps"], c["m0_alone_vs_rule"]["published_pct"],
             c["m0_alone_vs_rule"]["deployable_pct"]))
    A("| beta = 1.00, eps = 0.00 | %+.2f%% | %+.2f%% |"
      % (pc["headline_cell_published_policy_MzeroGain"],
         pc["headline_cell_deployable_policy_RoutedGain"]))
    return "\n".join(L) + "\n"


_TAG2MACRO = {"A": "RealBetaRouted", "B": "RealBetaEpsRouted",
              "C": "RealBetaHiRouted"}


def _fmt_p(p):
    if p != p:
        return "n/a"
    if p < 0.001:
        return "$<$0.001"
    return "%.3f" % p


def _fmt_num(x):
    s = "%.1f" % x
    whole, frac = s.split(".")
    neg = whole.startswith("-")
    if neg:
        whole = whole[1:]
    if len(whole) > 3:
        whole = whole[:-3] + "{,}" + whole[-3:]
    return ("$-$" if neg else "") + whole + "." + frac


def _sgn(v):
    return ("$-$%.1f\\%%" % abs(v)) if v < 0 else ("%.1f\\%%" % v)


def _sgnpp(v):
    return ("$-$%.1f" % abs(v)) if v < 0 else ("$+$%.1f" % v)


def _render_macros(summ, repro, mirror):
    L = []
    A = L.append
    A("% ===========================================================================")
    A("% P9b: THE CORPUS-ANCHORED RECOVERABLE-SHARE CELLS UNDER THE DEPLOYABLE")
    A("% REVIEW POLICY. Generated by scripts/y3_p9b_deployable_cells.py --part all.")
    A("% Campus 9, storm2 w80, utilisation 1.00, review budget 0.25, ten held-out")
    A("% instances, seeds 301-310. Same cells, same instances, same tuned rule,")
    A("% same independent validator and same paired Wilcoxon convention as")
    A("% results/y3_p9/; the ONLY difference is the review policy, which is now")
    A("% the deployable stability routing of results/y3_w1/ (split-conformal band")
    A("% on override-derived weak labels, alpha = 0.1, calibration fraction 0.3).")
    A("%")
    A("% These macros REPLACE the \\RealBeta* family in the abstract and the")
    A("% regime-map table, so the corpus-anchored figure and the")
    A("% full-recoverability figure (\\RoutedGain) come from one protocol. The")
    A("% \\RealBeta* macros remain valid as the published-policy measurement and")
    A("% are the right thing to cite when the two policies are compared.")
    A("%")
    A("% GATES. Reproduction of \\MzeroGain through this code path:")
    A("%%   results/y3_p9b/repro_check.json, %d/%d per-instance TWT* values equal"
      % (repro["n_bit_exact"], repro["n_per_instance_values_compared"]))
    A("%%   to results/y3_p4/cache, max|diff| = %r, recomputed gain %.10f%%."
      % (repro["max_abs_diff_vs_committed_cache"],
         repro["recomputed_MzeroGain_value"]))
    A("% Per-cell mirror: with the review policy set back to the published one,")
    A("%   this code path reproduces results/y3_p9/cache bit-for-bit at all three")
    A("%%   cells (results/y3_p9b/mirror_check.json, max|diff| = %r over %d"
      % (max(v["max_abs_diff"] for v in mirror["cells"].values()),
         sum(v["n_compared"] for v in mirror["cells"].values())))
    A("%   per-instance values), which is the empirical configuration diff.")
    A("% CONVENTION: Gain macros expand WITH a trailing percent sign. A macro whose")
    A("% value is negative carries an explicit LaTeX minus sign. Percentage-POINT")
    A("% differences carry an explicit sign and no percent sign.")
    A("% Macro names contain no digits (LaTeX forbids them); numerals are spelt out.")
    A("% ===========================================================================")
    A("")

    for cell in CELLS:
        c = summ["cells"][cell["key"]]
        cmp_ = summ["comparison_vs_p9"][cell["key"]]
        m = _TAG2MACRO[cell["tag"]]
        k = c["contrasts"]
        src = "results/y3_p9b/cell_summary.json:cells.%s" % cell["key"]
        A("%% ---- cell %s: %s, deployable review policy ----" % (cell["tag"],
                                                                  c["label"]))
        A("%% source %s (10 seeds, 10 held-out instances)" % src)
        A("\\newcommand{\\%sGain}{%s}        %% contrasts.m0_alone_vs_rule."
          "pct_vs_comparator %.4f" % (m, _sgn(k["m0_alone_vs_rule"]["pct_vs_comparator"]),
                                      k["m0_alone_vs_rule"]["pct_vs_comparator"]))
        A("\\newcommand{\\%sGainP}{%s}       %% contrasts.m0_alone_vs_rule."
          "wilcoxon_p %.6f" % (m, _fmt_p(k["m0_alone_vs_rule"]["wilcoxon_p"]),
                               k["m0_alone_vs_rule"]["wilcoxon_p"]))
        A("\\newcommand{\\%sGainWTL}{%d/%d/%d} %% contrasts.m0_alone_vs_rule.wtl"
          % (m, k["m0_alone_vs_rule"]["wtl"]["W"], k["m0_alone_vs_rule"]["wtl"]["T"],
             k["m0_alone_vs_rule"]["wtl"]["L"]))
        A("\\newcommand{\\%sSupGain}{%s}     %% contrasts.m0_sup_vs_rule."
          "pct_vs_comparator %.4f" % (m, _sgn(k["m0_sup_vs_rule"]["pct_vs_comparator"]),
                                      k["m0_sup_vs_rule"]["pct_vs_comparator"]))
        A("\\newcommand{\\%sSupVsSupGain}{%s} %% contrasts.m0_sup_vs_rule_sup."
          "pct_vs_comparator %.4f" % (m, _sgn(k["m0_sup_vs_rule_sup"]["pct_vs_comparator"]),
                                      k["m0_sup_vs_rule_sup"]["pct_vs_comparator"]))
        A("\\newcommand{\\%sSupVsSupP}{%s}   %% contrasts.m0_sup_vs_rule_sup."
          "wilcoxon_p %.6f" % (m, _fmt_p(k["m0_sup_vs_rule_sup"]["wilcoxon_p"]),
                               k["m0_sup_vs_rule_sup"]["wilcoxon_p"]))
        A("\\newcommand{\\%sSupVsSupWTL}{%d/%d/%d} %% contrasts.m0_sup_vs_rule_sup.wtl"
          % (m, k["m0_sup_vs_rule_sup"]["wtl"]["W"],
             k["m0_sup_vs_rule_sup"]["wtl"]["T"],
             k["m0_sup_vs_rule_sup"]["wtl"]["L"]))
        A("\\newcommand{\\%sSupOnly}{%s}     %% contrasts.rule_sup_vs_rule."
          "pct_vs_comparator %.4f" % (m, _sgn(k["rule_sup_vs_rule"]["pct_vs_comparator"]),
                                      k["rule_sup_vs_rule"]["pct_vs_comparator"]))
        A("\\newcommand{\\%sOracle}{%s}      %% contrasts.oracle_vs_rule."
          "pct_vs_comparator %.4f" % (m, _sgn(k["oracle_vs_rule"]["pct_vs_comparator"]),
                                      k["oracle_vs_rule"]["pct_vs_comparator"]))
        A("\\newcommand{\\%sGapClosed}{%s}   %% gap_closed_pct.m0_alone %.4f"
          % (m, _sgn(c["gap_closed_pct"]["m0_alone"]), c["gap_closed_pct"]["m0_alone"]))
        A("\\newcommand{\\%sSupGapClosed}{%s} %% gap_closed_pct.m0_sup %.4f"
          % (m, _sgn(c["gap_closed_pct"]["m0_sup"]), c["gap_closed_pct"]["m0_sup"]))
        A("\\newcommand{\\%sTwtRule}{%s}     %% ladder.rule.twt_mean %.4f"
          % (m, _fmt_num(c["ladder"]["rule"]["twt_mean"]),
             c["ladder"]["rule"]["twt_mean"]))
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
        # Bare-fraction form, so it is a drop-in for \RealBetaEpsRecoverySign.
        A("\\newcommand{\\%sRecoverySign}{%.2f} %% recovery_quality."
          "sign_acc_nonzero_mean %.4f (chance is 0.5); same quantity as "
          "\\%sSignAcc, as a fraction"
          % (m, c["recovery_quality"]["sign_acc_nonzero_mean"],
             c["recovery_quality"]["sign_acc_nonzero_mean"], m))
        A("\\newcommand{\\%sDeltaVsPublished}{%s} %% comparison_vs_p9.%s."
          "m0_alone_vs_rule.difference_pct_points %.4f (percentage points, "
          "deployable minus published)"
          % (m, _sgnpp(cmp_["m0_alone_vs_rule"]["difference_pct_points"]),
             cell["key"], cmp_["m0_alone_vs_rule"]["difference_pct_points"]))
        A("\\newcommand{\\%sSupVsSupDeltaVsPublished}{%s} %% comparison_vs_p9.%s."
          "m0_sup_vs_rule_sup.difference_pct_points %.4f"
          % (m, _sgnpp(cmp_["m0_sup_vs_rule_sup"]["difference_pct_points"]),
             cell["key"], cmp_["m0_sup_vs_rule_sup"]["difference_pct_points"]))
        r = cmp_["routing"]
        A("\\newcommand{\\%sReviewFrac}{%.1f\\%%} %% policy_proof.json arms."
          "stability.per_cell[%s].m0_sup_reviewed_fraction %.4f (budget rho = 0.25)"
          % (m, 100.0 * r["m0_sup_reviewed_fraction"], cell["tag"],
             r["m0_sup_reviewed_fraction"]))
        A("\\newcommand{\\%sUndetermined}{%.1f\\%%} %% policy_proof.json arms."
          "stability.per_cell[%s].m0_sup_undetermined_share %.4f (share of "
          "multi-candidate decisions the stability test cannot certify)"
          % (m, 100.0 * r["m0_sup_undetermined_share"], cell["tag"],
             r["m0_sup_undetermined_share"]))
        A("\\newcommand{\\%sBandQ}{%.2f} %% policy_proof.json arms.stability."
          "per_cell[%s].band_q_mean %.4f (class-shift units)"
          % (m, r["band_q_mean"], cell["tag"], r["band_q_mean"]))
        A("")

    A("% ---- shared protocol facts -------------------------------------------")
    A("\\newcommand{\\RealBetaRoutedSeeds}{10}     % seeds 301-310 "
      "(cell_summary.json:config.seeds_primary)")
    A("\\newcommand{\\RealBetaRoutedNinst}{10}     % held-out instances per cell")
    A("\\newcommand{\\RealBetaRoutedAlpha}{0.1}   % conformal level of the band "
      "(cell_summary.json:config.band.alpha)")
    A("\\newcommand{\\RealBetaRoutedCalFrac}{30\\%}  % calibration fold share "
      "(cell_summary.json:config.band.cal_frac 0.3)")
    A("\\newcommand{\\ReproMzeroGainPnineb}{45.4\\%%}  %% results/y3_p9b/"
      "repro_check.json recomputed_MzeroGain_value %.8f == published"
      % repro["recomputed_MzeroGain_value"])
    A("\\newcommand{\\ReproNExactPnineb}{%d/%d} %% results/y3_p9b/"
      "repro_check.json n_bit_exact / n_per_instance_values_compared"
      % (repro["n_bit_exact"], repro["n_per_instance_values_compared"]))
    A("\\newcommand{\\MirrorNExactPnineb}{%d/%d} %% results/y3_p9b/"
      "mirror_check.json, summed over the three cells: per-instance TWT* equal "
      "to results/y3_p9/cache when the review policy is set back to the "
      "published one"
      % (sum(v["n_bit_exact"] for v in mirror["cells"].values()),
         sum(v["n_compared"] for v in mirror["cells"].values())))
    A("")
    A("% ---- the two abstract figures, now under one protocol ------------------")
    pc = summ["published_context"]
    A("\\newcommand{\\RoutedGainGate}{%.1f\\%%}   %% results/y3_w1/"
      "head_summary.json:arms.stability.ladder.m0_alone.pct_below_rule %.4f "
      "(beta = 1.00, deployable policy; the abstract's full-recoverability figure)"
      % (pc["headline_cell_deployable_policy_RoutedGain"],
         pc["headline_cell_deployable_policy_RoutedGain"]))
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def _load_out(name):
    with open(os.path.join(_OUT, name)) as fh:
        return json.load(fh)


def _rebuild_store():
    store = {}
    for arm in ("stability", "targeted_split"):
        for cell in CELLS:
            recs = []
            for s in SEEDS:
                t = cell_task(cell, s, arm)
                p = os.path.join(_CACHE, "%s.json" % W1._cell_sig(t))
                assert os.path.exists(p), "missing cached record: %s" % p
                with open(p) as fh:
                    r = json.load(fh)
                ap = os.path.join(_ASSERT, "%s.json" % W1._cell_sig(t))
                with open(ap) as fh:
                    r["policy_proof"] = json.load(fh)
                recs.append((t, r))
            store[(arm, cell["key"])] = sorted(recs, key=lambda tr: tr[1]["seed"])
    return store


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["checks", "cfgdiff", "repro", "mirror",
                                       "cells", "tables", "all"], default="all")
    ap.add_argument("--workers", type=int, default=8,
                    help="at most 8: this run owns cores 10-19 only")
    args = ap.parse_args(argv)
    assert args.workers <= 8, "at most eight workers on the ten pinned cores"
    torch.set_num_threads(1)
    os.makedirs(_OUT, exist_ok=True)
    os.makedirs(_CACHE, exist_ok=True)
    os.makedirs(_ASSERT, exist_ok=True)

    checks = cfg = repro = mirror = store = proof = None
    if args.part in ("checks", "all"):
        checks = part_checks()
    if args.part in ("cfgdiff", "all"):
        cfg = part_configdiff()
    if args.part in ("repro", "all"):
        repro = part_repro(args.workers)
        assert repro["PASS"], ("the reproduction gate did not pass; everything "
                               "downstream would be meaningless")
    if args.part in ("mirror", "all"):
        mirror = part_mirror(args.workers)
    if args.part in ("cells", "all"):
        store = part_cells(args.workers)
    if args.part in ("tables", "all"):
        if store is None:
            store = _rebuild_store()
        checks = checks or _load_out("data_checks.json")
        cfg = cfg or _load_out("config_diff.json")
        repro = repro or _load_out("repro_check.json")
        mirror = mirror or _load_out("mirror_check.json")
        assert repro["PASS"] and cfg["PASS"] and mirror["PASS"], "gates did not pass"
        proof = part_policy_proof(store)
        part_tables(store, repro, mirror, checks, cfg, proof)

    print("[y3_p9b] part=%s complete." % args.part, flush=True)


if __name__ == "__main__":
    main()
