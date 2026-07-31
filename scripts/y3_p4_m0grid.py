#!/usr/bin/env python
"""Y3 Phase-4 CHEAP-decider grid (no PPO): M0 headline gate + E3 regime map +
boundary confirmations.

This is the NO-RL companion to scripts/y3_p3_eval.py. It scores ONLY the cheap
deciders that need no policy training:

    RULE          ATC on the recorded fields (the deployed rule).
    RULE+SUP      the rule with the in-loop supervisor (status quo).
    M0            augmented ATC; its estimator corrects BOTH the weight and the
                  deadline, trained on the RULE+SUP override log (no RL).
    M0+SUP        M0 with the in-loop supervisor.
    ORACLE-GREEDY myopic ATC on the TRUE (w*, d*) at every decision (skyline).

Every decider path, the M0 per-cell estimator training, and the TWT*(w*,d*)
scoring are REUSED verbatim from scripts/y3_p3_eval.py / the hitl package
(overlay channel=full_class_shift; the supervisor auto-injects d* as its due).
Nothing here trains or loads a PPO policy.

Config (locked, matches the P3 pilot): channel=full_class_shift, family=F-NL,
master_seed=12345, eps=0, mechanism=targeted, theta=1.0. Held-out eval =
files[n_train+n_probe : +n_eval] (default files[20:30]); disjoint from the M0
train/probe pools files[:20].

Parts
-----
A  M0-gate significance.  PRIMARY: campus 9, storm2 u{90,100}, beta{0.75,1.0},
   rho{0.25,0.5}, seeds 301-310 (8 cells x 10 seeds). SIGN CHECK: campus 10,
   same cells, seeds 301-303.  -> results/y3_p4/m0_gate.csv (+ _summary.json).
B  E3 regime map.  campus 9 AND 10, storm2 u{70,90,100,110,130}, beta{0,0.5,1.0},
   rho 0.25, seeds 301-303 (descriptive).  -> results/y3_p4/e3_map.csv (+ _summary).
C  Boundary cells.  (i) c12 u100 + c5 u100 storm2, beta 1.0 rho 0.25, 3 seeds;
   (ii) c9/c10 replay size 150/400 (slack capacity), beta 1.0 rho 0.25, 3 seeds.
   -> results/y3_p4/boundary.csv.

Every cell-seed is cached under results/y3_p4/cache/ so partial harvests resume
and cells shared between parts are computed once.

Run (in a y3_-prefixed tmux, CPU only, 8 workers, OMP=1/worker, niced):
    PYTHONPATH=src OMP_NUM_THREADS=1 nice -n 15 \
        python scripts/y3_p4_m0grid.py --part all --workers 8
"""

from __future__ import annotations

import os

# Single-threaded numeric libs BEFORE numpy/torch import (parent + forked workers).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import glob
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch                                                    # noqa: E402

from fmwos.env import DispatchEnv                                # noqa: E402
from fmwos.hitl import deciders as dec                           # noqa: E402
from fmwos.hitl import overlay as ov                             # noqa: E402
from fmwos.hitl.supervisor import Supervisor                     # noqa: E402
from fmwos.hitl import augmented_rule as AR                      # noqa: E402
from fmwos.hitl import true_objective as TO                      # noqa: E402

try:
    from scipy.stats import wilcoxon as _wilcoxon
    _HAVE_SCIPY = True
except Exception:                                               # pragma: no cover
    _HAVE_SCIPY = False

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_p4")
_CACHE = os.path.join(_OUT, "cache")
_TOL = 1e-9
HORIZON_BH = 80.0                       # storm2 fixed window (util denominator)

# Locked cell constants (match the P3 pilot / the locked overlay).
FAMILY = "F-NL"
MASTER_SEED = 12345
EPS = 0.0
THETA = 1.0
MECH = "targeted"
CHANNEL = "full_class_shift"

ALONE = ["rule", "m0_alone", "oracle"]
INLOOP = ["rule_sup", "m0_sup"]
DECIDERS = ALONE + INLOOP


# --------------------------------------------------------------------------- #
# Instance pools                                                              #
# --------------------------------------------------------------------------- #
def locate_files(campus, regime, u=None, size=None, w="w80"):
    cdir = "c%02d" % campus
    if regime == "storm2":
        pat = os.path.join(_INST, cdir, "storm2", w,
                           "%s_storm2_%s_u%d_*.json" % (cdir, w, u))
    elif regime == "replay":
        pat = os.path.join(_INST, cdir, "replay", str(size),
                           "%s_replay_%d_*.json" % (cdir, size))
    else:
        raise ValueError("regime %r not supported" % regime)
    return sorted(glob.glob(pat))


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def _utilization(inst):
    """Pooled + worst-trade utilization over the fixed storm2 window (copied from
    scripts/y3_cont_storm2-util.py). util_g = sum p_bh in trade g / (k_g * H)."""
    kg = defaultdict(int)
    for t in inst["technicians"]:
        kg[t["trade"]] += 1
    pg = defaultdict(float)
    for w in inst["work_orders"]:
        pg[w["trade"]] += float(w["p_bh"])
    total_crew = sum(kg.values())
    util_pool = (sum(pg.values()) / (total_crew * HORIZON_BH)) if total_crew else 0.0
    worst = 0.0
    for g, work in pg.items():
        k = kg.get(g, 0)
        if k <= 0:
            continue
        u = work / (k * HORIZON_BH)
        worst = max(worst, u)
    return util_pool, worst


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
def paired_wilcoxon(a, b):
    """Two-sided paired Wilcoxon signed-rank p on a-b (a=test, b=comparator).
    Identical settings to scripts/y3_p3_eval.py (zero_method='pratt')."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0.0):
        return 1.0
    if not _HAVE_SCIPY:
        return float("nan")
    try:
        return float(_wilcoxon(a, b, zero_method="pratt").pvalue)
    except Exception:
        return float("nan")


def win_tie_loss(a, b, tol=_TOL):
    """W/T/L of a vs b per element: a WIN = a strictly LOWER (better on TWT*)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = int(np.sum(a < b - tol))
    l = int(np.sum(a > b + tol))
    return {"W": w, "T": int(len(a) - w - l), "L": l}


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1).
    Returns a list aligned to the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: (float("inf") if pvals[i] is None
                   or np.isnan(pvals[i]) else pvals[i]))
    adj = [None] * m
    run = 0.0
    for rank, idx in enumerate(order):
        p = pvals[idx]
        if p is None or np.isnan(p):
            adj[idx] = float("nan")
            continue
        val = min(1.0, (m - rank) * p)
        run = max(run, val)              # enforce monotone non-decreasing
        adj[idx] = run
    return adj


# --------------------------------------------------------------------------- #
# Per (cell, seed) evaluation of the five cheap deciders                      #
# --------------------------------------------------------------------------- #
def _cell_sig(task):
    keys = ["campus", "regime", "u", "size", "beta", "rho", "eps", "theta",
            "mech", "channel", "family", "master_seed", "seed",
            "n_train", "n_probe", "n_eval", "m0_iters"]
    payload = {k: task.get(k) for k in keys}
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def evaluate_cell(task):
    """Train the M0 estimator for this (cell, seed) and score the five deciders
    on the held-out instances. Cached to disk by cell signature."""
    t0 = time.perf_counter()
    torch.set_num_threads(1)
    try:
        os.nice(5)
    except Exception:
        pass

    sig = _cell_sig(task)
    cache_path = os.path.join(_CACHE, "%s.json" % sig)
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                rec = json.load(fh)
            rec["cached"] = True
            return rec
        except Exception:
            pass                          # corrupt cache -> recompute

    campus, regime = task["campus"], task["regime"]
    u, size = task.get("u"), task.get("size")
    beta, rho, seed = task["beta"], task["rho"], task["seed"]
    n_train, n_probe, n_eval = task["n_train"], task["n_probe"], task["n_eval"]

    files = locate_files(campus, regime, u=u, size=size)
    need = n_train + n_probe + n_eval
    if len(files) < need:
        raise RuntimeError("only %d files at c%d %s u=%s size=%s (need %d)"
                           % (len(files), campus, regime, u, size, need))

    train = [_load(p) for p in files[:n_train]]
    probe = [_load(p) for p in files[n_train:n_train + n_probe]]
    eval_files = files[n_train + n_probe:n_train + n_probe + n_eval]
    eval_insts = [_load(p) for p in eval_files]
    assert not (set(eval_files) & set(files[:n_train + n_probe])), "eval overlaps train"

    overlay = ov.Overlay(ov.OverlayParams(
        beta=beta, family=task["family"], master_seed=task["master_seed"],
        channel=task["channel"]))
    assert overlay.params.channel == task["channel"]

    cell = {"beta": beta, "rho": rho, "eps": task["eps"], "theta": task["theta"],
            "mechanism": task["mech"]}

    # ---- M0 estimator (deterministic; RULE+SUP override log only) ----------- #
    torch.manual_seed(seed)
    np.random.seed(seed)
    res = AR.run_m0(train, probe, overlay,
                    beta_rho_eps=(beta, rho, task["eps"]),
                    outer_iters=task["m0_iters"], mechanism=task["mech"],
                    theta=task["theta"], seed=seed, device="cpu", verbose=False)
    estimator = res["estimator"]
    m0_last = res["per_iter"][-1]

    # ---- Per-instance TWT* for each decider --------------------------------- #
    per = {k: [] for k in DECIDERS}
    inst_ids = []
    rsup_rf, rsup_orr, m0sup_rf, m0sup_orr = [], [], [], []
    util_pool = util_worst = None
    if regime == "storm2":
        up, uw = [], []
    for inst in eval_insts:
        applied = overlay.apply(inst)
        inst_ids.append(inst["meta"]["id"])

        def sc(sched):
            return TO.score_true(inst, sched, overlay, applied)["TWT_true"]

        # ALONE
        per["rule"].append(sc(dec.run_rule(DispatchEnv(inst), "atc", seed=seed)))
        m0d = AR.augmented_atc_decider(estimator, inst, channel=task["channel"])
        m0_sched, _ = DispatchEnv(inst).run_supervised(
            m0d, supervisor=None, method="m0", seed=seed)
        per["m0_alone"].append(sc(m0_sched))
        osup = Supervisor(overlay, inst, rho=0.0, applied=applied)
        per["oracle"].append(sc(dec.run_oracle_greedy(DispatchEnv(inst), osup,
                                                       seed=seed)))
        # IN-LOOP
        rsup = Supervisor(overlay, inst, rho=rho, epsilon=task["eps"],
                          theta=task["theta"], mechanism=task["mech"],
                          seed=seed, applied=applied)
        rsched, _ = dec.run_rule_sup(DispatchEnv(inst), "atc", rsup, seed=seed)
        per["rule_sup"].append(sc(rsched))
        rs = rsup.summary()
        rsup_rf.append(rs["reviewed_fraction"]); rsup_orr.append(rs["override_rate_of_reviews"])

        m0d2 = AR.augmented_atc_decider(estimator, inst, channel=task["channel"])
        m0sup = Supervisor(overlay, inst, rho=rho, epsilon=task["eps"],
                           theta=task["theta"], mechanism=task["mech"],
                           seed=seed, applied=applied)
        m0s_sched, _ = DispatchEnv(inst).run_supervised(
            m0d2, supervisor=m0sup, method="m0_sup", seed=seed)
        per["m0_sup"].append(sc(m0s_sched))
        ms = m0sup.summary()
        m0sup_rf.append(ms["reviewed_fraction"]); m0sup_orr.append(ms["override_rate_of_reviews"])

        if regime == "storm2":
            p_, w_ = _utilization(inst)
            up.append(p_); uw.append(w_)

    if regime == "storm2":
        util_pool = float(np.mean(up)); util_worst = float(np.mean(uw))

    rec = {
        "sig": sig, "campus": campus, "regime": regime, "u": u, "size": size,
        "beta": beta, "rho": rho, "eps": task["eps"], "seed": seed,
        "channel": task["channel"], "n_train": n_train, "n_probe": n_probe,
        "n_eval": len(inst_ids), "n_eval_requested": n_eval,
        "subsampled": n_eval < task.get("n_eval_full", n_eval),
        "inst_ids": inst_ids, "n_wos": len(eval_insts[0]["work_orders"]),
        "per": {k: [float(x) for x in per[k]] for k in DECIDERS},
        "rule_sup_revfrac": [float(x) for x in rsup_rf],
        "rule_sup_orr": [float(x) for x in rsup_orr],
        "m0_sup_revfrac": [float(x) for x in m0sup_rf],
        "m0_sup_orr": [float(x) for x in m0sup_orr],
        "util_pool": util_pool, "util_worst": util_worst,
        "m0_final": {"override_rate": m0_last["override_rate"],
                     "pearson_r": m0_last["pearson_r"],
                     "sign_acc_nonzero": m0_last["sign_acc_nonzero"],
                     "zero_baseline_acc": m0_last["zero_baseline_acc"]},
        "elapsed_s": time.perf_counter() - t0,
        "cached": False,
    }
    os.makedirs(_CACHE, exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, cache_path)
    return rec


# --------------------------------------------------------------------------- #
# Task builders                                                               #
# --------------------------------------------------------------------------- #
def _base_task(**kw):
    t = {"regime": "storm2", "u": None, "size": None, "eps": EPS, "theta": THETA,
         "mech": MECH, "channel": CHANNEL, "family": FAMILY,
         "master_seed": MASTER_SEED, "n_train": 16, "n_probe": 4, "n_eval": 10,
         "n_eval_full": 10, "m0_iters": 8}
    t.update(kw)
    return t


def tasks_A(n_eval_c10=8):
    """PART A: gate significance. primary c9 seeds 301-310; sign check c10 301-303."""
    out = []
    cells = [(u, beta, rho) for u in (90, 100) for beta in (0.75, 1.0)
             for rho in (0.25, 0.5)]
    for (u, beta, rho) in cells:
        for seed in range(301, 311):
            out.append(_base_task(campus=9, u=u, beta=beta, rho=rho, seed=seed,
                                   scope="primary", part="A"))
        for seed in range(301, 304):
            out.append(_base_task(campus=10, u=u, beta=beta, rho=rho, seed=seed,
                                   scope="signcheck", part="A",
                                   n_eval=n_eval_c10))
    return out


def tasks_B(n_eval_c10=8):
    """PART B: E3 regime map. c9 & c10, u ladder, beta{0,.5,1}, rho .25, 3 seeds."""
    out = []
    for campus in (9, 10):
        for u in (70, 90, 100, 110, 130):
            for beta in (0.0, 0.5, 1.0):
                for seed in range(301, 304):
                    kw = dict(campus=campus, u=u, beta=beta, rho=0.25, seed=seed,
                              scope="e3", part="B")
                    if campus == 10:
                        kw["n_eval"] = n_eval_c10
                    out.append(_base_task(**kw))
    return out


def tasks_C(n_eval_c10=8):
    """PART C: boundary cells. (i) c12/c5 storm2 u100; (ii) replay slack cells."""
    out = []
    # (i) no-leverage / small-denominator campuses
    for campus in (12, 5):
        for seed in range(301, 304):
            out.append(_base_task(campus=campus, u=100, beta=1.0, rho=0.25,
                                  seed=seed, scope="boundary_noleverage", part="C"))
    # (ii) slack-capacity replay-default cells (original non-contention regime)
    for campus in (9, 10):
        for size in (150, 400):
            for seed in range(301, 304):
                out.append(_base_task(campus=campus, regime="replay", size=size,
                                      u=None, beta=1.0, rho=0.25, seed=seed,
                                      scope="boundary_slack", part="C"))
    return out


# --------------------------------------------------------------------------- #
# Runner: dispatch a task list across workers, write per-instance CSV rows     #
# --------------------------------------------------------------------------- #
def _cell_label(t):
    if t["regime"] == "replay":
        return "c%d replay%d b%.2f r%.2f s%d" % (t["campus"], t["size"], t["beta"],
                                                 t["rho"], t["seed"])
    return "c%d %s u%d b%.2f r%.2f s%d" % (t["campus"], t["regime"], t["u"],
                                           t["beta"], t["rho"], t["seed"])


CSV_COLS = ["part", "scope", "campus", "regime", "u", "size", "beta", "rho",
            "seed", "inst_id", "n_wos", "util_pool", "util_worst", "subsampled",
            "rule", "m0_alone", "oracle", "rule_sup", "m0_sup",
            "rule_sup_revfrac", "rule_sup_orr", "m0_sup_revfrac", "m0_sup_orr"]


def _append_rows(csv_path, task, rec):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        for i, iid in enumerate(rec["inst_ids"]):
            w.writerow({
                "part": task["part"], "scope": task["scope"],
                "campus": rec["campus"], "regime": rec["regime"], "u": rec["u"],
                "size": rec["size"], "beta": rec["beta"], "rho": rec["rho"],
                "seed": rec["seed"], "inst_id": iid, "n_wos": rec["n_wos"],
                "util_pool": rec["util_pool"], "util_worst": rec["util_worst"],
                "subsampled": int(rec["subsampled"]),
                "rule": "%.6f" % rec["per"]["rule"][i],
                "m0_alone": "%.6f" % rec["per"]["m0_alone"][i],
                "oracle": "%.6f" % rec["per"]["oracle"][i],
                "rule_sup": "%.6f" % rec["per"]["rule_sup"][i],
                "m0_sup": "%.6f" % rec["per"]["m0_sup"][i],
                "rule_sup_revfrac": "%.4f" % rec["rule_sup_revfrac"][i],
                "rule_sup_orr": "%.4f" % rec["rule_sup_orr"][i],
                "m0_sup_revfrac": "%.4f" % rec["m0_sup_revfrac"][i],
                "m0_sup_orr": "%.4f" % rec["m0_sup_orr"][i],
            })


def run_tasks(tasks, csv_path, workers, label, fresh=True):
    # `fresh` truncates the CSV so a re-run (cache-backed, hence fast) rebuilds it
    # cleanly instead of duplicate-appending. Append when a second part writes the
    # same CSV (e.g. A10 after A9 into m0_gate.csv).
    if fresh and os.path.exists(csv_path):
        os.remove(csv_path)
    print("[%s] %d cell-seed tasks, %d workers -> %s" % (label, len(tasks),
          workers, csv_path), flush=True)
    records = []
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(evaluate_cell, t): t for t in tasks}
        for f in as_completed(fut):
            t = fut[f]
            rec = f.result()
            records.append((t, rec))
            _append_rows(csv_path, t, rec)
            done += 1
            print("  [%s %d/%d] %-34s TWT* rule=%.0f m0=%.0f m0+sup=%.0f rule+sup=%.0f "
                  "or=%.0f | util=%s r=%.2f orr=%.3f %s (%.0fs, wall %.0fs)"
                  % (label, done, len(tasks), _cell_label(t),
                     np.mean(rec["per"]["rule"]), np.mean(rec["per"]["m0_alone"]),
                     np.mean(rec["per"]["m0_sup"]), np.mean(rec["per"]["rule_sup"]),
                     np.mean(rec["per"]["oracle"]),
                     ("%.2f" % rec["util_pool"]) if rec["util_pool"] else "n/a",
                     rec["m0_final"]["pearson_r"], rec["m0_final"]["override_rate"],
                     "CACHED" if rec.get("cached") else "",
                     rec["elapsed_s"], time.time() - t0), flush=True)
    return records


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #
def _group_by_cell(records):
    """key (campus,regime,u,size,beta,rho) -> list of (task, rec) over seeds."""
    g = defaultdict(list)
    for t, r in records:
        key = (r["campus"], r["regime"], r["u"], r["size"], r["beta"], r["rho"])
        g[key].append((t, r))
    return g


def _stack(recs, decider):
    """(S seeds x n instances) matrix, rows aligned by seed order; instances
    aligned by inst_id (the held-out set is identical across seeds, but keep only
    ids present in every seed to be robust to a subsample mismatch)."""
    recs = sorted(recs, key=lambda tr: tr[1]["seed"])
    ref_ids = recs[0][1]["inst_ids"]
    common = [iid for iid in ref_ids if all(iid in r["inst_ids"] for _t, r in recs)]
    mat = []
    for _t, r in recs:
        idx = {iid: i for i, iid in enumerate(r["inst_ids"])}
        mat.append([r["per"][decider][idx[iid]] for iid in common])
    return np.asarray(mat, float), common


def _seed_meanstd(recs, decider):
    mat, _ = _stack(recs, decider)          # S x n
    seed_means = mat.mean(axis=1)           # per-seed cell means
    return float(seed_means.mean()), float(seed_means.std(ddof=0)), int(mat.shape[0])


def _contrast(recs, test, comp):
    """Per-instance seed-averaged paired contrast: seed-average each decider's
    per-instance TWT*, then paired Wilcoxon + W/T/L on the n held-out instances."""
    a_mat, _ = _stack(recs, test)
    b_mat, _ = _stack(recs, comp)
    a = a_mat.mean(axis=0)                   # seed-averaged per instance
    b = b_mat.mean(axis=0)
    am, bm = float(a.mean()), float(b.mean())
    return {"test": test, "comparator": comp,
            "test_mean": am, "comparator_mean": bm,
            "pct_vs_comparator": (100.0 * (bm - am) / bm) if abs(bm) > 1e-12 else 0.0,
            "wtl": win_tie_loss(a, b), "wilcoxon_p": paired_wilcoxon(a, b),
            "n_instances": int(a.size)}


def summarize_gate(records, out_json):
    """PART A summary: per-cell seed-mean ladder + 3 contrasts; Holm within the
    primary (campus-9) gate scope, per contrast type."""
    g = _group_by_cell(records)
    cells = {}
    for key, recs in g.items():
        campus, regime, u, size, beta, rho = key
        scope = recs[0][0]["scope"]
        ladder = {}
        for d in DECIDERS:
            m, sd, S = _seed_meanstd(recs, d)
            ladder[d] = {"twt_mean": m, "twt_std": sd, "n_seeds": S}
        rule_m = ladder["rule"]["twt_mean"]
        for d in DECIDERS:
            ladder[d]["pct_below_rule"] = (100.0 * (rule_m - ladder[d]["twt_mean"]) / rule_m
                                           if rule_m > 1e-12 else float("nan"))
        contrasts = {
            "M0_vs_RULE": _contrast(recs, "m0_alone", "rule"),
            "M0sup_vs_RULEsup": _contrast(recs, "m0_sup", "rule_sup"),
            "M0sup_vs_ORACLE": _contrast(recs, "m0_sup", "oracle"),
        }
        ck = "c%d_%s_u%s_b%.2f_r%.2f" % (campus, regime, u, beta, rho)
        cells[ck] = {"campus": campus, "regime": regime, "u": u, "beta": beta,
                     "rho": rho, "scope": scope, "n_seeds": ladder["rule"]["n_seeds"],
                     "n_wos": recs[0][1]["n_wos"], "subsampled": recs[0][1]["subsampled"],
                     "util_pool": recs[0][1]["util_pool"],
                     "ladder": ladder, "contrasts": contrasts,
                     "m0_final_mean": {
                         "pearson_r": float(np.mean([r["m0_final"]["pearson_r"] for _t, r in recs])),
                         "override_rate": float(np.mean([r["m0_final"]["override_rate"] for _t, r in recs]))},
                     "rule_sup_revfrac": float(np.mean([np.mean(r["rule_sup_revfrac"]) for _t, r in recs])),
                     "m0_sup_revfrac": float(np.mean([np.mean(r["m0_sup_revfrac"]) for _t, r in recs]))}

    # ---- Holm within the PRIMARY (campus-9) gate scope, per contrast type ---- #
    holm_out = {}
    primary = {ck: c for ck, c in cells.items() if c["scope"] == "primary"}
    for cname in ["M0_vs_RULE", "M0sup_vs_RULEsup", "M0sup_vs_ORACLE"]:
        keys = sorted(primary.keys())
        pv = [primary[k]["contrasts"][cname]["wilcoxon_p"] for k in keys]
        adj = holm(pv)
        holm_out[cname] = {k: {"raw_p": pv[i], "holm_p": adj[i],
                               "pct": primary[k]["contrasts"][cname]["pct_vs_comparator"],
                               "wtl": primary[k]["contrasts"][cname]["wtl"]}
                           for i, k in enumerate(keys)}
        n_sig = sum(1 for k in keys if adj[keys.index(k)] is not None
                    and not np.isnan(adj[keys.index(k)]) and adj[keys.index(k)] < 0.05)
        holm_out[cname]["_n_cells"] = len(keys)
        holm_out[cname]["_n_sig_holm_0.05"] = n_sig

    summary = {"config": {"channel": CHANNEL, "family": FAMILY,
                          "master_seed": MASTER_SEED, "eps": EPS, "theta": THETA,
                          "mechanism": MECH,
                          "scoring": "TWT*(w*,d*) full_class_shift, independent validator",
                          "contrast_method": "seed-averaged per-instance paired Wilcoxon (pratt), "
                                             "W=test strictly lower TWT*",
                          "holm_scope": "primary campus-9 gate cells, per contrast type"},
               "cells": cells, "holm_primary_gate": holm_out}
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("[gate] wrote %s" % out_json, flush=True)
    return summary


def summarize_e3(records, out_json):
    """PART B summary: readable per-cell M0/M0+SUP headroom over RULE/RULE+SUP +
    utilization, organized as a load-sweep ladder per (campus, beta)."""
    g = _group_by_cell(records)
    cells = {}
    for key, recs in g.items():
        campus, regime, u, size, beta, rho = key
        row = {}
        for d in DECIDERS:
            m, sd, S = _seed_meanstd(recs, d)
            row[d] = m; row[d + "_std"] = sd
        rule_m, rsup_m = row["rule"], row["rule_sup"]
        row["n_seeds"] = S
        row["util_pool"] = recs[0][1]["util_pool"]
        row["util_worst"] = recs[0][1]["util_worst"]
        row["n_wos"] = recs[0][1]["n_wos"]
        row["subsampled"] = recs[0][1]["subsampled"]
        row["m0_over_rule_pct"] = (100.0 * (rule_m - row["m0_alone"]) / rule_m) if rule_m > 1e-12 else float("nan")
        row["m0sup_over_rulesup_pct"] = (100.0 * (rsup_m - row["m0_sup"]) / rsup_m) if rsup_m > 1e-12 else float("nan")
        row["m0sup_over_rule_pct"] = (100.0 * (rule_m - row["m0_sup"]) / rule_m) if rule_m > 1e-12 else float("nan")
        row["oracle_over_rule_pct"] = (100.0 * (rule_m - row["oracle"]) / rule_m) if rule_m > 1e-12 else float("nan")
        row["m0_final_r"] = float(np.mean([r["m0_final"]["pearson_r"] for _t, r in recs]))
        row["m0_final_orr"] = float(np.mean([r["m0_final"]["override_rate"] for _t, r in recs]))
        ck = "c%d_u%d_b%.2f" % (campus, u, beta)
        cells[ck] = row
    summary = {"config": {"channel": CHANNEL, "rho": 0.25, "n_seeds": 3,
                          "scoring": "TWT*(w*,d*) full_class_shift",
                          "note": "descriptive load-sweep; beta=0 is the no-recoverable-info boundary"},
               "cells": cells}
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("[e3] wrote %s" % out_json, flush=True)
    return summary


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A9", "A10", "A", "B", "C", "all"],
                    default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-eval-c10", type=int, default=8,
                    help="held-out eval size for the large c10 storm2 cells")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    os.makedirs(_OUT, exist_ok=True)
    os.makedirs(_CACHE, exist_ok=True)

    gate_csv = os.path.join(_OUT, "m0_gate.csv")
    e3_csv = os.path.join(_OUT, "e3_map.csv")
    bnd_csv = os.path.join(_OUT, "boundary.csv")

    allA = tasks_A(n_eval_c10=args.n_eval_c10)
    A9 = [t for t in allA if t["campus"] == 9]
    A10 = [t for t in allA if t["campus"] == 10]
    B = tasks_B(n_eval_c10=args.n_eval_c10)
    C = tasks_C(n_eval_c10=args.n_eval_c10)

    def do_A_summary():
        # rebuild the PART A record set from CACHE ONLY (skip cells not yet
        # computed, so a partial harvest still writes a summary of what exists).
        recs = []
        for t in allA:
            cache_path = os.path.join(_CACHE, "%s.json" % _cell_sig(t))
            if not os.path.exists(cache_path):
                continue
            with open(cache_path) as fh:
                recs.append((t, json.load(fh)))
        if recs:
            summarize_gate(recs, os.path.join(_OUT, "m0_gate_summary.json"))
        else:
            print("[gate] no cached A cells yet; summary skipped", flush=True)

    if args.part in ("A9", "all"):
        run_tasks(A9, gate_csv, args.workers, "A9", fresh=True)
    if args.part in ("C", "all"):
        run_tasks(C, bnd_csv, args.workers, "C", fresh=True)
    if args.part in ("A10", "all"):
        # append into the same gate CSV that A9 already opened
        run_tasks(A10, gate_csv, args.workers, "A10", fresh=(args.part == "A10"))
    if args.part in ("A9", "A10", "A", "all"):
        # write the gate summary once both campuses are on disk (cache-backed)
        try:
            do_A_summary()
        except Exception as e:                              # partial harvest ok
            print("[gate] summary deferred (%s)" % e, flush=True)
    if args.part in ("A",):
        run_tasks(A9, gate_csv, args.workers, "A9", fresh=True)
        run_tasks(A10, gate_csv, args.workers, "A10", fresh=False)
        do_A_summary()
    if args.part in ("B", "all"):
        recs = run_tasks(B, e3_csv, args.workers, "B", fresh=True)
        summarize_e3(recs, os.path.join(_OUT, "e3_map_summary.json"))

    print("[y3_p4] part=%s complete." % args.part, flush=True)


if __name__ == "__main__":
    main()
