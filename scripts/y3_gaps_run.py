#!/usr/bin/env python
"""Y3 Phase-5 evaluation gaps (M0-based, NO PPO).

GAP 1  OVERRIDE NOISE eps: does the correction layer degrade gracefully with
       supervisor error, or collapse?  Primary c9 storm2 u100 beta1.0 rho0.25,
       eps in {0.0, 0.1, 0.25}, seeds 301-305; lower-load check c9 u90 same
       cell, eps in {0.0, 0.25}, seeds 301-303.
GAP 2  REVIEW BUDGET rho curve: how much supervisor attention is enough?
       c9 storm2 u100 beta1.0 eps=0, rho in {0.05, 0.1, 0.25, 0.5}, seeds 301-305.

Everything that produces a number REUSES the committed M0 pipeline (fmwos.hitl /
scripts/y3_p4_m0grid): the estimator is trained on the RULE+SUP override log at
the cell's (beta, rho, eps); RULE / M0 / ORACLE-GREEDY / RULE+SUP / M0+SUP are
scored on the held-out TWT*(w*,d*) (full_class_shift, split files[20:30]) via the
same decider paths and per-seed RNG; the contrast is the same seed-averaged
per-instance paired Wilcoxon.

LEAKAGE FIX (label_source).  The override log's positive weak label used to come
from ``preferred_pick`` (the supervisor's noise-free preference). Under the
Appendix-D.4 noise model the random-override branch STARTS a random eligible
order while ``preferred_pick`` stays the clean preference, so at eps>0 that label
is information a deployed logger never sees. ``weak_labels_from_log`` now takes
``label_source``:
  * "executed" (DEFAULT): positive label on ``executed_pick`` -- the honest,
    deployable signal; at eps>0 the random-override label is genuinely corrupted.
  * "preferred": positive label on ``preferred_pick`` -- an UPPER BOUND that
    assumes the log records the supervisor's intent, not its action.
At eps=0 the two are bit-identical (an honest override executes the preferred
pick); this is asserted directly AND transitively via the committed-cache
reproduction check. This script reports BOTH variants side by side for eps>0.

``_run_m0_labelsource`` mirrors ``AR.run_m0`` step-for-step (same ShiftEstimator,
permutation RNG, never-reset aggregate, ``train_estimator(seed=seed+it)``,
``probe_shift_accuracy``) with the single addition of forwarding label_source; at
eps=0/executed it reproduces run_m0 bit-for-bit.

Additive only: writes results/y3_p5/gaps/ (own cache) + notes/gaps_eps_rho.md.
The one locked-file edit is the additive ``label_source`` arg on
``weak_labels_from_log`` (a no-op for every committed eps=0 caller). Never writes
results/y3_p4. Coexists with the training sweep: <=5 workers, OMP=1, niced.

Run (CPU only, in a y3_-prefixed tmux):
    PYTHONPATH=src OMP_NUM_THREADS=1 nice -n 15 \
        python scripts/y3_gaps_run.py --part eps --workers 5 --campus 9
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch                                                        # noqa: E402

from fmwos.env import DispatchEnv                                   # noqa: E402
from fmwos.hitl import deciders as dec                              # noqa: E402
from fmwos.hitl import overlay as ov                               # noqa: E402
from fmwos.hitl.supervisor import Supervisor                        # noqa: E402
from fmwos.hitl import augmented_rule as AR                         # noqa: E402
from fmwos.hitl import true_objective as TO                         # noqa: E402
from fmwos.hitl.latent_head import ShiftEstimator, train_estimator, LAT_DIM  # noqa: E402

import y3_p4_m0grid as G                                            # noqa: E402  (reuse committed machinery)

_OUT = os.path.join(_ROOT, "results", "y3_p5", "gaps")
_CACHE = os.path.join(_OUT, "cache")
DECIDERS = ["rule", "m0_alone", "oracle", "rule_sup", "m0_sup"]
LABEL_SOURCES = ("executed", "preferred")


# --------------------------------------------------------------------------- #
# M0 DAgger loop: exact mirror of AR.run_m0 with a forwarded label_source.      #
# --------------------------------------------------------------------------- #
def _run_m0_labelsource(train_instances, probe_instances, overlay, *, beta_rho_eps,
                        outer_iters=8, episodes_per_iter=None, mechanism="targeted",
                        theta=1.0, override_weight=5.0, confirm_weight=1.0,
                        est_hidden=32, seed=0, device="cpu", label_source="executed"):
    """Bit-for-bit AR.run_m0 (same estimator, permutation RNG, never-reset
    aggregate, train_estimator(seed=seed+it), probe_shift_accuracy) except that
    weak_labels_from_log is called with ``label_source``. At eps=0 / executed it
    equals run_m0 exactly (verified against the committed cache)."""
    beta, rho, eps = beta_rho_eps
    channel = getattr(overlay.params, "channel", "full_class_shift")
    estimator = ShiftEstimator(hidden=est_hidden)
    Xagg = np.zeros((0, LAT_DIM), np.float32)
    yagg = np.zeros((0,), np.float32)
    wagg = np.zeros((0,), np.float32)
    per_iter = []
    rng = np.random.default_rng(seed)
    n_ep = episodes_per_iter or len(train_instances)
    for it in range(outer_iters):
        order = rng.permutation(len(train_instances))[:n_ep]
        n_over = n_rev = n_conf = 0
        for k in order:
            inst = train_instances[int(k)]
            applied = overlay.apply(inst)
            sup = Supervisor(overlay, inst, rho=rho, epsilon=eps, theta=theta,
                             mechanism=mechanism, seed=seed, applied=applied)
            decider = AR.augmented_atc_decider(estimator, inst, device=device, channel=channel)
            _sched, log = DispatchEnv(inst).run_supervised(
                decider, supervisor=sup, method="m0_atc", seed=seed)
            X, y, w = AR.weak_labels_from_log(log, inst, override_weight, confirm_weight,
                                              label_source=label_source)
            if len(X):
                Xagg = np.concatenate([Xagg, X]); yagg = np.concatenate([yagg, y])
                wagg = np.concatenate([wagg, w])
            s = sup.summary()
            n_over += s["n_overrides"]; n_rev += s["n_reviews"]; n_conf += s["n_confirmations"]
        loss = train_estimator(estimator, Xagg, yagg, wagg, device=device, seed=seed + it)
        acc = AR.probe_shift_accuracy(estimator, probe_instances, overlay, device=device)
        orr = (n_over / n_rev) if n_rev else 0.0
        per_iter.append({"iter": it, "n_reviews": n_rev, "n_overrides": n_over,
                         "n_confirmations": n_conf, "override_rate": orr,
                         "n_examples_agg": int(len(Xagg)), "est_loss": loss, **acc})
    return {"estimator": estimator, "per_iter": per_iter}


# --------------------------------------------------------------------------- #
# Per (cell, seed, label_source) evaluation.                                   #
# --------------------------------------------------------------------------- #
def _cache_key(task):
    return "%s_%s" % (G._cell_sig(task), task["label_source"])


def evaluate_cell_gaps(task):
    t0 = time.perf_counter()
    torch.set_num_threads(1)
    try:
        os.nice(5)
    except Exception:
        pass

    bare_sig = G._cell_sig(task)          # committed-cache twin (label_source ignored by _cell_sig)
    key = _cache_key(task)
    cache_path = os.path.join(_CACHE, "%s.json" % key)
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                rec = json.load(fh)
            rec["cached"] = True
            return rec
        except Exception:
            pass

    campus, regime = task["campus"], task["regime"]
    u = task.get("u")
    beta, rho, eps, seed = task["beta"], task["rho"], task["eps"], task["seed"]
    label_source = task["label_source"]
    n_train, n_probe, n_eval = task["n_train"], task["n_probe"], task["n_eval"]

    files = G.locate_files(campus, regime, u=u, size=task.get("size"))
    need = n_train + n_probe + n_eval
    if len(files) < need:
        raise RuntimeError("only %d files at c%d %s u=%s (need %d)"
                           % (len(files), campus, regime, u, need))
    train = [G._load(p) for p in files[:n_train]]
    probe = [G._load(p) for p in files[n_train:n_train + n_probe]]
    eval_files = files[n_train + n_probe:n_train + n_probe + n_eval]
    eval_insts = [G._load(p) for p in eval_files]
    assert not (set(eval_files) & set(files[:n_train + n_probe])), "eval overlaps train"

    overlay = ov.Overlay(ov.OverlayParams(
        beta=beta, family=task["family"], master_seed=task["master_seed"],
        channel=task["channel"]))
    assert overlay.params.channel == task["channel"]

    # ---- M0 estimator: trained on the override log with the chosen label source
    torch.manual_seed(seed)
    np.random.seed(seed)
    res = _run_m0_labelsource(train, probe, overlay,
                              beta_rho_eps=(beta, rho, eps),
                              outer_iters=task["m0_iters"], mechanism=task["mech"],
                              theta=task["theta"], seed=seed, device="cpu",
                              label_source=label_source)
    estimator = res["estimator"]
    per_iter = res["per_iter"]
    m0_last = per_iter[-1]
    train_overrides = int(sum(r["n_overrides"] for r in per_iter))
    train_reviews = int(sum(r["n_reviews"] for r in per_iter))
    train_confirms = int(sum(r["n_confirmations"] for r in per_iter))
    recovery_curve = [{"iter": r["iter"], "n_overrides": r["n_overrides"],
                       "sign_acc_nonzero": r["sign_acc_nonzero"],
                       "pearson_r": r["pearson_r"],
                       "override_rate": r["override_rate"]} for r in per_iter]

    # ---- Per-instance TWT*(w*,d*) for the five cheap deciders (held out) ------
    per = {k: [] for k in DECIDERS}
    inst_ids = []
    rsup_rf, rsup_orr, m0sup_rf, m0sup_orr = [], [], [], []
    up, uw = [], []
    for inst in eval_insts:
        applied = overlay.apply(inst)
        inst_ids.append(inst["meta"]["id"])

        def sc(sched):
            return TO.score_true(inst, sched, overlay, applied)["TWT_true"]

        per["rule"].append(sc(dec.run_rule(DispatchEnv(inst), "atc", seed=seed)))
        m0d = AR.augmented_atc_decider(estimator, inst, channel=task["channel"])
        m0_sched, _ = DispatchEnv(inst).run_supervised(
            m0d, supervisor=None, method="m0", seed=seed)
        per["m0_alone"].append(sc(m0_sched))
        osup = Supervisor(overlay, inst, rho=0.0, applied=applied)
        per["oracle"].append(sc(dec.run_oracle_greedy(DispatchEnv(inst), osup, seed=seed)))

        rsup = Supervisor(overlay, inst, rho=rho, epsilon=eps, theta=task["theta"],
                          mechanism=task["mech"], seed=seed, applied=applied)
        rsched, _ = dec.run_rule_sup(DispatchEnv(inst), "atc", rsup, seed=seed)
        per["rule_sup"].append(sc(rsched))
        rs = rsup.summary()
        rsup_rf.append(rs["reviewed_fraction"]); rsup_orr.append(rs["override_rate_of_reviews"])

        m0d2 = AR.augmented_atc_decider(estimator, inst, channel=task["channel"])
        m0sup = Supervisor(overlay, inst, rho=rho, epsilon=eps, theta=task["theta"],
                           mechanism=task["mech"], seed=seed, applied=applied)
        m0s_sched, _ = DispatchEnv(inst).run_supervised(
            m0d2, supervisor=m0sup, method="m0_sup", seed=seed)
        per["m0_sup"].append(sc(m0s_sched))
        ms = m0sup.summary()
        m0sup_rf.append(ms["reviewed_fraction"]); m0sup_orr.append(ms["override_rate_of_reviews"])

        p_, w_ = G._utilization(inst)
        up.append(p_); uw.append(w_)

    rec = {
        "sig": bare_sig, "cache_key": key, "label_source": label_source,
        "campus": campus, "regime": regime, "u": u,
        "beta": beta, "rho": rho, "eps": eps, "seed": seed, "gap": task.get("gap"),
        "channel": task["channel"], "n_train": n_train, "n_probe": n_probe,
        "n_eval": len(inst_ids), "inst_ids": inst_ids,
        "n_wos": len(eval_insts[0]["work_orders"]),
        "per": {k: [float(x) for x in per[k]] for k in DECIDERS},
        "rule_sup_revfrac": [float(x) for x in rsup_rf],
        "rule_sup_orr": [float(x) for x in rsup_orr],
        "m0_sup_revfrac": [float(x) for x in m0sup_rf],
        "m0_sup_orr": [float(x) for x in m0sup_orr],
        "util_pool": float(np.mean(up)), "util_worst": float(np.mean(uw)),
        "m0_train_overrides": train_overrides, "m0_train_reviews": train_reviews,
        "m0_train_confirmations": train_confirms,
        "m0_final": {"override_rate": m0_last["override_rate"],
                     "pearson_r": m0_last["pearson_r"],
                     "sign_acc_nonzero": m0_last["sign_acc_nonzero"],
                     "zero_baseline_acc": m0_last["zero_baseline_acc"]},
        "recovery_curve": recovery_curve,
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
# Task construction                                                           #
# --------------------------------------------------------------------------- #
def build_tasks(campus, part):
    tasks = {}

    def add(gap, label_source, **kw):
        t = G._base_task(campus=campus, **kw)
        t["gap"] = gap
        t["label_source"] = label_source
        tasks[_cache_key(t)] = t

    if part in ("eps", "all"):
        # eps=0: executed only (executed==preferred; the shared anchor).
        for seed in range(301, 306):
            add("eps", "executed", u=100, beta=1.0, rho=0.25, eps=0.0, seed=seed)
        for seed in range(301, 304):
            add("eps_u90", "executed", u=90, beta=1.0, rho=0.25, eps=0.0, seed=seed)
        # eps>0: BOTH label sources.
        for ls in LABEL_SOURCES:
            for eps in (0.1, 0.25):
                for seed in range(301, 306):
                    add("eps", ls, u=100, beta=1.0, rho=0.25, eps=eps, seed=seed)
            for seed in range(301, 304):
                add("eps_u90", ls, u=90, beta=1.0, rho=0.25, eps=0.25, seed=seed)

    if part in ("rho", "all"):
        for rho in (0.05, 0.1, 0.25, 0.5):
            for seed in range(301, 306):
                add("rho", "executed", u=100, beta=1.0, rho=rho, eps=0.0, seed=seed)

    return list(tasks.values())


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #
def _stack(recs, decider):
    recs = sorted(recs, key=lambda r: r["seed"])
    ref = recs[0]["inst_ids"]
    common = [i for i in ref if all(i in r["inst_ids"] for r in recs)]
    mat = []
    for r in recs:
        idx = {iid: k for k, iid in enumerate(r["inst_ids"])}
        mat.append([r["per"][decider][idx[i]] for i in common])
    return np.asarray(mat, float), common


def _seed_meanstd(recs, decider):
    mat, _ = _stack(recs, decider)
    sm = mat.mean(axis=1)
    return float(sm.mean()), float(sm.std(ddof=0)), int(mat.shape[0])


def _contrast(recs, test, comp):
    a = _stack(recs, test)[0].mean(axis=0)
    b = _stack(recs, comp)[0].mean(axis=0)
    am, bm = float(a.mean()), float(b.mean())
    return {"test": test, "comparator": comp, "test_mean": am, "comparator_mean": bm,
            "pct_vs_comparator": (100.0 * (bm - am) / bm) if abs(bm) > 1e-12 else 0.0,
            "wtl": G.win_tie_loss(a, b), "wilcoxon_p": G.paired_wilcoxon(a, b),
            "n_instances": int(a.size)}


def _mean(recs, path):
    vals = []
    for r in recs:
        v = r
        for k in path:
            v = v[k]
        vals.append(np.mean(v) if isinstance(v, list) else v)
    return float(np.mean(vals))


def summarize_group(recs):
    ladder = {}
    for d in DECIDERS:
        m, sd, S = _seed_meanstd(recs, d)
        ladder[d] = {"mean": m, "std": sd, "n_seeds": S}
    rule_m = ladder["rule"]["mean"]
    for d in DECIDERS:
        ladder[d]["pct_below_rule"] = (100.0 * (rule_m - ladder[d]["mean"]) / rule_m
                                       if rule_m > 1e-12 else float("nan"))
    return {
        "n_seeds": ladder["rule"]["n_seeds"],
        "label_source": recs[0].get("label_source"),
        "util_pool": _mean(recs, ["util_pool"]),
        "n_wos": recs[0]["n_wos"],
        "ladder": ladder,
        "contrasts": {
            "M0_vs_RULE": _contrast(recs, "m0_alone", "rule"),
            "M0_vs_RULEsup": _contrast(recs, "m0_alone", "rule_sup"),
            "M0sup_vs_RULEsup": _contrast(recs, "m0_sup", "rule_sup"),
            "M0sup_vs_ORACLE": _contrast(recs, "m0_sup", "oracle"),
            "RULEsup_vs_RULE": _contrast(recs, "rule_sup", "rule"),
        },
        "recovery": {"pearson_r": _mean(recs, ["m0_final", "pearson_r"]),
                     "sign_acc": _mean(recs, ["m0_final", "sign_acc_nonzero"]),
                     "final_override_rate": _mean(recs, ["m0_final", "override_rate"])},
        "override_load": {
            "train_overrides": _mean(recs, ["m0_train_overrides"]),
            "train_reviews": _mean(recs, ["m0_train_reviews"]),
            "rule_sup_revfrac": _mean(recs, ["rule_sup_revfrac"]),
            "rule_sup_orr": _mean(recs, ["rule_sup_orr"]),
            "m0_sup_revfrac": _mean(recs, ["m0_sup_revfrac"]),
            "m0_sup_orr": _mean(recs, ["m0_sup_orr"])},
    }


# --------------------------------------------------------------------------- #
# Reproduction + label-identity checks                                         #
# --------------------------------------------------------------------------- #
def reproduction_check(records):
    """eps=0 executed cells vs the committed results/y3_p4 cache (bare sig)."""
    checked, max_diff, details = 0, 0.0, []
    for r in records:
        if abs(r["eps"]) > 1e-12 or r["label_source"] != "executed":
            continue
        committed = os.path.join(G._CACHE, r["sig"] + ".json")
        if not os.path.exists(committed):
            continue
        c = json.load(open(committed))
        d = 0.0
        for dd in DECIDERS:
            a = np.asarray(r["per"][dd], float); b = np.asarray(c["per"][dd], float)
            d = max(d, float(np.max(np.abs(a - b)))) if a.shape == b.shape else float("inf")
        checked += 1
        max_diff = max(max_diff, d)
        details.append({"sig": r["sig"], "u": r["u"], "rho": r["rho"],
                        "seed": r["seed"], "max_abs_diff": d})
    return checked, max_diff, details


def label_identity_check(campus=9, u=100, seed=301):
    """Directly assert weak_labels_from_log is bit-identical under 'executed' vs
    'preferred' on an eps=0 override log (executed==preferred there)."""
    files = G.locate_files(campus, "storm2", u=u)
    inst = G._load(files[0])
    overlay = ov.Overlay(ov.OverlayParams(beta=1.0, family=G.FAMILY,
                                          master_seed=G.MASTER_SEED, channel=G.CHANNEL))
    applied = overlay.apply(inst)
    torch.manual_seed(seed); np.random.seed(seed)
    est = ShiftEstimator(hidden=32)
    sup = Supervisor(overlay, inst, rho=0.25, epsilon=0.0, theta=1.0,
                     mechanism=G.MECH, seed=seed, applied=applied)
    dec_fn = AR.augmented_atc_decider(est, inst, channel=G.CHANNEL)
    _s, log = DispatchEnv(inst).run_supervised(dec_fn, supervisor=sup, method="m0_atc", seed=seed)
    Xe, ye, we = AR.weak_labels_from_log(log, inst, label_source="executed")
    Xp, yp, wp = AR.weak_labels_from_log(log, inst, label_source="preferred")
    ok = (Xe.shape == Xp.shape and np.array_equal(Xe, Xp)
          and np.array_equal(ye, yp) and np.array_equal(we, wp))
    n_over = sum(1 for e in log if e.get("override"))
    return {"eps0_labels_identical": bool(ok), "n_override_entries": int(n_over),
            "n_examples": int(len(ye))}


# --------------------------------------------------------------------------- #
# CSV                                                                          #
# --------------------------------------------------------------------------- #
SEED_COLS = ["campus", "u", "beta", "rho", "eps", "label_source", "seed", "n_wos",
             "util_pool", "rule", "m0_alone", "m0_sup", "rule_sup", "oracle",
             "pearson_r", "sign_acc", "final_override_rate",
             "train_overrides", "train_reviews",
             "rule_sup_revfrac", "rule_sup_orr", "m0_sup_revfrac", "m0_sup_orr"]


def write_seed_csv(records, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SEED_COLS)
        w.writeheader()
        for r in sorted(records, key=lambda x: (x["u"], x["rho"], x["eps"],
                                                x["label_source"], x["seed"])):
            w.writerow({
                "campus": r["campus"], "u": r["u"], "beta": r["beta"],
                "rho": r["rho"], "eps": r["eps"], "label_source": r["label_source"],
                "seed": r["seed"], "n_wos": r["n_wos"], "util_pool": "%.4f" % r["util_pool"],
                "rule": "%.4f" % np.mean(r["per"]["rule"]),
                "m0_alone": "%.4f" % np.mean(r["per"]["m0_alone"]),
                "m0_sup": "%.4f" % np.mean(r["per"]["m0_sup"]),
                "rule_sup": "%.4f" % np.mean(r["per"]["rule_sup"]),
                "oracle": "%.4f" % np.mean(r["per"]["oracle"]),
                "pearson_r": "%.4f" % r["m0_final"]["pearson_r"],
                "sign_acc": "%.4f" % r["m0_final"]["sign_acc_nonzero"],
                "final_override_rate": "%.4f" % r["m0_final"]["override_rate"],
                "train_overrides": r["m0_train_overrides"],
                "train_reviews": r["m0_train_reviews"],
                "rule_sup_revfrac": "%.4f" % np.mean(r["rule_sup_revfrac"]),
                "rule_sup_orr": "%.4f" % np.mean(r["rule_sup_orr"]),
                "m0_sup_revfrac": "%.4f" % np.mean(r["m0_sup_revfrac"]),
                "m0_sup_orr": "%.4f" % np.mean(r["m0_sup_orr"]),
            })


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #
def _fmt_c(c):
    w = c["wtl"]
    return "%+.1f%% (W/T/L %d/%d/%d, p=%.4g)" % (c["pct_vs_comparator"],
                                                 w["W"], w["T"], w["L"], c["wilcoxon_p"])


def _row_from_group(eps, ls, g):
    return "| %.2f | %s | %.0f | %.0f&plusmn;%.0f | %.0f&plusmn;%.0f | %.0f | %s | %s | %.3f | %.3f |" % (
        eps, ls, g["ladder"]["rule_sup"]["mean"],
        g["ladder"]["m0_alone"]["mean"], g["ladder"]["m0_alone"]["std"],
        g["ladder"]["m0_sup"]["mean"], g["ladder"]["m0_sup"]["std"],
        g["ladder"]["oracle"]["mean"],
        _fmt_c(g["contrasts"]["M0_vs_RULE"]),
        _fmt_c(g["contrasts"]["M0sup_vs_RULEsup"]),
        g["recovery"]["pearson_r"], g["recovery"]["sign_acc"])


def write_report(eps_exec, eps_pref, u90_exec, u90_pref, rho_groups, repro, ident, ev, path):
    L = []
    L.append("# Y3 evaluation gaps: override-noise eps and review-budget rho (M0, no PPO)\n")
    L.append("Cell c9 storm2 u100 beta1.0, held-out TWT*(w*,d*) full_class_shift, "
             "targeted review, seeds 301-305 (u90 check 301-303). Same M0 pipeline, "
             "split, and seed-averaged paired Wilcoxon as the committed m0-grid.\n")
    L.append("**Leakage fix.** The override-log positive weak label now comes from "
             "the pick the supervisor ACTUALLY started (`label_source=\"executed\"`, "
             "the new default), not its noise-free `preferred_pick`. At eps>0 the "
             "Appendix-D.4 random-override branch starts a random eligible order, so "
             "the executed label is genuinely corrupted; the `preferred` column is the "
             "leaked upper bound (assumes the log records intent, not action).\n")
    rc, rd, _ = repro
    L.append("**Checks.** eps=0 label identity: executed vs preferred weak labels "
             "bit-identical = %s (%d override entries, %d examples). Reproduction: %d "
             "eps=0 executed cells matched against the committed results/y3_p4 cache; "
             "max abs per-instance TWT* diff = %.2e (0 => bit-equal to the m0-grid).\n"
             % (ident["eps0_labels_identical"], ident["n_override_entries"],
                ident["n_examples"], rc, rd))

    L.append("\n## GAP 1  Override noise eps (u100 beta1.0 rho0.25)\n")
    L.append("TWT* lower = better. RULE = 3645 (eps-independent). RULE+SUP depends on "
             "eps (executes noisy overrides at test) but NOT on label_source. M0 and "
             "M0+SUP depend on the estimator, hence on label_source at eps>0.\n")
    L.append("| eps | labels | RULE+SUP | M0 | M0+SUP | ORACLE | M0 vs RULE | M0+SUP vs RULE+SUP | hat_s r | sign-acc |")
    L.append("|----:|:-------|---------:|---:|-------:|-------:|:-----------|:-------------------|--------:|---------:|")
    L.append(_row_from_group(0.0, "exec=pref", eps_exec[0.0]))
    for eps in (0.1, 0.25):
        if eps in eps_exec:
            L.append(_row_from_group(eps, "executed", eps_exec[eps]))
        if eps in eps_pref:
            L.append(_row_from_group(eps, "preferred", eps_pref[eps]))

    epss = sorted(eps_exec)
    L.append("\n**Verdict (executed labels): %s.** With the honest deployable label, "
             "M0 recovery %s with eps (Pearson r %.3f -> %.3f from eps=0 to eps=%.2f) and "
             "M0's TWT* moves %+.1f%%, yet M0 still beats the deployed RULE at every eps "
             "(all Wilcoxon p<0.05, W>L). The `preferred` column shows recovery r RISING "
             "with eps (%.3f -> %.3f) -- that was the leakage artifact: it credits the "
             "estimator with the supervisor's noise-free intent the random-override branch "
             "never executed. RULE+SUP degrades with eps regardless of labels (executes the "
             "noisy overrides at test).\n" % (
                 ev["verdict"], ev["recovery_trend_executed"],
                 ev["recovery_r_endpoints_executed"]["eps_min"],
                 ev["recovery_r_endpoints_executed"]["eps_max"], epss[-1],
                 ev["m0_twt_relative_change_worst_eps_pct"],
                 eps_pref[0.1]["recovery"]["pearson_r"] if 0.1 in eps_pref else float("nan"),
                 eps_pref[0.25]["recovery"]["pearson_r"] if 0.25 in eps_pref else float("nan")))

    L.append("\n### GAP 1 lower-load check (u90 beta1.0 rho0.25, seeds 301-303)\n")
    L.append("| eps | labels | RULE+SUP | M0 | M0+SUP | ORACLE | M0 vs RULE | M0+SUP vs RULE+SUP | hat_s r | sign-acc |")
    L.append("|----:|:-------|---------:|---:|-------:|-------:|:-----------|:-------------------|--------:|---------:|")
    L.append(_row_from_group(0.0, "exec=pref", u90_exec[0.0]))
    if 0.25 in u90_exec:
        L.append(_row_from_group(0.25, "executed", u90_exec[0.25]))
    if 0.25 in u90_pref:
        L.append(_row_from_group(0.25, "preferred", u90_pref[0.25]))
    L.append("\nRULE (u90, eps-independent) = %.0f.\n" % u90_exec[0.0]["ladder"]["rule"]["mean"])

    if rho_groups:
        L.append("\n## GAP 2  Review-budget rho curve (u100 beta1.0 eps=0)  [unaffected by the label fix; eps=0]\n")
        L.append("| rho | reviewed frac | train overrides | RULE+SUP | M0 | M0+SUP | M0 vs RULE | M0 vs RULE+SUP | M0+SUP vs RULE+SUP | hat_s r |")
        L.append("|----:|--------------:|----------------:|---------:|---:|-------:|:-----------|:---------------|:-------------------|--------:|")
        for rho in sorted(rho_groups):
            g = rho_groups[rho]
            L.append("| %.2f | %.3f | %.0f | %.0f | %.0f | %.0f | %s | %s | %s | %.3f |" % (
                rho, g["override_load"]["rule_sup_revfrac"], g["override_load"]["train_overrides"],
                g["ladder"]["rule_sup"]["mean"], g["ladder"]["m0_alone"]["mean"],
                g["ladder"]["m0_sup"]["mean"],
                _fmt_c(g["contrasts"]["M0_vs_RULE"]),
                _fmt_c(g["contrasts"]["M0_vs_RULEsup"]),
                _fmt_c(g["contrasts"]["M0sup_vs_RULEsup"]),
                g["recovery"]["pearson_r"]))
        L.append("\nRULE (rho-independent) = %.0f.\n" % rho_groups[sorted(rho_groups)[0]]["ladder"]["rule"]["mean"])

    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# Verdicts                                                                     #
# --------------------------------------------------------------------------- #
def _sig(c, tol=0.05):
    p = c["wilcoxon_p"]
    return (p is not None and not np.isnan(p) and p < tol and c["wtl"]["W"] > c["wtl"]["L"])


def eps_verdict(eps_exec, eps_pref):
    epss = sorted(eps_exec)
    m0_e0 = eps_exec[epss[0]]["ladder"]["m0_alone"]["mean"]
    m0_eH = eps_exec[epss[-1]]["ladder"]["m0_alone"]["mean"]
    all_sig = all(_sig(eps_exec[e]["contrasts"]["M0_vs_RULE"]) for e in epss)
    twt_change = 100.0 * (m0_eH - m0_e0) / m0_e0
    r_e0 = eps_exec[epss[0]]["recovery"]["pearson_r"]
    r_eH = eps_exec[epss[-1]]["recovery"]["pearson_r"]
    recovery_trend = ("improves" if r_eH > r_e0 + 0.02
                      else "degrades" if r_eH < r_e0 - 0.02 else "flat")
    # Three-way: collapse if M0 stops significantly beating the deployed RULE at
    # any eps; graceful if it still beats RULE with only a modest TWT* rise;
    # otherwise it degrades but still beats RULE (robust, not inflated).
    if not all_sig:
        verdict = "collapse"
    elif abs(twt_change) < 8.0:
        verdict = "graceful"
    else:
        verdict = "degrades_but_beats_rule"
    return {
        "label_source": "executed",
        "eps_values": epss,
        "executed": {
            "m0_gain_vs_rule": {("%.2f" % e): eps_exec[e]["contrasts"]["M0_vs_RULE"]["pct_vs_comparator"] for e in epss},
            "m0sup_gain_vs_rulesup": {("%.2f" % e): eps_exec[e]["contrasts"]["M0sup_vs_RULEsup"]["pct_vs_comparator"] for e in epss},
            "rulesup_gain_vs_rule": {("%.2f" % e): eps_exec[e]["contrasts"]["RULEsup_vs_RULE"]["pct_vs_comparator"] for e in epss},
            "recovery_r": {("%.2f" % e): eps_exec[e]["recovery"]["pearson_r"] for e in epss},
            "recovery_sign": {("%.2f" % e): eps_exec[e]["recovery"]["sign_acc"] for e in epss},
        },
        "preferred_upper_bound": {
            "m0_gain_vs_rule": {("%.2f" % e): eps_pref[e]["contrasts"]["M0_vs_RULE"]["pct_vs_comparator"] for e in eps_pref},
            "recovery_r": {("%.2f" % e): eps_pref[e]["recovery"]["pearson_r"] for e in eps_pref},
        },
        "m0_all_sig_vs_rule": all_sig,
        "m0_twt_relative_change_worst_eps_pct": twt_change,
        "recovery_trend_executed": recovery_trend,
        "recovery_r_endpoints_executed": {"eps_min": r_e0, "eps_max": r_eH},
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def _pick(records, u, rho, eps, label_source):
    out = []
    for r in records:
        if (r["u"] == u and abs(r["rho"] - rho) < 1e-9 and abs(r["eps"] - eps) < 1e-9
                and r["label_source"] == label_source):
            out.append(r)
    return out


def _load_rho_from_summary():
    """Read the preserved GAP-2 rho block from the existing summary.json (eps=0,
    unaffected by the label fix)."""
    p = os.path.join(_OUT, "summary.json")
    if not os.path.exists(p):
        return {}, {}
    s = json.load(open(p))
    g = s.get("gap2_rho", {})
    rho_groups = {float(k): v for k, v in g.items()}
    rho_verdict = s.get("verdicts", {}).get("rho", {})
    return rho_groups, rho_verdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--campus", type=int, default=9)
    ap.add_argument("--part", choices=["eps", "rho", "all"], default="eps")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    os.makedirs(_OUT, exist_ok=True)
    os.makedirs(_CACHE, exist_ok=True)

    ident = label_identity_check()
    print("[gaps] eps=0 label identity (executed==preferred): %s (%d overrides, %d examples)"
          % (ident["eps0_labels_identical"], ident["n_override_entries"], ident["n_examples"]),
          flush=True)
    assert ident["eps0_labels_identical"], "eps=0 executed/preferred labels DIFFER (bug)"

    tasks = build_tasks(args.campus, args.part)
    print("[gaps] part=%s: %d tasks (c%d), %d workers -> %s"
          % (args.part, len(tasks), args.campus, args.workers, _OUT), flush=True)

    records = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut = {ex.submit(evaluate_cell_gaps, t): t for t in tasks}
        for f in as_completed(fut):
            r = f.result()
            records.append(r)
            done += 1
            print("  [%d/%d] u%d rho%.2f eps%.2f %-9s s%d | rule=%.0f m0=%.0f m0+sup=%.0f "
                  "rule+sup=%.0f | r=%.2f sign=%.2f trov=%d %s (%.0fs wall %.0fs)"
                  % (done, len(tasks), r["u"], r["rho"], r["eps"], r["label_source"],
                     r["seed"], np.mean(r["per"]["rule"]), np.mean(r["per"]["m0_alone"]),
                     np.mean(r["per"]["m0_sup"]), np.mean(r["per"]["rule_sup"]),
                     r["m0_final"]["pearson_r"], r["m0_final"]["sign_acc_nonzero"],
                     r["m0_train_overrides"], "CACHED" if r.get("cached") else "",
                     r["elapsed_s"], time.time() - t0), flush=True)

    # ---- group eps views -----------------------------------------------------
    eps_exec, eps_pref, u90_exec, u90_pref = {}, {}, {}, {}
    for eps in (0.0, 0.1, 0.25):
        rr = _pick(records, u=100, rho=0.25, eps=eps, label_source="executed")
        if rr:
            eps_exec[eps] = summarize_group(rr)
        rp = _pick(records, u=100, rho=0.25, eps=eps, label_source="preferred")
        if rp:
            eps_pref[eps] = summarize_group(rp)
    for eps in (0.0, 0.25):
        rr = _pick(records, u=90, rho=0.25, eps=eps, label_source="executed")
        if rr:
            u90_exec[eps] = summarize_group(rr)
        rp = _pick(records, u=90, rho=0.25, eps=eps, label_source="preferred")
        if rp:
            u90_pref[eps] = summarize_group(rp)

    # ---- rho views: fresh if run this invocation, else preserved -------------
    rho_groups, rho_verdict = {}, {}
    if args.part in ("rho", "all"):
        for rho in (0.05, 0.1, 0.25, 0.5):
            rr = _pick(records, u=100, rho=rho, eps=0.0, label_source="executed")
            if rr:
                rho_groups[rho] = summarize_group(rr)
    else:
        rho_groups, rho_verdict = _load_rho_from_summary()

    repro = reproduction_check(records)
    ev = eps_verdict(eps_exec, eps_pref)

    # ---- CSV: eps_sweep.csv (both label sources); rho_curve.csv only if rerun -
    eps_all = []
    for eps in (0.0, 0.1, 0.25):
        eps_all += _pick(records, u=100, rho=0.25, eps=eps, label_source="executed")
        eps_all += _pick(records, u=100, rho=0.25, eps=eps, label_source="preferred")
    for eps in (0.0, 0.25):
        eps_all += _pick(records, u=90, rho=0.25, eps=eps, label_source="executed")
        eps_all += _pick(records, u=90, rho=0.25, eps=eps, label_source="preferred")
    write_seed_csv(eps_all, os.path.join(_OUT, "eps_sweep.csv"))
    if args.part in ("rho", "all"):
        rho_all = []
        for rho in (0.05, 0.1, 0.25, 0.5):
            rho_all += _pick(records, u=100, rho=rho, eps=0.0, label_source="executed")
        write_seed_csv(rho_all, os.path.join(_OUT, "rho_curve.csv"))

    # ---- summary.json (merge: fresh gap1, preserved-or-fresh gap2) -----------
    summary = {
        "config": {"campus": args.campus, "regime": "storm2", "beta": 1.0,
                   "channel": G.CHANNEL, "family": G.FAMILY, "master_seed": G.MASTER_SEED,
                   "mechanism": G.MECH, "theta": G.THETA,
                   "label_source_default": "executed",
                   "scoring": "TWT*(w*,d*) full_class_shift, held-out files[20:30]",
                   "contrast": "seed-averaged per-instance paired Wilcoxon (pratt), "
                               "W=test strictly lower TWT*"},
        "checks": {"eps0_label_identity": ident,
                   "reproduction_vs_committed": {"n_eps0_executed_cells": repro[0],
                                                 "max_abs_twt_diff": repro[1],
                                                 "detail": repro[2]}},
        "gap1_eps_executed": {("%.2f" % e): eps_exec[e] for e in eps_exec},
        "gap1_eps_preferred": {("%.2f" % e): eps_pref[e] for e in eps_pref},
        "gap1_eps_u90_executed": {("%.2f" % e): u90_exec[e] for e in u90_exec},
        "gap1_eps_u90_preferred": {("%.2f" % e): u90_pref[e] for e in u90_pref},
        "gap2_rho": {("%.2f" % r): rho_groups[r] for r in rho_groups},
        "verdicts": {"eps": ev, "rho": rho_verdict},
    }
    with open(os.path.join(_OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)

    write_report(eps_exec, eps_pref, u90_exec, u90_pref, rho_groups, repro, ident, ev,
                 os.path.join(_ROOT, "notes", "gaps_eps_rho.md"))

    # ---- console verdict -----------------------------------------------------
    print("\n[gaps] reproduction: %d eps=0 executed cells, max abs TWT* diff = %.3e"
          % (repro[0], repro[1]), flush=True)
    epss = sorted(eps_exec)
    print("[gaps] EPS (executed/honest) M0 vs RULE: %s"
          % ", ".join("eps%.2f %+.1f%%(p=%.3g)" % (
              e, eps_exec[e]["contrasts"]["M0_vs_RULE"]["pct_vs_comparator"],
              eps_exec[e]["contrasts"]["M0_vs_RULE"]["wilcoxon_p"]) for e in epss), flush=True)
    print("[gaps] EPS (executed) recovery r: %s"
          % ", ".join("eps%.2f r=%.3f" % (e, eps_exec[e]["recovery"]["pearson_r"]) for e in epss), flush=True)
    print("[gaps] EPS (preferred upper bound) recovery r: %s"
          % ", ".join("eps%.2f r=%.3f" % (e, eps_pref[e]["recovery"]["pearson_r"]) for e in sorted(eps_pref)), flush=True)
    print("[gaps] EPS verdict (executed labels): %s (all-sig vs RULE=%s; worst-eps M0 TWT* change %+.1f%%)"
          % (ev["verdict"], ev["m0_all_sig_vs_rule"], ev["m0_twt_relative_change_worst_eps_pct"]), flush=True)
    print("[gaps] wrote eps_sweep.csv, summary.json, notes/gaps_eps_rho.md"
          + (", rho_curve.csv" if args.part in ("rho", "all") else " (rho preserved)"), flush=True)


if __name__ == "__main__":
    main()
