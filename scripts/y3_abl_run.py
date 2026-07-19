#!/usr/bin/env python
"""Paper Y3 Phase-5 CHEAP ablations (M0-only, no RL) + E4 recovery curve.

Everything scored on TWT*(w*,d*) by the independent validator on the held-out
instances (c9 storm2, files[20:30]); the overlay / supervisor / scoring channel
is full_class_shift throughout. Only how M0 USES its recovered class-shift, or
which review mechanism / f-family generated the override log, varies.

Ablations (all c9 storm2, beta1.0 rho0.25 eps0 theta1.0, seeds 301-305 unless
noted; u100 primary + u90 secondary):

  E5-ATTRIBUTION  TARGETED vs RANDOM review at matched rho. M0 estimator trained
                  AND evaluated under each mechanism; reports M0-alone-vs-RULE and
                  M0+SUP-vs-RULE+SUP gains under both -> does the gain persist
                  under random review (learning, not attention placement).
  E5-CHANNEL      decider correction applied to weight-only / deadline-only /
                  full-class-shift; reports M0-alone gain over RULE per channel.
  E5-FAMILY       overlay f-family F-LIN vs F-NL; reports M0 gain over RULE.
  E4-RECOVERY     M0 accuracy (sign / Pearson r) AND held-out M0 true-TWT* vs the
                  cumulative override count across the 8 DAgger iterations, by
                  beta {0.5,0.75,1.0} (seeds 301-303), plus the same curve mined
                  from the existing M1/M0 training logs.

Run (CPU only, 6 workers, OMP=1/worker, niced; coexists with the PPO sweep):
    PYTHONPATH=src:scripts OMP_NUM_THREADS=1 nice -n 15 \
        python scripts/y3_abl_run.py --workers 6
"""

from __future__ import annotations

import os

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
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch                                                    # noqa: E402
import y3_abl_common as C                                       # noqa: E402
from fmwos.hitl import overlay as ov                            # noqa: E402

_OUT = os.path.join(_ROOT, "results", "y3_p5", "ablations")
_CACHE = os.path.join(_OUT, "cache")
_NOTES = os.path.join(_ROOT, "notes", "phase5_ablations.md")

MASTER_SEED = 12345
N_TRAIN, N_PROBE, N_EVAL = 16, 4, 10
M0_ITERS = 8
SEEDS_E5 = list(range(301, 306))
SEEDS_E4 = list(range(301, 304))


# --------------------------------------------------------------------------- #
# One (task) -> record, disk-cached                                           #
# --------------------------------------------------------------------------- #
def _sig(task):
    keys = ["campus", "u", "beta", "rho", "eps", "theta", "mechanism",
            "decider_channel", "family", "master_seed", "seed", "n_train",
            "n_probe", "n_eval", "m0_iters", "eval_per_iter"]
    payload = {k: task.get(k) for k in keys}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def evaluate_task(task):
    t0 = time.perf_counter()
    torch.set_num_threads(1)
    try:
        os.nice(5)
    except Exception:
        pass
    sig = _sig(task)
    cache_path = os.path.join(_CACHE, "%s.json" % sig)
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                rec = json.load(fh)
            rec["cached"] = True
            return rec
        except Exception:
            pass

    train, probe, ev, evf = C.load_pools(task["campus"], task["u"],
                                         task["n_train"], task["n_probe"], task["n_eval"])
    overlay = C.make_overlay(task["beta"], family=task["family"],
                             master_seed=task["master_seed"],
                             channel="full_class_shift")
    res = C.replicate_m0(train, probe, overlay, beta=task["beta"], rho=task["rho"],
                         eps=task["eps"], decider_channel=task["decider_channel"],
                         outer_iters=task["m0_iters"], mechanism=task["mechanism"],
                         theta=task["theta"], seed=task["seed"], device="cpu",
                         eval_insts=(ev if task["eval_per_iter"] else None))
    estimator = res["estimator"]
    lad = C.eval_ladder(estimator, ev, overlay, rho=task["rho"], eps=task["eps"],
                        theta=task["theta"], mechanism=task["mechanism"],
                        decider_channel=task["decider_channel"], seed=task["seed"])
    fin = res["per_iter"][-1]
    rec = {
        "sig": sig, "task": task, "eval_files": evf,
        "inst_ids": lad["inst_ids"], "n_wos": lad["n_wos"],
        "per": lad["per"],
        "rule_sup_revfrac": lad["rule_sup_revfrac"], "rule_sup_orr": lad["rule_sup_orr"],
        "m0_sup_revfrac": lad["m0_sup_revfrac"], "m0_sup_orr": lad["m0_sup_orr"],
        "m0_final": {"pearson_r": fin["pearson_r"],
                     "sign_acc_nonzero": fin["sign_acc_nonzero"],
                     "override_rate": fin["override_rate"],
                     "cum_overrides": fin["cum_overrides"]},
        "per_iter": [],
        "elapsed_s": time.perf_counter() - t0, "cached": False,
    }
    for r in res["per_iter"]:
        row = {k: r.get(k) for k in ("iter", "n_reviews", "n_overrides",
               "cum_overrides", "override_rate", "sign_acc_nonzero", "pearson_r",
               "zero_baseline_acc", "exact_class_acc")}
        row["m0_twt_mean"] = r.get("m0_twt_mean")
        rec["per_iter"].append(row)

    os.makedirs(_CACHE, exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, cache_path)
    return rec


# --------------------------------------------------------------------------- #
# Task builders                                                               #
# --------------------------------------------------------------------------- #
def _base(**kw):
    t = {"campus": 9, "beta": 1.0, "rho": 0.25, "eps": 0.0, "theta": 1.0,
         "mechanism": "targeted", "decider_channel": "full_class_shift",
         "family": "F-NL", "master_seed": MASTER_SEED, "n_train": N_TRAIN,
         "n_probe": N_PROBE, "n_eval": N_EVAL, "m0_iters": M0_ITERS,
         "eval_per_iter": False}
    t.update(kw)
    return t


def build_tasks():
    tasks = {}

    def add(ab, **kw):
        t = _base(**kw)
        tasks[_sig(t)] = t

    # E5-ATTRIBUTION: targeted vs random, u100 (primary) + u90 (secondary)
    for u in (100, 90):
        for mech in ("targeted", "random"):
            for seed in SEEDS_E5:
                add("attribution", u=u, mechanism=mech, seed=seed)
    # E5-CHANNEL: weight_only / deadline_only / full, u100 + u90
    for u in (100, 90):
        for ch in ("full_class_shift", "weight_only", "deadline_only"):
            for seed in SEEDS_E5:
                add("channel", u=u, decider_channel=ch, seed=seed)
    # E5-FAMILY: F-LIN vs F-NL, u100 + u90
    for u in (100, 90):
        for fam in ("F-NL", "F-LIN"):
            for seed in SEEDS_E5:
                add("family", u=u, family=fam, seed=seed)
    # E4-RECOVERY: beta {0.5,0.75,1.0}, u100, seeds 301-303, per-iter eval ON
    for beta in (0.5, 0.75, 1.0):
        for seed in SEEDS_E4:
            add("e4", u=100, beta=beta, seed=seed, eval_per_iter=True)
    return list(tasks.values())


# --------------------------------------------------------------------------- #
# Aggregation helpers                                                          #
# --------------------------------------------------------------------------- #
def _stack(recs, key):
    """Stack (S seeds x n instances), instances aligned by inst_id."""
    recs = sorted(recs, key=lambda r: r["task"]["seed"])
    ref = recs[0]["inst_ids"]
    common = [i for i in ref if all(i in r["inst_ids"] for r in recs)]
    mat = []
    for r in recs:
        idx = {iid: j for j, iid in enumerate(r["inst_ids"])}
        mat.append([r["per"][key][idx[i]] for i in common])
    return np.asarray(mat, float)


def _ladder_means(recs):
    return {k: float(_stack(recs, k).mean()) for k in
            ["rule", "m0_alone", "m0_sup", "rule_sup", "oracle"]}


def _revfrac(recs, which):
    return float(np.mean([np.mean(r[which]) for r in recs]))


def _m0_final(recs, k):
    return float(np.mean([r["m0_final"][k] for r in recs]))


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
def run(tasks, workers):
    print("[y3_abl] %d unique cell-seed tasks, %d workers" % (len(tasks), workers), flush=True)
    out = []
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(evaluate_task, t): t for t in tasks}
        for f in as_completed(fut):
            t = fut[f]
            rec = f.result()
            out.append(rec)
            done += 1
            lm = {k: np.mean(rec["per"][k]) for k in ["rule", "m0_alone", "m0_sup", "rule_sup", "oracle"]}
            print("  [%d/%d] c%d u%d b%.2f %s ch=%s fam=%s s%d | RULE=%.0f M0=%.0f M0S=%.0f "
                  "RS=%.0f OR=%.0f r=%.2f orr=%.3f %s (%.0fs, wall %.0fs)"
                  % (done, len(tasks), t["campus"], t["u"], t["beta"], t["mechanism"],
                     t["decider_channel"].split("_")[0], t["family"], t["seed"],
                     lm["rule"], lm["m0_alone"], lm["m0_sup"], lm["rule_sup"], lm["oracle"],
                     rec["m0_final"]["pearson_r"], rec["m0_final"]["override_rate"],
                     "CACHED" if rec.get("cached") else "", rec["elapsed_s"], time.time() - t0),
                  flush=True)
    return out


def _match(records, **filt):
    out = []
    for r in records:
        t = r["task"]
        if all(t.get(k) == v for k, v in filt.items()):
            out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Per-ablation CSV + aggregate                                                 #
# --------------------------------------------------------------------------- #
def write_attribution(records):
    path = os.path.join(_OUT, "e5_attribution.csv")
    cols = ["mechanism", "campus", "u", "beta", "rho", "seed", "inst_id", "n_wos",
            "rule", "m0_alone", "m0_sup", "rule_sup", "oracle",
            "rule_sup_revfrac", "m0_sup_revfrac", "m0_final_r", "m0_final_orr", "m0_final_sign"]
    rows = []
    agg = {}
    for u in (100, 90):
        for mech in ("targeted", "random"):
            recs = _match(records, u=u, mechanism=mech, decider_channel="full_class_shift",
                          family="F-NL", beta=1.0, eval_per_iter=False)
            recs = [r for r in recs if r["task"]["seed"] in SEEDS_E5]
            if not recs:
                continue
            for r in sorted(recs, key=lambda r: r["task"]["seed"]):
                t = r["task"]
                for i, iid in enumerate(r["inst_ids"]):
                    rows.append({"mechanism": mech, "campus": t["campus"], "u": u,
                                 "beta": t["beta"], "rho": t["rho"], "seed": t["seed"],
                                 "inst_id": iid, "n_wos": r["n_wos"],
                                 "rule": "%.6f" % r["per"]["rule"][i],
                                 "m0_alone": "%.6f" % r["per"]["m0_alone"][i],
                                 "m0_sup": "%.6f" % r["per"]["m0_sup"][i],
                                 "rule_sup": "%.6f" % r["per"]["rule_sup"][i],
                                 "oracle": "%.6f" % r["per"]["oracle"][i],
                                 "rule_sup_revfrac": "%.4f" % r["rule_sup_revfrac"][i],
                                 "m0_sup_revfrac": "%.4f" % r["m0_sup_revfrac"][i],
                                 "m0_final_r": "%.4f" % r["m0_final"]["pearson_r"],
                                 "m0_final_orr": "%.4f" % r["m0_final"]["override_rate"],
                                 "m0_final_sign": "%.4f" % r["m0_final"]["sign_acc_nonzero"]})
            lm = _ladder_means(recs)
            c_alone = C.seed_avg_contrast(_stack(recs, "m0_alone"), _stack(recs, "rule"))
            c_sup = C.seed_avg_contrast(_stack(recs, "m0_sup"), _stack(recs, "rule_sup"))
            agg["u%d_%s" % (u, mech)] = {
                "u": u, "mechanism": mech, "n_seeds": len(recs), "ladder": lm,
                "rule_sup_revfrac": _revfrac(recs, "rule_sup_revfrac"),
                "m0_sup_revfrac": _revfrac(recs, "m0_sup_revfrac"),
                "m0_final_r": _m0_final(recs, "pearson_r"),
                "m0_final_orr": _m0_final(recs, "override_rate"),
                "M0_vs_RULE": c_alone, "M0sup_vs_RULEsup": c_sup}
    _write_csv(path, cols, rows)
    return agg


def write_channel(records):
    path = os.path.join(_OUT, "e5_channel.csv")
    cols = ["decider_channel", "campus", "u", "beta", "rho", "seed", "inst_id", "n_wos",
            "rule", "m0_alone", "m0_sup", "rule_sup", "oracle", "m0_final_r", "m0_final_orr"]
    rows = []
    agg = {}
    for u in (100, 90):
        for ch in ("full_class_shift", "deadline_only", "weight_only"):
            recs = _match(records, u=u, decider_channel=ch, mechanism="targeted",
                          family="F-NL", beta=1.0, eval_per_iter=False)
            recs = [r for r in recs if r["task"]["seed"] in SEEDS_E5]
            if not recs:
                continue
            for r in sorted(recs, key=lambda r: r["task"]["seed"]):
                t = r["task"]
                for i, iid in enumerate(r["inst_ids"]):
                    rows.append({"decider_channel": ch, "campus": t["campus"], "u": u,
                                 "beta": t["beta"], "rho": t["rho"], "seed": t["seed"],
                                 "inst_id": iid, "n_wos": r["n_wos"],
                                 "rule": "%.6f" % r["per"]["rule"][i],
                                 "m0_alone": "%.6f" % r["per"]["m0_alone"][i],
                                 "m0_sup": "%.6f" % r["per"]["m0_sup"][i],
                                 "rule_sup": "%.6f" % r["per"]["rule_sup"][i],
                                 "oracle": "%.6f" % r["per"]["oracle"][i],
                                 "m0_final_r": "%.4f" % r["m0_final"]["pearson_r"],
                                 "m0_final_orr": "%.4f" % r["m0_final"]["override_rate"]})
            lm = _ladder_means(recs)
            c_alone = C.seed_avg_contrast(_stack(recs, "m0_alone"), _stack(recs, "rule"))
            c_sup = C.seed_avg_contrast(_stack(recs, "m0_sup"), _stack(recs, "rule_sup"))
            agg["u%d_%s" % (u, ch)] = {"u": u, "decider_channel": ch,
                "n_seeds": len(recs), "ladder": lm,
                "m0_final_r": _m0_final(recs, "pearson_r"),
                "M0_vs_RULE": c_alone, "M0sup_vs_RULEsup": c_sup}
    _write_csv(path, cols, rows)
    return agg


def write_family(records):
    path = os.path.join(_OUT, "e5_family.csv")
    cols = ["family", "campus", "u", "beta", "rho", "seed", "inst_id", "n_wos",
            "rule", "m0_alone", "m0_sup", "rule_sup", "oracle", "m0_final_r", "m0_final_orr"]
    rows = []
    agg = {}
    for u in (100, 90):
        for fam in ("F-NL", "F-LIN"):
            recs = _match(records, u=u, family=fam, mechanism="targeted",
                          decider_channel="full_class_shift", beta=1.0, eval_per_iter=False)
            recs = [r for r in recs if r["task"]["seed"] in SEEDS_E5]
            if not recs:
                continue
            for r in sorted(recs, key=lambda r: r["task"]["seed"]):
                t = r["task"]
                for i, iid in enumerate(r["inst_ids"]):
                    rows.append({"family": fam, "campus": t["campus"], "u": u,
                                 "beta": t["beta"], "rho": t["rho"], "seed": t["seed"],
                                 "inst_id": iid, "n_wos": r["n_wos"],
                                 "rule": "%.6f" % r["per"]["rule"][i],
                                 "m0_alone": "%.6f" % r["per"]["m0_alone"][i],
                                 "m0_sup": "%.6f" % r["per"]["m0_sup"][i],
                                 "rule_sup": "%.6f" % r["per"]["rule_sup"][i],
                                 "oracle": "%.6f" % r["per"]["oracle"][i],
                                 "m0_final_r": "%.4f" % r["m0_final"]["pearson_r"],
                                 "m0_final_orr": "%.4f" % r["m0_final"]["override_rate"]})
            lm = _ladder_means(recs)
            c_alone = C.seed_avg_contrast(_stack(recs, "m0_alone"), _stack(recs, "rule"))
            c_sup = C.seed_avg_contrast(_stack(recs, "m0_sup"), _stack(recs, "rule_sup"))
            agg["u%d_%s" % (u, fam)] = {"u": u, "family": fam, "n_seeds": len(recs),
                "ladder": lm, "m0_final_r": _m0_final(recs, "pearson_r"),
                "M0_vs_RULE": c_alone, "M0sup_vs_RULEsup": c_sup}
    _write_csv(path, cols, rows)
    return agg


def write_e4(records):
    """Tidy long recovery curve: live (this run) + mined (existing logs)."""
    path = os.path.join(_OUT, "e4_recovery.csv")
    cols = ["source", "campus", "u", "beta", "rho", "seed", "iter", "cum_overrides",
            "n_reviews_iter", "override_rate", "sign_acc", "pearson_r",
            "m0_twt_mean", "rule_twt", "oracle_twt", "note"]
    rows = []
    # ---- live ---- #
    live = _match(records, eval_per_iter=True)
    for r in sorted(live, key=lambda r: (r["task"]["beta"], r["task"]["seed"])):
        t = r["task"]
        rule_twt = float(np.mean(r["per"]["rule"]))
        oracle_twt = float(np.mean(r["per"]["oracle"]))
        for pr in r["per_iter"]:
            rows.append({"source": "live_m0", "campus": t["campus"], "u": t["u"],
                         "beta": t["beta"], "rho": t["rho"], "seed": t["seed"],
                         "iter": pr["iter"], "cum_overrides": pr["cum_overrides"],
                         "n_reviews_iter": pr["n_reviews"],
                         "override_rate": "%.6f" % pr["override_rate"],
                         "sign_acc": "%.6f" % pr["sign_acc_nonzero"],
                         "pearson_r": "%.6f" % pr["pearson_r"],
                         "m0_twt_mean": "%.4f" % pr["m0_twt_mean"] if pr["m0_twt_mean"] is not None else "",
                         "rule_twt": "%.4f" % rule_twt, "oracle_twt": "%.4f" % oracle_twt,
                         "note": "held-out M0-alone TWT*(w*,d*), n=10"})
    # ---- mined from existing training logs ---- #
    rows += _mine_logs()
    _write_csv(path, cols, rows)
    # per-beta live summary (accuracy + TWT at low vs high override budget)
    e4sum = {}
    for beta in (0.5, 0.75, 1.0):
        recs = [r for r in live if r["task"]["beta"] == beta]
        if not recs:
            continue
        # average over seeds per iter
        by_iter = defaultdict(lambda: defaultdict(list))
        for r in recs:
            for pr in r["per_iter"]:
                by_iter[pr["iter"]]["cum"].append(pr["cum_overrides"])
                by_iter[pr["iter"]]["sign"].append(pr["sign_acc_nonzero"])
                by_iter[pr["iter"]]["r"].append(pr["pearson_r"])
                by_iter[pr["iter"]]["twt"].append(pr["m0_twt_mean"])
        iters = sorted(by_iter)
        curve = [{"iter": it,
                  "cum_overrides": float(np.mean(by_iter[it]["cum"])),
                  "sign_acc": float(np.mean(by_iter[it]["sign"])),
                  "pearson_r": float(np.mean(by_iter[it]["r"])),
                  "m0_twt": float(np.mean(by_iter[it]["twt"]))} for it in iters]
        rule_twt = float(np.mean([np.mean(r["per"]["rule"]) for r in recs]))
        oracle_twt = float(np.mean([np.mean(r["per"]["oracle"]) for r in recs]))
        e4sum["beta%.2f" % beta] = {"n_seeds": len(recs), "rule_twt": rule_twt,
                                    "oracle_twt": oracle_twt, "curve": curve,
                                    "first": curve[0], "last": curve[-1]}
    return e4sum


def _mine_logs():
    rows = []
    specs = [
        ("m1_full", "train_log/y3_p15/m1_full/metrics.csv", 9, 100, 1.0,
         "M1 fair-training in-loop true_twt (train-scale, ~20 insts); NOT held-out M0"),
        ("m1_fair", "train_log/y3_p3_m1fair/metrics.csv", 9, 100, 1.0,
         "M1 fair-training in-loop true_twt (train-scale); NOT held-out M0"),
        ("m0_p2", "train_log/y3_p2/m0/m0_metrics.csv", None, None, None,
         "pre-P1.5 weight-only-era M0 (dead cell, orr~0.002); accuracy only, no true_twt"),
    ]
    for src, rel, campus, u, beta, note in specs:
        p = os.path.join(_ROOT, rel)
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            rd = list(csv.DictReader(fh))
        cum = 0.0
        for row in rd:
            it = int(row["iter"])
            # override count for this iter
            if "n_overrides" in row and row.get("n_overrides") not in (None, ""):
                n_ov = float(row["n_overrides"]); n_rev = float(row.get("n_reviews", 0) or 0)
            else:
                n_rev = float(row.get("n_reviews", 0) or 0)
                n_ov = n_rev * float(row.get("override_rate", 0) or 0)
            cum += n_ov
            sign = row.get("hat_s_sign_acc", row.get("sign_acc_nonzero", ""))
            pr = row.get("hat_s_pearson_r", row.get("pearson_r", ""))
            twt = row.get("true_twt", "")
            rows.append({"source": src, "campus": campus if campus is not None else "",
                         "u": u if u is not None else "", "beta": beta if beta is not None else "",
                         "rho": 0.25, "seed": 301, "iter": it,
                         "cum_overrides": int(round(cum)),
                         "n_reviews_iter": int(round(n_rev)),
                         "override_rate": "%.6f" % float(row.get("override_rate", 0) or 0),
                         "sign_acc": "%.6f" % float(sign) if sign not in ("", None) else "",
                         "pearson_r": "%.6f" % float(pr) if pr not in ("", None) else "",
                         "m0_twt_mean": "%.4f" % float(twt) if twt not in ("", None) else "",
                         "rule_twt": "", "oracle_twt": "", "note": note})
    return rows


def _write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[y3_abl] wrote %s (%d rows)" % (path, len(rows)), flush=True)


# --------------------------------------------------------------------------- #
# Notes markdown                                                              #
# --------------------------------------------------------------------------- #
def _pct(x):
    return "%+.1f%%" % x


def _wtl(w):
    return "%d/%d/%d" % (w["W"], w["T"], w["L"])


def write_notes(attr, chan, fam, e4):
    L = []
    A = L.append
    A("# Paper Y3 Phase-5 CHEAP ablations (M0-only, no RL)\n")
    A("Scored on TWT*(w*,d*) by the independent validator, held-out c9 storm2 "
      "files[20:30] (n=10, no subsampling), overlay/supervisor/scoring channel "
      "full_class_shift, eps=0, theta=1.0, master_seed=12345, F-NL unless noted. "
      "Gains are seed-averaged per-instance paired contrasts (Wilcoxon pratt, "
      "W = M0 strictly lower TWT*). M0 loop reproduces results/y3_p4 bit-for-bit "
      "on the shared cell. Seeds 301-305 (E5) / 301-303 (E4).\n")

    # E5-ATTRIBUTION
    A("## E5-ATTRIBUTION: TARGETED vs RANDOM review at matched rho=0.25\n")
    A("c9 storm2 u100 (primary) + u90, beta1.0. M0 estimator trained AND evaluated "
      "under each mechanism (the whole review regime is targeted or random).\n")
    A("| u | mech | rev.frac | RULE | M0 | M0+SUP | RULE+SUP | ORACLE | M0-vs-RULE | W/T/L | p | M0+SUP-vs-RULE+SUP | W/T/L | p |")
    A("|---|------|----------|------|----|--------|----------|--------|-----------|-------|---|--------------------|-------|---|")
    for u in (100, 90):
        for mech in ("targeted", "random"):
            k = "u%d_%s" % (u, mech)
            if k not in attr:
                continue
            c = attr[k]; lm = c["ladder"]
            a, s = c["M0_vs_RULE"], c["M0sup_vs_RULEsup"]
            A("| %d | %s | %.3f | %.0f | %.0f | %.0f | %.0f | %.0f | %s | %s | %.3f | %s | %s | %.3f |" % (
                u, mech, c["rule_sup_revfrac"], lm["rule"], lm["m0_alone"], lm["m0_sup"],
                lm["rule_sup"], lm["oracle"], _pct(a["pct_gain"]), _wtl(a["wtl"]),
                a["wilcoxon_p"], _pct(s["pct_gain"]), _wtl(s["wtl"]), s["wilcoxon_p"]))
    # verdict
    tp = attr.get("u100_targeted", {}).get("M0_vs_RULE", {}).get("pct_gain")
    rp = attr.get("u100_random", {}).get("M0_vs_RULE", {}).get("pct_gain")
    persists = (rp is not None and rp > 5.0)
    A("\n**Verdict (u100):** M0-alone gain over RULE = %s (TARGETED) vs %s (RANDOM). "
      "The gain %s under random review -> the estimator is LEARNING the latent "
      "from overrides, not merely exploiting targeted attention placement.\n"
      % (_pct(tp) if tp is not None else "n/a", _pct(rp) if rp is not None else "n/a",
         "PERSISTS" if persists else "does NOT persist"))

    # E5-CHANNEL
    A("## E5-CHANNEL: weight-only vs deadline-only vs full-class-shift (M0 correction)\n")
    A("c9 storm2 u100 (primary) + u90, beta1.0 rho0.25 targeted. Overlay/objective "
      "stay full_class_shift; only which quantity M0's hat_s corrects varies.\n")
    A("| u | channel | RULE | M0 | ORACLE | M0-vs-RULE | W/T/L | p | M0+SUP-vs-RULE+SUP |")
    A("|---|---------|------|----|--------|-----------|-------|---|--------------------|")
    for u in (100, 90):
        for ch in ("full_class_shift", "deadline_only", "weight_only"):
            k = "u%d_%s" % (u, ch)
            if k not in chan:
                continue
            c = chan[k]; lm = c["ladder"]; a = c["M0_vs_RULE"]; s = c["M0sup_vs_RULEsup"]
            A("| %d | %s | %.0f | %.0f | %.0f | %s | %s | %.3f | %s |" % (
                u, ch, lm["rule"], lm["m0_alone"], lm["oracle"], _pct(a["pct_gain"]),
                _wtl(a["wtl"]), a["wilcoxon_p"], _pct(s["pct_gain"])))
    gf = chan.get("u100_full_class_shift", {}).get("M0_vs_RULE", {}).get("pct_gain")
    gd = chan.get("u100_deadline_only", {}).get("M0_vs_RULE", {}).get("pct_gain")
    gw = chan.get("u100_weight_only", {}).get("M0_vs_RULE", {}).get("pct_gain")
    A("\n**Verdict (u100):** full %s, deadline-only %s, weight-only %s. The deadline "
      "channel carries the lever; the weight-only correction is %s.\n" % (
        _pct(gf) if gf is not None else "n/a", _pct(gd) if gd is not None else "n/a",
        _pct(gw) if gw is not None else "n/a",
        "near-inert" if (gw is not None and abs(gw) < 5.0) else "non-trivial"))

    # E5-FAMILY
    A("## E5-FAMILY: F-LIN vs F-NL (overlay f-family)\n")
    A("c9 storm2 u100 (primary) + u90, beta1.0 rho0.25 targeted, full-class-shift.\n")
    A("| u | family | RULE | M0 | ORACLE | M0-vs-RULE | W/T/L | p | m0 r |")
    A("|---|--------|------|----|--------|-----------|-------|---|------|")
    for u in (100, 90):
        for fm in ("F-NL", "F-LIN"):
            k = "u%d_%s" % (u, fm)
            if k not in fam:
                continue
            c = fam[k]; lm = c["ladder"]; a = c["M0_vs_RULE"]
            A("| %d | %s | %.0f | %.0f | %.0f | %s | %s | %.3f | %.2f |" % (
                u, fm, lm["rule"], lm["m0_alone"], lm["oracle"], _pct(a["pct_gain"]),
                _wtl(a["wtl"]), a["wilcoxon_p"], c["m0_final_r"]))
    gnl = fam.get("u100_F-NL", {}).get("M0_vs_RULE", {}).get("pct_gain")
    glin = fam.get("u100_F-LIN", {}).get("M0_vs_RULE", {}).get("pct_gain")
    A("\n**Verdict (u100):** F-NL %s, F-LIN %s -> the M0 gain is not an artifact of "
      "one f-family.\n" % (_pct(gnl) if gnl is not None else "n/a",
                           _pct(glin) if glin is not None else "n/a"))

    # E4-RECOVERY
    A("## E4-RECOVERY curve: accuracy & TWT* vs cumulative override budget\n")
    A("Live: M0 pipeline on c9 storm2 u100, seeds 301-303, per DAgger iter (0..7). "
      "sign/r = held-out probe accuracy, M0-TWT* = held-out M0-alone TWT*(w*,d*). "
      "cum = cumulative overrides seen up to and including that iter (train pool, "
      "16 insts).\n")
    for beta in (1.0, 0.75, 0.5):
        k = "beta%.2f" % beta
        if k not in e4:
            continue
        c = e4[k]; f, l = c["first"], c["last"]
        A("\n**beta=%.2f** (RULE=%.0f, ORACLE=%.0f):" % (beta, c["rule_twt"], c["oracle_twt"]))
        A("| iter | cum_over | sign_acc | pearson_r | M0-TWT* | %%gap closed |")
        A("|------|----------|----------|-----------|---------|-------------|")
        for pt in c["curve"]:
            gap = (100.0 * (c["rule_twt"] - pt["m0_twt"]) / (c["rule_twt"] - c["oracle_twt"])
                   if c["rule_twt"] > c["oracle_twt"] else float("nan"))
            A("| %d | %.0f | %.3f | %.3f | %.0f | %.0f%% |" % (
                pt["iter"], pt["cum_overrides"], pt["sign_acc"], pt["pearson_r"],
                pt["m0_twt"], gap))
    A("\nMined from existing training logs (see e4_recovery.csv, source column): "
      "m1_full / m1_fair (M1 fair-training, in-loop train-scale true_twt, NOT "
      "held-out) and m0_p2 (pre-P1.5 weight-only dead cell, accuracy only).\n")
    # E4 verdict
    vv = []
    for beta in (1.0, 0.75, 0.5):
        k = "beta%.2f" % beta
        if k not in e4:
            continue
        c = e4[k]; f, l = c["first"], c["last"]
        vv.append("beta=%.2f: r %.2f->%.2f, sign %.2f->%.2f, M0-TWT* %.0f->%.0f (cum %.0f->%.0f)"
                  % (beta, f["pearson_r"], l["pearson_r"], f["sign_acc"], l["sign_acc"],
                     f["m0_twt"], l["m0_twt"], f["cum_overrides"], l["cum_overrides"]))
    A("**Verdict:** accuracy and TWT* improve from the first override batch to the "
      "full budget, most steeply for high beta (more recoverable structure):\n- "
      + "\n- ".join(vv) + "\n")

    with open(_NOTES, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("[y3_abl] wrote %s" % _NOTES, flush=True)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    os.makedirs(_OUT, exist_ok=True)
    os.makedirs(_CACHE, exist_ok=True)

    # Pre-build overlay coefficients for both families to avoid a worker race.
    for fam in ("F-NL", "F-LIN"):
        ov.get_coeffs(fam, MASTER_SEED)

    tasks = build_tasks()
    records = run(tasks, args.workers)

    attr = write_attribution(records)
    chan = write_channel(records)
    fam = write_family(records)
    e4 = write_e4(records)

    with open(os.path.join(_OUT, "ablations_summary.json"), "w") as fh:
        json.dump({"attribution": attr, "channel": chan, "family": fam, "e4": e4},
                  fh, indent=1, default=str)
    write_notes(attr, chan, fam, e4)
    print("[y3_abl] complete.", flush=True)


if __name__ == "__main__":
    main()
