#!/usr/bin/env python
"""W1 sweep: the deployable review policy, its references, and the routing curve.

Structure, caching, instance splits, overlay, scoring and statistics are taken
verbatim from scripts/y3_p4_m0grid.py so a result from this script is directly
comparable with the committed headline grid. The ONLY thing that changes is the
review policy, and the conformal split the deployable policy needs.

Review policies compared (``--policy`` / the ``policy`` task field):

  stability  DEPLOYABLE HEADLINE. Refer a decision when the corrected top pick
             can be overturned by an admissible shift vector inside the
             conformal band; spend rho on undetermined decisions, worst
             instability margin first. Every input is an observable.
  margin     Observable control: the published TARGETED policy with its latent
             ``_has_plus2`` clause deleted. Deployable, but blind to
             uncertainty.
  targeted   ORACLE-INFORMED UPPER REFERENCE. The published policy. Its
             consequential test reads the realized latent shift of the pending
             queue, so it is NOT deployable and appears only as a reference.
  random     Lower control (iid Bernoulli(rho)), unchanged.

Estimator-fitting protocol. ``split_fit=True`` assigns every weak-label example
a permanent fold at creation: the calibration fold is never used to fit the
estimator, in this or any later DAgger iteration, so the conformal residuals are
genuinely out-of-sample. All four policies are run under the SAME split, so the
only difference between them is review placement. ``split_fit=False`` is the
published full-aggregate protocol and is used for the reproduction anchor.

Parts
-----
pilot  Headline cell, seed 301, policy=targeted, split_fit=False. Must reproduce
       the committed per-seed record and the published \\MzeroGain. Gate.
head   Headline cell (c9 storm2 u100, beta 1.0, rho 0.25, eps 0), seeds 301-310,
       five arms: targeted_pub + {stability, margin, targeted, random} on the
       split protocol.  -> results/y3_w1/head.csv, head_summary.json
curve  Routing curve: rho in {0.02,0.05,0.10,0.25,0.50}, stability policy, c9
       headline cell and the c10 confirmation cell, seeds 301-303.
       -> results/y3_w1/curve.csv, curve_summary.json
alpha  Conformal level sweep at the headline cell, rho 0.25, seeds 301-303.
       -> results/y3_w1/alpha.csv
cov    Band coverage against the true shift per beta (EVALUATION ONLY), beta in
       {0,0.25,0.5,0.75,1.0}, headline cell, seeds 301-303.
       -> results/y3_w1/coverage.csv

Compute. The machine is shared. Run pinned, one sweep at a time:
    PYTHONPATH=src taskset -c 0-9 python scripts/y3_w1_sweep.py --part head
Threads: the parent is capped at ``--threads`` (default 4, per the work-package
brief); each forked worker is capped at ONE thread, so ``--workers`` processes
use ``--workers`` threads and never exceed the ten pinned cores. Setting four
threads per worker as well would put 32 threads on 10 cores and the jobs would
fight, which is the failure the pinning is there to prevent.
"""

from __future__ import annotations

import os

# Cap the numeric runtimes BEFORE numpy/torch import (parent and forked workers).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

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

import torch                                                        # noqa: E402

from fmwos.env import DispatchEnv                                   # noqa: E402
from fmwos.hitl import deciders as dec                              # noqa: E402
from fmwos.hitl import overlay as ov                                # noqa: E402
from fmwos.hitl import augmented_rule as AR                         # noqa: E402
from fmwos.hitl import routing as R                                 # noqa: E402
from fmwos.hitl import true_objective as TO                         # noqa: E402
from fmwos.hitl.supervisor import Supervisor                        # noqa: E402

import y3_w1_band_coverage as COV                                   # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_w1")
_CACHE = os.path.join(_OUT, "cache")

# Locked cell constants, identical to scripts/y3_p4_m0grid.py.
FAMILY = "F-NL"
MASTER_SEED = 12345
EPS = 0.0
THETA = 1.0
CHANNEL = "full_class_shift"

DECIDERS = ["rule", "m0_alone", "oracle", "rule_sup", "m0_sup"]

# The headline cell of the manuscript: campus 9, the benchmark high-load track
# at saturation, full-class-shift overlay, beta high, rho low, no noise.
HEAD = dict(campus=9, regime="storm2", u=100, beta=1.0, rho=0.25)
CONF = dict(campus=10, regime="storm2", u=100, beta=1.0, rho=0.25)

# The eight-cell contention grid the manuscript Holm-corrects each gate contrast
# across (campus 9; utilisation x recoverable share x review budget).
GRID_CELLS = [dict(campus=9, regime="storm2", u=u, beta=b, rho=r)
              for u in (90, 100) for b in (0.75, 1.0) for r in (0.25, 0.5)]


# --------------------------------------------------------------------------- #
# Instance pools (identical to y3_p4_m0grid.locate_files)                      #
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


# --------------------------------------------------------------------------- #
# Task signature / cache                                                       #
# --------------------------------------------------------------------------- #
_SIG_KEYS = ["campus", "regime", "u", "size", "beta", "rho", "eps", "theta",
             "channel", "family", "master_seed", "seed", "n_train", "n_probe",
             "n_eval", "m0_iters", "policy", "split_fit", "cal_frac", "alpha",
             "band_mode"]


def _cell_sig(task):
    payload = {k: task.get(k) for k in _SIG_KEYS}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _base_task(**kw):
    t = {"regime": "storm2", "u": None, "size": None, "eps": EPS, "theta": THETA,
         "channel": CHANNEL, "family": FAMILY, "master_seed": MASTER_SEED,
         "n_train": 16, "n_probe": 4, "n_eval": 10, "m0_iters": 8,
         "policy": "stability", "split_fit": True, "cal_frac": 0.3,
         "alpha": 0.1, "band_mode": "global"}
    t.update(kw)
    return t


# --------------------------------------------------------------------------- #
# One (cell, seed, policy) evaluation                                          #
# --------------------------------------------------------------------------- #
def evaluate_cell(task):
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
            pass

    campus, regime = task["campus"], task["regime"]
    u, size = task.get("u"), task.get("size")
    beta, rho, seed = task["beta"], task["rho"], task["seed"]
    policy = task["policy"]
    n_train, n_probe, n_eval = task["n_train"], task["n_probe"], task["n_eval"]

    files = locate_files(campus, regime, u=u, size=size)
    need = n_train + n_probe + n_eval
    if len(files) < need:
        raise RuntimeError("only %d files at c%d %s u=%s (need %d)"
                           % (len(files), campus, regime, u, need))
    train_files = files[:n_train]
    probe_files = files[n_train:n_train + n_probe]
    eval_files = files[n_train + n_probe:n_train + n_probe + n_eval]
    assert not (set(eval_files) & set(train_files) & set(probe_files))
    assert not set(eval_files) & set(train_files), "eval overlaps train"
    assert not set(eval_files) & set(probe_files), "eval overlaps probe"
    train = [_load(p) for p in train_files]
    probe = [_load(p) for p in probe_files]
    eval_insts = [_load(p) for p in eval_files]

    overlay = ov.Overlay(ov.OverlayParams(
        beta=beta, family=task["family"], master_seed=task["master_seed"],
        channel=task["channel"]))
    assert overlay.params.channel == task["channel"]

    # ---- the correction layer, trained under this review policy ------------ #
    torch.manual_seed(seed)
    np.random.seed(seed)
    res = R.run_m0_routed(train, probe, overlay,
                          beta_rho_eps=(beta, rho, task["eps"]),
                          outer_iters=task["m0_iters"], policy=policy,
                          theta=task["theta"], seed=seed, device="cpu",
                          verbose=False, split_fit=task["split_fit"],
                          cal_frac=task["cal_frac"], alpha=task["alpha"],
                          band_mode=task["band_mode"])
    estimator, band = res["estimator"], res["band"]
    m0_last = res["per_iter"][-1]

    def sup_for(inst, applied):
        bm = None
        if policy == "stability" and band is not None:
            bm = R.band_for_instance(estimator, inst, band, channel=task["channel"])
        return R.make_supervisor(policy, overlay, inst, rho, epsilon=task["eps"],
                                 theta=task["theta"], seed=seed, applied=applied,
                                 band_map=bm, channel=task["channel"])

    # ---- held-out per-instance TWT* for every decider ---------------------- #
    per = {k: [] for k in DECIDERS}
    inst_ids = []
    rt = defaultdict(list)                     # routing telemetry, m0_sup arm
    rt_rule = defaultdict(list)                # routing telemetry, rule_sup arm
    verdict_rows = []
    eval_logs = []
    for inst in eval_insts:
        applied = overlay.apply(inst)
        inst_ids.append(inst["meta"]["id"])

        def sc(sched):
            return TO.score_true(inst, sched, overlay, applied)["TWT_true"]

        per["rule"].append(sc(dec.run_rule(DispatchEnv(inst), "atc", seed=seed)))
        m0d = AR.augmented_atc_decider(estimator, inst, channel=task["channel"])
        m0_sched, _ = DispatchEnv(inst).run_supervised(m0d, supervisor=None,
                                                       method="m0", seed=seed)
        per["m0_alone"].append(sc(m0_sched))
        osup = Supervisor(overlay, inst, rho=0.0, applied=applied)
        per["oracle"].append(sc(dec.run_oracle_greedy(DispatchEnv(inst), osup,
                                                      seed=seed)))

        rsup = sup_for(inst, applied)
        rsched, _ = dec.run_rule_sup(DispatchEnv(inst), "atc", rsup, seed=seed)
        per["rule_sup"].append(sc(rsched))
        for k, v in R.routing_summary_of(rsup).items():
            if isinstance(v, (int, float)):
                rt_rule[k].append(float(v))

        m0sup = sup_for(inst, applied)
        m0d2 = AR.augmented_atc_decider(estimator, inst, channel=task["channel"])
        m0s_sched, log = DispatchEnv(inst).run_supervised(
            m0d2, supervisor=m0sup, method="m0_sup", seed=seed)
        per["m0_sup"].append(sc(m0s_sched))
        for k, v in R.routing_summary_of(m0sup).items():
            if isinstance(v, (int, float)):
                rt[k].append(float(v))
        eval_logs.append((log, inst))

    # ---- automate/refer verdicts with NO budget cap (the referral demand) --- #
    verdict = {}
    if band is not None:
        cov_a = ref_a = 0
        for inst in eval_insts[:3]:
            vs = R.verdict_stream(estimator, inst, band, channel=task["channel"],
                                  seed=seed, max_records=8)
            cov_a += vs["counts"]["automate"]; ref_a += vs["counts"]["refer"]
            verdict_rows.extend(vs["records"][:4])
        verdict = {"automate": cov_a, "refer": ref_a,
                   "automation_coverage_unbudgeted":
                       cov_a / (cov_a + ref_a) if (cov_a + ref_a) else 1.0}

    # ---- band coverage: the QUARANTINED evaluation-only latent read -------- #
    coverage = {}
    if band is not None:
        coverage.update(COV.true_shift_coverage(estimator, band, eval_insts,
                                                overlay))
        coverage.update(COV.weak_label_coverage(estimator, band, eval_logs))
        coverage["band_q"] = band.q
        coverage["band_n_cal"] = band.n_cal

    def _m(d, k):
        return float(np.mean(d[k])) if d.get(k) else float("nan")

    rec = {
        "sig": sig, "campus": campus, "regime": regime, "u": u, "size": size,
        "beta": beta, "rho": rho, "eps": task["eps"], "seed": seed,
        "policy": policy, "split_fit": task["split_fit"],
        "alpha": task["alpha"], "cal_frac": task["cal_frac"],
        "band_mode": task["band_mode"], "channel": task["channel"],
        "n_train": n_train, "n_probe": n_probe, "n_eval": len(inst_ids),
        "inst_ids": inst_ids, "n_wos": len(eval_insts[0]["work_orders"]),
        "per": {k: [float(x) for x in per[k]] for k in DECIDERS},
        "m0_sup_revfrac": [float(x) for x in rt["reviewed_fraction"]],
        "m0_sup_orr": [float(x) for x in rt["override_rate_of_reviews"]],
        "rule_sup_revfrac": [float(x) for x in rt_rule["reviewed_fraction"]],
        "rule_sup_orr": [float(x) for x in rt_rule["override_rate_of_reviews"]],
        "routing": {
            "m0_sup_revfrac_mean": _m(rt, "reviewed_fraction"),
            "m0_sup_revfrac_all_mean": _m(rt, "reviewed_fraction_all"),
            "m0_sup_undetermined": _m(rt, "undetermined_rate"),
            "m0_sup_cov_all": _m(rt, "automation_coverage_all"),
            "m0_sup_cov_reviewable": _m(rt, "automation_coverage_reviewable"),
            "rule_sup_revfrac_mean": _m(rt_rule, "reviewed_fraction"),
            "rule_sup_undetermined": _m(rt_rule, "undetermined_rate"),
        },
        "verdict": verdict, "verdict_rows": verdict_rows,
        "coverage": coverage,
        "band": band.as_dict() if band is not None else None,
        "m0_final": {"override_rate": m0_last["override_rate"],
                     "pearson_r": m0_last.get("pearson_r"),
                     "sign_acc_nonzero": m0_last.get("sign_acc_nonzero"),
                     "exact_class_acc": m0_last.get("exact_class_acc"),
                     "zero_baseline_acc": m0_last.get("zero_baseline_acc"),
                     "n_examples_fit": m0_last["n_examples_fit"],
                     "n_examples_cal": m0_last["n_examples_cal"],
                     "undetermined_rate_train": m0_last["undetermined_rate_train"]},
        "per_iter": res["per_iter"], "run_config": res["config"],
        "elapsed_s": time.perf_counter() - t0, "cached": False,
    }
    os.makedirs(_CACHE, exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, cache_path)                 # atomic: no partial result file
    return rec


# --------------------------------------------------------------------------- #
# Task builders                                                                #
# --------------------------------------------------------------------------- #
ARMS = [("targeted_pub", dict(policy="targeted", split_fit=False)),
        ("targeted", dict(policy="targeted", split_fit=True)),
        ("stability", dict(policy="stability", split_fit=True)),
        ("margin", dict(policy="margin", split_fit=True)),
        ("random", dict(policy="random", split_fit=True))]


def tasks_pilot():
    return [_base_task(seed=301, policy="targeted", split_fit=False,
                       arm="targeted_pub", part="pilot", **HEAD)]


def tasks_head(seeds=range(301, 311)):
    out = []
    for arm, kw in ARMS:
        for seed in seeds:
            out.append(_base_task(seed=seed, arm=arm, part="head", **HEAD, **kw))
    return out


def tasks_grid(seeds=range(301, 311)):
    """The eight-cell contention grid under the deployable policy and its
    oracle-informed upper reference; the Holm family of the gate contrasts."""
    out = []
    for cell in GRID_CELLS:
        for arm, kw in (("stability", dict(policy="stability", split_fit=True)),
                        ("targeted", dict(policy="targeted", split_fit=True))):
            for seed in seeds:
                out.append(_base_task(seed=seed, arm=arm, part="grid",
                                      **cell, **kw))
    return out


def tasks_curve(rhos=(0.02, 0.05, 0.10, 0.25, 0.50), seeds=range(301, 304),
                n_eval_c10=8):
    out = []
    # C9 carries both the deployable policy and its oracle-informed reference.
    # C10 instances are four times larger (9,350 orders against 2,269) and cost
    # about fifteen times the wall-clock, so the confirmation cell runs the
    # deployable policy only; it confirms the curve's shape, it is not a second
    # place to measure the price of deployability.
    for cell, n_eval, arms in ((HEAD, 10, ("stability", "targeted")),
                               (CONF, n_eval_c10, ("stability",))):
        for rho in rhos:
            for seed in seeds:
                kw = dict(cell); kw["rho"] = rho
                for arm in arms:
                    out.append(_base_task(seed=seed, arm=arm, part="curve",
                                          policy=arm, split_fit=True,
                                          n_eval=n_eval, **kw))
    return out


def tasks_alpha(alphas=(0.05, 0.1, 0.2, 0.3, 0.5), seeds=range(301, 304)):
    out = []
    for a in alphas:
        for seed in seeds:
            out.append(_base_task(seed=seed, arm="stability", part="alpha",
                                  policy="stability", split_fit=True, alpha=a,
                                  **HEAD))
    for seed in seeds:                          # locally-adaptive variant
        out.append(_base_task(seed=seed, arm="stability_norm", part="alpha",
                              policy="stability", split_fit=True, alpha=0.1,
                              band_mode="normalized", **HEAD))
    return out


def tasks_cov(betas=(0.0, 0.25, 0.5, 0.75, 1.0), seeds=range(301, 304)):
    out = []
    for b in betas:
        for seed in seeds:
            kw = dict(HEAD); kw["beta"] = b
            out.append(_base_task(seed=seed, arm="stability", part="cov",
                                  policy="stability", split_fit=True, **kw))
    return out


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
CSV_COLS = ["part", "arm", "policy", "split_fit", "alpha", "band_mode",
            "campus", "regime", "u", "beta", "rho", "seed", "inst_id", "n_wos",
            "rule", "m0_alone", "oracle", "rule_sup", "m0_sup",
            "m0_sup_revfrac", "m0_sup_orr", "rule_sup_revfrac", "rule_sup_orr"]


def _append_rows(csv_path, task, rec):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        for i, iid in enumerate(rec["inst_ids"]):
            row = {"part": task["part"], "arm": task["arm"],
                   "policy": rec["policy"], "split_fit": int(rec["split_fit"]),
                   "alpha": rec["alpha"], "band_mode": rec["band_mode"],
                   "campus": rec["campus"], "regime": rec["regime"],
                   "u": rec["u"], "beta": rec["beta"], "rho": rec["rho"],
                   "seed": rec["seed"], "inst_id": iid, "n_wos": rec["n_wos"]}
            for k in DECIDERS:
                row[k] = "%.6f" % rec["per"][k][i]
            for k in ("m0_sup_revfrac", "m0_sup_orr", "rule_sup_revfrac",
                      "rule_sup_orr"):
                v = rec.get(k) or []
                row[k] = ("%.4f" % v[i]) if i < len(v) else ""
            w.writerow(row)


def run_tasks(tasks, csv_path, workers, label, fresh=True):
    if fresh and os.path.exists(csv_path):
        os.remove(csv_path)
    print("[%s] %d cell-seed tasks, %d workers -> %s"
          % (label, len(tasks), workers, csv_path), flush=True)
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
            rt = rec["routing"]
            print("  [%s %d/%d] %-12s c%d b%.2f r%.2f a%.2f s%d | rule=%.0f "
                  "m0=%.0f m0sup=%.0f gain=%.2f%% | rev=%.3f und=%.3f "
                  "cov=%.3f q=%s %s (%.0fs, wall %.0fs)"
                  % (label, done, len(tasks), t["arm"], rec["campus"],
                     rec["beta"], rec["rho"], rec["alpha"], rec["seed"],
                     np.mean(rec["per"]["rule"]), np.mean(rec["per"]["m0_alone"]),
                     np.mean(rec["per"]["m0_sup"]),
                     100.0 * (np.mean(rec["per"]["rule"]) - np.mean(rec["per"]["m0_alone"]))
                     / np.mean(rec["per"]["rule"]),
                     rt["m0_sup_revfrac_mean"], rt["m0_sup_undetermined"],
                     rt["m0_sup_cov_all"],
                     ("%.3f" % rec["band"]["q"]) if rec["band"] else "-",
                     "CACHED" if rec.get("cached") else "",
                     rec["elapsed_s"], time.time() - t0), flush=True)
    return records


def dump_records(records, path):
    """Full per-cell records (routing telemetry, coverage, verdict samples)."""
    out = []
    for t, r in records:
        rr = {k: v for k, v in r.items() if k not in ("per_iter",)}
        rr["arm"] = t["arm"]; rr["part"] = t["part"]
        rr["per_iter_last"] = None
        out.append(rr)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("[dump] wrote %s (%d records)" % (path, len(out)), flush=True)


# --------------------------------------------------------------------------- #
# Pilot gate: reproduce the committed headline numbers before anything else    #
# --------------------------------------------------------------------------- #
def _p4_sig(seed):
    """Signature of the committed y3_p4 cache record for the headline cell."""
    payload = {"campus": 9, "regime": "storm2", "u": 100, "size": None,
               "beta": 1.0, "rho": 0.25, "eps": EPS, "theta": THETA,
               "mech": "targeted", "channel": CHANNEL, "family": FAMILY,
               "master_seed": MASTER_SEED, "seed": seed, "n_train": 16,
               "n_probe": 4, "n_eval": 10, "m0_iters": 8}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def published_record(seed):
    p = os.path.join(_ROOT, "results", "y3_p4", "cache", "%s.json" % _p4_sig(seed))
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def run_pilot(workers):
    print("=" * 78)
    print("PILOT GATE -- headline cell, seed 301, published protocol")
    print("=" * 78)
    t = tasks_pilot()[0]
    rec = evaluate_cell(t)
    pub = published_record(301)
    if pub is None:
        print("  !! committed y3_p4 cache record missing; cannot gate")
        return False
    ok = True
    for k in DECIDERS:
        a = np.asarray(rec["per"][k]); b = np.asarray(pub["per"][k])
        same = bool(np.array_equal(a, b))
        ok = ok and same
        print("  %-10s per-instance TWT* identical to the committed record: %s "
              "(max |diff| = %.3e)" % (k, "YES" if same else "NO",
                                       float(np.abs(a - b).max())))
    rule = float(np.mean(rec["per"]["rule"]))
    gain = 100.0 * (rule - float(np.mean(rec["per"]["m0_alone"]))) / rule
    pgain = 100.0 * (float(np.mean(pub["per"]["rule"]))
                     - float(np.mean(pub["per"]["m0_alone"]))) \
        / float(np.mean(pub["per"]["rule"]))
    print("  seed-301 M0-vs-RULE gain: ours %.4f%%  committed %.4f%%  diff %.2e"
          % (gain, pgain, gain - pgain))
    summ = os.path.join(_ROOT, "results", "y3_p4", "m0_gate_summary.json")
    with open(summ) as fh:
        s = json.load(fh)
    cell = s["cells"]["c9_storm2_u100_b1.00_r0.25"]
    print("  published 10-seed \\MzeroGain = %.4f%% (m0_alone pct_below_rule)"
          % cell["ladder"]["m0_alone"]["pct_below_rule"])
    print("  PILOT GATE: %s" % ("PASS" if ok else "FAIL"))
    return ok


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["pilot", "head", "grid", "curve", "alpha",
                                       "cov", "all"], default="pilot")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4,
                    help="parent thread cap; workers are always single-threaded")
    ap.add_argument("--n-eval-c10", type=int, default=8)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    os.makedirs(_OUT, exist_ok=True)
    os.makedirs(_CACHE, exist_ok=True)

    if args.part in ("pilot", "all"):
        if not run_pilot(args.workers) and args.part == "all":
            print("pilot gate failed; refusing to run the sweep")
            sys.exit(1)
    if args.part in ("head", "all"):
        recs = run_tasks(tasks_head(), os.path.join(_OUT, "head.csv"),
                         args.workers, "head")
        dump_records(recs, os.path.join(_OUT, "head_records.json"))
    if args.part in ("grid", "all"):
        recs = run_tasks(tasks_grid(), os.path.join(_OUT, "grid.csv"),
                         args.workers, "grid")
        dump_records(recs, os.path.join(_OUT, "grid_records.json"))
    if args.part in ("curve", "all"):
        recs = run_tasks(tasks_curve(n_eval_c10=args.n_eval_c10),
                         os.path.join(_OUT, "curve.csv"), args.workers, "curve")
        dump_records(recs, os.path.join(_OUT, "curve_records.json"))
    if args.part in ("alpha", "all"):
        recs = run_tasks(tasks_alpha(), os.path.join(_OUT, "alpha.csv"),
                         args.workers, "alpha")
        dump_records(recs, os.path.join(_OUT, "alpha_records.json"))
    if args.part in ("cov", "all"):
        recs = run_tasks(tasks_cov(), os.path.join(_OUT, "coverage.csv"),
                         args.workers, "cov")
        dump_records(recs, os.path.join(_OUT, "coverage_records.json"))
    print("[y3_w1] part=%s complete." % args.part, flush=True)


if __name__ == "__main__":
    main()
