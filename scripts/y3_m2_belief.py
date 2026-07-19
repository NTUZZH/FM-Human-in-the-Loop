#!/usr/bin/env python
"""Y3 P5 ABLATION -- M2: Bayesian belief + active elicitation (no PPO, CPU only).

Completes the M0/M1/M2 ladder. M2 differs from M0 (scripts/y3_abl_common.
replicate_m0 / fmwos.hitl.augmented_rule) in EXACTLY ONE component -- the
estimator -- so this script REUSES verbatim: the override-log weak labels
(``AR.weak_labels_from_log``), the augmented-ATC decider (``AR.augmented_atc_
decider``, corrects BOTH weight and deadline via hat_s), the held-out
TWT*(w*,d*) ladder (``ABL.eval_ladder``) and the probe accuracy
(``AR.probe_shift_accuracy``). The only swap is M0's MLP ``ShiftEstimator`` ->
M2's conjugate Bayesian linear regression ``fmwos.hitl.belief.BayesianLinearShift``.

Three pipelines per (cell, seed), all sharing the same permutation RNG, the same
never-reset aggregate, the same supervisor seed, and the same 8 DAgger iters:
  M0        replicate_m0 (point estimator, TARGETED review)          -- comparator
  M2-TGT    Bayesian belief, SAME base TARGETED review (mean only)   -- parity test
  M2-ACT    Bayesian belief, ACTIVE review: budget steered by
            posterior variance (consequential AND uncertain), matched rho

Eval cell: c9 storm2 u100, beta 1.0 seeds 301-305 (primary) + beta 0.75 seeds
301-303 (robustness), rho 0.25, eps 0, theta 1.0, F-NL, master_seed 12345,
channel full_class_shift, held-out files[20:30] (n=10), TWT*(d*).

Outputs (results/y3_p5/m2/):
  m2_ladder_per_instance.csv   per (cell,seed,method,inst) TWT* for the ladder
  m2_recovery.csv              per (cell,seed,method,iter) hat_s accuracy + cum
                               overrides + held-out alone-TWT* (the recovery curve)
  m2_summary.json              seed-mean ladder + M2-vs-M0 / M2-vs-RULE / active-
                               vs-targeted contrasts (seed-averaged paired Wilcoxon)

Run (tmux y3_m2, CPU only, <=6 workers, OMP=1/worker, niced):
    PYTHONPATH=src OMP_NUM_THREADS=1 nice -n 15 \
        python scripts/y3_m2_belief.py --workers 6
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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch  # noqa: E402

from fmwos.env import DispatchEnv                                  # noqa: E402
from fmwos.hitl import augmented_rule as AR                        # noqa: E402
from fmwos.hitl import true_objective as TO                        # noqa: E402
from fmwos.hitl.supervisor import Supervisor                       # noqa: E402
from fmwos.hitl.latent_head import LAT_DIM                         # noqa: E402
from fmwos.hitl.belief import (BayesianLinearShift,                # noqa: E402
                               ActiveReviewSupervisor, belief_variance_map)

import y3_abl_common as ABL                                        # noqa: E402

_OUT = os.path.join(_ROOT, "results", "y3_p5", "m2")

# Locked cell constants (identical to scripts/y3_p4_m0grid.py / the M0 grid).
FAMILY = "F-NL"
MASTER_SEED = 12345
EPS = 0.0
THETA = 1.0
MECH = "targeted"
CHANNEL = "full_class_shift"
CAMPUS = 9
U = 100
RHO = 0.25
OUTER_ITERS = 8
N_TRAIN, N_PROBE, N_EVAL = 16, 4, 10
OVERRIDE_W, CONFIRM_W = 5.0, 1.0
ALPHA = 1.0            # BLR prior precision (ridge strength); weakly-informative
VAR_WEIGHT = 1.0       # active-review variance weight (z(-margin)+w*z(variance))


# --------------------------------------------------------------------------- #
# M2 pipeline (mirror of ABL.replicate_m0; ONLY the estimator differs)         #
# --------------------------------------------------------------------------- #
def run_m2(train, probe, overlay, *, beta, rho, eps, active=False,
           var_weight=VAR_WEIGHT, alpha=ALPHA, outer_iters=OUTER_ITERS,
           mechanism=MECH, theta=THETA, override_weight=OVERRIDE_W,
           confirm_weight=CONFIRM_W, seed=0, eval_insts=None):
    """Run the M2 DAgger pipeline; return {belief, per_iter}.

    Step-for-step identical to ABL.replicate_m0 (same permutation RNG, same
    never-reset aggregate, same supervisor seed, same probe accuracy) except:
      * the estimator is BayesianLinearShift, fit on the full aggregate each iter
        (batch == sequential for the conjugate Gaussian; symmetric with M0);
      * when active, the supervisor is ActiveReviewSupervisor, whose fixed review
        budget is steered by the belief's posterior variance (consequential AND
        uncertain) at the same rho.
    """
    np.random.seed(seed)
    belief = BayesianLinearShift(dim=LAT_DIM, alpha=alpha)
    Xagg = np.zeros((0, LAT_DIM), np.float32)
    yagg = np.zeros((0,), np.float32)
    wagg = np.zeros((0,), np.float32)
    per_iter = []
    rng = np.random.default_rng(seed)
    n_ep = len(train)
    cum_over = 0

    for it in range(outer_iters):
        order = rng.permutation(len(train))[:n_ep]
        n_over = n_rev = n_conf = 0
        rev_fracs = []
        for k in order:
            inst = train[int(k)]
            applied = overlay.apply(inst)
            if active:
                vmap = belief_variance_map(belief, inst)
                sup = ActiveReviewSupervisor(
                    overlay, inst, rho=rho, epsilon=eps, theta=theta,
                    mechanism=mechanism, seed=seed, applied=applied,
                    variance_map=vmap, var_weight=var_weight)
            else:
                sup = Supervisor(overlay, inst, rho=rho, epsilon=eps, theta=theta,
                                 mechanism=mechanism, seed=seed, applied=applied)
            decider = AR.augmented_atc_decider(belief, inst, channel=CHANNEL)
            _sched, log = DispatchEnv(inst).run_supervised(
                decider, supervisor=sup, method="m2_atc", seed=seed)
            X, y, w = AR.weak_labels_from_log(log, inst, override_weight, confirm_weight)
            if len(X):
                Xagg = np.concatenate([Xagg, X]); yagg = np.concatenate([yagg, y])
                wagg = np.concatenate([wagg, w])
            s = sup.summary()
            n_over += s["n_overrides"]; n_rev += s["n_reviews"]
            n_conf += s["n_confirmations"]
            rev_fracs.append(s["reviewed_fraction"])
        belief.fit(Xagg, yagg, wagg)                 # conjugate posterior from prior
        acc = AR.probe_shift_accuracy(belief, probe, overlay)
        orr = (n_over / n_rev) if n_rev else 0.0
        cum_over += n_over
        row = {"iter": it, "n_reviews": n_rev, "n_overrides": n_over,
               "cum_overrides": cum_over, "n_confirmations": n_conf,
               "override_rate": orr, "reviewed_fraction": float(np.mean(rev_fracs)),
               "n_examples_agg": int(len(Xagg)), **acc}
        if eval_insts is not None:
            twts = []
            for inst in eval_insts:
                applied = overlay.apply(inst)
                d = AR.augmented_atc_decider(belief, inst, channel=CHANNEL)
                sched, _ = DispatchEnv(inst).run_supervised(
                    d, supervisor=None, method="m2", seed=seed)
                twts.append(TO.score_true(inst, sched, overlay, applied)["TWT_true"])
            row["alone_twt_mean"] = float(np.mean(twts))
            row["alone_twt_per"] = [float(x) for x in twts]
        per_iter.append(row)
    return {"belief": belief, "per_iter": per_iter}


# --------------------------------------------------------------------------- #
# One (cell, seed): run M0 + M2-TGT + M2-ACT, score the held-out ladder         #
# --------------------------------------------------------------------------- #
def run_cell(task):
    t0 = time.perf_counter()
    torch.set_num_threads(1)
    try:
        os.nice(5)
    except Exception:
        pass
    beta, seed = task["beta"], task["seed"]
    train, probe, eval_insts, eval_names = ABL.load_pools(
        CAMPUS, U, n_train=N_TRAIN, n_probe=N_PROBE, n_eval=N_EVAL)
    overlay = ABL.make_overlay(beta, family=FAMILY, master_seed=MASTER_SEED,
                               channel=CHANNEL)

    out = {"beta": beta, "seed": seed, "eval_names": eval_names,
           "n_wos": len(eval_insts[0]["work_orders"]), "methods": {}}

    # ---- M0 (point estimator, TARGETED) -- comparator ---------------------- #
    torch.manual_seed(seed); np.random.seed(seed)
    m0 = ABL.replicate_m0(train, probe, overlay, beta=beta, rho=RHO, eps=EPS,
                          decider_channel=CHANNEL, outer_iters=OUTER_ITERS,
                          mechanism=MECH, theta=THETA, seed=seed,
                          eval_insts=eval_insts)
    m0_ladder = ABL.eval_ladder(m0["estimator"], eval_insts, overlay, rho=RHO,
                                eps=EPS, theta=THETA, mechanism=MECH,
                                decider_channel=CHANNEL, seed=seed)

    # ---- M2-TARGETED (Bayesian belief, base TARGETED review) -- parity ----- #
    m2t = run_m2(train, probe, overlay, beta=beta, rho=RHO, eps=EPS,
                 active=False, seed=seed, eval_insts=eval_insts)
    m2t_ladder = ABL.eval_ladder(m2t["belief"], eval_insts, overlay, rho=RHO,
                                 eps=EPS, theta=THETA, mechanism=MECH,
                                 decider_channel=CHANNEL, seed=seed)

    # ---- M2-ACTIVE (Bayesian belief, variance-steered review) -------------- #
    m2a = run_m2(train, probe, overlay, beta=beta, rho=RHO, eps=EPS,
                 active=True, var_weight=VAR_WEIGHT, seed=seed, eval_insts=eval_insts)
    m2a_ladder = ABL.eval_ladder(m2a["belief"], eval_insts, overlay, rho=RHO,
                                 eps=EPS, theta=THETA, mechanism=MECH,
                                 decider_channel=CHANNEL, seed=seed)

    for name, res, ladder in (("m0", m0, m0_ladder), ("m2t", m2t, m2t_ladder),
                              ("m2a", m2a, m2a_ladder)):
        pi = res["per_iter"]
        alone_key = "m0_twt_mean" if name == "m0" else "alone_twt_mean"
        alone_per_key = "m0_twt_per" if name == "m0" else "alone_twt_per"
        recovery = [{"iter": r["iter"], "cum_overrides": r["cum_overrides"],
                     "n_reviews": r["n_reviews"], "n_overrides": r["n_overrides"],
                     "override_rate": r["override_rate"],
                     "reviewed_fraction": r.get("reviewed_fraction"),
                     "sign_acc_nonzero": r["sign_acc_nonzero"],
                     "pearson_r": r["pearson_r"],
                     "exact_class_acc": r["exact_class_acc"],
                     "zero_baseline_acc": r["zero_baseline_acc"],
                     "alone_twt": r.get(alone_key)} for r in pi]
        out["methods"][name] = {
            "ladder": ladder["per"],
            "rule_sup_revfrac": ladder["rule_sup_revfrac"],
            "m0_sup_revfrac": ladder["m0_sup_revfrac"],
            "recovery": recovery,
            "final": {"cum_overrides": pi[-1]["cum_overrides"],
                      "override_rate": pi[-1]["override_rate"],
                      "reviewed_fraction": pi[-1].get("reviewed_fraction"),
                      "sign_acc_nonzero": pi[-1]["sign_acc_nonzero"],
                      "pearson_r": pi[-1]["pearson_r"],
                      "alone_twt_mean": float(np.mean(pi[-1][alone_per_key]))},
        }
    out["elapsed_s"] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #
def _safe_mean(vals):
    """Mean over non-None values; nan if all None (M0's training reviewed
    fraction is not tracked by replicate_m0, so it is None -- see note)."""
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else float("nan")


def _stack(records, method, decider):
    """(S seeds x n instances) matrix of TWT*, aligned by instance order (the
    held-out set is identical across seeds)."""
    records = sorted(records, key=lambda r: r["seed"])
    mat = [r["methods"][method]["ladder"][decider] for r in records]
    return np.asarray(mat, float)


def summarize(records_by_beta, out_json):
    summary = {"config": {"campus": CAMPUS, "u": U, "rho": RHO, "eps": EPS,
                          "theta": THETA, "family": FAMILY,
                          "master_seed": MASTER_SEED, "channel": CHANNEL,
                          "alpha": ALPHA, "var_weight": VAR_WEIGHT,
                          "outer_iters": OUTER_ITERS,
                          "scoring": "TWT*(w*,d*) full_class_shift, held-out n=%d" % N_EVAL,
                          "contrast": "seed-averaged per-instance paired Wilcoxon (pratt), "
                                      "W = test strictly lower TWT*"},
               "cells": {}}
    for beta, recs in sorted(records_by_beta.items()):
        n_seeds = len(recs)
        # seed-mean ladder per method
        ladder = {}
        for method in ("m0", "m2t", "m2a"):
            ladder[method] = {}
            for d in ("rule", "m0_alone", "m0_sup", "rule_sup", "oracle"):
                mat = _stack(recs, method, d)          # S x n
                ladder[method][d] = {"twt_mean": float(mat.mean(axis=1).mean()),
                                     "twt_std": float(mat.mean(axis=1).std())}
            ladder[method]["final_recovery"] = {
                "cum_overrides": _safe_mean([r["methods"][method]["final"]["cum_overrides"] for r in recs]),
                "override_rate": _safe_mean([r["methods"][method]["final"]["override_rate"] for r in recs]),
                "reviewed_fraction": _safe_mean([r["methods"][method]["final"]["reviewed_fraction"] for r in recs]),
                "sign_acc_nonzero": _safe_mean([r["methods"][method]["final"]["sign_acc_nonzero"] for r in recs]),
                "pearson_r": _safe_mean([r["methods"][method]["final"]["pearson_r"] for r in recs])}

        # contrasts (seed-averaged per-instance paired)
        def contrast(test_m, test_d, comp_m, comp_d):
            return ABL.seed_avg_contrast(_stack(recs, test_m, test_d),
                                         _stack(recs, comp_m, comp_d))
        contrasts = {
            # parity: M2-targeted vs M0 (both alone)
            "M2t_vs_M0_alone": contrast("m2t", "m0_alone", "m0", "m0_alone"),
            "M2t_alone_vs_RULE": contrast("m2t", "m0_alone", "m0", "rule"),
            "M0_alone_vs_RULE": contrast("m0", "m0_alone", "m0", "rule"),
            # active vs targeted (both M2, alone) at matched budget
            "M2a_vs_M2t_alone": contrast("m2a", "m0_alone", "m2t", "m0_alone"),
            "M2a_alone_vs_RULE": contrast("m2a", "m0_alone", "m2a", "rule"),
            "M2a_vs_M0_alone": contrast("m2a", "m0_alone", "m0", "m0_alone"),
            # in-loop parity (optional)
            "M2t_sup_vs_M0_sup": contrast("m2t", "m0_sup", "m0", "m0_sup"),
        }
        summary["cells"]["beta%.2f" % beta] = {
            "beta": beta, "n_seeds": n_seeds, "seeds": [r["seed"] for r in recs],
            "n_wos": recs[0]["n_wos"], "ladder": ladder, "contrasts": contrasts}
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("[m2] wrote %s" % out_json, flush=True)
    return summary


# --------------------------------------------------------------------------- #
# CSV writers                                                                  #
# --------------------------------------------------------------------------- #
def write_ladder_csv(records, path):
    cols = ["beta", "seed", "method", "inst_id", "rule", "alone", "plus_sup",
            "rule_sup", "oracle"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in records:
            for method in ("m0", "m2t", "m2a"):
                lad = r["methods"][method]["ladder"]
                for i, iid in enumerate(r["eval_names"]):
                    w.writerow({"beta": r["beta"], "seed": r["seed"], "method": method,
                                "inst_id": iid,
                                "rule": "%.6f" % lad["rule"][i],
                                "alone": "%.6f" % lad["m0_alone"][i],
                                "plus_sup": "%.6f" % lad["m0_sup"][i],
                                "rule_sup": "%.6f" % lad["rule_sup"][i],
                                "oracle": "%.6f" % lad["oracle"][i]})
    print("[m2] wrote %s" % path, flush=True)


def write_recovery_csv(records, path):
    cols = ["beta", "seed", "method", "iter", "cum_overrides", "n_reviews",
            "n_overrides", "override_rate", "reviewed_fraction",
            "sign_acc_nonzero", "pearson_r", "exact_class_acc", "alone_twt"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in records:
            for method in ("m0", "m2t", "m2a"):
                for row in r["methods"][method]["recovery"]:
                    w.writerow({"beta": r["beta"], "seed": r["seed"], "method": method,
                                "iter": row["iter"], "cum_overrides": row["cum_overrides"],
                                "n_reviews": row["n_reviews"], "n_overrides": row["n_overrides"],
                                "override_rate": "%.4f" % row["override_rate"],
                                "reviewed_fraction": ("%.4f" % row["reviewed_fraction"]
                                                      if row["reviewed_fraction"] is not None else ""),
                                "sign_acc_nonzero": "%.4f" % row["sign_acc_nonzero"],
                                "pearson_r": "%.4f" % row["pearson_r"],
                                "exact_class_acc": "%.4f" % row["exact_class_acc"],
                                "alone_twt": "%.4f" % (row["alone_twt"] or float("nan"))})
    print("[m2] wrote %s" % path, flush=True)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def build_tasks(smoke=False):
    tasks = []
    for seed in range(301, 306):
        tasks.append({"beta": 1.0, "seed": seed})
    for seed in range(301, 304):
        tasks.append({"beta": 0.75, "seed": seed})
    if smoke:
        tasks = [{"beta": 1.0, "seed": 301}]
    return tasks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    os.makedirs(_OUT, exist_ok=True)

    tasks = build_tasks(smoke=args.smoke)
    print("[m2] %d (cell,seed) tasks x 3 pipelines, %d workers -> %s"
          % (len(tasks), args.workers, _OUT), flush=True)
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut = {ex.submit(run_cell, t): t for t in tasks}
        for f in as_completed(fut):
            r = f.result()
            records.append(r)
            m0f = r["methods"]["m0"]["final"]
            m2tf = r["methods"]["m2t"]["final"]
            m2af = r["methods"]["m2a"]["final"]
            print("  [%d/%d] b%.2f s%d  M0 alone=%.0f r=%.2f cum=%d | "
                  "M2t alone=%.0f r=%.2f | M2a alone=%.0f r=%.2f cum=%d revf=%.3f "
                  "(%.0fs, wall %.0fs)"
                  % (len(records), len(tasks), r["beta"], r["seed"],
                     m0f["alone_twt_mean"], m0f["pearson_r"], int(m0f["cum_overrides"]),
                     m2tf["alone_twt_mean"], m2tf["pearson_r"],
                     m2af["alone_twt_mean"], m2af["pearson_r"], int(m2af["cum_overrides"]),
                     m2af["reviewed_fraction"], r["elapsed_s"], time.time() - t0), flush=True)

    by_beta = defaultdict(list)
    for r in records:
        by_beta[r["beta"]].append(r)
    write_ladder_csv(records, os.path.join(_OUT, "m2_ladder_per_instance.csv"))
    write_recovery_csv(records, os.path.join(_OUT, "m2_recovery.csv"))
    summarize(by_beta, os.path.join(_OUT, "m2_summary.json"))
    print("[m2] done (%.0fs)." % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
