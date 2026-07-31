#!/usr/bin/env python
"""P1.5 proof-of-life: the full method ladder on the headline contention gate
cell, scored on the TRUE full-class-shift objective TWT*(w*, d*).

Cell: campus 9, storm2 u100 (util ~1.0, saturated/surge), beta in {0.75, 1.0},
rho=0.25, eps=0, TARGETED, theta=1.0. Ladder, all scored on TWT*(w*,d*):

    RULE            ATC on recorded fields, no supervisor (deployed rule).
    RULE+SUP        ATC + simulated supervisor reviewing 25% of decisions,
                    overriding toward the true-(w*,d*) preferred pick.
    M0              the augmented ATC rule: its estimator (trained on the
                    RULE+SUP override stream over 8 outer iters) corrects BOTH
                    the weight AND the deadline of the ATC index.
    ORACLE          myopic ATC on true w*,d* at every decision (full-info ceiling).

The M0 estimator is trained on a disjoint TRAIN slice; the ladder is evaluated on
a held-out EVAL slice. hat_s recovery accuracy is probed per outer iteration on a
disjoint PROBE slice.

Run: PYTHONPATH=src nice python scripts/y3_p15_proof.py [--betas 0.75 1.0]
                                                        [--u 100] [--campus 9]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import csv
import glob
import json
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch                                              # noqa: E402

from fmwos.env import DispatchEnv                          # noqa: E402
from fmwos.hitl import deciders as dec                     # noqa: E402
from fmwos.hitl import overlay as ov                       # noqa: E402
from fmwos.hitl.supervisor import Supervisor               # noqa: E402
from fmwos.hitl import augmented_rule as AR                # noqa: E402
from fmwos.hitl import true_objective as TO                # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "train_log", "y3_p15")
SEED = 301
MASTER_SEED = 12345
FAMILY = "F-NL"
RHO = 0.25
THETA = 1.0


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def _cell_files(campus, u):
    cdir = "c%02d" % campus
    return sorted(glob.glob(os.path.join(
        _INST, cdir, "storm2", "w80", "%s_storm2_w80_u%d_*.json" % (cdir, u))))


def ladder_eval(estimator, eval_insts, overlay, channel):
    """Per-instance TWT*(w*,d*) for RULE / RULE+SUP / M0 / ORACLE, pooled."""
    tot = {"rule": 0.0, "sup": 0.0, "m0": 0.0, "oracle": 0.0}
    n = 0
    sup_reviews = sup_overrides = sup_reviewable = 0
    for inst in eval_insts:
        applied = overlay.apply(inst)

        def sc(sched):
            return TO.score_true(inst, sched, overlay, applied)["TWT_true"]

        # RULE (ATC, recorded)
        tot["rule"] += sc(dec.run_rule(DispatchEnv(inst), "atc", seed=SEED))

        # RULE+SUP (rho=0.25, targeted, eps=0)
        sup = Supervisor(overlay, inst, rho=RHO, epsilon=0.0, theta=THETA,
                         mechanism="targeted", seed=SEED, applied=applied)
        s_sched, _log = dec.run_rule_sup(DispatchEnv(inst), "atc", sup, seed=SEED)
        tot["sup"] += sc(s_sched)
        sm = sup.summary()
        sup_reviews += sm["n_reviews"]; sup_overrides += sm["n_overrides"]
        sup_reviewable += sm["n_reviewable"]

        # M0 (augmented ATC, corrected w AND d under full_class_shift)
        m0d = AR.augmented_atc_decider(estimator, inst, channel=channel)
        m0_sched, _ = DispatchEnv(inst).run_supervised(m0d, supervisor=None,
                                                       method="m0", seed=SEED)
        tot["m0"] += sc(m0_sched)

        # ORACLE (full info)
        osup = Supervisor(overlay, inst, rho=0.0, applied=applied)
        tot["oracle"] += sc(dec.run_oracle_greedy(DispatchEnv(inst), osup, seed=SEED))
        n += 1

    mean = {k: v / n for k, v in tot.items()}
    return {
        "n": n, "mean": mean,
        "sup_review_frac": (sup_reviews / sup_reviewable) if sup_reviewable else 0.0,
        "sup_override_rate": (sup_overrides / sup_reviews) if sup_reviews else 0.0,
        "sup_overrides_per_inst": sup_overrides / n,
    }


def run_beta(beta, campus, u, n_train, n_probe, n_eval, out_dir):
    channel = "full_class_shift"
    overlay = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                          master_seed=MASTER_SEED, channel=channel))
    files = _cell_files(campus, u)
    assert len(files) >= n_train + n_probe + n_eval, \
        "only %d instances at c%d storm2 u%d" % (len(files), campus, u)
    train = [_load(p) for p in files[:n_train]]
    probe = [_load(p) for p in files[n_train:n_train + n_probe]]
    eval_insts = [_load(p) for p in
                  files[n_train + n_probe:n_train + n_probe + n_eval]]

    print("\n===== beta=%.2f  c%d storm2 u%d  channel=%s =====" % (beta, campus, u, channel))
    print("train=%d probe=%d eval=%d  (n_wos~%d)"
          % (len(train), len(probe), len(eval_insts), len(train[0]["work_orders"])))

    t0 = time.perf_counter()
    res = AR.run_m0(train, probe, overlay, beta_rho_eps=(beta, RHO, 0.0),
                    outer_iters=8, mechanism="targeted", theta=THETA,
                    seed=SEED, device="cpu", verbose=True)
    m0_secs = time.perf_counter() - t0
    estimator = res["estimator"]

    lad = ladder_eval(estimator, eval_insts, overlay, channel)
    m = lad["mean"]
    rule, sup, m0, orc = m["rule"], m["sup"], m["m0"], m["oracle"]
    gap = rule - orc

    def pct(x):
        return 100.0 * (rule - x) / rule if rule > 1e-9 else 0.0

    print("\n----- LADDER (TWT*(w*,d*), pooled over %d held-out instances) -----" % lad["n"])
    print("  RULE (ATC)          = %10.1f" % rule)
    print("  RULE+SUP (rho=0.25) = %10.1f   (%.1f%% below RULE; rev_frac=%.3f, "
          "over_rate=%.3f, %.0f overrides/inst)"
          % (sup, pct(sup), lad["sup_review_frac"], lad["sup_override_rate"],
             lad["sup_overrides_per_inst"]))
    print("  M0 (augmented ATC)  = %10.1f   (%.1f%% below RULE; %.1f%% of RULE->ORACLE gap)"
          % (m0, pct(m0), 100.0 * (rule - m0) / gap if gap > 1e-9 else 0.0))
    print("  ORACLE (full info)  = %10.1f   (%.1f%% below RULE)" % (orc, pct(orc)))
    per = res["per_iter"]
    print("  M0 hat_s sign-acc trend: %s"
          % " ".join("%.3f" % r["sign_acc_nonzero"] for r in per))
    print("  M0 override-rate trend:  %s"
          % " ".join("%.3f" % r["override_rate"] for r in per))

    out = {
        "beta": beta, "campus": campus, "u": u, "channel": channel,
        "rho": RHO, "theta": THETA, "seed": SEED,
        "n_train": len(train), "n_probe": len(probe), "n_eval": lad["n"],
        "n_wos_train0": len(train[0]["work_orders"]),
        "ladder": {"RULE": rule, "RULE+SUP": sup, "M0": m0, "ORACLE": orc},
        "ladder_pct_below_rule": {"RULE+SUP": pct(sup), "M0": pct(m0), "ORACLE": pct(orc)},
        "m0_pct_of_gap": 100.0 * (rule - m0) / gap if gap > 1e-9 else 0.0,
        "sup_review_frac": lad["sup_review_frac"],
        "sup_override_rate": lad["sup_override_rate"],
        "sup_overrides_per_inst": lad["sup_overrides_per_inst"],
        "m0_secs": m0_secs,
        "per_iter": per,
    }
    with open(os.path.join(out_dir, "proof_beta%.2f.json" % beta), "w") as fh:
        json.dump(out, fh, indent=1, default=str)

    csv_path = os.path.join(out_dir, "m0_metrics_beta%.2f.csv" % beta)
    keys = ["iter", "n_reviews", "n_overrides", "override_rate", "n_examples_agg",
            "sign_acc_nonzero", "exact_class_acc", "zero_baseline_acc", "pearson_r",
            "est_loss"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(keys)
        for row in per:
            w.writerow([row.get(k) for k in keys])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", type=float, nargs="+", default=[0.75, 1.0])
    ap.add_argument("--campus", type=int, default=9)
    ap.add_argument("--u", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-probe", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=str, default=_OUT)
    args = ap.parse_args()
    torch.set_num_threads(int(args.threads))
    os.makedirs(args.out, exist_ok=True)

    allout = []
    for beta in args.betas:
        allout.append(run_beta(beta, args.campus, args.u, args.n_train,
                               args.n_probe, args.n_eval, args.out))
    with open(os.path.join(args.out, "proof_summary.json"), "w") as fh:
        json.dump(allout, fh, indent=1, default=str)
    print("\nwrote %s" % os.path.join(args.out, "proof_summary.json"))


if __name__ == "__main__":
    main()
