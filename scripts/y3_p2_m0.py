"""M0 (augmented-rule) full pipeline runner for one supervisor cell (P2).

Symmetric protocol with M1: 8 outer iterations of ATC+SUP log aggregation +
estimator training. Reports hat_s accuracy per iteration and the final M0
true-TWT* vs plain ATC on a held-out instance sample. Cheap (no RL); runs
directly.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.hitl.overlay import Overlay, OverlayParams          # noqa: E402
from fmwos.hitl import augmented_rule                          # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
CAMPUSES = ["c05", "c09", "c10", "c12"]


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _replay_train_files(campus, size, limit):
    files = []
    for f in sorted(glob.glob(os.path.join(_INST, campus, "replay", str(size), "*.json"))):
        try:
            ws = _load(f)["meta"]["window_start"]
        except Exception:
            continue
        if ws <= "2017-12-31":
            files.append(f)
        if len(files) >= limit:
            break
    return files


def collect(campuses, size, per_campus):
    insts = []
    for c in campuses:
        for f in _replay_train_files(c, size, per_campus):
            insts.append(_load(f))
    return insts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--mechanism", type=str, default="targeted")
    ap.add_argument("--family", type=str, default="F-NL")
    ap.add_argument("--master-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--size", type=int, default=150)
    ap.add_argument("--train-per-campus", type=int, default=24)
    ap.add_argument("--probe-per-campus", type=int, default=8)
    ap.add_argument("--eval-per-campus", type=int, default=16)
    ap.add_argument("--out", type=str, default="train_log/y3_p2/m0")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    overlay = Overlay(OverlayParams(beta=args.beta, family=args.family,
                                    master_seed=args.master_seed))

    # disjoint train / probe / eval slices (offset windows)
    all_train = collect(CAMPUSES, args.size, args.train_per_campus)
    probe = collect(CAMPUSES, args.size, args.probe_per_campus)   # first-k, for recovery
    # eval uses a later slice per campus (skip the training window)
    eval_insts = []
    for c in CAMPUSES:
        fs = _replay_train_files(c, args.size, args.train_per_campus + args.eval_per_campus)
        eval_insts += [_load(f) for f in fs[args.train_per_campus:]]

    print("[M0] cell beta=%.2f rho=%.2f eps=%.2f | train=%d probe=%d eval=%d"
          % (args.beta, args.rho, args.eps, len(all_train), len(probe), len(eval_insts)))

    res = augmented_rule.run_m0(
        all_train, probe, overlay,
        beta_rho_eps=(args.beta, args.rho, args.eps),
        outer_iters=8, mechanism=args.mechanism, theta=args.theta,
        seed=args.seed, device="cpu", verbose=True)

    # per-iteration CSV
    csv_path = os.path.join(args.out, "m0_metrics.csv")
    keys = ["iter", "n_reviews", "n_overrides", "override_rate", "n_examples_agg",
            "sign_acc_nonzero", "exact_class_acc", "zero_baseline_acc", "pearson_r",
            "est_loss"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(keys)
        for row in res["per_iter"]:
            w.writerow([row.get(k) for k in keys])

    # final held-out comparison
    cmp = augmented_rule.evaluate_m0_vs_atc(res["estimator"], eval_insts, overlay,
                                            seed=args.seed)
    print("[M0] FINAL held-out (n=%d): M0 true-TWT*=%.2f vs plain ATC=%.2f | "
          "W/T/L = %d/%d/%d  (delta=%.2f, %.2f%%)"
          % (cmp["n"], cmp["m0_true_twt"], cmp["atc_true_twt"], cmp["wins"],
             cmp["ties"], cmp["losses"], cmp["atc_true_twt"] - cmp["m0_true_twt"],
             100.0 * (cmp["atc_true_twt"] - cmp["m0_true_twt"]) / max(cmp["atc_true_twt"], 1e-9)))
    with open(os.path.join(args.out, "m0_final.json"), "w") as fh:
        json.dump(cmp, fh, indent=2)
    res["estimator"].save(os.path.join(args.out, "m0_estimator.pt"))


if __name__ == "__main__":
    main()
