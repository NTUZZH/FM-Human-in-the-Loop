#!/usr/bin/env python
"""W2 R1/R2: the estimator ladder at the headline cell.

Rungs (see results/y3_w2/RUN_PLAN.md):
  (i)    mse_published  the shipped weighted-squared-error pipeline, verbatim
  (i-es) mse_es         same loss and inputs, choice-rung fitting protocol
  (ii)   choice         CLAIM A alone: conditional logit, per-order features
  (iii)  choice_queue   CLAIMS A+B: conditional logit + Deep-Sets pool(Q)
Robustness: choice_tol, choice_queue_tol, choice_queue_k64.

Every rung is scored by the same battery: Pearson r and sign accuracy on the
probe instances, Kendall tau-b against the TRUE ranking at a common reference
trajectory on the held-out instances, held-out choice-model log-likelihood under
one common functional, and the deployed reduction in true weighted tardiness
against the tuned rule, scored by the shipped independent validator.

Before any rung is trained the script (a) diffs its resolved configuration
against rung (i)'s and aborts unless the ONLY differing field is `variant`, and
(b) asserts its parameter count.

Run (one sweep at a time, on the assigned cores):
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
    taskset -c 10-19 python scripts/y3_w2_ladder.py --seeds 301 302 303 304 305
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

import y3_w2_lib as L                                           # noqa: E402
from fmwos.hitl import choice_estimator as CE                   # noqa: E402

EXPECTED_PARAMS = {
    "mse_published": (1761, 1761), "mse_es": (1761, 1762),
    "choice": (1761, 1762), "choice_tol": (1761, 1762),
    "choice_queue": (3041, 3042), "choice_queue_tol": (3041, 3042),
    "choice_queue_k64": (3041, 3042),
}


def evaluate(model, cell, ds_val, ds_test, kcache, tag=""):
    """The common metric battery for one trained rung."""
    ov_, ch, seed = cell["overlay"], cell["channel"], cell["seed"]
    t0 = time.perf_counter()
    rec = L.recovery_metrics(model, cell["probe"], ov_, channel=ch)
    ken = L.kendall_on_instances(model, cell["eval"], ov_, channel=ch,
                                 seed=seed, cache=kcache)
    tau = L.CE.fit_temperature(model, ds_val, channel=ch) if len(ds_val) else float(model.tau())
    ll = CE.choice_loglik(model, ds_test, channel=ch, tau_override=tau)
    dep = L.deployed_twt(model, cell["eval"], ov_, channel=ch, seed=seed,
                         measure_latency=model.use_queue)
    con = L.contrast(np.asarray(dep["aug"]), np.asarray(dep["rule"]))
    out = {"recovery": rec, "kendall_eval": ken,
           "choice_ll": ll, "choice_tau_calibrated": tau,
           "twt_rule": dep["rule"], "twt_aug": dep["aug"],
           "contrast_vs_rule": con, "eval_secs_upper_bound": time.perf_counter() - t0}
    if "ms_per_decision" in dep:
        out["ms_per_decision"] = dep["ms_per_decision"]
        out["n_decisions_timed"] = dep["n_decisions_timed"]
    del tag
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[301, 302, 303, 304, 305])
    ap.add_argument("--variants", nargs="+",
                    default=["mse_published", "mse_es", "choice", "choice_queue"])
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads. DEFAULT 1, NOT 4: the "
                         "published M0 run used torch.set_num_threads(1), and "
                         "the pilot measured that 4 threads moves the seed-301 "
                         "headline by 1.56 percentage points (float reduction "
                         "order changes the fit, which changes the dispatch). "
                         "The only difference from the comparator must be the "
                         "estimator, so the thread count is pinned to 1.")
    ap.add_argument("--campus", type=int, default=9)
    ap.add_argument("--u", type=int, default=100)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--tag", default="ladder")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    os.makedirs(L.OUT, exist_ok=True)
    out_path = os.path.join(L.OUT, "%s.json" % a.tag)
    per_seed = {}

    for seed in a.seeds:
        cell = L.build_cell(campus=a.campus, u=a.u, beta=a.beta, rho=a.rho,
                            seed=seed)
        base_cfg = L.resolved_config(cell, a.variants[0])
        # -- variant-INDEPENDENT evaluation sets, built once per seed ---------- #
        # test: reviewed decisions of RULE+SUP on the 10 held-out instances.
        # calib: the same construction on the 4 validation TRAINING instances,
        # used only to calibrate each rung's choice scale. Building it the same
        # way for every rung (rather than from each rung's own DAgger aggregate)
        # keeps the likelihood comparison a comparison of estimators, not of the
        # decision streams the rungs happened to generate.
        ds_test = L.build_choice_testset(cell["eval"], cell["overlay"], cell)
        ds_calib = L.build_choice_testset(
            cell["train"][L.N_TRAIN - L.N_VAL_INSTANCES:], cell["overlay"], cell)
        kcache = {}
        zero = L.zero_model()

        rows = {}
        for v in a.variants:
            cfg = L.resolved_config(cell, v)
            diff = sorted(k for k in set(cfg) | set(base_cfg)
                          if cfg.get(k) != base_cfg.get(k))
            if diff != ["variant"] and v != a.variants[0]:
                raise SystemExit("config diff between %s and %s is %r, expected "
                                 "['variant'] only" % (v, a.variants[0], diff))
            print("\n=== seed %d  variant %s ===" % (seed, v), flush=True)
            tr = L.train_variant(cell, v, verbose=True)
            exp_e, exp_t = EXPECTED_PARAMS[v]
            assert tr["n_params_estimator"] == exp_e, (
                "%s estimator has %d params, expected %d"
                % (v, tr["n_params_estimator"], exp_e))
            assert tr["n_params_total"] == exp_t, (
                "%s has %d params, expected %d" % (v, tr["n_params_total"], exp_t))
            if v == "mse_published":
                # fidelity: the generalised recovery metric must reproduce the
                # shipped probe EXACTLY for a per-order estimator.
                ref = L.AR.probe_shift_accuracy(tr["model"].est.core,
                                                cell["probe"], cell["overlay"])
                got = L.recovery_metrics(tr["model"], cell["probe"],
                                         cell["overlay"], channel=cell["channel"])
                for k in ("pearson_r", "sign_acc_nonzero", "exact_class_acc"):
                    assert abs(ref[k] - got[k]) < 1e-12, (k, ref[k], got[k])
            ev = evaluate(tr["model"], cell, ds_calib, ds_test, kcache, tag=v)
            ev.update({"n_params_estimator": tr["n_params_estimator"],
                       "n_params_total": tr["n_params_total"],
                       "train_secs_upper_bound": tr["secs"],
                       "dataset_counts": tr["counts"],
                       "fit": tr["fit"],
                       "per_iter_last": tr["per_iter"][-1],
                       "n_overrides_total": int(sum(r["n_overrides"]
                                                    for r in tr["per_iter"]))})
            rows[v] = ev
            c = ev["contrast_vs_rule"]
            print("  -> pearson %.4f  sign %.4f  kendall %.4f  llhold %.4f  "
                  "TWT* %.2f (rule %.2f, %+.3f%%) p=%.4g"
                  % (ev["recovery"]["pearson_r"], ev["recovery"]["sign_acc_nonzero"],
                     ev["kendall_eval"]["kendall_tau"], ev["choice_ll"]["ll"],
                     c["test_mean"], c["comparator_mean"], c["pct_vs_comparator"],
                     c["wilcoxon_p"]), flush=True)

        # references that do not depend on any variant
        rows["_recorded_reference"] = {
            "kendall_eval": L.kendall_on_instances(zero, cell["eval"],
                                                   cell["overlay"],
                                                   channel=cell["channel"],
                                                   zero=True, seed=seed,
                                                   cache=kcache),
            "choice_ll": CE.choice_loglik(
                zero, ds_test, channel=cell["channel"],
                tau_override=L.CE.fit_temperature(zero, ds_calib,
                                                  channel=cell["channel"])),
            "note": "hat_s == 0: the deployed recorded-field ATC ranking"}
        rows["_testset_counts"] = ds_test.counts()
        rows["_calibset_counts"] = ds_calib.counts()
        rows["_config"] = base_cfg
        per_seed[str(seed)] = rows
        with open(out_path + ".tmp", "w") as fh:
            json.dump(per_seed, fh, indent=1)
        os.replace(out_path + ".tmp", out_path)
        print("\n[seed %d written to %s]" % (seed, out_path), flush=True)

    print("\nDONE. %s" % out_path)


if __name__ == "__main__":
    sys.exit(main())
