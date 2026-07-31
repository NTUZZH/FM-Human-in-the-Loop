#!/usr/bin/env python
"""W2 diagnostic: give the choice likelihood a fair shot before judging it.

The smoke run showed the conditional-logit rungs fitting the CHOICE very well
(top-1 accuracy 0.87) while recovering the latent badly (sign accuracy below
chance). Two very different explanations, and the package's conclusion depends
on which is true:

  (a) STRUCTURAL. The likelihood is dominated by confirmations (~95% of reviewed
      decisions), and a confirmation's term "the decider's pick IS the choice"
      is satisfied by any s_hat that preserves the ranking the decider already
      had. It is a relative constraint, where the squared-error loss's
      confirmation label (s = 0 on the decider's pick) is an ABSOLUTE anchor.
      If so, the honest finding is about the objective, not the optimiser.
  (b) OPTIMISATION. The learning rate, the scale initialisation or the epoch
      budget is simply wrong for this loss, and a negative result would be a
      false negative.

This script separates them, selecting only on the VALIDATION objective and never
on the test set or on TWT*:
  1. learning-rate search for every objective, on validation loss;
  2. the per-epoch validation curve, so a "best epoch 0" is visible;
  3. an OVERRIDES-ONLY ablation, which removes the confirmation term entirely
     and so isolates explanation (a);
  4. the tolerance-aware confirmation term at several tolerances delta.

Writes results/y3_w2/diag.json. Nothing here is quoted as a headline number;
it decides the hyperparameters the ladder then uses for every rung, and it is
reported in full including the settings that failed.
"""

import argparse
import copy
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

import y3_w2_lib as L                                           # noqa: E402
from fmwos.env import DispatchEnv                               # noqa: E402
from fmwos.hitl import choice_estimator as CE                   # noqa: E402


def collect_dataset(cell, iters=3, k_max=CE.K_MAX):
    """The DAgger aggregate under the SHIPPED squared-error decider.

    Using one fixed decision stream for every objective makes the comparison a
    comparison of losses; if each objective generated its own stream the
    learning-rate search would also be searching over trajectories.
    """
    from fmwos.hitl import augmented_rule as AR
    from fmwos.hitl.latent_head import ShiftEstimator, train_estimator, LAT_DIM
    torch.manual_seed(cell["seed"]); np.random.seed(cell["seed"])
    est = ShiftEstimator(hidden=32)
    ds = CE.ChoiceDataset()
    Xa = np.zeros((0, LAT_DIM), np.float32)
    ya = np.zeros((0,), np.float32); wa = np.zeros((0,), np.float32)
    rng = np.random.default_rng(cell["seed"])
    for it in range(iters):
        for k in rng.permutation(len(cell["train"])):
            inst = cell["train"][int(k)]
            applied = cell["overlay"].apply(inst)
            sup = CE.QueueLoggingSupervisor(
                cell["overlay"], inst, rho=cell["rho"], epsilon=cell["eps"],
                theta=cell["theta"], mechanism=cell["mech"], seed=cell["seed"],
                applied=applied)
            d = AR.augmented_atc_decider(est, inst, channel=cell["channel"])
            _s, log = DispatchEnv(inst).run_supervised(d, supervisor=sup,
                                                       method="m0_atc",
                                                       seed=cell["seed"])
            ds.add_log(log, inst, k_max=k_max)
            X, y, w = AR.weak_labels_from_log(log, inst, 5.0, 1.0)
            if len(X):
                Xa = np.concatenate([Xa, X]); ya = np.concatenate([ya, y])
                wa = np.concatenate([wa, w])
        train_estimator(est, Xa, ya, wa, device="cpu", seed=cell["seed"] + it)
    return ds


def val_curve(model, ds_tr, ds_va, objective, channel, lr, epochs, delta=1.0,
              seed=0, batch_size=512):
    """Train recording the per-epoch validation objective (lower is better)."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    best, best_state, best_ep = float("inf"), None, -1
    for ep in range(epochs):
        model.train()
        for b in ds_tr.batches(batch_size=batch_size, shuffle=True, rng=rng):
            w = b["w"]
            if objective == "mse":
                se, cnt = CE.mse_loss_terms(model, b)
                loss = (w * se).sum() / (w * cnt).sum().clamp_min(1e-8)
            else:
                ll = (CE.choice_logprob(model, b, channel) if objective == "choice"
                      else CE.tolerance_logprob(model, b, delta, channel))
                loss = (-w * ll).sum() / w.sum().clamp_min(1e-8)
            opt.zero_grad(); loss.backward(); opt.step()
        v = CE._epoch_objective(model, ds_va, objective, channel, delta)
        curve.append(float(v))
        if v < best - 1e-6:
            best, best_ep = v, ep
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"curve": curve, "best_val": best, "best_epoch": best_ep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--iters", type=int, default=3,
                    help="DAgger iterations used to build the fixed dataset")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(L.OUT, "diag.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    cell = L.build_cell(seed=a.seed)
    print("building the fixed decision stream (%d DAgger iterations) ..." % a.iters,
          flush=True)
    t0 = time.perf_counter()
    ds = collect_dataset(cell, iters=a.iters)
    val_ids = set(cell["train_ids"][L.N_TRAIN - L.N_VAL_INSTANCES:])
    ds_tr, ds_va = ds.split_by_instance(val_ids)
    print("dataset: %r  (%.0fs)" % (ds.counts(), time.perf_counter() - t0), flush=True)
    print("split: train=%d val=%d decisions" % (len(ds_tr), len(ds_va)), flush=True)

    # overrides-only view: the confirmation term removed entirely.
    ov_tr = ds_tr.subset([i for i in range(len(ds_tr)) if ds_tr.override[i]])
    ov_va = ds_va.subset([i for i in range(len(ds_va)) if ds_va.override[i]])
    print("overrides only: train=%d val=%d" % (len(ov_tr), len(ov_va)), flush=True)

    ch = cell["channel"]
    runs = []
    grid = []
    # (1) learning rate, per objective and per input set. Batch 512 matches the
    #     shipped latent_head.train_estimator exactly, so the squared-error rung
    #     of the ladder differs from the published one ONLY in split + early stop.
    for obj in ("mse", "choice"):
        for uq in (False, True):
            for lr in (1e-2, 3e-3, 1e-3):
                grid.append({"objective": obj, "use_queue": uq, "lr": lr,
                             "delta": None, "data": "all", "batch": 512,
                             "tau_fixed": False})
    # (2) does the choice loss simply want more gradient steps?
    for uq in (False, True):
        for lr in (1e-2, 3e-3):
            grid.append({"objective": "choice", "use_queue": uq, "lr": lr,
                         "delta": None, "data": "all", "batch": 128,
                         "tau_fixed": False})
    # (3) confirmations removed entirely: isolates "the confirmation term is the
    #     problem" from "the likelihood is the problem".
    for uq in (False, True):
        grid.append({"objective": "choice", "use_queue": uq, "lr": 3e-3,
                     "delta": None, "data": "overrides_only", "batch": 512,
                     "tau_fixed": False})
    # (4) the tolerance-aware confirmation term, the principled reading of a
    #     confirmation (and the only form in which a FIXED delta pins the scale).
    for d in (0.25, 1.0, 4.0):
        for uq in (False, True):
            grid.append({"objective": "tolerance", "use_queue": uq, "lr": 3e-3,
                         "delta": d, "data": "all", "batch": 512,
                         "tau_fixed": False})
    # (5) frozen choice scale: removes the (s_hat scale, tau) degeneracy, under
    #     which a conditional logit constrains only the RANKING and the shift's
    #     magnitude -- which clip(c - s, 1, 4) consumes -- floats free.
    for uq in (False, True):
        grid.append({"objective": "choice", "use_queue": uq, "lr": 3e-3,
                     "delta": None, "data": "all", "batch": 512,
                     "tau_fixed": True})

    for g in grid:
        tr_set, va_set = (ds_tr, ds_va) if g["data"] == "all" else (ov_tr, ov_va)
        torch.manual_seed(a.seed)
        model = CE.ChoiceModel(use_queue=g["use_queue"])
        if g["objective"] != "mse":
            CE.init_temperature(model, tr_set, channel=ch)
        if g.get("tau_fixed"):
            model.log_tau.requires_grad_(False)
        tau0 = model.tau_value()
        t1 = time.perf_counter()
        res = val_curve(model, tr_set, va_set, g["objective"], ch, g["lr"],
                        a.epochs, delta=(g["delta"] or 1.0), seed=a.seed,
                        batch_size=g["batch"])
        rec = L.recovery_metrics(model, cell["probe"], cell["overlay"], channel=ch)
        # held-out choice LL on the FULL validation decisions (never the test set)
        tau = CE.fit_temperature(model, ds_va, channel=ch)
        ll = CE.choice_loglik(model, ds_va, channel=ch, tau_override=tau)
        row = {**g, "tau_init": tau0, "tau_final": model.tau_value(),
               "tau_calibrated": tau, "best_val": res["best_val"],
               "best_epoch": res["best_epoch"], "curve": res["curve"],
               "pearson_r": rec["pearson_r"],
               "sign_acc_nonzero": rec["sign_acc_nonzero"],
               "mean_hat_s": rec["mean_hat_s"], "sd_hat_s": rec["sd_hat_s"],
               "val_choice_ll": ll["ll"], "val_ll_overrides": ll["ll_overrides"],
               "val_top1": ll["acc_top1"], "secs": time.perf_counter() - t1}
        runs.append(row)
        print("%-10s q=%-5s lr=%-6g b%-4d d=%-5s %-14s%s | bestep %2d val %8.4f | "
              "r %+.3f sign %.3f mean_s %+.3f sd_s %.3f | ll %+.3f llov %+.3f "
              "top1 %.3f tau %.2f->%.2f" %
              (g["objective"], g["use_queue"], g["lr"], g["batch"],
               g["delta"], g["data"], " taufix" if g.get("tau_fixed") else "",
               res["best_epoch"], res["best_val"], rec["pearson_r"],
               rec["sign_acc_nonzero"], rec["mean_hat_s"], rec["sd_hat_s"],
               ll["ll"], ll["ll_overrides"], ll["acc_top1"], tau0,
               model.tau_value()), flush=True)

    out = {"seed": a.seed, "iters": a.iters, "epochs": a.epochs,
           "dataset": ds.counts(), "n_train_dec": len(ds_tr),
           "n_val_dec": len(ds_va), "n_train_ov": len(ov_tr),
           "n_val_ov": len(ov_va), "runs": runs}
    with open(a.out + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print("\nwritten %s" % a.out)


if __name__ == "__main__":
    sys.exit(main())
