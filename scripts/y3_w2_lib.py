#!/usr/bin/env python
"""W2 shared machinery: cell construction, the four estimator variants, and the
common evaluation battery.

Nothing here forks a shipped module. The cell definition, the instance pools,
the overlay, the supervisor, the environment, the deployed augmented-ATC
decider, the true-objective validator and the recovery probe are all IMPORTED
from the shipped code; the only thing re-expressed is the ~20-line outer DAgger
loop of ``augmented_rule.run_m0``, which hard-wires its label constructor and
its fit routine and offers no hook for a different objective (the exact
one-function patch that would remove this duplication is named in the W2
report). The INCUMBENT rung never goes through the re-expressed loop: variant
``mse_published`` calls ``augmented_rule.run_m0`` verbatim, so the published
pipeline is reproduced, not re-implemented.

THE FOUR RUNGS
  mse_published  (i)    shipped weighted-squared-error fit on per-order features,
                        all 16 training instances, fixed 40 epochs. The published
                        M0 estimator; reproduced bit-for-bit by the pilot.
  mse_es         (i-es) same loss and same inputs, but the 12/4 instance split
                        and early stopping used by the choice rungs. This is the
                        PROTOCOL CONTROL: without it the (i) -> (ii) difference
                        would confound the objective with the fitting protocol.
  choice         (ii)   CLAIM A alone: conditional-logit likelihood over the
                        reviewed queue, inputs unchanged (per-order features).
  choice_queue   (iii)  CLAIMS A + B: the same likelihood with the feasible set
                        as an input through a parameter-free Deep-Sets encoder.
Robustness rungs: ``choice_tol`` / ``choice_queue_tol`` replace the
over-reading "a confirmation is a full choice" term with the tolerance-aware
one; ``*_k64`` caps the choice set at the plan's K <= 64.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

from fmwos import pdrs                                           # noqa: E402
from fmwos.env import DispatchEnv                                # noqa: E402
from fmwos.hitl import augmented_rule as AR                      # noqa: E402
from fmwos.hitl import deciders as dec                           # noqa: E402
from fmwos.hitl import overlay as ov                             # noqa: E402
from fmwos.hitl import true_objective as TO                      # noqa: E402
from fmwos.hitl.supervisor import Supervisor, _ATC_K             # noqa: E402
from fmwos.hitl import choice_estimator as CE                    # noqa: E402

try:
    from scipy.stats import wilcoxon as _wilcoxon
    _HAVE_SCIPY = True
except Exception:                                               # pragma: no cover
    _HAVE_SCIPY = False

_INST = os.path.join(_ROOT, "data", "processed", "instances")
OUT = os.path.join(_ROOT, "results", "y3_w2")
_TOL = 1e-9

# Locked headline-cell constants, copied from scripts/y3_p4_m0grid.py so a
# resolved-config diff against the published run is exact.
FAMILY = "F-NL"
MASTER_SEED = 12345
EPS = 0.0
THETA = 1.0
MECH = "targeted"
CHANNEL = "full_class_shift"
N_TRAIN, N_PROBE, N_EVAL = 16, 4, 10
M0_ITERS = 8
OVERRIDE_WEIGHT, CONFIRM_WEIGHT = 5.0, 1.0
N_VAL_INSTANCES = 4          # of the 16 training instances, for early stopping

VARIANTS = ("mse_published", "mse_es", "choice", "choice_queue",
            "choice_tol", "choice_queue_tol", "choice_queue_k64")


def set_threads(n=4):
    torch.set_num_threads(int(n))
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None


# --------------------------------------------------------------------------- #
# Cell                                                                        #
# --------------------------------------------------------------------------- #
def locate_files(campus, u, w="w80"):
    cdir = "c%02d" % campus
    return sorted(glob.glob(os.path.join(
        _INST, cdir, "storm2", w, "%s_storm2_%s_u%d_*.json" % (cdir, w, u))))


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def build_cell(campus=9, u=100, beta=1.0, rho=0.25, seed=301):
    """The headline cell, resolved exactly as scripts/y3_p4_m0grid.evaluate_cell."""
    files = locate_files(campus, u)
    need = N_TRAIN + N_PROBE + N_EVAL
    if len(files) < need:
        raise RuntimeError("only %d instance files, need %d" % (len(files), need))
    train = [_load(p) for p in files[:N_TRAIN]]
    probe = [_load(p) for p in files[N_TRAIN:N_TRAIN + N_PROBE]]
    eval_files = files[N_TRAIN + N_PROBE:N_TRAIN + N_PROBE + N_EVAL]
    assert not (set(eval_files) & set(files[:N_TRAIN + N_PROBE])), "eval overlaps train"
    evl = [_load(p) for p in eval_files]
    overlay = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                          master_seed=MASTER_SEED,
                                          channel=CHANNEL))
    assert overlay.params.channel == CHANNEL
    return {"campus": campus, "u": u, "beta": beta, "rho": rho, "seed": seed,
            "eps": EPS, "theta": THETA, "mech": MECH, "channel": CHANNEL,
            "family": FAMILY, "master_seed": MASTER_SEED,
            "n_train": N_TRAIN, "n_probe": N_PROBE, "n_eval": N_EVAL,
            "m0_iters": M0_ITERS, "train": train, "probe": probe, "eval": evl,
            "overlay": overlay,
            "train_ids": [i["meta"]["id"] for i in train],
            "probe_ids": [i["meta"]["id"] for i in probe],
            "eval_ids": [i["meta"]["id"] for i in evl],
            "files": {"train": files[:N_TRAIN],
                      "probe": files[N_TRAIN:N_TRAIN + N_PROBE],
                      "eval": eval_files}}


def resolved_config(cell, variant, extra=None):
    """The resolved configuration of a run, for the pre-launch config diff.

    Everything the estimator's result can depend on, EXCEPT the instance
    objects. Two runs of the ladder must differ in exactly one field: variant.
    """
    cfg = {k: cell[k] for k in ("campus", "u", "beta", "rho", "seed", "eps",
                                "theta", "mech", "channel", "family",
                                "master_seed", "n_train", "n_probe", "n_eval",
                                "m0_iters")}
    cfg.update({"variant": variant,
                "override_weight": OVERRIDE_WEIGHT,
                "confirm_weight": CONFIRM_WEIGHT,
                "n_val_instances": N_VAL_INSTANCES,
                "fit": dict(FIT), "lr_by_objective": dict(LR_BY_OBJECTIVE),
                "train_ids": cell["train_ids"], "probe_ids": cell["probe_ids"],
                "eval_ids": cell["eval_ids"],
                "torch": torch.__version__, "numpy": np.__version__})
    if extra:
        cfg.update(extra)
    return cfg


# --------------------------------------------------------------------------- #
# Variant training                                                            #
# --------------------------------------------------------------------------- #
# Fitting hyperparameters. Batch 512 and lr 1e-2 are the SHIPPED
# latent_head.train_estimator settings; the learning rate of each objective is
# selected on VALIDATION loss by scripts/y3_w2_diag.py and pinned here, so every
# rung is fitted at its own best rate rather than at the incumbent's.
# epochs 40 matches the shipped latent_head.train_estimator exactly (it runs
# 40 epochs per DAgger iteration); patience 8 early-stops within that budget.
FIT = {"batch_size": 512, "epochs": 40, "patience": 8, "delta": 1.0}
LR_BY_OBJECTIVE = {"mse": 1e-2, "choice": 1e-2, "tolerance": 3e-3}


def _variant_spec(variant):
    if variant not in VARIANTS:
        raise ValueError("unknown variant %r" % variant)
    obj = ("mse" if variant.startswith("mse")
           else ("tolerance" if variant.endswith("_tol") else "choice"))
    return {"use_queue": "queue" in variant, "objective": obj,
            "lr": LR_BY_OBJECTIVE[obj],
            "k_max": CE.K_MAX_PLAN if variant.endswith("_k64") else CE.K_MAX}


def wrap_shift_estimator(est):
    """Put a shipped ``ShiftEstimator`` inside a ``ChoiceModel`` for evaluation.

    ``QueueShiftEstimator(use_queue=False).core`` IS a ``ShiftEstimator`` of the
    same shape, so this is a state-dict move, not a re-fit: the incumbent is
    scored by exactly the same evaluation code as the choice rungs.
    """
    m = CE.ChoiceModel(use_queue=False)
    m.est.core.load_state_dict(est.state_dict())
    return m


def train_variant(cell, variant, verbose=False):
    """Train one rung. Returns {model, per_iter, fit, n_params, counts, secs}."""
    t0 = time.perf_counter()
    spec = _variant_spec(variant)
    train, probe, overlay = cell["train"], cell["probe"], cell["overlay"]
    seed, rho, eps, theta = cell["seed"], cell["rho"], cell["eps"], cell["theta"]
    channel = cell["channel"]

    # ---- rung (i): the PUBLISHED pipeline, called verbatim ------------------ #
    if variant == "mse_published":
        torch.manual_seed(seed)
        np.random.seed(seed)
        res = AR.run_m0(train, probe, overlay,
                        beta_rho_eps=(cell["beta"], rho, eps),
                        outer_iters=cell["m0_iters"], mechanism=cell["mech"],
                        theta=theta, override_weight=OVERRIDE_WEIGHT,
                        confirm_weight=CONFIRM_WEIGHT, seed=seed, device="cpu",
                        verbose=verbose)
        model = wrap_shift_estimator(res["estimator"])
        return {"model": model, "per_iter": res["per_iter"], "fit": [],
                "n_params_estimator": model.n_params_estimator(),
                "n_params_total": model.n_params_estimator(),   # no temperature
                "counts": {}, "secs": time.perf_counter() - t0,
                "dataset": None}

    # ---- rungs (i-es), (ii), (iii): re-expressed outer loop ------------------ #
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CE.ChoiceModel(use_queue=spec["use_queue"])
    ds = CE.ChoiceDataset()
    tables = {i["meta"]["id"]: CE.instance_tables(i) for i in train}
    val_ids = set(cell["train_ids"][N_TRAIN - N_VAL_INSTANCES:])
    rng = np.random.default_rng(seed)          # same stream as run_m0
    n_ep = len(train)
    per_iter, fits = [], []
    dp_cache = {}        # probe-instance decision points, collected once
    for it in range(cell["m0_iters"]):
        order = rng.permutation(len(train))[:n_ep]
        n_over = n_rev = n_conf = 0
        for k in order:
            inst = train[int(k)]
            applied = overlay.apply(inst)
            sup = CE.QueueLoggingSupervisor(overlay, inst, rho=rho, epsilon=eps,
                                            theta=theta, mechanism=cell["mech"],
                                            seed=seed, applied=applied)
            decider = make_decider(model, inst, channel,
                                   table=tables[inst["meta"]["id"]])
            _s, log = DispatchEnv(inst).run_supervised(
                decider, supervisor=sup, method="m0_atc", seed=seed)
            ds.add_log(log, inst, override_weight=OVERRIDE_WEIGHT,
                       confirm_weight=CONFIRM_WEIGHT, k_max=spec["k_max"])
            s = sup.summary()
            n_over += s["n_overrides"]; n_rev += s["n_reviews"]
            n_conf += s["n_confirmations"]
        ds_tr, ds_va = ds.split_by_instance(val_ids)
        if it == 0 and spec["objective"] != "mse":
            CE.init_temperature(model, ds_tr, channel=channel)
        fi = CE.fit_estimator(model, ds_tr, ds_va, objective=spec["objective"],
                              channel=channel, epochs=FIT["epochs"],
                              lr=spec["lr"], batch_size=FIT["batch_size"],
                              patience=FIT["patience"], delta=FIT["delta"],
                              seed=seed + it)
        fi.update({"iter": it, "n_train_dec": len(ds_tr), "n_val_dec": len(ds_va)})
        fits.append(fi)
        acc = recovery_metrics(model, probe, overlay, channel=channel,
                               seed=seed, dp_cache=dp_cache)
        orr = (n_over / n_rev) if n_rev else 0.0
        per_iter.append({"iter": it, "n_reviews": n_rev, "n_overrides": n_over,
                         "n_confirmations": n_conf, "override_rate": orr,
                         "n_examples_agg": len(ds), **acc})
        if verbose:
            print("[%s it%d] rev=%d over=%d orr=%.3f sign=%.3f r=%.3f "
                  "ep=%d/%d val=%.4f" % (variant, it, n_rev, n_over, orr,
                                         acc["sign_acc_nonzero"], acc["pearson_r"],
                                         fi["best_epoch"], fi["epochs_run"],
                                         fi["best_val"]))
    return {"model": model, "per_iter": per_iter, "fit": fits,
            "n_params_estimator": model.n_params_estimator(),
            "n_params_total": model.n_params(),
            "counts": ds.counts(), "secs": time.perf_counter() - t0,
            "dataset": ds, "val_ids": sorted(val_ids)}


def make_decider(model, instance, channel, table=None, stats=None):
    """The deployed augmented-ATC decider for a model.

    For a per-order model this is the SHIPPED ``augmented_rule.augmented_atc_
    decider`` on the wrapped ``ShiftEstimator``, so the deployment path of the
    choice rungs is byte-identical to the incumbent's; only the fitted weights
    differ. A queue-conditioned model has no static per-instance hat_s map, so
    it uses ``choice_estimator.queue_conditioned_atc_decider``, which computes
    the same index with hat_s recomputed from the live feasible set.
    """
    if model.use_queue:
        return CE.queue_conditioned_atc_decider(model, instance, channel=channel,
                                                table=table, stats=stats)
    return AR.augmented_atc_decider(model.est.core, instance, device="cpu",
                                    channel=channel)


# --------------------------------------------------------------------------- #
# Evaluation: recovery (EVAL-ONLY latent reads, quarantined here)             #
# --------------------------------------------------------------------------- #
def recovery_metrics(model, instances, overlay, channel=CHANNEL, seed=301,
                     dp_cache=None):
    """Pearson r / sign accuracy / exact-class accuracy of hat_s against s.

    Generalises ``augmented_rule.probe_shift_accuracy`` to a queue-conditioned
    estimator: hat_s for an order is averaged over the feasible sets in which
    that order actually appeared along the reference RULE(ATC) trajectory. For a
    per-order model hat_s does not depend on Q, so every average is over
    identical values and this returns EXACTLY the shipped metric (asserted in
    tests/test_choice_estimator.py).
    """
    hs_all, s_all = [], []
    for inst in instances:
        applied = overlay.apply(inst)                 # EVAL-ONLY latent read
        shift = applied["shift"]
        iid = inst["meta"]["id"]
        if dp_cache is not None and iid in dp_cache:
            tab, pts = dp_cache[iid]
        else:
            tab = CE.instance_tables(inst)
            pts = (collect_decision_points(inst, tab, seed=seed)
                   if model.use_queue else None)
            if dp_cache is not None:
                dp_cache[iid] = (tab, pts)
        hs = _mean_hat_s_per_order(model, inst, tab, decisions=pts)
        for wid, h in zip(tab.ids, hs):
            hs_all.append(float(h)); s_all.append(int(shift[wid]))
    hs_all = np.asarray(hs_all); s_all = np.asarray(s_all)
    nz = s_all != 0
    sign_acc = (float(np.mean(np.sign(hs_all[nz]) == np.sign(s_all[nz])))
                if nz.any() else float("nan"))
    pred = np.clip(np.round(hs_all), -2, 2).astype(int)
    r = (float(np.corrcoef(hs_all, s_all)[0, 1])
         if hs_all.std() > 1e-9 and s_all.std() > 1e-9 else float("nan"))
    return {"sign_acc_nonzero": sign_acc,
            "exact_class_acc": float(np.mean(pred == s_all)),
            "pearson_r": r, "zero_baseline_acc": float(np.mean(s_all == 0)),
            "mean_hat_s": float(hs_all.mean()), "sd_hat_s": float(hs_all.std()),
            "n_orders": int(s_all.size)}


def _mean_hat_s_per_order(model, instance, tab, decisions=None):
    if not model.use_queue:
        with torch.no_grad():
            return model.est(torch.as_tensor(tab.feats)).numpy()
    decisions = decisions if decisions is not None else collect_decision_points(instance, tab)
    tot = np.zeros(len(tab.ids), np.float64)
    cnt = np.zeros(len(tab.ids), np.float64)
    for chunk in _chunk(decisions, 64):
        k = max(len(d[0]) for d in chunk)
        feats = np.zeros((len(chunk), k, tab.feats.shape[1]), np.float32)
        mask = np.zeros((len(chunk), k), bool)
        for r, (rows, _now) in enumerate(chunk):
            feats[r, :len(rows)] = tab.feats[rows]; mask[r, :len(rows)] = True
        with torch.no_grad():
            hs = model.est(torch.as_tensor(feats), torch.as_tensor(mask)).numpy()
        for r, (rows, _now) in enumerate(chunk):
            np.add.at(tot, rows, hs[r, :len(rows)])
            np.add.at(cnt, rows, 1.0)
    return tot / np.maximum(cnt, 1.0)


def _chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --------------------------------------------------------------------------- #
# Evaluation: the common reference trajectory and Kendall tau                 #
# --------------------------------------------------------------------------- #
def collect_decision_points(instance, tab, seed=301):
    """Every (feasible set, clock) along the plain RULE(ATC) rollout.

    ALL variants are scored at the SAME decision points, so Kendall tau compares
    RANKINGS and not trajectories. Only decisions with >= 2 candidates count (a
    forced pick has no ranking to get right).
    """
    pts = []

    def _capture(queue, t, rng):
        if len(queue) >= 2:
            pts.append((np.asarray([tab.pos[j["id"]] for j in queue], np.int32),
                        float(t)))
        return pdrs.pick_with_margin("atc", queue, t, rng)

    DispatchEnv(instance).run_supervised(_capture, supervisor=None,
                                         method="atc", seed=seed)
    return pts


def true_atc_scores(instance, tab, decisions, overlay, applied=None):
    """The supervisor's TRUE-quantity ATC index at each decision point.

    EVAL-ONLY latent read, the same quarantine as ``probe_shift_accuracy``:
    ``w*`` and ``d*`` are read through a ``Supervisor`` bound to the overlay and
    are never seen by any estimator.
    """
    applied = applied if applied is not None else overlay.apply(instance)
    sup = Supervisor(overlay, instance, rho=0.0, applied=applied)
    wstar = np.asarray([sup.wstar[i] for i in tab.ids], np.float64)
    due = np.asarray([sup.due[i] for i in tab.ids], np.float64)
    out = []
    for rows, now in decisions:
        p = tab.p[rows].astype(np.float64)
        pbar = p.mean()
        slack = np.maximum(0.0, due[rows] - now - p)
        out.append((wstar[rows] / p) * np.exp(-slack / (_ATC_K * pbar)))
    return out


def corrected_scores_at(model, tab, decisions, channel=CHANNEL, zero=False):
    """The variant's corrected ATC index at each decision point (numpy path)."""
    out = []
    for chunk in _chunk(decisions, 64):
        if zero:
            hs_list = [np.zeros(len(rows), np.float32) for rows, _ in chunk]
        elif model.use_queue:
            k = max(len(d[0]) for d in chunk)
            feats = np.zeros((len(chunk), k, tab.feats.shape[1]), np.float32)
            mask = np.zeros((len(chunk), k), bool)
            for r, (rows, _n) in enumerate(chunk):
                feats[r, :len(rows)] = tab.feats[rows]; mask[r, :len(rows)] = True
            with torch.no_grad():
                hh = model.est(torch.as_tensor(feats), torch.as_tensor(mask)).numpy()
            hs_list = [hh[r, :len(rows)] for r, (rows, _n) in enumerate(chunk)]
        else:
            hs_list = None
        for r, (rows, now) in enumerate(chunk):
            if hs_list is None:
                with torch.no_grad():
                    hs = model.est(torch.as_tensor(tab.feats[rows])).numpy()
            else:
                hs = hs_list[r]
            p = tab.p[rows].astype(np.float64)
            pbar = p.mean()
            c_hat = np.clip(tab.prio[rows] - hs, 1.0, 4.0)
            w_corr = np.interp(c_hat, AR._CLASS_GRID, AR._W_GRID)
            if channel == "full_class_shift":
                d_corr = tab.rel[rows] + np.interp(c_hat, AR._CLASS_GRID, AR._SLA_GRID)
            else:
                d_corr = tab.due_rec[rows]
            slack = np.maximum(0.0, d_corr - now - p)
            out.append((w_corr / p) * np.exp(-slack / (_ATC_K * pbar)))
    return out


def kendall_on_instances(model, instances, overlay, channel=CHANNEL, zero=False,
                         seed=301, cache=None):
    """Mean Kendall tau-b between the corrected and the TRUE ranking.

    ``zero=True`` gives the RECORDED-field reference (hat_s == 0, i.e. what the
    deployed ATC rule ranks by), the honest floor this metric is read against.
    """
    from scipy.stats import kendalltau
    taus, n = [], 0
    for inst in instances:
        iid = inst["meta"]["id"]
        if cache is not None and iid in cache:
            tab, pts, tscores = cache[iid]
        else:
            tab = CE.instance_tables(inst)
            pts = collect_decision_points(inst, tab, seed=seed)
            tscores = true_atc_scores(inst, tab, pts, overlay)
            if cache is not None:
                cache[iid] = (tab, pts, tscores)
        cscores = corrected_scores_at(model, tab, pts, channel=channel, zero=zero)
        for cs, ts in zip(cscores, tscores):
            if len(ts) < 2:
                continue
            t = kendalltau(cs, ts).correlation
            if np.isfinite(t):
                taus.append(float(t)); n += 1
    if not taus:
        return {"kendall_tau": float("nan"), "n_decisions": 0}
    return {"kendall_tau": float(np.mean(taus)),
            "kendall_tau_sd": float(np.std(taus)), "n_decisions": n}


# --------------------------------------------------------------------------- #
# Evaluation: held-out choice log-likelihood                                  #
# --------------------------------------------------------------------------- #
def build_choice_testset(instances, overlay, cell, k_max=CE.K_MAX):
    """Reviewed decisions from RULE+SUP episodes on instances never trained on.

    RULE+SUP is the common protocol every variant's estimator would be fitted
    from at iteration 0, and it does not depend on the variant, so the test set
    is identical for all rungs.
    """
    ds = CE.ChoiceDataset()
    for inst in instances:
        applied = overlay.apply(inst)
        sup = CE.QueueLoggingSupervisor(overlay, inst, rho=cell["rho"],
                                        epsilon=cell["eps"], theta=cell["theta"],
                                        mechanism=cell["mech"], seed=cell["seed"],
                                        applied=applied)
        _s, log = dec.run_rule_sup(DispatchEnv(inst), "atc", sup, seed=cell["seed"])
        ds.add_log(log, inst, override_weight=OVERRIDE_WEIGHT,
                   confirm_weight=CONFIRM_WEIGHT, k_max=k_max)
    return ds


def zero_model():
    """A ChoiceModel whose hat_s is identically 0: the RECORDED-field reference."""
    m = CE.ChoiceModel(use_queue=False)
    with torch.no_grad():
        m.est.core.out.weight.zero_(); m.est.core.out.bias.zero_()
    return m


# --------------------------------------------------------------------------- #
# Evaluation: deployed true weighted tardiness                                #
# --------------------------------------------------------------------------- #
def deployed_twt(model, instances, overlay, channel=CHANNEL, seed=301,
                 measure_latency=False):
    """Per-instance TWT* of RULE(ATC) and of the augmented rule, same validator."""
    rule, aug = [], []
    stats = {"n": 0, "s": 0.0} if measure_latency else None
    for inst in instances:
        applied = overlay.apply(inst)

        def sc(sched):
            return TO.score_true(inst, sched, overlay, applied)["TWT_true"]

        rule.append(sc(dec.run_rule(DispatchEnv(inst), "atc", seed=seed)))
        d = make_decider(model, inst, channel, stats=stats)
        sched, _ = DispatchEnv(inst).run_supervised(d, supervisor=None,
                                                    method="m0", seed=seed)
        aug.append(sc(sched))
    out = {"rule": rule, "aug": aug}
    if stats and stats["n"]:
        out["ms_per_decision"] = 1000.0 * stats["s"] / stats["n"]
        out["n_decisions_timed"] = stats["n"]
    return out


def win_tie_loss(a, b, tol=_TOL):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = int(np.sum(a < b - tol)); l = int(np.sum(a > b + tol))
    return {"W": w, "T": int(len(a) - w - l), "L": l}


def paired_wilcoxon(a, b):
    """Two-sided paired Wilcoxon signed-rank p, zero_method='pratt' (the paper's)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a - b, 0.0):
        return 1.0
    if not _HAVE_SCIPY:
        return float("nan")
    try:
        return float(_wilcoxon(a, b, zero_method="pratt").pvalue)
    except Exception:
        return float("nan")


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    run = 0.0
    for r, i in enumerate(order):
        v = min(1.0, (m - r) * pvals[i])
        run = max(run, v)
        adj[i] = run
    return adj


def contrast(a, b):
    """Seed-averaged per-instance paired contrast, matching y3_p4_m0grid._contrast."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    am, bm = float(a.mean()), float(b.mean())
    return {"test_mean": am, "comparator_mean": bm,
            "pct_vs_comparator": (100.0 * (bm - am) / bm) if abs(bm) > 1e-12 else 0.0,
            "wtl": win_tie_loss(a, b), "wilcoxon_p": paired_wilcoxon(a, b),
            "n_instances": int(a.size)}
