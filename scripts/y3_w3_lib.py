#!/usr/bin/env python
"""W3 shared machinery: cell construction, the censored variants, and the
common evaluation battery.

Nothing here forks a shipped module. The cell definition, the instance pools,
the overlay, the supervisor, the environment, the deployed augmented-ATC
decider, the true-objective validator and the recovery probe are all IMPORTED
from the shipped code; the only thing re-expressed is the ~20-line outer DAgger
loop of ``augmented_rule.run_m0``, which hard-wires its label constructor and
its fit routine and offers no hook for a different likelihood. The INCUMBENT
rung never goes through the re-expressed loop: variant ``mse_published`` calls
``augmented_rule.run_m0`` verbatim, so the published pipeline is reproduced,
not re-implemented.

THE RUNGS
  mse_published   the shipped weighted-squared-error pipeline, called verbatim.
  mse_reexpr      the re-expressed loop with censoring switched OFF. The
                  BIT-EXACTNESS CONTROL: it must equal mse_published to the
                  last decimal, which is what licenses reading every other
                  difference as the likelihood and nothing else.
  tobit           two-limit censored (Tobit) likelihood, sigma fixed at 1,
                  textbook convention (an observation at a limit is censored),
                  plug-in deployment. THE PRIMARY.
  tobit_imp       the same, censoring ONLY the two structurally impossible
                  labels, so every attainable point label keeps its anchor.
  tobit_exp       tobit, deployed through the posterior-mean effective shift
                  E[clip(s, L, U)] instead of the plug-in mu.
  tobit_sig       tobit with the scale sigma fitted (1762 parameters).
  classmean_oracle  EVAL-ONLY REFERENCE, never a deliverable: the correction
                  that applies the TRUE population mean effective shift of each
                  recorded class. It measures how much of a beta = 0 reduction
                  a class-level constant can buy at all.
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
from fmwos.hitl import censored as CN                            # noqa: E402
from fmwos.hitl import deciders as dec                           # noqa: E402
from fmwos.hitl import overlay as ov                             # noqa: E402
from fmwos.hitl import true_objective as TO                      # noqa: E402
from fmwos.hitl.latent_head import ShiftEstimator                # noqa: E402
from fmwos.hitl.supervisor import Supervisor, _ATC_K             # noqa: E402

try:
    from scipy.stats import wilcoxon as _wilcoxon
    _HAVE_SCIPY = True
except Exception:                                               # pragma: no cover
    _HAVE_SCIPY = False

_INST = os.path.join(_ROOT, "data", "processed", "instances")
OUT = os.path.join(_ROOT, "results", "y3_w3")
_TOL = 1e-9

# Locked cell constants, copied from scripts/y3_p4_m0grid.py so a resolved-config
# diff against the published run is exact.
FAMILY = "F-NL"
MASTER_SEED = 12345
EPS = 0.0
THETA = 1.0
MECH = "targeted"
CHANNEL = "full_class_shift"
OVERRIDE_WEIGHT, CONFIRM_WEIGHT = 5.0, 1.0
N_PARAMS_INCUMBENT = 1761          # ShiftEstimator(lat_dim=20, hidden=32)

VARIANTS = ("mse_published", "mse_reexpr", "tobit", "tobit_imp", "tobit_exp",
            "tobit_sig", "classmean_oracle")

# variant -> (censor mode, deployment mapping, fitted sigma)
_SPEC = {
    "mse_reexpr": ("none", "plugin", False),
    "tobit": ("strict", "plugin", False),
    "tobit_imp": ("impossible", "plugin", False),
    "tobit_exp": ("strict", "expected", False),
    "tobit_sig": ("strict", "plugin", True),
}


def set_threads(n=1):
    """One numeric thread per process. Parallelism comes from processes only.

    The pipeline reproduces bit-exactly only at one intra-op thread: with more,
    the estimator refits with a different floating-point reduction order and the
    headline moves by more than a percentage point (measured in W2, +1.56 pp).
    """
    torch.set_num_threads(int(n))


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


def build_cell(campus, u, beta, rho=0.25, seed=301, n_train=16, n_probe=4,
               n_eval=10, m0_iters=8):
    """A cell, resolved exactly as scripts/y3_p4_m0grid.evaluate_cell."""
    files = locate_files(campus, u)
    need = n_train + n_probe + n_eval
    if len(files) < need:
        raise RuntimeError("only %d instance files, need %d" % (len(files), need))
    train = [_load(p) for p in files[:n_train]]
    probe = [_load(p) for p in files[n_train:n_train + n_probe]]
    eval_files = files[n_train + n_probe:n_train + n_probe + n_eval]
    assert not (set(eval_files) & set(files[:n_train + n_probe])), "eval overlaps train"
    evl = [_load(p) for p in eval_files]
    overlay = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                          master_seed=MASTER_SEED,
                                          channel=CHANNEL))
    assert overlay.params.channel == CHANNEL
    return {"campus": campus, "regime": "storm2", "u": u, "size": None,
            "beta": beta, "rho": rho, "seed": seed, "eps": EPS, "theta": THETA,
            "mech": MECH, "channel": CHANNEL, "family": FAMILY,
            "master_seed": MASTER_SEED, "n_train": n_train, "n_probe": n_probe,
            "n_eval": n_eval, "m0_iters": m0_iters,
            "train": train, "probe": probe, "eval": evl, "overlay": overlay,
            "train_ids": [i["meta"]["id"] for i in train],
            "probe_ids": [i["meta"]["id"] for i in probe],
            "eval_ids": [i["meta"]["id"] for i in evl],
            "files": {"train": files[:n_train],
                      "probe": files[n_train:n_train + n_probe],
                      "eval": eval_files}}


def resolved_config(cell, variant, extra=None):
    """Everything a run's result can depend on except the instance objects.

    Two runs on the same cell must differ in exactly one field: ``variant``.
    """
    cfg = {k: cell[k] for k in ("campus", "regime", "u", "size", "beta", "rho",
                                "seed", "eps", "theta", "mech", "channel",
                                "family", "master_seed", "n_train", "n_probe",
                                "n_eval", "m0_iters")}
    mode, deploy, fitsig = _SPEC.get(variant, ("-", "-", False))
    cfg.update({"variant": variant, "censor_mode": mode, "deploy": deploy,
                "fit_sigma": fitsig,
                "override_weight": OVERRIDE_WEIGHT,
                "confirm_weight": CONFIRM_WEIGHT,
                "est_hidden": 32, "epochs": 40, "lr": 1e-2, "batch_size": 512,
                "torch_threads": torch.get_num_threads(),
                "torch": torch.__version__, "numpy": np.__version__,
                "train_ids": cell["train_ids"], "probe_ids": cell["probe_ids"],
                "eval_ids": cell["eval_ids"]})
    if extra:
        cfg.update(extra)
    return cfg


def config_diff(a, b, ignore=("variant", "censor_mode", "deploy", "fit_sigma")):
    """Fields where two resolved configurations differ, excluding the ones the
    ladder is deliberately varying."""
    keys = set(a) | set(b)
    return {k: (a.get(k), b.get(k)) for k in sorted(keys)
            if k not in ignore and a.get(k) != b.get(k)}


# --------------------------------------------------------------------------- #
# The fitted object: one interface for every variant                          #
# --------------------------------------------------------------------------- #
class Fitted:
    """A fitted estimator plus the deployment mapping it is scored under."""

    def __init__(self, kind, est=None, deploy="plugin", const_by_class=None):
        self.kind = kind                    # "mse" | "censored" | "constant"
        self.est = est
        self.deploy = deploy
        self.const_by_class = const_by_class

    # -- what the estimator outputs (the "fitted shift" the paper quotes) ---- #
    def mu_map(self, instance) -> dict:
        if self.kind == "constant":
            return {w["id"]: float(self.const_by_class[int(w["priority"])])
                    for w in instance["work_orders"]}
        if self.kind == "mse":
            return AR.hat_s_map(self.est, instance)
        return CN.deployed_hat_s_map(self.est, instance, deploy="plugin")

    # -- what the decider is handed ----------------------------------------- #
    def hat_s_map(self, instance) -> dict:
        if self.kind == "censored":
            return CN.deployed_hat_s_map(self.est, instance, deploy=self.deploy)
        return self.mu_map(instance)

    # -- the deployed decider ------------------------------------------------ #
    def decider(self, instance, channel=CHANNEL):
        if self.kind == "mse" or (self.kind == "censored" and self.deploy == "plugin"):
            core = self.est if self.kind == "mse" else self.est.core
            # the SHIPPED decider, verbatim: the deployment path of the censored
            # plug-in rung is byte-identical to the incumbent's.
            return AR.augmented_atc_decider(core, instance, device="cpu",
                                            channel=channel)
        return CN.hat_s_map_atc_decider(self.hat_s_map(instance), instance,
                                        channel=channel)

    def sigma_value(self):
        return self.est.sigma_value() if self.kind == "censored" else float("nan")

    def n_params_estimator(self):
        if self.kind == "mse":
            return sum(p.numel() for p in self.est.parameters())
        if self.kind == "censored":
            return self.est.n_params_estimator()
        return 0

    def n_params_total(self):
        if self.kind == "censored":
            return self.est.n_params()
        return self.n_params_estimator()


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def train_variant(cell, variant, verbose=False):
    """Train one rung on one cell. Returns {model, per_iter, n_params, secs}."""
    t0 = time.perf_counter()
    if variant not in VARIANTS:
        raise ValueError("unknown variant %r" % variant)
    train, probe, overlay = cell["train"], cell["probe"], cell["overlay"]
    seed, rho, eps, theta = cell["seed"], cell["rho"], cell["eps"], cell["theta"]
    channel, mech = cell["channel"], cell["mech"]

    # ---- the EVAL-ONLY class-constant reference (no fitting at all) --------- #
    if variant == "classmean_oracle":
        const = true_class_mean_effective_shift(train + probe, overlay)
        model = Fitted("constant", const_by_class=const)
        return {"model": model, "per_iter": [], "censor": [],
                "n_params_estimator": 0, "n_params_total": 0,
                "secs": time.perf_counter() - t0, "class_constant": const}

    # ---- rung (i): the PUBLISHED pipeline, called verbatim ----------------- #
    if variant == "mse_published":
        torch.manual_seed(seed)
        np.random.seed(seed)
        res = AR.run_m0(train, probe, overlay,
                        beta_rho_eps=(cell["beta"], rho, eps),
                        outer_iters=cell["m0_iters"], mechanism=mech,
                        theta=theta, override_weight=OVERRIDE_WEIGHT,
                        confirm_weight=CONFIRM_WEIGHT, seed=seed, device="cpu",
                        verbose=verbose)
        model = Fitted("mse", est=res["estimator"], deploy="plugin")
        assert model.n_params_estimator() == N_PARAMS_INCUMBENT
        return {"model": model, "per_iter": res["per_iter"], "censor": [],
                "n_params_estimator": model.n_params_estimator(),
                "n_params_total": model.n_params_total(),
                "secs": time.perf_counter() - t0}

    # ---- censored rungs: the re-expressed outer loop ------------------------ #
    mode, deploy, fit_sigma = _SPEC[variant]
    torch.manual_seed(seed)
    np.random.seed(seed)
    est = CN.CensoredShiftEstimator(hidden=32, sigma=1.0, learn_sigma=fit_sigma)
    # PARAMETER-COUNT ASSERTION against the incumbent: a variant that quietly
    # grew cannot be compared, so abort before a single gradient step.
    if est.n_params_estimator() != N_PARAMS_INCUMBENT:
        raise SystemExit("parameter-count mismatch: estimator has %d, incumbent "
                         "has %d -- aborting" % (est.n_params_estimator(),
                                                 N_PARAMS_INCUMBENT))
    expect_total = N_PARAMS_INCUMBENT + (1 if fit_sigma else 0)
    if est.n_params() != expect_total:
        raise SystemExit("total parameter count %d != expected %d"
                         % (est.n_params(), expect_total))
    model = Fitted("censored", est=est, deploy=deploy)

    Xagg = np.zeros((0, CN.LAT_DIM), np.float32)
    yagg = np.zeros((0,), np.float32)
    wagg = np.zeros((0,), np.float32)
    cagg = np.zeros((0,), np.float32)
    per_iter, cens = [], []
    rng = np.random.default_rng(seed)          # same stream as run_m0
    n_ep = len(train)
    for it in range(cell["m0_iters"]):
        order = rng.permutation(len(train))[:n_ep]
        n_over = n_rev = n_conf = 0
        for k in order:
            inst = train[int(k)]
            applied = overlay.apply(inst)
            sup = Supervisor(overlay, inst, rho=rho, epsilon=eps, theta=theta,
                             mechanism=mech, seed=seed, applied=applied)
            decider = model.decider(inst, channel=channel)
            _s, log = DispatchEnv(inst).run_supervised(
                decider, supervisor=sup, method="m0_atc", seed=seed)
            X, y, w, c = CN.weak_labels_with_class(
                log, inst, override_weight=OVERRIDE_WEIGHT,
                confirm_weight=CONFIRM_WEIGHT)
            if len(X):
                Xagg = np.concatenate([Xagg, X]); yagg = np.concatenate([yagg, y])
                wagg = np.concatenate([wagg, w]); cagg = np.concatenate([cagg, c])
            s = sup.summary()
            n_over += s["n_overrides"]; n_rev += s["n_reviews"]
            n_conf += s["n_confirmations"]
        loss = CN.train_censored_estimator(est, Xagg, yagg, wagg, cagg, mode=mode,
                                           device="cpu", seed=seed + it)
        acc = probe_recovery(model, probe, overlay)
        cs = CN.censor_summary(yagg, cagg, mode=mode)
        cens.append(dict(cs, iter=it))
        orr = (n_over / n_rev) if n_rev else 0.0
        per_iter.append({"iter": it, "n_reviews": n_rev, "n_overrides": n_over,
                         "n_confirmations": n_conf, "override_rate": orr,
                         "n_examples_agg": int(len(Xagg)), "est_loss": loss,
                         "sigma": est.sigma_value(), **acc})
        if verbose:
            print("[%s it%d] rev=%d over=%d orr=%.3f sign=%.3f r=%+.3f "
                  "mean_hat_s=%+.4f mean_applied=%+.4f cens=%d/%d sigma=%.3f"
                  % (variant, it, n_rev, n_over, orr, acc["sign_acc_nonzero"],
                     acc["pearson_r"], acc["mean_hat_s"], acc["mean_applied_shift"],
                     cs["n_left"] + cs["n_right"], cs["n"], est.sigma_value()),
                  flush=True)
    return {"model": model, "per_iter": per_iter, "censor": cens,
            "n_params_estimator": model.n_params_estimator(),
            "n_params_total": model.n_params_total(),
            "secs": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# Evaluation: recovery (EVAL-ONLY latent reads, quarantined below)            #
# --------------------------------------------------------------------------- #
def probe_recovery(model, instances, overlay):
    """hat_s recovery against the TRUE shift. EVALUATION ONLY.

    Generalises ``augmented_rule.probe_shift_accuracy`` in two ways, both
    reported alongside the shipped numbers rather than in place of them:
      * ``mean_hat_s`` is the estimator's raw output, the quantity the
        manuscript quotes as the constant offset;
      * ``mean_applied_shift`` is ``clip(hat_s, c-4, c-1)``, the correction the
        corrected class actually applies -- the two differ exactly at the
        boundary classes, which is the whole subject of this package.
    Per-recorded-class means of both, and of the TRUE effective shift, are
    returned so the class-level bias can be read directly.
    """
    hs_all, ap_all, s_all, c_all, t_all = [], [], [], [], []
    for inst in instances:
        applied = overlay.apply(inst)               # EVAL-ONLY latent read
        shift = applied["shift"]
        cstar = applied["c_star"]
        mu = model.mu_map(inst)
        hsm = model.hat_s_map(inst)
        _ids, ap = CN.applied_shift(hsm, inst)
        for w, a in zip(inst["work_orders"], ap):
            wid = w["id"]
            hs_all.append(float(mu[wid])); ap_all.append(float(a))
            s_all.append(int(shift[wid])); c_all.append(int(w["priority"]))
            t_all.append(float(w["priority"]) - float(cstar[wid]))
    hs = np.asarray(hs_all); ap = np.asarray(ap_all)
    s = np.asarray(s_all); c = np.asarray(c_all); t = np.asarray(t_all)
    nz = s != 0
    sign_acc = (float(np.mean(np.sign(hs[nz]) == np.sign(s[nz]))) if nz.any()
                else float("nan"))
    pred = np.clip(np.round(hs), -2, 2).astype(int)
    r = (float(np.corrcoef(hs, s)[0, 1])
         if hs.std() > 1e-12 and s.std() > 1e-12 else float("nan"))
    r_ap = (float(np.corrcoef(ap, t)[0, 1])
            if ap.std() > 1e-12 and t.std() > 1e-12 else float("nan"))
    out = {"sign_acc_nonzero": sign_acc, "exact_class_acc": float(np.mean(pred == s)),
           "pearson_r": r, "pearson_r_applied_vs_true_effective": r_ap,
           "zero_baseline_acc": float(np.mean(s == 0)),
           "mean_hat_s": float(hs.mean()), "sd_hat_s": float(hs.std()),
           "mean_applied_shift": float(ap.mean()),
           "mean_true_effective_shift": float(t.mean()),
           "n_orders": int(s.size)}
    by = {}
    for k in (1, 2, 3, 4):
        m = c == k
        by[str(k)] = {"n": int(m.sum()),
                      "mean_hat_s": float(hs[m].mean()) if m.any() else float("nan"),
                      "sd_hat_s": float(hs[m].std()) if m.any() else float("nan"),
                      "mean_applied_shift": float(ap[m].mean()) if m.any() else float("nan"),
                      "mean_true_effective_shift": float(t[m].mean()) if m.any() else float("nan")}
    out["by_recorded_class"] = by
    return out


def true_class_mean_effective_shift(instances, overlay):
    """E[c - c* | recorded class c] over a pool. EVAL-ONLY LATENT READ.

    Used only to build the ``classmean_oracle`` upper reference, which measures
    how much a class-level constant can buy when it is exactly right. It is
    never fitted, never deployed as a deliverable, and never trained on.
    """
    tot = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    cnt = {1: 0, 2: 0, 3: 0, 4: 0}
    for inst in instances:
        applied = overlay.apply(inst)
        cstar = applied["c_star"]
        for w in inst["work_orders"]:
            c = int(w["priority"])
            tot[c] += c - float(cstar[w["id"]]); cnt[c] += 1
    return {c: (tot[c] / cnt[c] if cnt[c] else 0.0) for c in (1, 2, 3, 4)}


# --------------------------------------------------------------------------- #
# Evaluation: deployed true weighted tardiness                                #
# --------------------------------------------------------------------------- #
def deployed_twt(model, instances, overlay, channel=CHANNEL, seed=301):
    """Per-instance TWT* of RULE(ATC) and of the augmented rule, same validator."""
    rule, aug = [], []
    for inst in instances:
        applied = overlay.apply(inst)

        def sc(sched):
            return TO.score_true(inst, sched, overlay, applied)["TWT_true"]

        rule.append(sc(dec.run_rule(DispatchEnv(inst), "atc", seed=seed)))
        d = model.decider(inst, channel=channel)
        sched, _ = DispatchEnv(inst).run_supervised(d, supervisor=None,
                                                    method="m0", seed=seed)
        aug.append(sc(sched))
    return {"rule": rule, "aug": aug}


# --------------------------------------------------------------------------- #
# Evaluation: Kendall tau at a COMMON reference trajectory                    #
# --------------------------------------------------------------------------- #
def collect_decision_points(instance, seed=301):
    """Every (feasible-set ids, clock) along the plain RULE(ATC) rollout.

    All variants are scored at the SAME decision points, so Kendall tau compares
    RANKINGS and not trajectories. Only decisions with >= 2 candidates count.
    """
    pts = []

    def _capture(queue, t, rng):
        if len(queue) >= 2:
            pts.append(([j["id"] for j in queue], float(t)))
        return pdrs.pick_with_margin("atc", queue, t, rng)

    DispatchEnv(instance).run_supervised(_capture, supervisor=None, method="atc",
                                         seed=seed)
    return pts


def _atc_index(w, p, due, now, k=_ATC_K):
    p = np.asarray(p, float)
    pbar = p.mean()
    slack = np.maximum(0.0, np.asarray(due, float) - now - p)
    return (np.asarray(w, float) / p) * np.exp(-slack / (k * pbar))


def kendall_on_instances(model, instances, overlay, channel=CHANNEL, seed=301,
                         zero=False, cache=None):
    """Mean Kendall tau-b between the corrected ranking and the TRUE ranking.

    ``zero=True`` gives the RECORDED-field reference (hat_s == 0), the honest
    floor this metric is read against.
    """
    from scipy.stats import kendalltau
    taus, n = [], 0
    for inst in instances:
        iid = inst["meta"]["id"]
        if cache is not None and iid in cache:
            pts, tab, tscores = cache[iid]
        else:
            pts = collect_decision_points(inst, seed=seed)
            wo = {w["id"]: w for w in inst["work_orders"]}
            applied = overlay.apply(inst)                # EVAL-ONLY latent read
            sup = Supervisor(overlay, inst, rho=0.0, applied=applied)
            tab = {"wo": wo,
                   "wstar": {i: float(sup.wstar[i]) for i in wo},
                   "dstar": {i: float(sup.due[i]) for i in wo}}
            tscores = [_atc_index([tab["wstar"][i] for i in ids],
                                  [float(wo[i]["p_bh"]) for i in ids],
                                  [tab["dstar"][i] for i in ids], now)
                       for ids, now in pts]
            if cache is not None:
                cache[iid] = (pts, tab, tscores)
        wo = tab["wo"]
        hsm = ({i: 0.0 for i in wo} if zero else model.hat_s_map(inst))
        for (ids, now), ts in zip(pts, tscores):
            if len(ts) < 2:
                continue
            p = [float(wo[i]["p_bh"]) for i in ids]
            ce = [min(4.0, max(1.0, float(wo[i]["priority"]) - hsm.get(i, 0.0)))
                  for i in ids]
            w_c = np.interp(ce, AR._CLASS_GRID, AR._W_GRID)
            if channel == "full_class_shift":
                d_c = np.asarray([float(wo[i]["release_bh"]) for i in ids]) + \
                    np.interp(ce, AR._CLASS_GRID, AR._SLA_GRID)
            else:
                d_c = np.asarray([float(wo[i]["due_bh"]) for i in ids])
            cs = _atc_index(w_c, p, d_c, now)
            tt = kendalltau(cs, ts).correlation
            if np.isfinite(tt):
                taus.append(float(tt)); n += 1
    if not taus:
        return {"kendall_tau": float("nan"), "n_decisions": 0}
    return {"kendall_tau": float(np.mean(taus)),
            "kendall_tau_sd": float(np.std(taus)), "n_decisions": n}


# --------------------------------------------------------------------------- #
# Statistics (identical settings to y3_p4_m0grid)                             #
# --------------------------------------------------------------------------- #
def win_tie_loss(a, b, tol=_TOL):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = int(np.sum(a < b - tol)); l = int(np.sum(a > b + tol))
    return {"W": w, "T": int(len(a) - w - l), "L": l}


def paired_wilcoxon(a, b):
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


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #
def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(path + ".tmp", path)
