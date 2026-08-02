#!/usr/bin/env python
"""Paper Y3, W10 + W11 -- which parts of the correction layer are load-bearing?

Two cheap robustness experiments on the SHIPPED deployable protocol, run through
``scripts/y3_w1_sweep.evaluate_cell`` verbatim, with the locked configuration
tables imported from ``scripts/y3_w9_overlay_worlds.py`` so this runner is
checked against the same table the incumbent folds were produced under.

W11 -- OVERRIDE-WEIGHT SENSITIVITY. The weak labels harvested from the override
log weight an override five times a confirmation. The method section discloses
that five is a setting we chose rather than tuned, with no sensitivity sweep.
This sweeps it over {1, 2, 5, 10}. The parameter lives in
``augmented_rule.weak_labels_from_log(..., override_weight=5.0, ...)`` and is
forwarded by ``routing.run_m0_routed(..., override_weight=5.0, ...)``, which
``y3_w1_sweep.evaluate_cell`` calls without naming it, so the shipped value is
the function default. It is threaded here by wrapping ``routing.run_m0_routed``
in a pass-through that injects the requested weight and by wrapping
``routing.weak_labels_from_log`` in a pass-through that PROVES the weight
arrived: it records the value every call received and the distinct sample
weights every call returned, and the fold aborts unless they match.

W10 -- GRADIENT-BOOSTED-TREE ESTIMATOR. Replaces ONLY the estimator with
``sklearn.ensemble.HistGradientBoostingRegressor`` under fixed hyperparameters
stated in results/y3_w10_w11/RUN_PLAN.md before any result was seen. No tuning
loop of any kind. Weak labels, the DAgger loop, the clip, the interpolated
class->weight and class->deadline curves, the split-conformal band on absolute
errors and the stability routing are untouched: the wrapper exposes the single
method the rest of the pipeline calls on an estimator, ``predict_np``. Nothing
in ``src/`` is edited; ``routing.ShiftEstimator`` and ``routing.train_estimator``
are rebound in this script for the duration of a W10 fold.

CACHE SAFETY. Neither the override weight nor the estimator family is a field of
``y3_w1_sweep._SIG_KEYS``, so every variant of one cell/seed shares one cache
signature. Two independent defences: the module-level cache is redirected to a
PER-VARIANT directory, and every fold filename carries a variant-extended key
``sha1(cell signature + variant descriptor)``. The runner asserts the variants
produce distinct keys before running anything.

THREADS. Hard-set to one before numpy/torch/sklearn import (importing
y3_w9_overlay_worlds does this first, and it is repeated here for readers).
Every worker re-asserts the caps, ``torch.get_num_threads() == 1`` and
scikit-learn's effective OpenMP thread count. Parallelism comes from separate
processes. No wall-clock number produced by this run is reported as a
measurement of anything.

Run:
    PYTHONPATH=src taskset -c 0-9 python scripts/y3_w10_w11_variants.py --pilot
    PYTHONPATH=src taskset -c 0-9 python scripts/y3_w10_w11_variants.py --workers 6
"""

from __future__ import annotations

import os

# HARD, not setdefault: y3_w1_sweep's module-level setdefault would put four
# threads in every worker, which changes the floating-point reduction order.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse                                                    # noqa: E402
import glob                                                        # noqa: E402
import hashlib                                                     # noqa: E402
import json                                                        # noqa: E402
import sys                                                         # noqa: E402
import time                                                        # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed   # noqa: E402

import numpy as np                                                 # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import sklearn                                                     # noqa: E402
import torch                                                       # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor         # noqa: E402
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads  # noqa: E402

from fmwos.hitl import overlay as ov                               # noqa: E402
from fmwos.hitl import routing as R                                # noqa: E402
from fmwos.hitl.latent_head import LAT_DIM                         # noqa: E402

import y3_w1_sweep as W1                                           # noqa: E402
import y3_w9_overlay_worlds as W9                                  # noqa: E402

_OUT = os.path.join(_ROOT, "results", "y3_w10_w11")
_FOLDS = os.path.join(_OUT, "folds")
_CACHE_ROOT = os.path.join(_OUT, "cache")
_W9FOLDS = os.path.join(_ROOT, "results", "y3_w9", "folds")
_COEFF_DIR = os.path.join(_ROOT, "results", "y3_p1", "overlay_coeffs")
_P9CHECKS = os.path.join(_ROOT, "results", "y3_p9", "data_checks.json")
_HARVEST = os.path.join(_ROOT, "results", "y3_p5", "harvest",
                        "primary_multiseed_summary.json")

# Caches that must never be written to by this runner.
_FORBIDDEN_CACHES = {os.path.join(_ROOT, "results", "y3_w1", "cache"),
                     os.path.join(_ROOT, "results", "y3_w9", "cache"),
                     os.path.join(_ROOT, "results", "y3_p4", "cache")}

# --------------------------------------------------------------------------- #
# The varied inputs                                                            #
# --------------------------------------------------------------------------- #
WORLD = W9.PUBLISHED_WORLD                 # 12345, the published hidden world
SEEDS = (301, 302, 303)
RHO = W9.RHO                               # 0.25
CELLS = [W9.CELL_OF["A"], W9.CELL_OF["B"]]  # headline ceiling; corpus-anchored
CELL_OF = {c["key"]: c for c in CELLS}

# The shipped setting of the estimator-fitting path, and the sweep over it.
SHIPPED_OVERRIDE_WEIGHT = 5.0
SHIPPED_CONFIRM_WEIGHT = 1.0
OVERRIDE_WEIGHTS = (1.0, 2.0, 5.0, 10.0)

# W10 hyperparameters. FIXED and stated in RUN_PLAN.md before any result was
# seen. No tuning loop of any kind; nothing here was changed after a result.
GBT_PARAMS = dict(max_iter=200, learning_rate=0.1, max_depth=None,
                  max_leaf_nodes=31, min_samples_leaf=20,
                  l2_regularization=0.0, early_stopping=False)

NPARAMS_EXPECTED = W9.NPARAMS_EXPECTED     # 1761, the shipped ShiftEstimator

# The shipped bindings, captured once at import, BEFORE anything is wrapped.
# A pooled worker process runs many folds, so every fold restores these first:
# without that, a worker that has run a tree fold would fit trees for the next
# neural fold it is handed, silently.
_ORIG = {"ShiftEstimator": R.ShiftEstimator,
         "train_estimator": R.train_estimator,
         "run_m0_routed": R.run_m0_routed,
         "weak_labels_from_log": R.weak_labels_from_log,
         "make_supervisor": R.make_supervisor}


def _restore_shipped_bindings():
    for k, v in _ORIG.items():
        setattr(R, k, v)


# --------------------------------------------------------------------------- #
# Variants                                                                     #
# --------------------------------------------------------------------------- #
def variant_id(experiment, estimator, override_weight):
    if experiment == "W11":
        return "W11_ow%g" % override_weight
    return "W10_gbt_ow%g" % override_weight


VARIANTS = ([dict(experiment="W11", estimator="neural", override_weight=w,
                  vid=variant_id("W11", "neural", w)) for w in OVERRIDE_WEIGHTS]
            + [dict(experiment="W10", estimator="gbt",
                    override_weight=SHIPPED_OVERRIDE_WEIGHT,
                    vid=variant_id("W10", "gbt", SHIPPED_OVERRIDE_WEIGHT))])
VARIANT_OF = {v["vid"]: v for v in VARIANTS}

# The incumbent: the shipped layer at the shipped weight, i.e. the W11 weight-5
# arm, whose folds already exist in results/y3_w9.
INCUMBENT_VID = variant_id("W11", "neural", SHIPPED_OVERRIDE_WEIGHT)


# --------------------------------------------------------------------------- #
# Resolved configuration and the config diff against the incumbent             #
# --------------------------------------------------------------------------- #
def resolved_config(task, variant):
    """The FULL resolved configuration of one fold: every task field the cache
    signature covers, plus every estimator-fitting setting the task dict does
    not carry. This is the object the config diff is taken on."""
    cfg = {("task.%s" % k): task[k] for k in W9.LOCKED_SIG_KEYS}
    cfg["task.theta"] = task["theta"]
    cfg["fit.override_weight"] = float(variant["override_weight"])
    cfg["fit.confirm_weight"] = SHIPPED_CONFIRM_WEIGHT
    cfg["fit.label_source"] = "executed"          # weak_labels_from_log default
    if variant["estimator"] == "neural":
        cfg["fit.estimator_family"] = "ShiftEstimator (torch MLP)"
        cfg["fit.estimator_arch"] = "lat_dim=%d, hidden=32, 2 hidden layers" % LAT_DIM
        cfg["fit.estimator_nparams"] = NPARAMS_EXPECTED
        cfg["fit.estimator_training"] = ("weighted MSE, Adam lr=1e-2, epochs=40, "
                                         "batch=512, warm-started across DAgger "
                                         "iterations")
    else:
        cfg["fit.estimator_family"] = "HistGradientBoostingRegressor (sklearn)"
        cfg["fit.estimator_arch"] = ", ".join(
            "%s=%r" % (k, GBT_PARAMS[k]) for k in sorted(GBT_PARAMS))
        cfg["fit.estimator_nparams"] = None
        cfg["fit.estimator_training"] = ("weighted squared error via "
                                         "sample_weight, random_state=seed+iter, "
                                         "refit from scratch on the full "
                                         "aggregate each DAgger iteration")
    return cfg


def config_diff(task, variant):
    """Fields on which this variant's resolved configuration differs from the
    incumbent's at the same cell and seed."""
    inc = resolved_config(task, VARIANT_OF[INCUMBENT_VID])
    got = resolved_config(task, variant)
    if set(inc) != set(got):
        raise SystemExit("resolved-config key sets differ: %r"
                         % sorted(set(inc) ^ set(got)))
    return {k: {"incumbent": inc[k], "variant": got[k]}
            for k in sorted(inc) if inc[k] != got[k]}


# W11 may differ from the incumbent in the override weight and NOTHING else;
# W10 may differ in the estimator block and NOTHING else.
_ALLOWED_DIFF = {
    "W11": {"fit.override_weight"},
    "W10": {"fit.estimator_family", "fit.estimator_arch",
            "fit.estimator_nparams", "fit.estimator_training"},
}


def assert_config_diff(task, variant):
    d = config_diff(task, variant)
    keys = set(d)
    if variant["vid"] == INCUMBENT_VID:
        if keys:
            raise SystemExit("the weight-5 arm IS the incumbent but its resolved "
                             "configuration differs on %r" % sorted(keys))
        return d
    allowed = _ALLOWED_DIFF[variant["experiment"]]
    if keys != allowed:
        raise SystemExit("CONFIG DIFF against the incumbent is %r, the ONLY "
                         "intended change for %s is %r"
                         % (sorted(keys), variant["vid"], sorted(allowed)))
    return d


# --------------------------------------------------------------------------- #
# Tasks, keys, folds                                                           #
# --------------------------------------------------------------------------- #
def make_task(cell, seed, variant):
    """The incumbent's task, verbatim. The variant changes NO task field: the
    override weight and the estimator family live in the fitting path, not in
    the cell definition, which is precisely why the cache key must be
    variant-extended below."""
    t = W9.make_task(cell, WORLD, seed)          # asserts the locked defaults
    t["part"] = "W10_W11"
    t["variant"] = variant["vid"]
    t["experiment"] = variant["experiment"]
    return t


def variant_key(task, variant):
    """Cache/fold key = the cell signature EXTENDED by the variant descriptor.

    ``y3_w1_sweep._cell_sig`` covers the cell only, so four override weights at
    one cell and seed collide on one signature. This key cannot."""
    desc = json.dumps({"experiment": variant["experiment"],
                       "estimator": variant["estimator"],
                       "override_weight": float(variant["override_weight"]),
                       "confirm_weight": SHIPPED_CONFIRM_WEIGHT,
                       "gbt": GBT_PARAMS if variant["estimator"] == "gbt" else None},
                      sort_keys=True)
    return hashlib.sha1((W1._cell_sig(task) + "|" + desc).encode()).hexdigest()[:16]


def fold_path(task, variant):
    return os.path.join(_FOLDS, "%s__%s__world%d__seed%d__%s.json"
                        % (variant["vid"], task["cell_name"], WORLD,
                           task["seed"], variant_key(task, variant)))


def existing_fold(task, variant):
    stem = "%s__%s__world%d__seed%d__" % (variant["vid"], task["cell_name"],
                                          WORLD, task["seed"])
    hits = sorted(glob.glob(os.path.join(_FOLDS, stem + "*.json")))
    want = fold_path(task, variant)
    stale = [h for h in hits if h != want]
    if stale:
        raise SystemExit("CACHE-KEY DRIFT: %s exists under a different "
                         "configuration signature (%s). Delete it or explain "
                         "the change."
                         % (stem, [os.path.basename(s) for s in stale]))
    return want if os.path.exists(want) else None


def w9_fold(cell_key, seed):
    """The committed incumbent fold for one cell and seed. Read-only."""
    cell = W9.CELL_OF[cell_key]
    pat = os.path.join(_W9FOLDS, "%s__world%d__seed%d__*.json"
                       % (W9.cell_name(cell), WORLD, seed))
    hits = sorted(glob.glob(pat))
    if len(hits) != 1:
        raise SystemExit("expected exactly one incumbent fold at %s, found %d"
                         % (pat, len(hits)))
    with open(hits[0]) as fh:
        return json.load(fh), hits[0]


# --------------------------------------------------------------------------- #
# The gradient-boosted-tree estimator (W10)                                    #
# --------------------------------------------------------------------------- #
class GBTShiftEstimator:
    """A drop-in for ``ShiftEstimator`` exposing the ONE method the rest of the
    pipeline calls on an estimator: ``predict_np(feats, device=...)``.

    The augmented decider (``augmented_rule.hat_s_map``), the per-order band
    (``routing.band_for_instance``), the split-conformal calibration
    (``routing.calibrate_band``), the verdict stream and the evaluation-only
    recovery metrics all reach the estimator through that single method, so
    nothing else in the shipped modules needs to know which family fitted it.

    Deliberately carries neither ``apply`` nor ``params``, so
    ``routing._forbid_latent`` treats it exactly as it treats the shipped
    estimator.

    Before the first fit it predicts exactly zero, so the first DAgger iteration
    runs the plain tuned rule. The neural estimator instead starts at random
    initialisation. That difference is stated in RUN_PLAN.md; it cannot be
    removed, because an unfitted tree ensemble has no state to perturb with.
    """

    kind = "gbt"

    def __init__(self, lat_dim: int = LAT_DIM, hidden: int = 32):
        self.lat_dim = int(lat_dim)
        self.hidden = int(hidden)       # accepted and ignored: no such knob here
        self.model = None
        self.fits = []                  # one record per DAgger iteration

    # -- the estimator interface the pipeline uses ------------------------- #
    def predict_np(self, feats, device="cpu") -> np.ndarray:
        X = np.asarray(feats, dtype=np.float32)
        if X.size == 0:
            return np.zeros((0,), dtype=np.float64)
        if self.model is None:
            return np.zeros((X.shape[0],), dtype=np.float64)
        return self.model.predict(X).astype(np.float64)

    # -- fitting ----------------------------------------------------------- #
    def fit(self, X, y, w, seed):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        m = HistGradientBoostingRegressor(random_state=int(seed), **GBT_PARAMS)
        m.fit(X, y, sample_weight=w)
        self.model = m
        pred = m.predict(X).astype(np.float64)
        loss = float((w * (pred - y) ** 2).sum() / max(w.sum(), 1e-8))
        self.fits.append({"n_examples": int(X.shape[0]),
                          "n_iter": int(m.n_iter_),
                          "n_trees_per_iteration": int(m.n_trees_per_iteration_),
                          "n_trees_total": int(m.n_iter_) * int(m.n_trees_per_iteration_),
                          "weighted_mse": loss,
                          "random_state": int(seed)})
        return loss


def gbt_train_estimator(estimator, X, y, w, *, epochs=40, lr=1e-2,
                        batch_size=512, device="cpu", seed=0):
    """``latent_head.train_estimator``'s signature, fitting trees instead.

    The neural training knobs are accepted and ignored; they have no analogue
    here, and the tree hyperparameters are the fixed ones in ``GBT_PARAMS``.
    Returns the weighted mean squared error on the fitted examples, which is the
    same quantity the neural routine returns (its last minibatch loss), so the
    per-iteration record keeps its meaning."""
    if len(X) == 0:
        return float("nan")
    return estimator.fit(X, y, w, seed)


# --------------------------------------------------------------------------- #
# Pass-through assertions: the weight is threaded, nothing else moved          #
# --------------------------------------------------------------------------- #
_SEEN = {"nparams": [], "override_weight_calls": [], "returned_weights": set(),
         "gbt_fits": [], "estimator_kind": set()}


def _install_weak_label_probe(expect_ow):
    """PROOF that the override weight reached the weak-label builder.

    ``routing`` imported ``weak_labels_from_log`` into its own namespace, and
    calls it positionally, so this pass-through sees the value
    ``run_m0_routed`` actually forwarded."""
    orig = _ORIG["weak_labels_from_log"]

    def checked(log, instance, override_weight=5.0, confirm_weight=1.0,
                label_source="executed"):
        if float(override_weight) != float(expect_ow):
            raise SystemExit("OVERRIDE WEIGHT NOT THREADED: weak_labels_from_log "
                             "received %r, the variant asked for %r"
                             % (override_weight, expect_ow))
        if float(confirm_weight) != SHIPPED_CONFIRM_WEIGHT:
            raise SystemExit("confirmation weight drifted to %r" % (confirm_weight,))
        if label_source != "executed":
            raise SystemExit("label_source drifted to %r" % (label_source,))
        X, y, w = orig(log, instance, override_weight, confirm_weight,
                       label_source)
        _SEEN["override_weight_calls"].append(float(override_weight))
        if len(w):
            bad = set(np.unique(w).tolist()) - {float(expect_ow),
                                                SHIPPED_CONFIRM_WEIGHT}
            if bad:
                raise SystemExit("weak-label sample weights outside "
                                 "{%r, %r}: %r" % (expect_ow,
                                                   SHIPPED_CONFIRM_WEIGHT, bad))
            _SEEN["returned_weights"] |= set(np.unique(w).tolist())
        return X, y, w

    R.weak_labels_from_log = checked


def _install_run_m0_wrapper(expect_ow, estimator_kind):
    """Inject the override weight into every ``run_m0_routed`` call, and assert
    the fitted estimator is the one this variant asked for."""
    orig = _ORIG["run_m0_routed"]

    def wrapped(*a, **kw):
        if "override_weight" in kw:
            raise SystemExit("the caller already set override_weight; the "
                             "wrapper would mask a change made elsewhere")
        kw = dict(kw)
        kw["override_weight"] = float(expect_ow)
        res = orig(*a, **kw)
        est = res["estimator"]
        if estimator_kind == "neural":
            n = sum(p.numel() for p in est.parameters())
            if n != NPARAMS_EXPECTED:
                raise SystemExit("PARAM DRIFT: the fitted shift estimator has %d "
                                 "parameters, the shipped layer has %d"
                                 % (n, NPARAMS_EXPECTED))
            _SEEN["nparams"].append(n)
            _SEEN["estimator_kind"].add("neural")
        else:
            if not isinstance(est, GBTShiftEstimator):
                raise SystemExit("ESTIMATOR DRIFT: W10 fitted a %s"
                                 % type(est).__name__)
            if not est.fits:
                raise SystemExit("the tree estimator was never fitted")
            _SEEN["gbt_fits"] = list(est.fits)
            _SEEN["estimator_kind"].add("gbt")
        return res

    R.run_m0_routed = wrapped


def _install_gbt_estimator():
    """Rebind the estimator constructor and its training routine INSIDE the
    routing module for the duration of a W10 fold. ``src/`` is not edited, and
    ``_restore_shipped_bindings`` puts the shipped names back before the next
    fold this worker process is handed."""
    R.ShiftEstimator = GBTShiftEstimator
    R.train_estimator = gbt_train_estimator


def _assert_threads(where):
    if torch.get_num_threads() != 1:
        raise SystemExit("%s: torch threads = %d, expected 1"
                         % (where, torch.get_num_threads()))
    n = _openmp_effective_n_threads()
    if n != 1:
        raise SystemExit("%s: scikit-learn effective OpenMP threads = %d, "
                         "expected 1" % (where, n))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(v) != "1":
            raise SystemExit("%s: %s = %r, expected '1'"
                             % (where, v, os.environ.get(v)))


# --------------------------------------------------------------------------- #
# Worker                                                                       #
# --------------------------------------------------------------------------- #
def _repro_vs_incumbent(task, rec):
    """For the weight-5 arm only: bit-equality against the committed W9 fold.

    This is the verification that the sweep's weight-5 column and the incumbent
    are the same computation, so the incumbent need not be rerun."""
    pub, path = w9_fold(task["cell_key"], task["seed"])
    if pub["inst_ids"] != rec["inst_ids"]:
        raise SystemExit("held-out ids differ from the incumbent fold %s" % path)
    out = {"incumbent_fold": os.path.relpath(path, _ROOT), "max_abs_diff": {}}
    worst = 0.0
    for k in W9.DECIDERS:
        a, b = rec["per"][k], pub["per"][k]
        if len(a) != len(b):
            raise SystemExit("decider %s length differs from the incumbent" % k)
        d = max(abs(x - y) for x, y in zip(a, b))
        out["max_abs_diff"][k] = d
        worst = max(worst, d)
    out["bit_exact"] = (worst == 0.0)
    if not out["bit_exact"]:
        raise SystemExit("BIT-COMPATIBILITY GATE FAILED: the weight-5 arm "
                         "differs from the incumbent by up to %r; the W9 cache "
                         "may NOT be reused as the weight-5 column" % worst)
    return out


def worker(job):
    task, variant = job["task"], job["variant"]
    torch.set_num_threads(1)
    _assert_threads("worker")

    cache = os.path.join(_CACHE_ROOT, variant["vid"])
    if os.path.abspath(cache) in {os.path.abspath(p) for p in _FORBIDDEN_CACHES}:
        raise SystemExit("refusing to write a published cache")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(_FOLDS, exist_ok=True)
    W1._CACHE = cache                       # never read or write a published cache

    for k in ("nparams", "override_weight_calls", "gbt_fits"):
        _SEEN[k] = []
    _SEEN["returned_weights"] = set()
    _SEEN["estimator_kind"] = set()

    assert_config_diff(task, variant)
    _restore_shipped_bindings()             # a pooled worker runs many variants
    W9._install_policy_assertion()          # stability routing, train and eval
    _install_weak_label_probe(variant["override_weight"])
    _install_run_m0_wrapper(variant["override_weight"], variant["estimator"])
    if variant["estimator"] == "gbt":
        _install_gbt_estimator()
    else:
        if R.ShiftEstimator is not _ORIG["ShiftEstimator"] or \
                R.train_estimator is not _ORIG["train_estimator"]:
            raise SystemExit("a tree binding survived into a neural fold")

    t0 = time.perf_counter()
    rec = W1.evaluate_cell(task)
    wall = time.perf_counter() - t0

    if rec["policy"] != "stability" or bool(rec["split_fit"]) is not True:
        raise SystemExit("the record was not produced by the deployable policy")
    if rec["band"] is None:
        raise SystemExit("no conformal band: the deployable protocol did not run")
    if not rec.get("cached"):
        if not _SEEN["override_weight_calls"]:
            raise SystemExit("weak_labels_from_log was never called; the "
                             "override-weight probe did not fire")
        if variant["estimator"] == "neural" and not _SEEN["nparams"]:
            raise SystemExit("no estimator fit was observed")
        if variant["estimator"] == "gbt" and not _SEEN["gbt_fits"]:
            raise SystemExit("no tree fit was observed")

    r = float(np.mean(rec["per"]["rule"]))
    m = float(np.mean(rec["per"]["m0_alone"]))
    fold = {
        "part": "W10_W11", "experiment": variant["experiment"],
        "variant": variant["vid"], "estimator": variant["estimator"],
        "override_weight": float(variant["override_weight"]),
        "confirm_weight": SHIPPED_CONFIRM_WEIGHT,
        "gbt_params": GBT_PARAMS if variant["estimator"] == "gbt" else None,
        "sklearn_version": sklearn.__version__,
        "arm": "stability",
        "cell_key": task["cell_key"], "cell_name": task["cell_name"],
        "cell_tag": W9.CELL_OF[task["cell_key"]]["tag"],
        "world": WORLD, "seed": task["seed"],
        "campus": rec["campus"], "u": rec["u"], "beta": rec["beta"],
        "rho": rec["rho"], "eps": rec["eps"],
        "family": task["family"], "channel": rec["channel"],
        "master_seed": task["master_seed"],
        "policy": rec["policy"], "split_fit": bool(rec["split_fit"]),
        "cal_frac": rec["cal_frac"], "alpha": rec["alpha"],
        "band_mode": rec["band_mode"],
        "n_train": rec["n_train"], "n_probe": rec["n_probe"],
        "n_eval": rec["n_eval"], "m0_iters": task["m0_iters"],
        "cell_sig": rec["sig"], "variant_key": variant_key(task, variant),
        "inst_ids": rec["inst_ids"], "n_wos": rec["n_wos"],
        "nparams_observed": sorted(set(_SEEN["nparams"])) or None,
        "gbt_fits": _SEEN["gbt_fits"] or None,
        "override_weight_observed": sorted(set(_SEEN["override_weight_calls"])) or None,
        "weak_label_weights_observed": sorted(_SEEN["returned_weights"]) or None,
        "per": rec["per"],
        "rule_mean": r, "m0_alone_mean": m,
        "gain_pct_vs_rule": 100.0 * (r - m) / r if r else float("nan"),
        "routing": rec["routing"], "band": rec["band"],
        "coverage": rec["coverage"], "verdict": rec["verdict"],
        "m0_final": rec["m0_final"], "per_iter": rec["per_iter"],
        "run_config": rec["run_config"],
        "config_diff_vs_incumbent": config_diff(task, variant),
        "was_cached": bool(rec.get("cached")),
        "wall_s_not_a_measurement": wall,
    }
    if variant["vid"] == INCUMBENT_VID:
        fold["repro_vs_incumbent"] = _repro_vs_incumbent(task, rec)

    path = fold_path(task, variant)
    tmp = path + ".part"
    with open(tmp, "w") as fh:
        json.dump(fold, fh, indent=1, default=str)
    os.replace(tmp, path)                   # atomic: no half-written fold
    return {k: fold[k] for k in
            ("variant", "cell_name", "seed", "rule_mean", "m0_alone_mean",
             "gain_pct_vs_rule", "routing", "m0_final", "gbt_fits",
             "override_weight_observed", "weak_label_weights_observed",
             "wall_s_not_a_measurement", "was_cached")
            } | {"repro": fold.get("repro_vs_incumbent")}


# --------------------------------------------------------------------------- #
# Pre-flight                                                                   #
# --------------------------------------------------------------------------- #
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight():
    out = {"world": WORLD, "seeds": list(SEEDS), "rho": RHO,
           "cells": [dict(c, name=W9.cell_name(c)) for c in CELLS],
           "variants": VARIANTS, "incumbent_variant": INCUMBENT_VID,
           "override_weights": list(OVERRIDE_WEIGHTS),
           "gbt_params": GBT_PARAMS,
           "versions": {"sklearn": sklearn.__version__,
                        "numpy": np.__version__, "torch": torch.__version__},
           "pools": {}, "variant_keys": {}, "config_diffs": {}}

    # -- the published hidden world is still the published hidden world ------ #
    sha = _sha256(os.path.join(_COEFF_DIR, "F-NL_seed%d.json" % WORLD))
    if sha != W9.PUBLISHED_COEFF_SHA:
        raise SystemExit("the published overlay coefficients have changed")
    with open(_P9CHECKS) as fh:
        if json.load(fh)["overlay_coeff_sha256"] != W9.PUBLISHED_COEFF_SHA:
            raise SystemExit("results/y3_p9/data_checks.json disagrees on the "
                             "published overlay digest")
    out["overlay_coeff_sha256"] = sha

    # -- instances: the same pool and the same held-out slice as the incumbent - #
    with open(_HARVEST) as fh:
        pub_ids = json.load(fh)["eval_inst_ids"]
    for cell in CELLS:
        key = "c%02d_u%d" % (cell["campus"], cell["u"])
        if key in out["pools"]:
            continue
        files = W1.locate_files(cell["campus"], "storm2", u=cell["u"])
        need = (W9.LOCKED_BASE_TASK["n_train"] + W9.LOCKED_BASE_TASK["n_probe"]
                + W9.LOCKED_BASE_TASK["n_eval"])
        if len(files) != 30 or len(files) < need:
            raise SystemExit("%s: %d instance files, expected 30" % (key, len(files)))
        tr, pr, ev = files[:16], files[16:20], files[20:30]
        if set(ev) & (set(tr) | set(pr)):
            raise SystemExit("%s: the held-out slice overlaps train or probe" % key)
        ids = [os.path.basename(p)[:-len(".json")] for p in ev]
        if cell["campus"] == 9 and cell["u"] == 100 and ids != pub_ids:
            raise SystemExit("the c9 u100 held-out slice differs from the "
                             "published set")
        out["pools"][key] = {
            "dir": os.path.relpath(os.path.dirname(files[0]), _ROOT),
            "n_files": len(files), "train_slice": "files[0:16]",
            "probe_slice": "files[16:20]", "eval_slice": "files[20:30]",
            "eval_inst_ids": ids,
            "eval_file_sha256": {os.path.basename(p): _sha256(p) for p in ev},
            "matches_published_eval_ids": True}
        print("  [pool] %s  %d files, held-out %s..%s, ids match the published "
              "set" % (key, len(files), ids[0][-4:], ids[-1][-4:]), flush=True)

    # -- the variant keys must not collide ---------------------------------- #
    keys = {}
    for cell in CELLS:
        for seed in SEEDS:
            for v in VARIANTS:
                t = make_task(cell, seed, v)
                k = variant_key(t, v)
                tag = "%s|%s|%d" % (v["vid"], t["cell_name"], seed)
                if k in keys:
                    raise SystemExit("VARIANT KEY COLLISION: %s and %s share %s"
                                     % (tag, keys[k], k))
                keys[k] = tag
                out["variant_keys"][tag] = k
                out["config_diffs"][v["vid"]] = assert_config_diff(t, v)
    print("  [keys] %d distinct variant-extended fold keys over %d variants x "
          "%d cells x %d seeds"
          % (len(keys), len(VARIANTS), len(CELLS), len(SEEDS)), flush=True)

    # -- the incumbent folds exist and are readable -------------------------- #
    out["incumbent_folds"] = {}
    for cell in CELLS:
        for seed in SEEDS:
            pub, path = w9_fold(cell["key"], seed)
            r = float(np.mean(pub["per"]["rule"]))
            m = float(np.mean(pub["per"]["m0_alone"]))
            out["incumbent_folds"]["%s|%d" % (W9.cell_name(cell), seed)] = {
                "path": os.path.relpath(path, _ROOT), "sha256": _sha256(path),
                "rule_mean": r, "m0_alone_mean": m,
                "gain_pct_vs_rule": 100.0 * (r - m) / r,
                "pearson_r": pub["m0_final"]["pearson_r"],
                "sign_acc_nonzero": pub["m0_final"]["sign_acc_nonzero"],
                "nparams": pub["nparams"]}

    out["threads"] = {v: os.environ.get(v) for v in
                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                       "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                       "VECLIB_MAXIMUM_THREADS")}
    out["torch_num_threads_parent"] = int(torch.get_num_threads())
    out["sklearn_openmp_effective_threads_parent"] = int(_openmp_effective_n_threads())
    out["locked_base_task"] = W9.LOCKED_BASE_TASK
    out["sig_keys"] = W9.LOCKED_SIG_KEYS

    os.makedirs(_OUT, exist_ok=True)
    p = os.path.join(_OUT, "preflight.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    os.replace(tmp, p)
    with open(os.path.join(_OUT, "config_diff.json"), "w") as fh:
        json.dump({"incumbent_variant": INCUMBENT_VID,
                   "allowed_diff": {k: sorted(v) for k, v in _ALLOWED_DIFF.items()},
                   "resolved_incumbent": resolved_config(
                       make_task(CELLS[0], SEEDS[0], VARIANT_OF[INCUMBENT_VID]),
                       VARIANT_OF[INCUMBENT_VID]),
                   "diffs": out["config_diffs"]}, fh, indent=1,
                  sort_keys=True, default=str)
    print("[preflight] %d instance pools checked, %d variants config-diffed "
          "against the incumbent -> %s"
          % (len(out["pools"]), len(VARIANTS), os.path.relpath(p, _ROOT)),
          flush=True)
    return out


# --------------------------------------------------------------------------- #
# Sweep                                                                        #
# --------------------------------------------------------------------------- #
def all_jobs(variants, cells, seeds):
    jobs = []
    for v in variants:
        for c in cells:
            for s in seeds:
                jobs.append({"task": make_task(c, s, v), "variant": v})
    return jobs


def summarize():
    """Per variant and cell: mean over seeds of the reduction against the rule,
    the recovery correlation, sign accuracy, and (W10) the trees fitted."""
    rows = []
    folds = sorted(glob.glob(os.path.join(_FOLDS, "*.json")))
    by = {}
    for p in folds:
        with open(p) as fh:
            d = json.load(fh)
        by.setdefault((d["variant"], d["cell_name"]), []).append(d)
    # The incumbent arm's remaining seeds come from W9, but ONLY once at least
    # one recomputed weight-5 fold has been shown bit-equal to its incumbent.
    # Without that gate this function would quietly present W9 numbers as the
    # sweep's weight-5 column on the strength of an assumption.
    gated = [d for ds in by.values() for d in ds
             if d["variant"] == INCUMBENT_VID
             and (d.get("repro_vs_incumbent") or {}).get("bit_exact")]
    if not gated:
        raise SystemExit("no recomputed weight-5 fold has passed the "
                         "bit-equality gate; refusing to reuse the W9 folds as "
                         "the weight-5 column")
    for cell in CELLS:
        for seed in SEEDS:
            pub, path = w9_fold(cell["key"], seed)
            k = (INCUMBENT_VID, W9.cell_name(cell))
            have = {d["seed"] for d in by.get(k, [])}
            if seed in have:
                continue
            r = float(np.mean(pub["per"]["rule"]))
            m = float(np.mean(pub["per"]["m0_alone"]))
            by.setdefault(k, []).append(
                {"variant": INCUMBENT_VID, "cell_name": W9.cell_name(cell),
                 "seed": seed, "rule_mean": r, "m0_alone_mean": m,
                 "gain_pct_vs_rule": 100.0 * (r - m) / r,
                 "m0_final": pub["m0_final"], "routing": pub["routing"],
                 "gbt_fits": None, "source": os.path.relpath(path, _ROOT)})
    for (vid, cell_name), ds in sorted(by.items()):
        ds = sorted(ds, key=lambda d: d["seed"])
        g = [d["gain_pct_vs_rule"] for d in ds]
        pr = [d["m0_final"]["pearson_r"] for d in ds]
        sa = [d["m0_final"]["sign_acc_nonzero"] for d in ds]
        rev = [d["routing"]["m0_sup_revfrac_mean"] for d in ds]
        trees = [t["n_trees_total"] for d in ds if d.get("gbt_fits")
                 for t in d["gbt_fits"]]
        rows.append({
            "variant": vid, "experiment": VARIANT_OF[vid]["experiment"],
            "estimator": VARIANT_OF[vid]["estimator"],
            "override_weight": VARIANT_OF[vid]["override_weight"],
            "cell": cell_name, "n_seeds": len(ds),
            "seeds": [d["seed"] for d in ds],
            "gain_pct_mean": float(np.mean(g)), "gain_pct_sd": float(np.std(g, ddof=1)) if len(g) > 1 else 0.0,
            "gain_pct_per_seed": g,
            "pearson_r_mean": float(np.mean(pr)), "pearson_r_per_seed": pr,
            "sign_acc_mean": float(np.mean(sa)), "sign_acc_per_seed": sa,
            "reviewed_fraction_mean": float(np.mean(rev)),
            "rule_mean": float(np.mean([d["rule_mean"] for d in ds])),
            "m0_mean": float(np.mean([d["m0_alone_mean"] for d in ds])),
            "trees_per_fit_min": int(min(trees)) if trees else None,
            "trees_per_fit_max": int(max(trees)) if trees else None,
            "n_tree_fits": len(trees) or None,
            "sources": [d.get("source", "this run") for d in ds],
        })
    p = os.path.join(_OUT, "summary.json")
    with open(p, "w") as fh:
        json.dump(rows, fh, indent=1, default=str)
    print("\n%-16s %-24s %8s %7s %7s %7s %7s"
          % ("variant", "cell", "gain%", "sd", "r", "signacc", "rev"))
    for r in rows:
        print("%-16s %-24s %8.3f %7.3f %7.3f %7.3f %7.3f"
              % (r["variant"], r["cell"], r["gain_pct_mean"], r["gain_pct_sd"],
                 r["pearson_r_mean"], r["sign_acc_mean"],
                 r["reviewed_fraction_mean"]))
    print("\n[summary] -> %s" % os.path.relpath(p, _ROOT), flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="run one fold per experiment in the foreground and "
                         "print their wall times")
    ap.add_argument("--variants", default=",".join(v["vid"] for v in VARIANTS))
    ap.add_argument("--cells", default=",".join(c["key"] for c in CELLS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    W9.assert_locked_config()               # the incumbent's own locked table
    _assert_threads("parent")
    print("[config] sklearn %s, one OpenMP thread asserted for parent and "
          "workers" % sklearn.__version__, flush=True)
    os.makedirs(_FOLDS, exist_ok=True)
    os.makedirs(_CACHE_ROOT, exist_ok=True)

    if args.summarize_only:
        summarize()
        return

    preflight()

    if args.pilot:
        for vid in (variant_id("W11", "neural", 1.0),
                    variant_id("W10", "gbt", SHIPPED_OVERRIDE_WEIGHT)):
            v = VARIANT_OF[vid]
            task = make_task(CELL_OF["A"], 301, v)
            print("[pilot] %s  %s seed 301 (key %s)"
                  % (vid, task["cell_name"], variant_key(task, v)), flush=True)
            t0 = time.time()
            out = worker({"task": task, "variant": v})
            wall = time.time() - t0
            print("[pilot] rule=%.1f m0=%.1f gain=%+.2f%%  r=%.3f signacc=%.3f "
                  "(one fold; NO conclusion is drawn from it)"
                  % (out["rule_mean"], out["m0_alone_mean"],
                     out["gain_pct_vs_rule"], out["m0_final"]["pearson_r"],
                     out["m0_final"]["sign_acc_nonzero"]), flush=True)
            if out["gbt_fits"]:
                print("[pilot] trees fitted per DAgger iteration: %r"
                      % [f["n_trees_total"] for f in out["gbt_fits"]], flush=True)
            else:
                print("[pilot] override weights seen by the weak-label builder: "
                      "%r; sample weights returned: %r"
                      % (out["override_weight_observed"],
                         out["weak_label_weights_observed"]), flush=True)
            print("[pilot] WALL TIME %.1f s  (single fold, one thread; a "
                  "scheduling estimate, not a measurement)" % wall, flush=True)
        return

    variants = [VARIANT_OF[k] for k in args.variants.split(",") if k]
    cells = [CELL_OF[k] for k in args.cells.split(",") if k]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    jobs = all_jobs(variants, cells, seeds)
    todo = [j for j in jobs if existing_fold(j["task"], j["variant"]) is None]
    print("[sweep] %d variants x %d cells x %d seeds = %d folds; %d already on "
          "disk, %d to run, %d workers"
          % (len(variants), len(cells), len(seeds), len(jobs),
             len(jobs) - len(todo), len(todo), args.workers), flush=True)
    if args.dry_run:
        for j in todo:
            print("   would run", j["variant"]["vid"], j["task"]["cell_name"],
                  j["task"]["seed"], variant_key(j["task"], j["variant"]))
        return

    t0 = time.time()
    done = 0
    gates = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut = {ex.submit(worker, j): j for j in todo}
        for f in as_completed(fut):
            out = f.result()
            done += 1
            if out.get("repro") is not None:
                gates += 1
                print("  [gate] %s %s seed %d reproduces the committed W9 "
                      "incumbent fold bit for bit"
                      % (out["variant"], out["cell_name"], out["seed"]), flush=True)
            print("  [%3d/%3d] %-16s %-24s seed %d | rule=%8.1f m0=%8.1f "
                  "gain=%+6.2f%% | r=%.3f | %.0fs (elapsed %.0fs)"
                  % (done, len(todo), out["variant"], out["cell_name"],
                     out["seed"], out["rule_mean"], out["m0_alone_mean"],
                     out["gain_pct_vs_rule"], out["m0_final"]["pearson_r"],
                     out["wall_s_not_a_measurement"], time.time() - t0), flush=True)
    n_on_disk = len(glob.glob(os.path.join(_FOLDS, "*.json")))
    print("[done] %d folds run, %d fold files on disk, %d bit-equality gates "
          "passed" % (done, n_on_disk, gates), flush=True)
    summarize()


if __name__ == "__main__":
    main()
