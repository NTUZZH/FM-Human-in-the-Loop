#!/usr/bin/env python
"""Paper Y3, W9 -- the key cells re-measured across INDEPENDENT SUPERVISOR WORLDS.

Every published cell draws the hidden supervisor function ``f`` and the per-order
latent noise ``z`` from one overlay master seed (12345). The ten seeds the
manuscript reports are training and evaluation randomness INSIDE that one
simulated world, so nothing published so far separates "the correction layer
works" from "the correction layer works against this particular draw of the
hidden function". W9 repeats the key cells over ten independent overlay worlds
(master seeds 12345 and 20001..20009), three training seeds each.

WHAT AN INDEPENDENT WORLD IS (verified, not assumed; see RUN_PLAN.md). The
master seed feeds THREE independent draws in src/fmwos/hitl/overlay.py:

  * ``stable_seed("lin", master_seed)``  -> the linear coefficients ``a`` of f;
  * ``stable_seed("nl",  master_seed)``  -> the four sparse interactions of F-NL;
  * ``stable_seed("z", master_seed, instance_id)`` -> the per-order noise z.

So changing the master seed changes BOTH the hidden function and the latent
noise. What it does NOT change is the feature basis and its standardization
(feat_mean / feat_std are computed over the fixed training-campus order
population), so two worlds' hidden functions are two random coefficient vectors
in the SAME 20-dimensional basis and are uncorrelated only in expectation, not
by construction. Measured on one c9 instance: at beta=0 the two worlds' latent
xi correlate at 0.0001 and 74% of class shifts differ; at beta=1 they correlate
at -0.26 and 78% of class shifts differ.

PROTOCOL. Review placement is the SHIPPED deployable routing: policy="stability"
with split_fit=True, the decision-stability test under a split-conformal band
calibrated only on override-derived weak labels. The evaluation is
``scripts/y3_w1_sweep.evaluate_cell`` called verbatim, with its module-level
cache redirected into results/y3_w9/cache so no published cache is read or
written. Nothing is re-derived here.

CONFIG DISCIPLINE. Every locked constant is imported from y3_w1_sweep and
asserted against an explicit table, including the full resolved default task and
the cache-signature key list (so the master seed cannot silently drop out of the
cache key). The ONLY inputs that vary are the overlay master seed and the cell
parameters (campus, utilisation, beta, eps). Every fit asserts the estimator's
1761 parameters.

THREADS. Hard-set to one before numpy/torch import, not setdefault: y3_w1_sweep
setdefaults them to 4 and the pipeline reproduces bit-exactly at one numeric
thread only. Every worker re-asserts the caps and torch.get_num_threads() == 1.
Parallelism comes from separate processes. No wall-clock number produced by this
run is reported as a measurement of anything.

Run:
    PYTHONPATH=src taskset -c 0-11 python scripts/y3_w9_overlay_worlds.py --pilot
    PYTHONPATH=src taskset -c 0-11 python scripts/y3_w9_overlay_worlds.py \\
        --workers 10
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

import torch                                                       # noqa: E402

from fmwos.hitl import overlay as ov                               # noqa: E402
from fmwos.hitl import routing as R                                # noqa: E402

import y3_w1_sweep as W1                                           # noqa: E402

_OUT = os.path.join(_ROOT, "results", "y3_w9")
_FOLDS = os.path.join(_OUT, "folds")
_CACHE = os.path.join(_OUT, "cache")
_W1CACHE = os.path.join(_ROOT, "results", "y3_w1", "cache")
_COEFF_DIR = os.path.join(_ROOT, "results", "y3_p1", "overlay_coeffs")
_P9CHECKS = os.path.join(_ROOT, "results", "y3_p9", "data_checks.json")
_HARVEST = os.path.join(_ROOT, "results", "y3_p5", "harvest",
                        "primary_multiseed_summary.json")

# --------------------------------------------------------------------------- #
# The varied inputs: overlay worlds, training seeds, cells                     #
# --------------------------------------------------------------------------- #
PUBLISHED_WORLD = 12345
WORLDS = (12345, 20001, 20002, 20003, 20004, 20005,
          20006, 20007, 20008, 20009)
SEEDS = (301, 302, 303)
RHO = 0.25

CELLS = [
    dict(key="A", campus=9,  u=100, beta=1.00, eps=0.00,
         tag="headline ceiling"),
    dict(key="B", campus=9,  u=100, beta=0.20, eps=0.00,
         tag="corpus-anchored share"),
    dict(key="C", campus=9,  u=100, beta=0.50, eps=0.00,
         tag="mid share"),
    dict(key="D", campus=9,  u=100, beta=0.00, eps=0.00,
         tag="inert boundary, in band"),
    dict(key="E", campus=9,  u=130, beta=0.00, eps=0.00,
         tag="inert boundary, overload"),
    dict(key="F", campus=10, u=100, beta=0.00, eps=0.00,
         tag="anomaly campus, in band"),
    dict(key="G", campus=10, u=130, beta=0.00, eps=0.00,
         tag="anomaly campus, overload"),
    dict(key="H", campus=9,  u=100, beta=0.20, eps=0.25,
         tag="noisy supervisor at the anchored share"),
]
CELL_OF = {c["key"]: c for c in CELLS}


def cell_name(cell):
    return "%s_c%d_u%d_b%.2f_eps%.2f" % (cell["key"], cell["campus"], cell["u"],
                                         cell["beta"], cell["eps"])


# --------------------------------------------------------------------------- #
# Locked configuration, asserted against the machinery this runner calls       #
# --------------------------------------------------------------------------- #
# The resolved default task of scripts/y3_w1_sweep._base_task(). Asserted as a
# whole dict: a new field, a dropped field or a changed default all abort here.
LOCKED_BASE_TASK = {
    "regime": "storm2", "u": None, "size": None, "eps": 0.0, "theta": 1.0,
    "channel": "full_class_shift", "family": "F-NL", "master_seed": 12345,
    "n_train": 16, "n_probe": 4, "n_eval": 10, "m0_iters": 8,
    "policy": "stability", "split_fit": True, "cal_frac": 0.3,
    "alpha": 0.1, "band_mode": "global",
}

# The cache-signature keys. Asserted so the overlay master seed can never drop
# out of the cache key: if it did, ten worlds would silently share one result.
LOCKED_SIG_KEYS = ["campus", "regime", "u", "size", "beta", "rho", "eps",
                   "theta", "channel", "family", "master_seed", "seed",
                   "n_train", "n_probe", "n_eval", "m0_iters", "policy",
                   "split_fit", "cal_frac", "alpha", "band_mode"]

LOCKED_MODULE = {
    "FAMILY": "F-NL", "MASTER_SEED": 12345, "EPS": 0.0, "THETA": 1.0,
    "CHANNEL": "full_class_shift",
    "DECIDERS": ["rule", "m0_alone", "oracle", "rule_sup", "m0_sup"],
}

DECIDERS = list(LOCKED_MODULE["DECIDERS"])

# The shipped correction layer: ShiftEstimator(lat_dim=20, hidden=32).
NPARAMS_EXPECTED = 1761

# The published overlay coefficient digest, recorded by results/y3_p9.
PUBLISHED_COEFF_SHA = "21b3ef67c107d2c5fbbbe6a1ee28354851c6928cec64aa66749cf2f1c1b9413b"


def _assert_threads(where):
    if torch.get_num_threads() != 1:
        raise SystemExit("%s: torch threads = %d, expected 1"
                         % (where, torch.get_num_threads()))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(v) != "1":
            raise SystemExit("%s: %s = %r, expected '1'"
                             % (where, v, os.environ.get(v)))


def assert_locked_config():
    """Abort unless the machinery this runner calls is the machinery it expects."""
    bad = []
    for k, v in LOCKED_MODULE.items():
        got = getattr(W1, k)
        same = (list(got) == list(v)) if isinstance(v, list) else (got == v)
        if not same:
            bad.append("y3_w1_sweep.%s = %r, W9 expects %r" % (k, got, v))
    base = W1._base_task()
    if base != LOCKED_BASE_TASK:
        for k in sorted(set(base) | set(LOCKED_BASE_TASK)):
            a, b = base.get(k, "<absent>"), LOCKED_BASE_TASK.get(k, "<absent>")
            if a != b:
                bad.append("_base_task()[%r] = %r, W9 expects %r" % (k, a, b))
    if list(W1._SIG_KEYS) != LOCKED_SIG_KEYS:
        bad.append("_SIG_KEYS = %r, W9 expects %r" % (list(W1._SIG_KEYS),
                                                      LOCKED_SIG_KEYS))
    if "master_seed" not in W1._SIG_KEYS:
        bad.append("the cache signature does not include master_seed; ten "
                   "overlay worlds would collide on one cache entry")
    if ov.SIGMA_S != 1.0 or ov.DEFAULT_CHANNEL != "full_class_shift":
        bad.append("overlay constants drifted (sigma_s=%r, default channel=%r)"
                   % (ov.SIGMA_S, ov.DEFAULT_CHANNEL))
    if bad:
        raise SystemExit("CONFIG DRIFT against scripts/y3_w1_sweep.py:\n  "
                         + "\n  ".join(bad))
    _assert_threads("parent")
    print("[config] locked cell matches y3_w1_sweep; deployable stability "
          "routing; 1 numeric thread asserted", flush=True)


# --------------------------------------------------------------------------- #
# Tasks                                                                        #
# --------------------------------------------------------------------------- #
def make_task(cell, world, seed):
    t = W1._base_task(campus=cell["campus"], regime="storm2", u=cell["u"],
                      beta=cell["beta"], rho=RHO, eps=cell["eps"], seed=seed,
                      master_seed=world)
    # Everything except the varied inputs must still be the locked default.
    varied = {"campus", "u", "beta", "rho", "eps", "seed", "master_seed"}
    for k, v in LOCKED_BASE_TASK.items():
        if k not in varied and t[k] != v:
            raise SystemExit("task field %r drifted to %r (expected %r)"
                             % (k, t[k], v))
    assert t["policy"] == "stability" and t["split_fit"] is True, \
        "review placement is not the deployable stability routing"
    t["arm"] = "stability"
    t["part"] = "W9"
    t["cell_key"] = cell["key"]
    t["cell_name"] = cell_name(cell)
    t["world"] = world
    return t


def fold_path(task):
    return os.path.join(_FOLDS, "%s__world%d__seed%d__%s.json"
                        % (task["cell_name"], task["world"], task["seed"],
                           W1._cell_sig(task)))


def existing_fold(task):
    """The completed fold for this (cell, world, seed), or None.

    A file for the same (cell, world, seed) under a DIFFERENT cache key is a
    configuration change since it was written, and aborts rather than being
    silently mixed with fresh results."""
    stem = "%s__world%d__seed%d__" % (task["cell_name"], task["world"],
                                      task["seed"])
    hits = sorted(glob.glob(os.path.join(_FOLDS, stem + "*.json")))
    want = fold_path(task)
    stale = [h for h in hits if h != want]
    if stale:
        raise SystemExit(
            "CACHE-KEY DRIFT: %s exists under a different configuration "
            "signature (%s). Delete it or explain the change; it must not be "
            "mixed with the current sweep." % (stem, [os.path.basename(s) for s in stale]))
    return want if os.path.exists(want) else None


# --------------------------------------------------------------------------- #
# Worker                                                                       #
# --------------------------------------------------------------------------- #
_NPARAM_SEEN = []


def _install_param_assertion():
    """Wrap routing.run_m0_routed in a pass-through that asserts the fitted
    estimator's parameter count. Calls the original and returns its result, so
    no numeric behaviour changes."""
    orig = getattr(R, "_w9_orig_run_m0_routed", None) or R.run_m0_routed
    R._w9_orig_run_m0_routed = orig

    def checked(*a, **kw):
        res = orig(*a, **kw)
        n = sum(p.numel() for p in res["estimator"].parameters())
        if n != NPARAMS_EXPECTED:
            raise SystemExit("PARAM DRIFT: the fitted shift estimator has %d "
                             "parameters, the shipped layer has %d"
                             % (n, NPARAMS_EXPECTED))
        _NPARAM_SEEN.append(n)
        return res

    R.run_m0_routed = checked


def _install_policy_assertion():
    """Prove which review-policy object was constructed, in training and in
    evaluation. Both call sites go through the module attribute."""
    orig = getattr(R, "_w9_orig_make_supervisor", None) or R.make_supervisor
    R._w9_orig_make_supervisor = orig

    def checked(policy, overlay, instance, rho, **kw):
        sup = orig(policy, overlay, instance, rho, **kw)
        if policy != "stability":
            raise SystemExit("a non-deployable review policy %r was requested"
                             % (policy,))
        if not isinstance(sup, R.StabilityRoutingSupervisor):
            raise SystemExit("policy='stability' built a %s"
                             % type(sup).__name__)
        return sup

    R.make_supervisor = checked


def _repro_check(task, rec):
    """For the published world only: compare this fold against the committed
    results/y3_w1/cache record with the same cache key, when one exists.

    Cell A at master seed 12345 IS the manuscript's deployable headline cell, so
    its cache key coincides with a committed W1 record. Recomputing it here and
    demanding bit-equality proves the W9 code path is the published one."""
    p = os.path.join(_W1CACHE, "%s.json" % rec["sig"])
    if task["world"] != PUBLISHED_WORLD or not os.path.exists(p):
        return None
    with open(p) as fh:
        pub = json.load(fh)
    if pub["inst_ids"] != rec["inst_ids"]:
        raise SystemExit("held-out ids differ from the committed W1 record at %s"
                         % os.path.basename(p))
    out = {"committed_record": os.path.relpath(p, _ROOT), "max_abs_diff": {}}
    worst = 0.0
    for k in DECIDERS:
        a, b = rec["per"][k], pub["per"][k]
        if len(a) != len(b):
            raise SystemExit("decider %s length differs from the committed record" % k)
        d = max(abs(x - y) for x, y in zip(a, b))
        out["max_abs_diff"][k] = d
        worst = max(worst, d)
    out["bit_exact"] = (worst == 0.0)
    if not out["bit_exact"]:
        raise SystemExit("REPRODUCTION GATE FAILED: %s differs from the "
                         "committed W1 record by up to %r" % (task["cell_name"], worst))
    return out


def worker(task):
    torch.set_num_threads(1)
    _assert_threads("worker")
    os.makedirs(_CACHE, exist_ok=True)
    os.makedirs(_FOLDS, exist_ok=True)
    W1._CACHE = _CACHE                      # never read or write a published cache
    _NPARAM_SEEN.clear()
    _install_param_assertion()
    _install_policy_assertion()

    t0 = time.perf_counter()
    rec = W1.evaluate_cell(task)
    wall = time.perf_counter() - t0

    if rec["policy"] != "stability" or bool(rec["split_fit"]) is not True:
        raise SystemExit("the record was not produced by the deployable policy")
    if rec["band"] is None:
        raise SystemExit("no conformal band: the deployable protocol did not run")
    if not rec.get("cached") and not _NPARAM_SEEN:
        raise SystemExit("no estimator fit was observed; the assertion did not fire")

    fold = {
        "part": "W9", "arm": "stability",
        "cell_key": task["cell_key"], "cell_name": task["cell_name"],
        "cell_tag": CELL_OF[task["cell_key"]]["tag"],
        "world": task["world"], "seed": task["seed"],
        "campus": rec["campus"], "u": rec["u"], "beta": rec["beta"],
        "rho": rec["rho"], "eps": rec["eps"],
        "family": task["family"], "channel": rec["channel"],
        "master_seed": task["master_seed"],
        "policy": rec["policy"], "split_fit": bool(rec["split_fit"]),
        "cal_frac": rec["cal_frac"], "alpha": rec["alpha"],
        "band_mode": rec["band_mode"],
        "n_train": rec["n_train"], "n_probe": rec["n_probe"],
        "n_eval": rec["n_eval"], "m0_iters": task["m0_iters"],
        "sig": rec["sig"], "inst_ids": rec["inst_ids"], "n_wos": rec["n_wos"],
        "nparams": NPARAMS_EXPECTED,
        "nparams_observed": sorted(set(_NPARAM_SEEN)) or None,
        "per": rec["per"],
        "routing": rec["routing"], "band": rec["band"],
        "coverage": rec["coverage"], "verdict": rec["verdict"],
        "m0_final": rec["m0_final"], "per_iter": rec["per_iter"],
        "run_config": rec["run_config"],
        "was_cached": bool(rec.get("cached")),
        "wall_s_not_a_measurement": wall,
    }
    fold["repro_vs_y3_w1"] = _repro_check(task, rec)

    path = fold_path(task)
    tmp = path + ".part"
    with open(tmp, "w") as fh:
        json.dump(fold, fh, indent=1, default=str)
    os.replace(tmp, path)                   # atomic: no half-written fold
    return {k: fold[k] for k in
            ("cell_name", "world", "seed", "per", "routing", "sig",
             "wall_s_not_a_measurement", "was_cached", "repro_vs_y3_w1")}


# --------------------------------------------------------------------------- #
# Pre-flight: overlay worlds, instance pools, data accuracy                    #
# --------------------------------------------------------------------------- #
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(worlds, cells):
    """Build every overlay world serially in the parent, and check the data.

    Building the coefficient files here rather than in the workers removes a
    write race on results/y3_p1/overlay_coeffs/ (get_coeffs writes without an
    atomic replace) and makes the worlds an input recorded before the sweep,
    not a side effect of it."""
    out = {"worlds": {}, "pools": {}, "published_world": PUBLISHED_WORLD}

    if _sha256(os.path.join(_COEFF_DIR, "F-NL_seed12345.json")) != PUBLISHED_COEFF_SHA:
        raise SystemExit("the published overlay coefficients have changed; the "
                         "published world is not the published world")
    with open(_P9CHECKS) as fh:
        if json.load(fh)["overlay_coeff_sha256"] != PUBLISHED_COEFF_SHA:
            raise SystemExit("results/y3_p9/data_checks.json disagrees on the "
                             "published overlay digest")

    for w in worlds:
        c = ov.get_coeffs("F-NL", w)          # builds and records if absent
        p = os.path.join(_COEFF_DIR, "F-NL_seed%d.json" % w)
        rebuilt = ov.build_coeffs("F-NL", w)
        if [round(x, 12) for x in rebuilt["a"]] != [round(x, 12) for x in c["a"]]:
            raise SystemExit("overlay world %d does not rebuild to its record" % w)
        out["worlds"][w] = {
            "coeff_file": os.path.relpath(p, _ROOT), "sha256": _sha256(p),
            "a_l2": float(np.linalg.norm(c["a"])),
            "n_interactions": len(c["interactions"]),
            "population_n_orders": c["population"]["n_orders"],
        }
    a0 = np.asarray(ov.get_coeffs("F-NL", PUBLISHED_WORLD)["a"])
    for w in worlds:
        if w == PUBLISHED_WORLD:
            continue
        a = np.asarray(ov.get_coeffs("F-NL", w)["a"])
        if np.array_equal(a, a0):
            raise SystemExit("world %d drew the published hidden function" % w)
        out["worlds"][w]["cos_with_published_f_coeffs"] = float(
            a @ a0 / (np.linalg.norm(a) * np.linalg.norm(a0)))
    shas = {v["sha256"] for v in out["worlds"].values()}
    if len(shas) != len(worlds):
        raise SystemExit("two overlay worlds produced the same coefficient file")

    with open(_HARVEST) as fh:
        pub_ids = json.load(fh)["eval_inst_ids"]
    for cell in cells:
        key = "c%02d_u%d" % (cell["campus"], cell["u"])
        if key in out["pools"]:
            continue
        files = W1.locate_files(cell["campus"], "storm2", u=cell["u"])
        need = LOCKED_BASE_TASK["n_train"] + LOCKED_BASE_TASK["n_probe"] \
            + LOCKED_BASE_TASK["n_eval"]
        if len(files) != 30 or len(files) < need:
            raise SystemExit("%s: %d instance files, expected 30" % (key, len(files)))
        tr = files[:16]
        pr = files[16:20]
        ev = files[20:30]
        if set(ev) & (set(tr) | set(pr)):
            raise SystemExit("%s: the held-out slice overlaps train or probe" % key)
        ids = [os.path.basename(p)[:-len(".json")] for p in ev]
        if cell["campus"] == 9 and cell["u"] == 100 and ids != pub_ids:
            raise SystemExit("the c9 u100 held-out slice differs from the "
                             "published set")
        with open(ev[0]) as fh:
            n_wos = len(json.load(fh)["work_orders"])
        out["pools"][key] = {
            "dir": os.path.relpath(os.path.dirname(files[0]), _ROOT),
            "n_files": len(files),
            "train_slice": "files[0:16]", "probe_slice": "files[16:20]",
            "eval_slice": "files[20:30]",
            "eval_inst_ids": ids,
            "eval_file_sha256": {os.path.basename(p): _sha256(p) for p in ev},
            "matches_published_eval_ids": bool(cell["campus"] == 9 and cell["u"] == 100),
            "n_work_orders_first_eval_instance": n_wos,
        }
        print("  [pool] %s  %d files, held-out %s..%s, %d orders/instance"
              % (key, len(files), ids[0][-4:], ids[-1][-4:], n_wos), flush=True)

    out["threads"] = {v: os.environ.get(v) for v in
                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                       "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                       "VECLIB_MAXIMUM_THREADS")}
    out["torch_num_threads_parent"] = int(torch.get_num_threads())
    out["locked_base_task"] = LOCKED_BASE_TASK
    out["sig_keys"] = LOCKED_SIG_KEYS
    out["nparams_expected"] = NPARAMS_EXPECTED
    out["cells"] = [dict(c, name=cell_name(c), rho=RHO) for c in cells]
    out["worlds_order"] = list(worlds)
    out["seeds"] = list(SEEDS)
    os.makedirs(_OUT, exist_ok=True)
    p = os.path.join(_OUT, "preflight.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    os.replace(tmp, p)
    print("[preflight] %d overlay worlds built and recorded, %d instance pools "
          "checked -> %s" % (len(worlds), len(out["pools"]),
                             os.path.relpath(p, _ROOT)), flush=True)
    return out


# --------------------------------------------------------------------------- #
# Sweep                                                                        #
# --------------------------------------------------------------------------- #
def order_tasks(cells, worlds, seeds):
    """Cheap campus first, and the published world's headline cell first of all
    so the reproduction gate against results/y3_w1/cache fires early."""
    tasks = [make_task(c, w, s) for c in cells for w in worlds for s in seeds]
    gate = [t for t in tasks if t["cell_key"] == "A" and t["world"] == PUBLISHED_WORLD]
    rest = sorted([t for t in tasks if t not in gate],
                  key=lambda t: (t["campus"], t["cell_key"], t["world"], t["seed"]))
    return gate + rest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="run exactly one fold (cell A, world 20001, seed 301) "
                         "in the foreground and print its wall time")
    ap.add_argument("--cells", default=",".join(c["key"] for c in CELLS))
    ap.add_argument("--worlds", default=",".join(str(w) for w in WORLDS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    assert_locked_config()
    os.makedirs(_FOLDS, exist_ok=True)
    os.makedirs(_CACHE, exist_ok=True)

    if args.pilot:
        cell, world, seed = CELL_OF["A"], 20001, 301
        preflight([world], [cell])
        task = make_task(cell, world, seed)
        print("[pilot] one fold: %s world %d seed %d (sig %s)"
              % (task["cell_name"], world, seed, W1._cell_sig(task)), flush=True)
        t0 = time.time()
        out = worker(task)
        wall = time.time() - t0
        r = np.mean(out["per"]["rule"])
        m = np.mean(out["per"]["m0_alone"])
        print("[pilot] rule=%.1f m0_alone=%.1f oracle=%.1f (one fold; NO "
              "conclusion is drawn from it)"
              % (r, m, np.mean(out["per"]["oracle"])), flush=True)
        print("[pilot] reviewed fraction %.3f, undetermined %.3f"
              % (out["routing"]["m0_sup_revfrac_mean"],
                 out["routing"]["m0_sup_undetermined"]), flush=True)
        print("[pilot] WALL TIME %.1f s  (single fold, one thread; a scheduling "
              "estimate, not a measurement)" % wall, flush=True)
        return

    cells = [CELL_OF[k] for k in args.cells.split(",") if k]
    worlds = [int(w) for w in args.worlds.split(",") if w]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    preflight(worlds, cells)
    tasks = order_tasks(cells, worlds, seeds)
    todo = [t for t in tasks if existing_fold(t) is None]
    print("[sweep] %d cells x %d worlds x %d seeds = %d folds; %d already on "
          "disk, %d to run, %d workers"
          % (len(cells), len(worlds), len(seeds), len(tasks),
             len(tasks) - len(todo), len(todo), args.workers), flush=True)
    if args.dry_run:
        for t in todo[:5]:
            print("   would run", t["cell_name"], t["world"], t["seed"],
                  W1._cell_sig(t))
        return

    t0 = time.time()
    done = 0
    gates = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut = {ex.submit(worker, t): t for t in todo}
        for f in as_completed(fut):
            t = fut[f]
            out = f.result()
            done += 1
            if out["repro_vs_y3_w1"] is not None:
                gates += 1
                print("  [gate] %s world %d seed %d reproduces the committed "
                      "results/y3_w1 record bit for bit"
                      % (out["cell_name"], out["world"], out["seed"]), flush=True)
            r = float(np.mean(out["per"]["rule"]))
            m = float(np.mean(out["per"]["m0_alone"]))
            print("  [%3d/%3d] %-24s world %5d seed %d | rule=%8.1f m0=%8.1f "
                  "gain=%+6.2f%% | rev=%.3f und=%.3f | %.0fs (elapsed %.0fs)"
                  % (done, len(todo), out["cell_name"], out["world"], out["seed"],
                     r, m, 100.0 * (r - m) / r if r else float("nan"),
                     out["routing"]["m0_sup_revfrac_mean"],
                     out["routing"]["m0_sup_undetermined"],
                     out["wall_s_not_a_measurement"], time.time() - t0), flush=True)
    n_on_disk = len(glob.glob(os.path.join(_FOLDS, "*.json")))
    print("[done] %d folds run, %d fold files on disk, %d reproduction gates "
          "passed; estimator parameters asserted at %d on every fit"
          % (done, n_on_disk, gates, NPARAMS_EXPECTED), flush=True)


if __name__ == "__main__":
    main()
