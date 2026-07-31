#!/usr/bin/env python
"""W3 R6b: where the beta = 0 reduction actually comes from.

At beta = 0 nothing per-order is recoverable, yet the fitted layer reduces true
weighted tardiness by 14.7% in the extreme-overload column. The manuscript
attributes this to the estimator absorbing the clip's class-level bias as a
constant offset. That attribution has three separable ingredients, and this
script measures each by taking the FITTED hat_s map and degrading it in one way
at a time, without refitting anything:

  fitted        the estimator's map, unchanged (the control; must reproduce the
                published cell exactly).
  class_mean    every order replaced by the mean fitted hat_s of its RECORDED
                class. Keeps the class-level offsets, destroys all within-class
                variation. If the reduction is the class-level offset, this
                keeps it.
  centred       the global mean removed, all variation kept. If the reduction is
                the constant offset, this destroys it.
  level_only    every order replaced by the single global mean. The offset with
                nothing else.
  shuffled      the same multiset of fitted values, permuted across orders
                within the instance. Identical mean, identical spread, zero
                association between an order and its correction. If the
                reduction survives this, it is not carried by WHICH order gets
                WHICH correction.
  gaussian      a fresh N(mean, sd) draw per order, matching only the first two
                moments of the fitted map.
  zero          hat_s == 0, i.e. the tuned rule (the reference).

Nothing here is a deliverable and no number from it is a headline; it decides
which sentence the manuscript is allowed to write about the beta = 0 column.

Run:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-23 \
    python scripts/y3_w3_mechanism.py --cell c9_u130_b0 --workers 3
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse                                                  # noqa: E402
import sys                                                       # noqa: E402
import time                                                      # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np                                               # noqa: E402

from fmwos.env import DispatchEnv                                 # noqa: E402
from fmwos.hitl import censored as CN                             # noqa: E402
from fmwos.hitl import deciders as dec                            # noqa: E402
from fmwos.hitl import true_objective as TO                       # noqa: E402

import y3_w3_lib as L                                             # noqa: E402
from y3_w3_run import CELLS                                       # noqa: E402

ARMS = ("fitted", "class_mean", "centred", "level_only", "shuffled",
        "gaussian", "zero")


def degrade(hs, instance, arm, rng):
    """One degradation of a fitted hat_s map. Returns a new wo_id -> float map."""
    ids = [w["id"] for w in instance["work_orders"]]
    cls = np.asarray([int(w["priority"]) for w in instance["work_orders"]])
    v = np.asarray([hs[i] for i in ids], float)
    if arm == "fitted":
        out = v
    elif arm == "zero":
        out = np.zeros_like(v)
    elif arm == "level_only":
        out = np.full_like(v, v.mean())
    elif arm == "centred":
        out = v - v.mean()
    elif arm == "class_mean":
        out = np.empty_like(v)
        for k in (1, 2, 3, 4):
            m = cls == k
            if m.any():
                out[m] = v[m].mean()
    elif arm == "shuffled":
        out = v[rng.permutation(v.size)]
    elif arm == "gaussian":
        out = rng.normal(v.mean(), v.std(), size=v.size)
    else:
        raise ValueError(arm)
    return {i: float(x) for i, x in zip(ids, out)}


def job(args):
    tag, seed, perm_seed = args
    L.set_threads(1)
    spec = dict(CELLS[tag])
    spec.pop("published_csv"); spec.pop("protocol")
    cell = L.build_cell(seed=seed, **spec)
    tr = L.train_variant(cell, "mse_published", verbose=False)
    model = tr["model"]
    overlay, channel = cell["overlay"], cell["channel"]

    out = {"tag": tag, "seed": seed, "perm_seed": perm_seed, "arms": {}}
    stats = {}
    for arm in ARMS:
        rng = np.random.default_rng(perm_seed)
        rule, aug = [], []
        sd_by_class, mean_by_class = {}, {}
        for inst in cell["eval"]:
            applied = overlay.apply(inst)

            def sc(sched):
                return TO.score_true(inst, sched, overlay, applied)["TWT_true"]

            hs = degrade(model.hat_s_map(inst), inst, arm, rng)
            _ids, ap = CN.applied_shift(hs, inst)
            cls = np.asarray([int(w["priority"]) for w in inst["work_orders"]])
            for k in (1, 2, 3, 4):
                m = cls == k
                if m.any():
                    sd_by_class.setdefault(k, []).append(float(ap[m].std()))
                    mean_by_class.setdefault(k, []).append(float(ap[m].mean()))
            rule.append(sc(dec.run_rule(DispatchEnv(inst), "atc", seed=seed)))
            d = CN.hat_s_map_atc_decider(hs, inst, channel=channel)
            sched, _ = DispatchEnv(inst).run_supervised(d, supervisor=None,
                                                        method="m0", seed=seed)
            aug.append(sc(sched))
        r, a = np.asarray(rule), np.asarray(aug)
        out["arms"][arm] = {
            "rule": r.tolist(), "aug": a.tolist(),
            "pct_below_rule": 100.0 * (r.mean() - a.mean()) / r.mean(),
            "applied_mean_by_class": {str(k): float(np.mean(mean_by_class[k]))
                                      for k in sorted(mean_by_class)},
            "applied_sd_by_class": {str(k): float(np.mean(sd_by_class[k]))
                                    for k in sorted(sd_by_class)}}
        stats[arm] = out["arms"][arm]["pct_below_rule"]
    out["summary"] = stats
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="c9_u130_b0", choices=sorted(CELLS))
    ap.add_argument("--seeds", default="301,302,303")
    ap.add_argument("--perm-seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs = [(a.cell, s, a.perm_seed) for s in seeds]
    print("[mechanism] cell=%s seeds=%s" % (a.cell, seeds), flush=True)
    recs, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fut = [ex.submit(job, j) for j in jobs]
        for f in as_completed(fut):
            r = f.result(); recs.append(r)
            print("  seed %d done (wall %.0fs): %s"
                  % (r["seed"], time.time() - t0,
                     {k: round(v, 2) for k, v in r["summary"].items()}), flush=True)

    pooled = {}
    for arm in ARMS:
        rule = np.asarray([r["arms"][arm]["rule"] for r in recs], float)
        aug = np.asarray([r["arms"][arm]["aug"] for r in recs], float)
        con = L.contrast(aug.mean(axis=0), rule.mean(axis=0))
        pooled[arm] = {
            "pooled_pct_below_rule": 100.0 * (rule.sum() - aug.sum()) / rule.sum(),
            "wtl_vs_rule": con["wtl"], "wilcoxon_p_vs_rule": con["wilcoxon_p"],
            "twt_aug_seedavg_per_instance": aug.mean(axis=0).tolist(),
            "applied_mean_by_class": recs[0]["arms"][arm]["applied_mean_by_class"],
            "applied_sd_by_class": recs[0]["arms"][arm]["applied_sd_by_class"]}
    base = np.asarray(pooled["fitted"]["twt_aug_seedavg_per_instance"], float)
    for arm in ARMS:
        v = np.asarray(pooled[arm]["twt_aug_seedavg_per_instance"], float)
        c = L.contrast(v, base)
        pooled[arm]["vs_fitted"] = {"dTWT_pct": c["pct_vs_comparator"],
                                    "wtl": c["wtl"], "wilcoxon_p": c["wilcoxon_p"]}
    out = {"cell": a.cell, "seeds": seeds, "perm_seed": a.perm_seed,
           "arms": pooled, "per_seed": recs,
           "note": ("Degradations of the FITTED incumbent hat_s map at beta = 0, "
                    "no refitting between arms. 'shuffled' keeps the multiset of "
                    "fitted values and destroys the order-to-value association; "
                    "'class_mean' keeps the class-level offsets and destroys the "
                    "within-class variation.")}
    L.write_json(os.path.join(L.OUT, "mechanism_%s.json" % a.cell), out)

    print("\nWHERE THE beta = 0 REDUCTION COMES FROM (cell %s, seeds %s)"
          % (a.cell, seeds))
    print("%-12s %10s %9s %10s   %s" % ("arm", "%<RULE", "W/T/L", "p vs RULE",
                                        "mean/sd of the applied shift by class"))
    for arm in ARMS:
        p = pooled[arm]
        mb = p["applied_mean_by_class"]; sb = p["applied_sd_by_class"]
        cols = "  ".join("c%s %+.3f+-%.3f" % (k, mb[k], sb[k]) for k in sorted(mb))
        print("%-12s %10.3f %9s %10.4f   %s"
              % (arm, p["pooled_pct_below_rule"],
                 "%d/%d/%d" % (p["wtl_vs_rule"]["W"], p["wtl_vs_rule"]["T"],
                               p["wtl_vs_rule"]["L"]),
                 p["wilcoxon_p_vs_rule"], cols))
    print("\nwrote", os.path.join(L.OUT, "mechanism_%s.json" % a.cell))


if __name__ == "__main__":
    main()
