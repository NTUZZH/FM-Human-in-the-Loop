#!/usr/bin/env python
"""Paper Y3, W7 -- LEAVE-ONE-CAMPUS-OUT transfer of the correction layer (M0).

Generalises scripts/y3_p6_transfer.py from "two campuses held out by design"
to "every campus held out in turn", so the deliverable claim (one fitted layer
ships to a site it never saw) rests on six folds rather than two.

For each held-out campus H:
  * TRANSFER arm  -- fit the shift estimator on the other five campuses, apply
    it zero-shot to H.
  * NATIVE arm    -- fit the estimator on H's own training slice.
  * Both are scored on the SAME held-out evaluation slice of H, against the
    tuned rule (RULE) and the myopic full-information reference (ORACLE), on
    TWT*(w*,d*).
Retention = transfer gain over RULE divided by native gain over RULE.

CONFIG DISCIPLINE. Every locked constant is imported from y3_p6_transfer and
asserted, so this runner cannot drift from the comparator it extends. The ONLY
deliberate difference is the composition of the transfer training pool. Its
SIZE is held at the comparator's 16 instances (4+3+3+3+3 over five campuses
rather than 4x4 over four), because training-set size is a confound and the
question is which campuses, not how many instances.

THREADS. Hard-set to one before numpy/torch import (not setdefault), and each
fit asserts it. The pipeline reproduces exactly at one numeric thread only;
more changes the floating-point reduction order. No wall-clock figure from this
run is reported as a measurement of anything, because the machine is shared.

Run:
    PYTHONPATH=src taskset -c <cores> python scripts/y3_w7_loco.py
"""

from __future__ import annotations

import os

# HARD, not setdefault: y3_p6_transfer's own module-level setdefault would
# otherwise be a no-op only by luck, and a 4-thread run does not reproduce.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import json
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch                                                     # noqa: E402

import y3_p6_transfer as P6                                      # noqa: E402
from fmwos.hitl import augmented_rule as AR                      # noqa: E402
from fmwos.hitl import overlay as ov                             # noqa: E402

_OUT = os.path.join(_ROOT, "results", "y3_w7")
_FOLDS = os.path.join(_OUT, "folds")

ALL_CAMPUSES = (1, 2, 5, 9, 10, 12)
BETAS = (1.0, 0.75)          # 1.0 first: it is the cell the manuscript quotes
SEEDS = P6.SEEDS

# ---- the comparator's locked cell, asserted rather than retyped ------------- #
LOCKED = {
    "FAMILY": "F-NL", "MASTER_SEED": 12345, "EPS": 0.0, "THETA": 1.0,
    "MECH": "targeted", "CHANNEL": "full_class_shift", "RHO": 0.25,
    "M0_ITERS": 8, "SIZE": 150, "TRACK": "replay", "TARGET_UTIL": 1.0,
    "UTIL_BAND": (0.85, 1.20), "N_TRAIN": 16, "N_PROBE": 4, "N_EVAL": 10,
    "SEEDS": (301, 302, 303),
}


def assert_locked_config():
    bad = []
    for k, v in LOCKED.items():
        got = getattr(P6, k)
        if got != v:
            bad.append(f"{k}: comparator has {got!r}, W7 expects {v!r}")
    if bad:
        raise SystemExit("CONFIG DRIFT against y3_p6_transfer:\n  "
                         + "\n  ".join(bad))
    if torch.get_num_threads() != 1:
        raise SystemExit(f"torch threads = {torch.get_num_threads()}, expected 1")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(v) != "1":
            raise SystemExit(f"{v} = {os.environ.get(v)!r}, expected '1'")
    print("[config] locked cell matches y3_p6_transfer; 1 numeric thread asserted",
          flush=True)


def split_pool(n_total, n_campus):
    """Distribute n_total training instances over n_campus campuses as evenly as
    possible, extra going to the earlier campuses. 16 over 5 -> [4,3,3,3,3]."""
    base, extra = divmod(n_total, n_campus)
    return [base + (1 if i < extra else 0) for i in range(n_campus)]


def nparams(est):
    return sum(p.numel() for p in est.parameters())


def build_fold_pools(heldout, cache, sources=None):
    """Transfer train/probe pools from the other campuses, and the held-out
    campus's own disjoint train / probe / eval slices.

    ``sources`` restricts the transfer pool to a named subset of the other
    campuses (the source-composition diagnostic). It never includes ``heldout``.
    The pool SIZE is P6.N_TRAIN either way, so only composition varies."""
    others = [c for c in (sources or ALL_CAMPUSES) if c != heldout]
    quota = split_pool(P6.N_TRAIN, len(others))
    tr_train, tr_probe = [], []
    for c, q in zip(others, quota):
        picks = cache[c][: q + 1]
        if len(picks) < q + 1:
            raise SystemExit(f"campus {c}: only {len(picks)} qualifying instances")
        tr_train += [p[1] for p in picks[:q]]
        tr_probe += [p[1] for p in picks[q: q + 1]]
    assert len(tr_train) == P6.N_TRAIN, (heldout, len(tr_train))

    need = P6.N_TRAIN + P6.N_PROBE + P6.N_EVAL
    picks = cache[heldout][:need]
    if len(picks) < need:
        raise SystemExit(f"held-out campus {heldout}: only {len(picks)} qualifying")
    ho = {
        "train": [p[1] for p in picks[: P6.N_TRAIN]],
        "probe": [p[1] for p in picks[P6.N_TRAIN: P6.N_TRAIN + P6.N_PROBE]],
        "eval": picks[P6.N_TRAIN + P6.N_PROBE: need],
        "eval_orig": [p[0] for p in picks[P6.N_TRAIN + P6.N_PROBE: need]],
    }
    # Leakage guard: the transfer pool must share no instance id with the
    # evaluation slice, and the native train slice must not either.
    ev_ids = {p[0]["meta"]["id"] for p in ho["eval"]}
    for name, pool in (("transfer_train", tr_train), ("native_train", ho["train"])):
        ids = {i["meta"]["id"].split("__")[0] for i in pool}
        if ids & ev_ids:
            raise SystemExit(f"LEAKAGE: {name} overlaps the eval slice of c{heldout}")
    return tr_train, tr_probe, ho, others, quota


def run_fold(heldout, beta, seed, cache, ref_nparams, sources=None):
    overlay = ov.Overlay(ov.OverlayParams(beta=beta, family=P6.FAMILY,
                                          master_seed=P6.MASTER_SEED,
                                          channel=P6.CHANNEL))
    tr_train, tr_probe, ho, others, quota = build_fold_pools(heldout, cache, sources)

    est_t, _ = P6.train_estimator_pool(tr_train, tr_probe, overlay, beta, seed)
    est_n, _ = P6.train_estimator_pool(ho["train"], ho["probe"], overlay, beta, seed)
    for nm, e in (("transfer", est_t), ("native", est_n)):
        n = nparams(e)
        if ref_nparams[0] is None:
            ref_nparams[0] = n
        elif n != ref_nparams[0]:
            raise SystemExit(f"PARAM DRIFT: {nm} estimator has {n}, "
                             f"expected {ref_nparams[0]}")

    lad = P6.twt_ladder(overlay, ho["eval"], {"transfer": est_t, "native": est_n},
                        seed)
    per = lad["per"]
    rec_t = AR.probe_shift_accuracy(est_t, ho["eval_orig"], overlay)
    rec_n = AR.probe_shift_accuracy(est_n, ho["eval_orig"], overlay)
    return {
        "heldout": heldout, "beta": beta, "seed": seed,
        "train_campuses": others, "train_quota": quota,
        "n_train": len(tr_train), "n_eval": len(ho["eval"]),
        "nparams": ref_nparams[0],
        "util_med": float(np.median(lad["utils"])),
        "inst_ids": lad["inst_ids"],
        "twt": {k: [float(x) for x in v] for k, v in per.items()},
        "recovery": {
            "transfer": {k: float(rec_t[k]) for k in
                         ("sign_acc_nonzero", "pearson_r", "exact_class_acc")},
            "native": {k: float(rec_n[k]) for k in
                       ("sign_acc_nonzero", "pearson_r", "exact_class_acc")},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", default="1.0,0.75")
    ap.add_argument("--campuses", default=",".join(str(c) for c in ALL_CAMPUSES))
    ap.add_argument("--sources", default="", help="restrict the transfer pool to these campuses (diagnostic)")
    ap.add_argument("--tag", default="", help="suffix for the fold cache key")
    ap.add_argument("--seeds", default="", help="override SEEDS (adds power; the cell is unchanged)")
    args = ap.parse_args()
    betas = [float(b) for b in args.betas.split(",")]
    campuses = [int(c) for c in args.campuses.split(",")]
    sources = [int(c) for c in args.sources.split(",")] if args.sources else None

    global SEEDS
    if args.seeds:
        SEEDS = tuple(int(x) for x in args.seeds.split(","))
        print("[seeds] %s (comparator locks %s; extra seeds add power only)"
              % (SEEDS, LOCKED["SEEDS"]), flush=True)
    torch.set_num_threads(1)
    assert_locked_config()
    os.makedirs(_FOLDS, exist_ok=True)

    print("[select] crew-scaling instance pools (util band %s) ..."
          % (P6.UTIL_BAND,), flush=True)
    need = P6.N_TRAIN + P6.N_PROBE + P6.N_EVAL
    cache = {}
    for c in ALL_CAMPUSES:
        cache[c] = P6.select_scaled(c, need)
        us = [p[2] for p in cache[c]]
        print("  c%02d: %d qualifying, util median %.3f [%.3f,%.3f]"
              % (c, len(cache[c]), np.median(us), min(us), max(us)), flush=True)

    ref_nparams = [None]
    t0 = time.time()
    done = 0
    total = len(campuses) * len(betas) * len(SEEDS)
    for beta in betas:
        for h in campuses:
            for seed in SEEDS:
                key = "c%02d_b%.2f_s%d%s" % (h, beta, seed, args.tag)
                path = os.path.join(_FOLDS, key + ".json")
                if os.path.exists(path):
                    with open(path) as fh:
                        ref_nparams[0] = json.load(fh).get("nparams", ref_nparams[0])
                    done += 1
                    print("  [skip] %s (cached)" % key, flush=True)
                    continue
                rec = run_fold(h, beta, seed, cache, ref_nparams, sources)
                tmp = path + ".part"
                with open(tmp, "w") as fh:
                    json.dump(rec, fh, indent=1)
                os.replace(tmp, path)          # atomic: no half-written fold
                done += 1
                r = np.array(rec["twt"]["rule"])
                t = np.array(rec["twt"]["transfer"])
                n = np.array(rec["twt"]["native"])
                print("  [fold %2d/%2d] %s  rule %.1f  transfer %+.2f%%  "
                      "native %+.2f%%  (%.0fs elapsed)"
                      % (done, total, key, r.mean(),
                         100 * (r.mean() - t.mean()) / r.mean(),
                         100 * (r.mean() - n.mean()) / r.mean(),
                         time.time() - t0), flush=True)
    print("[done] %d folds; params asserted at %s" % (done, ref_nparams[0]),
          flush=True)


if __name__ == "__main__":
    main()
