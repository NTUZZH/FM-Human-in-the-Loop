#!/usr/bin/env python
"""P1.5 M1 DAgger runner on the headline contention cell (storm2), scored on the
full-class-shift objective TWT*(w*, d*).

Reuses the locked P2 loop ``scripts/y3_p2_train.train_m1`` verbatim (fair-compute
budget, PPO + intervention-weighted imitation, gated latent-shift head, no-leak
red lines) but INJECTS:
  * a storm2 sampler (campus c9, u100 by default) instead of the replay-campus
    sampler, so the loop trains on the contention regime where the deadline
    channel binds;
  * a storm2 probe set for hat_s recovery + policy-alone TWT* eval;
  * a full_class_shift overlay (channel default), so the supervisor's preferred
    pick and the true-objective eval both use the true deadline d*.

The RL reward stays the env's observable shaped reward (recorded fields); d*
reaches the learner only through the supervisor's override log + imitation target.

Smoke: python scripts/y3_p15_m1.py --smoke   (2 iters, 10% budget)
Full : python scripts/y3_p15_m1.py --beta 1.0 --seed 301 --out train_log/y3_p15/m1_full
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _SCRIPTS)                              # to import y3_p2_train

from y3_p2_train import train_m1                          # noqa: E402
from fmwos.hitl.overlay import Overlay, OverlayParams      # noqa: E402
from fmwos.hitl.intervention import OVERRIDE_WEIGHT, CONFIRM_WEIGHT  # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")


class Storm2Sampler:
    """Deterministic sampler over a storm2 train pool (campus/u fixed).

    ``sample()`` draws one instance from the pool with the seeded rng (with
    replacement, like the replay sampler), returning the pre-loaded dict. Carries
    a ``_cache`` attribute for API compatibility with the default runner path."""

    def __init__(self, campus, u, n_train, seed):
        cdir = "c%02d" % campus
        files = sorted(glob.glob(os.path.join(
            _INST, cdir, "storm2", "w80",
            "%s_storm2_w80_u%d_*.json" % (cdir, u))))
        if not files:
            raise RuntimeError("no storm2 instances at %s u%d" % (cdir, u))
        self.pool = []
        for f in files[:n_train]:
            with open(f) as fh:
                self.pool.append(json.load(fh))
        self.rng = random.Random(seed + 777)
        self._cache = {}

    def sample(self):
        return self.rng.choice(self.pool)


def _storm2_probe(campus, u, n_probe, skip):
    cdir = "c%02d" % campus
    files = sorted(glob.glob(os.path.join(
        _INST, cdir, "storm2", "w80", "%s_storm2_w80_u%d_*.json" % (cdir, u))))
    picked = files[skip:skip + n_probe]
    out = []
    for f in picked:
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--family", type=str, default="F-NL")
    ap.add_argument("--master-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--campus", type=int, default=9)
    ap.add_argument("--u", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-probe", type=int, default=6)
    ap.add_argument("--gate", type=float, default=1.0)
    ap.add_argument("--channel", type=str, default="full_class_shift",
                    choices=["full_class_shift", "weight_only"])
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--outer-iters", type=int, default=8)
    ap.add_argument("--budget-frac", type=float, default=1.0)
    ap.add_argument("--deadline-head", action="store_true",
                    help="fair M1: add the in-network ATC-slack deadline head "
                         "(E0-preserving, zero-init); default OFF = old M1")
    ap.add_argument("--label-source", type=str, default="preferred",
                    choices=["preferred", "executed"],
                    help="imitation/weak-label target on OVERRIDES: 'preferred' "
                         "(committed) or 'executed' (honest under eps>0 noise; "
                         "bit-identical at eps=0)")
    ap.add_argument("--il-pure", action="store_true",
                    help="IL-PURE ablation: zero the PPO loss, learn from the "
                         "imitation term only (rollouts/env-steps unchanged)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 outer iters, 10%% budget (fair-compute equality relaxed)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    torch.set_num_threads(int(args.threads))
    outer_iters = 2 if args.smoke else args.outer_iters
    budget_frac = 0.10 if args.smoke else args.budget_frac

    cell = {"beta": args.beta, "rho": args.rho, "eps": args.eps, "theta": args.theta,
            "mechanism": "targeted", "family": args.family,
            "master_seed": args.master_seed, "channel": args.channel,
            "regime": "storm2", "campus": args.campus, "u": args.u}

    overlay = Overlay(OverlayParams(beta=args.beta, family=args.family,
                                    master_seed=args.master_seed, channel=args.channel))
    sampler = Storm2Sampler(args.campus, args.u, args.n_train, args.seed)
    probe = _storm2_probe(args.campus, args.u, args.n_probe, skip=args.n_train)

    print("[y3_p15_m1] storm2 c%d u%d channel=%s beta=%.2f rho=%.2f gate=%.1f "
          "deadline_head=%s | train_pool=%d probe=%d | %s (%d iters, frac=%.2f)"
          % (args.campus, args.u, args.channel, args.beta, args.rho, args.gate,
             args.deadline_head, len(sampler.pool), len(probe),
             "SMOKE" if args.smoke else "FULL", outer_iters, budget_frac))

    train_m1(cell, args.seed, args.out, outer_iters=outer_iters,
             budget_frac=budget_frac, gate=args.gate, il_coef=1.0,
             smoke=args.smoke, sampler=sampler, probe=probe, overlay=overlay,
             n_probe=len(probe), deadline_head=args.deadline_head,
             label_source=args.label_source, il_pure=args.il_pure)


if __name__ == "__main__":
    main()
