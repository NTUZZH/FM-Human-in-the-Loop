#!/usr/bin/env python
"""PI-0: a frozen Y1-style PPO policy trained IN-REGIME with NO override learning.

PI-0 is the control that isolates the value of learning FROM the supervisor's
overrides (vs. merely being a PPO policy trained on the same contended
instances). It is the Y1 PPO recipe -- the SAME network, the SAME fair-compute
env-step budget (4,915,200 == Y1), the SAME seed -- run on the SAME contention
cell as M1 (c9 storm2 u100), but with the supervisor OFF and the latent head
disabled:

    rho = 0    -> the supervisor never reviews, so no override/confirmation is
                 ever logged (D_int stays empty, the imitation term is skipped).
    gate = 0   -> the additive latent-shift head contributes nothing; the forward
                 pass is bit-identical to the Y1 DispatchPolicy.

This reuses the LOCKED, fair-compute-asserted P2 loop (``train_m1``) verbatim, so
PI-0 sees the identical PPO on the identical observable shaped reward as M1 --
the ONLY differences are the two knobs above. PI-0 vs M1 (both alone and +SUP)
therefore isolates exactly the override-learning contribution.

Run (AFTER the M1 training tmux has finished; do not run two 8-thread trainings
at once on the shared machine):
    python scripts/y3_p3_pi0.py --seed 301 --out train_log/y3_p3/pi0_full
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _SCRIPTS)

from y3_p2_train import train_m1                              # noqa: E402
from y3_p15_m1 import Storm2Sampler, _storm2_probe            # noqa: E402
from fmwos.hitl.overlay import Overlay, OverlayParams          # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--family", type=str, default="F-NL")
    ap.add_argument("--master-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--campus", type=int, default=9)
    ap.add_argument("--u", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-probe", type=int, default=6)
    ap.add_argument("--channel", type=str, default="full_class_shift",
                    choices=["full_class_shift", "weight_only"])
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--outer-iters", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    torch.set_num_threads(int(args.threads))

    # PI-0 knobs: supervisor OFF (rho=0), latent head OFF (gate=0). The overlay
    # is only used for the (harmless) TWT* reporting during training; with rho=0
    # no latent quantity reaches the learner.
    cell = {"beta": args.beta, "rho": 0.0, "eps": 0.0, "theta": 1.0,
            "mechanism": "targeted", "family": args.family,
            "master_seed": args.master_seed, "channel": args.channel,
            "regime": "storm2", "campus": args.campus, "u": args.u}

    overlay = Overlay(OverlayParams(beta=args.beta, family=args.family,
                                    master_seed=args.master_seed, channel=args.channel))
    sampler = Storm2Sampler(args.campus, args.u, args.n_train, args.seed)
    probe = _storm2_probe(args.campus, args.u, args.n_probe, skip=args.n_train)

    print("[y3_p3_pi0] PI-0 (Y1 PPO, supervisor OFF, gate=0) storm2 c%d u%d "
          "beta=%.2f seed=%d | train_pool=%d probe=%d | %d iters, fair-compute"
          % (args.campus, args.u, args.beta, args.seed, len(sampler.pool),
             len(probe), args.outer_iters))

    train_m1(cell, args.seed, args.out, outer_iters=args.outer_iters,
             budget_frac=1.0, gate=0.0, il_coef=0.0, smoke=False,
             sampler=sampler, probe=probe, overlay=overlay, n_probe=len(probe))


if __name__ == "__main__":
    main()
