#!/usr/bin/env python
"""W1 system output: the per-decision automate/refer verdict on real orders.

Retrains the correction layer at the headline cell under the deployable policy
(seed 301, cached by scripts/y3_w1_sweep.py's signature so this is a lookup, not
a second fit), then runs the augmented rule over the first held-out instance with
no supervisor and records, at every dispatch event, what the shipped system would
tell a facility manager:

    order | recorded class c | s_hat | corrected class c_hat with its interval
          | instability margin | verdict (automate / refer)

and, for the referred decisions, the rival order that could overturn the pick.
The true class c* is printed in the last column for the reader's benefit only;
it is an EVALUATION-ONLY read of the simulator's latent, exactly like the
existing case table, and no part of the verdict depends on it.

Run:  PYTHONPATH=src python scripts/y3_w1_verdicts.py
Writes results/y3_w1/verdicts_head.csv (all decisions) and a printed excerpt
chosen by a stated rule: the widest-interval refer, the tightest automate, and
the decisions closest to the stability boundary from each side.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch                                                        # noqa: E402

from fmwos.hitl import overlay as ov                                # noqa: E402
from fmwos.hitl import routing as R                                 # noqa: E402
from fmwos.hitl.latent_head import ShiftEstimator                   # noqa: E402

import y3_w1_sweep as S                                             # noqa: E402


def build_layer(seed=301, cell=None, alpha=0.1):
    """Fit the layer at the headline cell under the deployable policy."""
    cell = cell or S.HEAD
    files = S.locate_files(cell["campus"], cell["regime"], u=cell.get("u"))
    train = [S._load(p) for p in files[:16]]
    probe = [S._load(p) for p in files[16:20]]
    eval_insts = [S._load(p) for p in files[20:30]]
    overlay = ov.Overlay(ov.OverlayParams(beta=cell["beta"], family=S.FAMILY,
                                          master_seed=S.MASTER_SEED,
                                          channel=S.CHANNEL))
    torch.manual_seed(seed)
    np.random.seed(seed)
    res = R.run_m0_routed(train, probe, overlay,
                          beta_rho_eps=(cell["beta"], cell["rho"], S.EPS),
                          outer_iters=8, policy="stability", theta=S.THETA,
                          seed=seed, device="cpu", verbose=False,
                          split_fit=True, cal_frac=0.3, alpha=alpha,
                          probe=False)
    return res["estimator"], res["band"], overlay, eval_insts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--instance", type=int, default=0,
                    help="index into the ten held-out instances")
    args = ap.parse_args(argv)
    torch.set_num_threads(4)

    est, band, overlay, evals = build_layer(seed=args.seed, alpha=args.alpha)
    inst = evals[args.instance]
    vs = R.verdict_stream(est, inst, band, channel=S.CHANNEL, seed=args.seed,
                          max_records=None)

    applied = overlay.apply(inst)                 # EVAL-ONLY latent read
    cstar = applied["c_star"]
    rows = vs["records"]
    for r in rows:
        r["c_star_eval_only"] = int(cstar[r["wo"]])
        r["interval_width"] = r["c_hat_hi"] - r["c_hat_lo"]

    out = os.path.join(S._OUT, "verdicts_head.csv")
    cols = ["t_bh", "wo", "n_cand", "recorded_class", "s_hat", "s_lo", "s_hi",
            "c_hat", "c_hat_lo", "c_hat_hi", "interval_width", "margin",
            "rival", "verdict", "c_star_eval_only"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    n = len(rows)
    aut = [r for r in rows if r["verdict"] == "automate"]
    ref = [r for r in rows if r["verdict"] == "refer"]
    summary = {
        "instance": inst["meta"]["id"], "seed": args.seed, "alpha": args.alpha,
        "band_q": band.q, "n_decisions_recorded": n,
        "n_decisions_total": vs["n_decisions"],
        "counts": vs["counts"],
        "automation_coverage": vs["automation_coverage"],
        "mean_interval_width_class_units": float(np.mean([r["interval_width"]
                                                         for r in rows])) if rows else 0.0,
        "corrected_class_matches_true_automate":
            float(np.mean([abs(r["c_hat"] - r["c_star_eval_only"]) <= 0.5
                           for r in aut])) if aut else float("nan"),
        "corrected_class_matches_true_refer":
            float(np.mean([abs(r["c_hat"] - r["c_star_eval_only"]) <= 0.5
                           for r in ref])) if ref else float("nan"),
        "true_class_inside_interval_automate":
            float(np.mean([r["c_hat_lo"] - 1e-9 <= r["c_star_eval_only"] <= r["c_hat_hi"] + 1e-9
                           for r in aut])) if aut else float("nan"),
        "true_class_inside_interval_refer":
            float(np.mean([r["c_hat_lo"] - 1e-9 <= r["c_star_eval_only"] <= r["c_hat_hi"] + 1e-9
                           for r in ref])) if ref else float("nan"),
    }
    with open(os.path.join(S._OUT, "verdicts_head_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    print("instance %s, seed %d, alpha %.2f, band q = %.3f"
          % (inst["meta"]["id"], args.seed, args.alpha, band.q))
    print("decisions: %d automate, %d refer, %d forced (single candidate); "
          "automation coverage %.3f"
          % (vs["counts"]["automate"], vs["counts"]["refer"],
             vs["counts"]["forced"], vs["automation_coverage"]))
    print("mean corrected-class interval width: %.3f classes"
          % summary["mean_interval_width_class_units"])
    print("true class inside the printed interval: automate %.3f, refer %.3f"
          % (summary["true_class_inside_interval_automate"],
             summary["true_class_inside_interval_refer"]))

    def show(title, rs):
        print("\n%s" % title)
        print("  %-8s %4s %6s %6s %6s %-16s %10s %-9s %5s"
              % ("order", "cand", "c", "s_hat", "c_hat", "interval", "margin",
                 "verdict", "c*"))
        for r in rs:
            print("  %-8s %4d %6d %+6.2f %6.2f [%5.2f, %5.2f]    %10.4f %-9s %5d"
                  % (r["wo"], r["n_cand"], r["recorded_class"], r["s_hat"],
                     r["c_hat"], r["c_hat_lo"], r["c_hat_hi"], r["margin"],
                     r["verdict"], r["c_star_eval_only"]))

    if ref:
        show("widest corrected-class interval among REFERRED decisions",
             sorted(ref, key=lambda r: -r["interval_width"])[:3])
        show("closest to the stability boundary from the referred side",
             sorted(ref, key=lambda r: -r["margin"])[:3])
    if aut:
        show("closest to the stability boundary from the automated side",
             sorted(aut, key=lambda r: r["margin"])[:3])
        show("most comfortably automated (largest margin)",
             sorted(aut, key=lambda r: -r["margin"])[:3])
    print("\nwrote %s (%d rows) and verdicts_head_summary.json" % (out, n))


if __name__ == "__main__":
    main()
