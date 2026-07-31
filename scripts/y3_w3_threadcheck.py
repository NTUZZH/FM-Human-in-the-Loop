#!/usr/bin/env python
"""W3 reproduction check on the published \\BetaZeroHatSMean provenance run.

``paper/macros.tex`` sources ``\\BetaZeroHatSMean`` = 0.04 to
``results/y3_p5/beta0_check.json``, field
``hat_s_c9_u130_beta0_seed301.overall_mean_hat_s``. That file was produced by
``scripts/y3_beta0_check.py``, which sets ``torch.set_num_threads(4)`` at import
(line 20), whereas every regime-map and gate number in ``results/y3_p4`` was
produced with ``torch.set_num_threads(1)`` (``y3_p4_m0grid.evaluate_cell``).

W2 measured that changing only the intra-op thread count changes the estimator's
floating-point reduction order and moves the headline by 1.56 percentage points.
This script therefore refits the SAME protocol at 1 and at 4 threads and reports
both against the published value, so the difference is measured rather than
assumed. It quotes nothing; it decides which value W3's before/after table is
allowed to call "the published number".

Run:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-23 \
    python scripts/y3_w3_threadcheck.py
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

import y3_w3_lib as L                                            # noqa: E402

_ROOT = L._ROOT
_PUB = os.path.join(_ROOT, "results", "y3_p5", "beta0_check.json")

# transcribed from scripts/y3_beta0_check.py::main ->
#   train_hat_s(9, 130, beta=0.0, seed=301, n_train=12, iters=5)
# with train_hat_s's own defaults rho=0.25, n_probe=4, n_eval=10.
PROTOCOL = dict(campus=9, u=130, beta=0.0, rho=0.25, seed=301, n_train=12,
                n_probe=4, n_eval=10, m0_iters=5)


def main():
    pub = json.load(open(_PUB))["hat_s_c9_u130_beta0_seed301"]
    out = {"run": "W3 reproduction check on the BetaZeroHatSMean provenance",
           "protocol": PROTOCOL,
           "published_file": os.path.relpath(_PUB, _ROOT),
           "published": {"overall_mean_hat_s": pub["overall_mean_hat_s"],
                         "overall_std_hat_s": pub["overall_std_hat_s"],
                         "final_pearson_r": pub["final_pearson_r"],
                         "final_sign_acc": pub["final_sign_acc"],
                         "E_hat_s_by_recorded_class": pub["E_hat_s_by_recorded_class"],
                         "producer_torch_threads": 4},
           "mine": {}}
    for threads in (1, 4):
        L.set_threads(threads)
        cell = L.build_cell(**PROTOCOL)
        tr = L.train_variant(cell, "mse_published", verbose=False)
        rec = L.probe_recovery(tr["model"], cell["eval"], cell["overlay"])
        out["mine"][str(threads)] = {
            "torch_threads": threads,
            "overall_mean_hat_s": rec["mean_hat_s"],
            "overall_std_hat_s": rec["sd_hat_s"],
            "pearson_r": rec["pearson_r"],
            "sign_acc_nonzero": rec["sign_acc_nonzero"],
            "mean_applied_shift": rec["mean_applied_shift"],
            "by_recorded_class": rec["by_recorded_class"],
            "d_vs_published": rec["mean_hat_s"] - pub["overall_mean_hat_s"]}
        print("threads=%d  mean_hat_s=%+.6f (published %+.6f, d=%+.6f)  "
              "pearson=%+.6f (published %+.6f)"
              % (threads, rec["mean_hat_s"], pub["overall_mean_hat_s"],
                 rec["mean_hat_s"] - pub["overall_mean_hat_s"],
                 rec["pearson_r"], pub["final_pearson_r"]), flush=True)
    L.write_json(os.path.join(L.OUT, "threadcheck.json"), out)
    print("wrote", os.path.join(L.OUT, "threadcheck.json"))


if __name__ == "__main__":
    main()
