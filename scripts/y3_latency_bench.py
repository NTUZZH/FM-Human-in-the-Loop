#!/usr/bin/env python
"""Paper Y3 -- committed source for the \\MzeroLatency macro.

Micro-benchmark of the correction layer's per-decision inference cost: a single
forward pass of the trained ShiftEstimator (MLP 20 -> 32 -> 32 -> 1, ~1.8k params)
predicting one order's class shift hat_s at dispatch time. This is the ONLY extra
work M0 adds on top of the base rule's own scoring.

Pinned to 4 threads (OMP/MKL/torch) to match the reported protocol; reports the
single-order min and median over many calls. Run once and record the value in the
macros.tex comment for \\MzeroLatency.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fmwos.hitl.latent_head import ShiftEstimator, LAT_DIM  # noqa: E402

torch.set_num_threads(4)


def main(n_warmup: int = 500, n_iter: int = 5000, seed: int = 301) -> None:
    rng = np.random.default_rng(seed)
    est = ShiftEstimator(lat_dim=LAT_DIM, hidden=32)
    est.eval()
    feats = rng.standard_normal((1, LAT_DIM)).astype(np.float32)  # one order

    for _ in range(n_warmup):
        est.predict_np(feats)

    times_ms = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        x = rng.standard_normal((1, LAT_DIM)).astype(np.float32)
        t0 = time.perf_counter()
        est.predict_np(x)
        times_ms[i] = (time.perf_counter() - t0) * 1e3

    print(f"ShiftEstimator single-order predict_np  (LAT_DIM={LAT_DIM}, hidden=32, "
          f"{sum(p.numel() for p in est.parameters())} params)")
    print(f"  n={n_iter}, threads=4")
    print(f"  min    {times_ms.min():.4f} ms")
    print(f"  median {np.median(times_ms):.4f} ms")
    print(f"  mean   {times_ms.mean():.4f} ms")


if __name__ == "__main__":
    main()
