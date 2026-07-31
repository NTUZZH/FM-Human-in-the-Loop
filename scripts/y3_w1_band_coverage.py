#!/usr/bin/env python
"""W1 band coverage: the QUARANTINED evaluation-only read of the latent.

The conformal band of ``fmwos.hitl.routing`` is calibrated on override-derived
weak labels and nothing else; ``routing.py`` never imports the overlay's latent
maps and its calibration functions refuse an overlay at run time. This file is
the single place where the simulator's TRUE shift is compared against the band,
and it exists only to report empirical coverage as a RESULT. Nothing here feeds
back into calibration, into the estimator, or into any routing decision:
``calibrate_band`` is never called from this module, and the functions below
return numbers, never a band.

Two coverage numbers are reported, and they answer different questions.

* Coverage of the WEAK labels on held-out reviewed decisions. This is what the
  split-conformal construction actually targets, so it should land at the
  nominal ``1 - alpha``. It is computed from an override log, using the same
  ``weak_labels_from_log`` the estimator trains on, so it reads no latent.

* Coverage of the TRUE shift over the held-out orders. This is the honest
  question a reader will ask, and there is no reason for it to reach ``1 -
  alpha``: the band targets a noisy, censored, budget-limited proxy for the
  latent, not the latent itself. The gap between the two numbers is the price of
  having only override evidence, and it is a finding, not a defect.

Run as a script for a standalone check on one cell:
    PYTHONPATH=src python scripts/y3_w1_band_coverage.py --help
"""

from __future__ import annotations

import numpy as np

from fmwos.hitl.overlay import base_features
from fmwos.hitl.augmented_rule import weak_labels_from_log


def true_shift_coverage(estimator, band, instances, overlay, device="cpu"):
    """Empirical coverage of the TRUE shift by the calibrated band.

    EVALUATION ONLY. Returns coverage, mean half-width, mean signed error, and
    the coverage split by whether the order carries a nonzero true shift (a
    band centred near zero covers the many s = 0 orders for free, so the
    nonzero-shift figure is the informative one).
    """
    s_hat_all, s_true_all, hw_all = [], [], []
    for inst in instances:
        applied = overlay.apply(inst)                     # EVAL-ONLY latent read
        shift = applied["shift"]
        wos = inst["work_orders"]
        feats = np.stack([base_features(w) for w in wos]).astype(np.float32)
        s_hat_all.append(estimator.predict_np(feats, device=device).astype(np.float64))
        hw_all.append(band.half_width(feats))
        s_true_all.append(np.asarray([shift[w["id"]] for w in wos], dtype=np.float64))
    s_hat = np.concatenate(s_hat_all)
    s_true = np.concatenate(s_true_all)
    hw = np.concatenate(hw_all)
    # The band is clipped to the protocol's shift range, exactly as the router
    # uses it, so the coverage reported is the coverage of the band in use.
    lo = np.clip(s_hat - hw, -2.0, 2.0)
    hi = np.clip(s_hat + hw, -2.0, 2.0)
    inside = (s_true >= lo - 1e-12) & (s_true <= hi + 1e-12)
    nz = s_true != 0
    return {
        "n_orders": int(s_true.size),
        "coverage_true": float(inside.mean()),
        "coverage_true_nonzero": float(inside[nz].mean()) if nz.any() else float("nan"),
        "coverage_true_zero": float(inside[~nz].mean()) if (~nz).any() else float("nan"),
        "frac_nonzero_shift": float(nz.mean()),
        "mean_half_width": float(hw.mean()),
        "mean_band_width_clipped": float((hi - lo).mean()),
        "mean_abs_error": float(np.abs(s_hat - s_true).mean()),
        "mean_signed_error": float((s_hat - s_true).mean()),
        "mean_s_hat": float(s_hat.mean()),
    }


def weak_label_coverage(estimator, band, logs_and_instances, device="cpu"):
    """Coverage of held-out OVERRIDE-DERIVED weak labels: the quantity the
    split-conformal construction targets. Reads no latent."""
    X, y = [], []
    for log, inst in logs_and_instances:
        xi, yi, _wi = weak_labels_from_log(log, inst)
        if len(xi):
            X.append(xi); y.append(yi)
    if not X:
        return {"n_weak": 0, "coverage_weak": float("nan")}
    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y).astype(np.float64)
    pred = estimator.predict_np(X, device=device).astype(np.float64)
    hw = band.half_width(X)
    inside = np.abs(pred - y) <= hw + 1e-12
    ov = y != 0
    return {
        "n_weak": int(y.size),
        "coverage_weak": float(inside.mean()),
        "coverage_weak_override": float(inside[ov].mean()) if ov.any() else float("nan"),
        "coverage_weak_confirm": float(inside[~ov].mean()) if (~ov).any() else float("nan"),
        "frac_override_labels": float(ov.mean()),
        "mean_abs_weak_residual": float(np.abs(pred - y).mean()),
    }


if __name__ == "__main__":                                  # pragma: no cover
    print(__doc__)
