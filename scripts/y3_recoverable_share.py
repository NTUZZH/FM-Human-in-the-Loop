"""Empirical recoverable-share report (Paper Y3, Phase P1, deliverable H).

Quantization (s = clip(round(sigma_s*xi), -2, +2)) bends the exact
variance-share semantics of beta slightly. This script measures the INDUCED
class-level recoverable share after quantization on the training-campus order
population, for beta in {0, .25, .5, .75, 1} and both f families. It is the
Appendix-sentence check that quantization preserves the sweep's semantics.

For each order, xi = sqrt(beta)*f(x) + sqrt(1-beta)*z with z ~ N(0,1), so given
x the pre-round latent is Normal(mu, sd^2) with mu = sqrt(beta)*f(x),
sd = sqrt(1-beta). The distribution of the integer shift s is obtained exactly
from the normal CDF over the rounding/clipping bin edges; no Monte Carlo. Then:

  recoverable_share = Var_x( E[s | x] ) / Var(s)          (variance of the
                                                          conditional mean over
                                                          the total variance)
  argmax_accuracy   = E_x[ P(s = argmax_k P(s=k|x) | x) ] (accuracy of predicting
                                                          the realized s from x
                                                          using the true f)
  base_accuracy     = E_x[ P(s = 0 | x) ]                 (always-predict-0)

Writes results/y3_p1/recoverable_share.csv and prints a summary.
"""

import csv
import os
import sys

import numpy as np
from scipy.stats import norm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.hitl import overlay as ov                  # noqa: E402

BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
FAMILIES = ["F-LIN", "F-NL"]
MASTER_SEED = 12345
_EDGES = np.array([-1.5, -0.5, 0.5, 1.5])             # round/clip bin edges
_KVALS = np.array([-2, -1, 0, 1, 2], dtype=float)
_OUT = os.path.join(_ROOT, "results", "y3_p1", "recoverable_share.csv")


def _f_over_population(family):
    """Standardized f(x) for every training-population order (in one pass)."""
    coeffs = ov.get_coeffs(family, MASTER_SEED)
    pop = ov.load_training_population()
    base = pop["base"]
    feat_mean = np.asarray(coeffs["feat_mean"])
    feat_std = np.asarray(coeffs["feat_std"])
    a = np.asarray(coeffs["a"])
    f_raw = (base - feat_mean) / feat_std @ a
    for it in coeffs["interactions"]:
        if it["type"] == "trade_bucket":
            other = pop["bucket_idx"]
        else:
            other = pop["day_idx"]
        g = ((pop["trade_idx"] == it["trade_idx"]) & (other == it["other_idx"])).astype(float)
        f_raw = f_raw + it["b"] * (g - it["g_mean"]) / it["g_std"]
    return (f_raw - coeffs["f_mean"]) / coeffs["f_std"]


def _shift_probs(mu, sd):
    """P(s=k|x) for k in [-2..2], shape [N, 5], via the normal CDF (sd may be 0)."""
    n = mu.shape[0]
    if sd <= 1e-12:
        s = np.clip(np.round(mu), -2, 2)
        p = np.zeros((n, 5))
        p[np.arange(n), (s + 2).astype(int)] = 1.0
        return p
    cdf = norm.cdf((_EDGES[None, :] - mu[:, None]) / sd)     # [N,4]
    p = np.zeros((mu.shape[0], 5))
    p[:, 0] = cdf[:, 0]
    p[:, 1] = cdf[:, 1] - cdf[:, 0]
    p[:, 2] = cdf[:, 2] - cdf[:, 1]
    p[:, 3] = cdf[:, 3] - cdf[:, 2]
    p[:, 4] = 1.0 - cdf[:, 3]
    return p


def compute(family):
    f = _f_over_population(family)
    rows = []
    for beta in BETAS:
        mu = np.sqrt(beta) * f
        sd = np.sqrt(1.0 - beta)
        p = _shift_probs(mu, sd)                     # [N,5]
        e_s = p @ _KVALS                             # E[s|x]
        e_s2 = p @ (_KVALS ** 2)                     # E[s^2|x]
        mean_s = float(e_s.mean())
        var_cond_mean = float((e_s ** 2).mean() - mean_s ** 2)   # Var_x(E[s|x])
        var_s = float(e_s2.mean() - mean_s ** 2)                 # total Var(s)
        share = (var_cond_mean / var_s) if var_s > 1e-12 else 0.0
        argmax_acc = float(p.max(axis=1).mean())
        base_acc = float(p[:, 2].mean())             # always predict s=0
        rows.append({
            "family": family, "beta": beta,
            "var_conditional_mean": var_cond_mean, "var_s": var_s,
            "recoverable_share": share,
            "argmax_accuracy": argmax_acc, "base_accuracy": base_acc,
            "n_orders": int(f.shape[0]),
        })
    return rows


def main():
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    all_rows = []
    for fam in FAMILIES:
        all_rows.extend(compute(fam))
    cols = ["family", "beta", "recoverable_share", "var_conditional_mean",
            "var_s", "argmax_accuracy", "base_accuracy", "n_orders"]
    with open(_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in cols})
    print("wrote %s (%d rows)\n" % (_OUT, len(all_rows)))
    print("%-6s %5s  %-16s %-14s %-13s" %
          ("family", "beta", "recoverable_share", "argmax_acc", "var_s"))
    for r in all_rows:
        print("%-6s %5.2f  %-16.4f %-14.4f %-13.4f" %
              (r["family"], r["beta"], r["recoverable_share"],
               r["argmax_accuracy"], r["var_s"]))


if __name__ == "__main__":
    main()
