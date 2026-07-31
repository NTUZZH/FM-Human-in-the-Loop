#!/usr/bin/env python
"""W3 runner: one (cell, variant, seed) per process, four processes at most.

Each job writes results/y3_w3/<tag>/<variant>_s<seed>.json and never touches a
file another job owns, so a killed sweep cannot leave a half-written number for
the summary to sweep in.

Every job asserts, before a gradient step is taken:
  * the estimator's parameter count equals the incumbent's 1761 (in
    ``y3_w3_lib.train_variant``);
  * the resolved configuration differs from rung (i)'s in nothing but the
    variant fields;
  * the resolved held-out instance ids equal the published run's ``inst_id``
    column for the same (campus, u, beta, seed), when that cell is published.

Run (single-threaded, pinned, four workers):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-23 \
    python scripts/y3_w3_run.py --cell c9_u130_b0 --workers 4
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse                                                  # noqa: E402
import csv                                                       # noqa: E402
import json                                                      # noqa: E402
import sys                                                       # noqa: E402
import time                                                      # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

import y3_w3_lib as L                                            # noqa: E402

_ROOT = L._ROOT
_E3_CSV = os.path.join(_ROOT, "results", "y3_p4", "e3_map.csv")
_GATE_CSV = os.path.join(_ROOT, "results", "y3_p4", "m0_gate.csv")

# The cells. `protocol` names the published call site whose fitting protocol is
# reproduced, so a reduced-protocol result can never be pooled with a full one.
CELLS = {
    # the published \BetaZeroHatSMean provenance: y3_beta0_check.train_hat_s
    # (campus 9, u130, beta 0, seed 301) runs 12 training instances / 5 iters.
    "c9_u130_b0_pub": dict(campus=9, u=130, beta=0.0, rho=0.25, n_train=12,
                           n_probe=4, n_eval=10, m0_iters=5,
                           protocol="y3_beta0_check.train_hat_s (12/5)",
                           published_csv=None),
    # the regime map: y3_p4_m0grid._base_task + tasks_B
    "c9_u130_b0": dict(campus=9, u=130, beta=0.0, rho=0.25, n_train=16,
                       n_probe=4, n_eval=10, m0_iters=8,
                       protocol="y3_p4_m0grid tasks_B (16/8)",
                       published_csv="e3"),
    "c9_u100_b0": dict(campus=9, u=100, beta=0.0, rho=0.25, n_train=16,
                       n_probe=4, n_eval=10, m0_iters=8,
                       protocol="y3_p4_m0grid tasks_B (16/8)",
                       published_csv="e3"),
    "c10_u130_b0": dict(campus=10, u=130, beta=0.0, rho=0.25, n_train=16,
                        n_probe=4, n_eval=10, m0_iters=8,
                        protocol="y3_p4_m0grid tasks_B (16/8)",
                        published_csv="e3"),
    "c10_u100_b0": dict(campus=10, u=100, beta=0.0, rho=0.25, n_train=16,
                        n_probe=4, n_eval=10, m0_iters=8,
                        protocol="y3_p4_m0grid tasks_B (16/8)",
                        published_csv="e3"),
    # the headline cell: y3_p4_m0grid tasks_A
    "c9_u100_b1": dict(campus=9, u=100, beta=1.0, rho=0.25, n_train=16,
                       n_probe=4, n_eval=10, m0_iters=8,
                       protocol="y3_p4_m0grid tasks_A (16/8)",
                       published_csv="gate"),
}


def published_rows(kind, campus, u, beta, rho, seed):
    """Per-instance published rule / m0_alone TWT* for a cell-seed, or None."""
    if kind is None:
        return None
    path = _E3_CSV if kind == "e3" else _GATE_CSV
    out = []
    for r in csv.DictReader(open(path)):
        if (r["campus"] == str(campus) and int(float(r["u"])) == int(u)
                and float(r["beta"]) == float(beta)
                and float(r["rho"]) == float(rho) and int(r["seed"]) == int(seed)):
            out.append(r)
    if not out:
        return None
    return {"inst_ids": [r["inst_id"] for r in out],
            "rule": [float(r["rule"]) for r in out],
            "m0_alone": [float(r["m0_alone"]) for r in out]}


def job(args):
    tag, variant, seed, skip_kendall = args
    L.set_threads(1)
    try:
        os.nice(5)
    except Exception:
        pass
    spec = dict(CELLS[tag])
    pub_kind = spec.pop("published_csv")
    protocol = spec.pop("protocol")
    t0 = time.perf_counter()

    cell = L.build_cell(seed=seed, **spec)
    cfg = L.resolved_config(cell, variant, extra={"protocol": protocol})
    ref = L.resolved_config(cell, "mse_published", extra={"protocol": protocol})
    d = L.config_diff(ref, cfg)
    if d:
        raise SystemExit("config drift on %s/%s/s%d: %r" % (tag, variant, seed, d))

    pub = published_rows(pub_kind, cell["campus"], cell["u"], cell["beta"],
                         cell["rho"], seed)
    if pub is not None and cell["eval_ids"] != pub["inst_ids"]:
        raise SystemExit("held-out instances differ from the published run on "
                         "%s s%d:\n mine=%r\n pub =%r"
                         % (tag, seed, cell["eval_ids"], pub["inst_ids"]))

    tr = L.train_variant(cell, variant, verbose=False)
    dep = L.deployed_twt(tr["model"], cell["eval"], cell["overlay"],
                         channel=cell["channel"], seed=seed)
    rec = L.probe_recovery(tr["model"], cell["eval"], cell["overlay"])
    if skip_kendall:
        # Kendall tau dominates the cost on the large campus-10 instances and is
        # not needed for the before/after TWT* comparison on those cells; it is
        # reported as not computed rather than silently defaulted.
        ken = ken0 = {"kendall_tau": float("nan"), "n_decisions": 0,
                      "skipped": True}
    else:
        ken_cache = {}
        ken = L.kendall_on_instances(tr["model"], cell["eval"], cell["overlay"],
                                     channel=cell["channel"], seed=seed,
                                     cache=ken_cache)
        ken0 = L.kendall_on_instances(tr["model"], cell["eval"], cell["overlay"],
                                      channel=cell["channel"], seed=seed,
                                      zero=True, cache=ken_cache)

    rule = np.asarray(dep["rule"]); aug = np.asarray(dep["aug"])
    out = {
        "tag": tag, "variant": variant, "seed": seed, "config": cfg,
        "files": cell["files"], "eval_ids": cell["eval_ids"],
        "n_params_estimator": tr["n_params_estimator"],
        "n_params_total": tr["n_params_total"],
        "twt": {"rule": rule.tolist(), "aug": aug.tolist(),
                "pct_below_rule": 100.0 * (rule.mean() - aug.mean()) / rule.mean()},
        "recovery_eval": rec,
        "recovery_probe_last_iter": (tr["per_iter"][-1] if tr["per_iter"] else {}),
        "kendall": ken, "kendall_recorded_field_floor": ken0,
        "censor": (tr["censor"][-1] if tr["censor"] else {}),
        "per_iter": tr["per_iter"],
        "sigma": tr["model"].sigma_value(),
        "class_constant": tr.get("class_constant"),
        "elapsed_s_upper_bound": time.perf_counter() - t0,
    }
    if pub is not None:
        out["published"] = {
            "rule": pub["rule"], "m0_alone": pub["m0_alone"],
            "pct_below_rule": 100.0 * (np.mean(pub["rule"]) - np.mean(pub["m0_alone"]))
            / np.mean(pub["rule"]),
            "max_abs_dTWT_rule": float(np.abs(rule - np.asarray(pub["rule"])).max()),
            "max_abs_dTWT_m0_vs_mine": float(np.abs(aug - np.asarray(pub["m0_alone"])).max()),
        }
    path = os.path.join(L.OUT, tag, "%s_s%d.json" % (variant, seed))
    L.write_json(path, out)
    return {"tag": tag, "variant": variant, "seed": seed,
            "pct": out["twt"]["pct_below_rule"],
            "mean_hat_s": rec["mean_hat_s"],
            "mean_applied": rec["mean_applied_shift"],
            "pearson": rec["pearson_r"], "kendall": ken["kendall_tau"],
            "secs": out["elapsed_s_upper_bound"], "path": path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--variants", default=",".join(L.VARIANTS))
    ap.add_argument("--seeds", default="301,302,303")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-kendall", action="store_true",
                    help="do not compute Kendall tau (its cost dominates on the "
                         "campus-10 cells); it is recorded as not computed")
    a = ap.parse_args()
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    seeds = [int(s) for s in a.seeds.split(",")]
    for v in variants:
        if v not in L.VARIANTS:
            raise SystemExit("unknown variant %r" % v)
    jobs = [(a.cell, v, s, a.skip_kendall) for s in seeds for v in variants]
    print("[w3] cell=%s  %d jobs (%d variants x %d seeds), %d workers"
          % (a.cell, len(jobs), len(variants), len(seeds), a.workers), flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fut = {ex.submit(job, j): j for j in jobs}
        for f in as_completed(fut):
            r = f.result()
            done += 1
            print("  [%2d/%2d] %-16s s%d  %7.3f%% <RULE  mean_hat_s=%+.4f "
                  "applied=%+.4f  r=%+.3f  tau=%.3f  (%.0fs, wall %.0fs)"
                  % (done, len(jobs), r["variant"], r["seed"], r["pct"],
                     r["mean_hat_s"], r["mean_applied"], r["pearson"],
                     r["kendall"], r["secs"], time.time() - t0), flush=True)
    print("[w3] cell=%s complete." % a.cell, flush=True)


if __name__ == "__main__":
    main()
