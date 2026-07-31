#!/usr/bin/env python
"""W2 R0: pipeline-fidelity pilot.

Reproduces, through the W2 code path with the INCUMBENT squared-error estimator
selected, the published M0 headline number at the headline cell (campus 9,
storm2 w80 u100, beta 1.0, rho 0.25, eps 0, theta 1, seed 301, 10 held-out
instances). Nothing else in W2 is launched until this agrees.

The comparison target is recomputed from results/y3_p4/m0_gate.csv, not copied
from paper/macros.tex, and the resolved configuration is diffed field-by-field
against the published y3_p4_m0grid task before the run starts.

Run:
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
    taskset -c 10-19 python scripts/y3_w2_pilot.py --threads 1
"""

import argparse
import csv
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

import y3_w2_lib as L                                           # noqa: E402

_ROOT = L._ROOT
_GATE_CSV = os.path.join(_ROOT, "results", "y3_p4", "m0_gate.csv")
_MACROS = os.path.join(_ROOT, "paper", "macros.tex")

# The published task, transcribed from scripts/y3_p4_m0grid.py::_base_task and
# tasks_A. Any drift between this and L.build_cell aborts the run.
PUBLISHED_TASK = {
    "campus": 9, "regime": "storm2", "u": 100, "size": None, "beta": 1.0,
    "rho": 0.25, "eps": 0.0, "theta": 1.0, "mech": "targeted",
    "channel": "full_class_shift", "family": "F-NL", "master_seed": 12345,
    "n_train": 16, "n_probe": 4, "n_eval": 10, "m0_iters": 8,
}


def published_target(seed=301):
    """Per-instance published m0_alone / rule TWT* at the headline cell."""
    rows = [r for r in csv.DictReader(open(_GATE_CSV))
            if r["campus"] == "9" and r["regime"] == "storm2"
            and r["u"] == "100" and r["beta"] == "1.0" and r["rho"] == "0.25"]
    if not rows:
        raise RuntimeError("headline cell not found in %s" % _GATE_CSV)
    mine = [r for r in rows if int(r["seed"]) == seed]
    inst_ids = [r["inst_id"] for r in mine]
    rule = np.asarray([float(r["rule"]) for r in mine])
    m0 = np.asarray([float(r["m0_alone"]) for r in mine])
    # 10-seed aggregation, exactly y3_p4_m0grid._contrast: seed-average per
    # instance, then the pooled percentage. This is what \MzeroGain quotes.
    by_r, by_m = {}, {}
    for r in rows:
        by_r.setdefault(r["inst_id"], []).append(float(r["rule"]))
        by_m.setdefault(r["inst_id"], []).append(float(r["m0_alone"]))
    R = float(np.mean([np.mean(v) for v in by_r.values()]))
    M = float(np.mean([np.mean(v) for v in by_m.values()]))
    return {"inst_ids": inst_ids, "rule": rule, "m0_alone": m0,
            "pct_seed": 100.0 * (rule.mean() - m0.mean()) / rule.mean(),
            "pct_allseeds": 100.0 * (R - M) / R,
            "n_seeds": len({r["seed"] for r in rows})}


def macro_value():
    for line in open(_MACROS):
        if line.startswith("\\newcommand{\\MzeroGain}"):
            return line.strip()
    return "(not found)"


def check_config(cell):
    got = {"campus": cell["campus"], "regime": "storm2", "u": cell["u"],
           "size": None, "beta": cell["beta"], "rho": cell["rho"],
           "eps": cell["eps"], "theta": cell["theta"], "mech": cell["mech"],
           "channel": cell["channel"], "family": cell["family"],
           "master_seed": cell["master_seed"], "n_train": cell["n_train"],
           "n_probe": cell["n_probe"], "n_eval": cell["n_eval"],
           "m0_iters": cell["m0_iters"]}
    diff = {k: (PUBLISHED_TASK[k], got[k]) for k in PUBLISHED_TASK
            if PUBLISHED_TASK[k] != got[k]}
    if diff:
        raise SystemExit("resolved config differs from the published task: %r" % diff)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads; the published run used 1")
    ap.add_argument("--out", default=os.path.join(L.OUT, "pilot.json"))
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    tgt = published_target(a.seed)
    cell = L.build_cell(campus=9, u=100, beta=1.0, rho=0.25, seed=a.seed)
    cfg = check_config(cell)
    assert cell["eval_ids"] == tgt["inst_ids"], (
        "held-out instance set differs from the published run:\n  mine=%r\n  pub =%r"
        % (cell["eval_ids"], tgt["inst_ids"]))

    print("config diff vs published task: NONE (%d fields checked)" % len(cfg))
    print("held-out instances match the published run: %d" % len(tgt["inst_ids"]))
    print("training rung (i) mse_published (torch threads=%d) ..." % a.threads)
    t0 = time.perf_counter()
    tr = L.train_variant(cell, "mse_published", verbose=True)
    dep = L.deployed_twt(tr["model"], cell["eval"], cell["overlay"],
                         channel=cell["channel"], seed=a.seed)
    secs = time.perf_counter() - t0

    rule = np.asarray(dep["rule"]); aug = np.asarray(dep["aug"])
    pct = 100.0 * (rule.mean() - aug.mean()) / rule.mean()
    d_m0 = np.abs(aug - tgt["m0_alone"])
    d_rule = np.abs(rule - tgt["rule"])

    out = {
        "run": "R0 pipeline-fidelity pilot", "seed": a.seed,
        "torch_threads": a.threads, "torch": torch.__version__,
        "numpy": np.__version__, "elapsed_s_upper_bound": secs,
        "config": cfg, "files": cell["files"], "eval_ids": cell["eval_ids"],
        "published": {"macro_line": macro_value(),
                      "MzeroGain_pct_allseeds": tgt["pct_allseeds"],
                      "n_seeds": tgt["n_seeds"],
                      "pct_seed%d" % a.seed: tgt["pct_seed"],
                      "rule_per_instance": tgt["rule"].tolist(),
                      "m0_alone_per_instance": tgt["m0_alone"].tolist()},
        "mine": {"pct_seed%d" % a.seed: pct,
                 "rule_per_instance": rule.tolist(),
                 "aug_per_instance": aug.tolist()},
        "difference": {"pct_points": pct - tgt["pct_seed"],
                       "max_abs_twt_m0_alone": float(d_m0.max()),
                       "max_abs_twt_rule": float(d_rule.max()),
                       "n_instances": int(rule.size)},
        "recovery_final_iter": {k: tr["per_iter"][-1][k] for k in
                                ("pearson_r", "sign_acc_nonzero",
                                 "exact_class_acc", "override_rate")},
        "n_params_estimator": tr["n_params_estimator"],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(a.out + ".tmp", a.out)

    print()
    print("published (macro)      : %s" % out["published"]["macro_line"])
    print("published 10-seed mean : %.4f %%" % tgt["pct_allseeds"])
    print("published seed %-4d    : %.4f %%" % (a.seed, tgt["pct_seed"]))
    print("mine     seed %-4d     : %.4f %%" % (a.seed, pct))
    print("difference             : %+.6f pct points" % (pct - tgt["pct_seed"]))
    print("max |dTWT*| m0_alone   : %.10g   over %d instances" % (d_m0.max(), rule.size))
    print("max |dTWT*| rule       : %.10g" % d_rule.max())
    print("elapsed (upper bound, contended box): %.1f s" % secs)
    ok = d_m0.max() < 1e-6 and d_rule.max() < 1e-6
    print("FIDELITY: %s" % ("EXACT" if ok else "MISMATCH -- do not proceed"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
