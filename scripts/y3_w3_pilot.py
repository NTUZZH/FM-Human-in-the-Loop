#!/usr/bin/env python
"""W3 R0: pipeline-fidelity pilot.

Two gates, both of which must pass before any W3 comparison is launched.

  (a) REPRODUCTION. The incumbent rung, run through the W3 harness, reproduces
      the published M0 headline number at the headline cell (campus 9, storm2
      w80 u100, beta 1.0, rho 0.25, eps 0, theta 1, seed 301, 10 held-out
      instances). The comparison target is recomputed from
      results/y3_p4/m0_gate.csv, never copied from paper/macros.tex, and the
      resolved configuration is diffed field-by-field against the published
      y3_p4_m0grid task before the run starts.
  (b) BIT-EXACT RE-EXPRESSION. The re-expressed outer loop with censoring
      switched off (``mse_reexpr``) reproduces the shipped pipeline
      (``mse_published``) to the last decimal on every held-out instance. This
      is what licenses reading every censored-rung difference as the likelihood
      and nothing else.

Run:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-23 \
    python scripts/y3_w3_pilot.py
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

import y3_w3_lib as L                                           # noqa: E402

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


def macro_value(name="MzeroGain"):
    for line in open(_MACROS):
        if line.startswith("\\newcommand{\\%s}" % name):
            return line.strip()
    return "(not found)"


def check_config(cell):
    got = {k: cell[k] for k in PUBLISHED_TASK}
    diff = {k: (PUBLISHED_TASK[k], got[k]) for k in PUBLISHED_TASK
            if PUBLISHED_TASK[k] != got[k]}
    if diff:
        raise SystemExit("resolved config differs from the published task: %r" % diff)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(L.OUT, "pilot.json"))
    a = ap.parse_args()

    L.set_threads(a.threads)
    tgt = published_target(a.seed)
    cell = L.build_cell(campus=9, u=100, beta=1.0, rho=0.25, seed=a.seed)
    cfg = check_config(cell)
    assert cell["eval_ids"] == tgt["inst_ids"], (
        "held-out instance set differs from the published run:\n mine=%r\n pub =%r"
        % (cell["eval_ids"], tgt["inst_ids"]))
    print("config diff vs published task: NONE (%d fields checked)" % len(cfg))
    print("held-out instances match the published run: %d" % len(tgt["inst_ids"]))

    out = {"run": "W3 R0 pipeline-fidelity pilot", "seed": a.seed,
           "torch_threads": a.threads, "torch": torch.__version__,
           "numpy": np.__version__, "config": cfg, "files": cell["files"],
           "eval_ids": cell["eval_ids"],
           "published": {"macro_line": macro_value(),
                         "MzeroGain_pct_allseeds": tgt["pct_allseeds"],
                         "n_seeds": tgt["n_seeds"],
                         "pct_seed%d" % a.seed: tgt["pct_seed"]},
           "rungs": {}}

    cfg_ref = None
    for variant in ("mse_published", "mse_reexpr"):
        t0 = time.perf_counter()
        print("training %s (torch threads=%d) ..." % (variant, a.threads), flush=True)
        rc = L.resolved_config(cell, variant)
        if cfg_ref is None:
            cfg_ref = rc
        else:
            d = L.config_diff(cfg_ref, rc)
            if d:
                raise SystemExit("config drift between rungs: %r" % d)
        tr = L.train_variant(cell, variant, verbose=False)
        dep = L.deployed_twt(tr["model"], cell["eval"], cell["overlay"],
                             channel=cell["channel"], seed=a.seed)
        rule = np.asarray(dep["rule"]); aug = np.asarray(dep["aug"])
        pct = 100.0 * (rule.mean() - aug.mean()) / rule.mean()
        out["rungs"][variant] = {
            "pct_below_rule_seed%d" % a.seed: pct,
            "rule_per_instance": rule.tolist(), "aug_per_instance": aug.tolist(),
            "max_abs_dTWT_vs_published_m0": float(np.abs(aug - tgt["m0_alone"]).max()),
            "max_abs_dTWT_vs_published_rule": float(np.abs(rule - tgt["rule"]).max()),
            "d_pct_points_vs_published": pct - tgt["pct_seed"],
            "n_params_estimator": tr["n_params_estimator"],
            "n_params_total": tr["n_params_total"],
            "recovery_final_iter": ({k: tr["per_iter"][-1][k] for k in
                                     ("pearson_r", "sign_acc_nonzero",
                                      "exact_class_acc", "override_rate")}
                                    if tr["per_iter"] else {}),
            "elapsed_s_upper_bound": time.perf_counter() - t0,
        }
        print("  %-14s %.6f %% below rule (published seed %d: %.6f %%)"
              % (variant, pct, a.seed, tgt["pct_seed"]), flush=True)

    A = np.asarray(out["rungs"]["mse_published"]["aug_per_instance"])
    B = np.asarray(out["rungs"]["mse_reexpr"]["aug_per_instance"])
    out["reexpression_bit_exact"] = {
        "max_abs_dTWT": float(np.abs(A - B).max()),
        "exact": bool(np.array_equal(A, B))}

    L.write_json(a.out, out)

    p = out["rungs"]["mse_published"]
    print()
    print("published (macro)      : %s" % out["published"]["macro_line"])
    print("published 10-seed mean : %.4f %%" % tgt["pct_allseeds"])
    print("published seed %-4d    : %.4f %%" % (a.seed, tgt["pct_seed"]))
    print("mine     seed %-4d     : %.4f %%" % (a.seed, p["pct_below_rule_seed%d" % a.seed]))
    print("difference             : %+.6f pct points" % p["d_pct_points_vs_published"])
    print("max |dTWT*| m0_alone   : %.10g" % p["max_abs_dTWT_vs_published_m0"])
    print("max |dTWT*| rule       : %.10g" % p["max_abs_dTWT_vs_published_rule"])
    print("re-expressed loop bit-exact vs shipped: %s (max |dTWT*| = %.10g)"
          % (out["reexpression_bit_exact"]["exact"],
             out["reexpression_bit_exact"]["max_abs_dTWT"]))
    ok = (p["max_abs_dTWT_vs_published_m0"] < 1e-6
          and p["max_abs_dTWT_vs_published_rule"] < 1e-6
          and out["reexpression_bit_exact"]["max_abs_dTWT"] < 1e-6)
    print("FIDELITY: %s" % ("PASS" if ok else "FAIL -- do not proceed"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
