#!/usr/bin/env python
"""P3 full-roster frozen-decider evaluator (Paper Y3).

The REUSABLE Tier-1 evaluation harness. For one contention gate cell and one
seed it scores every method in the ladder, in BOTH deployment modes where
applicable, on the TRUE objective TWT*(w*, d*) computed by the INDEPENDENT
validator path (``true_objective.score_true``); no policy or solver ever scores
itself. Every decider is FROZEN at test (no test-time updates).

Roster
------
ALONE   (no supervisor at test):
    RULE          ATC on recorded fields (the deployed rule).
    M0            augmented ATC (its estimator corrects BOTH the weight and the
                  deadline; trained on the RULE+SUP override log, no RL).
    PI-0          a frozen Y1-style PPO policy trained in-regime with the
                  supervisor OFF (no override learning); gate = 0.
    M1-FROZEN     the DAgger-trained latent-shift policy; gate as trained.
    ORACLE-GREEDY myopic ATC on the true (w*, d*) at every decision (full-info
                  skyline; NOT a certified optimum, so it can invert -- flagged).

IN-LOOP (supervisor active at test, rho/eps/theta as in the cell):
    RULE+SUP, M0+SUP, PI-0+SUP, M1+SUP.

Held-out evaluation. The M0 estimator trains on ``files[:n_train]``; the M1 and
PI-0 policies train on ``files[:m1_pool]`` (default 20); the harness evaluates on
``files[n_train+n_probe : n_train+n_probe+n_eval]`` (default files[20:30]), which
is DISJOINT from every training pool.

Rule-family deciders (RULE, M0, ORACLE, and their +SUP) run through
``env.run_supervised`` and present the FULL trade queue to the supervisor.
Policy deciders (PI-0, M1, and their +SUP) run through the RL ``reset/step`` path
and present the K=64-candidate set the policy was trained on. This asymmetry is
inherent to how a rule vs a Y1 policy interfaces with the benchmark (the policy
is capped at K=64 by the Y1 observation design); each method is evaluated exactly
as it is defined/trained. The supervisor budget (rho/eps/theta) is identical for
both, and the realized review fractions are reported for parity.

Run
---
PYTHONPATH=src python scripts/y3_p3_eval.py \
    --campus 9 --u 100 --regime storm2 --beta 1.0 --rho 0.25 --eps 0.0 \
    --channel full_class_shift --seed 301 \
    --m1-ckpt train_log/y3_p15/m1_full/final.pt \
    --pi0-ckpt train_log/y3_p3/pi0_full/final.pt \
    --m1-metrics train_log/y3_p15/m1_full/metrics.csv \
    --out results/y3_p3 --tag pilot_c9u100_s301
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import csv
import glob
import json
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch                                                    # noqa: E402

from fmwos.env import DispatchEnv                                # noqa: E402
from fmwos.hitl import deciders as dec                           # noqa: E402
from fmwos.hitl import overlay as ov                             # noqa: E402
from fmwos.hitl.supervisor import Supervisor                     # noqa: E402
from fmwos.hitl import augmented_rule as AR                      # noqa: E402
from fmwos.hitl import true_objective as TO                      # noqa: E402
from fmwos.hitl.latent_head import (                             # noqa: E402
    LatentDispatchPolicy, latfeat_for_candidates)

try:
    from scipy.stats import wilcoxon as _wilcoxon
    _HAVE_SCIPY = True
except Exception:                                                # pragma: no cover
    _HAVE_SCIPY = False

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_TOL = 1e-9

# Deployment-mode ladder. Key -> (label, mode).
ALONE_KEYS = ["rule", "m0_alone", "pi0_alone", "m1_alone", "oracle"]
INLOOP_KEYS = ["rule_sup", "m0_sup", "pi0_sup", "m1_sup"]


# --------------------------------------------------------------------------- #
# Instance pool                                                               #
# --------------------------------------------------------------------------- #
def cell_files(campus, u, regime, w="w80"):
    cdir = "c%02d" % campus
    return sorted(glob.glob(os.path.join(
        _INST, cdir, regime, w, "%s_%s_%s_u%d_*.json" % (cdir, regime, w, u))))


def _load(p):
    with open(p) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Policy rollouts (RL path; FROZEN, greedy, no test-time updates)             #
# --------------------------------------------------------------------------- #
def rollout_policy_alone(policy, inst, device="cpu"):
    """Greedy frozen policy ALONE (no supervisor). Returns a schedule dict."""
    policy.eval()
    env = DispatchEnv(inst)
    obs = env.reset()
    fc = {}
    done = env._done
    while not done:
        lat = latfeat_for_candidates(env._candidates, fc)
        a, _lp, _v, _m = policy.act_with_margin(obs, latfeat=lat, greedy=True,
                                                device=device)
        obs, _r, done, _i = env.step(a)
    return env.to_schedule("policy_alone")


def rollout_policy_sup(policy, inst, overlay, applied, cell, seed, device="cpu"):
    """Greedy frozen policy WITH the in-loop supervisor (mirrors the M1 training
    rollout: policy proposes, supervisor may override, env executes the override).
    Returns (schedule, supervisor summary)."""
    policy.eval()
    sup = Supervisor(overlay, inst, rho=cell["rho"], epsilon=cell["eps"],
                     theta=cell["theta"], mechanism=cell["mechanism"],
                     seed=seed, applied=applied)
    env = DispatchEnv(inst)
    obs = env.reset()
    fc = {}
    done = env._done
    while not done:
        cands = env._candidates
        now = env._cur_now
        lat = latfeat_for_candidates(cands, fc)
        a_pi, _lp, _v, margin = policy.act_with_margin(obs, latfeat=lat,
                                                       greedy=True, device=device)
        decider_pick = cands[a_pi]
        executed_pick, _entry = sup.review(decider_pick, cands, now, margin)
        exec_action = cands.index(executed_pick)
        obs, _r, done, _i = env.step(exec_action)
    return env.to_schedule("policy_sup"), sup.summary()


# --------------------------------------------------------------------------- #
# M0 estimator (trained in-harness, deterministic; RULE+SUP override log only) #
# --------------------------------------------------------------------------- #
def train_m0_estimator(train, probe, overlay, cell, seed, outer_iters=8,
                       device="cpu", verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    res = AR.run_m0(train, probe, overlay,
                    beta_rho_eps=(cell["beta"], cell["rho"], cell["eps"]),
                    outer_iters=outer_iters, mechanism=cell["mechanism"],
                    theta=cell["theta"], seed=seed, device=device, verbose=verbose)
    return res["estimator"], res["per_iter"]


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
def paired_wilcoxon(a, b):
    """Two-sided paired Wilcoxon signed-rank p on a-b. a,b are per-instance
    vectors (a = decider under test, b = comparator). Returns p (float)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0.0):
        return 1.0
    if not _HAVE_SCIPY:
        return float("nan")
    try:
        # zero_method='pratt' keeps zero-diff pairs in the ranking (conservative);
        # default mode handles small n exactly.
        return float(_wilcoxon(a, b, zero_method="pratt").pvalue)
    except Exception:
        return float("nan")


def win_tie_loss(a, b, tol=_TOL):
    """W/T/L of a vs b, per instance: a WIN = a is LOWER (better on TWT*)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = int(np.sum(a < b - tol))
    l = int(np.sum(a > b + tol))
    t = int(len(a) - w - l)
    return {"W": w, "T": t, "L": l}


def contrast(name, test_key, comp_key, per):
    """A paired contrast: test_key vs comp_key on per-instance TWT*."""
    a = per[test_key]
    b = per[comp_key]
    am, bm = float(np.mean(a)), float(np.mean(b))
    return {
        "name": name, "test": test_key, "comparator": comp_key,
        "test_mean": am, "comparator_mean": bm,
        "delta_mean": am - bm,
        "pct_vs_comparator": 100.0 * (bm - am) / bm if abs(bm) > 1e-12 else 0.0,
        "wtl": win_tie_loss(a, b),
        "wilcoxon_p": paired_wilcoxon(a, b),
    }


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
# --------------------------------------------------------------------------- #
def evaluate(args):
    cell = {"beta": args.beta, "rho": args.rho, "eps": args.eps,
            "theta": args.theta, "mechanism": args.mechanism,
            "family": args.family, "master_seed": args.master_seed,
            "channel": args.channel, "regime": args.regime,
            "campus": args.campus, "u": args.u}

    overlay = ov.Overlay(ov.OverlayParams(
        beta=args.beta, family=args.family, master_seed=args.master_seed,
        channel=args.channel))
    assert overlay.params.channel == args.channel, "overlay channel mismatch"

    files = cell_files(args.campus, args.u, args.regime)
    need = max(args.n_train + args.n_probe + args.n_eval, args.m1_pool + args.n_eval)
    assert len(files) >= need, "only %d instances at c%d %s u%d (need %d)" % (
        len(files), args.campus, args.regime, args.u, need)

    train = [_load(p) for p in files[:args.n_train]]
    probe = [_load(p) for p in files[args.n_train:args.n_train + args.n_probe]]
    eval_start = args.n_train + args.n_probe
    eval_files = files[eval_start:eval_start + args.n_eval]
    eval_insts = [_load(p) for p in eval_files]

    # Guard: the eval split must be disjoint from every training pool.
    train_pool = set(files[:max(args.n_train, args.m1_pool)])
    assert not (set(eval_files) & train_pool), "eval split overlaps a training pool"

    print("[y3_p3] cell c%d %s u%d beta=%.2f rho=%.2f eps=%.2f theta=%.2f %s "
          "channel=%s seed=%d" % (args.campus, args.regime, args.u, args.beta,
          args.rho, args.eps, args.theta, args.mechanism, args.channel, args.seed))
    print("[y3_p3] pools: M0-train=%d probe=%d | policy-train-pool=%d | eval=%d "
          "(files[%d:%d]); n_wos~%d" % (len(train), len(probe), args.m1_pool,
          len(eval_insts), eval_start, eval_start + args.n_eval,
          len(eval_insts[0]["work_orders"])))

    # ---- M0 estimator (deterministic, RULE+SUP override log only) ---------- #
    t0 = time.perf_counter()
    estimator, m0_per_iter = train_m0_estimator(train, probe, overlay, cell,
                                                args.seed, outer_iters=args.m0_iters,
                                                device="cpu", verbose=True)
    print("[y3_p3] M0 estimator trained in %.1fs" % (time.perf_counter() - t0))

    # ---- Load frozen policies --------------------------------------------- #
    m1 = None
    if args.m1_ckpt and os.path.exists(args.m1_ckpt):
        m1 = LatentDispatchPolicy.load(args.m1_ckpt).to("cpu")
        m1.eval()
        print("[y3_p3] M1-FROZEN loaded: %s (gate=%.3f, correction_mode=%s)"
              % (args.m1_ckpt, float(m1.gate), m1.correction_mode))
    else:
        print("[y3_p3] WARNING: M1 checkpoint missing (%s); skipping M1 rows"
              % args.m1_ckpt)

    pi0 = None
    if args.pi0_ckpt and os.path.exists(args.pi0_ckpt):
        pi0 = LatentDispatchPolicy.load(args.pi0_ckpt).to("cpu")
        pi0.eval()
        assert float(pi0.gate) == 0.0, "PI-0 must have gate=0 (no override head)"
        print("[y3_p3] PI-0 loaded: %s (gate=%.3f)" % (args.pi0_ckpt, float(pi0.gate)))
    else:
        print("[y3_p3] NOTE: PI-0 checkpoint missing (%s); PI-0 rows skipped "
              "(re-run with --pi0-ckpt after training)" % args.pi0_ckpt)

    # ---- Per-instance TWT* for every decider ------------------------------ #
    per = {k: [] for k in ALONE_KEYS + INLOOP_KEYS}
    inst_ids = []
    sup_stats = {k: [] for k in ["rule_sup", "m0_sup", "pi0_sup", "m1_sup"]}

    for inst in eval_insts:
        applied = overlay.apply(inst)
        iid = inst["meta"]["id"]
        inst_ids.append(iid)

        def sc(sched):
            return TO.score_true(inst, sched, overlay, applied)["TWT_true"]

        # ---- ALONE ------------------------------------------------------- #
        per["rule"].append(sc(dec.run_rule(DispatchEnv(inst), "atc", seed=args.seed)))

        m0d = AR.augmented_atc_decider(estimator, inst, channel=args.channel)
        m0_sched, _ = DispatchEnv(inst).run_supervised(m0d, supervisor=None,
                                                       method="m0", seed=args.seed)
        per["m0_alone"].append(sc(m0_sched))

        osup = Supervisor(overlay, inst, rho=0.0, applied=applied)
        per["oracle"].append(sc(dec.run_oracle_greedy(DispatchEnv(inst), osup,
                                                       seed=args.seed)))

        per["pi0_alone"].append(sc(rollout_policy_alone(pi0, inst)) if pi0 else np.nan)
        per["m1_alone"].append(sc(rollout_policy_alone(m1, inst)) if m1 else np.nan)

        # ---- IN-LOOP (supervisor active) --------------------------------- #
        rsup = Supervisor(overlay, inst, rho=args.rho, epsilon=args.eps,
                          theta=args.theta, mechanism=args.mechanism,
                          seed=args.seed, applied=applied)
        rsched, _log = dec.run_rule_sup(DispatchEnv(inst), "atc", rsup, seed=args.seed)
        per["rule_sup"].append(sc(rsched))
        sup_stats["rule_sup"].append(rsup.summary())

        m0d2 = AR.augmented_atc_decider(estimator, inst, channel=args.channel)
        m0sup = Supervisor(overlay, inst, rho=args.rho, epsilon=args.eps,
                           theta=args.theta, mechanism=args.mechanism,
                           seed=args.seed, applied=applied)
        m0s_sched, _ = DispatchEnv(inst).run_supervised(m0d2, supervisor=m0sup,
                                                        method="m0_sup", seed=args.seed)
        per["m0_sup"].append(sc(m0s_sched))
        sup_stats["m0_sup"].append(m0sup.summary())

        if pi0:
            ps, pstat = rollout_policy_sup(pi0, inst, overlay, applied, cell, args.seed)
            per["pi0_sup"].append(sc(ps)); sup_stats["pi0_sup"].append(pstat)
        else:
            per["pi0_sup"].append(np.nan)

        if m1:
            ms, mstat = rollout_policy_sup(m1, inst, overlay, applied, cell, args.seed)
            per["m1_sup"].append(sc(ms)); sup_stats["m1_sup"].append(mstat)
        else:
            per["m1_sup"].append(np.nan)

    per = {k: np.asarray(v, float) for k, v in per.items()}
    n = len(inst_ids)

    def mean(k):
        return float(np.nanmean(per[k])) if np.any(~np.isnan(per[k])) else float("nan")

    rule_mean = mean("rule")

    def pct_below_rule(k):
        m = mean(k)
        return 100.0 * (rule_mean - m) / rule_mean if rule_mean > 1e-12 else float("nan")

    have = {"m1": m1 is not None, "pi0": pi0 is not None}

    # ---- Ladder table ------------------------------------------------------ #
    ladder = {}
    for k in ALONE_KEYS + INLOOP_KEYS:
        m = mean(k)
        if np.isnan(m):
            continue
        wtl = win_tie_loss(per[k], per["rule"])
        ladder[k] = {"twt_star": m, "pct_below_rule": pct_below_rule(k),
                     "wtl_vs_rule": wtl}

    # ---- Headline contrasts ------------------------------------------------ #
    contrasts = {}
    contrasts["H1_m1sup_vs_rulesup"] = contrast(
        "H1 equal-budget dominance (M1+SUP vs RULE+SUP)", "m1_sup", "rule_sup", per) if have["m1"] else None
    contrasts["H2_m1alone_vs_rule"] = contrast(
        "H2 internalization (M1-FROZEN alone vs RULE alone)", "m1_alone", "rule", per) if have["m1"] else None
    contrasts["M1_vs_M0_alone"] = contrast(
        "M1 alone vs M0 alone", "m1_alone", "m0_alone", per) if have["m1"] else None
    contrasts["M1_vs_M0_sup"] = contrast(
        "M1+SUP vs M0+SUP", "m1_sup", "m0_sup", per) if have["m1"] else None
    contrasts["PI0sup_vs_M1sup"] = contrast(
        "PI-0+SUP vs M1+SUP (learning-from-overrides isolation)", "m1_sup", "pi0_sup", per) if (have["m1"] and have["pi0"]) else None
    # supporting contrasts
    contrasts["M0_vs_RULE_alone"] = contrast("M0 alone vs RULE", "m0_alone", "rule", per)
    contrasts["RULEsup_vs_RULE"] = contrast("RULE+SUP vs RULE", "rule_sup", "rule", per)
    contrasts["M1sup_vs_M1alone"] = contrast(
        "M1+SUP vs M1 alone", "m1_sup", "m1_alone", per) if have["m1"] else None

    # ---- H3: falling burden ------------------------------------------------ #
    m1_over_traj = None
    if args.m1_metrics and os.path.exists(args.m1_metrics):
        with open(args.m1_metrics) as fh:
            rows = list(csv.DictReader(fh))
        m1_over_traj = [{"iter": int(r["iter"]),
                         "override_rate": float(r["override_rate"]),
                         "true_twt": float(r["true_twt"])} for r in rows]
    # flat RULE+SUP override rate over the eval instances (a fixed rule never learns)
    rule_sup_orr = [s["override_rate_of_reviews"] for s in sup_stats["rule_sup"]]
    rule_sup_revf = [s["reviewed_fraction"] for s in sup_stats["rule_sup"]]
    h3 = {
        "m1_train_override_rate_trajectory": m1_over_traj,
        "rule_sup_override_rate_flat_mean": float(np.mean(rule_sup_orr)),
        "rule_sup_override_rate_flat_per_inst": [float(x) for x in rule_sup_orr],
        "rule_sup_review_fraction_mean": float(np.mean(rule_sup_revf)),
    }

    # ---- Supervisor budget parity (review fractions, override rates) ------- #
    def sup_agg(key):
        if not sup_stats.get(key):
            return None
        rf = [s["reviewed_fraction"] for s in sup_stats[key]]
        orr = [s["override_rate_of_reviews"] for s in sup_stats[key]]
        return {"review_fraction_mean": float(np.mean(rf)),
                "override_rate_mean": float(np.mean(orr))}
    budget = {k: sup_agg(k) for k in sup_stats}

    # ---- Write per-instance CSV ------------------------------------------- #
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "%s.csv" % args.tag)
    cols = ["instance_id"] + ALONE_KEYS + INLOOP_KEYS
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, iid in enumerate(inst_ids):
            row = [iid] + ["%.6f" % per[k][i] if not np.isnan(per[k][i]) else ""
                           for k in ALONE_KEYS + INLOOP_KEYS]
            w.writerow(row)
    print("[y3_p3] wrote %s" % csv_path)

    # ---- Write summary JSON ----------------------------------------------- #
    summary = {
        "cell": cell, "seed": args.seed, "n_eval": n,
        "eval_files": [os.path.basename(f) for f in eval_files],
        "n_wos_eval0": len(eval_insts[0]["work_orders"]),
        "m1_ckpt": args.m1_ckpt if have["m1"] else None,
        "m1_gate": float(m1.gate) if m1 else None,
        "pi0_ckpt": args.pi0_ckpt if have["pi0"] else None,
        "ladder": ladder,
        "contrasts": {k: v for k, v in contrasts.items() if v is not None},
        "H3": h3,
        "supervisor_budget": budget,
        "m0_per_iter": m0_per_iter,
        "have": have,
    }
    json_path = os.path.join(args.out, "%s_summary.json" % args.tag)
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("[y3_p3] wrote %s" % json_path)

    # ---- Console ladder ---------------------------------------------------- #
    _print_ladder(ladder, contrasts, h3, budget, rule_mean, n, have)
    return summary


def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "  n/a"
    return "%.4f" % p


def _print_ladder(ladder, contrasts, h3, budget, rule_mean, n, have):
    label = {"rule": "RULE (ATC) alone", "m0_alone": "M0 alone",
             "pi0_alone": "PI-0 alone", "m1_alone": "M1-FROZEN alone",
             "oracle": "ORACLE-GREEDY", "rule_sup": "RULE+SUP",
             "m0_sup": "M0+SUP", "pi0_sup": "PI-0+SUP", "m1_sup": "M1+SUP"}
    print("\n===== LADDER  (TWT*(w*,d*), n=%d held-out; W/T/L vs RULE) =====" % n)
    print("%-18s %12s %14s %12s" % ("decider", "TWT*", "%below RULE", "W/T/L vs RULE"))
    for k in ALONE_KEYS + INLOOP_KEYS:
        if k not in ladder:
            continue
        e = ladder[k]
        wtl = e["wtl_vs_rule"]
        print("%-18s %12.1f %13.1f%% %4d/%d/%d"
              % (label[k], e["twt_star"], e["pct_below_rule"],
                 wtl["W"], wtl["T"], wtl["L"]))
    print("\n----- headline contrasts (paired, n=%d) -----" % n)
    for key in ["H1_m1sup_vs_rulesup", "H2_m1alone_vs_rule", "M1_vs_M0_alone",
                "M1_vs_M0_sup", "PI0sup_vs_M1sup", "M0_vs_RULE_alone",
                "RULEsup_vs_RULE", "M1sup_vs_M1alone"]:
        c = contrasts.get(key)
        if c is None:
            continue
        wtl = c["wtl"]
        print("  %-42s: %8.1f vs %8.1f  (%+.1f%%)  W/T/L %d/%d/%d  p=%s"
              % (c["name"][:42], c["test_mean"], c["comparator_mean"],
                 c["pct_vs_comparator"], wtl["W"], wtl["T"], wtl["L"],
                 _fmt_p(c["wilcoxon_p"])))
    print("\n----- H3 falling burden -----")
    if h3["m1_train_override_rate_trajectory"]:
        traj = " ".join("%.3f" % r["override_rate"]
                        for r in h3["m1_train_override_rate_trajectory"])
        print("  M1 train override-rate/iter: %s" % traj)
    print("  RULE+SUP override-rate (flat): %.3f  (review_frac %.3f)"
          % (h3["rule_sup_override_rate_flat_mean"], h3["rule_sup_review_fraction_mean"]))
    print("\n----- supervisor budget parity (review_frac / override_rate) -----")
    for k in ["rule_sup", "m0_sup", "pi0_sup", "m1_sup"]:
        b = budget.get(k)
        if b:
            print("  %-9s review_frac=%.3f  override_rate=%.3f"
                  % (k, b["review_fraction_mean"], b["override_rate_mean"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--campus", type=int, default=9)
    ap.add_argument("--u", type=int, default=100)
    ap.add_argument("--regime", type=str, default="storm2")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--mechanism", type=str, default="targeted",
                    choices=["targeted", "random"])
    ap.add_argument("--channel", type=str, default="full_class_shift",
                    choices=["full_class_shift", "weight_only"])
    ap.add_argument("--family", type=str, default="F-NL")
    ap.add_argument("--master-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--n-train", type=int, default=16, help="M0 estimator train slice")
    ap.add_argument("--n-probe", type=int, default=4, help="M0 hat_s probe slice")
    ap.add_argument("--n-eval", type=int, default=10, help="held-out eval slice")
    ap.add_argument("--m1-pool", type=int, default=20,
                    help="policy (M1/PI-0) training pool size (files[:m1_pool])")
    ap.add_argument("--m0-iters", type=int, default=8)
    ap.add_argument("--m1-ckpt", type=str, default="")
    ap.add_argument("--pi0-ckpt", type=str, default="")
    ap.add_argument("--m1-metrics", type=str, default="")
    ap.add_argument("--out", type=str, default=os.path.join(_ROOT, "results", "y3_p3"))
    ap.add_argument("--tag", type=str, default="pilot_c9u100_s301")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    torch.set_num_threads(int(args.threads))
    evaluate(args)


if __name__ == "__main__":
    main()
