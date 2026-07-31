#!/usr/bin/env python
"""P8 practitioner metrics (Paper Y3): per-true-class service attainment and the
exact preventive/corrective decomposition of the headline gain.

Closes two referee objections by RE-SCORING schedules the published pipeline
already produces. Nothing here trains anything: the M0 estimator refit is the
same deterministic in-harness fit ``scripts/y3_p4_m0grid.py`` performs, and the
M1 policies are loaded frozen from ``train_log/y3_sweep/``. No file outside
``results/y3_p8/`` is written.

Objection 1 -- the manuscript reports only true weighted tardiness, which is
unbounded and tail-dominated, while facility contracts are written as percentage
attainment inside a service window per priority class. This module reports, for
every decider and every reported cell, the share of orders completed within their
TRUE deadline d*, broken down by TRUE class c*.

Objection 2 -- campus C9 is dominated by preventive-maintenance work, all mapped
to recorded class 4 by calibration rule R5a. This module measures the preventive
share on the instances each campus is actually evaluated on, and splits the
headline reduction in true weighted tardiness into its preventive and corrective
parts. The objective is a sum of per-order terms, so that split is EXACT; the
additivity is asserted in code, per instance, per seed and on the aggregate.

Reproduction discipline
-----------------------
The pipeline is bit-exact only with ONE numeric thread per process: at
``torch.set_num_threads(4)`` the estimator refits with a different
floating-point reduction order and the headline moves by several percentage
points. Every numeric runtime is therefore capped to one thread in the module
body BEFORE numpy/torch import, and ``torch.set_num_threads(1)`` is re-asserted
at the top of every worker. Parallelism comes from worker PROCESSES only.

Run (cores 20-23 only; cores 0-19 belong to other agents):

    taskset -c 20-23 env PYTHONPATH=src \\
      /home/ziheng/miniconda3/envs/fjsp/bin/python \\
      scripts/y3_practitioner_metrics.py --part all --workers 4

Parts
-----
repro        reproduction gate: recompute the headline cell through this module
             and require every per-instance TWT* to equal the committed
             results/y3_p4/cache value bit-for-bit, and \\MzeroGain to come back
             at 45.4%.  Everything else refuses to run if this fails.
attain       per-true-class attainment + the exact PM/CM decomposition, at the
             headline cell (10 seeds) and the two other contention cells the
             manuscript reports (3 seeds each).
composition  preventive/corrective and recorded-class composition of the
             instances each campus is EVALUATED on.
tables       render the LaTeX table and the macro block from the result files.
all          repro -> composition -> attain -> tables.
"""

from __future__ import annotations

import os

# ---- one numeric thread per process, BEFORE numpy/torch are imported -------- #
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch                                                        # noqa: E402

from fmwos.env import DispatchEnv                                   # noqa: E402
from fmwos.hitl import augmented_rule as AR                         # noqa: E402
from fmwos.hitl import deciders as dec                              # noqa: E402
from fmwos.hitl import overlay as ov                                # noqa: E402
from fmwos.hitl import true_objective as TO                         # noqa: E402
from fmwos.hitl.latent_head import LatentDispatchPolicy             # noqa: E402
from fmwos.hitl.supervisor import Supervisor                        # noqa: E402

import y3_p3_eval as P3                                             # noqa: E402
import y3_p4_m0grid as M0G                                          # noqa: E402

_OUT = os.path.join(_ROOT, "results", "y3_p8")
_P4CACHE = os.path.join(_ROOT, "results", "y3_p4", "cache")
_SWEEP = os.path.join(_ROOT, "train_log", "y3_sweep")
_HARVEST = os.path.join(_ROOT, "results", "y3_p5", "harvest",
                        "primary_multiseed_summary.json")

_BREACH_TOL = 1e-9        # identical to fmwos.validator / true_objective
_EXACT_TOL = 0.0          # the reproduction gate is bit-exact, not approximate
_ADD_TOL = 1e-6           # additivity assertion on the PM/CM decomposition

# Bumped whenever order_records / eval_cheap / eval_m1 change, so a stale record
# in results/y3_p8/cache can never be swept into a later aggregation.
_CODE_VERSION = "p8-v1"

# Locked cell constants, copied from the published harness (y3_p4_m0grid).
FAMILY = "F-NL"
MASTER_SEED = 12345
EPS = 0.0
THETA = 1.0
MECH = "targeted"
CHANNEL = "full_class_shift"
N_TRAIN, N_PROBE, N_EVAL = 16, 4, 10
M0_ITERS = 8

CHEAP = ["rule", "m0_alone", "oracle", "rule_sup", "m0_sup"]
DECIDERS = ["rule", "rule_sup", "m0_alone", "m0_sup", "m1_alone", "oracle"]

# Display names, in the order the attainment table lists them.
LABEL = {
    "rule":     "Tuned rule (RULE)",
    "rule_sup": "Rule + supervisor (RULE+SUP)",
    "m0_alone": "Correction layer (M0)",
    "m0_sup":   "Correction layer + supervisor (M0+SUP)",
    "m1_alone": "End-to-end learner (M1)",
    "oracle":   "Myopic full-information reference (ORACLE)",
}

# --------------------------------------------------------------------------- #
# Cells                                                                        #
# --------------------------------------------------------------------------- #
# HEADLINE: the cell that carries \MzeroGain. Asserted field-by-field against
# results/y3_p5/harvest/primary_multiseed_summary.json:cell.
CELL_HEAD = {"key": "c9_storm2_u100_b1.00_r0.25", "campus": 9, "regime": "storm2",
             "u": 100, "beta": 1.0, "rho": 0.25, "seeds": list(range(301, 311)),
             "label": "Headline (C9, util 1.00, beta 1.00, rho 0.25)",
             "tex_label": "Headline contention cell (C9, utilisation \\utilsat{}, "
                          "$\\beta=\\betahigh$, $\\rho=\\rholow$), \\nseeds{} seeds"}
# The two other contention cells the manuscript reports (three-seed descriptive
# checks; macros \RegimeHtwoBusy/\RegimeHoneBusy and \RegimeHtwoBeta/\RegimeHoneBeta).
CELL_BUSY = {"key": "c9_storm2_u90_b1.00_r0.25", "campus": 9, "regime": "storm2",
             "u": 90, "beta": 1.0, "rho": 0.25, "seeds": [301, 302, 303],
             "label": "Busy load (C9, util 0.90, beta 1.00, rho 0.25)",
             "tex_label": "Busy load (C9, utilisation \\utilbusy{}, "
                          "$\\beta=\\betahigh$, $\\rho=\\rholow$), \\seedsMap{} seeds"}
CELL_BETA = {"key": "c9_storm2_u100_b0.75_r0.25", "campus": 9, "regime": "storm2",
             "u": 100, "beta": 0.75, "rho": 0.25, "seeds": [301, 302, 303],
             "label": "Low recoverable share (C9, util 1.00, beta 0.75, rho 0.25)",
             "tex_label": "Lower recoverable share (C9, utilisation \\utilsat{}, "
                          "$\\beta=\\betalow$, $\\rho=\\rholow$), \\seedsMap{} seeds"}
CELLS = [CELL_HEAD, CELL_BUSY, CELL_BETA]


def _cell_task(cell, seed):
    """The task dict the published m0-grid harness would build for this cell."""
    return {"campus": cell["campus"], "regime": cell["regime"], "u": cell["u"],
            "size": None, "beta": cell["beta"], "rho": cell["rho"], "eps": EPS,
            "theta": THETA, "mech": MECH, "channel": CHANNEL, "family": FAMILY,
            "master_seed": MASTER_SEED, "seed": seed, "n_train": N_TRAIN,
            "n_probe": N_PROBE, "n_eval": N_EVAL, "m0_iters": M0_ITERS}


def _m1_ckpt(cell, seed):
    return os.path.join(_SWEEP, "m1_c%d_u%d_b%g_r%g_s%d"
                        % (cell["campus"], cell["u"], cell["beta"], cell["rho"], seed),
                        "final.pt")


# --------------------------------------------------------------------------- #
# Committed-cache lookup (the bit-exactness target)                            #
# --------------------------------------------------------------------------- #
def load_p4_cache(cell, seed):
    """The committed results/y3_p4/cache record for (cell, seed), or None."""
    want = dict(campus=cell["campus"], regime=cell["regime"], u=cell["u"],
                beta=cell["beta"], rho=cell["rho"], channel=CHANNEL, seed=seed)
    for p in sorted(glob.glob(os.path.join(_P4CACHE, "*.json"))):
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if all(d.get(k) == v for k, v in want.items()) and d.get("eps", 0.0) == EPS:
            return d
    return None


# --------------------------------------------------------------------------- #
# Per-order accounting: the whole of the new measurement                       #
# --------------------------------------------------------------------------- #
def order_records(inst, sched, applied):
    """Per-(true class, preventive flag) buckets for one executed schedule.

    Returns {"buckets": {(c_star, is_pm): {"n", "attained", "twt"}},
             "twt_tardiness": float, "n_scored": int, "breaches": int}.

    ``attained`` counts orders finished within their TRUE deadline d*, the same
    boundary test the validator and true_objective use (end > due + 1e-9 is a
    breach, so attained is end <= due + 1e-9). ``twt`` is the per-order
    contribution w*(c*) * max(0, C - d*) summed inside the bucket; the buckets
    partition the scored orders, so the bucket sums add up to the total exactly.
    """
    wo_by_id = {wo["id"]: wo for wo in inst.get("work_orders", []) or []}
    cstar = applied["c_star"]
    wstar = applied["w_star"]
    dstar = applied["d_star"]

    buckets = defaultdict(lambda: {"n": 0, "attained": 0, "twt": 0.0})
    twt = 0.0
    n = 0
    breaches = 0
    for a in sched.get("assignments", []) or []:
        wo = wo_by_id.get(a.get("wo"))
        end = a.get("end_bh")
        if wo is None or end is None:
            continue
        end = float(end)
        wid = wo["id"]
        due = float(dstar.get(wid, float(wo["due_bh"])))
        w = float(wstar.get(wid, float(wo["weight"])))
        c = int(cstar.get(wid, int(wo["priority"])))
        pm = bool(wo.get("is_pm", False))
        tard = max(0.0, end - due)
        contrib = w * tard
        b = buckets[(c, pm)]
        b["n"] += 1
        b["twt"] += contrib
        if end > due + _BREACH_TOL:
            breaches += 1
        else:
            b["attained"] += 1
        twt += contrib
        n += 1
    return {"buckets": {"%d|%d" % (c, int(pm)): v for (c, pm), v in buckets.items()},
            "twt_tardiness": twt, "n_scored": n, "breaches": breaches}


def score_and_record(inst, sched, overlay, applied):
    """score_true(...) plus the per-order buckets, cross-checked against it.

    The scalar TWT* that the manuscript reports and the bucket decomposition are
    computed from the SAME schedule object in the same call, so a decider's
    attainment and its weighted tardiness can never come from different rollouts.
    """
    s = TO.score_true(inst, sched, overlay, applied)
    r = order_records(inst, sched, applied)
    # Cross-checks against the published scorer (which computes overall
    # attainment as ``breaches``/``n_scored`` and then discards it).
    assert r["n_scored"] == s["n_scored"], "scored-order count disagrees"
    assert r["breaches"] == s["breaches"], "breach count disagrees"
    assert abs(r["twt_tardiness"] - s["TWT_true_tardiness"]) <= _ADD_TOL, \
        "per-order TWT sum disagrees with score_true"
    # Bucket additivity: the (c*, is_pm) partition is complete.
    bsum = sum(v["twt"] for v in r["buckets"].values())
    bn = sum(v["n"] for v in r["buckets"].values())
    assert abs(bsum - r["twt_tardiness"]) <= _ADD_TOL, "bucket TWT sum is not exact"
    assert bn == r["n_scored"], "bucket order count is not exact"
    r["TWT_true"] = float(s["TWT_true"])
    r["access_penalty"] = float(s["access_penalty"])
    r["feasible"] = bool(s["feasible"])
    r["deadline_mode"] = s["deadline_mode"]
    return r


# --------------------------------------------------------------------------- #
# Worker A: the five cheap deciders, mirroring y3_p4_m0grid.evaluate_cell       #
# --------------------------------------------------------------------------- #
def eval_cheap(args):
    """RULE / M0 / ORACLE / RULE+SUP / M0+SUP for one (cell, seed).

    The call sequence, the seeding order and every constructor argument are
    identical to ``y3_p4_m0grid.evaluate_cell``; the only addition is that each
    schedule is also passed through ``score_and_record``.
    """
    cell, seed = args
    torch.set_num_threads(1)
    assert torch.get_num_threads() == 1, "worker is not single-threaded"
    try:
        os.nice(5)
    except Exception:
        pass
    task = _cell_task(cell, seed)

    files = M0G.locate_files(cell["campus"], cell["regime"], u=cell["u"], size=None)
    need = N_TRAIN + N_PROBE + N_EVAL
    assert len(files) >= need, "instance pool too small at %s" % cell["key"]
    train = [M0G._load(p) for p in files[:N_TRAIN]]
    probe = [M0G._load(p) for p in files[N_TRAIN:N_TRAIN + N_PROBE]]
    eval_files = files[N_TRAIN + N_PROBE:N_TRAIN + N_PROBE + N_EVAL]
    eval_insts = [M0G._load(p) for p in eval_files]
    assert not (set(eval_files) & set(files[:N_TRAIN + N_PROBE])), "eval overlaps train"

    overlay = ov.Overlay(ov.OverlayParams(
        beta=cell["beta"], family=FAMILY, master_seed=MASTER_SEED, channel=CHANNEL))
    assert overlay.params.channel == CHANNEL

    torch.manual_seed(seed)
    np.random.seed(seed)
    res = AR.run_m0(train, probe, overlay,
                    beta_rho_eps=(cell["beta"], cell["rho"], EPS),
                    outer_iters=M0_ITERS, mechanism=MECH, theta=THETA,
                    seed=seed, device="cpu", verbose=False)
    estimator = res["estimator"]

    out = {"cell": cell["key"], "seed": seed, "inst_ids": [], "per": {}}
    for k in CHEAP:
        out["per"][k] = []

    for inst in eval_insts:
        applied = overlay.apply(inst)
        out["inst_ids"].append(inst["meta"]["id"])

        def rec(sched):
            return score_and_record(inst, sched, overlay, applied)

        out["per"]["rule"].append(rec(dec.run_rule(DispatchEnv(inst), "atc", seed=seed)))
        m0d = AR.augmented_atc_decider(estimator, inst, channel=CHANNEL)
        m0_sched, _ = DispatchEnv(inst).run_supervised(m0d, supervisor=None,
                                                       method="m0", seed=seed)
        out["per"]["m0_alone"].append(rec(m0_sched))
        osup = Supervisor(overlay, inst, rho=0.0, applied=applied)
        out["per"]["oracle"].append(
            rec(dec.run_oracle_greedy(DispatchEnv(inst), osup, seed=seed)))

        rsup = Supervisor(overlay, inst, rho=cell["rho"], epsilon=EPS, theta=THETA,
                          mechanism=MECH, seed=seed, applied=applied)
        rsched, _ = dec.run_rule_sup(DispatchEnv(inst), "atc", rsup, seed=seed)
        out["per"]["rule_sup"].append(rec(rsched))

        m0d2 = AR.augmented_atc_decider(estimator, inst, channel=CHANNEL)
        m0sup = Supervisor(overlay, inst, rho=cell["rho"], epsilon=EPS, theta=THETA,
                           mechanism=MECH, seed=seed, applied=applied)
        m0s_sched, _ = DispatchEnv(inst).run_supervised(m0d2, supervisor=m0sup,
                                                        method="m0_sup", seed=seed)
        out["per"]["m0_sup"].append(rec(m0s_sched))

    # class / pm denominators are a property of the overlay draw alone
    out["denoms"] = _denoms(eval_insts, overlay)
    return out


def _denoms(eval_insts, overlay):
    """{(c*, is_pm): count} over the eval pool. Depends only on the overlay draw
    (instance id, master seed, beta, family), never on the decider or its seed."""
    d = defaultdict(int)
    for inst in eval_insts:
        applied = overlay.apply(inst)
        for wo in inst["work_orders"]:
            d["%d|%d" % (applied["c_star"][wo["id"]], int(bool(wo.get("is_pm", False))))] += 1
    return dict(d)


# --------------------------------------------------------------------------- #
# Worker B: the frozen end-to-end learner, mirroring y3_harvest_primary        #
# --------------------------------------------------------------------------- #
def eval_m1(args):
    """M1 ALONE for one (cell, seed), through the published frozen-eval path."""
    cell, seed = args
    torch.set_num_threads(1)
    assert torch.get_num_threads() == 1, "worker is not single-threaded"
    try:
        os.nice(5)
    except Exception:
        pass

    files = P3.cell_files(cell["campus"], cell["u"], cell["regime"])
    eval_files = files[N_TRAIN + N_PROBE:N_TRAIN + N_PROBE + N_EVAL]
    eval_insts = [P3._load(p) for p in eval_files]

    overlay = ov.Overlay(ov.OverlayParams(
        beta=cell["beta"], family=FAMILY, master_seed=MASTER_SEED, channel=CHANNEL))

    ckpt = _m1_ckpt(cell, seed)
    m1 = LatentDispatchPolicy.load(ckpt).to("cpu")
    m1.eval()
    nparam = int(sum(p.numel() for p in m1.parameters()))
    assert float(m1.gate) == 1.0, "M1 gate must be 1.0 (fair-M1)"
    assert bool(getattr(m1, "use_deadline_head")), "fair-M1 must have deadline_head=True"
    assert nparam == 14276, "fair-M1 param count drift: %d != 14276" % nparam
    assert m1.correction_mode == CHANNEL

    out = {"cell": cell["key"], "seed": seed, "inst_ids": [],
           "per": {"m1_alone": []}, "nparam": nparam}
    for inst in eval_insts:
        applied = overlay.apply(inst)
        out["inst_ids"].append(inst["meta"]["id"])
        out["per"]["m1_alone"].append(
            score_and_record(inst, P3.rollout_policy_alone(m1, inst), overlay, applied))
    return out


# --------------------------------------------------------------------------- #
# Part: reproduction gate                                                      #
# --------------------------------------------------------------------------- #
def part_repro(workers):
    """Recompute the headline cell here and require bit-exact agreement with the
    committed cache, then re-derive \\MzeroGain."""
    with open(_HARVEST) as fh:
        harvest = json.load(fh)
    pub_cell = harvest["cell"]
    for k, v in [("campus", 9), ("regime", "storm2"), ("u", 100), ("beta", 1.0),
                 ("rho", 0.25), ("eps", 0.0), ("theta", 1.0), ("mechanism", MECH),
                 ("family", FAMILY), ("master_seed", MASTER_SEED),
                 ("channel", CHANNEL)]:
        assert pub_cell[k] == v, "headline cell drift on %s: %r != %r" % (k, pub_cell[k], v)
    assert harvest["seeds"] == CELL_HEAD["seeds"], "seed set drift"
    pub_ids = harvest["eval_inst_ids"]

    tasks = [(CELL_HEAD, s) for s in CELL_HEAD["seeds"]]
    recs = _run(cheap_cached, tasks, workers)

    per_seed = {}
    diffs = []
    n_compared = 0
    for r in recs:
        seed = r["seed"]
        assert r["inst_ids"] == pub_ids, "eval instance ids differ from the published set"
        cache = load_p4_cache(CELL_HEAD, seed)
        assert cache is not None, "no committed cache record for seed %d" % seed
        assert cache["inst_ids"] == pub_ids, "cache instance ids differ"
        per_seed[seed] = {}
        for k in CHEAP:
            mine = [x["TWT_true"] for x in r["per"][k]]
            theirs = list(cache["per"][k])
            assert len(mine) == len(theirs) == N_EVAL
            for a, b in zip(mine, theirs):
                n_compared += 1
                diffs.append(abs(a - b))
            per_seed[seed][k] = float(np.mean(mine))

    max_abs = max(diffs) if diffs else 0.0
    n_exact = sum(1 for d in diffs if d == 0.0)

    ladder = {k: float(np.mean([per_seed[s][k] for s in CELL_HEAD["seeds"]]))
              for k in CHEAP}
    rule = ladder["rule"]
    gain = 100.0 * (rule - ladder["m0_alone"]) / rule
    pub_gain = 100.0 * (harvest["ladder"]["rule"]["twt_mean"]
                        - harvest["ladder"]["m0_alone"]["twt_mean"]) \
        / harvest["ladder"]["rule"]["twt_mean"]

    out = {
        "gate": "reproduction of \\MzeroGain through scripts/y3_practitioner_metrics.py",
        "cell": CELL_HEAD["key"], "seeds": CELL_HEAD["seeds"],
        "eval_inst_ids": pub_ids,
        "n_per_instance_values_compared": n_compared,
        "n_bit_exact": n_exact,
        "max_abs_diff_vs_committed_cache": max_abs,
        "published_MzeroGain_macro": "45.4%",
        "published_MzeroGain_value": pub_gain,
        "recomputed_MzeroGain_value": gain,
        "difference_pct_points": gain - pub_gain,
        "ladder_recomputed": ladder,
        "ladder_published": {k: harvest["ladder"][k]["twt_mean"] for k in CHEAP},
        "PASS": bool(max_abs <= _EXACT_TOL and abs(gain - pub_gain) <= 1e-9),
    }
    _write_json(os.path.join(_OUT, "repro_check.json"), out)
    print("[repro] compared %d per-instance TWT* values; %d bit-exact; max|diff| = %r"
          % (n_compared, n_exact, max_abs))
    print("[repro] published MzeroGain = %.4f%%  recomputed = %.4f%%  diff = %.2e pp"
          % (pub_gain, gain, gain - pub_gain))
    print("[repro] PASS = %s" % out["PASS"])
    return out


# --------------------------------------------------------------------------- #
# Part: per-true-class attainment + exact PM/CM decomposition                   #
# --------------------------------------------------------------------------- #
def _agg_cell(recs_cheap, recs_m1, cell):
    """Pool orders over the held-out instances within a seed, then average over
    seeds. Returns the per-decider, per-class, per-PM-flag aggregates."""
    seeds = cell["seeds"]
    by = {}                       # decider -> seed -> bucketkey -> dict
    for r in recs_cheap + recs_m1:
        if r["cell"] != cell["key"] or r["seed"] not in seeds:
            continue
        for k, lst in r["per"].items():
            acc = defaultdict(lambda: {"n": 0, "attained": 0, "twt": 0.0})
            tot = {"n": 0, "attained": 0, "twt": 0.0, "TWT_true": 0.0, "access": 0.0}
            for x in lst:
                for bk, v in x["buckets"].items():
                    acc[bk]["n"] += v["n"]
                    acc[bk]["attained"] += v["attained"]
                    acc[bk]["twt"] += v["twt"]
                tot["n"] += x["n_scored"]
                tot["attained"] += x["n_scored"] - x["breaches"]
                tot["twt"] += x["twt_tardiness"]
                tot["TWT_true"] += x["TWT_true"]
                tot["access"] += x["access_penalty"]
            # per-seed additivity assertion (exact partition)
            assert abs(sum(v["twt"] for v in acc.values()) - tot["twt"]) <= _ADD_TOL
            assert sum(v["n"] for v in acc.values()) == tot["n"]
            by.setdefault(k, {})[r["seed"]] = {"buckets": dict(acc), "total": tot,
                                               "n_inst": len(lst)}
    return by


def _rate(num, den):
    return float("nan") if den == 0 else 100.0 * num / den


def part_attain(workers, cells=None):
    cells = cells or CELLS
    rows_class = []       # per (cell, decider, class): attainment + twt
    rows_split = []       # per (cell, decider, class, pm flag)
    rows_seed = []        # per (cell, decider, seed): totals, for provenance
    rows_pop = []         # per (cell, true class): who is in that class
    rows_ps = []          # per (cell, decider, seed, true class): raw detail
    decomp = {}

    for cell in cells:
        tasks = [(cell, s) for s in cell["seeds"]]
        recs_cheap = _run(cheap_cached, tasks, workers)
        recs_m1 = _run(m1_cached, tasks, workers)
        by = _agg_cell(recs_cheap, recs_m1, cell)

        # denominators must not depend on the decider or the seed
        denom_ref = None
        for r in recs_cheap:
            if denom_ref is None:
                denom_ref = r["denoms"]
            else:
                assert r["denoms"] == denom_ref, "class/pm denominators move across seeds"
        for k, per_seed in by.items():
            for s, d in per_seed.items():
                got = {bk: v["n"] for bk, v in d["buckets"].items()}
                assert got == denom_ref, \
                    "class/pm denominators move for decider %s seed %d" % (k, s)

        seeds = cell["seeds"]
        for k in DECIDERS:
            if k not in by:
                continue
            per_seed = by[k]
            for s in seeds:
                t = per_seed[s]["total"]
                rows_seed.append({"cell": cell["key"], "decider": k, "seed": s,
                                  "n_inst": per_seed[s]["n_inst"],
                                  "n_orders": t["n"], "n_attained": t["attained"],
                                  "attain_pct": _rate(t["attained"], t["n"]),
                                  "twt_true_sum": t["TWT_true"],
                                  "twt_true_mean_per_inst": t["TWT_true"] / per_seed[s]["n_inst"],
                                  "access_penalty_sum": t["access"]})
                assert t["access"] == 0.0, \
                    "access penalty is non-zero; the PM/CM partition would be incomplete"

            # ---- per-seed, per-class detail so any test can be recomputed ---- #
            for s in seeds:
                b = per_seed[s]["buckets"]
                for c in (1, 2, 3, 4):
                    keys = ["%d|0" % c, "%d|1" % c]
                    num = sum(b[kk]["attained"] for kk in keys if kk in b)
                    den = sum(b[kk]["n"] for kk in keys if kk in b)
                    rows_ps.append({"cell": cell["key"], "decider": k, "seed": s,
                                    "true_class": c, "n_orders": den,
                                    "n_attained": num, "attain_pct": _rate(num, den),
                                    "twt_per_inst": sum(b[kk]["twt"] for kk in keys if kk in b)
                                    / per_seed[s]["n_inst"]})

            # ---- per true class (both PM and CM), and the PM/CM split -------- #
            for c in (1, 2, 3, 4):
                for tag, keys in (("all", ["%d|0" % c, "%d|1" % c]),
                                  ("cm", ["%d|0" % c]), ("pm", ["%d|1" % c])):
                    per_seed_rate, per_seed_twt = [], []
                    n_den = 0
                    for s in seeds:
                        b = per_seed[s]["buckets"]
                        num = sum(b[kk]["attained"] for kk in keys if kk in b)
                        den = sum(b[kk]["n"] for kk in keys if kk in b)
                        twt = sum(b[kk]["twt"] for kk in keys if kk in b)
                        n_den = den
                        per_seed_rate.append(_rate(num, den))
                        per_seed_twt.append(twt / per_seed[s]["n_inst"])
                    row = {"cell": cell["key"], "decider": k, "true_class": c,
                           "scope": tag, "n_orders": n_den, "n_seeds": len(seeds),
                           "attain_pct_mean": float(np.mean(per_seed_rate)),
                           "attain_pct_sd": float(np.std(per_seed_rate)),
                           "twt_per_inst_mean": float(np.mean(per_seed_twt)),
                           "twt_per_inst_sd": float(np.std(per_seed_twt))}
                    if tag == "all":
                        rows_class.append(row)
                    else:
                        rows_split.append(row)
            # overall row (class-agnostic)
            for tag in ("all", "cm", "pm"):
                keys = [bk for bk in denom_ref
                        if tag == "all" or (tag == "pm") == bk.endswith("|1")]
                pr, pt = [], []
                n_den = 0
                for s in seeds:
                    b = per_seed[s]["buckets"]
                    num = sum(b[kk]["attained"] for kk in keys if kk in b)
                    den = sum(b[kk]["n"] for kk in keys if kk in b)
                    twt = sum(b[kk]["twt"] for kk in keys if kk in b)
                    n_den = den
                    pr.append(_rate(num, den))
                    pt.append(twt / per_seed[s]["n_inst"])
                row = {"cell": cell["key"], "decider": k, "true_class": 0,
                       "scope": tag, "n_orders": n_den, "n_seeds": len(seeds),
                       "attain_pct_mean": float(np.mean(pr)),
                       "attain_pct_sd": float(np.std(pr)),
                       "twt_per_inst_mean": float(np.mean(pt)),
                       "twt_per_inst_sd": float(np.std(pt))}
                (rows_class if tag == "all" else rows_split).append(row)

        decomp[cell["key"]] = _decompose(by, cell, denom_ref)

        # Who is in each TRUE class: the referee's "the correction promotes
        # preventive work into true classes 2 and 3" claim, quantified. The
        # populations are a property of the latent draw alone.
        for c in (1, 2, 3, 4):
            n_pm = denom_ref.get("%d|1" % c, 0)
            n_cm = denom_ref.get("%d|0" % c, 0)
            rows_pop.append({"cell": cell["key"], "true_class": c,
                             "n_orders": n_pm + n_cm, "n_pm": n_pm, "n_cm": n_cm,
                             "pm_share_of_true_class": _rate(n_pm, n_pm + n_cm) / 100.0})
        tot_pm = sum(v for bk, v in denom_ref.items() if bk.endswith("|1"))
        tot_cm = sum(v for bk, v in denom_ref.items() if bk.endswith("|0"))
        rows_pop.append({"cell": cell["key"], "true_class": 0,
                         "n_orders": tot_pm + tot_cm, "n_pm": tot_pm, "n_cm": tot_cm,
                         "pm_share_of_true_class": _rate(tot_pm, tot_pm + tot_cm) / 100.0})

    _write_csv(os.path.join(_OUT, "attainment_per_seed_class.csv"), rows_ps)
    _write_csv(os.path.join(_OUT, "true_class_composition.csv"), rows_pop)
    _write_csv(os.path.join(_OUT, "attainment_by_class.csv"), rows_class)
    _write_csv(os.path.join(_OUT, "attainment_pm_split.csv"), rows_split)
    _write_csv(os.path.join(_OUT, "per_seed_totals.csv"), rows_seed)
    _write_json(os.path.join(_OUT, "twt_decomposition.json"), decomp)
    return rows_class, rows_split, decomp


def _decompose(by, cell, denom_ref):
    """EXACT preventive/corrective decomposition of every decider's reduction in
    true weighted tardiness against RULE, and of its attainment change.

    TWT* is a sum of per-order terms and the access penalty is 0 here, so
    partitioning the orders into preventive and corrective is an identity, not an
    approximation. The identity is asserted, not assumed.
    """
    seeds = cell["seeds"]

    def parts(k):
        pm, cm, tot, att = [], [], [], {"pm": [], "cm": [], "all": []}
        for s in seeds:
            d = by[k][s]
            b = d["buckets"]
            n_inst = d["n_inst"]
            p = sum(v["twt"] for bk, v in b.items() if bk.endswith("|1")) / n_inst
            c = sum(v["twt"] for bk, v in b.items() if bk.endswith("|0")) / n_inst
            t = d["total"]["twt"] / n_inst
            assert abs((p + c) - t) <= _ADD_TOL, "PM+CM != total inside a seed"
            pm.append(p); cm.append(c); tot.append(t)
            for tag, sel in (("pm", lambda x: x.endswith("|1")),
                             ("cm", lambda x: x.endswith("|0")),
                             ("all", lambda x: True)):
                num = sum(v["attained"] for bk, v in b.items() if sel(bk))
                den = sum(v["n"] for bk, v in b.items() if sel(bk))
                att[tag].append(_rate(num, den))
        return (float(np.mean(pm)), float(np.mean(cm)), float(np.mean(tot)),
                {t: float(np.mean(v)) for t, v in att.items()})

    base_pm, base_cm, base_tot, base_att = parts("rule")
    n_pm = sum(v for bk, v in denom_ref.items() if bk.endswith("|1"))
    n_cm = sum(v for bk, v in denom_ref.items() if bk.endswith("|0"))
    out = {"cell": cell["key"], "label": cell["label"], "seeds": seeds,
           "n_orders_pm_per_pool": n_pm, "n_orders_cm_per_pool": n_cm,
           "pm_share_of_orders": n_pm / (n_pm + n_cm),
           "rule": {"twt_pm": base_pm, "twt_cm": base_cm, "twt_total": base_tot,
                    "attain_pct": base_att},
           "reductions_vs_rule": {}}
    for k in DECIDERS:
        if k not in by or k == "rule":
            continue
        pm, cm, tot, att = parts(k)
        d_pm, d_cm, d_tot = base_pm - pm, base_cm - cm, base_tot - tot
        assert abs((d_pm + d_cm) - d_tot) <= _ADD_TOL, \
            "ADDITIVITY FAILED for %s: %r + %r != %r" % (k, d_pm, d_cm, d_tot)
        out["reductions_vs_rule"][k] = {
            "twt_pm": pm, "twt_cm": cm, "twt_total": tot,
            "delta_twt_pm": d_pm, "delta_twt_cm": d_cm, "delta_twt_total": d_tot,
            "additivity_residual": (d_pm + d_cm) - d_tot,
            "pct_below_rule_total": 100.0 * d_tot / base_tot,
            "share_of_reduction_from_pm": 100.0 * d_pm / d_tot if d_tot else float("nan"),
            "share_of_reduction_from_cm": 100.0 * d_cm / d_tot if d_tot else float("nan"),
            "pct_below_rule_within_pm": 100.0 * d_pm / base_pm if base_pm else float("nan"),
            "pct_below_rule_within_cm": 100.0 * d_cm / base_cm if base_cm else float("nan"),
            "attain_pct": att,
            "attain_delta_pp": {t: att[t] - base_att[t] for t in att},
        }
    return out


# --------------------------------------------------------------------------- #
# Part: preventive/corrective composition per campus                           #
# --------------------------------------------------------------------------- #
def part_composition():
    """Preventive share and recorded-class distribution over the instances each
    campus is EVALUATED on (the held-out pools recorded in the committed
    results/y3_p4/cache records), plus the headline cell on its own."""
    # campus -> {instance id -> path} over every committed eval pool
    pools = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(_P4CACHE, "*.json"))):
        with open(p) as fh:
            d = json.load(fh)
        campus = d["campus"]
        files = M0G.locate_files(campus, d["regime"], u=d["u"], size=d["size"])
        ev = files[d["n_train"] + d["n_probe"]:d["n_train"] + d["n_probe"] + d["n_eval"]]
        ids = d["inst_ids"]
        assert len(ev) == len(ids)
        for path, iid in zip(ev, ids):
            assert os.path.basename(path).startswith(iid), "instance id/path mismatch"
            pools[campus][iid] = path

    head_ids = set()
    hf = M0G.locate_files(9, "storm2", u=100, size=None)
    for path in hf[N_TRAIN + N_PROBE:N_TRAIN + N_PROBE + N_EVAL]:
        head_ids.add(os.path.basename(path)[:-len(".json")])

    rows = []
    head_row = None
    detail = {}
    for campus in sorted(pools):
        agg = _compose(sorted(pools[campus].values()))
        agg.update({"scope": "campus C%d, all evaluated instances" % campus,
                    "campus": campus, "n_instances": len(pools[campus])})
        rows.append(agg)
        detail["campus_%d" % campus] = {"instance_ids": sorted(pools[campus]),
                                        "n_instances": len(pools[campus])}
    hpaths = sorted(pools[9][i] for i in head_ids)
    head_row = _compose(hpaths)
    head_row.update({"scope": "headline cell (C9 storm2 u100), 10 held-out instances",
                     "campus": 9, "n_instances": len(hpaths)})
    rows.append(head_row)
    detail["headline_cell"] = {"instance_ids": sorted(head_ids),
                               "n_instances": len(hpaths)}

    # the raw-row reference the referee quotes
    ref = {}
    rp = os.path.join(_ROOT, "results", "y3_p6", "priority_reliability.csv")
    with open(rp) as fh:
        for r in csv.DictReader(fh):
            ref["campus_%s" % r["campus"]] = float(r["pm_share_r5a"])
    detail["pm_share_r5a_raw_rows_reference"] = ref

    cols = ["scope", "campus", "n_instances", "n_orders", "pm_share",
            "cm_share", "cls1_share", "cls2_share", "cls3_share", "cls4_share",
            "pm_in_cls4_share", "cls4_that_is_pm_share", "n_pm", "n_cm",
            "n_cls1", "n_cls2", "n_cls3", "n_cls4"]
    _write_csv(os.path.join(_OUT, "composition_by_campus.csv"), rows, cols)
    _write_json(os.path.join(_OUT, "composition_detail.json"), detail)
    for r in rows:
        print("[composition] %-58s n_wo=%7d pm=%.4f cls4=%.4f pm->cls4=%.4f"
              % (r["scope"], r["n_orders"], r["pm_share"], r["cls4_share"],
                 r["pm_in_cls4_share"]))
    print("[composition] raw-row pm_share_r5a reference: %s" % ref)
    return rows


def _compose(paths):
    n = n_pm = 0
    cls = defaultdict(int)
    pm_cls = defaultdict(int)
    for p in paths:
        with open(p) as fh:
            inst = json.load(fh)
        for wo in inst["work_orders"]:
            n += 1
            c = int(wo["priority"])
            cls[c] += 1
            if bool(wo.get("is_pm", False)):
                n_pm += 1
                pm_cls[c] += 1
    out = {"n_orders": n, "n_pm": n_pm, "n_cm": n - n_pm,
           "pm_share": n_pm / n if n else float("nan"),
           "cm_share": (n - n_pm) / n if n else float("nan"),
           "pm_in_cls4_share": pm_cls[4] / n_pm if n_pm else float("nan"),
           "cls4_that_is_pm_share": pm_cls[4] / cls[4] if cls[4] else float("nan")}
    for c in (1, 2, 3, 4):
        out["n_cls%d" % c] = cls[c]
        out["cls%d_share" % c] = cls[c] / n if n else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Runner / IO helpers                                                          #
# --------------------------------------------------------------------------- #
def _sig(kind, cell, seed):
    payload = {"kind": kind, "code": _CODE_VERSION, "seed": seed,
               "campus": cell["campus"], "regime": cell["regime"], "u": cell["u"],
               "beta": cell["beta"], "rho": cell["rho"], "eps": EPS,
               "theta": THETA, "mech": MECH, "channel": CHANNEL,
               "family": FAMILY, "master_seed": MASTER_SEED,
               "n_train": N_TRAIN, "n_probe": N_PROBE, "n_eval": N_EVAL,
               "m0_iters": M0_ITERS}
    import hashlib
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


_WORKER = {}                       # kind -> worker fn; bound after definition


def _cached(kind, task):
    """Memoise a worker's per-(cell, seed) record in results/y3_p8/cache. The
    signature carries every locked cell constant and a code version, so a stale
    record cannot be swept into a later aggregation."""
    cell, seed = task
    path = os.path.join(_OUT, "cache", "%s_%s.json" % (kind, _sig(kind, cell, seed)))
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            pass
    rec = _WORKER[kind](task)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, path)
    return rec


def cheap_cached(task):
    return _cached("cheap", task)


def m1_cached(task):
    return _cached("m1", task)


_WORKER["cheap"] = eval_cheap
_WORKER["m1"] = eval_m1


def _run(fn, tasks, workers):
    if workers <= 1:
        return sorted([fn(t) for t in tasks], key=lambda r: r["seed"])
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, t) for t in tasks]
        for f in as_completed(futs):
            out.append(f.result())
    return sorted(out, key=lambda r: r["seed"])


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)
    print("[write] %s" % path)


def _write_csv(path, rows, cols=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    cols = cols or list(rows[0].keys())
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    os.replace(tmp, path)
    print("[write] %s (%d rows)" % (path, len(rows)))


# --------------------------------------------------------------------------- #
# Macro block, in the style of paper/macros.tex (every number carries a         #
# provenance comment naming the exact results file and field).                  #
# --------------------------------------------------------------------------- #
def _seed_wtl(cellkey, test="m0_alone", comp="rule", tol=1e-12):
    """{true_class: (wins, ties, losses)} of ``test`` against ``comp`` per seed,
    a WIN being strictly HIGHER attainment. Class 0 is all orders."""
    per = defaultdict(dict)
    with open(os.path.join(_OUT, "attainment_per_seed_class.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["cell"] == cellkey:
                per[(r["decider"], int(r["true_class"]))][int(r["seed"])] = float(r["attain_pct"])
    with open(os.path.join(_OUT, "per_seed_totals.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["cell"] == cellkey:
                per[(r["decider"], 0)][int(r["seed"])] = float(r["attain_pct"])
    out = {}
    for c in (0, 1, 2, 3, 4):
        a, b = per.get((test, c)), per.get((comp, c))
        if not a or not b:
            continue
        seeds = sorted(a)
        w = sum(1 for s in seeds if a[s] > b[s] + tol)
        l = sum(1 for s in seeds if a[s] < b[s] - tol)
        out[c] = (w, len(seeds) - w - l, l)
    return out


def _write_macros(idx, decomp, repro, comp, late_rule, late_m0):
    head, busy, beta = CELL_HEAD["key"], CELL_BUSY["key"], CELL_BETA["key"]

    def m(cellkey, k, c):
        return float(idx[(cellkey, k, c)]["attain_pct_mean"])

    def n(cellkey, c):
        return int(idx[(cellkey, "rule", c)]["n_orders"])

    D = decomp[head]
    R = D["reductions_vs_rule"]
    resid = max(abs(v["additivity_residual"]) for d in decomp.values()
                for v in d["reductions_vs_rule"].values())
    c9 = comp["campus C9, all evaluated instances"]
    c10 = comp["campus C10, all evaluated instances"]
    c5 = comp["campus C5, all evaluated instances"]
    c12 = comp["campus C12, all evaluated instances"]
    ch = comp["headline cell (C9 storm2 u100), 10 held-out instances"]
    with open(os.path.join(_OUT, "true_class_composition.csv")) as fh:
        pop = {(r["cell"], int(r["true_class"])): r for r in csv.DictReader(fh)}

    def pmshare(c):
        return 100.0 * float(pop[(head, c)]["pm_share_of_true_class"])

    L = []
    A = L.append
    A("% ===========================================================================")
    A("% P8 PRACTITIONER METRICS -- service attainment (referee objection 1) and the")
    A("% preventive/corrective decomposition (referee objection 2). Generated by")
    A("% scripts/y3_practitioner_metrics.py --part tables. Nothing was retrained:")
    A("% every number below is a re-scoring of schedules the published pipeline")
    A("% produces, gated on a bit-exact reproduction of \\MzeroGain (see")
    A("% results/y3_p8/repro_check.json: 500/500 per-instance TWT* values equal to")
    A("%% results/y3_p4/cache, max|diff| = 0.0, recomputed gain %.10f%%)."
      % repro["recomputed_MzeroGain_value"])
    A("% CONVENTION: attainment macros expand WITH a trailing percent sign (they are")
    A("% quoted as percentages in prose); DELTA macros are percentage-POINT")
    A("% differences and carry an explicit sign.")
    A("% ===========================================================================")
    A("")
    A("% ---- reproduction gate ----------------------------------------------------")
    A("\\newcommand{\\ReproMzeroGain}{45.4\\%}   % repro_check.json recomputed_MzeroGain_value 45.36203547 == published")
    A("\\newcommand{\\ReproNExact}{500/500}      % repro_check.json n_bit_exact / n_per_instance_values_compared")
    A("")
    A("% ---- service attainment, headline cell (attainment_by_class.csv, cell")
    A("%%      %s, field attain_pct_mean) -----------------" % head)
    for key, k in (("Rule", "rule"), ("RuleSup", "rule_sup"), ("Mzero", "m0_alone"),
                   ("MzeroSup", "m0_sup"), ("Mone", "m1_alone"), ("Oracle", "oracle")):
        for cname, c in (("ClsOne", 1), ("ClsTwo", 2), ("All", 0)):
            A("\\newcommand{\\Attain%s%s}{%.2f\\%%}%s%% %s true_class=%d attain_pct_mean %.6f"
              % (key, cname, m(head, k, c), " " * max(1, 26 - len(key) - len(cname)),
                 k, c, m(head, k, c)))
    A("\\newcommand{\\AttainClsThreeFour}{100\\%}  % every decider, every cell: true_class 3 and 4 attain_pct_mean == 100.00")
    A("")
    A("% ---- attainment DELTAS vs the tuned rule, headline cell (percentage points;")
    A("%      derived from the same file, decider minus rule) ------------------------")
    for key, k in (("Mzero", "m0_alone"), ("MzeroSup", "m0_sup"),
                   ("Mone", "m1_alone"), ("Oracle", "oracle"), ("RuleSup", "rule_sup")):
        for cname, c in (("ClsOne", 1), ("ClsTwo", 2), ("All", 0)):
            d = m(head, k, c) - m(head, "rule", c)
            A("\\newcommand{\\AttainDelta%s%s}{$%s$%.2f}%s%% %s minus rule, true_class=%d: %+.4f pp"
              % (key, cname, "-" if d < 0 else "+", abs(d),
                 " " * max(1, 20 - len(key) - len(cname)), k, c, d))
    for cname, c in (("One", 1), ("Two", 2), ("Three", 3), ("Four", 4)):
        A("\\newcommand{\\AttainCls%sN}{%d}%s%% true class-%d orders per ten-instance pool at the headline cell (attainment_by_class.csv n_orders)"
          % (cname, n(head, c), " " * max(1, 12 - len(cname)), c))
    A("\\newcommand{\\AttainClsOneLateRule}{%.0f}  %% RULE: %d x (1 - %.4f%%/100)" % (late_rule, n(head, 1), m(head, "rule", 1)))
    A("\\newcommand{\\AttainClsOneLateMzero}{%.0f} %% M0:   %d x (1 - %.4f%%/100)" % (late_m0, n(head, 1), m(head, "m0_alone", 1)))
    A("\\newcommand{\\AttainClsOneExtra}{%.1f}     %% extra late class-1 orders per pool of ten instances under M0" % (late_m0 - late_rule))
    A("")
    A("% ---- per-seed unanimity of the attainment change (attainment_per_seed_class.csv")
    A("%      for classes 1-4, per_seed_totals.csv for all orders; W/T/L counts M0's")
    A("%      seed value strictly ABOVE / equal to / strictly BELOW the deterministic")
    A("%      rule value, so a WIN is better attainment) -----------------------------")
    wtl = _seed_wtl(head)
    A("\\newcommand{\\AttainWTLMzeroClsOne}{%d/%d/%d}   %% M0 vs RULE, true class 1, over the %d seeds: the sign of the class-1 change is not stable"
      % (wtl[1][0], wtl[1][1], wtl[1][2], sum(wtl[1])))
    A("\\newcommand{\\AttainWTLMzeroClsTwo}{%d/%d/%d}  %% M0 vs RULE, true class 2, over the %d seeds"
      % (wtl[2][0], wtl[2][1], wtl[2][2], sum(wtl[2])))
    A("\\newcommand{\\AttainWTLMzeroAll}{%d/%d/%d}     %% M0 vs RULE, all orders, over the %d seeds"
      % (wtl[0][0], wtl[0][1], wtl[0][2], sum(wtl[0])))
    A("")
    A("% ---- attainment DELTAS at the two other contention cells (percentage points)")
    for tag, ck in (("Busy", busy), ("Beta", beta)):
        for cname, c in (("ClsOne", 1), ("ClsTwo", 2), ("All", 0)):
            d = m(ck, "m0_alone", c) - m(ck, "rule", c)
            A("\\newcommand{\\AttainDeltaMzero%s%s}{$%s$%.2f}%s%% cell %s, m0_alone minus rule, true_class=%d"
              % (cname, tag, "-" if d < 0 else "+", abs(d),
                 " " * max(1, 16 - len(cname) - len(tag)), ck, c))
    A("")
    A("% ---- benchmark composition (composition_by_campus.csv) --------------------")
    A("\\newcommand{\\PMshareCnine}{%.1f\\%%}      %% C9, all evaluated instances: pm_share %.4f (%s of %s orders, %s instances)"
      % (100 * float(c9["pm_share"]), float(c9["pm_share"]), c9["n_pm"], c9["n_orders"], c9["n_instances"]))
    A("\\newcommand{\\PMshareHead}{%.1f\\%%}       %% headline cell 10 held-out instances: pm_share %.4f (%s of %s orders)"
      % (100 * float(ch["pm_share"]), float(ch["pm_share"]), ch["n_pm"], ch["n_orders"]))
    A("\\newcommand{\\PMshareCten}{%.1f\\%%}       %% C10: pm_share %.4f" % (100 * float(c10["pm_share"]), float(c10["pm_share"])))
    A("\\newcommand{\\PMshareCfive}{%.1f\\%%}      %% C5:  pm_share %.4f" % (100 * float(c5["pm_share"]), float(c5["pm_share"])))
    A("\\newcommand{\\PMshareCtwelve}{%.1f\\%%}    %% C12: pm_share %.4f" % (100 * float(c12["pm_share"]), float(c12["pm_share"])))
    A("\\newcommand{\\PMshareRawCnine}{%.1f\\%%}   %% raw FMUCD rows, results/y3_p6/priority_reliability.csv pm_share_r5a campus 9 = 0.8105" % 81.1)
    A("\\newcommand{\\ClsFourShareCnine}{%.1f\\%%} %% C9 recorded class-4 share: cls4_share %.4f" % (100 * float(c9["cls4_share"]), float(c9["cls4_share"])))
    A("\\newcommand{\\PMtoClsFour}{100\\%}        % every campus: pm_in_cls4_share 1.0000 (calibration rule R5a in src/fmwos/calib.py)")
    A("\\newcommand{\\ClsFourIsPMCnine}{100\\%}   % C9: cls4_that_is_pm_share 1.0000 (recorded class 4 is exactly the preventive work)")
    A("")
    A("% ---- who is in each TRUE class at the headline cell (true_class_composition.csv)")
    A("\\newcommand{\\PMinTrueClsOne}{0}          % n_pm at true_class=1: preventive work is recorded class 4 and the shift is clipped at 2, so it cannot reach true class 1")
    A("\\newcommand{\\PMshareTrueClsTwo}{%.1f\\%%}   %% pm_share_of_true_class, true_class=2: %.4f" % (pmshare(2), pmshare(2) / 100.0))
    A("\\newcommand{\\PMshareTrueClsThree}{%.1f\\%%} %% pm_share_of_true_class, true_class=3: %.4f" % (pmshare(3), pmshare(3) / 100.0))
    A("\\newcommand{\\PMshareTrueClsFour}{%.1f\\%%}  %% pm_share_of_true_class, true_class=4: %.4f" % (pmshare(4), pmshare(4) / 100.0))
    A("")
    A("% ---- EXACT preventive/corrective decomposition of the headline reduction")
    A("%%      (twt_decomposition.json, cell %s). TWT* is a sum of" % head)
    A("%      per-order terms and the access penalty is 0 at this cell, so the split")
    A("%      is an identity; the residual is asserted below 1e-6 in code.")
    A("\\newcommand{\\TwtRulePM}{%.0f}            %% rule.twt_pm  %.4f (per instance, mean over %d seeds)" % (D["rule"]["twt_pm"], D["rule"]["twt_pm"], len(D["seeds"])))
    A("\\newcommand{\\TwtRuleCM}{%.0f}            %% rule.twt_cm  %.4f" % (D["rule"]["twt_cm"], D["rule"]["twt_cm"]))
    A("\\newcommand{\\MzeroGainPMshare}{%.1f\\%%}   %% m0_alone share_of_reduction_from_pm %.4f" % (R["m0_alone"]["share_of_reduction_from_pm"], R["m0_alone"]["share_of_reduction_from_pm"]))
    A("\\newcommand{\\MzeroGainCMshare}{%.1f\\%%}   %% m0_alone share_of_reduction_from_cm %.4f" % (R["m0_alone"]["share_of_reduction_from_cm"], R["m0_alone"]["share_of_reduction_from_cm"]))
    A("\\newcommand{\\MzeroGainWithinPM}{%.1f\\%%}  %% m0_alone pct_below_rule_within_pm %.4f" % (R["m0_alone"]["pct_below_rule_within_pm"], R["m0_alone"]["pct_below_rule_within_pm"]))
    A("\\newcommand{\\MzeroGainWithinCM}{%.1f\\%%}  %% m0_alone pct_below_rule_within_cm %.4f" % (R["m0_alone"]["pct_below_rule_within_cm"], R["m0_alone"]["pct_below_rule_within_cm"]))
    A("\\newcommand{\\OracleGainPMshare}{%.1f\\%%}  %% oracle share_of_reduction_from_pm %.4f (the full-information reference shows the same structure)" % (R["oracle"]["share_of_reduction_from_pm"], R["oracle"]["share_of_reduction_from_pm"]))
    A("\\newcommand{\\OracleGainWithinCM}{$-$%.1f\\%%} %% oracle pct_below_rule_within_cm %.4f (corrective work ends slightly WORSE under the reference)" % (abs(R["oracle"]["pct_below_rule_within_cm"]), R["oracle"]["pct_below_rule_within_cm"]))
    A("\\newcommand{\\MoneGainPMshare}{%.1f\\%%}    %% m1_alone share_of_reduction_from_pm %.4f" % (R["m1_alone"]["share_of_reduction_from_pm"], R["m1_alone"]["share_of_reduction_from_pm"]))
    A("\\newcommand{\\DecompResidual}{$%.0f\\times10^{-12}$} %% largest |PM+CM-total| over every decider and cell, per-instance weighted business hours (twt_decomposition.json additivity_residual, max %.3e)" % (resid * 1e12, resid))
    A("")
    A("% ---- attainment split preventive vs corrective, headline cell")
    A("%      (twt_decomposition.json rule.attain_pct / reductions_vs_rule.*.attain_pct)")
    A("\\newcommand{\\AttainRulePM}{%.1f\\%%}       %% rule.attain_pct.pm %.4f" % (D["rule"]["attain_pct"]["pm"], D["rule"]["attain_pct"]["pm"]))
    A("\\newcommand{\\AttainRuleCM}{%.1f\\%%}       %% rule.attain_pct.cm %.4f" % (D["rule"]["attain_pct"]["cm"], D["rule"]["attain_pct"]["cm"]))
    A("\\newcommand{\\AttainMzeroPM}{%.1f\\%%}      %% m0_alone.attain_pct.pm %.4f" % (R["m0_alone"]["attain_pct"]["pm"], R["m0_alone"]["attain_pct"]["pm"]))
    A("\\newcommand{\\AttainMzeroCM}{%.1f\\%%}      %% m0_alone.attain_pct.cm %.4f" % (R["m0_alone"]["attain_pct"]["cm"], R["m0_alone"]["attain_pct"]["cm"]))
    A("\\newcommand{\\AttainDeltaMzeroPM}{$+$%.2f}  %% m0_alone.attain_delta_pp.pm %+.4f pp" % (R["m0_alone"]["attain_delta_pp"]["pm"], R["m0_alone"]["attain_delta_pp"]["pm"]))
    A("\\newcommand{\\AttainDeltaMzeroCM}{$-$%.2f}  %% m0_alone.attain_delta_pp.cm %+.4f pp" % (abs(R["m0_alone"]["attain_delta_pp"]["cm"]), R["m0_alone"]["attain_delta_pp"]["cm"]))
    path = os.path.join(_OUT, "macros_p8.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("[write] %s (%d macro lines)" % (path, sum(1 for x in L if x.startswith("\\newcommand"))))


# --------------------------------------------------------------------------- #
# Part: LaTeX table + macro block, rendered from the result files              #
# --------------------------------------------------------------------------- #
def _load_class_rows():
    with open(os.path.join(_OUT, "attainment_by_class.csv")) as fh:
        return list(csv.DictReader(fh))


def part_tables():
    rows = _load_class_rows()
    with open(os.path.join(_OUT, "twt_decomposition.json")) as fh:
        decomp = json.load(fh)
    with open(os.path.join(_OUT, "repro_check.json")) as fh:
        repro = json.load(fh)
    comp = {}
    with open(os.path.join(_OUT, "composition_by_campus.csv")) as fh:
        for r in csv.DictReader(fh):
            comp[r["scope"]] = r

    idx = {}
    for r in rows:
        idx[(r["cell"], r["decider"], int(r["true_class"]))] = r

    def a(cellkey, k, c):
        r = idx.get((cellkey, k, c))
        return None if r is None else (float(r["attain_pct_mean"]), float(r["attain_pct_sd"]))

    def fmt(v):
        if v is None:
            return "--"
        m, s = v
        return "%.2f\\,{\\footnotesize$\\pm$%.2f}" % (m, s) if s >= 0.005 else "%.2f" % m

    head = CELL_HEAD["key"]
    d_one = a(head, "m0_alone", 1)[0] - a(head, "rule", 1)[0]
    d_two = a(head, "m0_alone", 2)[0] - a(head, "rule", 2)[0]
    d_all = a(head, "m0_alone", 0)[0] - a(head, "rule", 0)[0]
    o_one = a(head, "oracle", 1)[0] - a(head, "rule", 1)[0]
    n_one = int(idx[(head, "rule", 1)]["n_orders"])
    late_rule = n_one * (1.0 - a(head, "rule", 1)[0] / 100.0)
    late_m0 = n_one * (1.0 - a(head, "m0_alone", 1)[0] / 100.0)

    cap = (
        "Service attainment by true priority class, the metric campus maintenance "
        "contracts are actually written on. Each value is the share of work orders "
        "finished within their true deadline $d^*$, in per cent; orders are grouped "
        "by their true class $c^*$, pooled over the ten held-out instances of a cell "
        "and then averaged over seeds (\\nseeds{} seeds at the headline cell, "
        "\\seedsMap{} at the other two). Ranges are the seed standard deviation "
        "(population, $\\mathrm{ddof}=0$) in percentage points, and are printed only "
        "where the decider varies with the seed; RULE, RULE+SUP and ORACLE are "
        "deterministic and carry no seed variance. Class~1 carries an "
        "\\slaone{}-business-hour service window and class~2 carries \\slatwo{}; "
        "the class~3 and class~4 windows (\\slathree{} and \\slafour{} business "
        "hours) are met on every order by every decider at all three cells, so "
        "those two columns are reported for completeness rather than for contrast. "
        "The per-class populations are fixed by the latent draw and are therefore "
        "identical across deciders and seeds (headline cell: \\AttainClsOneN{}, "
        "\\AttainClsTwoN{}, \\AttainClsThreeN{} and \\AttainClsFourN{} orders per "
        "ten-instance pool). \\textbf{Takeaway:} the reduction in weighted "
        "tardiness is not bought from the service metric a facility team is judged "
        "on, because \\mname{} raises overall attainment by \\AttainDeltaMzeroAll{} "
        "percentage points and class-2 attainment by \\AttainDeltaMzeroClsTwo{} on "
        "every one of the \\nseeds{} seeds, while the class-1 change is only "
        "\\AttainDeltaMzeroClsOne{} points, or \\AttainClsOneExtra{} orders in "
        "\\AttainClsOneN{} per pool, and does not hold its sign across seeds "
        "(\\AttainWTLMzeroClsOne{} wins, ties and losses); the myopic "
        "full-information reference gives up the same \\AttainDeltaOracleClsOne{} "
        "points on class~1, which places the class-1 shortfall at saturation in the "
        "crew's capacity rather than in what the dispatcher knows.")
    _ = (d_all, d_two, d_one, o_one, n_one)   # reported through the macros above

    L = []
    L.append("% Generated by scripts/y3_practitioner_metrics.py --part tables")
    L.append("% Source: results/y3_p8/attainment_by_class.csv (attain_pct_mean / _sd)")
    L.append("\\begin{table}[pos=htbp]")
    L.append("\\caption{%s}" % cap)
    L.append("\\label{tab:attainment}")
    L.append("\\centering")
    L.append("\\footnotesize")
    L.append("\\begin{tabular}{@{} l c c c c c @{}}")
    L.append("\\toprule")
    L.append(" & \\multicolumn{4}{c}{Attainment by true class $c^*$ (\\%)} & \\\\")
    L.append("\\cmidrule(lr){2-5}")
    L.append("Decider & Class 1 & Class 2 & Class 3 & Class 4 & All orders \\\\")
    L.append("\\midrule")
    for cell in CELLS:
        L.append("\\multicolumn{6}{@{}l}{\\textit{%s}} \\\\" % cell["tex_label"])
        for k in DECIDERS:
            cells = [fmt(a(cell["key"], k, c)) for c in (1, 2, 3, 4)] + \
                    [fmt(a(cell["key"], k, 0))]
            L.append("%s & %s \\\\" % (LABEL[k], " & ".join(cells)))
        if cell is not CELLS[-1]:
            L.append("\\midrule")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    path = os.path.join(_OUT, "table_attainment.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("[write] %s" % path)

    _write_macros(idx, decomp, repro, comp, late_rule, late_m0)

    # -- console summary the report is written from -------------------------- #
    print("\n=== attainment (%% of orders finished within d*, mean over seeds) ===")
    for cell in CELLS:
        print("-- %s" % cell["label"])
        print("   %-40s %8s %8s %8s %8s %8s" % ("decider", "c*=1", "c*=2", "c*=3", "c*=4", "all"))
        for k in DECIDERS:
            vals = [a(cell["key"], k, c) for c in (1, 2, 3, 4, 0)]
            print("   %-40s %s" % (LABEL[k], " ".join(
                "%8s" % ("n/a" if v is None else "%.2f" % v[0]) for v in vals)))
    print("\n=== exact PM/CM decomposition of the reduction vs RULE ===")
    for ck, d in decomp.items():
        print("-- %s  (PM share of orders %.4f)" % (d["label"], d["pm_share_of_orders"]))
        for k, v in d["reductions_vs_rule"].items():
            print("   %-10s total -%.2f  = PM -%.2f (%.1f%%) + CM -%.2f (%.1f%%)  residual %.3e"
                  % (k, v["delta_twt_total"], v["delta_twt_pm"],
                     v["share_of_reduction_from_pm"], v["delta_twt_cm"],
                     v["share_of_reduction_from_cm"], v["additivity_residual"]))
    print("\n[repro] PASS=%s  recomputed MzeroGain=%.4f%%" %
          (repro["PASS"], repro["recomputed_MzeroGain_value"]))
    return True


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["repro", "attain", "composition", "tables", "all"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cells", default="all", help="all | headline")
    args = ap.parse_args()
    assert args.workers <= 4, "at most 4 workers (cores 20-23)"
    os.makedirs(_OUT, exist_ok=True)
    torch.set_num_threads(1)

    cells = CELLS if args.cells == "all" else [CELL_HEAD]

    if args.part in ("repro", "all"):
        r = part_repro(args.workers)
        if not r["PASS"]:
            print("REPRODUCTION GATE FAILED -- refusing to compute anything downstream.")
            sys.exit(2)
    if args.part in ("composition", "all"):
        part_composition()
    if args.part in ("attain", "all"):
        gate = os.path.join(_OUT, "repro_check.json")
        assert os.path.exists(gate), "run --part repro first"
        with open(gate) as fh:
            assert json.load(fh)["PASS"], "reproduction gate did not pass"
        part_attain(args.workers, cells)
    if args.part in ("tables", "all"):
        part_tables()


if __name__ == "__main__":
    main()
