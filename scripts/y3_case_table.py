#!/usr/bin/env python
"""Y3 Phase-7 deliverable 1: the WORKED-EXAMPLE table + the reproduction gate.

What this produces
------------------
results/y3_p7/verify_gate.json   Independent recomputation of the published
                                 headline cell, compared per-instance against
                                 the committed grid cache and against the
                                 published seed-mean ladder.
results/y3_p7/case_table.json    The worked example: what the correction layer
                                 (M0 / ORCA) did to individual work orders on
                                 ONE held-out instance at ONE seed, plus the
                                 instance-level bottom line.
results/y3_p7/case_table.csv     The same selected rows, tabular (a row_type
                                 column separates the order rows from the
                                 instance-total row).
results/y3_p7/case_orders_all.csv  EVERY order of the case instance with the
                                 same fields, so the row selection below can be
                                 re-derived and audited (no cherry-picking).
results/y3_p7/case_table.tex     Ready-to-paste LaTeX: the \newcommand block for
                                 macros.tex and the table itself, generated from
                                 the same numbers so the two cannot drift.

The cell
--------
Read from scripts/y3_p4_m0grid.py rather than restated here, so it cannot drift
from the run that produced the published numbers: campus 9, the released
storm2 high-load track at u=100 (saturation), full-class-shift overlay, family
F-NL, master seed 12345, beta=1.00, rho=0.25, eps=0, theta=1.0, targeted
review, 8 DAgger iterations, estimator fit on files[:16], probe files[16:20],
evaluation on the held-out files[20:30], seeds 301-310. Identical to
macros.tex's E1 headline cell c9_storm2_u100_b1.00_r0.25.

Worked example: FIRST held-out instance (c09_storm2_w80_u100_0020) and FIRST
seed (301). Not chosen for a favourable result; the choice is fixed by position
in the held-out slice and in the seed list, and --case-index / --case-seed exist
only so a reviewer can check any other one.

Row selection (deterministic, sign-symmetric, stated so it can be audited)
-------------------------------------------------------------------------
Every order carries ``delta = w* * L*_M0 - w* * L*_RULE``, its own signed
contribution to the change in the scored objective (these sum EXACTLY to the
instance-level change; the script asserts it). Ranking by |delta| is symmetric
in sign: an order the layer harmed ranks exactly like one it helped.

Each order falls in one of six behaviour groups: the corrected class is more
urgent than the recorded class (promoted), less urgent (demoted), or effectively
unchanged, crossed with whether the correction moved the class TOWARD or AWAY
from the true class. Behaviour is read off the class the layer actually scores
with, ``c_hat = clip(c - hat_s, 1, 4)``, so an order whose correction the clip
absorbs counts as unchanged, which is what the printed columns show.

  * every non-empty group contributes its largest-|delta| order (ties by id);
  * the two largest-|delta| orders overall are added if missing;
  * the largest-|delta| order the layer made WORSE off is added if missing.

Rows are printed in descending estimated shift, so promotions read at the top
and demotions at the bottom.

Compute
-------
CPU only. The committed grid ran ONE numeric thread per process
(OMP/MKL/torch=1, 8 worker processes). That setting is part of the artifact:
torch's reduction order depends on the thread count, and refitting the same
estimator with torch.set_num_threads(4) moves the seed-301 held-out mean from
1997.69 to 1940.82. This script therefore keeps ONE numeric thread per worker
and gets its parallelism from worker PROCESSES, so the numbers reproduce
bit-for-bit; pin it to four cores from the shell:

    cd <repo> && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-3 \
        python scripts/y3_case_table.py --workers 4
"""

from __future__ import annotations

import os

# One numeric thread per process, set BEFORE numpy/torch import (parent and
# forked workers) -- see "Compute" above; this is what the published run used.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch                                                    # noqa: E402

from fmwos.env import DispatchEnv                                # noqa: E402
from fmwos.hitl import deciders as dec                           # noqa: E402
from fmwos.hitl import overlay as ov                             # noqa: E402
from fmwos.hitl.supervisor import Supervisor                     # noqa: E402
from fmwos.hitl import augmented_rule as AR                      # noqa: E402
from fmwos.hitl import true_objective as TO                      # noqa: E402

import y3_p4_m0grid as G                                         # noqa: E402

_OUT = os.path.join(_ROOT, "results", "y3_p7")
_CACHE = os.path.join(_ROOT, "results", "y3_p4", "cache")
_HARVEST = os.path.join(_ROOT, "results", "y3_p5", "harvest",
                        "primary_multiseed_summary.json")
_GATE = os.path.join(_ROOT, "results", "y3_p4", "m0_gate_summary.json")

SEEDS = list(range(301, 311))
CELL_KEY = "c9_storm2_u100_b1.00_r0.25"

# Published values this run must reproduce (macros.tex E1 block).
PUBLISHED_MACROS = {
    "MzeroGain": "45.4%",       # m0_alone pct_below_rule 45.3620
    "TwtRule": 3644.8,          # rule twt_mean
    "TwtMzero": 1991.5,         # m0_alone twt_mean
    "TwtMzeroStd": 62.3,        # m0_alone twt_std_pop
    "SupGain": "26.1%",         # rule_sup pct_below_rule 26.1207
    "MzeroSupGain": "49.0%",    # m0_sup pct_below_rule 48.9816
    "OracleGap": "50.2%",       # oracle pct_below_rule 50.2029
}

# Behaviour thresholds (stated here, used by the table caption).
SHIFT_TOL = 0.05        # |hat_s| <= this  ->  class left effectively unchanged
EPSF = 1e-9


# --------------------------------------------------------------------------- #
# One (cell, seed): fit the estimator, score the held-out set                  #
# --------------------------------------------------------------------------- #
def _load(p):
    with open(p) as fh:
        return json.load(fh)


def _task(seed):
    """The locked headline-cell task dict, built by the grid driver itself."""
    return G._base_task(campus=9, u=100, beta=1.0, rho=0.25, seed=seed,
                        scope="primary", part="A")


def _fit_and_score(seed, case_index=None):
    """Reproduce the grid's per-(cell, seed) evaluation. Returns per-instance
    TWT* for the five deciders, plus (optionally) the full per-order record of
    the ``case_index``-th held-out instance."""
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    task = _task(seed)
    n_train, n_probe, n_eval = task["n_train"], task["n_probe"], task["n_eval"]

    files = G.locate_files(task["campus"], task["regime"], u=task["u"])
    train = [_load(p) for p in files[:n_train]]
    probe = [_load(p) for p in files[n_train:n_train + n_probe]]
    eval_files = files[n_train + n_probe:n_train + n_probe + n_eval]
    eval_insts = [_load(p) for p in eval_files]
    assert not (set(eval_files) & set(files[:n_train + n_probe])), "eval overlaps train"

    overlay = ov.Overlay(ov.OverlayParams(
        beta=task["beta"], family=task["family"], master_seed=task["master_seed"],
        channel=task["channel"]))

    # Estimator: same seeding, same call, same arguments as the grid driver.
    torch.manual_seed(seed)
    np.random.seed(seed)
    res = AR.run_m0(train, probe, overlay,
                    beta_rho_eps=(task["beta"], task["rho"], task["eps"]),
                    outer_iters=task["m0_iters"], mechanism=task["mech"],
                    theta=task["theta"], seed=seed, device="cpu", verbose=False)
    estimator = res["estimator"]

    per = {k: [] for k in G.DECIDERS}
    inst_ids = []
    case = None
    for idx, inst in enumerate(eval_insts):
        applied = overlay.apply(inst)
        inst_ids.append(inst["meta"]["id"])

        def sc(sched):
            return TO.score_true(inst, sched, overlay, applied)["TWT_true"]

        rule_sched = dec.run_rule(DispatchEnv(inst), "atc", seed=seed)
        per["rule"].append(sc(rule_sched))
        m0d = AR.augmented_atc_decider(estimator, inst, channel=task["channel"])
        m0_sched, _ = DispatchEnv(inst).run_supervised(m0d, supervisor=None,
                                                       method="m0", seed=seed)
        per["m0_alone"].append(sc(m0_sched))
        osup = Supervisor(overlay, inst, rho=0.0, applied=applied)
        per["oracle"].append(sc(dec.run_oracle_greedy(DispatchEnv(inst), osup,
                                                      seed=seed)))
        rsup = Supervisor(overlay, inst, rho=task["rho"], epsilon=task["eps"],
                          theta=task["theta"], mechanism=task["mech"],
                          seed=seed, applied=applied)
        rsched, _ = dec.run_rule_sup(DispatchEnv(inst), "atc", rsup, seed=seed)
        per["rule_sup"].append(sc(rsched))
        m0d2 = AR.augmented_atc_decider(estimator, inst, channel=task["channel"])
        m0sup = Supervisor(overlay, inst, rho=task["rho"], epsilon=task["eps"],
                           theta=task["theta"], mechanism=task["mech"],
                           seed=seed, applied=applied)
        m0s_sched, _ = DispatchEnv(inst).run_supervised(m0d2, supervisor=m0sup,
                                                        method="m0_sup", seed=seed)
        per["m0_sup"].append(sc(m0s_sched))

        if case_index is not None and idx == case_index:
            case = build_case_record(inst, applied, estimator, rule_sched,
                                     m0_sched, task, seed)

    return {"seed": seed, "inst_ids": inst_ids,
            "per": {k: [float(x) for x in v] for k, v in per.items()},
            "m0_final": res["per_iter"][-1], "case": case,
            "elapsed_s": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# Per-order record for one instance                                           #
# --------------------------------------------------------------------------- #
def _dispatch_index(sched, trade_of):
    """wo_id -> (global dispatch rank, rank within its own trade, start, end).

    ``assignments`` is appended in execution order by the shared event loop
    (fmwos.env.DispatchEnv._driver), so its index IS the realised dispatch
    sequence position."""
    out = {}
    seen_in_trade = {}
    for i, a in enumerate(sched["assignments"]):
        wid = a["wo"]
        tr = trade_of[wid]
        seen_in_trade[tr] = seen_in_trade.get(tr, 0) + 1
        out[wid] = (i + 1, seen_in_trade[tr], float(a["start_bh"]),
                    float(a["end_bh"]))
    return out


def build_case_record(inst, applied, estimator, rule_sched, m0_sched, task, seed):
    """Every order of ``inst``: what the layer estimated, what class it scored
    the order with, the true class, and the consequence under both deciders."""
    wos = inst["work_orders"]
    trade_of = {w["id"]: w["trade"] for w in wos}
    hs = AR.hat_s_map(estimator, inst)
    po = applied["per_order"]

    ix_rule = _dispatch_index(rule_sched, trade_of)
    ix_m0 = _dispatch_index(m0_sched, trade_of)

    rows = []
    for w in wos:
        wid = w["id"]
        c = int(w["priority"])
        s_hat = float(hs[wid])
        # The class the layer actually scores the order with (augmented_rule.
        # corrected_weight / corrected_deadline clip c - hat_s into [1, 4]).
        c_hat = float(min(4.0, max(1.0, c - s_hat)))
        lat = po[wid]
        c_star, w_star, d_star = int(lat["c_star"]), float(lat["w_star"]), float(lat["d_star"])

        r_rank, r_trank, r_start, r_end = ix_rule[wid]
        m_rank, m_trank, m_start, m_end = ix_m0[wid]
        L_rule = max(0.0, r_end - d_star)
        L_m0 = max(0.0, m_end - d_star)
        contrib_rule = w_star * L_rule
        contrib_m0 = w_star * L_m0

        # Behaviour is read off the class the layer SCORES with (clip included),
        # so it matches the printed c / c_hat columns.
        move = c_hat - c
        behaviour = ("promoted" if move <= -SHIFT_TOL else
                     ("demoted" if move >= SHIFT_TOL else "unchanged"))
        miscorrected = abs(c_hat - c_star) > abs(c - c_star) + SHIFT_TOL

        rows.append({
            "wo_id": wid, "trade": w["trade"], "p_bh": float(w["p_bh"]),
            "release_bh": float(w["release_bh"]), "due_recorded_bh": float(w["due_bh"]),
            "class_recorded": c, "hat_s": s_hat, "class_corrected": c_hat,
            "class_true": c_star, "true_shift": int(lat["s"]),
            "w_recorded": float(w["weight"]), "w_star": w_star, "d_star_bh": d_star,
            "rank_rule": r_rank, "rank_m0": m_rank,
            "rank_trade_rule": r_trank, "rank_trade_m0": m_trank,
            "start_rule_bh": r_start, "start_m0_bh": m_start,
            "completion_rule_bh": r_end, "completion_m0_bh": m_end,
            "lateness_rule_bh": L_rule, "lateness_m0_bh": L_m0,
            "twt_contrib_rule": contrib_rule, "twt_contrib_m0": contrib_m0,
            "delta_twt": contrib_m0 - contrib_rule,
            "behaviour": behaviour, "miscorrected": bool(miscorrected),
            "harmed": bool(contrib_m0 - contrib_rule > EPSF),
        })

    twt_rule = float(sum(r["twt_contrib_rule"] for r in rows))
    twt_m0 = float(sum(r["twt_contrib_m0"] for r in rows))
    return {"instance_id": inst["meta"]["id"], "seed": seed,
            "n_work_orders": len(wos), "rows": rows,
            "twt_rule": twt_rule, "twt_m0": twt_m0}


# --------------------------------------------------------------------------- #
# Row selection                                                               #
# --------------------------------------------------------------------------- #
def select_rows(rows, n_top=2, n_min=6, n_max=10):
    """Deterministic, sign-symmetric selection (see module docstring)."""
    order = sorted(range(len(rows)),
                   key=lambda i: (-abs(rows[i]["delta_twt"]), rows[i]["wo_id"]))
    picked, why = [], {}

    def take(i, reason):
        if i not in picked:
            picked.append(i)
            why[i] = reason

    # (1) one representative per non-empty behaviour group: its largest mover.
    label = {(True, True): "corrected away from the true class",
             (True, False): "corrected toward the true class"}
    groups = {}
    for i in order:                                   # already |delta|-sorted
        key = (rows[i]["behaviour"], rows[i]["miscorrected"])
        groups.setdefault(key, i)
    for (beh, mis), i in sorted(groups.items()):
        take(i, "largest mover among orders %s and %s"
             % (beh, label[(True, mis)]))
    # (2) the biggest movers overall, either direction.
    for rank, i in enumerate(order[:n_top]):
        take(i, "largest |change in TWT*| overall (#%d)" % (rank + 1))
    # (3) the order the layer made worst off.
    harmed = [i for i in order if rows[i]["harmed"]]
    if harmed:
        take(harmed[0], "largest increase in TWT* (the layer's worst call here)")
    # (4) top up if a degenerate instance left fewer than n_min rows.
    for i in order:
        if len(picked) >= n_min:
            break
        take(i, "fill to n_min")
    picked = picked[:n_max]
    # Print order: most promoted first, most demoted last.
    picked.sort(key=lambda i: (rows[i]["class_corrected"] - rows[i]["class_recorded"],
                               rows[i]["wo_id"]))
    return [(i, why[i]) for i in picked]


# --------------------------------------------------------------------------- #
# Verification gate                                                           #
# --------------------------------------------------------------------------- #
def _pct_below(rule_mean, x):
    return 100.0 * (rule_mean - x) / rule_mean


def verification_gate(records):
    """Compare the recomputation against (a) the committed per-(cell, seed)
    cache, per instance, and (b) the published seed-mean ladder + macros."""
    pub = json.load(open(_HARVEST))
    gate_json = json.load(open(_GATE))
    cell = gate_json["cells"][CELL_KEY]

    # (a) per-instance identity against the committed grid cache.
    per_inst = []
    for rec in records:
        sig = G._cell_sig(_task(rec["seed"]))
        path = os.path.join(_CACHE, "%s.json" % sig)
        entry = {"seed": rec["seed"], "cache_sig": sig,
                 "cache_file": os.path.relpath(path, _ROOT),
                 "cache_found": os.path.exists(path)}
        if entry["cache_found"]:
            cached = json.load(open(path))
            entry["inst_ids_match"] = (cached["inst_ids"] == rec["inst_ids"])
            worst = {}
            for d in G.DECIDERS:
                diff = np.abs(np.asarray(cached["per"][d])
                              - np.asarray(rec["per"][d]))
                worst[d] = float(diff.max())
            entry["max_abs_per_instance_diff"] = worst
            entry["exact"] = bool(max(worst.values()) == 0.0)
        per_inst.append(entry)

    # (b) seed-mean ladder.
    ladder = {}
    rule_mean = float(np.mean([np.mean(r["per"]["rule"]) for r in records]))
    for d in G.DECIDERS:
        seed_means = np.asarray([np.mean(r["per"][d]) for r in records])
        mine = float(seed_means.mean())
        pub_d = pub["ladder"][d]
        ladder[d] = {
            "recomputed_twt_mean": mine,
            "published_twt_mean": pub_d["twt_mean"],
            "abs_diff_twt": abs(mine - pub_d["twt_mean"]),
            "recomputed_pct_below_rule": _pct_below(rule_mean, mine),
            "published_pct_below_rule": pub_d["pct_below_rule"],
            "abs_diff_pct": abs(_pct_below(rule_mean, mine) - pub_d["pct_below_rule"]),
            "recomputed_twt_std_pop": float(seed_means.std(ddof=0)),
            "published_twt_std_pop": pub_d["twt_std_pop"],
        }

    headline = ladder["m0_alone"]
    gate = {
        "cell_key": CELL_KEY,
        "cell": pub["cell"],
        "seeds": [r["seed"] for r in records],
        "n_eval_instances": len(records[0]["inst_ids"]),
        "eval_instance_ids": records[0]["inst_ids"],
        "published_eval_instance_ids": pub["eval_inst_ids"],
        "eval_ids_match_published": records[0]["inst_ids"] == pub["eval_inst_ids"],
        "sources": {
            "published_ladder": os.path.relpath(_HARVEST, _ROOT),
            "published_grid_cell": os.path.relpath(_GATE, _ROOT),
            "committed_cell_cache": os.path.relpath(_CACHE, _ROOT),
        },
        "published_grid_cell_check": {
            "m0_alone_twt_mean": cell["ladder"]["m0_alone"]["twt_mean"],
            "m0_alone_pct_below_rule": cell["ladder"]["m0_alone"]["pct_below_rule"],
        },
        "headline_macro": {
            "macro": "\\MzeroGain",
            "published_printed": PUBLISHED_MACROS["MzeroGain"],
            "published_full_precision": pub["ladder"]["m0_alone"]["pct_below_rule"],
            "recomputed": headline["recomputed_pct_below_rule"],
            "abs_diff_pct_points": headline["abs_diff_pct"],
            "recomputed_printed": "%.1f%%" % headline["recomputed_pct_below_rule"],
            "agrees_at_printed_precision":
                ("%.1f%%" % headline["recomputed_pct_below_rule"]) == PUBLISHED_MACROS["MzeroGain"],
        },
        "ladder": ladder,
        "per_seed_cache_comparison": per_inst,
        "all_seeds_bit_exact_vs_cache":
            all(e.get("exact", False) for e in per_inst),
        "published_macros_checked": PUBLISHED_MACROS,
        "thread_note": ("reproduces only with ONE numeric thread per process "
                        "(OMP/MKL/torch = 1), the committed grid setting; four "
                        "torch threads refit a different estimator"),
    }
    return gate


# --------------------------------------------------------------------------- #
# Writers                                                                     #
# --------------------------------------------------------------------------- #
CSV_COLS = ["row_type", "selection_reason", "wo_id", "trade", "p_bh",
            "class_recorded", "hat_s", "class_corrected", "class_true",
            "true_shift", "w_star", "d_star_bh", "release_bh",
            "due_recorded_bh", "rank_rule", "rank_m0", "rank_trade_rule",
            "rank_trade_m0", "completion_rule_bh", "completion_m0_bh",
            "lateness_rule_bh", "lateness_m0_bh", "twt_contrib_rule",
            "twt_contrib_m0", "delta_twt", "behaviour", "miscorrected",
            "harmed"]


def _csv_row(r, row_type, reason=""):
    out = {k: "" for k in CSV_COLS}
    out["row_type"] = row_type
    out["selection_reason"] = reason
    for k, v in r.items():
        if k in out:
            out[k] = ("%.6f" % v) if isinstance(v, float) else v
    return out


def write_csv(path, rows_selected, case):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for r, reason in rows_selected:
            w.writerow(_csv_row(r, "order", reason))
        tot = {"wo_id": "__INSTANCE_TOTAL__",
               "twt_contrib_rule": case["twt_rule"],
               "twt_contrib_m0": case["twt_m0"],
               "delta_twt": case["twt_m0"] - case["twt_rule"]}
        w.writerow(_csv_row(tot, "instance_total",
                            "TWT*(w*,d*) summed over all %d orders of %s"
                            % (case["n_work_orders"], case["instance_id"])))


def write_all_orders(path, case):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for r in sorted(case["rows"], key=lambda x: x["wo_id"]):
            w.writerow(_csv_row(r, "order", ""))


# --------------------------------------------------------------------------- #
# Ready-to-paste LaTeX (generated here so the table cannot drift from the data) #
# --------------------------------------------------------------------------- #
def _th(x, dp=0):
    """Thousands-separated number in the macros.tex style (2{,}253)."""
    return format(float(x), ",.%df" % dp).replace(",", "{,}")


def _sg(x, dp=1):
    """Signed math number; a value that rounds to zero carries no sign."""
    if abs(x) < 0.5 * 10 ** (-dp):
        return "$%.*f$" % (dp, 0.0)
    return "$%s%.*f$" % ("+" if x > 0 else "-", dp, abs(x))


def write_tex(path, rows_selected, case, out):
    tot = out["instance_totals"]
    cnt = out["instance_order_counts"]
    macros = [
        ("CaseInstance", case["instance_id"].replace("_", r"\_"),
         "results/y3_p7/case_table.json:instance_id"),
        ("CaseNOrders", _th(case["n_work_orders"]),
         "results/y3_p7/case_table.json:n_work_orders"),
        ("CaseTwtRule", _th(tot["twt_rule"], 1),
         "results/y3_p7/case_table.json:instance_totals.twt_rule"),
        ("CaseTwtMzero", _th(tot["twt_m0"], 1),
         "results/y3_p7/case_table.json:instance_totals.twt_m0"),
        ("CaseTwtDrop", _th(tot["reduction_abs"], 1),
         "results/y3_p7/case_table.json:instance_totals.reduction_abs"),
        ("CaseGain", "%.1f\\%%" % tot["reduction_pct"],
         "results/y3_p7/case_table.json:instance_totals.reduction_pct"),
        ("CaseNPromoted", _th(cnt["promoted"]),
         "results/y3_p7/case_table.json:instance_order_counts.promoted"),
        ("CaseNDemoted", _th(cnt["demoted"]),
         "results/y3_p7/case_table.json:instance_order_counts.demoted"),
        ("CaseNUnchanged", _th(cnt["unchanged"]),
         "results/y3_p7/case_table.json:instance_order_counts.unchanged"),
        ("CaseNAway", _th(cnt["corrected_away_from_true_class"]),
         "results/y3_p7/case_table.json:instance_order_counts."
         "corrected_away_from_true_class"),
        ("CaseNMoved", _th(cnt["dispatch_position_changed_in_trade"]),
         "results/y3_p7/case_table.json:instance_order_counts."
         "dispatch_position_changed_in_trade"),
        ("CaseNHelped", _th(cnt["helped"]),
         "results/y3_p7/case_table.json:instance_order_counts.helped"),
        ("CaseNHarmed", _th(cnt["harmed"]),
         "results/y3_p7/case_table.json:instance_order_counts.harmed"),
        ("CaseNLateRule", _th(cnt["late_under_rule"]),
         "results/y3_p7/case_table.json:instance_order_counts.late_under_rule"),
        ("CaseNLateMzero", _th(cnt["late_under_m0"]),
         "results/y3_p7/case_table.json:instance_order_counts.late_under_m0"),
    ]
    L = []
    L.append("% ---------------------------------------------------------------"
             "------------")
    L.append("% GENERATED by scripts/y3_case_table.py -- do not hand-edit; re-run"
             " the script.")
    L.append("%% Worked-example table for the headline cell %s, instance %s, "
             "seed %d." % (CELL_KEY, case["instance_id"], case["seed"]))
    L.append("% ---------------------------------------------------------------"
             "------------")
    L.append("")
    L.append("%% --- macros for macros.tex (paste into the RESULTS block) ---")
    for name, val, src in macros:
        L.append("\\newcommand{\\%s}{%s} %% %s" % (name, val, src))
    L.append("")
    L.append(r"\begin{table}[pos=htbp]")
    L.append(r"""\caption{What the correction layer did to individual work orders. The instance
is the first held-out instance of the headline cell (\CaseInstance{}: campus C9,
the benchmark's high-load track at utilisation \utilsat{}, $\beta=\betahigh$,
$\rho=\rholow$), and the seed is the first of the ten, \seedlo{}; neither was
chosen on its result. The recorded class $c$ is the class in the work-order
record, $\hat{s}$ is the class shift the fitted estimator predicts from
observable fields alone, $\hat{c}=\mathrm{clip}(c-\hat{s},1,4)$ is the class the
layer scores the order with, and $c^*$ is the operationally true class. Queue
position is the order's place in its own trade's realised dispatch sequence, so
an order can move up while being demoted if the orders around it move further;
lateness is measured against the true deadline $d^*$; and $\Delta\mathrm{TWT}^*$
is the order's own signed contribution to the change in the scored objective,
which sums over all \CaseNOrders{} orders to the instance total in the last row.
The seven orders are the largest mover of each behaviour the layer can produce,
together with the two largest movers overall and its worst call, chosen by a
stated rule that ranks orders by the size of that contribution irrespective of
its sign. This single instance gives a \CaseGain{} reduction; the cell-level
figure over the ten held-out instances and \nseeds{} seeds is \MzeroGain{}.
\textbf{Takeaway:} the layer buys its \CaseGain{} reduction on this instance from
a small number of consequential orders, promoting under-recorded work that the
rule leaves late, and it pays for that with a few orders it pushes back,
including one it demoted away from the true class.}""")
    L.append(r"\label{tab:caseout}")
    L.append(r"\centering")
    L.append(r"\footnotesize")
    L.append(r"\begin{tabular}{@{} l l r c r r c c c r @{}}")
    L.append(r"\toprule")
    L.append(r"      &       & $p$  & Class &           & Corr.     & True  & "
             r"\multicolumn{2}{c}{Rule $\to$ \mname{}} & $\Delta\mathrm{TWT}^*$ \\")
    L.append(r"\cmidrule(lr){8-9}")
    L.append(r"Order & Trade & (bh) & $c$   & $\hat{s}$ & $\hat{c}$ & $c^*$ & "
             r"queue position & lateness (bh) & (weighted bh) \\")
    L.append(r"\midrule")
    for r, _why in rows_selected:
        L.append("%s%s & %s & %.2f & %d & %s & %.2f & %d & "
                 r"%d\,$\to$\,%d & %.1f\,$\to$\,%.1f & %s \\"
                 % (r["wo_id"], r"$^{\dagger}$" if r["miscorrected"] else "",
                    r["trade"], r["p_bh"], r["class_recorded"],
                    _sg(r["hat_s"], 2), r["class_corrected"], r["class_true"],
                    r["rank_trade_rule"], r["rank_trade_m0"],
                    r["lateness_rule_bh"], r["lateness_m0_bh"],
                    _sg(r["delta_twt"], 1)))
    L.append(r"\midrule")
    L.append(r"\multicolumn{7}{@{}l}{$\mathrm{TWT}^*(w^*,d^*)$, all "
             r"\CaseNOrders{} orders of this instance} & "
             r"\multicolumn{2}{c}{\CaseTwtRule{} $\to$ \CaseTwtMzero{}} & "
             r"$-$\CaseTwtDrop{} \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append("")
    L.append(r"\vspace{2pt}")
    L.append(r"""{\footnotesize $^{\dagger}$The correction moved the class away from the true
class. Across the whole instance the layer promoted \CaseNPromoted{} orders,
demoted \CaseNDemoted{}, left \CaseNUnchanged{} effectively unchanged, and moved
\CaseNAway{} away from the true class; \CaseNMoved{} orders changed position in
their trade's sequence, \CaseNHelped{} finished with less true weighted tardiness
and \CaseNHarmed{} with more, and the number of late orders fell from
\CaseNLateRule{} to \CaseNLateMzero{}.}""")
    L.append(r"\end{table}")
    L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L))


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--case-seed", type=int, default=301,
                    help="seed for the worked example (default: the first seed)")
    ap.add_argument("--case-index", type=int, default=0,
                    help="index into the held-out slice (default: the first)")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    os.makedirs(_OUT, exist_ok=True)
    assert args.case_seed in args.seeds, "case seed must be among the scored seeds"

    t0 = time.time()
    records = []
    jobs = [(s, args.case_index if s == args.case_seed else None)
            for s in args.seeds]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut = {ex.submit(_fit_and_score, s, ci): s for s, ci in jobs}
            for f in as_completed(fut):
                rec = f.result()
                records.append(rec)
                print("  [seed %d] rule=%.5f m0=%.5f (%.0fs)"
                      % (rec["seed"], np.mean(rec["per"]["rule"]),
                         np.mean(rec["per"]["m0_alone"]), rec["elapsed_s"]),
                      flush=True)
    else:
        for s, ci in jobs:
            rec = _fit_and_score(s, ci)
            records.append(rec)
            print("  [seed %d] rule=%.5f m0=%.5f (%.0fs)"
                  % (rec["seed"], np.mean(rec["per"]["rule"]),
                     np.mean(rec["per"]["m0_alone"]), rec["elapsed_s"]), flush=True)
    records.sort(key=lambda r: r["seed"])

    # ---- gate ---- #
    gate = verification_gate(records)
    with open(os.path.join(_OUT, "verify_gate.json"), "w") as fh:
        json.dump(gate, fh, indent=1)
    hm = gate["headline_macro"]
    print("\n[GATE] \\MzeroGain published %s (%.4f%%) | recomputed %s (%.4f%%) "
          "| diff %.4f pct-pts | per-instance bit-exact vs cache: %s"
          % (hm["published_printed"], hm["published_full_precision"],
             hm["recomputed_printed"], hm["recomputed"],
             hm["abs_diff_pct_points"], gate["all_seeds_bit_exact_vs_cache"]),
          flush=True)
    if not hm["agrees_at_printed_precision"]:
        print("[GATE] FAILED -- stopping before the case table.", flush=True)
        return 1

    # ---- case table ---- #
    case = [r["case"] for r in records if r["case"] is not None][0]
    rows = case["rows"]
    # The per-order deltas must sum EXACTLY to the instance-level change.
    delta_sum = sum(r["delta_twt"] for r in rows)
    assert abs(delta_sum - (case["twt_m0"] - case["twt_rule"])) < 1e-6, delta_sum
    # ... and the instance totals must equal the scored per-instance values.
    rec_case = [r for r in records if r["seed"] == args.case_seed][0]
    k = args.case_index
    assert abs(case["twt_rule"] - rec_case["per"]["rule"][k]) < 1e-6
    assert abs(case["twt_m0"] - rec_case["per"]["m0_alone"][k]) < 1e-6

    sel = select_rows(rows)
    rows_selected = [(rows[i], why) for i, why in sel]

    counts = {
        "promoted": sum(1 for r in rows if r["behaviour"] == "promoted"),
        "demoted": sum(1 for r in rows if r["behaviour"] == "demoted"),
        "unchanged": sum(1 for r in rows if r["behaviour"] == "unchanged"),
        "corrected_away_from_true_class": sum(1 for r in rows if r["miscorrected"]),
        "harmed": sum(1 for r in rows if r["harmed"]),
        "helped": sum(1 for r in rows if r["delta_twt"] < -EPSF),
        "own_twt_unchanged": sum(1 for r in rows if abs(r["delta_twt"]) <= EPSF),
        "dispatch_position_changed_in_trade":
            sum(1 for r in rows if r["rank_trade_rule"] != r["rank_trade_m0"]),
        "completion_time_changed":
            sum(1 for r in rows
                if abs(r["completion_rule_bh"] - r["completion_m0_bh"]) > EPSF),
        "late_under_rule": sum(1 for r in rows if r["lateness_rule_bh"] > EPSF),
        "late_under_m0": sum(1 for r in rows if r["lateness_m0_bh"] > EPSF),
    }
    red = 100.0 * (case["twt_rule"] - case["twt_m0"]) / case["twt_rule"]
    out = {
        "cell_key": CELL_KEY,
        "cell": _task(args.case_seed),
        "case_seed": args.case_seed,
        "case_instance_index_in_heldout": args.case_index,
        "instance_id": case["instance_id"],
        "n_work_orders": case["n_work_orders"],
        "why_this_instance_and_seed":
            "first held-out instance (index 0 of files[20:30]) and first seed "
            "(301 of 301-310); not selected on the result",
        "selection_rule": {
            "groups": "largest |delta_twt| order of each non-empty (behaviour x "
                      "corrected toward/away from the true class) group",
            "plus": ["the two largest |delta_twt| orders overall",
                     "the order with the largest increase in TWT*"],
            "ranking_is_sign_symmetric": True,
            "class_tolerance": SHIFT_TOL,
            "print_order": "ascending (class_corrected - class_recorded), "
                           "promotions first",
        },
        "definitions": {
            "hat_s": "estimator output; positive = more urgent than recorded",
            "class_corrected": "clip(class_recorded - hat_s, 1, 4); the class the "
                               "layer scores the order with (weight AND deadline)",
            "class_true": "overlay c* = clip(c - s, 1, 4)",
            "behaviour": "promoted / demoted / unchanged, from "
                         "class_corrected - class_recorded against a tolerance "
                         "of %.2f class (so a correction absorbed by the clip "
                         "reads as unchanged)" % SHIFT_TOL,
            "miscorrected": "|class_corrected - class_true| > "
                            "|class_recorded - class_true| + %.2f: the "
                            "correction moved the class away from the true one"
                            % SHIFT_TOL,
            "lateness_*_bh": "max(0, completion - d*), d* = release + SLA(c*)",
            "delta_twt": "w* * (lateness_m0 - lateness_rule); sums over all "
                         "orders to the instance-level change in TWT*(w*,d*)",
        },
        "instance_totals": {
            "twt_rule": case["twt_rule"], "twt_m0": case["twt_m0"],
            "reduction_abs": case["twt_rule"] - case["twt_m0"],
            "reduction_pct": red,
        },
        "instance_order_counts": counts,
        "seed_mean_context": {
            "seed_301_heldout_mean_rule": float(np.mean(rec_case["per"]["rule"])),
            "seed_301_heldout_mean_m0": float(np.mean(rec_case["per"]["m0_alone"])),
            "ten_seed_heldout_mean_rule": gate["ladder"]["rule"]["recomputed_twt_mean"],
            "ten_seed_heldout_mean_m0": gate["ladder"]["m0_alone"]["recomputed_twt_mean"],
            "ten_seed_pct_below_rule": gate["ladder"]["m0_alone"]["recomputed_pct_below_rule"],
        },
        "estimator_last_iter": {k2: rec_case["m0_final"][k2] for k2 in
                                ["iter", "n_reviews", "n_overrides", "override_rate",
                                 "pearson_r", "sign_acc_nonzero", "exact_class_acc",
                                 "zero_baseline_acc"]},
        "rows": [dict(r, selection_reason=why) for r, why in rows_selected],
    }
    with open(os.path.join(_OUT, "case_table.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    write_csv(os.path.join(_OUT, "case_table.csv"), rows_selected, case)
    write_all_orders(os.path.join(_OUT, "case_orders_all.csv"), case)
    write_tex(os.path.join(_OUT, "case_table.tex"), rows_selected, case, out)

    print("[case] %s seed %d: TWT* rule %.2f -> M0 %.2f (%.1f%% lower); "
          "%d rows selected of %d orders"
          % (case["instance_id"], args.case_seed, case["twt_rule"],
             case["twt_m0"], red, len(rows_selected), len(rows)))
    print("[case] order counts: %s" % counts)
    for r, why in rows_selected:
        print("   %-7s %-4s p=%5.2f c=%d  hat_s=%+.2f  c_hat=%.2f  c*=%d  "
              "L*: %7.2f -> %7.2f  dTWT=%+9.2f  [%s%s] %s"
              % (r["wo_id"], r["trade"], r["p_bh"], r["class_recorded"],
                 r["hat_s"], r["class_corrected"], r["class_true"],
                 r["lateness_rule_bh"], r["lateness_m0_bh"], r["delta_twt"],
                 r["behaviour"], ", MISCORRECTED" if r["miscorrected"] else "",
                 why))
    print("[y3_p7] wrote %s (%.0fs total)" % (_OUT, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
