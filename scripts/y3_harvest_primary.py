#!/usr/bin/env python
"""P5 Tier-1 PRIMARY-cell harvest (Paper Y3).

Evaluates the 10 completed fair-M1 (deadline_head=True) checkpoints for the
primary gate cell -- c9 storm2 u100, beta=1.0, rho=0.25, eps=0, TARGETED,
channel=full_class_shift, seeds 301..310 -- in ALONE (M1-FROZEN) and IN-LOOP
(M1+SUP) mode on the true objective TWT*(d*), then combines them with the
COMMITTED m0-grid per-instance values (RULE, RULE+SUP, M0, M0+SUP, ORACLE) from
results/y3_p4/cache to produce the multi-seed H1/H2/H3 + M1-vs-M0 headline.

Additive only. Reuses the frozen-decider eval from scripts/y3_p3_eval.py
(rollout_policy_alone / rollout_policy_sup, the TWT*(d*) validator scoring path,
paired Wilcoxon, W/T/L). M0 is NOT recomputed; its committed cache values are
reused and cross-checked against results/y3_p4/m0_gate_summary.json.

Held-out eval split = files[20:30] (n_train=16, n_probe=4, n_eval=10), the SAME
split the m0-grid and y3_p3_eval used. NO training.

Run (COEXISTENCE: cap workers, OMP=1/worker, nice):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 15 /home/ziheng/miniconda3/envs/fjsp/bin/python \
    scripts/y3_harvest_primary.py --workers 5
"""

from __future__ import annotations

import os

# Cap numeric runtimes to one thread per worker BEFORE torch/numpy import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import glob
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch                                                        # noqa: E402

# Reuse the frozen-decider eval harness verbatim (no reimplementation).
import y3_p3_eval as P3                                             # noqa: E402
from fmwos.hitl import overlay as ov                               # noqa: E402
from fmwos.hitl import true_objective as TO                        # noqa: E402
from fmwos.hitl.latent_head import LatentDispatchPolicy            # noqa: E402

# --------------------------------------------------------------------------- #
# Primary gate cell (LOCKED)                                                   #
# --------------------------------------------------------------------------- #
CELL = {"campus": 9, "regime": "storm2", "u": 100, "beta": 1.0, "rho": 0.25,
        "eps": 0.0, "theta": 1.0, "mechanism": "targeted", "family": "F-NL",
        "master_seed": 12345, "channel": "full_class_shift"}
SEEDS = list(range(301, 311))
CELL_KEY = "c9_storm2_u100_b1.00_r0.25"
SWEEP = os.path.join(_ROOT, "train_log", "y3_sweep")
CACHE = os.path.join(_ROOT, "results", "y3_p4", "cache")

N_TRAIN, N_PROBE, N_EVAL = 16, 4, 10       # eval = files[20:30]

# Deciders reused from the committed m0-grid cache.
CACHE_KEYS = ["rule", "m0_alone", "oracle", "rule_sup", "m0_sup"]
# Full ladder (M1 rows are computed here).
LADDER = ["rule", "m0_alone", "oracle", "m1_alone", "rule_sup", "m0_sup", "m1_sup"]


# --------------------------------------------------------------------------- #
# Committed m0-grid cache loader                                              #
# --------------------------------------------------------------------------- #
def load_cache():
    """Return {seed: cache_dict} for the primary cell, matched on cell keys."""
    match = dict(campus=CELL["campus"], regime=CELL["regime"], u=CELL["u"],
                 beta=CELL["beta"], rho=CELL["rho"], channel=CELL["channel"])
    out = {}
    for p in glob.glob(os.path.join(CACHE, "*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if all(d.get(k) == v for k, v in match.items()) and d.get("eps", 0.0) == CELL["eps"]:
            out[d["seed"]] = d
    return out


# --------------------------------------------------------------------------- #
# Per-seed M1 evaluation (ALONE + IN-LOOP), one worker                        #
# --------------------------------------------------------------------------- #
def eval_seed(seed):
    """Evaluate the fair-M1 checkpoint for `seed` on the 10 held-out instances,
    ALONE (M1-FROZEN) and IN-LOOP (M1+SUP, rho=0.25). Returns per-instance TWT*
    plus the override-rate trajectory from metrics.csv (H3)."""
    torch.set_num_threads(1)
    t0 = time.perf_counter()

    files = P3.cell_files(CELL["campus"], CELL["u"], CELL["regime"])
    eval_start = N_TRAIN + N_PROBE
    eval_files = files[eval_start:eval_start + N_EVAL]
    eval_insts = [P3._load(p) for p in eval_files]
    inst_ids = [inst["meta"]["id"] for inst in eval_insts]

    overlay = ov.Overlay(ov.OverlayParams(
        beta=CELL["beta"], family=CELL["family"],
        master_seed=CELL["master_seed"], channel=CELL["channel"]))
    assert overlay.params.channel == CELL["channel"]

    ckpt = os.path.join(SWEEP, "m1_c9_u100_b1_r0.25_s%d" % seed, "final.pt")
    m1 = LatentDispatchPolicy.load(ckpt).to("cpu")
    m1.eval()
    gate = float(m1.gate)
    nparam = int(sum(p.numel() for p in m1.parameters()))
    assert gate == 1.0, "M1 gate must be 1.0 (fair-M1)"
    assert bool(getattr(m1, "use_deadline_head")), "fair-M1 must have deadline_head=True"
    assert nparam == 14276, "fair-M1 param count drift: %d != 14276" % nparam
    assert m1.correction_mode == CELL["channel"]

    m1_alone, m1_sup = [], []
    sup_summ = []
    for inst in eval_insts:
        applied = overlay.apply(inst)

        def sc(sched):
            return TO.score_true(inst, sched, overlay, applied)["TWT_true"]

        m1_alone.append(sc(P3.rollout_policy_alone(m1, inst)))
        ms, mstat = P3.rollout_policy_sup(m1, inst, overlay, applied, CELL, seed)
        m1_sup.append(sc(ms))
        sup_summ.append(mstat)

    # H3: M1 DAgger override-rate trajectory per outer iter, from metrics.csv.
    mpath = os.path.join(SWEEP, "m1_c9_u100_b1_r0.25_s%d" % seed, "metrics.csv")
    traj = None
    if os.path.exists(mpath):
        rows = list(csv.DictReader(open(mpath)))
        traj = [{"iter": int(r["iter"]),
                 "override_rate": float(r["override_rate"])} for r in rows]

    return {
        "seed": seed, "inst_ids": inst_ids,
        "m1_ckpt": ckpt, "gate": gate, "nparam": nparam,
        "m1_alone": m1_alone, "m1_sup": m1_sup,
        "m1_sup_revfrac": float(np.mean([s["reviewed_fraction"] for s in sup_summ])),
        "m1_sup_orr": float(np.mean([s["override_rate_of_reviews"] for s in sup_summ])),
        "override_traj": traj,
        "secs": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
def contrast(mat, test, comp):
    """Paired contrast test vs comp on the per-seed x per-instance matrices.

    Returns three views:
      seedavg   -- average each instance over seeds (n=10 instance means), the
                   committed m0-grid method ("seed-averaged per-instance Wilcoxon").
      pooled    -- all seed x instance pairs (n=100).
      per_seed  -- fraction of seeds whose seed-mean(test) < seed-mean(comp).
    Lower TWT* = better = a WIN for `test`.
    """
    A = np.asarray(mat[test], float)    # (n_seed, n_inst)
    B = np.asarray(mat[comp], float)
    # seed-averaged per instance
    a_i, b_i = A.mean(axis=0), B.mean(axis=0)
    sa = {"test_mean": float(a_i.mean()), "comp_mean": float(b_i.mean()),
          "delta_mean": float(a_i.mean() - b_i.mean()),
          "pct_gain": 100.0 * (b_i.mean() - a_i.mean()) / b_i.mean(),
          "wtl": P3.win_tie_loss(a_i, b_i),
          "wilcoxon_p": P3.paired_wilcoxon(a_i, b_i), "n": int(len(a_i))}
    # pooled
    a_f, b_f = A.reshape(-1), B.reshape(-1)
    pl = {"wtl": P3.win_tie_loss(a_f, b_f),
          "wilcoxon_p": P3.paired_wilcoxon(a_f, b_f), "n": int(len(a_f))}
    # per-seed win fraction (seed-mean test beats seed-mean comp)
    tm, cm = A.mean(axis=1), B.mean(axis=1)
    seed_wins = int(np.sum(tm < cm - P3._TOL))
    ps = {"seeds_test_beats_comp": seed_wins, "n_seeds": int(len(tm)),
          "frac": seed_wins / len(tm)}
    return {"test": test, "comp": comp, "seedavg": sa, "pooled": pl, "per_seed": ps}


def holm(pdict):
    """Holm step-down on {name: raw_p}. Returns {name: holm_p} with monotonicity."""
    items = sorted(pdict.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        run = max(run, adj)      # enforce monotone non-decreasing
        out[name] = run
    return out


def seed_stats(mat, key):
    """Seed MEAN +/- STD of the per-seed instance-means (never best-of-seeds)."""
    per_seed_mean = np.asarray(mat[key], float).mean(axis=1)   # (n_seed,)
    return {"twt_mean": float(per_seed_mean.mean()),
            "twt_std_pop": float(per_seed_mean.std(ddof=0)),
            "twt_std_sample": float(per_seed_mean.std(ddof=1)),
            "per_seed_mean": [float(x) for x in per_seed_mean],
            "n_seeds": int(len(per_seed_mean))}


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_ROOT, "results", "y3_p5", "harvest"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- committed m0-grid cache + equivalence check ---------------------- #
    cache = load_cache()
    assert set(cache) == set(SEEDS), "cache seeds %s != %s" % (sorted(cache), SEEDS)
    base_ids = cache[SEEDS[0]]["inst_ids"]
    for s in SEEDS:
        assert cache[s]["inst_ids"] == base_ids, "cache inst order drift seed %d" % s
        assert cache[s]["n_wos"] == 2253

    comm = json.load(open(os.path.join(_ROOT, "results", "y3_p4",
                     "m0_gate_summary.json")))["cells"][CELL_KEY]["ladder"]
    equiv = {}
    for k in CACHE_KEYS:
        seed_means = np.array([np.mean(cache[s]["per"][k]) for s in SEEDS])
        got, want = float(seed_means.mean()), comm[k]["twt_mean"]
        assert abs(got - want) < 1e-6, "m0-grid drift %s: %.6f != %.6f" % (k, got, want)
        equiv[k] = {"harvest_seed_mean": got, "committed_mean": want}
    print("[harvest] m0-grid equivalence PASS (RULE/M0/ORACLE/RULE+SUP/M0+SUP "
          "seed-means == committed m0_gate_summary.json)")

    # ---- evaluate 10 M1 checkpoints (parallel, capped, 1 thread each) ------ #
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        m1res = {r["seed"]: r for r in ex.map(eval_seed, SEEDS)}
    print("[harvest] 10 M1 evals done in %.1fs (workers=%d)"
          % (time.perf_counter() - t0, args.workers))
    for s in SEEDS:
        assert m1res[s]["inst_ids"] == base_ids, "M1 eval inst mismatch seed %d" % s

    # ---- assemble per-seed x per-instance matrices ------------------------ #
    n_inst = len(base_ids)
    mat = {k: np.zeros((len(SEEDS), n_inst)) for k in LADDER}
    for si, s in enumerate(SEEDS):
        for k in CACHE_KEYS:
            mat[k][si] = cache[s]["per"][k]
        mat["m1_alone"][si] = m1res[s]["m1_alone"]
        mat["m1_sup"][si] = m1res[s]["m1_sup"]

    # ---- per-instance CSV -------------------------------------------------- #
    csv_path = os.path.join(args.out, "primary_multiseed.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "inst_id"] + LADDER)
        for si, s in enumerate(SEEDS):
            for ii, iid in enumerate(base_ids):
                w.writerow([s, iid] + ["%.6f" % mat[k][si, ii] for k in LADDER])
    print("[harvest] wrote %s" % csv_path)

    # ---- ladder (seed mean +/- std) --------------------------------------- #
    ladder = {k: seed_stats(mat, k) for k in LADDER}
    rule_mean = ladder["rule"]["twt_mean"]
    for k in LADDER:
        ladder[k]["pct_below_rule"] = 100.0 * (rule_mean - ladder[k]["twt_mean"]) / rule_mean

    # ---- headline contrasts (Holm family = these 4) ----------------------- #
    C = {
        "H1_M1sup_vs_RULEsup": contrast(mat, "m1_sup", "rule_sup"),
        "H2_M1alone_vs_RULE":  contrast(mat, "m1_alone", "rule"),
        "M1alone_vs_M0alone":  contrast(mat, "m1_alone", "m0_alone"),
        "M1sup_vs_M0sup":      contrast(mat, "m1_sup", "m0_sup"),
    }
    # supporting (not in Holm family)
    S = {
        "M1sup_vs_M1alone":   contrast(mat, "m1_sup", "m1_alone"),
        "M0alone_vs_RULE":    contrast(mat, "m0_alone", "rule"),
        "RULEsup_vs_RULE":    contrast(mat, "rule_sup", "rule"),
    }
    holm_raw = {k: v["seedavg"]["wilcoxon_p"] for k, v in C.items()}
    holm_p = holm(holm_raw)
    for k in C:
        C[k]["raw_p_seedavg"] = holm_raw[k]
        C[k]["holm_p"] = holm_p[k]

    # ---- H3 override-rate trajectory (mean across seeds) ------------------- #
    trajs = [m1res[s]["override_traj"] for s in SEEDS]
    n_iter = len(trajs[0])
    orr_mat = np.array([[t[i]["override_rate"] for i in range(n_iter)] for t in trajs])
    h3 = {
        "n_iter": n_iter,
        "mean_trajectory": [float(x) for x in orr_mat.mean(axis=0)],
        "std_trajectory": [float(x) for x in orr_mat.std(axis=0, ddof=0)],
        "per_seed_first": [float(orr_mat[i, 0]) for i in range(len(SEEDS))],
        "per_seed_last": [float(orr_mat[i, -1]) for i in range(len(SEEDS))],
        "mean_first": float(orr_mat[:, 0].mean()),
        "mean_last": float(orr_mat[:, -1].mean()),
        "falls_all_seeds": bool(np.all(orr_mat[:, -1] < orr_mat[:, 0])),
        "rule_sup_override_rate_flat_mean": float(np.mean(
            [np.mean(cache[s]["per"].get("rule_sup_orr", cache[s].get("rule_sup_orr", [np.nan])))
             for s in SEEDS])) if "rule_sup_orr" in cache[SEEDS[0]] else None,
    }
    # RULE+SUP flat override rate lives at top level of the cache dict
    rs_orr = [np.mean(cache[s]["rule_sup_orr"]) for s in SEEDS if "rule_sup_orr" in cache[s]]
    h3["rule_sup_override_rate_flat_mean"] = float(np.mean(rs_orr)) if rs_orr else None

    # ---- supervisor budget parity ----------------------------------------- #
    budget = {
        "m1_sup": {"review_fraction_mean": float(np.mean([m1res[s]["m1_sup_revfrac"] for s in SEEDS])),
                   "override_rate_mean": float(np.mean([m1res[s]["m1_sup_orr"] for s in SEEDS]))},
        "rule_sup": {"review_fraction_mean": float(np.mean([np.mean(cache[s]["rule_sup_revfrac"]) for s in SEEDS])),
                     "override_rate_mean": float(np.mean([np.mean(cache[s]["rule_sup_orr"]) for s in SEEDS]))},
        "m0_sup": {"review_fraction_mean": float(np.mean([np.mean(cache[s]["m0_sup_revfrac"]) for s in SEEDS])),
                   "override_rate_mean": float(np.mean([np.mean(cache[s]["m0_sup_orr"]) for s in SEEDS]))},
    }

    # ---- verdicts ---------------------------------------------------------- #
    verdict = {
        "H1_M1sup_beats_RULEsup": bool(ladder["m1_sup"]["twt_mean"] < ladder["rule_sup"]["twt_mean"]
                                       and C["H1_M1sup_vs_RULEsup"]["holm_p"] < 0.05),
        "H2_M1alone_beats_RULE": bool(ladder["m1_alone"]["twt_mean"] < ladder["rule"]["twt_mean"]
                                      and C["H2_M1alone_vs_RULE"]["holm_p"] < 0.05),
        "H3_override_rate_falls": h3["falls_all_seeds"],
        "M0_dominates_M1_alone": bool(ladder["m0_alone"]["twt_mean"] < ladder["m1_alone"]["twt_mean"]),
        "M0_dominates_M1_sup": bool(ladder["m0_sup"]["twt_mean"] < ladder["m1_sup"]["twt_mean"]),
        "pi0_attribution": "PENDING (PI-0 seeds still training; not evaluated)",
        "headline_holds_10_seeds": None,   # filled below
    }
    verdict["headline_holds_10_seeds"] = bool(
        verdict["H1_M1sup_beats_RULEsup"] and verdict["H2_M1alone_beats_RULE"]
        and verdict["H3_override_rate_falls"] and verdict["M0_dominates_M1_alone"]
        and verdict["M0_dominates_M1_sup"])

    summary = {
        "cell": CELL, "cell_key": CELL_KEY, "seeds": SEEDS, "n_eval": n_inst,
        "eval_inst_ids": base_ids, "n_wos": cache[SEEDS[0]]["n_wos"],
        "m1_ckpts": {s: m1res[s]["m1_ckpt"] for s in SEEDS},
        "m1_gate": 1.0, "m1_nparam": 14276, "m1_deadline_head": True,
        "std_convention": "twt_std_pop = population std over 10 seeds (matches committed m0-grid); twt_std_sample = ddof=1",
        "m0_grid_equivalence": equiv,
        "ladder": ladder,
        "contrasts_holm_family": C,
        "holm_raw_p": holm_raw,
        "holm_p": holm_p,
        "contrasts_supporting": S,
        "H3_override_rate": h3,
        "supervisor_budget": budget,
        "verdict": verdict,
        "notes": "TWT*(w*,d*) independent validator; held-out files[20:30]; "
                 "M0/RULE/ORACLE/RULE+SUP/M0+SUP reused from results/y3_p4 cache (not recomputed); "
                 "fair-M1 deadline_head=True is the standard M1.",
    }
    json_path = os.path.join(args.out, "primary_multiseed_summary.json")
    json.dump(summary, open(json_path, "w"), indent=1, default=str)
    print("[harvest] wrote %s" % json_path)

    _print_console(summary)
    _write_md(summary, os.path.join(_ROOT, "notes", "harvest_primary.md"))
    return summary


def _pm(st):
    return "%.1f +/- %.1f" % (st["twt_mean"], st["twt_std_pop"])


def _print_console(s):
    la, C = s["ladder"], s["contrasts_holm_family"]
    print("\n===== PRIMARY CELL %s  TWT*(d*), 10 seeds, n=%d held-out =====" % (s["cell_key"], s["n_eval"]))
    print("%-14s %20s %12s" % ("decider", "TWT* (mean+/-std)", "%below RULE"))
    for k in LADDER:
        print("%-14s %20s %11.1f%%" % (k, _pm(la[k]), la[k]["pct_below_rule"]))
    print("\n----- Holm-family contrasts (seed-avg per-instance Wilcoxon) -----")
    for k, v in C.items():
        sa = v["seedavg"]
        print("  %-22s %8.1f vs %8.1f  (%+.1f%%)  W/T/L %d/%d/%d  raw_p=%.4g  holm_p=%.4g  seeds_beat=%d/10"
              % (k, sa["test_mean"], sa["comp_mean"], sa["pct_gain"],
                 sa["wtl"]["W"], sa["wtl"]["T"], sa["wtl"]["L"],
                 v["raw_p_seedavg"], v["holm_p"], v["per_seed"]["seeds_test_beats_comp"]))
    h3 = s["H3_override_rate"]
    print("\n----- H3 M1 override-rate trajectory (mean across 10 seeds) -----")
    print("  " + " ".join("%.4f" % x for x in h3["mean_trajectory"]))
    print("  first=%.4f last=%.4f falls_all_seeds=%s  (RULE+SUP flat=%.4f)"
          % (h3["mean_first"], h3["mean_last"], h3["falls_all_seeds"],
             h3["rule_sup_override_rate_flat_mean"] or float("nan")))
    print("\n----- VERDICT -----")
    for k, v in s["verdict"].items():
        print("  %-28s %s" % (k, v))


def _write_md(s, path):
    la, C = s["ladder"], s["contrasts_holm_family"]
    h3, vd = s["H3_override_rate"], s["verdict"]
    L = []
    L.append("# Y3 P5 primary-cell multi-seed harvest\n")
    L.append("Cell **%s**: c9 storm2 u100, beta=1.0, rho=0.25, eps=0, TARGETED, "
             "channel=full_class_shift, seeds 301-310 (10 done). "
             "Objective TWT*(w*,d*), independent validator, held-out n=%d (files[20:30]).\n"
             % (s["cell_key"], s["n_eval"]))
    L.append("fair-M1 = deadline_head=True standard M1 (gate=1.0, %d params). "
             "M0/RULE/ORACLE/RULE+SUP/M0+SUP reused from the committed results/y3_p4 cache "
             "(seed-means verified bit-equal to m0_gate_summary.json). No training; NO M0 recompute.\n"
             % s["m1_nparam"])
    L.append("STD = population std over 10 seeds (matches the committed m0-grid convention).\n")

    L.append("## Ladder (seed mean +/- std)\n")
    L.append("| decider | TWT* | %below RULE |")
    L.append("|---|---|---|")
    for k in LADDER:
        L.append("| %s | %.1f +/- %.1f | %.1f%% |" % (k, la[k]["twt_mean"], la[k]["twt_std_pop"], la[k]["pct_below_rule"]))
    L.append("")

    L.append("## Holm-family contrasts (seed-averaged per-instance paired Wilcoxon, n=10 instances)\n")
    L.append("| contrast | test | comp | %gain | W/T/L | seeds test-beats-comp | raw p | Holm p |")
    L.append("|---|---|---|---|---|---|---|---|")
    lab = {"H1_M1sup_vs_RULEsup": "H1 M1+SUP vs RULE+SUP",
           "H2_M1alone_vs_RULE": "H2 M1-FROZEN vs RULE",
           "M1alone_vs_M0alone": "M1 alone vs M0 alone",
           "M1sup_vs_M0sup": "M1+SUP vs M0+SUP"}
    for k, v in C.items():
        sa = v["seedavg"]
        L.append("| %s | %.1f | %.1f | %+.1f%% | %d/%d/%d | %d/10 | %.4g | %.4g |"
                 % (lab[k], sa["test_mean"], sa["comp_mean"], sa["pct_gain"],
                    sa["wtl"]["W"], sa["wtl"]["T"], sa["wtl"]["L"],
                    v["per_seed"]["seeds_test_beats_comp"], v["raw_p_seedavg"], v["holm_p"]))
    L.append("")
    L.append("Pooled (n=100 seed x instance pairs), same contrasts:\n")
    L.append("| contrast | W/T/L | pooled p |")
    L.append("|---|---|---|")
    for k, v in C.items():
        pl = v["pooled"]
        L.append("| %s | %d/%d/%d | %.4g |" % (lab[k], pl["wtl"]["W"], pl["wtl"]["T"], pl["wtl"]["L"], pl["wilcoxon_p"]))
    L.append("")

    L.append("## H3 override-rate trajectory (mean across 10 seeds, DAgger iters 0-7)\n")
    L.append("`" + " ".join("%.4f" % x for x in h3["mean_trajectory"]) + "`\n")
    L.append("First iter %.4f -> last iter %.4f; falls on all 10 seeds: **%s**. "
             "RULE+SUP flat override-rate %.4f (a fixed rule never learns).\n"
             % (h3["mean_first"], h3["mean_last"], h3["falls_all_seeds"],
                h3["rule_sup_override_rate_flat_mean"] or float("nan")))

    L.append("## Supervisor budget parity\n")
    b = s["supervisor_budget"]
    L.append("| mode | review_frac | override_rate |")
    L.append("|---|---|---|")
    for k in ["rule_sup", "m0_sup", "m1_sup"]:
        L.append("| %s | %.3f | %.3f |" % (k, b[k]["review_fraction_mean"], b[k]["override_rate_mean"]))
    L.append("")

    L.append("## Verdict\n")
    for k, v in vd.items():
        L.append("- **%s**: %s" % (k, v))
    L.append("")
    L.append("PI-0 attribution (PI-0+SUP vs M1+SUP) is PENDING: the PI-0 seeds are still "
             "training in the sweep and were not evaluated here.")
    open(path, "w").write("\n".join(L) + "\n")
    print("[harvest] wrote %s" % path)


if __name__ == "__main__":
    main()
