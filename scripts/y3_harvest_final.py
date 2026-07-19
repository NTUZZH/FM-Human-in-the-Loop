#!/usr/bin/env python
"""P5 FINAL harvest (Paper Y3): PI-0 multi-seed attribution + M1 regime robustness.

Two additive harvests on top of the committed primary multi-seed harvest
(results/y3_p5/harvest/primary_multiseed_summary.json, fair-M1 10 seeds), reusing
the frozen-decider eval harness from scripts/y3_p3_eval.py verbatim and the
committed results/y3_p4 M0-grid cache (never recomputing M0/RULE/ORACLE):

  (A) PI-0 multi-seed attribution -- the 10 PI-0 checkpoints (c9 u100, seeds
      301-310) in ALONE and IN-LOOP mode on TWT*(w*,d*), held-out files[20:30].
      The fair-M1 per-instance values (rule/rule_sup/m1_alone/m1_sup/m0*) are read
      from the committed primary_multiseed.csv (NOT recomputed). Attribution
      family (Holm over 3): PI-0 alone vs RULE, PI-0+SUP vs RULE+SUP, and the key
      contrast fair-M1+SUP vs PI-0+SUP (isolating learning-from-overrides). The 3
      PI-0 u90 seeds give the lower-load check.

  (B) M1 regime robustness -- the 6 regime M1 checkpoints (u90 b1.0 s301-303;
      u100 b0.75 s301-303) ALONE and +SUP vs their cells' RULE/RULE+SUP/M0 from
      the committed cache (seeds 301-303). Descriptive H1/H2 + M1-vs-M0 at 3 seeds.

Run (coexistence: cap workers, OMP=1/worker, nice):
  OMP_NUM_THREADS=1 nice -n 15 python scripts/y3_harvest_final.py --workers 5
"""

from __future__ import annotations

import os

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

import y3_p3_eval as P3                                             # noqa: E402
from y3_harvest_primary import contrast, holm, seed_stats          # noqa: E402
from fmwos.hitl import overlay as ov                               # noqa: E402
from fmwos.hitl import true_objective as TO                        # noqa: E402
from fmwos.hitl.latent_head import LatentDispatchPolicy            # noqa: E402

SWEEP = os.path.join(_ROOT, "results", "y3_checkpoints", "sweep")
CACHE = os.path.join(_ROOT, "results", "y3_p4", "cache")
PRIMARY_CSV = os.path.join(_ROOT, "results", "y3_p5", "harvest", "primary_multiseed.csv")
N_TRAIN, N_PROBE, N_EVAL = 16, 4, 10       # eval = files[20:30]

FAMILY = "F-NL"
MASTER_SEED = 12345
CHANNEL = "full_class_shift"
REGIME = "storm2"
THETA = 1.0
MECH = "targeted"
FAIR_M1_NPARAM = 14276
PI0_NPARAM = 12323


# --------------------------------------------------------------------------- #
# General committed-cache loader (any cell)                                    #
# --------------------------------------------------------------------------- #
def load_cache_cell(campus, u, beta, rho, eps, seeds):
    """{seed: cache_dict} for the requested cell, matched on cell keys."""
    match = dict(campus=campus, regime=REGIME, u=u, beta=beta, rho=rho,
                 channel=CHANNEL)
    out = {}
    for p in glob.glob(os.path.join(CACHE, "*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if all(d.get(k) == v for k, v in match.items()) and d.get("eps", 0.0) == eps:
            if d["seed"] in seeds:
                out[d["seed"]] = d
    missing = set(seeds) - set(out)
    assert not missing, "cache miss u%d b%.2f seeds %s" % (u, beta, sorted(missing))
    return out


# --------------------------------------------------------------------------- #
# One frozen-policy eval (ALONE + IN-LOOP), one worker                        #
# --------------------------------------------------------------------------- #
def eval_policy_cell(job):
    """Evaluate one frozen policy checkpoint on the 10 held-out instances of its
    cell, ALONE and IN-LOOP (rho/eps as in job). Returns per-instance TWT*."""
    torch.set_num_threads(1)
    t0 = time.perf_counter()

    files = P3.cell_files(job["campus"], job["u"], REGIME)
    eval_start = N_TRAIN + N_PROBE
    eval_files = files[eval_start:eval_start + N_EVAL]
    eval_insts = [P3._load(p) for p in eval_files]
    inst_ids = [inst["meta"]["id"] for inst in eval_insts]

    overlay = ov.Overlay(ov.OverlayParams(
        beta=job["beta"], family=FAMILY, master_seed=MASTER_SEED, channel=CHANNEL))
    assert overlay.params.channel == CHANNEL

    pol = LatentDispatchPolicy.load(job["ckpt"]).to("cpu")
    pol.eval()
    gate = float(pol.gate)
    nparam = int(sum(p.numel() for p in pol.parameters()))

    cell = {"rho": job["rho"], "eps": job["eps"], "theta": THETA, "mechanism": MECH}
    alone, sup, sup_summ = [], [], []
    for inst in eval_insts:
        applied = overlay.apply(inst)

        def sc(sched):
            return TO.score_true(inst, sched, overlay, applied)["TWT_true"]

        alone.append(sc(P3.rollout_policy_alone(pol, inst)))
        ms, mstat = P3.rollout_policy_sup(pol, inst, overlay, applied, cell, job["seed"])
        sup.append(sc(ms))
        sup_summ.append(mstat)

    traj = None
    mpath = os.path.join(os.path.dirname(job["ckpt"]), "metrics.csv")
    if os.path.exists(mpath):
        rows = list(csv.DictReader(open(mpath)))
        try:
            traj = [{"iter": int(r["iter"]),
                     "override_rate": float(r["override_rate"])} for r in rows]
        except Exception:
            traj = None

    return {
        "tag": job["tag"], "seed": job["seed"], "inst_ids": inst_ids,
        "ckpt": job["ckpt"], "gate": gate, "nparam": nparam,
        "alone": alone, "sup": sup,
        "sup_revfrac": float(np.mean([s["reviewed_fraction"] for s in sup_summ])),
        "sup_orr": float(np.mean([s["override_rate_of_reviews"] for s in sup_summ])),
        "override_traj": traj, "secs": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- #
# Read the committed fair-M1 per-instance matrices from primary_multiseed.csv  #
# --------------------------------------------------------------------------- #
def load_primary_csv(seeds):
    """Return (base_ids, {key: (n_seed,n_inst) array}) for the committed fair-M1
    primary harvest. Keys: rule,m0_alone,oracle,m1_alone,rule_sup,m0_sup,m1_sup."""
    rows = list(csv.DictReader(open(PRIMARY_CSV)))
    keys = ["rule", "m0_alone", "oracle", "m1_alone", "rule_sup", "m0_sup", "m1_sup"]
    by_seed = {}
    for r in rows:
        s = int(r["seed"])
        by_seed.setdefault(s, []).append(r)
    assert set(by_seed) == set(seeds), "primary_multiseed.csv seeds %s != %s" % (
        sorted(by_seed), seeds)
    base_ids = [r["inst_id"] for r in by_seed[seeds[0]]]
    mat = {k: np.zeros((len(seeds), len(base_ids))) for k in keys}
    for si, s in enumerate(seeds):
        srows = by_seed[s]
        assert [r["inst_id"] for r in srows] == base_ids, "csv inst order drift s%d" % s
        for ii, r in enumerate(srows):
            for k in keys:
                mat[k][si, ii] = float(r[k])
    return base_ids, mat


def cache_mat(cache, seeds, base_ids):
    """(n_seed,n_inst) arrays for rule/m0_alone/oracle/rule_sup/m0_sup from cache."""
    keys = ["rule", "m0_alone", "oracle", "rule_sup", "m0_sup"]
    mat = {k: np.zeros((len(seeds), len(base_ids))) for k in keys}
    for si, s in enumerate(seeds):
        assert cache[s]["inst_ids"] == base_ids, "cache inst order drift s%d" % s
        for k in keys:
            mat[k][si] = cache[s]["per"][k]
    return mat


def pct_std_points(mat, key, rule_mean):
    """Seed s.d. of a decider's TWT expressed in percentage points of RULE."""
    st = seed_stats(mat, key)
    return 100.0 * st["twt_std_pop"] / rule_mean


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

    PI0_SEEDS = list(range(301, 311))
    U90_SEEDS = [301, 302, 303]

    # ---- build the eval job list ------------------------------------------ #
    jobs = []
    for s in PI0_SEEDS:
        jobs.append(dict(tag="pi0_u100", kind="pi0", campus=9, u=100, beta=1.0,
                         rho=0.25, eps=0.0, seed=s,
                         ckpt=os.path.join(SWEEP, "pi0_c9_u100_s%d" % s, "final.pt")))
    for s in U90_SEEDS:
        jobs.append(dict(tag="pi0_u90", kind="pi0", campus=9, u=90, beta=1.0,
                         rho=0.25, eps=0.0, seed=s,
                         ckpt=os.path.join(SWEEP, "pi0_c9_u90_s%d" % s, "final.pt")))
    for s in U90_SEEDS:
        jobs.append(dict(tag="m1_u90_b1", kind="m1", campus=9, u=90, beta=1.0,
                         rho=0.25, eps=0.0, seed=s,
                         ckpt=os.path.join(SWEEP, "m1_c9_u90_b1_r0.25_s%d" % s, "final.pt")))
    for s in U90_SEEDS:
        jobs.append(dict(tag="m1_u100_b0.75", kind="m1", campus=9, u=100, beta=0.75,
                         rho=0.25, eps=0.0, seed=s,
                         ckpt=os.path.join(SWEEP, "m1_c9_u100_b0.75_r0.25_s%d" % s, "final.pt")))

    for j in jobs:
        assert os.path.exists(j["ckpt"]), "missing checkpoint %s" % j["ckpt"]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(eval_policy_cell, jobs))
    print("[final] %d policy evals done in %.1fs (workers=%d)"
          % (len(res), time.perf_counter() - t0, args.workers))

    by = {}
    for r in res:
        by.setdefault(r["tag"], {})[r["seed"]] = r

    # sanity: PI-0 gate 0 + param count; regime M1 gate 1 + fair param count
    for s in PI0_SEEDS:
        assert by["pi0_u100"][s]["gate"] == 0.0, "PI-0 gate != 0 (s%d)" % s
        assert by["pi0_u100"][s]["nparam"] == PI0_NPARAM, "PI-0 param drift s%d" % s
    for s in U90_SEEDS:
        assert by["pi0_u90"][s]["gate"] == 0.0
        assert by["pi0_u90"][s]["nparam"] == PI0_NPARAM
        for tag in ("m1_u90_b1", "m1_u100_b0.75"):
            assert by[tag][s]["gate"] == 1.0, "%s gate != 1 s%d" % (tag, s)
            assert by[tag][s]["nparam"] == FAIR_M1_NPARAM, "%s param drift s%d" % (tag, s)

    summary = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "notes": "TWT*(w*,d*) independent validator; held-out files[20:30]; "
                        "M0/RULE/ORACLE/RULE+SUP/M0+SUP reused from committed "
                        "results/y3_p4 cache and results/y3_p5 primary harvest (NOT recomputed). "
                        "PI-0 gate=0 (no override head, no deadline head, %d params); "
                        "fair-M1 deadline_head=True (%d params)." % (PI0_NPARAM, FAIR_M1_NPARAM),
               "std_convention": "twt_std_pop = population std over seeds; "
                                 "pct-point std = twt_std_pop / RULE_seed_mean * 100"}

    # ====================================================================== #
    # (A) PI-0 multi-seed attribution (u100, 10 seeds)                        #
    # ====================================================================== #
    base_ids, pmat = load_primary_csv(PI0_SEEDS)
    # attach PI-0 rows (verify inst order matches the committed primary harvest)
    pi0_alone = np.zeros((len(PI0_SEEDS), len(base_ids)))
    pi0_sup = np.zeros((len(PI0_SEEDS), len(base_ids)))
    for si, s in enumerate(PI0_SEEDS):
        r = by["pi0_u100"][s]
        assert r["inst_ids"] == base_ids, "PI-0 inst order != primary harvest s%d" % s
        pi0_alone[si] = r["alone"]
        pi0_sup[si] = r["sup"]
    A = dict(pmat)
    A["pi0_alone"] = pi0_alone
    A["pi0_sup"] = pi0_sup

    rule_mean = float(A["rule"].mean(axis=0).mean())

    # attribution family (Holm over these 3)
    fam = {
        "PI0alone_vs_RULE": contrast(A, "pi0_alone", "rule"),
        "PI0sup_vs_RULEsup": contrast(A, "pi0_sup", "rule_sup"),
        "M1sup_vs_PI0sup": contrast(A, "m1_sup", "pi0_sup"),   # THE attribution
    }
    fam_raw = {k: v["seedavg"]["wilcoxon_p"] for k, v in fam.items()}
    fam_holm = holm(fam_raw)
    for k in fam:
        fam[k]["raw_p_seedavg"] = fam_raw[k]
        fam[k]["holm_p"] = fam_holm[k]

    support = {
        "M1alone_vs_PI0alone": contrast(A, "m1_alone", "pi0_alone"),
        "PI0sup_vs_PI0alone": contrast(A, "pi0_sup", "pi0_alone"),
        "M0sup_vs_PI0sup": contrast(A, "m0_sup", "pi0_sup"),
    }

    pi0_ladder = {}
    for k in ["rule", "pi0_alone", "m1_alone", "m0_alone", "rule_sup",
              "pi0_sup", "m1_sup", "m0_sup", "oracle"]:
        st = seed_stats(A, k)
        st["pct_below_rule"] = 100.0 * (rule_mean - st["twt_mean"]) / rule_mean
        st["pct_std_points"] = 100.0 * st["twt_std_pop"] / rule_mean
        pi0_ladder[k] = st

    attribution = {
        "cell": "c9_storm2_u100_b1.00_r0.25_eps0", "seeds": PI0_SEEDS,
        "n_eval": len(base_ids), "eval_inst_ids": base_ids,
        "pi0_ckpts": {s: by["pi0_u100"][s]["ckpt"] for s in PI0_SEEDS},
        "pi0_gate": 0.0, "pi0_nparam": PI0_NPARAM,
        "pi0_sup_review_fraction_mean": float(np.mean(
            [by["pi0_u100"][s]["sup_revfrac"] for s in PI0_SEEDS])),
        "pi0_sup_override_rate_mean": float(np.mean(
            [by["pi0_u100"][s]["sup_orr"] for s in PI0_SEEDS])),
        "ladder": pi0_ladder,
        "attribution_family_holm": fam,
        "family_raw_p": fam_raw, "family_holm_p": fam_holm,
        "supporting": support,
        "verdict": {
            "PI0_alone_worse_than_RULE": bool(pi0_ladder["pi0_alone"]["twt_mean"]
                                              > pi0_ladder["rule"]["twt_mean"]),
            "M1sup_beats_PI0sup": bool(pi0_ladder["m1_sup"]["twt_mean"]
                                       < pi0_ladder["pi0_sup"]["twt_mean"]),
            "M1sup_vs_PI0sup_holm_p": fam["M1sup_vs_PI0sup"]["holm_p"],
            "attribution_significant_holm_0.05": bool(
                fam["M1sup_vs_PI0sup"]["holm_p"] < 0.05
                and pi0_ladder["m1_sup"]["twt_mean"] < pi0_ladder["pi0_sup"]["twt_mean"]),
        },
    }
    summary["A_pi0_attribution"] = attribution

    # ---- PI-0 u90 lower-load check (3 seeds, descriptive) ----------------- #
    cache90 = load_cache_cell(9, 90, 1.0, 0.25, 0.0, U90_SEEDS)
    ids90 = cache90[U90_SEEDS[0]]["inst_ids"]
    U = cache_mat(cache90, U90_SEEDS, ids90)
    pu_alone = np.zeros((len(U90_SEEDS), len(ids90)))
    pu_sup = np.zeros((len(U90_SEEDS), len(ids90)))
    for si, s in enumerate(U90_SEEDS):
        r = by["pi0_u90"][s]
        assert r["inst_ids"] == ids90, "PI-0 u90 inst order != cache s%d" % s
        pu_alone[si] = r["alone"]
        pu_sup[si] = r["sup"]
    U["pi0_alone"] = pu_alone
    U["pi0_sup"] = pu_sup
    rule90 = float(U["rule"].mean(axis=0).mean())
    pi0_u90 = {
        "cell": "c9_storm2_u90_b1.00_r0.25_eps0", "seeds": U90_SEEDS,
        "n_eval": len(ids90), "descriptive": True,
        "ladder": {k: {**seed_stats(U, k),
                       "pct_below_rule": 100.0 * (rule90 - seed_stats(U, k)["twt_mean"]) / rule90}
                   for k in ["rule", "pi0_alone", "rule_sup", "pi0_sup"]},
        "PI0alone_vs_RULE": contrast(U, "pi0_alone", "rule"),
        "PI0sup_vs_RULEsup": contrast(U, "pi0_sup", "rule_sup"),
    }
    summary["A_pi0_u90_lowload"] = pi0_u90

    # ====================================================================== #
    # (B) M1 regime robustness (3 seeds each, descriptive)                    #
    # ====================================================================== #
    regime = {}
    for tag, u, beta, ckey in [
            ("m1_u90_b1", 90, 1.0, "c9_storm2_u90_b1.00_r0.25"),
            ("m1_u100_b0.75", 100, 0.75, "c9_storm2_u100_b0.75_r0.25")]:
        cache = load_cache_cell(9, u, beta, 0.25, 0.0, U90_SEEDS)
        ids = cache[U90_SEEDS[0]]["inst_ids"]
        M = cache_mat(cache, U90_SEEDS, ids)
        m1a = np.zeros((len(U90_SEEDS), len(ids)))
        m1s = np.zeros((len(U90_SEEDS), len(ids)))
        for si, s in enumerate(U90_SEEDS):
            r = by[tag][s]
            assert r["inst_ids"] == ids, "%s inst order != cache s%d" % (tag, s)
            m1a[si] = r["alone"]
            m1s[si] = r["sup"]
        M["m1_alone"] = m1a
        M["m1_sup"] = m1s
        rmean = float(M["rule"].mean(axis=0).mean())
        lad = {}
        for k in ["rule", "m1_alone", "m0_alone", "rule_sup", "m1_sup", "m0_sup", "oracle"]:
            st = seed_stats(M, k)
            st["pct_below_rule"] = 100.0 * (rmean - st["twt_mean"]) / rmean
            lad[k] = st
        H1 = contrast(M, "m1_sup", "rule_sup")
        H2 = contrast(M, "m1_alone", "rule")
        regime[tag] = {
            "cell": ckey, "u": u, "beta": beta, "seeds": U90_SEEDS,
            "n_eval": len(ids), "descriptive": True,
            "m1_gate": 1.0, "m1_nparam": FAIR_M1_NPARAM,
            "m1_sup_review_fraction_mean": float(np.mean([by[tag][s]["sup_revfrac"] for s in U90_SEEDS])),
            "ladder": lad,
            "H1_M1sup_vs_RULEsup": H1,
            "H2_M1alone_vs_RULE": H2,
            "M1alone_vs_M0alone": contrast(M, "m1_alone", "m0_alone"),
            "M1sup_vs_M0sup": contrast(M, "m1_sup", "m0_sup"),
            "override_traj_mean_last": float(np.mean(
                [by[tag][s]["override_traj"][-1]["override_rate"]
                 for s in U90_SEEDS if by[tag][s]["override_traj"]])),
            "verdict": {
                "H1_holds_qual": bool(lad["m1_sup"]["twt_mean"] < lad["rule_sup"]["twt_mean"]),
                "H2_holds_qual": bool(lad["m1_alone"]["twt_mean"] < lad["rule"]["twt_mean"]),
                "H1_all3seeds": H1["per_seed"]["seeds_test_beats_comp"] == 3,
                "H2_all3seeds": H2["per_seed"]["seeds_test_beats_comp"] == 3,
                "M0_still_dominates_M1_alone": bool(lad["m0_alone"]["twt_mean"] < lad["m1_alone"]["twt_mean"]),
                "M0_still_dominates_M1_sup": bool(lad["m0_sup"]["twt_mean"] < lad["m1_sup"]["twt_mean"]),
            },
        }
    summary["B_m1_regime_robustness"] = regime

    # ---- write ------------------------------------------------------------ #
    jpath = os.path.join(args.out, "final_harvest_summary.json")
    json.dump(summary, open(jpath, "w"), indent=1, default=str)
    print("[final] wrote %s" % jpath)

    _print(summary)
    _write_md(summary, os.path.join(_ROOT, "notes", "harvest_final.md"))
    return summary


def _pm(st):
    return "%.1f +/- %.1f" % (st["twt_mean"], st["twt_std_pop"])


def _print(s):
    a = s["A_pi0_attribution"]
    print("\n===== (A) PI-0 attribution  c9 u100 b1.0 rho0.25 eps0, 10 seeds, n=%d ====="
          % a["n_eval"])
    print("%-12s %20s %12s" % ("decider", "TWT* (mean+/-std)", "%below RULE"))
    for k in ["rule", "pi0_alone", "m1_alone", "m0_alone", "rule_sup",
              "pi0_sup", "m1_sup", "m0_sup", "oracle"]:
        e = a["ladder"][k]
        print("%-12s %20s %11.1f%%" % (k, _pm(e), e["pct_below_rule"]))
    print("\n--- attribution family (seed-avg per-instance Wilcoxon, Holm over 3) ---")
    for k, v in a["attribution_family_holm"].items():
        sa = v["seedavg"]
        print("  %-20s %8.1f vs %8.1f (%+.1f%%) W/T/L %d/%d/%d seeds=%d/10 raw=%.4g holm=%.4g"
              % (k, sa["test_mean"], sa["comp_mean"], sa["pct_gain"],
                 sa["wtl"]["W"], sa["wtl"]["T"], sa["wtl"]["L"],
                 v["per_seed"]["seeds_test_beats_comp"], v["raw_p_seedavg"], v["holm_p"]))
    print("  VERDICT:", json.dumps(a["verdict"]))
    u = s["A_pi0_u90_lowload"]
    print("\n--- PI-0 u90 lower-load (3 seeds, descriptive) ---")
    for k in ["rule", "pi0_alone", "rule_sup", "pi0_sup"]:
        e = u["ladder"][k]
        print("  %-10s %20s %+.1f%%" % (k, _pm(e), e["pct_below_rule"]))
    for c in ["PI0alone_vs_RULE", "PI0sup_vs_RULEsup"]:
        sa = u[c]["seedavg"]
        print("  %-18s %+.1f%% W/T/L %d/%d/%d seeds=%d/3 raw=%.4g"
              % (c, sa["pct_gain"], sa["wtl"]["W"], sa["wtl"]["T"], sa["wtl"]["L"],
                 u[c]["per_seed"]["seeds_test_beats_comp"], sa["wilcoxon_p"]))

    print("\n===== (B) M1 regime robustness (3 seeds each, descriptive) =====")
    for tag, R in s["B_m1_regime_robustness"].items():
        print("\n-- %s  %s --" % (tag, R["cell"]))
        for k in ["rule", "m1_alone", "m0_alone", "rule_sup", "m1_sup", "m0_sup", "oracle"]:
            e = R["ladder"][k]
            print("   %-10s %20s %+.1f%%" % (k, _pm(e), e["pct_below_rule"]))
        for c in ["H1_M1sup_vs_RULEsup", "H2_M1alone_vs_RULE", "M1alone_vs_M0alone", "M1sup_vs_M0sup"]:
            sa = R[c]["seedavg"]
            print("   %-20s %+.1f%% W/T/L %d/%d/%d seeds=%d/3 raw=%.4g"
                  % (c, sa["pct_gain"], sa["wtl"]["W"], sa["wtl"]["T"], sa["wtl"]["L"],
                     R[c]["per_seed"]["seeds_test_beats_comp"], sa["wilcoxon_p"]))
        print("   VERDICT:", json.dumps(R["verdict"]))


def _write_md(s, path):
    a = s["A_pi0_attribution"]
    L = ["# Y3 P5 FINAL harvest: PI-0 attribution + M1 regime robustness\n"]
    L.append(s["notes"] + "\n")
    L.append("## (A) PI-0 multi-seed attribution -- c9 storm2 u100 beta1.0 rho0.25 eps0, "
             "seeds 301-310, held-out n=%d\n" % a["n_eval"])
    L.append("PI-0 = blind PPO control, gate=0, %d params (no override head, no deadline "
             "head). fair-M1 rows read from the committed primary_multiseed.csv.\n"
             % a["pi0_nparam"])
    L.append("### Ladder (seed mean +/- std, %below RULE)\n")
    L.append("| decider | TWT* | %below RULE | std (pct-pts) |")
    L.append("|---|---|---|---|")
    for k in ["rule", "pi0_alone", "m1_alone", "m0_alone", "rule_sup",
              "pi0_sup", "m1_sup", "m0_sup", "oracle"]:
        e = a["ladder"][k]
        L.append("| %s | %.1f +/- %.1f | %+.1f%% | %.2f |"
                 % (k, e["twt_mean"], e["twt_std_pop"], e["pct_below_rule"], e["pct_std_points"]))
    L.append("")
    L.append("### Attribution family (seed-averaged per-instance paired Wilcoxon, "
             "Holm over the 3 contrasts)\n")
    L.append("| contrast | test | comp | %gain | W/T/L | seeds test-beats-comp | raw p | Holm p |")
    L.append("|---|---|---|---|---|---|---|---|")
    lab = {"PI0alone_vs_RULE": "PI-0 alone vs RULE",
           "PI0sup_vs_RULEsup": "PI-0+SUP vs RULE+SUP",
           "M1sup_vs_PI0sup": "fair-M1+SUP vs PI-0+SUP (attribution)"}
    for k, v in a["attribution_family_holm"].items():
        sa = v["seedavg"]
        L.append("| %s | %.1f | %.1f | %+.1f%% | %d/%d/%d | %d/10 | %.4g | %.4g |"
                 % (lab[k], sa["test_mean"], sa["comp_mean"], sa["pct_gain"],
                    sa["wtl"]["W"], sa["wtl"]["T"], sa["wtl"]["L"],
                    v["per_seed"]["seeds_test_beats_comp"], v["raw_p_seedavg"], v["holm_p"]))
    L.append("")
    L.append("Pooled (n=100 seed x instance) for the attribution contrast: "
             "M1sup_vs_PI0sup W/T/L %d/%d/%d, pooled p=%.4g.\n"
             % (a["attribution_family_holm"]["M1sup_vs_PI0sup"]["pooled"]["wtl"]["W"],
                a["attribution_family_holm"]["M1sup_vs_PI0sup"]["pooled"]["wtl"]["T"],
                a["attribution_family_holm"]["M1sup_vs_PI0sup"]["pooled"]["wtl"]["L"],
                a["attribution_family_holm"]["M1sup_vs_PI0sup"]["pooled"]["wilcoxon_p"]))
    L.append("PI-0+SUP supervisor budget: review fraction %.3f, override rate %.3f.\n"
             % (a["pi0_sup_review_fraction_mean"], a["pi0_sup_override_rate_mean"]))
    L.append("**Verdict:** " + json.dumps(a["verdict"]) + "\n")

    u = s["A_pi0_u90_lowload"]
    L.append("### PI-0 lower-load check (u90 beta1.0, 3 seeds, descriptive)\n")
    L.append("| decider | TWT* | %below RULE |")
    L.append("|---|---|---|")
    for k in ["rule", "pi0_alone", "rule_sup", "pi0_sup"]:
        e = u["ladder"][k]
        L.append("| %s | %.1f +/- %.1f | %+.1f%% |" % (k, e["twt_mean"], e["twt_std_pop"], e["pct_below_rule"]))
    for c in ["PI0alone_vs_RULE", "PI0sup_vs_RULEsup"]:
        sa = u[c]["seedavg"]
        L.append("- %s: %+.1f%%, W/T/L %d/%d/%d, %d/3 seeds, raw p=%.4g"
                 % (c, sa["pct_gain"], sa["wtl"]["W"], sa["wtl"]["T"], sa["wtl"]["L"],
                    u[c]["per_seed"]["seeds_test_beats_comp"], sa["wilcoxon_p"]))
    L.append("")

    L.append("## (B) M1 regime robustness (H1/H2 off the primary cell, 3 seeds, descriptive)\n")
    for tag, R in s["B_m1_regime_robustness"].items():
        L.append("### %s -- %s (u%d beta%.2f)\n" % (tag, R["cell"], R["u"], R["beta"]))
        L.append("| decider | TWT* | %below RULE |")
        L.append("|---|---|---|")
        for k in ["rule", "m1_alone", "m0_alone", "rule_sup", "m1_sup", "m0_sup", "oracle"]:
            e = R["ladder"][k]
            L.append("| %s | %.1f +/- %.1f | %+.1f%% |" % (k, e["twt_mean"], e["twt_std_pop"], e["pct_below_rule"]))
        for c, nm in [("H1_M1sup_vs_RULEsup", "H1 M1+SUP vs RULE+SUP"),
                      ("H2_M1alone_vs_RULE", "H2 M1-FROZEN vs RULE"),
                      ("M1alone_vs_M0alone", "M1 alone vs M0 alone"),
                      ("M1sup_vs_M0sup", "M1+SUP vs M0+SUP")]:
            sa = R[c]["seedavg"]
            L.append("- %s: %+.1f%%, W/T/L %d/%d/%d, %d/3 seeds, raw p=%.4g"
                     % (nm, sa["pct_gain"], sa["wtl"]["W"], sa["wtl"]["T"], sa["wtl"]["L"],
                        R[c]["per_seed"]["seeds_test_beats_comp"], sa["wilcoxon_p"]))
        L.append("\n**Verdict:** " + json.dumps(R["verdict"]) + "\n")
    open(path, "w").write("\n".join(L) + "\n")
    print("[final] wrote %s" % path)


if __name__ == "__main__":
    main()
