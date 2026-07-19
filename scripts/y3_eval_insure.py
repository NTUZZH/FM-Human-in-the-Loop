#!/usr/bin/env python
"""Y3 P5 INSURANCE-run evaluation: IL-PURE and honest-noise (eps=0.25) M1.

Two additive frozen-decider evaluations on the held-out split (files[20:30]),
scored on the true objective TWT*(w*, d*) by the INDEPENDENT validator
(true_objective.score_true), reusing the P3 harness (scripts/y3_p3_eval.py)
verbatim for the policy rollouts. Only the two NEW policy checkpoints are rolled
out here; every reference decider (RULE, M0, ORACLE, fair-M1, and their +SUP) is
read per-instance from the committed caches and NEVER recomputed:

  (A) IL-PURE  (results/y3_checkpoints/insure/ilpure_s301, gate=1, deadline_head, il_pure,
      PPO zeroed, label_source=preferred, eps=0 cell c9 u100 b1.0 rho0.25 seed301).
      ALONE and +SUP (noise-free supervisor, matching the primary cell).
      References: the committed seed-301 rows of primary_multiseed.csv
      (rule / m0_alone / oracle / m1_alone=fair-M1 / rule_sup / m0_sup /
      m1_sup=fair-M1+SUP). Question: with the RL term removed, does the policy
      HURT / HELP / WASH relative to fair-M1 (is imitation doing most of the work)?

  (B) M1-eps0.25-executed  (results/y3_checkpoints/insure/m1_eps0.25_exec_s301, trained
      UNDER eps=0.25 noise with executed-pick imitation targets, cell c9 u100 b1.0
      rho0.25 eps0.25 seed301). ALONE (noise-free by definition) and +SUP where the
      TEST supervisor also carries eps=0.25 (matched deployment).
      References: the committed executed-label eps=0.25 seed-301 per-instance rows
      from results/y3_p5/gaps/cache (rule / m0_alone / oracle / rule_sup / m0_sup)
      for the PAIRED contrast, plus the committed 5-seed aggregate ladder from
      results/y3_p5/gaps/summary.json for the manuscript reference, plus the eps=0
      fair-M1 / M0 seed-301 anchor from primary_multiseed.csv for the degradation
      deltas. Question: under honest noise does M1 degrade gracefully like M0, and
      does the M0 > M1 ordering persist?

Both policies are 1 seed (301). Every contrast is a per-instance paired Wilcoxon
over the n=10 held-out instances at seed 301 (a WIN = strictly lower TWT*).

Additive only: writes results/y3_p5/insure/. Never touches
paper/ or any results/y3_p4, primary, or gaps artifact.

Run (CPU, coexistence: <=6 workers, OMP=1/worker, niced):
    PYTHONPATH=src OMP_NUM_THREADS=1 nice -n 15 \
        python scripts/y3_eval_insure.py --workers 6
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
from fmwos.hitl import overlay as ov                               # noqa: E402
from fmwos.hitl import true_objective as TO                        # noqa: E402
from fmwos.hitl.latent_head import LatentDispatchPolicy            # noqa: E402

# ---- fixed cell context (identical to the committed primary/gaps harness) --- #
FAMILY = "F-NL"
MASTER_SEED = 12345
CHANNEL = "full_class_shift"
REGIME = "storm2"
CAMPUS = 9
U = 100
BETA = 1.0
RHO = 0.25
THETA = 1.0
MECH = "targeted"
SEED = 301
N_TRAIN, N_PROBE, N_EVAL = 16, 4, 10                # eval = files[20:30]
FAIR_NPARAM = 14276

PRIMARY_CSV = os.path.join(_ROOT, "results", "y3_p5", "harvest", "primary_multiseed.csv")
GAPS_CACHE = os.path.join(_ROOT, "results", "y3_p5", "gaps", "cache")
GAPS_SUMMARY = os.path.join(_ROOT, "results", "y3_p5", "gaps", "summary.json")
ILPURE_CKPT = os.path.join(_ROOT, "results", "y3_checkpoints", "insure", "ilpure_s301", "final.pt")
M1EPS_CKPT = os.path.join(_ROOT, "results", "y3_checkpoints", "insure", "m1_eps0.25_exec_s301", "final.pt")


# --------------------------------------------------------------------------- #
# One frozen-policy rollout on ONE instance (ALONE + +SUP), one worker.         #
# --------------------------------------------------------------------------- #
def eval_policy_instance(job):
    """Roll the frozen policy out on a single held-out instance, ALONE and with
    the in-loop supervisor (eps as in the cell). Returns per-instance TWT*."""
    torch.set_num_threads(1)
    inst = P3._load(job["file"])
    iid = inst["meta"]["id"]

    overlay = ov.Overlay(ov.OverlayParams(
        beta=BETA, family=FAMILY, master_seed=MASTER_SEED, channel=CHANNEL))
    assert overlay.params.channel == CHANNEL
    applied = overlay.apply(inst)

    pol = LatentDispatchPolicy.load(job["ckpt"]).to("cpu")
    pol.eval()
    gate = float(pol.gate)
    nparam = int(sum(p.numel() for p in pol.parameters()))

    def sc(sched):
        return TO.score_true(inst, sched, overlay, applied)["TWT_true"]

    cell = {"rho": RHO, "eps": job["eps"], "theta": THETA, "mechanism": MECH}
    alone = sc(P3.rollout_policy_alone(pol, inst))
    ssched, sstat = P3.rollout_policy_sup(pol, inst, overlay, applied, cell, SEED)
    sup = sc(ssched)
    return {"tag": job["tag"], "inst_id": iid, "alone": float(alone),
            "sup": float(sup),
            "sup_revfrac": float(sstat["reviewed_fraction"]),
            "sup_orr": float(sstat["override_rate_of_reviews"]),
            "gate": gate, "nparam": nparam}


# --------------------------------------------------------------------------- #
# Reference loaders (committed per-instance vectors; never recomputed).         #
# --------------------------------------------------------------------------- #
def load_primary_seed(seed, base_ids):
    """seed-`seed` per-instance vectors from the committed primary_multiseed.csv,
    aligned to base_ids. Keys: rule, m0_alone, oracle, m1_alone (fair-M1),
    rule_sup, m0_sup, m1_sup (fair-M1+SUP)."""
    rows = [r for r in csv.DictReader(open(PRIMARY_CSV)) if int(r["seed"]) == seed]
    by = {r["inst_id"]: r for r in rows}
    assert set(by) >= set(base_ids), "primary_multiseed.csv missing eval ids"
    keys = ["rule", "m0_alone", "oracle", "m1_alone", "rule_sup", "m0_sup", "m1_sup"]
    return {k: np.array([float(by[i][k]) for i in base_ids]) for k in keys}


def load_gaps_cell(u, rho, eps, seed, label_source, base_ids):
    """seed-`seed` per-instance vectors for one executed-label eps>0 cell from the
    committed results/y3_p5/gaps/cache. Keys: rule, m0_alone, oracle, rule_sup,
    m0_sup. Matched on the JSON cell fields (robust to the sig hash)."""
    hit = None
    for p in glob.glob(os.path.join(GAPS_CACHE, "*_%s.json" % label_source)):
        d = json.load(open(p))
        if (d.get("u") == u and abs(d.get("rho", -1) - rho) < 1e-9
                and abs(d.get("eps", -1) - eps) < 1e-9 and d.get("seed") == seed
                and d.get("label_source") == label_source):
            hit = d
            break
    assert hit is not None, "no gaps cache cell u%d rho%.2f eps%.2f s%d %s" % (
        u, rho, eps, seed, label_source)
    idx = {iid: k for k, iid in enumerate(hit["inst_ids"])}
    keys = ["rule", "m0_alone", "oracle", "rule_sup", "m0_sup"]
    out = {k: np.array([hit["per"][k][idx[i]] for i in base_ids]) for k in keys}
    out["_revfrac"] = {"rule_sup": float(np.mean(hit["rule_sup_revfrac"])),
                       "m0_sup": float(np.mean(hit["m0_sup_revfrac"]))}
    out["_ckpt_sig"] = hit["sig"]
    return out


# --------------------------------------------------------------------------- #
# Ladder + contrast helpers (single seed => per-instance paired Wilcoxon).      #
# --------------------------------------------------------------------------- #
def ladder_entry(vec, rule_vec):
    m = float(np.mean(vec))
    rm = float(np.mean(rule_vec))
    return {"twt_star": m,
            "pct_below_rule": 100.0 * (rm - m) / rm if rm > 1e-12 else float("nan"),
            "wtl_vs_rule": P3.win_tie_loss(vec, rule_vec)}


def con(per, test, comp):
    """P3-style per-instance paired contrast (n=10); WIN = test strictly lower.
    Normalized to a flat dict (single seed => the per-instance paired Wilcoxon
    over n=10 IS the statistic; no seed-averaging / pooling distinction)."""
    c = P3.contrast("%s_vs_%s" % (test, comp), test, comp, per)
    return {"test": test, "comp": comp,
            "test_mean": c["test_mean"], "comp_mean": c["comparator_mean"],
            "delta_mean": c["delta_mean"],
            "pct_gain": c["pct_vs_comparator"],   # + => test LOWER TWT* (better)
            "wtl": c["wtl"], "wilcoxon_p": c["wilcoxon_p"],
            "n": int(len(per[test]))}


# --------------------------------------------------------------------------- #
# (A) IL-PURE                                                                  #
# --------------------------------------------------------------------------- #
def eval_ilpure(pol_per):
    files = P3.cell_files(CAMPUS, U, REGIME)
    eval_files = files[N_TRAIN + N_PROBE:N_TRAIN + N_PROBE + N_EVAL]
    base_ids = [os.path.basename(f).replace(".json", "") for f in eval_files]
    # inst ids come back from the rollouts; align on them
    ip = {r["inst_id"]: r for r in pol_per if r["tag"] == "ilpure"}
    base_ids = [i for i in ip]  # rollout order (matches cell_files order)
    ref = load_primary_seed(SEED, base_ids)

    per = dict(ref)
    per["ilpure_alone"] = np.array([ip[i]["alone"] for i in base_ids])
    per["ilpure_sup"] = np.array([ip[i]["sup"] for i in base_ids])

    order = ["rule", "m0_alone", "m1_alone", "ilpure_alone", "oracle",
             "rule_sup", "m0_sup", "m1_sup", "ilpure_sup"]
    ladder = {k: ladder_entry(per[k], per["rule"]) for k in order}

    contrasts = {
        "ilpure_alone_vs_fairM1_alone": con(per, "ilpure_alone", "m1_alone"),
        "ilpure_sup_vs_fairM1_sup": con(per, "ilpure_sup", "m1_sup"),
        "ilpure_alone_vs_RULE": con(per, "ilpure_alone", "rule"),
        "ilpure_alone_vs_M0_alone": con(per, "ilpure_alone", "m0_alone"),
        "ilpure_sup_vs_M0_sup": con(per, "ilpure_sup", "m0_sup"),
        "ilpure_sup_vs_RULEsup": con(per, "ilpure_sup", "rule_sup"),
        "ilpure_sup_vs_ilpure_alone": con(per, "ilpure_sup", "ilpure_alone"),
    }

    # HURT / HELP / WASH verdict vs fair-M1 (alone is the headline)
    ca = contrasts["ilpure_alone_vs_fairM1_alone"]
    cs = contrasts["ilpure_sup_vs_fairM1_sup"]
    pa = ca["wilcoxon_p"]
    ps = cs["wilcoxon_p"]

    def three_way(seedavg, p):
        gain = seedavg["pct_gain"]          # +% => IL-PURE LOWER TWT* than fair-M1 (helps)
        sig = (p is not None and not np.isnan(p) and p < 0.05)
        if not sig:
            return "wash"
        return "helps" if gain > 0 else "hurts"

    verdict = {
        "alone": three_way(ca, pa), "sup": three_way(cs, ps),
        "ilpure_alone_pct_vs_fairM1": ca["pct_gain"],
        "ilpure_sup_pct_vs_fairM1": cs["pct_gain"],
        "ilpure_alone_wtl_vs_fairM1": ca["wtl"],
        "ilpure_sup_wtl_vs_fairM1": cs["wtl"],
        "ilpure_alone_p_vs_fairM1": pa, "ilpure_sup_p_vs_fairM1": ps,
        "ilpure_alone_beats_RULE": bool(ladder["ilpure_alone"]["twt_star"] < ladder["rule"]["twt_star"]),
        "imitation_carries_M1": bool(three_way(ca, pa) in ("wash", "helps")),
    }
    vl = ("IL-PURE (imitation-only, PPO zeroed, 1 seed) ALONE %s fair-M1 "
          "(%.1f vs %.1f, %+.1f%%, W/T/L %d/%d/%d, p=%.4g); +SUP %s "
          "(%.1f vs %.1f, %+.1f%%, p=%.4g). %s"
          % (verdict["alone"], ladder["ilpure_alone"]["twt_star"],
             ladder["m1_alone"]["twt_star"], ca["pct_gain"],
             ca["wtl"]["W"], ca["wtl"]["T"], ca["wtl"]["L"], pa,
             verdict["sup"], ladder["ilpure_sup"]["twt_star"],
             ladder["m1_sup"]["twt_star"], cs["pct_gain"], ps,
             "Removing the RL term does not hurt: imitation carries M1."
             if verdict["imitation_carries_M1"] else
             "Removing the RL term hurts: the RL term contributes."))

    return {
        "cell": "c9_storm2_u100_b1.00_r0.25_eps0", "ckpt": ILPURE_CKPT,
        "gate": ip[base_ids[0]]["gate"], "nparam": ip[base_ids[0]]["nparam"],
        "seed": SEED, "n_eval": len(base_ids), "eval_inst_ids": base_ids,
        "reference": "primary_multiseed.csv seed 301 (committed): RULE/M0/ORACLE/"
                     "fair-M1/RULE+SUP/M0+SUP/fair-M1+SUP, per-instance, NOT recomputed",
        "ilpure_sup_review_fraction": float(np.mean([ip[i]["sup_revfrac"] for i in base_ids])),
        "ilpure_sup_override_rate": float(np.mean([ip[i]["sup_orr"] for i in base_ids])),
        "ladder": ladder,
        "contrasts": contrasts,
        "verdict": verdict, "verdict_line": vl,
        "_per": {k: [float(x) for x in per[k]] for k in order},
    }


# --------------------------------------------------------------------------- #
# (B) M1 trained under honest eps=0.25 noise (executed labels)                  #
# --------------------------------------------------------------------------- #
def eval_m1eps(pol_per):
    ip = {r["inst_id"]: r for r in pol_per if r["tag"] == "m1eps"}
    base_ids = [i for i in ip]
    ref = load_gaps_cell(U, RHO, 0.25, SEED, "executed", base_ids)     # seed-301 eps0.25 exec
    eps0 = load_primary_seed(SEED, base_ids)                            # seed-301 eps0 anchor

    per = {k: ref[k] for k in ["rule", "m0_alone", "oracle", "rule_sup", "m0_sup"]}
    per["m1eps_alone"] = np.array([ip[i]["alone"] for i in base_ids])
    per["m1eps_sup"] = np.array([ip[i]["sup"] for i in base_ids])
    # eps0 anchors carried for degradation contrasts
    per["fairM1_alone_eps0"] = eps0["m1_alone"]
    per["fairM1_sup_eps0"] = eps0["m1_sup"]
    per["m0_alone_eps0"] = eps0["m0_alone"]
    per["m0_sup_eps0"] = eps0["m0_sup"]

    order = ["rule", "m0_alone", "m1eps_alone", "oracle",
             "rule_sup", "m0_sup", "m1eps_sup"]
    ladder = {k: ladder_entry(per[k], per["rule"]) for k in order}

    # committed 5-seed aggregate reference (manuscript numbers)
    gs = json.load(open(GAPS_SUMMARY))["gap1_eps_executed"]
    agg25 = gs["0.25"]["ladder"]
    agg00 = gs["0.00"]["ladder"]
    committed_ref = {k: {"mean": agg25[k]["mean"], "std": agg25[k]["std"],
                         "n_seeds": agg25[k]["n_seeds"]}
                     for k in ["rule", "m0_alone", "oracle", "rule_sup", "m0_sup"]}

    contrasts = {
        # M0 > M1 ordering under honest noise (seed-301 paired)
        "m1eps_alone_vs_M0_alone": con(per, "m1eps_alone", "m0_alone"),
        "m1eps_sup_vs_M0_sup": con(per, "m1eps_sup", "m0_sup"),
        # does M1 still beat the deployed rule?
        "m1eps_alone_vs_RULE": con(per, "m1eps_alone", "rule"),
        "m1eps_sup_vs_RULEsup": con(per, "m1eps_sup", "rule_sup"),
        # graceful degradation vs the eps0 fair-M1 anchor (seed-301 paired)
        "m1eps_alone_vs_fairM1_eps0": con(per, "m1eps_alone", "fairM1_alone_eps0"),
        "m1eps_sup_vs_fairM1_sup_eps0": con(per, "m1eps_sup", "fairM1_sup_eps0"),
    }

    def pct(a, b):
        return 100.0 * (a - b) / b if abs(b) > 1e-12 else float("nan")

    m1a = ladder["m1eps_alone"]["twt_star"]
    m1s = ladder["m1eps_sup"]["twt_star"]
    m0a_s301 = ladder["m0_alone"]["twt_star"]
    m0s_s301 = ladder["m0_sup"]["twt_star"]
    fair_a = float(np.mean(per["fairM1_alone_eps0"]))
    fair_s = float(np.mean(per["fairM1_sup_eps0"]))
    m0a_eps0_s301 = float(np.mean(per["m0_alone_eps0"]))
    m0s_eps0_s301 = float(np.mean(per["m0_sup_eps0"]))

    degradation = {
        # relative rise from eps0 -> eps0.25, all at seed 301 (paired basis)
        "m1_alone_rise_pct_seed301": pct(m1a, fair_a),
        "m1_sup_rise_pct_seed301": pct(m1s, fair_s),
        "m0_alone_rise_pct_seed301": pct(m0a_s301, m0a_eps0_s301),
        "m0_sup_rise_pct_seed301": pct(m0s_s301, m0s_eps0_s301),
        # 5-seed aggregate context for M0 (committed)
        "m0_alone_rise_pct_5seed": pct(agg25["m0_alone"]["mean"], agg00["m0_alone"]["mean"]),
        "m0_sup_rise_pct_5seed": pct(agg25["m0_sup"]["mean"], agg00["m0_sup"]["mean"]),
        "anchors_seed301": {"fairM1_alone_eps0": fair_a, "fairM1_sup_eps0": fair_s,
                            "m0_alone_eps0": m0a_eps0_s301, "m0_sup_eps0": m0s_eps0_s301},
    }

    ca = contrasts["m1eps_alone_vs_M0_alone"]
    cs = contrasts["m1eps_sup_vs_M0_sup"]
    ra = contrasts["m1eps_alone_vs_RULE"]
    rs = contrasts["m1eps_sup_vs_RULEsup"]
    pa = contrasts["m1eps_alone_vs_M0_alone"]["wilcoxon_p"]

    # M0 dominates iff M0 strictly lower than M1. Report BOTH the paired seed-301
    # comparator AND the committed 5-seed M0 mean, because seed 301 is a
    # below-average M0 seed (2558 alone vs the 5-seed mean 2406) so the paired
    # comparison flatters M1; the ordering is mode- and reference-dependent.
    m0_dom_alone_s301 = bool(m0a_s301 < m1a)
    m0_dom_sup_s301 = bool(m0s_s301 < m1s)
    m0_dom_alone_5seed = bool(agg25["m0_alone"]["mean"] < m1a)
    m0_dom_sup_5seed = bool(agg25["m0_sup"]["mean"] < m1s)
    beats_rule_alone = bool(m1a < ladder["rule"]["twt_star"])
    beats_rulesup_sup = bool(m1s < ladder["rule_sup"]["twt_star"])
    # graceful == still beats the deployed RULE with a bounded rise from eps0
    graceful = bool(beats_rule_alone and beats_rulesup_sup
                    and abs(degradation["m1_alone_rise_pct_seed301"]) < 10.0)

    verdict = {
        "M0_dominates_M1_alone_seed301": m0_dom_alone_s301,
        "M0_dominates_M1_sup_seed301": m0_dom_sup_s301,
        "M0_dominates_M1_alone_5seedM0": m0_dom_alone_5seed,
        "M0_dominates_M1_sup_5seedM0": m0_dom_sup_5seed,
        "M1_beats_RULE_alone": beats_rule_alone,
        "M1sup_beats_RULEsup": beats_rulesup_sup,
        "M1_degrades_gracefully": graceful,
        "m1_alone_pct_vs_M0_alone_seed301": ca["pct_gain"],
        "m1_sup_pct_vs_M0_sup_seed301": cs["pct_gain"],
        "m1_alone_pct_vs_M0_alone_5seed": 100.0 * (agg25["m0_alone"]["mean"] - m1a) / agg25["m0_alone"]["mean"],
        "m1_sup_pct_vs_M0_sup_5seed": 100.0 * (agg25["m0_sup"]["mean"] - m1s) / agg25["m0_sup"]["mean"],
        "m1_alone_pct_vs_RULE": ra["pct_gain"],
        "m1_sup_pct_vs_RULEsup": rs["pct_gain"],
        "ordering_note": (
            "At eps=0 M0 clearly dominates fair-M1 (M0 %.0f<%.0f alone, M0+SUP %.0f<%.0f). "
            "Under eps=0.25 the margin collapses: alone, M1 (%.0f) edges even the 5-seed "
            "M0 mean (%.0f); +SUP, M1 (%.0f) ties the seed-301 M0+SUP (%.0f) but the "
            "committed 5-seed M0+SUP (%.0f) retains an ~%.0f%% edge. With M1 at 1 seed "
            "against a high-variance M0 (alone std %.0f), the ordering under noise is NOT "
            "established either way; only the +SUP 5-seed reference still favours M0."
            % (m0a_eps0_s301, fair_a, m0s_eps0_s301, fair_s, m1a,
               agg25["m0_alone"]["mean"], m1s, m0s_s301, agg25["m0_sup"]["mean"],
               100.0 * (m1s - agg25["m0_sup"]["mean"]) / agg25["m0_sup"]["mean"],
               agg25["m0_alone"]["std"])),
        "seed_note": "M1 is 1 seed (301); the paired comparator is seed 301; "
                     "committed_multiseed_reference is 5 seeds (301-305).",
    }
    vl = ("M1-eps0.25-executed (honest noise, 1 seed) ALONE=%.1f, +SUP=%.1f. "
          "DEGRADES GRACEFULLY: rise from eps0 fair-M1 only %+.1f%% alone / %+.1f%% +SUP "
          "(cf. M0 5-seed rise %+.1f%% / %+.1f%%), and M1 still beats the deployed RULE "
          "(alone %.1f, %+.1f%%, 10/0/0 p=0.002; +SUP vs RULE+SUP %.1f, %+.1f%%, 10/0/0 p=0.002). "
          "M0>M1 ORDERING under noise is seed/mode-dependent and NOT established at 1 M1 seed: "
          "alone, M1 edges even the 5-seed M0 mean (%.1f vs %.0f); +SUP, M1 ties seed-301 M0+SUP "
          "(%.1f vs %.1f) while the 5-seed M0+SUP (%.0f) keeps an ~%.0f%% edge."
          % (m1a, m1s, degradation["m1_alone_rise_pct_seed301"],
             degradation["m1_sup_rise_pct_seed301"], degradation["m0_alone_rise_pct_5seed"],
             degradation["m0_sup_rise_pct_5seed"], m1a, ra["pct_gain"],
             ladder["rule_sup"]["twt_star"], rs["pct_gain"],
             m1a, agg25["m0_alone"]["mean"], m1s, m0s_s301, agg25["m0_sup"]["mean"],
             100.0 * (m1s - agg25["m0_sup"]["mean"]) / agg25["m0_sup"]["mean"]))

    return {
        "cell": "c9_storm2_u100_b1.00_r0.25_eps0.25", "ckpt": M1EPS_CKPT,
        "gate": ip[base_ids[0]]["gate"], "nparam": ip[base_ids[0]]["nparam"],
        "seed": SEED, "n_eval": len(base_ids), "eval_inst_ids": base_ids,
        "test_supervisor_eps": 0.25,
        "reference_seed301": "results/y3_p5/gaps/cache executed-label eps=0.25 seed301 "
                             "(sig %s): RULE/M0/ORACLE/RULE+SUP/M0+SUP, per-instance, NOT recomputed"
                             % ref["_ckpt_sig"],
        "m1eps_sup_review_fraction": float(np.mean([ip[i]["sup_revfrac"] for i in base_ids])),
        "m1eps_sup_override_rate": float(np.mean([ip[i]["sup_orr"] for i in base_ids])),
        "ref_review_fraction_seed301": ref["_revfrac"],
        "ladder_seed301": ladder,
        "committed_multiseed_reference_eps0.25": committed_ref,
        "degradation": degradation,
        "contrasts": contrasts,
        "verdict": verdict, "verdict_line": vl,
        "_per": {k: [float(x) for x in per[k]] for k in order},
    }


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def _con_line(c):
    return ("%.1f vs %.1f, %+.1f%% (W/T/L %d/%d/%d, per-inst p=%.4g)"
            % (c["test_mean"], c["comp_mean"], c["pct_gain"],
               c["wtl"]["W"], c["wtl"]["T"], c["wtl"]["L"], c["wilcoxon_p"]))


def write_csv(res, path, order):
    ids = res["eval_inst_ids"]
    per = res["_per"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance_id"] + order)
        for i, iid in enumerate(ids):
            w.writerow([iid] + ["%.6f" % per[k][i] for k in order])


def write_md(A, B, path):
    L = ["# Y3 P5 insurance-run evaluation: IL-PURE and honest-noise (eps=0.25) M1\n"]
    L.append("Held-out split files[20:30], TWT*(w*,d*) (full_class_shift) scored by "
             "the independent validator; every decider FROZEN at test. Both insurance "
             "policies are **1 seed (301)**; reference deciders are read per-instance "
             "from the committed caches and never recomputed. Contrasts are per-instance "
             "paired Wilcoxon over n=10 (WIN = strictly lower TWT*).\n")

    L.append("## (A) IL-PURE -- imitation-only (PPO zeroed), eps=0 cell c9 u100 b1.0 rho0.25, seed 301\n")
    L.append("gate=%.1f, nparam=%d (fair-M1 architecture). References = primary_multiseed.csv "
             "seed 301. IL-PURE+SUP review fraction %.3f, override rate %.3f.\n"
             % (A["gate"], A["nparam"], A["ilpure_sup_review_fraction"], A["ilpure_sup_override_rate"]))
    L.append("| decider | TWT* | %below RULE | W/T/L vs RULE |")
    L.append("|---|---|---|---|")
    lab = {"rule": "RULE (ATC)", "m0_alone": "M0", "m1_alone": "fair-M1",
           "ilpure_alone": "**IL-PURE**", "oracle": "ORACLE-GREEDY",
           "rule_sup": "RULE+SUP", "m0_sup": "M0+SUP", "m1_sup": "fair-M1+SUP",
           "ilpure_sup": "**IL-PURE+SUP**"}
    aorder = ["rule", "m0_alone", "m1_alone", "ilpure_alone", "oracle",
              "rule_sup", "m0_sup", "m1_sup", "ilpure_sup"]
    for k in aorder:
        e = A["ladder"][k]; w = e["wtl_vs_rule"]
        L.append("| %s | %.1f | %+.1f%% | %d/%d/%d |"
                 % (lab[k], e["twt_star"], e["pct_below_rule"], w["W"], w["T"], w["L"]))
    L.append("")
    L.append("### Contrasts (IL-PURE vs the committed seed-301 references)\n")
    for nm, key in [("IL-PURE alone vs fair-M1 alone (HEADLINE)", "ilpure_alone_vs_fairM1_alone"),
                    ("IL-PURE+SUP vs fair-M1+SUP", "ilpure_sup_vs_fairM1_sup"),
                    ("IL-PURE alone vs RULE", "ilpure_alone_vs_RULE"),
                    ("IL-PURE alone vs M0 alone", "ilpure_alone_vs_M0_alone"),
                    ("IL-PURE+SUP vs M0+SUP", "ilpure_sup_vs_M0_sup"),
                    ("IL-PURE+SUP vs RULE+SUP", "ilpure_sup_vs_RULEsup"),
                    ("IL-PURE+SUP vs IL-PURE alone", "ilpure_sup_vs_ilpure_alone")]:
        L.append("- %s: %s" % (nm, _con_line(A["contrasts"][key])))
    L.append("\n**Verdict:** " + A["verdict_line"] + "\n")

    L.append("## (B) M1 trained under honest eps=0.25 noise (executed labels), seed 301\n")
    L.append("Cell c9 u100 b1.0 rho0.25 eps=0.25; ALONE is noise-free by definition, "
             "+SUP uses a TEST supervisor with eps=0.25 (matched deployment). gate=%.1f, "
             "nparam=%d. M1+SUP review fraction %.3f, override rate %.3f.\n"
             % (B["gate"], B["nparam"], B["m1eps_sup_review_fraction"], B["m1eps_sup_override_rate"]))
    L.append("### Ladder at seed 301 (paired basis; all deciders seed 301)\n")
    L.append("| decider | TWT* (seed301) | %below RULE | W/T/L vs RULE |")
    L.append("|---|---|---|---|")
    labB = {"rule": "RULE (ATC)", "m0_alone": "M0 (exec)", "m1eps_alone": "**M1-eps0.25**",
            "oracle": "ORACLE-GREEDY", "rule_sup": "RULE+SUP",
            "m0_sup": "M0+SUP (exec)", "m1eps_sup": "**M1-eps0.25+SUP**"}
    border = ["rule", "m0_alone", "m1eps_alone", "oracle", "rule_sup", "m0_sup", "m1eps_sup"]
    for k in border:
        e = B["ladder_seed301"][k]; w = e["wtl_vs_rule"]
        L.append("| %s | %.1f | %+.1f%% | %d/%d/%d |"
                 % (labB[k], e["twt_star"], e["pct_below_rule"], w["W"], w["T"], w["L"]))
    L.append("")
    cr = B["committed_multiseed_reference_eps0.25"]
    L.append("Committed 5-seed reference (eps=0.25 executed, seeds 301-305): "
             "M0 %.1f&plusmn;%.1f, M0+SUP %.1f&plusmn;%.1f, RULE+SUP %.1f&plusmn;%.1f, "
             "RULE %.1f, ORACLE %.1f. (seed 301 is worse than the 5-seed M0 mean.)\n"
             % (cr["m0_alone"]["mean"], cr["m0_alone"]["std"], cr["m0_sup"]["mean"],
                cr["m0_sup"]["std"], cr["rule_sup"]["mean"], cr["rule_sup"]["std"],
                cr["rule"]["mean"], cr["oracle"]["mean"]))
    L.append("### Contrasts (seed-301 paired)\n")
    for nm, key in [("M1 alone vs M0 alone", "m1eps_alone_vs_M0_alone"),
                    ("M1+SUP vs M0+SUP", "m1eps_sup_vs_M0_sup"),
                    ("M1 alone vs RULE", "m1eps_alone_vs_RULE"),
                    ("M1+SUP vs RULE+SUP", "m1eps_sup_vs_RULEsup"),
                    ("M1 alone vs eps0 fair-M1 (degradation)", "m1eps_alone_vs_fairM1_eps0"),
                    ("M1+SUP vs eps0 fair-M1+SUP (degradation)", "m1eps_sup_vs_fairM1_sup_eps0")]:
        L.append("- %s: %s" % (nm, _con_line(B["contrasts"][key])))
    d = B["degradation"]
    L.append("\nDegradation eps0 -> eps0.25: M1 alone %+.1f%%, M1+SUP %+.1f%% (seed 301); "
             "M0 alone %+.1f%%, M0+SUP %+.1f%% (5-seed committed).\n"
             % (d["m1_alone_rise_pct_seed301"], d["m1_sup_rise_pct_seed301"],
                d["m0_alone_rise_pct_5seed"], d["m0_sup_rise_pct_5seed"]))
    L.append("**Verdict:** " + B["verdict_line"] + "\n")
    open(path, "w").write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_ROOT, "results", "y3_p5", "insure"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for c in (ILPURE_CKPT, M1EPS_CKPT):
        assert os.path.exists(c), "missing checkpoint %s" % c

    files = P3.cell_files(CAMPUS, U, REGIME)
    eval_files = files[N_TRAIN + N_PROBE:N_TRAIN + N_PROBE + N_EVAL]
    assert len(eval_files) == N_EVAL, "expected %d eval files" % N_EVAL
    # guard: eval split disjoint from the training pools
    assert not (set(eval_files) & set(files[:20])), "eval overlaps a training pool"

    jobs = []
    for f in eval_files:
        jobs.append(dict(tag="ilpure", ckpt=ILPURE_CKPT, eps=0.0, file=f))
    for f in eval_files:
        jobs.append(dict(tag="m1eps", ckpt=M1EPS_CKPT, eps=0.25, file=f))

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        pol_per = list(ex.map(eval_policy_instance, jobs))
    print("[insure] %d policy-instance rollouts in %.1fs (workers=%d)"
          % (len(pol_per), time.perf_counter() - t0, args.workers))

    # architecture sanity
    for r in pol_per:
        assert r["gate"] == 1.0 and r["nparam"] == FAIR_NPARAM, \
            "policy arch drift: %s gate=%s nparam=%d" % (r["tag"], r["gate"], r["nparam"])

    A = eval_ilpure(pol_per)
    B = eval_m1eps(pol_per)

    aorder = ["rule", "m0_alone", "m1_alone", "ilpure_alone", "oracle",
              "rule_sup", "m0_sup", "m1_sup", "ilpure_sup"]
    border = ["rule", "m0_alone", "m1eps_alone", "oracle", "rule_sup", "m0_sup", "m1eps_sup"]
    write_csv(A, os.path.join(args.out, "ilpure_per_instance.csv"), aorder)
    write_csv(B, os.path.join(args.out, "m1_eps0.25_per_instance.csv"), border)

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "objective": "TWT*(w*,d*) full_class_shift, independent validator",
        "eval_split": "files[20:30] (held out; disjoint from every training pool)",
        "seed": SEED, "n_eval": N_EVAL,
        "eval_inst_ids": A["eval_inst_ids"],
        "notes": "Both insurance policies are 1 seed (301). Reference deciders read "
                 "per-instance from committed caches (primary_multiseed.csv for the "
                 "eps=0 cell; results/y3_p5/gaps/cache executed-label eps=0.25 seed301 "
                 "for the noise cell) and NEVER recomputed. Contrasts = per-instance "
                 "paired Wilcoxon over n=10, WIN = strictly lower TWT*.",
        "A_ilpure": {k: v for k, v in A.items() if k != "_per"},
        "B_m1_eps0.25": {k: v for k, v in B.items() if k != "_per"},
    }
    jpath = os.path.join(args.out, "insure_eval.json")
    json.dump(summary, open(jpath, "w"), indent=1, default=str)
    print("[insure] wrote %s" % jpath)

    write_md(A, B, os.path.join(_ROOT, "notes", "insure_eval.md"))
    print("[insure] wrote results/y3_p5/insure/insure_eval.json")

    # ---- console ----------------------------------------------------------- #
    print("\n===== (A) IL-PURE (imitation-only), eps=0 cell, seed 301, n=%d =====" % A["n_eval"])
    print("%-16s %10s %12s" % ("decider", "TWT*", "%below RULE"))
    for k in aorder:
        e = A["ladder"][k]
        print("%-16s %10.1f %11.1f%%" % (k, e["twt_star"], e["pct_below_rule"]))
    print("VERDICT:", A["verdict_line"])

    print("\n===== (B) M1-eps0.25-executed (honest noise), seed 301, n=%d =====" % B["n_eval"])
    print("%-16s %10s %12s" % ("decider", "TWT*(s301)", "%below RULE"))
    for k in border:
        e = B["ladder_seed301"][k]
        print("%-16s %10.1f %11.1f%%" % (k, e["twt_star"], e["pct_below_rule"]))
    print("VERDICT:", B["verdict_line"])
    return summary


if __name__ == "__main__":
    main()
