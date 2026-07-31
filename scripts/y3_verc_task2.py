#!/usr/bin/env python
"""y3_verc TASK 2 -- why is c12 near-zero headroom (and falling with beta),
unlike c9/c10?  Quantified diagnostics on storm2 u100, beta=1.0, 12 instances,
c12 vs c9 (contrast).

Metrics per campus:
  A. class-shift structure: P(s=0), P(s!=0), Pearson r(c_recorded, c_star),
     fraction c*==c.  (Does recorded priority already track true urgency?)
  B. ordering freedom: mean/median candidate-queue size at each ATC decision,
     fraction of decisions that are FORCED (1 candidate).
  C. irreducible floor: oracle/rule TWT* ratio (1-headroom); per-order late
     magnitude (C-d*)+ for tardy jobs (structural if huge); shared tardy mass.
  D. trade concentration: share of TWT*_oracle from the single worst trade, and
     the worst-trade utilization (bottleneck so overloaded ordering is moot?).
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import glob, json, sys, statistics
from collections import defaultdict
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from fmwos.env import DispatchEnv
from fmwos.hitl import deciders as dec
from fmwos.hitl import overlay as ov
from fmwos.hitl.supervisor import Supervisor
from fmwos import validator as _validator
from fmwos import pdrs

_INST = os.path.join(_ROOT, "data", "processed", "instances")
SEED = 301
SLA = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}
BETA = 1.0
N = 12
HORIZON = 80.0


def _dstar_map(inst, applied):
    cs = applied["c_star"]
    return {w["id"]: float(w["release_bh"]) + SLA[cs[w["id"]]]
            for w in inst["work_orders"]}


def _run_atc_logged(inst):
    """Run ATC via run_policy but record queue size at each decision."""
    env = DispatchEnv(inst)
    qsizes = []
    base_pick = pdrs.get_rule("atc")

    def pick(queue, t, rng):
        qsizes.append(len(queue))
        return base_pick(queue, t, rng)
    sched = env.run_policy(pick, method="atc", seed=SEED)
    return sched, qsizes


def analyze(campus, u):
    cdir = "c%02d" % campus
    fs = sorted(glob.glob(os.path.join(_INST, cdir, "storm2", "w80",
                "%s_storm2_w80_u%d_*.json" % (cdir, u))))[:N]
    overlay = ov.Overlay(ov.OverlayParams(beta=BETA, family="F-NL", master_seed=12345))

    s0 = snz = 0
    c_rec_all, c_star_all = [], []
    qsz_all = []
    forced = qdec = 0
    ratio_list = []
    late_mag_rule, late_mag_or = [], []
    worst_trade_share = []
    worst_trade_util = []
    n_wos_list = []

    for p in fs:
        inst = json.load(open(p))
        applied = overlay.apply(inst)
        wstar = applied["w_star"]
        cstar = applied["c_star"]
        dstar = _dstar_map(inst, applied)
        n_wos_list.append(len(inst["work_orders"]))

        # A: shift structure
        for w in inst["work_orders"]:
            wid = w["id"]
            c = int(w["priority"]); cs = cstar[wid]
            c_rec_all.append(c); c_star_all.append(cs)
            if applied["shift"][wid] == 0:
                s0 += 1
            else:
                snz += 1

        # B: ordering freedom (ATC decisions)
        rule_sched, qsizes = _run_atc_logged(inst)
        qsz_all.extend(qsizes)
        for q in qsizes:
            qdec += 1
            if q <= 1:
                forced += 1

        # ORACLE
        sup = Supervisor(overlay, inst, rho=0.0, applied=applied)
        sup.due = dstar
        or_sched = dec.run_oracle_greedy(DispatchEnv(inst), sup, seed=SEED)

        def twt_and_late(sched):
            tot = 0.0
            per_trade = defaultdict(float)
            lates = []
            tr_of = {w["id"]: w["trade"] for w in inst["work_orders"]}
            for a in sched.get("assignments", []):
                wid = a["wo"]; end = float(a["end_bh"])
                if wid not in dstar:
                    continue
                late = max(0.0, end - dstar[wid])
                contrib = wstar[wid] * late
                tot += contrib
                per_trade[tr_of[wid]] += contrib
                if late > 0:
                    lates.append(late)
            return tot, per_trade, lates

        twt_r, pt_r, lates_r = twt_and_late(rule_sched)
        twt_o, pt_o, lates_o = twt_and_late(or_sched)
        if twt_r > 1e-9:
            ratio_list.append(twt_o / twt_r)
        if lates_r:
            late_mag_rule.append(statistics.mean(lates_r))
        if lates_o:
            late_mag_or.append(statistics.mean(lates_o))
        # worst trade share of ORACLE TWT*
        if twt_o > 1e-9 and pt_o:
            wt_trade = max(pt_o, key=pt_o.get)
            worst_trade_share.append(pt_o[wt_trade] / twt_o)
        # worst-trade utilization
        kg = defaultdict(int)
        for t in inst["technicians"]:
            kg[t["trade"]] += 1
        pg = defaultdict(float)
        for w in inst["work_orders"]:
            pg[w["trade"]] += float(w["p_bh"])
        worst = 0.0
        for g, work in pg.items():
            k = kg.get(g, 0)
            if k > 0:
                worst = max(worst, work / (k * HORIZON))
        worst_trade_util.append(worst)

    r = np.corrcoef(c_rec_all, c_star_all)[0, 1]
    frac_same = np.mean([1.0 if a == b else 0.0
                         for a, b in zip(c_rec_all, c_star_all)])
    return dict(
        campus=campus, u=u, n_inst=len(fs),
        mean_n_wos=np.mean(n_wos_list),
        p_s0=s0 / (s0 + snz), p_snz=snz / (s0 + snz),
        pearson_c_cstar=float(r), frac_cstar_eq_c=float(frac_same),
        mean_q=np.mean(qsz_all), median_q=np.median(qsz_all),
        p95_q=np.percentile(qsz_all, 95),
        frac_forced=forced / qdec,
        mean_oracle_over_rule=np.mean(ratio_list),
        headroom_pct=100.0 * (1 - np.mean(ratio_list)),
        mean_late_mag_rule=np.mean(late_mag_rule),
        mean_late_mag_oracle=np.mean(late_mag_or),
        worst_trade_share_oracle=np.mean(worst_trade_share),
        worst_trade_util=np.mean(worst_trade_util),
    )


if __name__ == "__main__":
    out = []
    for (c, u) in [(12, 100), (9, 100), (10, 100)]:
        d = analyze(c, u)
        out.append(d)
        print("\n=== c%d u%d (n=%d, mean_n_wos=%.0f) ===" % (c, u, d["n_inst"], d["mean_n_wos"]))
        print(" A shift: P(s=0)=%.3f P(s!=0)=%.3f  r(c,c*)=%.3f  frac c*==c=%.3f"
              % (d["p_s0"], d["p_snz"], d["pearson_c_cstar"], d["frac_cstar_eq_c"]))
        print(" B freedom: mean_q=%.1f median_q=%.0f p95_q=%.0f  frac_forced(1 cand)=%.3f"
              % (d["mean_q"], d["median_q"], d["p95_q"], d["frac_forced"]))
        print(" C floor: oracle/rule=%.4f (headroom=%.2f%%)  mean_late_mag rule=%.1f oracle=%.1f bh"
              % (d["mean_oracle_over_rule"], d["headroom_pct"],
                 d["mean_late_mag_rule"], d["mean_late_mag_oracle"]))
        print(" D bottleneck: worst-trade share of ORACLE TWT*=%.3f  worst_trade_util=%.2f"
              % (d["worst_trade_share_oracle"], d["worst_trade_util"]))
    with open(os.path.join(_ROOT, "results", "y3_verc", "task2_c12_diag.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote task2_c12_diag.json")
