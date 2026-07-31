#!/usr/bin/env python
"""Fast probe: reproduce the myopic full-class-shift headroom and test whether a
COMPETENT, non-myopic recorded planner (best-of-portfolio + GA) closes the gap,
plus the beta=0 collapse test. Small N, short GA budget: signal only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import copy, glob, json, sys
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from fmwos import pdrs, tightness, ga
from fmwos.env import DispatchEnv
from fmwos.validator import validate
from fmwos.hitl import deciders as dec
from fmwos.hitl import overlay as ov
from fmwos.hitl.supervisor import Supervisor

SLA = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}
SEED = 301
MASTER_SEED = 12345
FAMILY = "F-NL"
_OV = {}
def overlay(beta):
    o = _OV.get(beta)
    if o is None:
        o = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY, master_seed=MASTER_SEED, sigma_s=1.0))
        _OV[beta] = o
    return o

def score(instance, sched, wstar, dstar):
    base = validate(instance, sched)
    wo_by = {w["id"]: w for w in instance["work_orders"]}
    twt = 0.0
    for a in sched.get("assignments", []) or []:
        wid = a.get("wo"); end = a.get("end_bh")
        if wid not in wo_by or end is None: continue
        twt += wstar[wid] * max(0.0, float(end) - dstar[wid])
    return twt, bool(base["feasible"])

def true_instance(inst_run, wstar, dstar, cstar):
    ti = copy.deepcopy(inst_run)
    for w in ti["work_orders"]:
        wid = w["id"]
        w["weight"] = wstar[wid]; w["due_bh"] = dstar[wid]; w["priority"] = cstar[wid]
    return ti

def pick_spt(q, t, rng): return min(q, key=lambda j: (j["p_bh"], j["id"]))
def pick_cr(q, t, rng):  return min(q, key=lambda j: ((j["due_bh"] - t) / j["p_bh"], j["id"]))
EXTRA = {"spt": pick_spt, "cr": pick_cr}
PDR_RULES = ("atc", "edd", "wspt", "pfifo", "mor")

def run_rule_on(inst, rule):
    if rule in EXTRA:
        sched, _ = DispatchEnv(inst).run_supervised(
            lambda q, t, rng: (EXTRA[rule](q, t, rng), pdrs._BIG_MARGIN),
            supervisor=None, method=rule, seed=SEED)
        return sched
    return dec.run_rule(DispatchEnv(inst), rule, seed=SEED)

def oracle_atc(inst_run, applied, dstar):
    sup = Supervisor(overlay(0.5), inst_run, rho=0.0, applied=applied)
    sup.due = dict(dstar)
    return dec.run_oracle_greedy(DispatchEnv(inst_run), sup, seed=SEED)

def rule_sup(inst_run, applied, dstar, beta, rho=0.25, override_due=True):
    sup = Supervisor(overlay(beta), inst_run, rho=rho, applied=applied, seed=SEED)
    if override_due:
        sup.due = dict(dstar)
    sched, log = DispatchEnv(inst_run).run_supervised(
        dec.rule_decider("atc"), supervisor=sup, method="atc+sup", seed=SEED)
    return sched, sup.summary()

def main():
    m = 0.25
    N = 8
    GA_BUDGET = 10.0
    d = os.path.join(_ROOT, "data", "processed", "instances", "c09", "replay", "400")
    files = sorted(glob.glob(os.path.join(d, "*.json")))[:N]
    betas = (0.0, 0.5, 1.0)
    all_rules = list(PDR_RULES) + list(EXTRA)

    agg = {b: {"rule_atc": [], "oracle_atc": [], "port_rec": [], "port_true": [],
               "ga_rec": [], "ga_true": [], "rulesup": []} for b in betas}
    revfrac = []
    for fp in files:
        orig = json.load(open(fp))
        inst_run = tightness.scale_crew(orig, m)
        # recorded schedules (beta-independent)
        rec_sched = {r: run_rule_on(inst_run, r) for r in all_rules}
        ga_rec = ga.solve_ga(inst_run, budget_s=GA_BUDGET, seed=SEED)
        for b in betas:
            ap = overlay(b).apply(orig)
            wstar = ap["w_star"]; cstar = ap["c_star"]
            dstar = {w["id"]: float(w["release_bh"]) + SLA[cstar[w["id"]]] for w in orig["work_orders"]}
            ti = true_instance(inst_run, wstar, dstar, cstar)
            # recorded portfolio scored on TWT*
            rec_scores = {r: score(inst_run, rec_sched[r], wstar, dstar)[0] for r in all_rules}
            true_sched = {r: run_rule_on(ti, r) for r in all_rules}
            true_scores = {r: score(inst_run, true_sched[r], wstar, dstar)[0] for r in all_rules}
            # GA
            ga_true = ga.solve_ga(ti, budget_s=GA_BUDGET, seed=SEED)
            s_ga_rec = score(inst_run, ga_rec, wstar, dstar)[0]
            s_ga_true = score(inst_run, ga_true, wstar, dstar)[0]
            # myopic oracle + rule
            s_rule_atc = rec_scores["atc"]
            s_or_atc = score(inst_run, oracle_atc(inst_run, ap, dstar), wstar, dstar)[0]
            # rule+sup
            rs_sched, summ = rule_sup(inst_run, ap, dstar, b)
            s_rulesup = score(inst_run, rs_sched, wstar, dstar)[0]
            if b == 0.5: revfrac.append(summ["reviewed_fraction"])
            agg[b]["rule_atc"].append(s_rule_atc)
            agg[b]["oracle_atc"].append(s_or_atc)
            agg[b]["port_rec"].append(min(rec_scores.values()))
            agg[b]["port_true"].append(min(true_scores.values()))
            agg[b]["ga_rec"].append(s_ga_rec)
            agg[b]["ga_true"].append(s_ga_true)
            agg[b]["rulesup"].append(s_rulesup)

    def hr(a, b):
        A = np.mean(a); B = np.mean(b)
        return 100.0 * (A - B) / A if A > 1e-9 else 0.0
    print("c09 replay m=%.2f  N=%d  GA=%.0fs" % (m, N, GA_BUDGET))
    print("beta | myopicATC | portfolio | GA      | rule+sup(gap-captured%%)")
    for b in betas:
        A = agg[b]
        myo = hr(A["rule_atc"], A["oracle_atc"])
        port = hr(A["port_rec"], A["port_true"])
        gah = hr(A["ga_rec"], A["ga_true"])
        # rule+sup gap captured vs myopic oracle
        num = np.mean(A["rule_atc"]) - np.mean(A["rulesup"])
        den = np.mean(A["rule_atc"]) - np.mean(A["oracle_atc"])
        capt = 100.0 * num / den if den > 1e-9 else 0.0
        print("%.2f | %8.2f%% | %7.2f%% | %6.2f%% | %6.1f%%" % (b, myo, port, gah, capt))
    print("mean reviewed_fraction(beta=.5) = %.3f" % np.mean(revfrac))
    # portfolio rule winners
    print("\nrecorded best rule counts / true best rule counts (beta=0.5):")
    b = 0.5
    # recompute winners
    rc, tc = {}, {}
    for fp in files:
        orig = json.load(open(fp)); inst_run = tightness.scale_crew(orig, m)
        ap = overlay(b).apply(orig); wstar = ap["w_star"]; cstar = ap["c_star"]
        dstar = {w["id"]: float(w["release_bh"]) + SLA[cstar[w["id"]]] for w in orig["work_orders"]}
        ti = true_instance(inst_run, wstar, dstar, cstar)
        rs = {r: score(inst_run, run_rule_on(inst_run, r), wstar, dstar)[0] for r in all_rules}
        ts = {r: score(inst_run, run_rule_on(ti, r), wstar, dstar)[0] for r in all_rules}
        rc[min(rs, key=rs.get)] = rc.get(min(rs, key=rs.get), 0) + 1
        tc[min(ts, key=ts.get)] = tc.get(min(ts, key=ts.get), 0) + 1
    print(" recorded:", rc)
    print(" true:    ", tc)

if __name__ == "__main__":
    main()
