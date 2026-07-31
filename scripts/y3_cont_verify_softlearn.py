#!/usr/bin/env python
"""Robustness add-on to y3_cont_verify_lever: a POSTERIOR-MEAN ("soft") learnable
planner that brackets the plug-in PREDICTED ceiling from above.

Given features x, the class shift s = clip(round(xi),-2,2) with
xi ~ N(sqrt(beta) f(x), 1-beta). The exact posterior P(s=k | x) is available in
closed form (clipped-rounded normal). A feature-based learned policy at best
knows this posterior, so its honest ceiling uses:
    w_hat = E[w*(clip(c-s)) | x]   (posterior-mean true weight)
    c_hat = clip(round(c - E[s|x]), 1, 4);  d_hat = r + SLA(c_hat)
This is >= the plug-in proxy (which drops the round/clip nonlinearity and noise
mean-shift) and still -> RULE at beta=0 and -> TRUE at beta=1. Dispatching only
(ATC + best-of-portfolio); no GA. Scored on the SAME TWT*(w*, d*).

Run: PYTHONPATH=src OMP_NUM_THREADS=1 nice python scripts/y3_cont_verify_softlearn.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import copy, glob, json, math, sys
from collections import defaultdict, Counter
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from fmwos import pdrs, tightness
from fmwos.env import DispatchEnv
from fmwos.validator import validate
from fmwos.hitl import deciders as dec
from fmwos.hitl import overlay as ov

SLA = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}
SEED = 301; MASTER_SEED = 12345; FAMILY = "F-NL"
BETAS = (0.0, 0.25, 0.5, 1.0)
PDR_RULES = ("atc", "edd", "wspt", "pfifo", "mor")
CELLS = [("replay", 9, 0.25, 24), ("replay", 9, 0.35, 24), ("replay", 12, 0.25, 16)]
_INST = os.path.join(_ROOT, "data", "processed", "instances")

_OV = {}
def overlay(beta):
    o = _OV.get(beta)
    if o is None:
        o = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY, master_seed=MASTER_SEED, sigma_s=1.0)); _OV[beta] = o
    return o

def _Phi(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def posterior_s(mu, sd):
    """P(s=k) for k in -2..2, s=clip(round(N(mu,sd^2)),-2,2)."""
    if sd < 1e-9:
        k = int(min(2, max(-2, round(mu))))
        return {j: (1.0 if j == k else 0.0) for j in (-2, -1, 0, 1, 2)}
    def cdf(x): return _Phi((x - mu) / sd)
    p = {}
    p[-2] = cdf(-1.5)
    p[-1] = cdf(-0.5) - cdf(-1.5)
    p[0] = cdf(0.5) - cdf(-0.5)
    p[1] = cdf(1.5) - cdf(0.5)
    p[2] = 1.0 - cdf(1.5)
    return p

def soft_maps(orig, beta):
    wos = sorted(orig["work_orders"], key=lambda w: w["id"])
    f = ov.eval_f_matrix(overlay(beta).coeffs, wos)
    sd = math.sqrt(max(0.0, 1.0 - beta))
    wmap = {}; dmap = {}; cmap = {}
    for i, w in enumerate(wos):
        c = int(w["priority"]); mu = math.sqrt(beta) * float(f[i])
        p = posterior_s(mu, sd)
        Ew = sum(p[k] * ov.W_OF_CLASS[int(min(4, max(1, c - k)))] for k in p)
        Es = sum(p[k] * k for k in p)
        chat = int(min(4, max(1, round(c - Es))))
        wmap[w["id"]] = Ew; dmap[w["id"]] = float(w["release_bh"]) + SLA[chat]; cmap[w["id"]] = chat
    return wmap, dmap, cmap

def make_instance(inst_run, wmap, dmap, cmap):
    ti = copy.deepcopy(inst_run)
    for w in ti["work_orders"]:
        wid = w["id"]; w["weight"] = wmap[wid]; w["due_bh"] = dmap[wid]; w["priority"] = int(cmap[wid])
    return ti

def pick_spt(q, t, rng): return min(q, key=lambda j: (j["p_bh"], j["id"]))
def pick_cr(q, t, rng):  return min(q, key=lambda j: ((j["due_bh"] - t) / j["p_bh"], j["id"]))
EXTRA = {"spt": pick_spt, "cr": pick_cr}
ALL_RULES = list(PDR_RULES) + list(EXTRA)

def run_rule_on(inst, rule):
    if rule in EXTRA:
        sched, _ = DispatchEnv(inst).run_supervised(
            lambda q, t, rng: (EXTRA[rule](q, t, rng), pdrs._BIG_MARGIN), supervisor=None, method=rule, seed=SEED)
        return sched
    return dec.run_rule(DispatchEnv(inst), rule, seed=SEED)

def score(instance, sched, wstar, dstar):
    validate(instance, sched)
    twt = 0.0
    for a in sched.get("assignments", []) or []:
        wid = a.get("wo"); end = a.get("end_bh")
        if end is None: continue
        twt += wstar[wid] * max(0.0, float(end) - dstar[wid])
    return twt

def main():
    print("softlearn: posterior-mean learnable planner (dispatching only)")
    print("%-8s c%-3s m    beta | rec_atc  soft_atc  | atc_learn_soft%%  port_learn_soft%%" % ("src", ""))
    for source, campus, m, n in CELLS:
        d = os.path.join(_INST, "c%02d" % campus, source, "400")
        files = sorted(glob.glob(os.path.join(d, "*.json")))[:n]
        for beta in BETAS:
            R_atc = []; S_atc = []; R_port = []; S_port = []
            for fp in files:
                orig = json.load(open(fp))
                inst_run = tightness.scale_crew(orig, m)
                ap = overlay(beta).apply(orig)
                wstar = ap["w_star"]; cstar = ap["c_star"]
                dstar = {w["id"]: float(w["release_bh"]) + SLA[cstar[w["id"]]] for w in orig["work_orders"]}
                swmap, sdmap, scmap = soft_maps(orig, beta)
                si = make_instance(inst_run, swmap, sdmap, scmap)
                rec_atc = score(inst_run, run_rule_on(inst_run, "atc"), wstar, dstar)
                soft_atc = score(inst_run, run_rule_on(si, "atc"), wstar, dstar)
                rec_port = min(score(inst_run, run_rule_on(inst_run, r), wstar, dstar) for r in ALL_RULES)
                soft_port = min(score(inst_run, run_rule_on(si, r), wstar, dstar) for r in ALL_RULES)
                R_atc.append(rec_atc); S_atc.append(soft_atc); R_port.append(rec_port); S_port.append(soft_port)
            def hr(a, b): A = np.mean(a); B = np.mean(b); return 100.0 * (A - B) / A if A > 1e-9 else 0.0
            print("%-8s c%02d %.2f %.2f | %8.1f %8.1f | %8.2f       %8.2f"
                  % (source, campus, m, beta, np.mean(R_atc), np.mean(S_atc), hr(R_atc, S_atc), hr(R_port, S_port)))

if __name__ == "__main__":
    main()
