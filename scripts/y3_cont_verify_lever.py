#!/usr/bin/env python
"""Y3 continuation: VERIFY the full-class-shift crew-starvation headroom is a
genuine, LEARNABLE, non-myopic information lever -- not a relabeling/park-late
artifact (the failure mode that faked +50-89%% in the CP-SAT probes).

For each contended cell (campus, crew_m) and each beta, on the TRUE objective
TWT* = sum w*(c*) max(0, C - d*) we compare four planner families, each run with
THREE information sets:

  RECORDED : recorded fields (w, d = r+SLA(c)).      Sees nothing latent.
  PREDICTED: the feature-RECOVERABLE class shift only, s_hat = clip(round(
             sqrt(beta) f(x)), -2, 2), c_hat = clip(c - s_hat), w_hat/d_hat from
             c_hat. Plug-in conditional-mean proxy = the honest ceiling a
             feature-based LEARNED policy can reach (z is unpredictable). At
             beta=0, s_hat==0 -> PREDICTED==RECORDED by construction.
  TRUE     : the realized true fields (w*, d*, c*). Full-information ceiling.

Planner families:
  ATC        : myopic dispatch (headline). RECORDED=RULE; TRUE=ORACLE (supervisor
               preferred-pick with sup.due overridden to d*, matching the probe).
  PORTFOLIO  : best-of {atc,edd,wspt,pfifo,mor,spt,cr} on the info set, ex-post
               best on TWT* (a COMPETENT non-myopic-ish recorded planner).
  GA         : the repo permutation GA (gap-aware insertion decoder; strictly
               stronger than non-delay dispatch) minimizing WWT on the info set.
               TRUE-GA minimizes TWT* directly; RECORDED-GA a strong optimizer of
               the WRONG objective.

Also: RULE+SUP at rho=0.25 (locked supervisor, targeted, sup.due->d*) to test
whether the in-the-loop mechanism already captures the myopic gap without
learning.

Headroom(info) = (TWT*_RECORDED - TWT*_info)/TWT*_RECORDED, per family.
  full-headroom      = headroom(TRUE)      (the ceiling the ladder reports)
  learnable-headroom = headroom(PREDICTED) (what a learned policy can reach)
A genuine lever needs: learnable ~ full, learnable->0 as beta->0, learnable grows
with beta. An artifact shows: full large even at beta=0, learnable << full.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 nice python scripts/y3_cont_verify_lever.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import copy, csv, glob, json, sys
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
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
BETAS = (0.0, 0.25, 0.5, 1.0)
GA_BUDGET = 30.0
TIE_TOL = 1.0
MAX_WORKERS = 8
PDR_RULES = ("atc", "edd", "wspt", "pfifo", "mor")

# cells: (source, campus, crew_m, n)
CELLS = [
    ("replay", 9, 0.25, 24),
    ("replay", 9, 0.35, 24),
    ("replay", 12, 0.25, 16),
]
_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_cont", "verify-lever")

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
    win = float(instance["meta"]["window_bh"])
    twt = 0.0; incomplete = 0
    for a in sched.get("assignments", []) or []:
        wid = a.get("wo"); end = a.get("end_bh")
        if wid not in wo_by or end is None: continue
        end = float(end)
        twt += wstar[wid] * max(0.0, end - dstar[wid])
        if end > win: incomplete += 1
    return twt, bool(base["feasible"]), incomplete

def make_instance(inst_run, wmap, dmap, cmap):
    ti = copy.deepcopy(inst_run)
    for w in ti["work_orders"]:
        wid = w["id"]
        w["weight"] = wmap[wid]; w["due_bh"] = dmap[wid]; w["priority"] = int(cmap[wid])
    return ti

def pick_spt(q, t, rng): return min(q, key=lambda j: (j["p_bh"], j["id"]))
def pick_cr(q, t, rng):  return min(q, key=lambda j: ((j["due_bh"] - t) / j["p_bh"], j["id"]))
EXTRA = {"spt": pick_spt, "cr": pick_cr}
ALL_RULES = list(PDR_RULES) + list(EXTRA)

def run_rule_on(inst, rule):
    if rule in EXTRA:
        sched, _ = DispatchEnv(inst).run_supervised(
            lambda q, t, rng: (EXTRA[rule](q, t, rng), pdrs._BIG_MARGIN),
            supervisor=None, method=rule, seed=SEED)
        return sched
    return dec.run_rule(DispatchEnv(inst), rule, seed=SEED)

def portfolio_best(inst, inst_run, wstar, dstar):
    best = None
    for r in ALL_RULES:
        s = score(inst_run, run_rule_on(inst, r), wstar, dstar)[0]
        if best is None or s < best: best = s
    return best

def oracle_atc(inst_run, applied, dstar):
    sup = Supervisor(overlay(0.5), inst_run, rho=0.0, applied=applied)
    sup.due = dict(dstar)
    return dec.run_oracle_greedy(DispatchEnv(inst_run), sup, seed=SEED)

def rule_sup(inst_run, applied, dstar, beta, rho=0.25):
    sup = Supervisor(overlay(beta), inst_run, rho=rho, applied=applied, seed=SEED)
    sup.due = dict(dstar)   # full-class-shift: supervisor sees d* too
    sched, _ = DispatchEnv(inst_run).run_supervised(
        dec.rule_decider("atc"), supervisor=sup, method="atc+sup", seed=SEED)
    return sched, sup.summary()

def utilization(inst_run):
    win = float(inst_run["meta"]["window_bh"])
    p_by = defaultdict(float)
    for w in inst_run["work_orders"]: p_by[w["trade"]] += float(w["p_bh"])
    c_by = Counter(t["trade"] for t in inst_run["technicians"])
    tot_p = sum(p_by.values()); tot_c = len(inst_run["technicians"])
    pooled = tot_p / (tot_c * win) if tot_c * win > 0 else float("inf")
    worst = max((pp / (c_by.get(tr, 0) * win) if c_by.get(tr, 0) > 0 else float("inf"))
                for tr, pp in p_by.items())
    return pooled, worst

def predicted_maps(orig, beta):
    """s_hat = clip(round(sqrt(beta) f(x)), -2, 2); c_hat, w_hat, d_hat."""
    wos = sorted(orig["work_orders"], key=lambda w: w["id"])
    f = ov.eval_f_matrix(overlay(beta).coeffs, wos)
    import math
    shat = np.clip(np.round(math.sqrt(beta) * f), -2, 2).astype(int)
    wmap = {}; dmap = {}; cmap = {}
    for i, w in enumerate(wos):
        c = int(w["priority"]); chat = int(min(4, max(1, c - int(shat[i]))))
        cmap[w["id"]] = chat
        wmap[w["id"]] = ov.W_OF_CLASS[chat]
        dmap[w["id"]] = float(w["release_bh"]) + SLA[chat]
    return wmap, dmap, cmap

def process(task):
    source, campus, m, path = task
    orig = json.load(open(path))
    inst_id = orig["meta"]["id"]
    inst_run = orig if m == 1.0 else tightness.scale_crew(orig, m)
    pooled_u, worst_u = utilization(inst_run)

    # recorded schedules (beta-independent)
    rec_atc = dec.run_rule(DispatchEnv(inst_run), "atc", seed=SEED)
    rec_port_scheds = {r: run_rule_on(inst_run, r) for r in ALL_RULES}
    ga_rec = ga.solve_ga(inst_run, budget_s=GA_BUDGET, seed=SEED)

    out = []
    for beta in BETAS:
        ap = overlay(beta).apply(orig)
        wstar = ap["w_star"]; cstar = ap["c_star"]
        dstar = {w["id"]: float(w["release_bh"]) + SLA[cstar[w["id"]]] for w in orig["work_orders"]}
        # true & predicted instances
        ti = make_instance(inst_run, wstar, dstar, cstar)
        pwmap, pdmap, pcmap = predicted_maps(orig, beta)
        pi = make_instance(inst_run, pwmap, pdmap, pcmap)

        # ATC family
        s_rec_atc = score(inst_run, rec_atc, wstar, dstar)[0]
        s_or_atc = score(inst_run, oracle_atc(inst_run, ap, dstar), wstar, dstar)[0]
        s_pred_atc = score(inst_run, run_rule_on(pi, "atc"), wstar, dstar)[0]

        # PORTFOLIO family
        s_rec_port = min(score(inst_run, rec_port_scheds[r], wstar, dstar)[0] for r in ALL_RULES)
        s_true_port = portfolio_best(ti, inst_run, wstar, dstar)
        s_pred_port = portfolio_best(pi, inst_run, wstar, dstar)

        # GA family
        ga_true = ga.solve_ga(ti, budget_s=GA_BUDGET, seed=SEED)
        ga_pred = ga.solve_ga(pi, budget_s=GA_BUDGET, seed=SEED)
        s_rec_ga = score(inst_run, ga_rec, wstar, dstar)[0]
        s_true_ga = score(inst_run, ga_true, wstar, dstar)[0]
        s_pred_ga = score(inst_run, ga_pred, wstar, dstar)[0]

        # RULE+SUP at rho=0.25
        rs_sched, summ = rule_sup(inst_run, ap, dstar, beta)
        s_rulesup = score(inst_run, rs_sched, wstar, dstar)[0]

        rec_inc = score(inst_run, rec_atc, wstar, dstar)[2]
        out.append(dict(
            source=source, campus=campus, crew_m=m, beta=beta, inst_id=inst_id,
            pooled_u=pooled_u, worst_u=worst_u, incomplete=rec_inc,
            rec_atc=s_rec_atc, or_atc=s_or_atc, pred_atc=s_pred_atc,
            rec_port=s_rec_port, true_port=s_true_port, pred_port=s_pred_port,
            rec_ga=s_rec_ga, true_ga=s_true_ga, pred_ga=s_pred_ga,
            rulesup=s_rulesup, rev_frac=summ["reviewed_fraction"],
            or_rate=summ["override_rate_of_reviews"],
        ))
    return out

def tasks():
    out = []
    for source, campus, m, n in CELLS:
        d = os.path.join(_INST, "c%02d" % campus, source, "400")
        files = sorted(glob.glob(os.path.join(d, "*.json")))[:n]
        for p in files:
            out.append((source, campus, m, p))
    return out

def hr(rec, info):
    A = np.mean(rec); B = np.mean(info)
    return 100.0 * (A - B) / A if A > 1e-9 else 0.0

def wtl(recs, rk, ik):
    w = t = l = 0
    for r in recs:
        diff = r[rk] - r[ik]
        if abs(diff) <= TIE_TOL: t += 1
        elif diff > TIE_TOL: w += 1
        else: l += 1
    return w, t, l

def main():
    os.makedirs(_OUT, exist_ok=True)
    tk = tasks()
    print("verify-lever: %d instance-tasks x %d betas, GA=%.0fs" % (len(tk), len(BETAS), GA_BUDGET))
    records = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        done = 0
        for res in ex.map(process, tk, chunksize=1):
            records.extend(res); done += 1
            if done % 10 == 0: print("  %d/%d instance-tasks" % (done, len(tk)))

    cols = list(records[0].keys())
    with open(os.path.join(_OUT, "records.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in records: w.writerow(r)
    print("wrote records.csv (%d rows)" % len(records))

    # per-cell x beta summary
    print("\n%-8s c%-3s m=    beta | u_pool u_worst inc  | ATC: rec->pred->orac  head_full/head_learn | PORT h_full/h_learn | GA h_full/h_learn | SUP capt%%"
          % ("source", ""))
    summary = []
    for source, campus, m, n in CELLS:
        for beta in BETAS:
            sub = [r for r in records if r["source"] == source and r["campus"] == campus
                   and r["crew_m"] == m and abs(r["beta"] - beta) < 1e-9]
            if not sub: continue
            up = float(np.median([r["pooled_u"] for r in sub]))
            uw = float(np.median([r["worst_u"] for r in sub]))
            inc = float(np.mean([r["incomplete"] for r in sub]))
            atc_full = hr([r["rec_atc"] for r in sub], [r["or_atc"] for r in sub])
            atc_learn = hr([r["rec_atc"] for r in sub], [r["pred_atc"] for r in sub])
            port_full = hr([r["rec_port"] for r in sub], [r["true_port"] for r in sub])
            port_learn = hr([r["rec_port"] for r in sub], [r["pred_port"] for r in sub])
            ga_full = hr([r["rec_ga"] for r in sub], [r["true_ga"] for r in sub])
            ga_learn = hr([r["rec_ga"] for r in sub], [r["pred_ga"] for r in sub])
            num = np.mean([r["rec_atc"] for r in sub]) - np.mean([r["rulesup"] for r in sub])
            den = np.mean([r["rec_atc"] for r in sub]) - np.mean([r["or_atc"] for r in sub])
            capt = 100.0 * num / den if den > 1e-9 else 0.0
            rf = float(np.mean([r["rev_frac"] for r in sub]))
            print("%-8s c%02d %.2f  %.2f | %6.2f %6.2f %5.1f | %8.1f/%8.1f/%8.1f  %5.1f/%5.1f | %5.1f/%5.1f | %5.1f/%5.1f | %5.1f (rf=%.2f)"
                  % (source, campus, m, beta, up, uw, inc,
                     np.mean([r["rec_atc"] for r in sub]), np.mean([r["pred_atc"] for r in sub]),
                     np.mean([r["or_atc"] for r in sub]), atc_full, atc_learn,
                     port_full, port_learn, ga_full, ga_learn, capt, rf))
            summary.append(dict(source=source, campus=campus, crew_m=m, beta=beta, n=len(sub),
                                pooled_u=up, worst_u=uw, incomplete=inc,
                                atc_full=atc_full, atc_learn=atc_learn,
                                port_full=port_full, port_learn=port_learn,
                                ga_full=ga_full, ga_learn=ga_learn, sup_capt=capt, rev_frac=rf))
    with open(os.path.join(_OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\nwrote summary.json")

if __name__ == "__main__":
    main()
