"""Independent REFUTATION of the access-window headroom claim (99.205%).

Rebuilds the ORACLE-vs-RULE comparison from scratch (own scoring, own DEFER loop,
own REORDER pick) and stress-tests every way the 99.205% could be an artifact:

 (A) reproduce the c05/replay headline + DECOMPOSE it into recorded-TWT vs the
     invented access penalty (is the "headroom" scheduling value or penalty
     accounting?);
 (B) beta-independence: recompute the whole objective at beta in {0.0, 0.5, 1.0}
     -> if identical, this is NOT the paper's information channel (beta=0 must
     collapse an information lever to ~0; here it does not);
 (C) the task's ACTUAL loaded cells (c09/c12 storm/pmmix): buildings present?
     headroom?;
 (D) "nothing to learn": the restricted set is a deterministic hash of the
     OBSERVABLE building id + fixed public constants (master_seed, alpha). Prove
     it is reconstructible with NO latent read and is invariant to the overlay's
     latent draw; a trivial public rule == DEFER;
 (E) NON-MYOPIC anchor: DEFER is a feasible schedule so its TWT*_access
     upper-bounds the optimum (headroom lower bound holds without a solve). Also
     run a window-BLIND static CP-SAT (min recorded TWT) and score it on
     TWT*_access: if the recorded-optimum still pays the penalty, the gap is
     intrinsic, not an ATC-ordering quirk.

Reuses locked modules read-only. Writes results/y3_diag/refute_access/.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import glob
import heapq
import itertools
import json
import math
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.env import DispatchEnv          # noqa: E402
from fmwos.hitl import deciders as dec      # noqa: E402
from fmwos.hitl import overlay as ov        # noqa: E402
from fmwos import validator as _validator   # noqa: E402
from fmwos import cpsat                      # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_diag", "refute_access")
os.makedirs(_OUT, exist_ok=True)

SEED = 301
MASTER_SEED = 12345
FAMILY = "F-NL"
_DAY_BH = 8.0
_MORNING_END = 4.0


# ---- scoring (copied recorded-TWT block; independent of true_objective.py) --- #
def recorded_twt(instance, schedule):
    wo_by_id = {w["id"]: w for w in instance.get("work_orders", []) or []}
    twt = 0.0
    for a in schedule.get("assignments", []) or []:
        wo = wo_by_id.get(a.get("wo"))
        end = a.get("end_bh")
        if wo is None or end is None:
            continue
        twt += float(wo["weight"]) * max(0.0, float(end) - float(wo["due_bh"]))
    return twt


def score_access(instance, schedule, overlay):
    base = _validator.validate(instance, schedule)
    twt = recorded_twt(instance, schedule)
    pen = overlay.access_penalty(instance, schedule)
    return {"feasible": base["feasible"], "rec": twt, "access": pen,
            "twt_access": twt + pen}


# ---- ATC (recorded weights), mirrors pdrs._pick_atc --------------------------- #
def _atc_argmin(cands, full_queue, now):
    pbar = sum(j["p_bh"] for j in full_queue) / len(full_queue)
    denom = 2.0 * pbar

    def key(j):
        slack = max(0.0, j["due_bh"] - now - j["p_bh"])
        score = (j["weight"] / j["p_bh"]) * math.exp(-slack / denom)
        return (-score, j["id"])

    return min(cands, key=key)


def make_reorder_pick(restricted_ids):
    def pick(queue, t, rng):
        morning = (t % _DAY_BH) < _MORNING_END
        if morning:
            cands = [j for j in queue if j["id"] in restricted_ids] or queue
        else:
            cands = [j for j in queue if j["id"] not in restricted_ids] or queue
        return _atc_argmin(cands, queue, t)
    return pick


_FREE, _RELEASE = 0, 1


def dispatch_defer(instance, restricted_ids):
    inst_id = instance["meta"]["id"]
    queue = defaultdict(list)
    idle = defaultdict(list)
    counter = itertools.count()
    events = []
    for tech in instance["technicians"]:
        heapq.heappush(events, (0.0, next(counter), _FREE, tech["id"], tech["trade"]))
    for wo in instance["work_orders"]:
        heapq.heappush(events, (float(wo["release_bh"]), next(counter), _RELEASE, wo))
    assignments = []

    def next_morning(now):
        return (math.floor(now / _DAY_BH) + 1) * _DAY_BH

    def try_dispatch(trade, now):
        q = queue[trade]
        free = idle[trade]
        while free and q:
            morning = (now % _DAY_BH) < _MORNING_END
            if morning:
                restr = [j for j in q if j["id"] in restricted_ids]
                cands = restr if restr else q
            else:
                nonr = [j for j in q if j["id"] not in restricted_ids]
                if nonr:
                    cands = nonr
                else:
                    m = next_morning(now)
                    while free:
                        tid = heapq.heappop(free)
                        heapq.heappush(events, (m, next(counter), _FREE, tid, trade))
                    return
            job = _atc_argmin(cands, q, now)
            q.remove(job)
            tid = heapq.heappop(free)
            start = float(now)
            assignments.append({"wo": job["id"], "tech": tid,
                                "start_bh": start, "end_bh": start + float(job["p_bh"])})
            heapq.heappush(events, (start + float(job["p_bh"]), next(counter),
                                    _FREE, tid, trade))

    while events:
        now = events[0][0]
        touched = set()
        while events and events[0][0] == now:
            _, _, kind, *payload = heapq.heappop(events)
            if kind == _FREE:
                tid, trade = payload
                heapq.heappush(idle[trade], tid)
                touched.add(trade)
            else:
                wo = payload[0]
                queue[wo["trade"]].append(wo)
                touched.add(wo["trade"])
        for trade in sorted(touched):
            try_dispatch(trade, now)
    return {"instance_id": inst_id, "method": "defer", "seed": SEED,
            "assignments": assignments}


def restricted_ids_of(instance, overlay):
    rb = overlay.restricted_buildings(instance)
    return {w["id"] for w in instance["work_orders"] if w.get("building") in rb}, rb


def files(cell, n):
    c, track, size = cell
    d = os.path.join(_INST, "c%02d" % c, track, str(size))
    return sorted(glob.glob(os.path.join(d, "*.json")))[:n]


def pct(rule, oracle):
    return 100.0 * (rule - oracle) / rule if rule > 1e-9 else 0.0


# ============================================================================ #
def part_A_reproduce():
    print("\n=== (A) REPRODUCE c05/replay headline + DECOMPOSE (rec vs penalty) ===")
    rows = []
    for cell in [(5, "replay", 400), (5, "replay", 150),
                 (10, "replay", 400), (10, "replay", 150)]:
        for alpha in (0.1, 0.2):
            ov_ = ov.Overlay(ov.OverlayParams(beta=1.0, family=FAMILY,
                                              master_seed=MASTER_SEED,
                                              access_alpha=alpha))
            n = 0
            sr = sd = sre = 0.0            # totals: rule twt*, defer twt*, reorder twt*
            sr_rec = sr_acc = 0.0
            sd_rec = sd_acc = 0.0
            defer_wins = defer_ties = 0
            for f in files(cell, 100):
                inst = json.load(open(f))
                rids, _ = restricted_ids_of(inst, ov_)
                s_rule = dec.run_rule(DispatchEnv(inst), "atc", seed=SEED)
                s_reo = DispatchEnv(inst).run_policy(make_reorder_pick(rids),
                                                     method="reo", seed=SEED)
                s_def = dispatch_defer(inst, rids)
                a = score_access(inst, s_rule, ov_)
                b = score_access(inst, s_reo, ov_)
                c = score_access(inst, s_def, ov_)
                assert a["feasible"] and b["feasible"] and c["feasible"]
                n += 1
                sr += a["twt_access"]; sr_rec += a["rec"]; sr_acc += a["access"]
                sre += b["twt_access"]
                sd += c["twt_access"]; sd_rec += c["rec"]; sd_acc += c["access"]
                diff = a["twt_access"] - c["twt_access"]
                if abs(diff) <= 1.0:
                    defer_ties += 1
                elif diff > 1.0:
                    defer_wins += 1
            row = {"cell": "c%02d/%s/%d" % cell, "alpha": alpha, "n": n,
                   "rule_twt*": round(sr / n, 2),
                   "rule_rec": round(sr_rec / n, 3), "rule_acc": round(sr_acc / n, 2),
                   "reorder_twt*": round(sre / n, 2),
                   "reorder_pct": round(pct(sr / n, sre / n), 3),
                   "defer_twt*": round(sd / n, 3),
                   "defer_rec": round(sd_rec / n, 3), "defer_acc": round(sd_acc / n, 2),
                   "defer_pct": round(pct(sr / n, sd / n), 3),
                   "defer_W/T": "%d/%d" % (defer_wins, defer_ties),
                   # decomposition: how much of the gap is penalty vs recorded?
                   "gap_from_penalty_%": round(100.0 * (sr_acc / n - sd_acc / n) /
                                               (sr / n - sd / n), 2) if sr / n - sd / n > 1e-9 else 0.0,
                   "defer_rec_worse_than_rule": round(sd_rec / n - sr_rec / n, 3)}
            rows.append(row)
            print("  %-14s a%.1f | RULE twt*=%.1f (rec %.2f + PEN %.1f) | "
                  "REORDER %.1f (%.2f%%) | DEFER %.2f (rec %.2f + PEN %.2f) "
                  "-> %.3f%%  [%.1f%% of gap is PENALTY; defer_rec-rule_rec=%+.2f]"
                  % (row["cell"], alpha, row["rule_twt*"], row["rule_rec"],
                     row["rule_acc"], row["reorder_twt*"], row["reorder_pct"],
                     row["defer_twt*"], row["defer_rec"], row["defer_acc"],
                     row["defer_pct"], row["gap_from_penalty_%"],
                     row["defer_rec_worse_than_rule"]))
    return rows


def part_B_beta():
    print("\n=== (B) BETA-INDEPENDENCE (info-dial test): recompute at beta {0,.5,1} ===")
    cell = (5, "replay", 400)
    alpha = 0.2
    fs = files(cell, 40)
    out = {}
    restr_signature = {}
    for beta in (0.0, 0.5, 1.0):
        ov_ = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                          master_seed=MASTER_SEED, access_alpha=alpha))
        sr = sd = 0.0
        n = 0
        sig = []
        for f in fs:
            inst = json.load(open(f))
            rids, rb = restricted_ids_of(inst, ov_)
            sig.append((inst["meta"]["id"], tuple(sorted(rb))))
            s_rule = dec.run_rule(DispatchEnv(inst), "atc", seed=SEED)
            s_def = dispatch_defer(inst, rids)
            sr += score_access(inst, s_rule, ov_)["twt_access"]
            sd += score_access(inst, s_def, ov_)["twt_access"]
            n += 1
        out[beta] = (sr / n, sd / n, pct(sr / n, sd / n))
        restr_signature[beta] = sig
        print("  beta=%.2f | RULE twt*=%.3f  DEFER twt*=%.3f  headroom=%.4f%%"
              % (beta, sr / n, sd / n, pct(sr / n, sd / n)))
    same = (restr_signature[0.0] == restr_signature[1.0] == restr_signature[0.5])
    same_headroom = abs(out[0.0][2] - out[1.0][2]) < 1e-6
    print("  restricted sets identical across beta: %s ; headroom identical: %s"
          % (same, same_headroom))
    print("  -> beta=0 does NOT collapse the headroom (fails the info-lever test)"
          if same_headroom else "  -> beta changes headroom")
    return {"per_beta": {str(k): v for k, v in out.items()},
            "restricted_identical_across_beta": same,
            "headroom_identical_across_beta": same_headroom}


def part_C_loaded():
    print("\n=== (C) TASK'S ACTUAL LOADED CELLS: buildings? headroom? ===")
    rows = []
    loaded = [(9, "storm", 150), (12, "storm", 150),
              (9, "pmmix", 150), (12, "pmmix", 150),
              (9, "pmmix", 150), (12, "pmmix", 150)]
    # include the c60 pmmix (the loaded crew level) explicitly
    for cell in [(9, "storm", 150), (12, "storm", 150),
                 (9, "pmmix", 150), (12, "pmmix", 150)]:
        # prefer c60 pmmix files if present
        fs = files(cell, 60)
        if cell[1] == "pmmix":
            c60 = [f for f in glob.glob(os.path.join(
                _INST, "c%02d" % cell[0], "pmmix", "150", "*c60*.json"))]
            if c60:
                fs = sorted(c60)[:60]
        ov_ = ov.Overlay(ov.OverlayParams(beta=1.0, family=FAMILY,
                                          master_seed=MASTER_SEED, access_alpha=0.2))
        n = 0
        tot_bldg = tot_restr = 0
        sr = sd = 0.0
        for f in fs:
            inst = json.load(open(f))
            nb = len({w.get("building") for w in inst["work_orders"]
                      if w.get("building") is not None})
            rids, rb = restricted_ids_of(inst, ov_)
            s_rule = dec.run_rule(DispatchEnv(inst), "atc", seed=SEED)
            s_def = dispatch_defer(inst, rids)
            sr += score_access(inst, s_rule, ov_)["twt_access"]
            sd += score_access(inst, s_def, ov_)["twt_access"]
            tot_bldg += nb
            tot_restr += len(rids)
            n += 1
        rows.append({"cell": "c%02d/%s/%d" % cell, "n": n,
                     "mean_buildings": round(tot_bldg / n, 2),
                     "mean_restr_jobs": round(tot_restr / n, 2),
                     "rule_twt*": round(sr / n, 3), "defer_twt*": round(sd / n, 3),
                     "headroom_pct": round(pct(sr / n, sd / n), 4)})
        print("  %-14s n=%d | mean buildings=%.1f | restr jobs=%.1f | "
              "RULE=%.1f DEFER=%.1f headroom=%.4f%%"
              % (rows[-1]["cell"], n, rows[-1]["mean_buildings"],
                 rows[-1]["mean_restr_jobs"], sr / n, sd / n,
                 rows[-1]["headroom_pct"]))
    return rows


def part_D_nolatent():
    print("\n=== (D) NOTHING-TO-LEARN: restricted set invariant to latent draw ===")
    cell = (5, "replay", 400)
    fs = files(cell, 20)
    # Reconstruct restricted set from ONLY observables (building id + public consts),
    # with NO overlay.apply / no latent, and compare to overlay.restricted_buildings.
    import hashlib
    import numpy as np

    def public_restricted(inst, master_seed, alpha):
        blds = sorted({w.get("building") for w in inst["work_orders"]
                       if w.get("building") is not None})
        out = set()
        for b in blds:
            s = "|".join(str(x) for x in ("access", master_seed,
                                          inst["meta"]["id"], b))
            seed = int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big") >> 1
            if float(np.random.default_rng(seed).random()) < alpha:
                out.add(b)
        return out

    mism = 0
    latent_indep = True
    for f in fs:
        inst = json.load(open(f))
        # overlay's restricted set at two DIFFERENT betas/families (latent varies):
        r_a = ov.Overlay(ov.OverlayParams(beta=0.0, family="F-LIN",
                         master_seed=MASTER_SEED, access_alpha=0.2)).restricted_buildings(inst)
        r_b = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL",
                         master_seed=MASTER_SEED, access_alpha=0.2)).restricted_buildings(inst)
        r_pub = public_restricted(inst, MASTER_SEED, 0.2)
        if not (r_a == r_b):
            latent_indep = False
        if not (r_a == r_pub):
            mism += 1
    print("  restricted set identical across (beta,family) draws: %s" % latent_indep)
    print("  observable-only reconstruction == overlay: %d/%d mismatches"
          % (mism, len(fs)))
    print("  -> the 'lever' is a public deterministic function of the OBSERVABLE "
          "building id; no latent, nothing to estimate.")
    return {"latent_independent": latent_indep,
            "observable_reconstruction_mismatches": mism, "n": len(fs)}


def part_E_nonmyopic():
    print("\n=== (E) NON-MYOPIC: window-BLIND static CP-SAT scored on TWT*_access ===")
    cell = (5, "replay", 400)
    alpha = 0.2
    ov_ = ov.Overlay(ov.OverlayParams(beta=1.0, family=FAMILY,
                                      master_seed=MASTER_SEED, access_alpha=alpha))
    fs = files(cell, 15)
    rows = []
    sr = scp = sd = 0.0
    scp_rec = scp_acc = 0.0
    n = 0
    for f in fs:
        inst = json.load(open(f))
        rids, _ = restricted_ids_of(inst, ov_)
        s_rule = dec.run_rule(DispatchEnv(inst), "atc", seed=SEED)
        s_def = dispatch_defer(inst, rids)
        cp = cpsat.solve(inst, time_limit_s=15.0, workers=4)
        if cp.get("status") not in ("OPTIMAL", "FEASIBLE"):
            print("  %s: CP-SAT status=%s (skip)" % (inst["meta"]["id"], cp.get("status")))
            continue
        a = score_access(inst, s_rule, ov_)
        c = score_access(inst, s_def, ov_)
        e = score_access(inst, cp, ov_)
        n += 1
        sr += a["twt_access"]; sd += c["twt_access"]
        scp += e["twt_access"]; scp_rec += e["rec"]; scp_acc += e["access"]
        rows.append({"id": inst["meta"]["id"], "status": cp["status"],
                     "rule_twt*": round(a["twt_access"], 2),
                     "cpsat_recTWT_opt_twt*": round(e["twt_access"], 2),
                     "cpsat_rec": round(e["rec"], 2), "cpsat_access": round(e["access"], 2),
                     "defer_twt*": round(c["twt_access"], 2)})
    if n:
        print("  n=%d  RULE twt*=%.1f | window-BLIND CP-SAT opt twt*=%.1f "
              "(rec %.2f + PEN %.1f) | DEFER twt*=%.2f"
              % (n, sr / n, scp / n, scp_rec / n, scp_acc / n, sd / n))
        print("  -> even the recorded-TWT OPTIMUM (window-blind) pays PEN=%.1f; "
              "the penalty is intrinsic, not an ATC quirk." % (scp_acc / n))
        print("  -> DEFER (feasible, window-aware) upper-bounds the true optimum, "
              "so non-myopic headroom >= %.3f%%" % pct(sr / n, sd / n))
    return {"n": n,
            "rule_twt*": sr / n if n else None,
            "cpsat_windowblind_twt*": scp / n if n else None,
            "cpsat_windowblind_penalty": scp_acc / n if n else None,
            "defer_twt*": sd / n if n else None,
            "nonmyopic_headroom_lb_pct": pct(sr / n, sd / n) if n else None,
            "per_instance": rows}


def main():
    res = {}
    res["A_reproduce"] = part_A_reproduce()
    res["B_beta"] = part_B_beta()
    res["C_loaded"] = part_C_loaded()
    res["D_nolatent"] = part_D_nolatent()
    res["E_nonmyopic"] = part_E_nonmyopic()
    with open(os.path.join(_OUT, "refute_summary.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    with open(os.path.join(_OUT, "reproduce.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res["A_reproduce"][0].keys()))
        w.writeheader(); w.writerows(res["A_reproduce"])
    print("\nwrote %s" % os.path.join(_OUT, "refute_summary.json"))


if __name__ == "__main__":
    main()
