"""Verify the beta=0 overload mechanism (R2 F3/boundary item).

Claim to test: at beta=0 the per-order latent is unrecoverable (recoverable
share is 0), yet the clip at the class boundaries, c*=clip(c-s,1,4), leaves a
class-level CONSTANT bias in the true shift that is present at every beta, and
under extreme overload even a constant learned correction reorders enough to
matter.

Outputs results/y3_p5/beta0_check.json:
 (a) E[c*-c | c] per recorded class at beta=0 (clip asymmetry), empirical over
     the c10 storm2 eval instances AND a large synthetic draw.
 (b) the trained hat_s distribution at the c10 u130 beta=0 cell: overall mean
     (constant component) and E[hat_s | c] per recorded class.
 (c) pooled RULE vs M0 TWT* at c10 u130 b0 / c10 u110 b0 / c9 u130 b0 (raw e3).
"""
import os, sys, json, glob, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
torch.set_num_threads(4)

from fmwos.hitl import overlay as ov
from fmwos.hitl import augmented_rule as AR
from fmwos.hitl.overlay import W_OF_CLASS, SLA_OF_CLASS

ROOT = os.path.join(os.path.dirname(__file__), "..")
INST = os.path.join(ROOT, "data", "processed", "instances")
MASTER_SEED = 12345
SIGMA_S = 1.0


def locate(campus, u, w="w80"):
    cdir = "c%02d" % campus
    pat = os.path.join(INST, cdir, "storm2", w, "%s_storm2_%s_u%d_*.json" % (cdir, w, u))
    return sorted(glob.glob(pat))


def load(p):
    with open(p) as fh:
        return json.load(fh)


def clip_asym_over_instances(insts, beta=0.0, family="F-NL"):
    """Empirical E[c*-c | c] and E[s | c] over instances at the given beta."""
    overlay = ov.Overlay(ov.OverlayParams(beta=beta, family=family, master_seed=MASTER_SEED))
    by_c = {1: [], 2: [], 3: [], 4: []}   # c*-c
    s_by_c = {1: [], 2: [], 3: [], 4: []}
    for inst in insts:
        ap = overlay.apply(inst)
        for wid, rec in ap["per_order"].items():
            c = rec["c_recorded"]; cs = rec["c_star"]; s = rec["s"]
            by_c[c].append(cs - c)
            s_by_c[c].append(s)
    out = {}
    for c in (1, 2, 3, 4):
        arr = np.array(by_c[c], float)
        sarr = np.array(s_by_c[c], float)
        out[c] = {"n": len(arr),
                  "E_cstar_minus_c": float(arr.mean()) if len(arr) else float("nan"),
                  "E_s": float(sarr.mean()) if len(sarr) else float("nan")}
    return out


def clip_asym_synthetic(n=2_000_000, seed=0):
    """Theoretical E[c*-c | c] for symmetric noise s=clip(round(z),-2,2), z~N(0,1)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    s = np.clip(np.round(SIGMA_S * z), -2, 2).astype(int)
    out = {}
    for c in (1, 2, 3, 4):
        cstar = np.clip(c - s, 1, 4)
        out[c] = {"E_cstar_minus_c": float((cstar - c).mean()), "E_s_effective": float((c - cstar).mean())}
    return out


def train_hat_s(campus, u, beta=0.0, rho=0.25, seed=301, n_train=16, n_probe=4, n_eval=10, iters=8):
    files = locate(campus, u)
    train = [load(p) for p in files[:n_train]]
    probe = [load(p) for p in files[n_train:n_train + n_probe]]
    eval_insts = [load(p) for p in files[n_train + n_probe:n_train + n_probe + n_eval]]
    overlay = ov.Overlay(ov.OverlayParams(beta=beta, family="F-NL", master_seed=MASTER_SEED))
    torch.manual_seed(seed); np.random.seed(seed)
    res = AR.run_m0(train, probe, overlay, beta_rho_eps=(beta, rho, 0.0),
                    outer_iters=iters, mechanism="targeted", theta=1.0,
                    seed=seed, device="cpu", verbose=False)
    est = res["estimator"]
    # hat_s per recorded class over eval instances
    hs_all, c_all = [], []
    for inst in eval_insts:
        ap = overlay.apply(inst)
        hs = AR.hat_s_map(est, inst)
        for wid, v in hs.items():
            hs_all.append(v); c_all.append(ap["per_order"][wid]["c_recorded"])
    hs_all = np.array(hs_all, float); c_all = np.array(c_all, int)
    per_c = {}
    for c in (1, 2, 3, 4):
        m = c_all == c
        per_c[c] = {"n": int(m.sum()),
                    "E_hat_s": float(hs_all[m].mean()) if m.any() else float("nan"),
                    "std_hat_s": float(hs_all[m].std()) if m.any() else float("nan")}
    return {"overall_mean_hat_s": float(hs_all.mean()),
            "overall_std_hat_s": float(hs_all.std()),
            "final_pearson_r": res["per_iter"][-1]["pearson_r"],
            "final_sign_acc": res["per_iter"][-1]["sign_acc_nonzero"],
            "E_hat_s_by_recorded_class": per_c}


def pooled_twt(campus, u, beta):
    """Sum RULE and M0 TWT* over the e3_map rows for this cell (matches decisions.md)."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "y3_p4", "e3_map.csv"))))
    rule = m0 = 0.0
    n = 0
    for r in rows:
        if (r["campus"] == str(campus) and r["u"] in (str(u), "%d.0" % u)
                and float(r["beta"]) == beta):
            rule += float(r["rule"]); m0 += float(r["m0_alone"]); n += 1
    return {"n_rows": n, "rule_sum": rule, "m0_sum": m0,
            "pct_below_rule": (100.0 * (rule - m0) / rule) if rule else float("nan")}


def main():
    c10 = [load(p) for p in locate(10, 130)]
    c9 = [load(p) for p in locate(9, 130)]
    result = {
        "note": ("beta=0 overload mechanism check; F-NL, master_seed 12345. "
                 "Clip asymmetry: c*=clip(c-s,1,4) bounds class 4 to more-urgent-only "
                 "and class 1 to less-urgent-only, a class-level constant bias present "
                 "at every beta. Recoverable per-order share is 0 at beta=0."),
        "clip_asymmetry_synthetic": clip_asym_synthetic(),
        "clip_asymmetry_empirical_c10_beta0": clip_asym_over_instances(c10, beta=0.0),
        "clip_asymmetry_empirical_c9_beta0": clip_asym_over_instances(c9, beta=0.0),
        "hat_s_c9_u130_beta0_seed301": train_hat_s(9, 130, beta=0.0, seed=301, n_train=12, iters=5),
        "pooled_twt": {
            "c10_u130_b0": pooled_twt(10, 130, 0.0),
            "c10_u110_b0": pooled_twt(10, 110, 0.0),
            "c9_u130_b0": pooled_twt(9, 130, 0.0),
            "c9_u100_b0": pooled_twt(9, 100, 0.0),
        },
    }
    out = os.path.join(ROOT, "results", "y3_p5", "beta0_check.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
