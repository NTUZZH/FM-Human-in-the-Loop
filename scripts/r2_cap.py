#!/usr/bin/env python
"""R2 / B5 -- candidate-cap ablation (robustness question: the learned policy picks
among the 64 smallest-slack candidates; do the PDRs win only because that cap
binds under overload?).

We re-evaluate the three v2 MLP policies (results/p3_train/v2/seed{301,302,303}/
best.pt) on the CONTENDED dynamic cells with the env candidate cap raised from 64
to 256, and pair each RL rollout against a cap=64 rerun THROUGH THIS SAME SCRIPT
(not the archived results.csv) so the comparison is apples-to-apples.

Cap-independence (why an eval-time cap change is sound without retraining)
-------------------------------------------------------------------------
The env truncates the queue to the K smallest-slack jobs (env._make_obs). Every
per-candidate feature is computed identically regardless of K: feature 11
(p_bh / queue-total-work) divides by the FULL-queue work `qtw` summed over the
whole trade queue, not the candidate slice, and the F_CTX context vector is
likewise computed over the full queue. Raising K therefore only EXPOSES MORE
candidates (higher-slack jobs that cap=64 would truncate); it changes neither the
surviving candidates' feature values nor the context. The policy is a
per-candidate scorer + masked softmax over the K axis (fmwos.policy), so it
consumes any K without a shape change. Empirically, on instances whose queue
never forces a bind the cap=64 and cap=256 rollouts are assignment-identical.
Caveat (documented): the policy was TRAINED at cap=64, so cap=256 presents
decision contexts (more candidates) never seen in training -- a mild train/eval
distribution shift -- even though each feature value is on the identical scale.

Contended cells (campuses restricted to {5,9,10,12})
----------------------------------------------------
  replay-tight : every replay TEST instance (sizes {150,400}) x crew m in
                 {0.6,0.8}  (fmwos.tightness.scale_crew)
  storm2 u>=1.0: storm2 utilization-sweep instances with u_target in
                 {1.0,1.1,1.3} (crew 1.0, on-disk)

Per config we run the 6 PDRs (seed 301, cap-agnostic) for a best-PDR reference,
and each v2 policy seed at cap in {64,256} (greedy argmax). Every schedule has
travel=0 -> feasible -> scored by the independent validator (WWT == TWT).

Outputs (results/r2_sens/)
--------------------------
  cap256.csv     id, regime, campus, size, method, seed, cap, twt, feasible
                 (RL rows carry cap in {64,256}; PDR rows carry cap='' )
  cap_summary.md paired cap64-vs-cap256 per seed (mean/median diff, Wilcoxon p,
                 win/loss/tie counts) + verdict-flip analysis (does cap=256 let
                 RL overtake the best PDR where cap=64 did not?).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from fmwos import pdrs, tightness            # noqa: E402
from fmwos.env import DispatchEnv             # noqa: E402
from fmwos.validator import validate          # noqa: E402

INST_ROOT = _ROOT / "data" / "processed" / "instances"
INDEX_CSV = INST_ROOT / "index.csv"
OUT_DIR = _ROOT / "results" / "r2_sens"
V2_DIR = _ROOT / "results" / "p3_train" / "v2"

CAMPUSES = {5, 9, 10, 12}
REPLAY_SIZES = {150, 400}
TIGHT_MULTS = [0.6, 0.8]
STORM2_U_MIN = 1.0
RL_SEEDS = [301, 302, 303]
PDR_RULES = ["edd", "wspt", "atc", "pfifo", "mor", "random"]
CAPS = [64, 256]
SEED = 301
TORCH_THREADS = 2


# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #
def _read_index():
    with open(INDEX_CSV, newline="") as f:
        return list(csv.DictReader(f))


def build_targets():
    idx = _read_index()
    configs = []
    # replay-tight
    replay = [r for r in idx
              if str(r.get("split", "")).strip().lower() == "test"
              and str(r.get("track", "")).strip().lower() == "replay"
              and int(r["campus"]) in CAMPUSES
              and int(r["size_class"]) in REPLAY_SIZES]
    for m in TIGHT_MULTS:
        for r in replay:
            configs.append({
                "id": "%s_m%s" % (r["id"], m), "regime": "replay-tight",
                "campus": int(r["campus"]), "size": int(r["size_class"]),
                "path": str(INST_ROOT / r["path"]), "m": float(m),
                "u_target": "",
            })
    # storm2 u>=1.0
    import re
    for r in idx:
        if str(r.get("track", "")).strip().lower() != "storm2":
            continue
        if int(r["campus"]) not in CAMPUSES:
            continue
        mobj = re.search(r"_u(\d+)_", r["id"])
        if not mobj:
            continue
        u = int(mobj.group(1)) / 100.0
        if u < STORM2_U_MIN:
            continue
        configs.append({
            "id": r["id"], "regime": "storm2", "campus": int(r["campus"]),
            "size": int(r["size_class"]), "path": str(INST_ROOT / r["path"]),
            "m": 1.0, "u_target": u,
        })
    configs.sort(key=lambda c: (c["regime"], c["campus"], c["size"], c["id"]))
    return configs


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
_POLICY_CACHE = {}
_MISMATCH = []


def _get_policy(seed):
    pol = _POLICY_CACHE.get(seed)
    if pol is None:
        import torch
        from fmwos.policy import DispatchPolicy
        torch.set_num_threads(TORCH_THREADS)
        pol = DispatchPolicy.load(str(V2_DIR / ("seed%d" % seed) / "best.pt"),
                                  map_location="cpu")
        pol.eval()
        _POLICY_CACHE[seed] = pol
    return pol


def _rl_twt(instance, seed, cap):
    pol = _get_policy(seed)
    env = DispatchEnv(instance, k_cand=cap)
    obs = env.reset()
    done = False
    while not done:
        a, _, _, _ = pol.act(obs, greedy=True, device="cpu")
        obs, _, done, _ = env.step(a)
    sched = env.to_schedule("rl%d" % seed, seed=seed)
    res = validate(instance, sched)
    # cross-check: env-accumulated realized WWT == validator WWT (travel=0)
    d = abs(float(env._realized) - float(res["metrics"]["WWT"]))
    return res["metrics"]["WWT"], int(bool(res["feasible"])), d


def _run_one(config):
    t0 = time.perf_counter()
    try:
        with open(config["path"]) as f:
            instance = json.load(f)
        if config["regime"] == "replay-tight" and config["m"] != 1.0:
            instance = tightness.scale_crew(instance, config["m"])

        rows = []
        # PDR reference (cap-agnostic, seed 301)
        for rule in PDR_RULES:
            sched = pdrs.dispatch(instance, rule, seed=SEED)
            res = validate(instance, sched)
            rows.append({
                "id": config["id"], "regime": config["regime"],
                "campus": config["campus"], "size": config["size"],
                "method": rule, "seed": SEED, "cap": "",
                "twt": res["metrics"]["WWT"],
                "feasible": int(bool(res["feasible"])),
            })
        # RL: 3 seeds x 2 caps
        max_d = 0.0
        for seed in RL_SEEDS:
            for cap in CAPS:
                twt, feas, d = _rl_twt(instance, seed, cap)
                max_d = max(max_d, d)
                rows.append({
                    "id": config["id"], "regime": config["regime"],
                    "campus": config["campus"], "size": config["size"],
                    "method": "rl%d" % seed, "seed": seed, "cap": cap,
                    "twt": twt, "feasible": feas,
                })
        return {"ok": True, "rows": rows, "id": config["id"],
                "max_d": max_d, "wall": time.perf_counter() - t0}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"ok": False, "id": config["id"], "error": "%s: %s" % (
            type(e).__name__, e), "traceback": traceback.format_exc()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="B5 candidate-cap ablation.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = build_targets()
    if args.limit:
        configs = configs[:args.limit]

    by_regime = {}
    for c in configs:
        by_regime[c["regime"]] = by_regime.get(c["regime"], 0) + 1
    print("B5 candidate-cap ablation")
    print("  configs         : %d  %s" % (len(configs), by_regime))
    print("  per config      : 6 PDR + 3 seed x 2 cap RL = %d rollouts"
          % (6 + len(RL_SEEDS) * len(CAPS)))
    print("  v2 policies     : %s" % V2_DIR)
    print("  workers=%d" % args.workers)

    all_rows = []
    max_d = 0.0
    n_err = 0
    t_start = time.perf_counter()
    total = len(configs)
    completed = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(_run_one, configs):
            completed += 1
            if not res.get("ok"):
                n_err += 1
                print("[ERROR] %s: %s" % (res["id"], res.get("error")))
            else:
                all_rows.extend(res["rows"])
                max_d = max(max_d, res.get("max_d", 0.0))
            if completed % 100 == 0 or completed == total:
                el = time.perf_counter() - t_start
                eta = el / completed * (total - completed)
                print("  progress %d/%d  elapsed %.0fs  eta %.0fs  (%d err)"
                      % (completed, total, el, eta, n_err))

    out_csv = OUT_DIR / "cap256.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "regime", "campus", "size",
                                          "method", "seed", "cap", "twt",
                                          "feasible"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print("Wrote %s (%d rows)" % (out_csv, len(all_rows)))
    print("env-realized vs validator WWT max |diff| = %.3e (should be ~0)" % max_d)
    print("errors: %d" % n_err)

    analyse(all_rows)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyse(rows):
    import numpy as np
    from scipy.stats import wilcoxon

    # index: (id, method, cap) -> twt  (RL only), and (id) -> best PDR twt
    rl = {}
    pdr = {}
    meta = {}
    for r in rows:
        meta[r["id"]] = (r["regime"], r["campus"], r["size"])
        if str(r["method"]).startswith("rl"):
            rl[(r["id"], r["method"], int(r["cap"]))] = float(r["twt"])
        else:
            pdr.setdefault(r["id"], []).append(float(r["twt"]))
    best_pdr = {i: min(v) for i, v in pdr.items()}
    ids = sorted(best_pdr)

    L = []
    L.append("# B5 candidate-cap ablation summary")
    L.append("")
    L.append("Source: `cap256.csv`. v2 policies (seed 301/302/303) on the "
             "contended cells (replay-tight m in {0.6,0.8} + storm2 u>=1.0, "
             "campuses {5,9,10,12}). Cap raised 64 -> 256 at EVAL only (no "
             "retraining); features are cap-independent (feature 11 and the "
             "context divide by the full-queue work, not the candidate slice), so "
             "raising the cap only exposes more candidates. Caveat: the policy was "
             "trained at cap=64, so cap=256 is a mild train/eval distribution "
             "shift. Both caps rerun through THIS script for an apples-to-apples "
             "pairing.")
    L.append("")

    L.append("## Paired cap64 vs cap256 (per v2 seed, over %d contended configs)"
             % len(ids))
    L.append("")
    L.append("diff = TWT(cap256) - TWT(cap64); negative => cap256 helps. Wilcoxon "
             "signed-rank over the non-tied pairs.")
    L.append("")
    L.append("| seed | n | n_tied | mean diff | median diff | wins(256<64) | "
             "losses | Wilcoxon p | mean TWT c64 | mean TWT c256 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    print("\nPaired cap64 vs cap256 per seed:")
    for seed in RL_SEEDS:
        m = "rl%d" % seed
        d = []
        c64 = []
        c256 = []
        for i in ids:
            a = rl.get((i, m, 64))
            b = rl.get((i, m, 256))
            if a is None or b is None:
                continue
            c64.append(a)
            c256.append(b)
            d.append(b - a)
        d = np.array(d)
        nz = d[d != 0.0]
        n_tied = int((d == 0.0).sum())
        wins = int((d < 0).sum())
        losses = int((d > 0).sum())
        if nz.size >= 1:
            try:
                _, p = wilcoxon(nz)
            except ValueError:
                p = float("nan")
        else:
            p = float("nan")
        L.append("| rl%d | %d | %d | %+.4f | %+.4f | %d | %d | %.3g | %.3f | "
                 "%.3f |" % (seed, len(d), n_tied, float(np.mean(d)),
                             float(np.median(d)), wins, losses, p,
                             float(np.mean(c64)), float(np.mean(c256))))
        print("  rl%d: n=%d tied=%d mean_diff=%+.4f wins=%d losses=%d "
              "wilcoxon_p=%.3g" % (seed, len(d), n_tied, float(np.mean(d)),
                                   wins, losses, p))
    L.append("")

    # Verdict flip: does cap256 let RL overtake the best PDR where cap64 did not?
    L.append("## Verdict-flip: RL vs best-PDR on each config")
    L.append("")
    L.append("For each (config, seed): is RL's TWT <= best-PDR TWT (RL wins)? "
             "Counts at cap=64 vs cap=256; 'flips' = configs that switch from "
             "RL-loses at cap64 to RL-wins at cap256.")
    L.append("")
    L.append("| seed | RL-wins @cap64 | RL-wins @cap256 | flips (lose->win) | "
             "un-flips (win->lose) |")
    L.append("|---|---|---|---|---|")
    print("\nVerdict-flip (RL vs best PDR):")
    tol = 1e-9
    for seed in RL_SEEDS:
        m = "rl%d" % seed
        win64 = win256 = flips = unflips = n = 0
        for i in ids:
            a = rl.get((i, m, 64))
            b = rl.get((i, m, 256))
            bp = best_pdr.get(i)
            if a is None or b is None or bp is None:
                continue
            n += 1
            w64 = a <= bp + tol
            w256 = b <= bp + tol
            win64 += int(w64)
            win256 += int(w256)
            if (not w64) and w256:
                flips += 1
            if w64 and (not w256):
                unflips += 1
        L.append("| rl%d | %d/%d | %d/%d | %d | %d |"
                 % (seed, win64, n, win256, n, flips, unflips))
        print("  rl%d: RL-wins cap64=%d/%d cap256=%d/%d flips=%d unflips=%d"
              % (seed, win64, n, win256, n, flips, unflips))
    L.append("")

    # per-regime mean diff (cap256 - cap64), pooled over seeds
    L.append("## Mean cap256-cap64 TWT diff by regime (pooled over seeds)")
    L.append("")
    L.append("| regime | n_configs | mean diff | mean |diff| |")
    L.append("|---|---|---|---|")
    for regime in ("replay-tight", "storm2"):
        diffs = []
        for i in ids:
            if meta[i][0] != regime:
                continue
            for seed in RL_SEEDS:
                m = "rl%d" % seed
                a = rl.get((i, m, 64))
                b = rl.get((i, m, 256))
                if a is not None and b is not None:
                    diffs.append(b - a)
        diffs = np.array(diffs)
        nconf = len({i for i in ids if meta[i][0] == regime})
        if diffs.size:
            L.append("| %s | %d | %+.4f | %.4f |"
                     % (regime, nconf, float(np.mean(diffs)),
                        float(np.mean(np.abs(diffs)))))
    L.append("")
    (OUT_DIR / "cap_summary.md").write_text("\n".join(L))
    print("\nWrote %s" % (OUT_DIR / "cap_summary.md"))


if __name__ == "__main__":
    main()
