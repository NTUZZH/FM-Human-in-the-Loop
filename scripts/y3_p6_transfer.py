#!/usr/bin/env python
"""Paper Y3, Phase P6 -- ZERO-SHOT TRANSFER of the M0 correction layer to the
held-out campuses 1 and 2.

The overlay latent uses CAMPUS-AGNOSTIC features only (trade one-hot, log1p p_bh,
release day-of-week) and its coefficients f are drawn on the TRAINING-campus order
population (campuses 5/9/10/12). The shift estimator predicts the class shift from
exactly those observable features. Transfer to a new campus is therefore
well-posed: the feature->shift mapping is the same object everywhere; only the
campus's own feature distribution and load differ.

Two sub-tasks (both REUSE the locked hitl package verbatim; nothing here edits a
locked file):

  TASK 1a  ESTIMATOR TRANSFER (recovery; contention-independent metric).
    Train the M0 shift-estimator on the TRAINING campuses (5/9/10/12), apply it
    ZERO-SHOT to campus 1 and 2 orders, and measure hat_s recovery vs the TRUE
    shift there (sign accuracy on s!=0, Pearson r). Compare to an estimator
    trained NATIVELY on campus 1 / campus 2. Also report the in-distribution
    reference (transfer estimator probed on held-out TRAINING-campus orders).
    ``AR.probe_shift_accuracy`` is the single, quarantined eval-only overlay read.

  TASK 1b  M0 TRANSFER GAIN UNDER INDUCED CONTENTION (TWT*(d*)).
    Campuses 1/2 carry only replay+generator (no storm2), so contention is
    induced with the crew multiplier (``fmwos.tightness.scale_crew``, exactly as a contention scan) to bring pooled utilisation to ~1.0. On
    the crew-scaled held-out instances, score RULE (plain ATC), the TRANSFERRED
    M0 (estimator from 5/9/10/12), the NATIVE M0 (estimator from campus 1/2) and
    ORACLE-GREEDY on TWT*(w*,d*). Report the transfer gain over RULE and how much
    of the native-M0 gain it retains.

Regime is held FIXED at replay / util~1.0 for BOTH the estimator training and the
evaluation, differing only by campus, so this isolates CAMPUS transfer (not a
track or load confound). The estimator is trained under induced contention
because the supervisor only overrides when a mis-ordering costs more than theta
(no contention => no overrides => no training signal); the recovery metric itself
is contention-independent (it reads features + the overlay's true s only).

Config (locked, matches the M0 grid): channel=full_class_shift, family=F-NL,
master_seed=12345, eps=0, theta=1.0, mechanism=targeted, rho=0.25, 8 DAgger iters.
Betas {1.0, 0.75}; seeds 301-303. CPU only.

Run (in tmux y3_p6, coexisting with the PPO sweep; single process, niced):
    PYTHONPATH=src OMP_NUM_THREADS=1 nice -n 15 python scripts/y3_p6_transfer.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import glob
import json
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch                                                    # noqa: E402

from fmwos import tightness                                     # noqa: E402
from fmwos.env import DispatchEnv                               # noqa: E402
from fmwos.hitl import overlay as ov                            # noqa: E402
from fmwos.hitl import augmented_rule as AR                     # noqa: E402
from fmwos.hitl import true_objective as TO                     # noqa: E402
from fmwos.hitl import deciders as dec                          # noqa: E402
from fmwos.hitl.supervisor import Supervisor                    # noqa: E402

# stats helpers (identical settings to the M0 grid)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from y3_abl_common import paired_wilcoxon, win_tie_loss         # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
_OUT = os.path.join(_ROOT, "results", "y3_p6")

# Locked cell constants.
FAMILY = "F-NL"
MASTER_SEED = 12345
EPS = 0.0
THETA = 1.0
MECH = "targeted"
CHANNEL = "full_class_shift"
RHO = 0.25
M0_ITERS = 8

SIZE = 150
TRACK = "replay"
BETAS = (1.0, 0.75)
SEEDS = (301, 302, 303)
TARGET_UTIL = 1.0
UTIL_BAND = (0.85, 1.20)           # keep only instances that scale cleanly to ~1
MMIN = 0.05

TRAIN_CAMPUSES = (5, 9, 10, 12)
HELDOUT_CAMPUSES = (1, 2)

N_TRAIN = 16                       # DAgger training instances per estimator
N_PROBE = 4                        # per-iter probe (logging only)
N_EVAL = 10                        # held-out eval instances
N_PER_TRAINCAMP = 4               # train pool: this many per training campus (x4 = 16)


# --------------------------------------------------------------------------- #
# Instance loading + crew-induced contention                                  #
# --------------------------------------------------------------------------- #
def _load(p):
    with open(p) as fh:
        return json.load(fh)


def pooled_util(inst):
    win = float(inst["meta"]["window_bh"])
    p = sum(float(w["p_bh"]) for w in inst["work_orders"])
    c = len(inst["technicians"])
    return p / (c * win) if c * win > 0 else float("inf")


def scale_to_util(inst, target=TARGET_UTIL):
    """Crew-scale ``inst`` so pooled utilisation is ~target. util scales roughly
    inversely with crew, so m = util(m=1)/target; the per-trade floor (>=1 tech)
    keeps it approximate, so the ACHIEVED util is measured + reported."""
    u1 = pooled_util(inst)
    if not np.isfinite(u1) or u1 <= 0:
        return inst, u1
    m = min(1.0, max(MMIN, u1 / target))
    run = inst if m >= 0.999 else tightness.scale_crew(inst, m)
    return run, pooled_util(run)


def campus_files(campus, split=None):
    d = os.path.join(_INST, "c%02d" % campus, TRACK, str(SIZE))
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    return files


def select_scaled(campus, n_want, skip=0):
    """Return up to n_want (orig_instance, scaled_instance, achieved_util) whose
    achieved util after crew-scaling lands in UTIL_BAND, skipping the first
    ``skip`` qualifying ones (for disjoint train/probe/eval slices)."""
    out = []
    passed = 0
    for f in campus_files(campus):
        orig = _load(f)
        run, u = scale_to_util(orig)
        if not (UTIL_BAND[0] <= u <= UTIL_BAND[1]):
            continue
        if passed < skip:
            passed += 1
            continue
        out.append((orig, run, u))
        if len(out) >= n_want:
            break
    return out


# --------------------------------------------------------------------------- #
# Estimator training (M0 DAgger on crew-scaled instances)                     #
# --------------------------------------------------------------------------- #
def train_estimator_pool(scaled_train, scaled_probe, overlay, beta, seed):
    """Run AR.run_m0 on the (already crew-scaled) train pool. Returns estimator +
    the last per-iter probe row (recovery on the training-campus probe set)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    res = AR.run_m0(scaled_train, scaled_probe, overlay,
                    beta_rho_eps=(beta, RHO, EPS), outer_iters=M0_ITERS,
                    mechanism=MECH, theta=THETA, seed=seed, device="cpu",
                    verbose=False)
    return res["estimator"], res["per_iter"][-1]


# --------------------------------------------------------------------------- #
# TWT* ladder on a crew-scaled held-out set                                   #
# --------------------------------------------------------------------------- #
def twt_ladder(overlay, eval_pack, estimators, seed):
    """eval_pack : list of (orig, scaled, util). estimators : dict name->estimator
    (M0 deciders). Returns per-instance dict of TWT*(d*) lists for rule/oracle +
    each estimator, plus the achieved utils. The latent is applied to the ORIGINAL
    instance (crew scaling does not touch work orders), matching crew-starvation."""
    per = {"rule": [], "oracle": []}
    for name in estimators:
        per[name] = []
    utils = []
    inst_ids = []
    for orig, run, u in eval_pack:
        applied = overlay.apply(orig)          # latent on the ORIGINAL orders
        utils.append(u)
        inst_ids.append(orig["meta"]["id"])

        def sc(sched):
            return TO.score_true(orig, sched, overlay, applied)["TWT_true"]

        per["rule"].append(sc(dec.run_rule(DispatchEnv(run), "atc", seed=seed)))
        # Oracle: bind the supervisor to the ORIGINAL instance so its latent
        # (w*,d*) and per-order p_bh use the unscaled ids (crew scaling appends a
        # suffix to meta.id, which would reseed the latent noise z). The env runs
        # on the crew-scaled instance; work orders + ids are identical, so
        # preferred_pick keys line up. Same discipline as the contention replay.
        osup = Supervisor(overlay, orig, rho=0.0, applied=applied)
        per["oracle"].append(sc(dec.run_oracle_greedy(DispatchEnv(run), osup, seed=seed)))
        for name, est in estimators.items():
            d = AR.augmented_atc_decider(est, run, channel=CHANNEL)
            msched, _ = DispatchEnv(run).run_supervised(d, supervisor=None,
                                                        method="m0", seed=seed)
            per[name].append(sc(msched))
    return {"per": per, "utils": utils, "inst_ids": inst_ids}


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    torch.set_num_threads(1)
    try:
        os.nice(15)
    except Exception:
        pass
    os.makedirs(_OUT, exist_ok=True)

    # ---- fixed instance selections (same across seeds; util-band filtered) --- #
    print("[select] building crew-scaled instance pools (util band %s) ..."
          % (UTIL_BAND,), flush=True)

    # transfer train pool: N_PER_TRAINCAMP per training campus + 1 probe each
    transfer_train, transfer_probe = [], []
    for c in TRAIN_CAMPUSES:
        picks = select_scaled(c, N_PER_TRAINCAMP + 1)
        transfer_train += [p[1] for p in picks[:N_PER_TRAINCAMP]]
        transfer_probe += [p[1] for p in picks[N_PER_TRAINCAMP:N_PER_TRAINCAMP + 1]]
    # in-distribution held-out reference (training-campus orders, disjoint from train)
    indist_eval_orig = []
    for c in TRAIN_CAMPUSES:
        picks = select_scaled(c, 3, skip=N_PER_TRAINCAMP + 1)
        indist_eval_orig += [p[0] for p in picks]      # recovery uses ORIG orders

    # held-out campuses: disjoint train / probe / eval slices from the same pool
    heldout = {}
    for c in HELDOUT_CAMPUSES:
        picks = select_scaled(c, N_TRAIN + N_PROBE + N_EVAL)
        if len(picks) < N_TRAIN + N_PROBE + N_EVAL:
            print("  [warn] c%02d only %d qualifying instances" % (c, len(picks)),
                  flush=True)
        heldout[c] = {
            "train": [p[1] for p in picks[:N_TRAIN]],
            "probe": [p[1] for p in picks[N_TRAIN:N_TRAIN + N_PROBE]],
            "eval": picks[N_TRAIN + N_PROBE:N_TRAIN + N_PROBE + N_EVAL],  # (orig,run,u)
            "eval_orig": [p[0] for p in picks[N_TRAIN + N_PROBE:N_TRAIN + N_PROBE + N_EVAL]],
        }
        us = [p[2] for p in picks[:N_TRAIN + N_PROBE + N_EVAL]]
        print("  c%02d: %d qualifying, achieved util median %.2f [%.2f,%.2f]"
              % (c, len(picks), np.median(us), min(us), max(us)), flush=True)
    tt_us = [pooled_util(x) for x in transfer_train]
    print("  transfer train pool (n=%d) util median %.2f" % (len(transfer_train),
          np.median(tt_us)), flush=True)

    # ---- accumulate results over betas x seeds --------------------------------- #
    rec_rows = []          # 1a recovery rows
    twt_rows = []          # 1b per-instance TWT* rows
    # for seed-averaged 1b contrasts: cell -> {decider: [per-seed per-instance vecs]}
    twt_cells = {}

    for beta in BETAS:
        overlay = ov.Overlay(ov.OverlayParams(beta=beta, family=FAMILY,
                                              master_seed=MASTER_SEED, channel=CHANNEL))
        for seed in SEEDS:
            print("[run] beta=%.2f seed=%d" % (beta, seed), flush=True)
            # --- train the three estimators ---
            est_transfer, tr_probe = train_estimator_pool(
                transfer_train, transfer_probe, overlay, beta, seed)
            natives = {}
            native_probe = {}
            for c in HELDOUT_CAMPUSES:
                est_c, pc = train_estimator_pool(
                    heldout[c]["train"], heldout[c]["probe"], overlay, beta, seed)
                natives[c] = est_c
                native_probe[c] = pc

            # --- 1a: recovery (transfer -> c01/c02/indist ; native -> own campus) ---
            def add_rec(estimator, est_name, heldout_orig, ho_label):
                acc = AR.probe_shift_accuracy(estimator, heldout_orig, overlay)
                rec_rows.append({
                    "beta": beta, "seed": seed, "estimator": est_name,
                    "heldout": ho_label, "sign_acc_nonzero": acc["sign_acc_nonzero"],
                    "exact_class_acc": acc["exact_class_acc"],
                    "pearson_r": acc["pearson_r"],
                    "zero_baseline_acc": acc["zero_baseline_acc"],
                    "n_orders": acc["n_orders"]})
                return acc

            a_ind = add_rec(est_transfer, "transfer", indist_eval_orig, "train_indist")
            for c in HELDOUT_CAMPUSES:
                a_t = add_rec(est_transfer, "transfer", heldout[c]["eval_orig"], "c%02d" % c)
                a_n = add_rec(natives[c], "native", heldout[c]["eval_orig"], "c%02d" % c)
                print("   1a c%02d: transfer sign=%.3f r=%.3f | native sign=%.3f r=%.3f "
                      "| indist sign=%.3f r=%.3f" % (c, a_t["sign_acc_nonzero"],
                      a_t["pearson_r"], a_n["sign_acc_nonzero"], a_n["pearson_r"],
                      a_ind["sign_acc_nonzero"], a_ind["pearson_r"]), flush=True)

            # --- 1b: TWT* ladder under induced contention ---
            for c in HELDOUT_CAMPUSES:
                estimators = {"transfer_m0": est_transfer, "native_m0": natives[c]}
                lad = twt_ladder(overlay, heldout[c]["eval"], estimators, seed)
                per = lad["per"]
                R = float(np.mean(per["rule"]))
                Tr = float(np.mean(per["transfer_m0"]))
                Na = float(np.mean(per["native_m0"]))
                Or = float(np.mean(per["oracle"]))
                print("   1b c%02d: RULE %.1f | transfer-M0 %.1f (%+.1f%%) | native-M0 %.1f "
                      "(%+.1f%%) | ORACLE %.1f (%+.1f%%) util~%.2f"
                      % (c, R, Tr, 100 * (R - Tr) / R, Na, 100 * (R - Na) / R,
                         Or, 100 * (R - Or) / R, float(np.median(lad["utils"]))), flush=True)
                for i, iid in enumerate(lad["inst_ids"]):
                    twt_rows.append({
                        "beta": beta, "seed": seed, "campus": c, "inst_id": iid,
                        "util": lad["utils"][i], "rule": per["rule"][i],
                        "transfer_m0": per["transfer_m0"][i],
                        "native_m0": per["native_m0"][i], "oracle": per["oracle"][i]})
                cell = twt_cells.setdefault((c, beta), {k: [] for k in
                        ["rule", "transfer_m0", "native_m0", "oracle", "util"]})
                for k in ["rule", "transfer_m0", "native_m0", "oracle"]:
                    cell[k].append(np.asarray(per[k], float))
                cell["util"].append(np.asarray(lad["utils"], float))

    # ---- write 1a recovery CSV + seed-averaged summary ------------------------ #
    rec_csv = os.path.join(_OUT, "estimator_transfer.csv")
    with open(rec_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["beta", "seed", "estimator", "heldout",
            "sign_acc_nonzero", "exact_class_acc", "pearson_r",
            "zero_baseline_acc", "n_orders"])
        w.writeheader()
        for r in rec_rows:
            w.writerow(r)
    print("[write] %s (%d rows)" % (rec_csv, len(rec_rows)), flush=True)

    rec_sum_csv = os.path.join(_OUT, "estimator_transfer_summary.csv")
    keys = {}
    for r in rec_rows:
        keys.setdefault((r["beta"], r["estimator"], r["heldout"]), []).append(r)
    with open(rec_sum_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["beta", "estimator", "heldout", "n_seeds",
            "sign_acc_mean", "sign_acc_std", "pearson_r_mean", "pearson_r_std",
            "exact_acc_mean", "zero_baseline_mean", "n_orders"])
        w.writeheader()
        for (beta, est, ho), rows in sorted(keys.items()):
            sa = [x["sign_acc_nonzero"] for x in rows]
            pr = [x["pearson_r"] for x in rows]
            ex = [x["exact_class_acc"] for x in rows]
            zb = [x["zero_baseline_acc"] for x in rows]
            w.writerow({"beta": beta, "estimator": est, "heldout": ho,
                "n_seeds": len(rows),
                "sign_acc_mean": "%.4f" % np.mean(sa), "sign_acc_std": "%.4f" % np.std(sa),
                "pearson_r_mean": "%.4f" % np.mean(pr), "pearson_r_std": "%.4f" % np.std(pr),
                "exact_acc_mean": "%.4f" % np.mean(ex),
                "zero_baseline_mean": "%.4f" % np.mean(zb),
                "n_orders": rows[0]["n_orders"]})
    print("[write] %s" % rec_sum_csv, flush=True)

    # ---- write 1b raw + seed-averaged summary --------------------------------- #
    raw_csv = os.path.join(_OUT, "m0_contention_raw.csv")
    with open(raw_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["beta", "seed", "campus", "inst_id",
            "util", "rule", "transfer_m0", "native_m0", "oracle"])
        w.writeheader()
        for r in twt_rows:
            row = dict(r)
            for k in ("util", "rule", "transfer_m0", "native_m0", "oracle"):
                row[k] = "%.6f" % r[k]
            w.writerow(row)
    print("[write] %s (%d rows)" % (raw_csv, len(twt_rows)), flush=True)

    sum_csv = os.path.join(_OUT, "m0_contention_summary.csv")
    with open(sum_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["campus", "beta", "n_seeds", "n_inst",
            "util_med", "rule", "transfer_m0", "native_m0", "oracle",
            "transfer_pct_below_rule", "native_pct_below_rule", "oracle_pct_below_rule",
            "transfer_retention_of_native", "transfer_pct_of_oracle_gap",
            "wtl_transfer_vs_rule", "wilcoxon_transfer_vs_rule",
            "wtl_transfer_vs_native", "wilcoxon_transfer_vs_native"])
        w.writeheader()
        for (c, beta), cell in sorted(twt_cells.items()):
            # seed-average per instance (rows aligned; eval set identical across seeds)
            def seedavg(key):
                return np.mean(np.stack(cell[key], axis=0), axis=0)
            rule = seedavg("rule"); tr = seedavg("transfer_m0")
            na = seedavg("native_m0"); orc = seedavg("oracle")
            R, Tr, Na, Or = map(lambda v: float(np.mean(v)), (rule, tr, na, orc))
            tr_pct = 100 * (R - Tr) / R if R > 1e-12 else float("nan")
            na_pct = 100 * (R - Na) / R if R > 1e-12 else float("nan")
            or_pct = 100 * (R - Or) / R if R > 1e-12 else float("nan")
            retention = (tr_pct / na_pct) if abs(na_pct) > 1e-9 else float("nan")
            gap = (R - Or)
            tr_of_gap = (100 * (R - Tr) / gap) if abs(gap) > 1e-9 else float("nan")
            wtl_tr = win_tie_loss(tr, rule)
            wtl_trn = win_tie_loss(tr, na)
            w.writerow({"campus": c, "beta": beta, "n_seeds": len(cell["rule"]),
                "n_inst": int(rule.size),
                "util_med": "%.3f" % float(np.median(np.concatenate(cell["util"]))),
                "rule": "%.3f" % R, "transfer_m0": "%.3f" % Tr,
                "native_m0": "%.3f" % Na, "oracle": "%.3f" % Or,
                "transfer_pct_below_rule": "%.2f" % tr_pct,
                "native_pct_below_rule": "%.2f" % na_pct,
                "oracle_pct_below_rule": "%.2f" % or_pct,
                "transfer_retention_of_native": "%.3f" % retention,
                "transfer_pct_of_oracle_gap": "%.2f" % tr_of_gap,
                "wtl_transfer_vs_rule": "%d/%d/%d" % (wtl_tr["W"], wtl_tr["T"], wtl_tr["L"]),
                "wilcoxon_transfer_vs_rule": "%.4g" % paired_wilcoxon(tr, rule),
                "wtl_transfer_vs_native": "%d/%d/%d" % (wtl_trn["W"], wtl_trn["T"], wtl_trn["L"]),
                "wilcoxon_transfer_vs_native": "%.4g" % paired_wilcoxon(tr, na)})
    print("[write] %s" % sum_csv, flush=True)
    print("[done] %.1fs total" % (time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
