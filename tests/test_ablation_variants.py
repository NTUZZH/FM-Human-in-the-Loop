"""E5 ablation-variant tests -- plain python script (no pytest).

Run:  PYTHONPATH=src python tests/test_ablation_variants.py

Covers the backward-compatible ablation knobs added to ``fmwos.env.DispatchEnv``
and ``fmwos.train`` (the training spec §2 reward variants, §3 feature groups, E5):

(a) REWARD MODES: on one replay instance, roll a full EDD episode via the
    step() API under each reward_mode and check the telescoping identity against
    the validator WWT of the emitted schedule:
      * 'shaped'   sum(rewards) == -finalWWT/100  (within 1e-6)
      * 'realized' sum(rewards) == -finalWWT/100  (within 1e-6)
      * 'terminal' every reward is exactly 0 except the LAST, which equals
                   -finalWWT/100  (within 1e-6)
    (All three modes make the same EDD picks, so the schedule -- hence finalWWT
    -- is identical across modes; we assert that too.)

(b) FEATURE DROP: verify the 1-indexed spec feature numbers map onto env.py's
    actual cand columns (by recomputing slack_days / tardy_already / log1p p /
    p-share for candidate 0), then check that
      * feature_drop='context'  zeroes the ENTIRE ctx vector, cand untouched;
      * feature_drop='urgency'  zeroes EXACTLY cand cols (1,2), nothing else;
      * feature_drop='workload' zeroes EXACTLY cand cols (0,10), nothing else;
    at reset and across a short rollout (drop applies to every observation).

(c) SMOKE TRAIN: run `python -m fmwos.train --smoke --reward terminal
    --feature-drop urgency --seed 999` to completion (cpu, single-threaded) and
    check config.json records the ablation and curves.csv is all-finite.

Prints a report and finally 'ALL ABLATION VARIANT TESTS PASSED'.
"""

import csv
import glob
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import validator                                    # noqa: E402
from fmwos.env import DispatchEnv, F_CTX, _DROP_CAND_COLS      # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")

# c02/150 is the tight campus with large non-zero WWT under EDD (see
# tests/test_env.py) -- needed so the 'terminal' sparse check is non-vacuous.
REWARD_CAMPUS, REWARD_SIZE = "c02", "150"
# c05/150 is a train campus (the training spec §4); the drop test works on any instance.
DROP_CAMPUS, DROP_SIZE = "c05", "150"
N_ROLL = 12          # steps to spot-check the drop across a short rollout
TOL = 1e-6


def first_file(campus, size):
    files = sorted(glob.glob(os.path.join(_INST, campus, "replay", size, "*.json")))
    return files[0]


def load(path):
    with open(path) as fh:
        return json.load(fh)


def _edd_action(env):
    """Index of the earliest-due candidate (id tiebreak) -- deterministic."""
    cands = env._candidates
    return min(range(len(cands)),
               key=lambda i: (cands[i]["due_bh"], cands[i]["id"]))


def _roll_edd(inst, reward_mode):
    """Roll one full EDD episode under ``reward_mode``; return (rewards, wwt)."""
    env = DispatchEnv(inst, reward_mode=reward_mode)
    env.reset()
    rewards, done = [], False
    while not done:
        _obs, r, done, _info = env.step(_edd_action(env))
        rewards.append(r)
    sched = env.to_schedule("edd")
    wwt = validator.validate(inst, sched)["metrics"]["WWT"]
    return rewards, wwt


# --------------------------------------------------------------------------- #
# (a) reward modes                                                            #
# --------------------------------------------------------------------------- #
def test_reward_modes(failures):
    print("(a) REWARD MODES: telescoping per reward_mode (EDD rollout)")
    inst = load(first_file(REWARD_CAMPUS, REWARD_SIZE))

    results = {m: _roll_edd(inst, m) for m in ("shaped", "realized", "terminal")}
    wwts = [results[m][1] for m in ("shaped", "realized", "terminal")]

    # Identical EDD picks -> identical schedule -> identical finalWWT.
    if max(abs(w - wwts[0]) for w in wwts) > 1e-9:
        failures.append("reward modes produced different WWT (picks diverged): %r"
                        % wwts)
    wwt = wwts[0]
    target = -wwt / 100.0
    print("    instance=%s  finalWWT(validator)=%.6f  target=-WWT/100=%.6f"
          % (inst["meta"]["id"], wwt, target))
    if wwt <= 0.0:
        failures.append("reward test vacuous: finalWWT=%.6f is not > 0 (pick a "
                        "tighter instance)" % wwt)

    # 'shaped' and 'realized' both telescope to -finalWWT/100.
    for mode in ("shaped", "realized"):
        rewards, _ = results[mode]
        s = float(sum(rewards))
        diff = abs(s - target)
        print("    %-9s n_steps=%4d  sum=%.6f  |sum-target|=%.2e"
              % (mode, len(rewards), s, diff))
        if diff > TOL:
            failures.append("%s telescoping: |sum-target|=%.3e > %.1e"
                            % (mode, diff, TOL))

    # 'terminal': exactly zero except the last, which equals target.
    rewards, _ = results["terminal"]
    head, last = rewards[:-1], rewards[-1]
    max_head = max((abs(r) for r in head), default=0.0)
    s = float(sum(rewards))
    print("    %-9s n_steps=%4d  last=%.6f  max|head|=%.2e  sum=%.6f"
          % ("terminal", len(rewards), last, max_head, s))
    if max_head != 0.0:
        failures.append("terminal: non-last reward nonzero (max|head|=%.3e)" % max_head)
    if abs(last - target) > TOL:
        failures.append("terminal: last reward %.6f != target %.6f (diff %.3e)"
                        % (last, target, abs(last - target)))
    if abs(s - target) > TOL:
        failures.append("terminal: sum %.6f != target %.6f" % (s, target))


# --------------------------------------------------------------------------- #
# (b) feature drop                                                            #
# --------------------------------------------------------------------------- #
def test_feature_drop(failures):
    print("(b) FEATURE DROP: column mapping + group zeroing")
    inst = load(first_file(DROP_CAMPUS, DROP_SIZE))

    # ---- verify the spec->column mapping against env.py's real layout ----- #
    env0 = DispatchEnv(inst)
    obs0 = env0.reset()
    t0 = env0._cur_now
    trade0 = env0._cur_trade
    j = env0._candidates[0]
    p, d = float(j["p_bh"]), float(j["due_bh"])
    qtw0 = sum(float(x["p_bh"]) for x in env0.queue[trade0])

    exp = {
        0: math.log1p(p),                                     # feat 1  log1p p
        1: max(-30.0, min(30.0, (d - t0 - p) / 8.0)),         # feat 2  slack_days
        2: 1.0 if (t0 + p > d) else 0.0,                      # feat 3  tardy_already
        10: p / (qtw0 + 1e-6),                                # feat 11 p/queue-work
    }
    for col, want in exp.items():
        got = float(obs0["cand"][0, col])
        if abs(got - want) > 1e-6:
            failures.append("col-map: cand[0,%d]=%.6f != expected %.6f"
                            % (col, got, want))
    print("    [map] verified vs _fill_job_features on candidate 0:")
    print("          urgency  -> cand cols %s = spec feats 2,3  (slack_days, tardy_already)"
          % (list(_DROP_CAND_COLS["urgency"]),))
    print("          workload -> cand cols %s = spec feats 1,11 (log1p p, p/queue-work)"
          % (list(_DROP_CAND_COLS["workload"]),))
    print("          context  -> entire ctx vector (F_CTX=%d) zeroed" % F_CTX)

    base = DispatchEnv(inst).reset()

    # ---- context: whole ctx zeroed, cand untouched ------------------------ #
    ctx_env = DispatchEnv(inst, feature_drop="context")
    ctx_obs = ctx_env.reset()
    if ctx_obs["ctx"].any():
        failures.append("context drop: ctx not all-zero")
    if not base["ctx"].any():
        failures.append("context drop vacuous: baseline ctx already all-zero")
    if not np.array_equal(ctx_obs["cand"], base["cand"]):
        failures.append("context drop: cand was modified (should be untouched)")
    if not np.array_equal(ctx_obs["mask"], base["mask"]):
        failures.append("context drop: mask was modified")

    # ---- urgency / workload: exactly the named cand cols zeroed ----------- #
    for group in ("urgency", "workload"):
        cols = list(_DROP_CAND_COLS[group])
        drop_obs = DispatchEnv(inst, feature_drop=group).reset()
        if drop_obs["cand"][:, cols].any():
            failures.append("%s drop: cols %s not all-zero" % (group, cols))
        if not base["cand"][:, cols].any():
            failures.append("%s drop vacuous: baseline cols %s already all-zero"
                            % (group, cols))
        expected = base["cand"].copy()
        expected[:, cols] = 0.0
        if not np.array_equal(drop_obs["cand"], expected):
            failures.append("%s drop: columns OTHER than %s changed" % (group, cols))
        if not np.array_equal(drop_obs["ctx"], base["ctx"]):
            failures.append("%s drop: ctx changed (should be untouched)" % group)
        print("    %-9s -> cand cols %-8s zeroed; other cand cols + ctx unchanged: OK"
              % (group, cols))

    # ---- drop applies to EVERY observation, not just reset ---------------- #
    ce = DispatchEnv(inst, feature_drop="context")
    ue = DispatchEnv(inst, feature_drop="urgency")
    ce.reset(); ue.reset()
    ce_done = ue_done = False
    for _ in range(N_ROLL):
        if not ce_done:
            o, _r, ce_done, _i = ce.step(_edd_action(ce))
            if not ce_done and o["ctx"].any():
                failures.append("context drop: ctx nonzero mid-rollout")
        if not ue_done:
            o, _r, ue_done, _i = ue.step(_edd_action(ue))
            if not ue_done and o["cand"][:, list(_DROP_CAND_COLS["urgency"])].any():
                failures.append("urgency drop: cols nonzero mid-rollout")
    print("    drop holds across a %d-step rollout for context & urgency: OK" % N_ROLL)


# --------------------------------------------------------------------------- #
# (c) smoke train                                                             #
# --------------------------------------------------------------------------- #
def test_smoke_train(failures):
    print("(c) SMOKE TRAIN: --reward terminal --feature-drop urgency (cpu)")
    out_dir = tempfile.mkdtemp(prefix="abl_smoke_")
    cmd = [sys.executable, "-m", "fmwos.train", "--smoke",
           "--reward", "terminal", "--feature-drop", "urgency",
           "--seed", "999", "--out", out_dir, "--device", "cpu"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(_ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Be a polite neighbour to the running dyneval batch: single-threaded.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    proc = subprocess.run(cmd, cwd=_ROOT, env=env, capture_output=True,
                          text=True, timeout=600)
    print("    out=%s  rc=%d" % (out_dir, proc.returncode))
    if proc.returncode != 0:
        failures.append("smoke train exited %d; stderr tail:\n%s"
                        % (proc.returncode, proc.stderr[-1500:]))
        return

    with open(os.path.join(out_dir, "config.json")) as fh:
        cfg = json.load(fh)
    if cfg.get("reward_mode") != "terminal" or cfg.get("feature_drop") != "urgency":
        failures.append("config.json ablation not recorded: reward_mode=%r feature_drop=%r"
                        % (cfg.get("reward_mode"), cfg.get("feature_drop")))
    else:
        print("    config.json records reward_mode=%r feature_drop=%r"
              % (cfg["reward_mode"], cfg["feature_drop"]))

    curves_path = os.path.join(out_dir, "curves.csv")
    with open(curves_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        failures.append("curves.csv is empty")
        return
    num_cols = ("mean_train_return", "dev_wwt_mean", "entropy", "value_loss", "seconds")
    all_finite = True
    for row in rows:
        for key in num_cols:
            v = float(row[key])
            if not math.isfinite(v):
                all_finite = False
                failures.append("curves.csv non-finite %s=%r at update %s"
                                % (key, v, row["update"]))
    last = rows[-1]
    print("    curves.csv rows=%d  all-finite=%s  last: ret=%s dev_wwt=%s ent=%s vloss=%s"
          % (len(rows), all_finite, last["mean_train_return"],
             last["dev_wwt_mean"], last["entropy"], last["value_loss"]))


# --------------------------------------------------------------------------- #
def main():
    failures = []
    test_reward_modes(failures)
    print()
    test_feature_drop(failures)
    print()
    test_smoke_train(failures)
    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("ALL ABLATION VARIANT TESTS PASSED")


if __name__ == "__main__":
    main()
