#!/usr/bin/env python
"""Tier-1 multi-seed training-sweep orchestrator (Paper Y3, Phase P5).

Builds and runs the 29-run Tier-1 training sweep on the headline contention gate
cells: the fair end-to-end policy fair-M1 (deadline_head=True; the STANDARD M1 per
the M1-FAIRNESS VERDICT in notes/decisions.md) and the control PI-0 (Y1-style PPO
with the supervisor OFF, gate=0). Every run reuses the LOCKED, fair-compute-asserted
trainers as subprocesses:

    fair-M1  ->  scripts/y3_p15_m1.py --deadline-head
    PI-0     ->  scripts/y3_p3_pi0.py

This script only ORCHESTRATES: it resolves each run's full config, diffs it against
the committed pilot config, asserts the ONLY differences are the intended axes
(seed, u, beta, rho for M1; seed, u for PI-0) plus deadline_head=True for M1,
asserts the fair-M1 network parameter count matches the pilot fair-M1 checkpoint,
asserts the fair-compute env-step budget resolves to exactly 4,915,200 for every
run, writes the eval manifest (results/y3_p5/sweep_manifest.json), and then runs the
jobs through a concurrency-2 queue. Each child is niced (nice -n 10) and capped at 8
threads (OMP/MKL/OpenBLAS env + --threads 8 -> torch.set_num_threads(8)), so at
concurrency 2 the sweep uses <= 16 threads and coexists with the ~8-thread y3_p4
batch inside the 24-core budget.

RUN LIST (fair-M1: deadline_head=True; channel=full_class_shift, F-NL,
master_seed=12345, eps=0, mechanism=targeted, regime=storm2):
  fair-M1 PRIMARY : c9 u100 beta1.0 rho0.25 seeds 301-310        (10 runs)
  fair-M1 REGIME  : c9 u90  beta1.0 rho0.25 seeds 301-303        (3 runs)
                    c9 u100 beta0.75 rho0.25 seeds 301-303       (3 runs)
  PI-0 PRIMARY    : c9 u100 seeds 301-310                        (10 runs)
  PI-0 REGIME     : c9 u90  seeds 301-303                        (3 runs)
  ------------------------------------------------------------  29 runs total

PI-0 depends only on (campus, u, seed): the same PI-0 checkpoint serves every
(beta, rho) eval cell at its (campus, u, seed), so no beta/rho fan-out is trained.

Modes
-----
  --verify-only     resolve + diff every config, assert param/budget, write the
                    manifest, print the PASS/FAIL summary, and EXIT (no training).
  (default)         verify (same asserts, aborts on any failure), then run the
                    concurrency-N queue to completion, updating manifest statuses.

Usage
-----
  python scripts/y3_sweep_train.py --verify-only          # pre-launch check
  python scripts/y3_sweep_train.py --concurrency 2        # full sweep (in tmux)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _SCRIPTS)

# Reuse the LOCKED budget-resolution + policy code so the resolved config is built
# by exactly the same path the trainer uses (no reimplementation of the budget).
from y3_p2_train import y1_single_run_budget, resolve_schedule            # noqa: E402
from fmwos.hitl.latent_head import LatentDispatchPolicy                   # noqa: E402
from fmwos.hitl.intervention import OVERRIDE_WEIGHT, CONFIRM_WEIGHT       # noqa: E402

# ---- pilot references (committed) ----------------------------------------- #
PILOT_M1_CONFIG = os.path.join(_ROOT, "train_log", "y3_p15", "m1_full", "config.json")
PILOT_M1FAIR_CKPT = os.path.join(_ROOT, "train_log", "y3_p3_m1fair", "final.pt")
PILOT_PI0_CONFIG = os.path.join(_ROOT, "train_log", "y3_p3", "pi0_full", "config.json")

SWEEP_DIR = os.path.join(_ROOT, "train_log", "y3_sweep")
MANIFEST_DIR = os.path.join(_ROOT, "results", "y3_p5")
MANIFEST_PATH = os.path.join(MANIFEST_DIR, "sweep_manifest.json")

FAIR_COMPUTE_ENV_STEPS = 4915200          # == Y1 single-run PPO budget
OUTER_ITERS = 8
N_ENVS, STEPS_PER_ENV = 16, 512

# Fixed (cell-invariant) parts of the trainer config, mirroring y3_p2_train.train_m1.
COMMON_CELL = dict(eps=0.0, theta=1.0, mechanism="targeted", family="F-NL",
                   master_seed=12345, channel="full_class_shift", regime="storm2")
PPO_BLOCK = {"gamma": 1.0, "lam": 0.98, "clip": 0.2, "epochs": 4,
             "ent_coef": 0.01, "val_coef": 0.5, "lr": 0.0003, "minibatch": 1024}


# --------------------------------------------------------------------------- #
# Run list                                                                    #
# --------------------------------------------------------------------------- #
def build_run_list():
    runs = []
    # fair-M1 PRIMARY: c9 u100 beta1.0 rho0.25 seeds 301-310
    for s in range(301, 311):
        runs.append(dict(decider="m1", campus=9, u=100, beta=1.0, rho=0.25,
                         seed=s, group="m1_primary"))
    # fair-M1 REGIME: c9 u90 beta1.0 rho0.25 seeds 301-303
    for s in range(301, 304):
        runs.append(dict(decider="m1", campus=9, u=90, beta=1.0, rho=0.25,
                         seed=s, group="m1_regime_u90"))
    # fair-M1 REGIME: c9 u100 beta0.75 rho0.25 seeds 301-303
    for s in range(301, 304):
        runs.append(dict(decider="m1", campus=9, u=100, beta=0.75, rho=0.25,
                         seed=s, group="m1_regime_beta075"))
    # PI-0 PRIMARY: c9 u100 seeds 301-310  (beta/rho fixed by the pilot recipe)
    for s in range(301, 311):
        runs.append(dict(decider="pi0", campus=9, u=100, beta=1.0, rho=0.0,
                         seed=s, group="pi0_primary"))
    # PI-0 REGIME: c9 u90 seeds 301-303
    for s in range(301, 304):
        runs.append(dict(decider="pi0", campus=9, u=90, beta=1.0, rho=0.0,
                         seed=s, group="pi0_regime_u90"))
    for r in runs:
        r["run_id"] = run_id(r)
        r["out_dir"] = os.path.join(SWEEP_DIR, r["run_id"])
    return runs


def run_id(r):
    if r["decider"] == "m1":
        return "m1_c%d_u%d_b%s_r%s_s%d" % (
            r["campus"], r["u"], _fnum(r["beta"]), _fnum(r["rho"]), r["seed"])
    return "pi0_c%d_u%d_s%d" % (r["campus"], r["u"], r["seed"])


def _fnum(x):
    return ("%g" % x)


# --------------------------------------------------------------------------- #
# Config resolution (mirror of y3_p2_train.train_m1's config dict)            #
# --------------------------------------------------------------------------- #
def resolve_config(r):
    """Reconstruct the EXACT config dict train_m1 writes for this run, reusing the
    locked budget-resolution code (y1_single_run_budget + resolve_schedule)."""
    y1_budget, y1_cfg = y1_single_run_budget()
    updates_per_iter, configured = resolve_schedule(
        y1_budget, OUTER_ITERS, N_ENVS, STEPS_PER_ENV, y1_budget, 1.0)

    is_m1 = r["decider"] == "m1"
    gate = 1.0 if is_m1 else 0.0
    il_coef = 1.0 if is_m1 else 0.0
    deadline_head = True if is_m1 else False

    cell = dict(beta=r["beta"], rho=r["rho"], **COMMON_CELL,
                campus=r["campus"], u=r["u"])
    # Reorder cell keys to match the trainer's construction order for a clean diff.
    cell = {"beta": cell["beta"], "rho": cell["rho"], "eps": cell["eps"],
            "theta": cell["theta"], "mechanism": cell["mechanism"],
            "family": cell["family"], "master_seed": cell["master_seed"],
            "channel": cell["channel"], "regime": cell["regime"],
            "campus": cell["campus"], "u": cell["u"]}

    config = {
        "cell": cell,
        "seed": r["seed"],
        "outer_iters": OUTER_ITERS,
        "updates_per_iter": updates_per_iter,
        "n_envs": N_ENVS,
        "steps_per_env": STEPS_PER_ENV,
        "configured_env_steps": configured,
        "y1_budget": y1_budget,
        "budget_frac": 1.0,
        "fair_compute_ok": (configured == y1_budget),
        "gate": gate,
        "channel": cell["channel"],
        "deadline_head": bool(deadline_head),
        "override_weight": OVERRIDE_WEIGHT,
        "confirm_weight": CONFIRM_WEIGHT,
        "il_coef": il_coef,
        "buffer_capacity": 60000,
        "head_epochs": 30,
        "ppo": dict(PPO_BLOCK),
        "y1_config_ref": {k: y1_cfg[k] for k in ("updates", "n_envs", "steps_per_env")},
    }
    return config


# --------------------------------------------------------------------------- #
# Diff helpers                                                                #
# --------------------------------------------------------------------------- #
def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = "%s.%s" % (prefix, k) if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _num_eq(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-12
    return a == b


def diff_config(resolved, pilot):
    """Return {dotted_key: [pilot_val, resolved_val]} for every difference.
    Both configs are normalized so an absent deadline_head reads as False (the
    pre-fairness schema default), so adopting deadline_head=True shows as a real,
    intended diff rather than a spurious added key."""
    resolved = copy.deepcopy(resolved)
    pilot = copy.deepcopy(pilot)
    resolved.setdefault("deadline_head", False)
    pilot.setdefault("deadline_head", False)
    fr, fp = _flatten(resolved), _flatten(pilot)
    keys = set(fr) | set(fp)
    diffs = {}
    for k in sorted(keys):
        rv, pv = fr.get(k, "<absent>"), fp.get(k, "<absent>")
        if rv == "<absent>" or pv == "<absent>" or not _num_eq(rv, pv):
            diffs[k] = [pv, rv]
    return diffs


# Keys allowed to differ from the pilot, per decider.
ALLOWED_AXES = {
    "m1": {"seed", "cell.u", "cell.beta", "cell.rho", "deadline_head"},
    "pi0": {"seed", "cell.u"},
}


def verify_run(r, pilot_m1, pilot_pi0, fair_param_count):
    """Config-diff + param-count + budget assertions for one run. Returns a dict
    with the verification detail; raises AssertionError on any failure."""
    resolved = resolve_config(r)
    pilot = pilot_m1 if r["decider"] == "m1" else pilot_pi0
    diffs = diff_config(resolved, pilot)
    allowed = ALLOWED_AXES[r["decider"]]

    bad = {k: v for k, v in diffs.items() if k not in allowed}
    assert not bad, ("CONFIG-DIFF FAIL [%s]: unintended difference(s) vs pilot: %s"
                     % (r["run_id"], bad))

    if r["decider"] == "m1":
        # deadline_head must resolve to True (fair-M1) -- the one allowed extra.
        assert resolved["deadline_head"] is True, \
            "CONFIG FAIL [%s]: fair-M1 must have deadline_head=True" % r["run_id"]
        param_count = _policy_params(gate=1.0, deadline_head=True)
        param_ok = (param_count == fair_param_count)
        assert param_ok, ("PARAM FAIL [%s]: fair-M1 params %d != pilot fair-M1 %d"
                          % (r["run_id"], param_count, fair_param_count))
    else:
        assert resolved["deadline_head"] is False, \
            "CONFIG FAIL [%s]: PI-0 must have deadline_head=False" % r["run_id"]
        assert resolved["gate"] == 0.0, \
            "CONFIG FAIL [%s]: PI-0 must have gate=0" % r["run_id"]
        param_count = _policy_params(gate=0.0, deadline_head=False)
        param_ok = None  # param-count assertion is defined against fair-M1 only

    assert resolved["configured_env_steps"] == FAIR_COMPUTE_ENV_STEPS, \
        ("BUDGET FAIL [%s]: %d != %d" % (r["run_id"], resolved["configured_env_steps"],
                                         FAIR_COMPUTE_ENV_STEPS))
    assert resolved["fair_compute_ok"] is True, "BUDGET FAIL [%s]: fair_compute_ok" % r["run_id"]

    return {"config": resolved, "config_diff_vs_pilot": diffs,
            "param_count": param_count, "param_ok": param_ok,
            "budget_env_steps": resolved["configured_env_steps"],
            "budget_ok": True}


_PARAM_CACHE = {}


def _policy_params(gate, deadline_head):
    key = (gate != 0.0, deadline_head)
    if key not in _PARAM_CACHE:
        p = LatentDispatchPolicy(gate=gate, deadline_head=deadline_head)
        _PARAM_CACHE[key] = sum(t.numel() for t in p.parameters())
    return _PARAM_CACHE[key]


def ckpt_param_count(path):
    p = LatentDispatchPolicy.load(path)
    return sum(t.numel() for t in p.parameters())


# --------------------------------------------------------------------------- #
# Command construction                                                        #
# --------------------------------------------------------------------------- #
def build_cmd(r):
    py = sys.executable
    if r["decider"] == "m1":
        return ["nice", "-n", "10", py,
                os.path.join(_SCRIPTS, "y3_p15_m1.py"),
                "--beta", _fnum(r["beta"]), "--rho", _fnum(r["rho"]),
                "--eps", "0.0", "--seed", str(r["seed"]),
                "--campus", str(r["campus"]), "--u", str(r["u"]),
                "--channel", "full_class_shift", "--deadline-head",
                "--out", r["out_dir"], "--threads", "8"]
    return ["nice", "-n", "10", py,
            os.path.join(_SCRIPTS, "y3_p3_pi0.py"),
            "--beta", "1.0", "--seed", str(r["seed"]),
            "--campus", str(r["campus"]), "--u", str(r["u"]),
            "--channel", "full_class_shift",
            "--out", r["out_dir"], "--threads", "8"]


def child_env():
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = "8"
    env["PYTHONUNBUFFERED"] = "1"   # stream run.log live (metrics.csv flushes per-iter)
    return env


# --------------------------------------------------------------------------- #
# Manifest                                                                    #
# --------------------------------------------------------------------------- #
def build_manifest(runs, verifications, fair_param_count, pi0_param_count):
    entries = []
    for r in runs:
        v = verifications[r["run_id"]]
        entries.append({
            "run_id": r["run_id"],
            "decider": r["decider"],          # "m1" (fair, deadline_head=True) | "pi0"
            "group": r["group"],
            "campus": r["campus"], "u": r["u"], "beta": r["beta"],
            "rho": r["rho"], "seed": r["seed"],
            "deadline_head": (r["decider"] == "m1"),
            "regime": "storm2", "channel": "full_class_shift",
            "mechanism": "targeted", "eps": 0.0, "family": "F-NL",
            "master_seed": 12345,
            "out_dir": os.path.relpath(r["out_dir"], _ROOT),
            "checkpoint": os.path.relpath(os.path.join(r["out_dir"], "final.pt"), _ROOT),
            "metrics": os.path.relpath(os.path.join(r["out_dir"], "metrics.csv"), _ROOT),
            "config": os.path.relpath(os.path.join(r["out_dir"], "config.json"), _ROOT),
            "run_log": os.path.relpath(os.path.join(r["out_dir"], "run.log"), _ROOT),
            "cmd": build_cmd(r),
            "verify": {
                "config_diff_vs_pilot": v["config_diff_vs_pilot"],
                "param_count": v["param_count"],
                "param_ok": v["param_ok"],
                "budget_env_steps": v["budget_env_steps"],
                "budget_ok": v["budget_ok"],
            },
            "status": "pending",
            "started": None, "finished": None, "returncode": None,
        })
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": "P5 Tier-1 multi-seed training sweep",
        "sweep_dir": os.path.relpath(SWEEP_DIR, _ROOT),
        "n_runs": len(entries),
        "n_m1": sum(1 for e in entries if e["decider"] == "m1"),
        "n_pi0": sum(1 for e in entries if e["decider"] == "pi0"),
        "fair_compute_env_steps": FAIR_COMPUTE_ENV_STEPS,
        "outer_iters": OUTER_ITERS,
        "param_counts": {"fair_m1": fair_param_count, "pi0": pi0_param_count},
        "pilot_refs": {
            "m1_config": os.path.relpath(PILOT_M1_CONFIG, _ROOT),
            "m1_fair_param_ckpt": os.path.relpath(PILOT_M1FAIR_CKPT, _ROOT),
            "pi0_config": os.path.relpath(PILOT_PI0_CONFIG, _ROOT),
        },
        "note": ("PI-0 depends only on (campus,u,seed); its checkpoint serves every "
                 "(beta,rho) eval cell at that (campus,u,seed). fair-M1 = the standard "
                 "M1 (deadline_head=True). Eval with scripts/y3_p3_eval.py: for each M1 "
                 "cell (campus,u,beta,rho,seed) pass --m1-ckpt <its final.pt> "
                 "--m1-metrics <its metrics.csv> and --pi0-ckpt <PI-0 final.pt at the "
                 "matching campus,u,seed>."),
        "sweep_status": "verified",
        "concurrency": None,
        "runs": entries,
    }
    return manifest


def write_manifest(manifest):
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, MANIFEST_PATH)


def _set_status(manifest, run_id, **fields):
    for e in manifest["runs"]:
        if e["run_id"] == run_id:
            e.update(fields)
            break
    write_manifest(manifest)


# --------------------------------------------------------------------------- #
# Verification pass (prints the required report)                             #
# --------------------------------------------------------------------------- #
def run_verification(runs):
    with open(PILOT_M1_CONFIG) as fh:
        pilot_m1 = json.load(fh)
    with open(PILOT_PI0_CONFIG) as fh:
        pilot_pi0 = json.load(fh)

    fair_param_count = ckpt_param_count(PILOT_M1FAIR_CKPT)
    pi0_param_count = ckpt_param_count(PILOT_PI0_CONFIG.replace("config.json", "final.pt"))

    print("=" * 78)
    print("PRE-LAUNCH VERIFICATION  (Tier-1 sweep, %d runs)" % len(runs))
    print("=" * 78)
    print("Pilot fair-M1 checkpoint : %s  (%d params)"
          % (os.path.relpath(PILOT_M1FAIR_CKPT, _ROOT), fair_param_count))
    print("Pilot M1 config (old-M1) : %s" % os.path.relpath(PILOT_M1_CONFIG, _ROOT))
    print("Pilot PI-0 config        : %s  (%d params)"
          % (os.path.relpath(PILOT_PI0_CONFIG, _ROOT), pi0_param_count))
    print("Fair-compute budget      : %d env-steps (must equal Y1 for every run)"
          % FAIR_COMPUTE_ENV_STEPS)
    print("-" * 78)
    print("%-32s %-22s %8s %7s" % ("run_id", "config-diff vs pilot",
                                    "params", "budget"))
    print("-" * 78)

    verifications = {}
    all_ok = True
    for r in runs:
        try:
            v = verify_run(r, pilot_m1, pilot_pi0, fair_param_count)
            verifications[r["run_id"]] = v
            diff_str = ",".join("%s=%s" % (k.split(".")[-1], v["config_diff_vs_pilot"][k][1])
                                for k in sorted(v["config_diff_vs_pilot"]))
            if not diff_str:
                diff_str = "(identical)"
            pstr = "%d%s" % (v["param_count"],
                             "==pilot" if v["param_ok"] else ("" if v["param_ok"] is None else "!!"))
            print("%-32s %-22s %8s %7d"
                  % (r["run_id"], diff_str, pstr, v["budget_env_steps"]))
        except AssertionError as e:
            all_ok = False
            print("%-32s  ABORT: %s" % (r["run_id"], e))

    print("-" * 78)
    # Group-level summary of exactly which axes vary vs the pilot.
    print("Intended axes that vary vs pilot, by group:")
    for grp in ["m1_primary", "m1_regime_u90", "m1_regime_beta075",
                "pi0_primary", "pi0_regime_u90"]:
        grp_runs = [r for r in runs if r["group"] == grp]
        if not grp_runs:
            continue
        axes = set()
        for r in grp_runs:
            for k in verifications[r["run_id"]]["config_diff_vs_pilot"]:
                axes.add(k)
        axes_str = ", ".join(sorted(axes)) if axes else "(none: identical to pilot)"
        print("  %-20s (%2d runs): %s" % (grp, len(grp_runs), axes_str))
    print("-" * 78)
    print("PARAM-COUNT ASSERTION : fair-M1 runs = %d  == pilot fair-M1 ckpt = %d  -> %s"
          % (_policy_params(1.0, True), fair_param_count,
             "PASS" if _policy_params(1.0, True) == fair_param_count else "FAIL"))
    print("BUDGET ASSERTION      : every run configured_env_steps == %d  -> %s"
          % (FAIR_COMPUTE_ENV_STEPS,
             "PASS" if all(verifications[r["run_id"]]["budget_env_steps"]
                           == FAIR_COMPUTE_ENV_STEPS for r in runs) else "FAIL"))
    print("CONFIG-DIFF ASSERTION : only intended axes (+deadline_head=True for M1) "
          "differ -> %s" % ("PASS" if all_ok else "FAIL"))
    print("=" * 78)
    print("VERIFY %s  (%d/%d runs verified)"
          % ("PASS" if all_ok else "FAIL", len(verifications), len(runs)))
    print("=" * 78)

    if not all_ok:
        raise SystemExit(2)
    return verifications, fair_param_count, pi0_param_count


# --------------------------------------------------------------------------- #
# On-disk config fidelity check (diff what the program ACTUALLY resolved)     #
# --------------------------------------------------------------------------- #
def check_ondisk_config(r, resolved, timeout=120):
    """After a run launches, its trainer writes config.json BEFORE the loop. Wait
    for it and assert it equals the resolved config we diffed. This closes the
    CLAUDE.md loop: we compare the config the PROGRAM actually resolved, not only
    the one we reconstructed. Returns (ok, message)."""
    path = os.path.join(r["out_dir"], "config.json")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    ondisk = json.load(fh)
            except (json.JSONDecodeError, OSError):
                time.sleep(0.5)
                continue
            exp = copy.deepcopy(resolved)
            got = copy.deepcopy(ondisk)
            exp.setdefault("deadline_head", False)
            got.setdefault("deadline_head", False)
            fe, fg = _flatten(exp), _flatten(got)
            keys = set(fe) | set(fg)
            mism = {k: [fe.get(k, "<absent>"), fg.get(k, "<absent>")]
                    for k in keys if not _num_eq(fe.get(k), fg.get(k))}
            if mism:
                return False, "on-disk config != resolved: %s" % mism
            return True, "on-disk config == resolved"
        time.sleep(0.5)
    return False, "config.json did not appear within %ds" % timeout


# --------------------------------------------------------------------------- #
# Queue                                                                       #
# --------------------------------------------------------------------------- #
def run_queue(runs, verifications, manifest, concurrency):
    manifest["concurrency"] = concurrency
    manifest["sweep_status"] = "running"
    write_manifest(manifest)

    pending = list(runs)
    running = {}   # popen -> (run, logfh)
    checked = set()
    done = 0
    t_start = time.time()

    print("[sweep] starting queue: %d runs, concurrency=%d" % (len(runs), concurrency))
    sys.stdout.flush()

    while pending or running:
        while len(running) < concurrency and pending:
            r = pending.pop(0)
            os.makedirs(r["out_dir"], exist_ok=True)
            logfh = open(os.path.join(r["out_dir"], "run.log"), "w")
            cmd = build_cmd(r)
            popen = subprocess.Popen(cmd, cwd=_ROOT, env=child_env(),
                                     stdout=logfh, stderr=subprocess.STDOUT)
            running[popen] = (r, logfh)
            _set_status(manifest, r["run_id"], status="running",
                        started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), pid=popen.pid)
            print("[sweep] START %-32s pid=%d  (%d running, %d pending, %d done)"
                  % (r["run_id"], popen.pid, len(running), len(pending), done))
            sys.stdout.flush()

        # On-disk config fidelity check for freshly-started runs (once each).
        for popen, (r, _fh) in list(running.items()):
            if r["run_id"] not in checked and popen.poll() is None:
                ok, msg = check_ondisk_config(r, verifications[r["run_id"]]["config"],
                                              timeout=180)
                checked.add(r["run_id"])
                _set_status(manifest, r["run_id"], ondisk_config_ok=ok,
                            ondisk_config_msg=msg)
                print("[sweep] CONFIG-CHECK %-28s %s : %s"
                      % (r["run_id"], "OK" if ok else "MISMATCH", msg))
                sys.stdout.flush()
                if not ok:
                    print("[sweep] FATAL: on-disk config mismatch -> killing run %s"
                          % r["run_id"])
                    popen.terminate()

        # Reap finished runs.
        for popen in list(running):
            rc = popen.poll()
            if rc is not None:
                r, logfh = running.pop(popen)
                logfh.close()
                done += 1
                final_ok = os.path.exists(os.path.join(r["out_dir"], "final.pt"))
                status = "done" if (rc == 0 and final_ok) else "failed"
                _set_status(manifest, r["run_id"], status=status,
                            finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            returncode=rc, final_pt=final_ok)
                el = time.time() - t_start
                print("[sweep] %-6s %-32s rc=%s final_pt=%s  (%d/%d done, %.1f min elapsed)"
                      % (status.upper(), r["run_id"], rc, final_ok, done, len(runs), el / 60.0))
                sys.stdout.flush()

        if pending or running:
            time.sleep(5)

    manifest["sweep_status"] = "complete"
    manifest["completed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_manifest(manifest)
    n_done = sum(1 for e in manifest["runs"] if e["status"] == "done")
    n_fail = sum(1 for e in manifest["runs"] if e["status"] == "failed")
    print("[sweep] COMPLETE: %d done, %d failed, %.1f min total"
          % (n_done, n_fail, (time.time() - t_start) / 60.0))


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true",
                    help="resolve+diff+assert+write manifest, then exit (no training)")
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args(argv)

    runs = build_run_list()
    assert len(runs) == 29, "expected 29 runs, built %d" % len(runs)

    verifications, fair_pc, pi0_pc = run_verification(runs)
    manifest = build_manifest(runs, verifications, fair_pc, pi0_pc)
    write_manifest(manifest)
    print("[sweep] manifest written -> %s" % os.path.relpath(MANIFEST_PATH, _ROOT))

    if args.verify_only:
        print("[sweep] --verify-only: not launching. %d runs ready." % len(runs))
        return

    os.makedirs(SWEEP_DIR, exist_ok=True)
    run_queue(runs, verifications, manifest, args.concurrency)


if __name__ == "__main__":
    main()
