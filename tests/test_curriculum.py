"""Policy-v2 curriculum tests -- plain python script (no pytest).

Run:  PYTHONPATH=src python tests/test_curriculum.py

Covers the backward-compatible ``--curriculum {v1,v2}`` knob added to
``fmwos.train`` (docs/decision_log.md "Policy v2 retraining planned"):

(1) KNOB FREQUENCIES: instantiate the v2 sampler with a seeded RNG, draw 200
    episode specs, and check the empirical knob frequencies are within +-10
    percentage points of the target weights --
      * replay half: m=1.0 w.p. 1/3, m in {0.5,0.6,0.8} uniform (2/9 each);
      * generator half: crew_multiplier {0.5,0.6,0.8,1.0} ~ {0.2,0.25,0.25,0.3},
        arrival_multiplier {1.0,1.25,1.5} ~ {0.5,0.3,0.2}.

(2) SCALED INSTANCES: materialize scaled replay draws and assert
    ``meta.crew_multiplier`` is set and the per-trade technician counts follow
    ``max(1, round(count*m))`` (i.e. really scaled, cache left intact).

(3) V1 UNAFFECTED: draw from the v1 sampler and assert no replay instance is
    ever crew-scaled (``tightness.scale_crew`` never applied: no
    ``meta.crew_multiplier`` and no ``_m`` id suffix on replay draws); the
    generator half keeps its native crew_multiplier in {0.75, 1.0}.

(4) SMOKE TRAIN: run ``python -m fmwos.train --smoke --curriculum v2 --seed 998``
    to completion (cpu, single-threaded) and check config.json records
    curriculum v2 + its knobs and curves.csv is all-finite incl. dev_wwt_tight.

Prints a report and finally 'ALL CURRICULUM TESTS PASSED'.
"""

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.train import InstanceSampler          # noqa: E402
from fmwos import tightness                       # noqa: E402

CAMPUSES = [5, 9, 10, 12]
SIZES = [150, 400]
SIZES_SMALL = [50]         # fast path for the v1 sweep (materializes instances)
FREQ_SEED = 202            # comfortable margin (empirical max-dev ~0.045 over 200)
TOL = 0.10                 # +-10 percentage points


def _frac(items, key, val):
    return sum(1 for it in items if it[key] == val) / len(items) if items else float("nan")


def _check(name, emp, tgt, failures):
    ok = abs(emp - tgt) <= TOL
    print("      %-22s emp=%.3f  tgt=%.3f  d=%+.3f  %s"
          % (name, emp, tgt, emp - tgt, "OK" if ok else "FAIL"))
    if not ok:
        failures.append("%s: emp=%.3f tgt=%.3f (|d|=%.3f > %.2f)"
                        % (name, emp, tgt, abs(emp - tgt), TOL))


# --------------------------------------------------------------------------- #
def test_knob_frequencies(failures):
    print("(1) v2 knob frequencies over 200 episode specs (seed=%d)" % FREQ_SEED)
    s = InstanceSampler(CAMPUSES, SIZES, FREQ_SEED, curriculum="v2")
    if not s.gen_enabled:
        failures.append("generator not enabled; cannot check generator-half knobs")
        return
    specs = [s._draw_spec_v2() for _ in range(200)]
    replay = [sp for sp in specs if sp["track"] == "replay"]
    gen = [sp for sp in specs if sp["track"] == "generator"]
    print("    n_replay=%d  n_generator=%d" % (len(replay), len(gen)))

    print("    replay half -- crew_multiplier:")
    _check("m=1.0", _frac(replay, "crew_multiplier", 1.0), 1.0 / 3.0, failures)
    for m in (0.5, 0.6, 0.8):
        _check("m=%.1f" % m, _frac(replay, "crew_multiplier", m), 2.0 / 9.0, failures)
    # replay half never touches arrival
    if any(sp["arrival_multiplier"] != 1.0 for sp in replay):
        failures.append("replay-half spec has arrival_multiplier != 1.0")

    print("    generator half -- crew_multiplier:")
    for c, w in [(0.5, 0.2), (0.6, 0.25), (0.8, 0.25), (1.0, 0.3)]:
        _check("crew=%.2f" % c, _frac(gen, "crew_multiplier", c), w, failures)
    print("    generator half -- arrival_multiplier:")
    for a, w in [(1.0, 0.5), (1.25, 0.3), (1.5, 0.2)]:
        _check("arr=%.2f" % a, _frac(gen, "arrival_multiplier", a), w, failures)

    # knobs stay inside their declared support
    bad_c = {sp["crew_multiplier"] for sp in gen} - {0.5, 0.6, 0.8, 1.0}
    bad_a = {sp["arrival_multiplier"] for sp in gen} - {1.0, 1.25, 1.5}
    bad_m = {sp["crew_multiplier"] for sp in replay} - {0.5, 0.6, 0.8, 1.0}
    if bad_c or bad_a or bad_m:
        failures.append("out-of-support knobs: crew=%r arr=%r m=%r"
                        % (bad_c, bad_a, bad_m))


# --------------------------------------------------------------------------- #
def test_scaled_instances(failures):
    print("(2) scaled replay instances: meta.crew_multiplier + scaled tech counts")
    s = InstanceSampler(CAMPUSES, SIZES, FREQ_SEED, curriculum="v2")
    checked = 0
    for _ in range(2000):
        spec = s._draw_spec_v2()
        if spec["track"] != "replay" or spec["crew_multiplier"] >= 1.0:
            continue
        m = spec["crew_multiplier"]
        with open(spec["path"]) as fh:
            base = json.load(fh)
        scaled = s._materialize(spec)

        # meta.crew_multiplier is set to m; id carries the _m suffix
        if scaled["meta"].get("crew_multiplier") != m:
            failures.append("scaled meta.crew_multiplier=%r != %r"
                            % (scaled["meta"].get("crew_multiplier"), m))
        if "_m" not in scaled["meta"]["id"]:
            failures.append("scaled id has no _m suffix: %s" % scaled["meta"]["id"])

        # per-trade counts follow max(1, round(count*m)); total strictly reduced
        base_c = Counter(t["trade"] for t in base["technicians"])
        scaled_c = Counter(t["trade"] for t in scaled["technicians"])
        for tr, cnt in base_c.items():
            exp = max(1, int(round(cnt * m)))
            if scaled_c[tr] != exp:
                failures.append("trade %s: scaled tech %d != expected %d (m=%s)"
                                % (tr, scaled_c[tr], exp, m))
        n_base, n_scaled = len(base["technicians"]), len(scaled["technicians"])
        if not (n_scaled < n_base):
            failures.append("scaled techs %d not < base %d (m=%s, id=%s)"
                            % (n_scaled, n_base, m, base["meta"]["id"]))
        # the shared cache original must remain UNSCALED (deep-copy guarantee)
        cached = s._cache.get(spec["path"])
        if cached is not None and len(cached["technicians"]) != n_base:
            failures.append("cache mutated by scaling: %d != %d"
                            % (len(cached["technicians"]), n_base))

        print("      %-24s m=%.1f  techs %d -> %d  OK"
              % (base["meta"]["id"], m, n_base, n_scaled))
        checked += 1
        if checked >= 3:
            break
    if checked == 0:
        failures.append("no scaled replay draws produced in 2000 tries")


# --------------------------------------------------------------------------- #
def test_v1_unaffected(failures):
    print("(3) v1 sampler draws are unaffected (no crew scaling ever)")
    s = InstanceSampler(CAMPUSES, SIZES_SMALL, 202, curriculum="v1")
    n_replay = n_gen = 0
    for _ in range(200):
        inst = s.sample()
        track = inst["meta"].get("track")
        cm = inst["meta"].get("crew_multiplier")
        if track == "generator":
            n_gen += 1
            if cm not in (0.75, 1.0):
                failures.append("v1 generator crew_multiplier=%r not in {0.75,1.0}" % cm)
        else:  # replay: scale_crew must NEVER have been applied
            n_replay += 1
            if cm is not None:
                failures.append("v1 replay instance carries crew_multiplier=%r" % cm)
            if "_m" in inst["meta"]["id"]:
                failures.append("v1 replay id shows scaling: %s" % inst["meta"]["id"])
    print("    replay=%d (never scaled)  generator=%d (crew in {0.75,1.0})"
          % (n_replay, n_gen))
    if n_replay == 0:
        failures.append("v1 sweep drew no replay instances (cannot assert no-scaling)")


# --------------------------------------------------------------------------- #
def test_smoke_train_v2(failures):
    print("(4) SMOKE TRAIN: --curriculum v2 --seed 998 (cpu, single-thread)")
    out_dir = tempfile.mkdtemp(prefix="v2smoke_")
    cmd = [sys.executable, "-m", "fmwos.train", "--smoke", "--curriculum", "v2",
           "--seed", "998", "--out", out_dir, "--device", "cpu"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(_ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Be a polite neighbour to the running abl/dyneval sessions: single-threaded.
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
    if cfg.get("curriculum") != "v2":
        failures.append("config.json curriculum=%r != 'v2'" % cfg.get("curriculum"))
    knobs = cfg.get("curriculum_knobs", {})
    if knobs.get("gen_arrival_weights") != [0.5, 0.3, 0.2] \
            or knobs.get("gen_crew_weights") != [0.2, 0.25, 0.25, 0.3]:
        failures.append("config.json curriculum_knobs not recorded: %r" % knobs)

    curves_path = os.path.join(out_dir, "curves.csv")
    with open(curves_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        failures.append("curves.csv is empty")
        return
    if "dev_wwt_tight" not in rows[0]:
        failures.append("curves.csv missing dev_wwt_tight column: %r" % list(rows[0]))
        return
    num_cols = ("mean_train_return", "dev_wwt_mean", "dev_wwt_tight",
                "entropy", "value_loss", "seconds")
    all_finite = True
    for row in rows:
        for key in num_cols:
            if not math.isfinite(float(row[key])):
                all_finite = False
                failures.append("curves.csv non-finite %s=%r at update %s"
                                % (key, row[key], row["update"]))
    last = rows[-1]
    print("    curves rows=%d all-finite=%s  last: dev_wwt=%s dev_tight=%s"
          % (len(rows), all_finite, last["dev_wwt_mean"], last["dev_wwt_tight"]))


# --------------------------------------------------------------------------- #
def main():
    failures = []
    test_knob_frequencies(failures)
    test_scaled_instances(failures)
    test_v1_unaffected(failures)
    test_smoke_train_v2(failures)

    print()
    if failures:
        print("CURRICULUM TESTS FAILED (%d):" % len(failures))
        for f in failures:
            print("  - %s" % f)
        sys.exit(1)
    print("ALL CURRICULUM TESTS PASSED")


if __name__ == "__main__":
    main()
