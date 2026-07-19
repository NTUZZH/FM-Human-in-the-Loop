"""Supervisor-oracle tests (Paper Y3, Phase P1) -- plain python, no pytest.

Run:  PYTHONPATH=src python tests/test_supervisor.py

Covers:
(a) OVERRIDE LOGIC  -- hand-built 2-candidate micro-cases with a controlled RNG:
                       the theta boundary (strict >), and the two epsilon
                       branches (miss / random-override) forced by the RNG.
(b) TARGETED ~= rho -- on a loaded campus the realized reviewed fraction over
                       reviewable decisions tracks rho.
(c) RANDOM   ~= rho -- iid Bernoulli(rho) review fraction tracks rho.
(d) CONFIRMATIONS   -- reviewed-but-not-overridden decisions are logged as
                       confirmations; the log records every decision.
(e) E0 ANCHOR       -- scripts/y3_e0_anchor.py runs and exits 0.
Prints a report and finally 'ALL SUPERVISOR TESTS PASSED'.
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.env import DispatchEnv                     # noqa: E402
from fmwos.hitl import deciders as dec                # noqa: E402
from fmwos.hitl import overlay as ov                  # noqa: E402
from fmwos.hitl.supervisor import Supervisor          # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")


class _FakeRNG:
    """Deterministic stand-in for numpy Generator: fixed .random() and
    .integers()."""
    def __init__(self, u=0.9, integer=0):
        self._u = u
        self._int = integer

    def random(self):
        return self._u

    def integers(self, n):
        return self._int


def _micro_instance():
    """Two past-due candidates A, B in one trade; p=2 each, due=0 (both late)."""
    wos = [
        {"id": "A", "trade": "D30", "p_bh": 2.0, "release_bh": 0.0,
         "due_bh": 0.0, "priority": 4, "weight": 1.0, "building": "b"},
        {"id": "B", "trade": "D30", "p_bh": 2.0, "release_bh": 0.0,
         "due_bh": 0.0, "priority": 4, "weight": 1.0, "building": "b"},
    ]
    return {"meta": {"id": "micro"}, "work_orders": wos}


def _micro_sup(w_star, theta, epsilon, rng):
    """A supervisor over the micro-instance with hand-set true weights.

    With both jobs past due and p equal, the true-ATC preferred pick is the
    higher-w* job, and the one-step pairwise improvement of serving A before B
    is 2*(w*_B - w*_A)."""
    inst = _micro_instance()
    applied = {
        "shift": {"A": 0, "B": 0},
        "w_star": dict(w_star),
        "c_star": {"A": 4, "B": 4},
        "per_order": {},
    }
    sup = Supervisor(overlay=None, instance=inst, rho=1.0, epsilon=epsilon,
                     theta=theta, mechanism="targeted", seed=0, applied=applied)
    sup.rng = rng
    return sup, inst


def test_override_logic(failures):
    print("(a) OVERRIDE LOGIC: theta boundary + epsilon branches (forced RNG)")
    A = _micro_instance()["work_orders"][0]
    B = _micro_instance()["work_orders"][1]
    cands = [A, B]

    # improvement(serve A first) = 2*(w*_B - w*_A). Take w*_A=1, w*_B=4 -> 6.
    wstar = {"A": 1.0, "B": 4.0}

    # theta boundary: honest branch (u=0.9 >= epsilon=0). improvement=6.
    #   theta=6.0 -> 6 > 6 is False -> NO override.
    #   theta=5.9 -> 6 > 5.9      -> override to B.
    sup, _ = _micro_sup(wstar, theta=6.0, epsilon=0.0, rng=_FakeRNG(u=0.9))
    ex, e = sup.review(A, cands, now=10.0, margin=0.0)
    b1 = (e["reviewed"] and not e["override"] and ex["id"] == "A"
          and abs(e["improvement"] - 6.0) < 1e-9 and e["confirmation"])
    sup, _ = _micro_sup(wstar, theta=5.9, epsilon=0.0, rng=_FakeRNG(u=0.9))
    ex, e = sup.review(A, cands, now=10.0, margin=0.0)
    b2 = (e["override"] and ex["id"] == "B" and e["preferred_pick"] == "B")
    if not b1:
        failures.append("theta boundary: improvement==theta should NOT override")
    if not b2:
        failures.append("theta boundary: improvement>theta should override to preferred")
    print("    theta==imp -> no override: %s   theta<imp -> override: %s" % (b1, b2))

    # epsilon MISS branch: u < eps/2. eps=0.5 -> half=0.25; u=0.1 -> miss.
    sup, _ = _micro_sup(wstar, theta=1.0, epsilon=0.5, rng=_FakeRNG(u=0.1))
    ex, e = sup.review(A, cands, now=10.0, margin=0.0)
    miss = (e["reviewed"] and not e["override"] and ex["id"] == "A"
            and e["noise"] == "miss")
    # epsilon RANDOM-OVERRIDE branch: eps/2 <= u < eps. u=0.4; integers->1 (B).
    sup, _ = _micro_sup(wstar, theta=1.0, epsilon=0.5, rng=_FakeRNG(u=0.4, integer=1))
    ex, e = sup.review(A, cands, now=10.0, margin=0.0)
    rnd = (e["override"] and ex["id"] == "B" and e["noise"] == "random_override")
    if not miss:
        failures.append("epsilon miss branch did not fire (should fail-to-override)")
    if not rnd:
        failures.append("epsilon random-override branch did not fire")
    print("    miss branch: %s   random-override branch: %s" % (miss, rnd))


def _loaded_files(n):
    return sorted(glob.glob(os.path.join(_INST, "c02", "replay", "150", "*.json")))[:n]


def test_targeted_fraction(failures):
    print("(b) TARGETED reviewed fraction ~= rho (loaded campus c02)")
    o = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL", master_seed=12345))
    ok = True
    for rho in [0.1, 0.25, 0.5]:
        fr = []
        for path in _loaded_files(20):
            inst = json.load(open(path))
            ap = o.apply(inst)
            sup = Supervisor(o, inst, rho=rho, epsilon=0.0, mechanism="targeted",
                             seed=301, applied=ap)
            dec.run_rule_sup(DispatchEnv(inst), "atc", sup)
            sm = sup.summary()
            if sm["n_reviewable"] > 0:
                fr.append(sm["reviewed_fraction"])
        m = float(np.mean(fr))
        good = abs(m - rho) <= 0.08
        ok = ok and good
        print("    rho=%.2f  realized=%.3f  ok=%s" % (rho, m, good))
        if not good:
            failures.append("TARGETED fraction %.3f far from rho=%.2f" % (m, rho))


def test_random_fraction(failures):
    print("(c) RANDOM reviewed fraction ~= rho")
    o = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL", master_seed=12345))
    for rho in [0.25, 0.5]:
        tot_dec = tot_rev = 0
        for path in _loaded_files(20):
            inst = json.load(open(path))
            ap = o.apply(inst)
            sup = Supervisor(o, inst, rho=rho, epsilon=0.0, mechanism="random",
                             seed=301, applied=ap)
            dec.run_rule_sup(DispatchEnv(inst), "atc", sup)
            sm = sup.summary()
            tot_dec += sm["n_decisions"]; tot_rev += sm["n_reviews"]
        frac = tot_rev / tot_dec
        good = abs(frac - rho) <= 0.05
        print("    rho=%.2f  realized(all decisions)=%.3f  ok=%s" % (rho, frac, good))
        if not good:
            failures.append("RANDOM fraction %.3f far from rho=%.2f" % (frac, rho))


def test_confirmations(failures):
    print("(d) CONFIRMATIONS logged; log covers every decision")
    o = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL", master_seed=12345))
    inst = json.load(open(_loaded_files(1)[0]))
    ap = o.apply(inst)
    sup = Supervisor(o, inst, rho=0.5, epsilon=0.0, mechanism="targeted",
                     seed=301, applied=ap)
    dec.run_rule_sup(DispatchEnv(inst), "atc", sup)
    sm = sup.summary()
    n_conf = sum(1 for e in sup.log if e["confirmation"])
    n_over = sum(1 for e in sup.log if e["override"])
    n_rev = sum(1 for e in sup.log if e["reviewed"])
    ok = (len(sup.log) == sm["n_decisions"]
          and n_conf == sm["n_confirmations"]
          and n_over == sm["n_overrides"]
          and n_conf + n_over == n_rev)          # reviewed = confirm + override
    print("    log_len=%d decisions=%d reviews=%d overrides=%d confirmations=%d ok=%s"
          % (len(sup.log), sm["n_decisions"], n_rev, n_over, n_conf, ok))
    if not ok:
        failures.append("confirmation / log accounting inconsistent")


def test_e0(failures):
    print("(e) E0 ANCHOR: scripts/y3_e0_anchor.py exits 0")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(_ROOT, "src")
    r = subprocess.run([sys.executable, os.path.join(_ROOT, "scripts", "y3_e0_anchor.py")],
                       capture_output=True, text=True, env=env)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)"
    print("    exit=%d  last: %s" % (r.returncode, last))
    if r.returncode != 0:
        failures.append("E0 anchor failed (exit %d)" % r.returncode)


def main():
    failures = []
    test_override_logic(failures)
    print()
    test_targeted_fraction(failures)
    print()
    test_random_fraction(failures)
    print()
    test_confirmations(failures)
    print()
    test_e0(failures)
    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("ALL SUPERVISOR TESTS PASSED")


if __name__ == "__main__":
    main()
