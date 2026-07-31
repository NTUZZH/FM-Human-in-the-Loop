"""Supervisor-overlay tests (Paper Y3, Phase P1) -- plain python, no pytest.

Run:  PYTHONPATH=src python tests/test_overlay.py

Covers:
(a) DETERMINISM     -- build_coeffs twice => byte-identical; Overlay.apply twice
                       => identical shift / w* maps.
(b) VARIANCE PRESERV-- total latent variance Var(xi) ~ constant across beta
                       (the sqrt(beta) mixture, proposal Sec.4.2).
(c) FEATURE HYGIENE -- base features never depend on the recorded class, the
                       building, or the campus (campus-agnostic, Sec.4.2).
(d) SEMANTICS       -- s in [-2,2]; c* = clip(c - s, 1, 4); w* matches the class
                       weight table; F-NL linear part == F-LIN linear part.
Prints a report and finally 'ALL OVERLAY TESTS PASSED'.
"""

import copy
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos.hitl import overlay as ov                  # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")


def _an_instance():
    path = sorted(glob.glob(os.path.join(_INST, "c05", "replay", "150", "*.json")))[0]
    with open(path) as fh:
        return json.load(fh)


def test_determinism(failures):
    print("(a) DETERMINISM: build_coeffs + Overlay.apply reproducible")
    c1 = ov.build_coeffs("F-NL", 4242)
    c2 = ov.build_coeffs("F-NL", 4242)
    j1 = json.dumps(c1, sort_keys=True)
    j2 = json.dumps(c2, sort_keys=True)
    if j1 != j2:
        failures.append("build_coeffs not byte-identical on rebuild")
    inst = _an_instance()
    o = ov.Overlay(ov.OverlayParams(beta=0.6, family="F-NL", master_seed=4242), coeffs=c1)
    a1 = o.apply(inst)
    a2 = o.apply(inst)
    same = (a1["shift"] == a2["shift"] and a1["w_star"] == a2["w_star"]
            and a1["c_star"] == a2["c_star"])
    if not same:
        failures.append("Overlay.apply not deterministic")
    print("    coeffs byte-equal=%s  apply-deterministic=%s" % (j1 == j2, same))


def test_variance_preservation(failures):
    print("(b) VARIANCE PRESERVATION: Var(xi) ~ constant across beta")
    coeffs = ov.get_coeffs("F-NL", 12345)
    # A sample of training-population orders (reuse the cached population).
    pop = ov.load_training_population()
    n = pop["base"].shape[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=6000, replace=False)
    # Standardized f over the sample (unit-var over the FULL population).
    feat_mean = np.asarray(coeffs["feat_mean"]); feat_std = np.asarray(coeffs["feat_std"])
    a = np.asarray(coeffs["a"])
    f = ((pop["base"][idx] - feat_mean) / feat_std) @ a
    for it in coeffs["interactions"]:
        other = pop["bucket_idx"] if it["type"] == "trade_bucket" else pop["day_idx"]
        g = ((pop["trade_idx"][idx] == it["trade_idx"]) & (other[idx] == it["other_idx"])).astype(float)
        f = f + it["b"] * (g - it["g_mean"]) / it["g_std"]
    f = (f - coeffs["f_mean"]) / coeffs["f_std"]
    z = np.random.default_rng(1).standard_normal(len(idx))
    variances = []
    for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        xi = np.sqrt(beta) * f + np.sqrt(1.0 - beta) * z
        variances.append(float(np.var(xi)))
    spread = max(variances) - min(variances)
    print("    Var(xi) by beta = %s  spread=%.3f" %
          ([round(v, 3) for v in variances], spread))
    if spread > 0.15:
        failures.append("Var(xi) not ~constant across beta (spread=%.3f)" % spread)


def test_feature_hygiene(failures):
    print("(c) FEATURE HYGIENE: base features ignore class / building / campus")
    wo = {"id": "W1", "trade": "D30", "p_bh": 2.0, "release_bh": 10.0,
          "due_bh": 80.0, "priority": 3, "weight": 2.0, "building": "0065"}
    base = ov.base_features(wo)
    ok = True
    for field, val in [("priority", 1), ("weight", 8.0), ("building", "9999")]:
        w2 = dict(wo); w2[field] = val
        if not np.array_equal(base, ov.base_features(w2)):
            ok = False
            failures.append("base features depend on %r (must be campus-agnostic)" % field)
    print("    class/weight/building independence: %s" % ok)


def test_semantics(failures):
    print("(d) SEMANTICS: s in [-2,2], c*=clip(c-s,1,4), w* table, F-NL=F-LIN+inter")
    inst = _an_instance()
    o = ov.Overlay(ov.OverlayParams(beta=0.75, family="F-NL", master_seed=12345))
    ap = o.apply(inst)
    by_id = {w["id"]: w for w in inst["work_orders"]}
    ok = True
    for wid, rec in ap["per_order"].items():
        s, c = rec["s"], rec["c_recorded"]
        if not (-2 <= s <= 2):
            ok = False; failures.append("shift out of range for %s: %d" % (wid, s))
        if rec["c_star"] != min(4, max(1, c - s)):
            ok = False; failures.append("c* wrong for %s" % wid)
        if rec["w_star"] != ov.W_OF_CLASS[rec["c_star"]]:
            ok = False; failures.append("w* wrong for %s" % wid)
        if rec["c_recorded"] != by_id[wid]["priority"]:
            ok = False; failures.append("recorded class mismatch for %s" % wid)
    lin = ov.build_coeffs("F-LIN", 12345)
    nl = ov.build_coeffs("F-NL", 12345)
    share = (lin["a"] == nl["a"]) and (len(lin["interactions"]) == 0) \
        and (len(nl["interactions"]) == 4)
    if not share:
        failures.append("F-NL linear part does not equal F-LIN, or interaction count wrong")
    print("    per-order semantics ok=%s  F-NL=F-LIN+4interactions=%s" % (ok, share))


def test_dstar(failures):
    print("(e) D_STAR: d*=r+SLA(c*); d*==recorded due when s=0; latent bit-exact")
    inst = _an_instance()
    o = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL", master_seed=12345))
    # regenerate twice -> the whole latent (incl d_star) must be byte-identical.
    a1 = o.apply(inst)
    a2 = o.apply(inst)
    bit_exact = (a1["shift"] == a2["shift"] and a1["w_star"] == a2["w_star"]
                 and a1["c_star"] == a2["c_star"] and a1["d_star"] == a2["d_star"])
    if not bit_exact:
        failures.append("overlay latent (incl d_star) not bit-exact on regenerate")
    by_id = {w["id"]: w for w in inst["work_orders"]}
    bad_d = 0
    zero_tot = zero_ok = 0
    for wid, dstar in a1["d_star"].items():
        w = by_id[wid]
        cstar = a1["c_star"][wid]
        exp = float(w["release_bh"]) + ov.SLA_OF_CLASS[cstar]
        if abs(dstar - exp) > 1e-9:
            bad_d += 1
        if a1["shift"][wid] == 0:               # unshifted => d* == recorded due
            zero_tot += 1
            if abs(dstar - float(w["due_bh"])) <= 1e-9:
                zero_ok += 1
    if bad_d:
        failures.append("d* != r + SLA(c*) for %d orders" % bad_d)
    if zero_ok != zero_tot:
        failures.append("d* != recorded due on %d/%d s==0 orders"
                        % (zero_tot - zero_ok, zero_tot))
    # the recorded due itself must satisfy d == r + SLA(c) (the invariant d*
    # generalises). Verify on this instance.
    bad_rec = sum(1 for w in inst["work_orders"]
                  if abs(float(w["due_bh"]) - (float(w["release_bh"])
                         + ov.SLA_OF_CLASS[int(w["priority"])])) > 1e-6)
    if bad_rec:
        failures.append("recorded due != r+SLA(c) on %d orders" % bad_rec)
    print("    d*=r+SLA(c*) exact (bad=%d); s=0 -> d*==recorded %d/%d; "
          "recorded d==r+SLA(c) bad=%d; latent bit-exact=%s"
          % (bad_d, zero_ok, zero_tot, bad_rec, bit_exact))


def main():
    failures = []
    test_determinism(failures)
    print()
    test_variance_preservation(failures)
    print()
    test_feature_hygiene(failures)
    print()
    test_semantics(failures)
    print()
    test_dstar(failures)
    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("ALL OVERLAY TESTS PASSED")


if __name__ == "__main__":
    main()
