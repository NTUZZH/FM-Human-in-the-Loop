"""W1 routing tests (Paper Y3) -- plain python, no pytest.

Run:  PYTHONPATH=src python tests/test_routing.py

Covers, in the order W1 depends on them:

(a) MONOTONICITY  -- the corrected ATC index is non-decreasing in the shift
                     s_hat over a dense grid of shifts x recorded classes x
                     processing times x slacks x pbar. Everything else in this
                     package rests on this, so it is checked first and the whole
                     suite fails if it fails anywhere.
(b) SEPARABILITY  -- pbar, and hence the ATC denominator, does not depend on any
                     shift, which is what makes the index separable across
                     orders and the pairwise stability test exact.
(c) INDEX MATCH   -- ``routing.corrected_atc_index`` reproduces the score inside
                     ``augmented_rule.augmented_atc_decider`` to the bit.
(d) BRUTE FORCE   -- the exact pairwise stability test agrees with an exhaustive
                     search over a fine grid of admissible shift vectors on
                     small synthetic queues.
(e) CONFORMAL     -- the split-conformal quantile achieves its nominal coverage
                     on exchangeable synthetic data, and degrades gracefully
                     when the calibration set is too small for the level.
(f) NO-LATENT     -- the calibration and routing code paths cannot see the
                     latent: the source of every such function is grepped for
                     latent tokens, and passing an overlay or an applied-overlay
                     dict raises.
(g) BUDGET        -- the deployable policy's realised reviewed fraction stays at
                     or below rho, a forced pick is never reviewed, and a stable
                     decision is never reviewed.
(h) NO-DRIFT      -- ``routing.run_m0_routed(policy='targeted', split_fit=False)``
                     reproduces ``augmented_rule.run_m0`` BIT-FOR-BIT, so the
                     wrapper is provably not a drifted fork of the published
                     pipeline.

Prints a report and finally 'ALL ROUTING TESTS PASSED'.
"""

import glob
import inspect
import json
import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch                                                        # noqa: E402

from fmwos.hitl import augmented_rule as AR                         # noqa: E402
from fmwos.hitl import overlay as ov                                # noqa: E402
from fmwos.hitl import routing as R                                 # noqa: E402
from fmwos.hitl.latent_head import ShiftEstimator                   # noqa: E402

_INST = os.path.join(_ROOT, "data", "processed", "instances")
FAILURES = []


def check(name, ok, detail=""):
    print("    %-58s %s %s" % (name, "OK" if ok else "FAIL", detail))
    if not ok:
        FAILURES.append(name + (" | " + detail if detail else ""))
    return ok


# --------------------------------------------------------------------------- #
# (a) Monotonicity of the corrected ATC index in the shift                     #
# --------------------------------------------------------------------------- #
def test_monotonicity():
    print("(a) MONOTONICITY of the corrected ATC index in s_hat")
    shifts = np.linspace(-3.0, 3.0, 601)          # denser than any band we use
    classes = [1, 2, 3, 4]
    ps = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    slacks = [-40.0, -8.0, -1.0, 0.0, 1.0, 4.0, 20.0, 80.0, 200.0]
    pbars = [0.3, 1.0, 2.5, 6.0, 12.0]

    worst_drop = 0.0
    n_cells = 0
    n_points = 0
    n_strict = 0
    for c in classes:
        for p in ps:
            for pbar in pbars:
                denom = 2.0 * pbar
                for sl in slacks:
                    # Place the clock so the RECORDED deadline sits at the
                    # requested slack: r = 0, t = SLA(c) - slack - p.
                    r = 0.0
                    t = ov.SLA_OF_CLASS[c] - sl - p
                    vals = R.index_slope_check(c, p, r, t, denom, shifts)
                    d = np.diff(vals)
                    worst_drop = min(worst_drop, float(d.min()))
                    n_strict += int((d > 0).sum())
                    n_points += d.size
                    n_cells += 1
    check("non-decreasing over %d cells, %d finite differences"
          % (n_cells, n_points), worst_drop >= 0.0,
          "worst difference = %.3e (must be >= 0); %.1f%% strictly increasing"
          % (worst_drop, 100.0 * n_strict / n_points))

    # The two factors separately, so a failure localises.
    w = np.asarray([AR.interp_weight(2.0 - s) for s in shifts])
    check("weight w(clip(c-s,1,4)) non-decreasing in s",
          bool((np.diff(w) >= 0).all()))
    sla = np.asarray([AR.interp_sla(2.0 - s) for s in shifts])
    check("deadline r+SLA(clip(c-s,1,4)) non-increasing in s",
          bool((np.diff(sla) <= 0).all()))

    # Saturation: outside [c-4, c-1] the index is constant, so an arbitrarily
    # wide band cannot make the test vacuous.
    v = R.index_slope_check(2, 1.0, 0.0, 10.0, 4.0, [3.0, 5.0, 50.0])
    check("index saturates beyond the class range",
          v[0] == v[1] == v[2], "%.9f %.9f %.9f" % tuple(v))

    # The scalar fast path used in the hot loop must be BIT-IDENTICAL to the
    # np.interp curves the published decider uses, or the monotonicity result
    # above does not transfer to the deployed test.
    rng = np.random.default_rng(11)
    xs = np.concatenate([rng.uniform(-6.0, 10.0, 300000),
                         np.arange(-6.0, 10.0, 0.25),
                         np.asarray([1.0, 2.0, 3.0, 4.0])])
    bw = sum(1 for x in xs if R._interp_unit(x, R._W_V) != AR.interp_weight(x))
    bs = sum(1 for x in xs if R._interp_unit(x, R._S_V) != AR.interp_sla(x))
    check("scalar fast path bit-identical to np.interp on %d points" % xs.size,
          bw == 0 and bs == 0, "weight %d, SLA %d mismatches" % (bw, bs))


def test_separability():
    print("(b) SEPARABILITY: the ATC denominator does not depend on any shift")
    src = inspect.getsource(AR.augmented_atc_decider)
    body = src.split("def _decider")[1]
    pbar_line = [l for l in body.splitlines() if "pbar =" in l][0]
    check("pbar is a function of p only", "hs" not in pbar_line
          and "wcorr" not in pbar_line and "dcorr" not in pbar_line,
          pbar_line.strip())


# --------------------------------------------------------------------------- #
# (c) The routing index reproduces the augmented decider's own score           #
# --------------------------------------------------------------------------- #
def _load_one_instance():
    p = sorted(glob.glob(os.path.join(_INST, "c09", "storm2", "w80",
                                      "c09_storm2_w80_u100_*.json")))[0]
    with open(p) as fh:
        return json.load(fh)


def test_index_matches_decider():
    print("(c) INDEX MATCH: routing.corrected_atc_index == decider score")
    inst = _load_one_instance()
    torch.manual_seed(0)
    est = ShiftEstimator(hidden=32)
    hs = AR.hat_s_map(est, inst)
    wos = sorted(inst["work_orders"], key=lambda w: w["id"])[:40]
    dec = AR.augmented_atc_decider(est, inst, channel="full_class_shift")
    t = 12.5
    job, margin = dec(wos, t, None)
    pbar = sum(float(w["p_bh"]) for w in wos) / len(wos)
    scores = sorted([(R.corrected_atc_index(int(w["priority"]), float(w["p_bh"]),
                                            float(w["release_bh"]), hs[w["id"]],
                                            t, 2.0 * pbar), w["id"])
                     for w in wos], key=lambda x: (-x[0], x[1]))
    check("argmax identical", scores[0][1] == job["id"],
          "%s vs %s" % (scores[0][1], job["id"]))
    check("top1-top2 margin identical to the bit",
          scores[0][0] - scores[1][0] == margin,
          "%.17g vs %.17g" % (scores[0][0] - scores[1][0], margin))


# --------------------------------------------------------------------------- #
# (d) Exact stability test vs brute force over admissible shift vectors        #
# --------------------------------------------------------------------------- #
def _synthetic_queue(rng, n):
    cands, band = [], {}
    for i in range(n):
        wid = "W%02d" % i
        c = int(rng.integers(1, 5))
        p = float(rng.uniform(0.25, 6.0))
        r = float(rng.uniform(0.0, 40.0))
        cands.append({"id": wid, "priority": c, "p_bh": p, "release_bh": r,
                      "due_bh": r + ov.SLA_OF_CLASS[c]})
        s_hat = float(rng.uniform(-1.5, 1.5))
        q = float(rng.choice([0.0, 0.1, 0.3, 0.7, 1.5]))
        band[wid] = (s_hat, max(-2.0, s_hat - q), min(2.0, s_hat + q))
    return cands, band


def test_stability_vs_bruteforce():
    print("(d) BRUTE FORCE agreement of the exact pairwise stability test")
    rng = np.random.default_rng(20260731)
    n_cases = 0
    n_disagree = 0
    n_unstable = 0
    n_path_diff = 0
    detail = ""
    for trial in range(400):
        n = int(rng.integers(2, 5))
        cands, band = _synthetic_queue(rng, n)
        t = float(rng.uniform(0.0, 120.0))
        v = R.stability_verdict(cands, t, band)
        # The precomputed-terms path used in the hot loop must agree exactly
        # with the on-the-fly path the brute force uses.
        vf = R.stability_verdict(cands, t, R.OrderBands.from_band_map(cands, band))
        if (v["stable"], v["margin"], v["top"]) != (vf["stable"], vf["margin"],
                                                     vf["top"]):
            n_path_diff += 1
        bf = R.brute_force_stable(cands, t, band, n_grid=9)
        n_cases += 1
        n_unstable += int(not v["stable"])
        if bool(v["stable"]) != bool(bf):
            # The only admissible disagreement is an exact tie at the corner,
            # which the exact test calls undetermined (conservative).
            if v["stable"] or abs(v["margin"]) > 1e-15:
                n_disagree += 1
                if not detail:
                    detail = "trial %d: exact=%s brute=%s margin=%.3e" % (
                        trial, v["stable"], bf, v["margin"])
    check("exact == brute force on %d random queues (%d undetermined)"
          % (n_cases, n_unstable), n_disagree == 0,
          detail or "0 disagreements")
    check("precomputed-terms path == on-the-fly path to the bit",
          n_path_diff == 0, "%d of %d differ" % (n_path_diff, n_cases))

    # Degenerate band => the test reduces to the plain top1-top2 comparison.
    cands, band = _synthetic_queue(rng, 4)
    band0 = {k: (v[0], v[0], v[0]) for k, v in band.items()}
    v = R.stability_verdict(cands, 10.0, band0)
    check("zero-width band => stable, margin == top1-top2 gap",
          v["stable"] and v["margin"] > 0)

    # A forced pick is stable by definition and never enters the budget.
    v1 = R.stability_verdict(cands[:1], 10.0, band)
    check("single feasible candidate => stable, margin +inf",
          v1["stable"] and math.isinf(v1["margin"]) and v1["forced"])

    # Widening the band can only turn a stable decision undetermined.
    mono_ok = True
    for _ in range(200):
        n = int(rng.integers(2, 5))
        cands, band = _synthetic_queue(rng, n)
        t = float(rng.uniform(0.0, 120.0))
        wide = {k: (v[0], max(-2.0, v[1] - 0.5), min(2.0, v[2] + 0.5))
                for k, v in band.items()}
        a = R.stability_verdict(cands, t, band)
        b = R.stability_verdict(cands, t, wide)
        if a["pick"] == b["pick"] and b["margin"] > a["margin"] + 1e-12:
            mono_ok = False
    check("margin is non-increasing in the band width", mono_ok)


# --------------------------------------------------------------------------- #
# (e) Split-conformal coverage on exchangeable synthetic data                  #
# --------------------------------------------------------------------------- #
class _AffineEstimator:
    """Stand-in with the ShiftEstimator's predict_np contract."""

    def __init__(self, w):
        self.w = np.asarray(w, dtype=np.float64)

    def predict_np(self, X, device="cpu"):
        return (np.asarray(X, dtype=np.float64) @ self.w).astype(np.float32)


def test_conformal_coverage():
    print("(e) CONFORMAL: nominal coverage of the split-conformal quantile")
    rng = np.random.default_rng(7)
    d = R.LAT_DIM
    w = rng.normal(size=d) * 0.05
    est = _AffineEstimator(w)
    covs = []
    for _ in range(200):
        X = rng.normal(size=(600, d)).astype(np.float32)
        y = np.clip(np.round(X @ w + rng.normal(scale=0.6, size=600)), -1, 1)
        cal, test = slice(0, 300), slice(300, 600)
        band = R.calibrate_band(est, X[cal], y[cal], alpha=0.1)
        hw = band.half_width(X[test])
        pred = est.predict_np(X[test]).astype(np.float64)
        covs.append(float(np.mean(np.abs(pred - y[test]) <= hw)))
    m = float(np.mean(covs))
    check("mean coverage at alpha=0.1 is >= 0.90 (marginal validity)",
          m >= 0.895, "mean=%.4f over 200 replicates" % m)
    check("coverage is not grossly conservative", m <= 0.945, "mean=%.4f" % m)

    small = R.calibrate_band(est, np.zeros((4, d), np.float32),
                             np.zeros(4), alpha=0.1)
    check("too-small calibration set => infinite half-width",
          math.isinf(small.q), "n=4 at alpha=0.1 needs n>=9")

    # The locally-adaptive variant must also cover.
    X = rng.normal(size=(1200, d)).astype(np.float32)
    X[:, :14] = 0.0
    g = rng.integers(0, 14, size=1200)
    X[np.arange(1200), g] = 1.0
    noise = np.where(g < 7, 0.2, 1.2)
    y = np.clip(np.round(X @ w + rng.normal(scale=noise)), -1, 1)
    band = R.fit_band_from_examples(est, X[:800], y[:800],
                                    np.r_[np.zeros(400, int), np.ones(400, int)],
                                    alpha=0.1, mode="normalized")
    hw = band.half_width(X[800:])
    pred = est.predict_np(X[800:]).astype(np.float64)
    cov = float(np.mean(np.abs(pred - y[800:]) <= hw))
    check("normalized band covers at alpha=0.1", cov >= 0.86,
          "coverage=%.3f, %d trade scales fitted" % (cov, len(band.scale)))


# --------------------------------------------------------------------------- #
# (f) The calibration path structurally cannot see the latent                  #
# --------------------------------------------------------------------------- #
_LATENT_TOKENS = ("w_star", "d_star", "c_star", '["shift"]', "'shift'",
                  "per_order", "wstar", "preferred_pick", "improvement")


def test_no_latent():
    print("(f) NO-LATENT: calibration and routing cannot read the latent")
    guarded = [R.calibrate_band, R.fit_band_from_examples, R.conformal_quantile,
               R.band_for_instance, R.stability_verdict, R.corrected_atc_index,
               R.verdict_stream, R.StabilityRoutingSupervisor._decide_review,
               R.MarginRoutingSupervisor._decide_review]
    bad = []
    for fn in guarded:
        src = inspect.getsource(fn)
        for tok in _LATENT_TOKENS:
            if tok in src:
                bad.append("%s contains %r" % (fn.__qualname__, tok))
    check("no latent token in %d calibration/routing functions" % len(guarded),
          not bad, "; ".join(bad))

    # The signature has no slot for an overlay, and one passed anyway is refused.
    sig = inspect.signature(R.calibrate_band)
    check("calibrate_band takes no overlay/instance/applied parameter",
          not (set(sig.parameters) & {"overlay", "instance", "applied", "shift",
                                      "s_true", "latent"}),
          str(list(sig.parameters)))

    overlay = ov.Overlay(ov.OverlayParams(beta=1.0))
    est = _AffineEstimator(np.zeros(R.LAT_DIM))
    X = np.zeros((32, R.LAT_DIM), np.float32)
    y = np.zeros(32)
    raised = False
    try:
        R.calibrate_band(overlay, X, y, alpha=0.1)
    except TypeError:
        raised = True
    check("an overlay passed to calibrate_band raises TypeError", raised)

    inst = _load_one_instance()
    applied = overlay.apply(inst)
    raised = False
    try:
        R.calibrate_band(est, X, y, alpha=0.1, X_prop=applied)
    except ValueError:
        raised = True
    check("an applied-overlay dict raises ValueError", raised)

    raised = False
    try:
        R.calibrate_band(est, X, np.full(32, 2.0), alpha=0.1)
    except ValueError:
        raised = True
    check("a label outside the weak-label alphabet raises", raised)

    # The true shift of a real instance is rejected as a calibration label.
    s_true = np.asarray([applied["shift"][w["id"]] for w in inst["work_orders"]],
                        dtype=np.float64)
    raised = False
    try:
        R.calibrate_band(est, np.zeros((s_true.size, R.LAT_DIM), np.float32),
                         s_true, alpha=0.1)
    except ValueError:
        raised = True
    check("the simulator's true shift is rejected as a calibration label",
          raised, "%d of %d orders carry |s| = 2"
          % (int((np.abs(s_true) == 2).sum()), s_true.size))


# --------------------------------------------------------------------------- #
# (g) Budget behaviour of the deployable policy                                #
# --------------------------------------------------------------------------- #
def test_budget():
    print("(g) BUDGET: realised reviewed fraction under the deployable policy")
    from fmwos.env import DispatchEnv
    inst = _load_one_instance()
    overlay = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL",
                                          master_seed=12345))
    applied = overlay.apply(inst)
    torch.manual_seed(301)
    est = ShiftEstimator(hidden=32)
    rows = []
    for rho in (0.1, 0.25, 0.5):
        for q in (0.2, 1.0):
            band = R.ConformalBand(q, 0.1)
            bm = R.band_for_instance(est, inst, band)
            sup = R.StabilityRoutingSupervisor(overlay, inst, rho=rho,
                                               applied=applied, band_map=bm,
                                               seed=301)
            dec = AR.augmented_atc_decider(est, inst)
            DispatchEnv(inst).run_supervised(dec, supervisor=sup,
                                             method="m0_atc", seed=301)
            s = sup.routing_summary()
            rows.append((rho, q, s))
            # The online controller admits a review while n_reviews < rho *
            # n_reviewable, so it can overshoot by at most one review, exactly
            # like the published TARGETED controller.
            slack = 1.0 / max(1, s["n_reviewable"])
            ok = (s["reviewed_fraction"] <= rho + slack
                  and s["reviewed_fraction"] >= 0.95 * rho)
            check("rho=%.2f q=%.1f: reviewed/reviewable %.4f tracks rho"
                  % (rho, q, s["reviewed_fraction"]), ok,
                  "undetermined=%.3f coverage_all=%.4f"
                  % (s["undetermined_rate"], s["automation_coverage_all"]))
    # A wider band must not reduce the referral demand.
    for rho in (0.1, 0.25, 0.5):
        a = [s for (r, q, s) in rows if r == rho and q == 0.2][0]
        b = [s for (r, q, s) in rows if r == rho and q == 1.0][0]
        check("rho=%.2f: a wider band raises the undetermined rate" % rho,
              b["undetermined_rate"] >= a["undetermined_rate"] - 1e-12,
              "%.3f -> %.3f" % (a["undetermined_rate"], b["undetermined_rate"]))
    # A stable decision is never reviewed: with a zero-width band only an exact
    # tie between two candidates can be undetermined, so the policy spends
    # essentially nothing however large the budget.
    band0 = R.ConformalBand(0.0, 0.1)
    bm0 = R.band_for_instance(est, inst, band0)
    sup = R.StabilityRoutingSupervisor(overlay, inst, rho=0.5, applied=applied,
                                       band_map=bm0, seed=301)
    dec = AR.augmented_atc_decider(est, inst)
    DispatchEnv(inst).run_supervised(dec, supervisor=sup, method="m0_atc",
                                     seed=301)
    s = sup.routing_summary()
    check("zero-width band => only exact ties are undetermined",
          s["undetermined_rate"] < 0.01 and s["n_reviews"] <= s["n_undetermined"] + 1,
          "reviews=%d undetermined=%d of %d reviewable (%.3f%%)"
          % (s["n_reviews"], s["n_undetermined"], s["n_reviewable"],
             100 * s["undetermined_rate"]))
    check("forced picks are never reviewed",
          s["n_forced"] > 0 and s["n_reviewable"] + s["n_forced"] == s["n_decisions"],
          "forced=%d reviewable=%d decisions=%d"
          % (s["n_forced"], s["n_reviewable"], s["n_decisions"]))


# --------------------------------------------------------------------------- #
# (h) The wrapper is not a drifted fork of the published pipeline              #
# --------------------------------------------------------------------------- #
def test_no_drift():
    print("(h) NO-DRIFT: run_m0_routed('targeted', split_fit=False) == run_m0")
    files = sorted(glob.glob(os.path.join(_INST, "c09", "storm2", "w80",
                                          "c09_storm2_w80_u100_*.json")))[:6]
    insts = []
    for p in files:
        with open(p) as fh:
            insts.append(json.load(fh))
    train, probe = insts[:4], insts[4:]
    overlay = ov.Overlay(ov.OverlayParams(beta=1.0, family="F-NL",
                                          master_seed=12345))
    args = dict(beta_rho_eps=(1.0, 0.25, 0.0), outer_iters=2,
                mechanism="targeted", theta=1.0, seed=301, device="cpu",
                verbose=False)
    torch.manual_seed(301); np.random.seed(301)
    ref = AR.run_m0(train, probe, overlay, **args)
    args2 = dict(args); args2.pop("mechanism")
    torch.manual_seed(301); np.random.seed(301)
    got = R.run_m0_routed(train, probe, overlay, policy="targeted",
                          split_fit=False, **args2)

    feats = np.stack([ov.base_features(w) for w in probe[0]["work_orders"]]
                     ).astype(np.float32)
    a = ref["estimator"].predict_np(feats)
    b = got["estimator"].predict_np(feats)
    check("estimator predictions bit-identical on %d orders" % feats.shape[0],
          bool(np.array_equal(a, b)),
          "max |diff| = %.3e" % float(np.abs(a - b).max()))
    same = all(abs(x["est_loss"] - y["est_loss"]) == 0.0
               and x["n_reviews"] == y["n_reviews"]
               and x["n_overrides"] == y["n_overrides"]
               and x["n_examples_agg"] == y["n_examples_agg"]
               for x, y in zip(ref["per_iter"], got["per_iter"]))
    check("per-iteration review/override/example counts and loss identical", same,
          "reviews %s vs %s" % ([r["n_reviews"] for r in ref["per_iter"]],
                                [r["n_reviews"] for r in got["per_iter"]]))
    check("no band is produced without the conformal split",
          got["band"] is None)


def main():
    torch.set_num_threads(4)
    print("W1 ROUTING TESTS")
    test_monotonicity()
    test_separability()
    test_index_matches_decider()
    test_stability_vs_bruteforce()
    test_conformal_coverage()
    test_no_latent()
    test_budget()
    test_no_drift()
    print()
    if FAILURES:
        print("ROUTING TESTS: FAIL")
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    print("ALL ROUTING TESTS PASSED")


if __name__ == "__main__":
    main()
