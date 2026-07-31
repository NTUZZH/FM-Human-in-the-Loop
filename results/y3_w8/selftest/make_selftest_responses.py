"""SYNTHETIC TEST FIXTURES for the W8 pilot analysis. NOT DATA. NOT RESULTS.

Everything this file writes is machine-generated from a known statistical model
with a fixed seed. No practitioner has been recruited, no response has been
collected, and nothing produced here may appear in the manuscript as a finding.
Its only two jobs are:

  (a) exercise ``scripts/y3_w8_pilot_analyse.py`` end to end, including its
      degradation path at one and two responses, so that the analysis is known
      to run before any real sheet arrives;
  (b) compute the pre-registered power statement by simulation, so the
      pre-registration says what this design can and cannot detect.

Every file it writes is under ``results/y3_w8/selftest/`` and every generated
response CSV carries ``SYNTHETIC`` in its name.

GENERATIVE MODEL (the fixture's, not the paper's)
-------------------------------------------------
For pair i with observable attribute differences d = (trade prior, log job size,
days waited), standardised over the fifty pairs, a consensus latent difference is

    Delta_i = sqrt(beta) * (a . d_i) + sqrt(1 - beta) * z_i,   z_i ~ N(0, 1)

with a fixed coefficient vector a normalised so that a . d has unit variance.
The consensus answer is A when Delta_i > 0. Each simulated rater reports the
consensus with probability 1 - q and the other order with probability q,
independently per item, and answers a repeated pair with an independent draw, so
within-rater consistency is (1-q)^2 + q^2 by construction.

Usage
-----
  # fixtures for the analysis smoke test (writes several response sets)
  python results/y3_w8/selftest/make_selftest_responses.py --fixtures

  # pre-registered power statement (slow-ish; a few minutes on four cores)
  python results/y3_w8/selftest/make_selftest_responses.py --power
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import y3_w8_pilot_analyse as A  # noqa: E402

MANIFEST = ROOT / "pilot" / "y3_w8_manifest.csv"
SEED = 4242

# Fixture scenarios: (name, n_raters, beta, per-rater error rate q)
FIXTURES = [
    ("R5_beta0.50_q0.20", 5, 0.50, 0.20),   # the design's expected operating point
    ("R3_beta0.50_q0.20", 3, 0.50, 0.20),   # the smallest sample the plan targets
    ("R2_beta0.50_q0.20", 2, 0.50, 0.20),   # the degradation floor the pre-registration names
    ("R1_beta0.50_q0.20", 1, 0.50, 0.20),   # single response: most statistics vanish
    ("R5_beta0.00_q0.20", 5, 0.00, 0.20),   # attributes carry nothing: H2 must fail
    ("R5_beta0.50_q0.50", 5, 0.50, 0.50),   # raters answer at random: H1 must fail
]


def load_pairs(manifest: Path):
    rows = A.read_manifest(manifest)
    first = [m for m in rows if m["presentation"] == 1]
    by_pair = {}
    for m in rows:
        by_pair.setdefault(m["pair_id"], {})[m["presentation"]] = m
    D = np.column_stack([
        np.array([m["trade_prior_a"] - m["trade_prior_b"] for m in first]),
        np.array([m["log1p_labor_a"] - m["log1p_labor_b"] for m in first]),
        np.array([m["wait_days_a"] - m["wait_days_b"] for m in first]),
    ])
    D = (D - D.mean(axis=0)) / np.where(D.std(axis=0) < 1e-12, 1.0, D.std(axis=0))
    return rows, first, by_pair, D


def signal(D: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A fixed linear combination of the attribute differences, unit variance."""
    a = np.array([0.6, -0.35, 0.72])          # fixed, not fitted
    s = D @ a
    sd = float(s.std())
    return s / (sd if sd > 1e-12 else 1.0)


def simulate(first, by_pair, D, n_raters, beta, q, rng):
    """Return {rater: {item_id: chosen_order_id}} under the fixture model."""
    s = signal(D, rng)
    delta = np.sqrt(beta) * s + np.sqrt(1.0 - beta) * rng.standard_normal(len(s))
    consensus = {m["pair_id"]: (m["order_a"] if d > 0 else m["order_b"])
                 for m, d in zip(first, delta)}
    other = {m["pair_id"]: (m["order_b"] if consensus[m["pair_id"]] == m["order_a"]
                            else m["order_a"]) for m in first}
    out = {}
    for r in range(n_raters):
        rid = f"R{r + 1}"
        ans = {}
        for pid, pres in by_pair.items():
            for p, m in pres.items():
                flip = rng.random() < q
                ans[m["item_id"]] = other[pid] if flip else consensus[pid]
        out[rid] = ans
    return out


def write_fixture(tag, resp, first_by_item, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for rid, ans in resp.items():
        path = out_dir / f"SYNTHETIC_{tag}_{rid}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rater_id", "rater_role", "rater_years_fm", "item_id",
                        "pair_id", "presented_left_order", "presented_right_order",
                        "choice_side", "chosen_order_id", "confidence", "reason",
                        "seconds_from_start"])
            for iid, chosen in sorted(ans.items()):
                m = first_by_item[iid]
                side = "L" if chosen == m["left_order"] else "R"
                w.writerow([f"SYNTHETIC-{rid}", "SYNTHETIC FIXTURE", "",
                            iid, m["pair_id"], m["left_order"], m["right_order"],
                            side, chosen, "", "", ""])


def cmd_fixtures(args):
    rows, first, by_pair, D = load_pairs(args.manifest)
    by_item = {m["item_id"]: m for m in rows}
    root = HERE / "responses"
    made = []
    # Seeds come from the fixture's position, not hash(tag): Python salts string
    # hashing per process, so a hash-derived seed would not reproduce.
    for slot, (tag, R, beta, q) in enumerate(FIXTURES):
        rng = np.random.default_rng(SEED + 977 * (slot + 1))
        resp = simulate(first, by_pair, D, R, beta, q, rng)
        d = root / tag
        write_fixture(tag, resp, by_item, d)
        made.append(dict(tag=tag, n_raters=R, beta=beta, q=q, dir=str(d)))
    (HERE / "SYNTHETIC_DO_NOT_USE.txt").write_text(
        "Every CSV under this directory is machine-generated from a known model "
        "with a fixed seed, for the sole purpose of testing that "
        "scripts/y3_w8_pilot_analyse.py runs and degrades correctly. No "
        "practitioner was recruited and no response was collected. Nothing here "
        "is a finding and nothing here may be quoted in the manuscript.\n")
    with open(HERE / "fixtures.json", "w") as fh:
        json.dump(dict(model=__doc__.split("GENERATIVE MODEL")[1].strip(),
                       seed=SEED, fixtures=made), fh, indent=2)
    print("wrote synthetic fixtures (NOT DATA):")
    for m in made:
        print(f"  {m['dir']}  R={m['n_raters']} beta={m['beta']} q={m['q']}")


# --------------------------------------------------------------------------- #
# Power, by simulation, for the pre-registration                              #
# --------------------------------------------------------------------------- #
def cmd_power(args):
    rows, first, by_pair, D = load_pairs(args.manifest)
    rng0 = np.random.default_rng(SEED + 1)
    grid = []
    for R in (2, 3, 5, 8):
        grid.append(dict(R=R, beta=0.40, q=0.20))
    for beta in (0.10, 0.20, 0.40, 0.60):
        grid.append(dict(R=5, beta=beta, q=0.20))
    for q in (0.10, 0.30):
        grid.append(dict(R=5, beta=0.40, q=q))
    seen, cells = set(), []
    for g in grid:
        k = (g["R"], g["beta"], g["q"])
        if k not in seen:
            seen.add(k)
            cells.append(g)

    out = []
    for g in cells:
        R, beta, q = g["R"], g["beta"], g["q"]
        h1 = h2 = h4 = 0
        n_ok_h2 = 0
        alphas, aucs = [], []
        for s in range(args.sims):
            rng = np.random.default_rng(SEED + 10000 * s + R * 97 + int(beta * 100))
            resp = simulate(first, by_pair, D, R, beta, q, rng)

            # H1: permutation test on Krippendorff's alpha (reduced resamples)
            units, used = [], []
            for m in first:
                u = [1 if resp[r][m["item_id"]] == m["order_a"] else 0
                     for r in sorted(resp)]
                units.append(u)
                used.append(m)
            if R >= 2:
                a_obs = A.krippendorff_alpha_nominal(units)
                alphas.append(a_obs)
                arr = np.array(units)
                ge = 0
                for _ in range(args.perm):
                    perm = np.array([row[rng.permutation(len(units))] for row in arr.T]).T
                    ap = A.krippendorff_alpha_nominal([list(perm[i]) for i in range(len(units))])
                    if np.isfinite(ap) and ap >= a_obs:
                        ge += 1
                if (1.0 + ge) / (1.0 + args.perm) < 0.05:
                    h1 += 1

            # H2: attribute model, out-of-fold AUC with a bootstrap lower bound
            y, keep = [], []
            for m in first:
                votes = [resp[r][m["item_id"]] for r in sorted(resp)]
                c = Counter(votes)
                top, n_top = c.most_common(1)[0]
                if len(c) > 1 and c.most_common(2)[1][1] == n_top:
                    continue
                keep.append(m)
                y.append(1 if top == m["order_a"] else 0)
            y = np.array(y)
            if len(y) >= 8 and len(np.unique(y)) > 1:
                n_ok_h2 += 1
                F = np.column_stack([
                    np.array([m["trade_prior_a"] - m["trade_prior_b"] for m in keep]),
                    np.array([m["log1p_labor_a"] - m["log1p_labor_b"] for m in keep]),
                    np.array([m["wait_days_a"] - m["wait_days_b"] for m in keep])])
                p = A.cv_out_of_fold(F, y, np.random.default_rng(SEED + s), repeats=5)
                auc = A.auc_score(y, p)
                aucs.append(auc)
                lo, hi, _ = A.boot_ci(lambda idx: A.auc_score(y[idx], p[idx]),
                                      len(y), np.random.default_rng(SEED + 3 * s),
                                      n_boot=args.boot)
                if np.isfinite(lo) and lo > 0.5:
                    h2 += 1

            # H4: pooled repeat consistency against chance
            reps = sorted({m["pair_id"] for m in rows if m["is_repeat"] == 1})
            hits = trials = 0
            for pid in reps:
                pres = by_pair[pid]
                for r in sorted(resp):
                    a1 = resp[r].get(pres[1]["item_id"])
                    a2 = resp[r].get(pres[2]["item_id"])
                    if a1 and a2:
                        trials += 1
                        hits += int(a1 == a2)
            from scipy import stats as st
            if trials and st.binomtest(hits, trials, 0.5, alternative="greater").pvalue < 0.05:
                h4 += 1

        out.append(dict(
            n_raters=R, beta=beta, q=q, sims=args.sims,
            mean_alpha=float(np.mean(alphas)) if alphas else float("nan"),
            power_H1_agreement=h1 / args.sims,
            mean_auc=float(np.mean(aucs)) if aucs else float("nan"),
            power_H2_predictability=(h2 / n_ok_h2) if n_ok_h2 else float("nan"),
            h2_estimable_share=n_ok_h2 / args.sims,
            power_H4_within_rater=h4 / args.sims))
        print(f"R={R} beta={beta:.2f} q={q:.2f} | alpha={out[-1]['mean_alpha']:.3f} "
              f"powH1={out[-1]['power_H1_agreement']:.2f} "
              f"auc={out[-1]['mean_auc']:.3f} "
              f"powH2={out[-1]['power_H2_predictability']:.2f} "
              f"powH4={out[-1]['power_H4_within_rater']:.2f}", flush=True)

    with open(HERE / "power.json", "w") as fh:
        json.dump(dict(
            note=("Power by simulation under the fixture's generative model, which "
                  "is a stand-in for practitioner behaviour and not a measurement "
                  "of it. Uncorrected per-hypothesis alpha = 0.05; the analysis "
                  "applies Holm across the four primary tests, so realised power is "
                  "somewhat lower than shown."),
            sims=args.sims, permutations=args.perm, bootstrap=args.boot,
            cells=out), fh, indent=2)
    print(f"\nwrote {HERE / 'power.json'}")


def main():
    ap = argparse.ArgumentParser(description="SYNTHETIC fixtures and power for W8")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--perm", type=int, default=400)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    if not args.fixtures and not args.power:
        ap.error("choose --fixtures and/or --power")
    if args.fixtures:
        cmd_fixtures(args)
    if args.power:
        cmd_power(args)


if __name__ == "__main__":
    main()
