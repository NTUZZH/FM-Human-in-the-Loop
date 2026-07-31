#!/usr/bin/env python
"""W1b analysis: the deployable regime map against the published one.

Reads the cell records this package's runner wrote, rebuilds the published map's
own per-cell statistics from `y3_p4_m0grid`'s helpers (imported, not rewritten),
and emits:

  results/y3_w1b/map_summary.json   per-cell contrasts, deployable and published
  results/y3_w1b/config_diff.json   resolved-config diff, cell by cell
  results/y3_w1b/policy_proof.json  which supervisor class ran, and how often the
                                    oracle clause was reached
  results/y3_w1b/macros_w1b.tex     \\newcommand definitions with provenance
  results/y3_w1b/map_table.md       the cell-by-cell table for the report

Two contrasts are reported per cell, because the manuscript names both:
  m0sup_over_rulesup_pct  the correction layer with the supervisor against the
                          tuned rule with the same supervisor. This is what the
                          published figure colours.
  m0_over_rule_pct        the layer alone against the rule alone.

Run:  PYTHONPATH=src python scripts/y3_w1b_analyze.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse                                                    # noqa: E402
import csv                                                         # noqa: E402
import json                                                        # noqa: E402
import sys                                                         # noqa: E402

import numpy as np                                                 # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import y3_w1_sweep as W1                                            # noqa: E402
import y3_p4_m0grid as P4                                           # noqa: E402
import y3_w1b_map as M                                              # noqa: E402

_OUT = M._OUT
_CACHE = M._CACHE
DECIDERS = P4.DECIDERS


# --------------------------------------------------------------------------- #
def load_part(tasks):
    """Cell records for a task list, from this package's cache only."""
    out = []
    missing = []
    for t in tasks:
        p = os.path.join(_CACHE, "%s.json" % W1._cell_sig(t))
        if not os.path.exists(p):
            missing.append(t)
            continue
        with open(p) as fh:
            out.append((t, json.load(fh)))
    return out, missing


def cell_key(campus, u, beta):
    return "c%d_u%d_b%.2f" % (campus, u, beta)


# --------------------------------------------------------------------------- #
# Per-cell statistics, using the published summariser's own helpers            #
# --------------------------------------------------------------------------- #
def cell_stats(recs):
    """The four percentages the published map reports, computed exactly as
    `y3_p4_m0grid.summarize_e3` computes them (mean of per-seed cell means)."""
    row = {}
    for d in DECIDERS:
        m, sd, S = P4._seed_meanstd(recs, d)
        row[d] = m
        row[d + "_std"] = sd
    row["n_seeds"] = S
    row["n_instances"] = len(P4._stack(recs, "rule")[1])
    rule_m, rsup_m = row["rule"], row["rule_sup"]
    row["m0_over_rule_pct"] = (100.0 * (rule_m - row["m0_alone"]) / rule_m) \
        if rule_m > 1e-12 else float("nan")
    row["m0sup_over_rulesup_pct"] = (100.0 * (rsup_m - row["m0_sup"]) / rsup_m) \
        if rsup_m > 1e-12 else float("nan")
    row["m0sup_over_rule_pct"] = (100.0 * (rule_m - row["m0_sup"]) / rule_m) \
        if rule_m > 1e-12 else float("nan")
    row["oracle_over_rule_pct"] = (100.0 * (rule_m - row["oracle"]) / rule_m) \
        if rule_m > 1e-12 else float("nan")
    # Routing telemetry, seed-averaged.
    for k in ("m0_sup_revfrac_mean", "m0_sup_undetermined", "m0_sup_cov_all",
              "rule_sup_revfrac_mean", "rule_sup_undetermined"):
        vals = [r["routing"][k] for _t, r in recs]
        row[k] = float(np.mean(vals)) if vals else float("nan")
    # Descriptive paired contrasts over the seed-averaged held-out instances.
    # The published map reports no significance (three seeds, descriptive), so
    # these are additional, not a comparison.
    for name, (test, comp) in (("M0sup_vs_RULEsup", ("m0_sup", "rule_sup")),
                               ("M0_vs_RULE", ("m0_alone", "rule"))):
        c = P4._contrast(recs, test, comp)
        row[name] = {"wtl": c["wtl"], "wilcoxon_p": c["wilcoxon_p"],
                     "n_instances": c["n_instances"]}
    bq = [r["band"]["q"] for _t, r in recs if r.get("band")]
    row["band_q"] = float(np.mean(bq)) if bq else float("nan")
    unb = [r["verdict"]["automation_coverage_unbudgeted"] for _t, r in recs
           if r.get("verdict")]
    row["automation_coverage_unbudgeted"] = float(np.mean(unb)) if unb else float("nan")
    return row


def pooled_pct(recs, test, comp):
    """The FIGURE's own formula: pool every instance of every seed, then take the
    ratio of the two pooled means. Equal instance counts per seed make this equal
    to the mean-of-seed-means form; asserted rather than assumed."""
    a, _ = P4._stack(recs, test)
    b, _ = P4._stack(recs, comp)
    am, bm = float(a.mean()), float(b.mean())
    return 100.0 * (bm - am) / bm


# --------------------------------------------------------------------------- #
# Data-accuracy gate: same instances, same overlay, same scoring               #
# --------------------------------------------------------------------------- #
def policy_free_check(campus, u, beta, rho, seed, n_eval, rec):
    """The tuned rule and the omniscient reference never consult the supervisor,
    so their per-instance TWT* cannot depend on the review policy. Bit-comparing
    them against the committed record proves this run is on the same instances,
    the same overlay draw, the same scoring and the same seed handling."""
    pub = M.published_record(campus, u, beta, rho, seed, n_eval)
    if pub is None:
        return {"published_record": False}
    out = {"published_record": True,
           "inst_ids_identical": list(rec["inst_ids"]) == list(pub["inst_ids"])}
    for d in ("rule", "oracle"):
        a = np.asarray(rec["per"][d], float)
        b = np.asarray(pub["per"][d], float)
        out[d + "_bit_identical"] = bool(np.array_equal(a, b))
        out[d + "_max_abs_diff"] = float(np.abs(a - b).max())
    return out


# --------------------------------------------------------------------------- #
# Config diff                                                                  #
# --------------------------------------------------------------------------- #
_BOOKKEEPING = {"scope", "part", "arm", "n_eval_full"}


def config_diff(mine, published):
    """Key-by-key diff of two resolved task dictionaries, bookkeeping removed."""
    a = {k: v for k, v in mine.items() if k not in _BOOKKEEPING}
    b = {k: v for k, v in published.items() if k not in _BOOKKEEPING}
    same, differs, only_mine, only_pub = {}, {}, {}, {}
    for k in sorted(set(a) | set(b)):
        if k in a and k in b:
            (same if a[k] == b[k] else differs)[k] = \
                a[k] if a[k] == b[k] else {"deployable": a[k], "published": b[k]}
        elif k in a:
            only_mine[k] = a[k]
        else:
            only_pub[k] = b[k]
    return {"identical": same, "differs": differs,
            "only_in_deployable": only_mine, "only_in_published": only_pub}


# --------------------------------------------------------------------------- #
# The map's three qualitative claims, checked on both policies                  #
# --------------------------------------------------------------------------- #
# "Inert" means within this many percentage points of no reduction at all. The
# published map's own slack row sits at 1.25 pp or below, so 2.0 is the smallest
# threshold that calls every published inert cell inert.
_INERT_PP = 2.0


def _series(cells, which, campus, field, fix_beta=None, fix_u=None,
            u_levels=None, betas=None):
    """The reduction along one row or column, or None if any cell is missing."""
    out = []
    keys = [(u, fix_beta) for u in u_levels] if fix_beta is not None \
        else [(fix_u, b) for b in betas]
    for u, b in keys:
        v = cells.get(cell_key(campus, u, b), {}).get(which)
        if v is None:
            return None
        out.append(v[field])
    return out


def _monotone(xs, tol=0.5):
    """Non-decreasing up to a tolerance, so a sub-half-point wobble between two
    essentially equal cells is not called a violation."""
    return all(b >= a - tol for a, b in zip(xs, xs[1:]))


def qualitative(cells, grid, field="m0sup_over_rulesup_pct"):
    """Does the deployable map still support the three claims the published map
    supports: the reduction grows with load, grows with the recoverable share,
    and collapses where either is absent?"""
    us, bs = grid["u_levels"], grid["betas"]
    band = [u for u in us if 0.85 <= {70: 0.70, 90: 0.91, 100: 1.00,
                                      110: 1.10, 130: 1.30}.get(u, 0) <= 1.02]
    out = {"field": field, "inert_threshold_pp": _INERT_PP,
           "realistic_load_band_u": band}
    for which in ("published", "deployable"):
        r = {"grows_with_load": {}, "grows_with_recoverable_share": {},
             "inert_at_slack": {}, "inert_at_zero_recoverable_in_band": {}}
        for campus in grid["campuses"]:
            for b in bs:
                s = _series(cells, which, campus, field, fix_beta=b, u_levels=us)
                if s is not None:
                    r["grows_with_load"]["c%d_b%.2f" % (campus, b)] = \
                        {"series": [round(x, 3) for x in s],
                         "monotone": _monotone(s)}
            for u in us:
                s = _series(cells, which, campus, field, fix_u=u, betas=bs)
                if s is not None:
                    r["grows_with_recoverable_share"]["c%d_u%d" % (campus, u)] = \
                        {"series": [round(x, 3) for x in s],
                         "monotone": _monotone(s)}
            slack = min(us)
            s = _series(cells, which, campus, field, fix_u=slack, betas=bs)
            if s is not None:
                r["inert_at_slack"]["c%d_u%d" % (campus, slack)] = \
                    {"series": [round(x, 3) for x in s],
                     "all_inert": all(abs(x) <= _INERT_PP for x in s)}
            s = _series(cells, which, campus, field, fix_beta=min(bs),
                        u_levels=band)
            if s is not None:
                r["inert_at_zero_recoverable_in_band"]["c%d_b%.2f" %
                                                       (campus, min(bs))] = \
                    {"series": [round(x, 3) for x in s],
                     "all_inert": all(abs(x) <= _INERT_PP for x in s)}
        for claim in list(r):
            vals = [v.get("monotone", v.get("all_inert")) for v in r[claim].values()]
            r[claim + "_holds_everywhere"] = bool(vals) and all(vals)
        out[which] = r
    out["conclusions_unchanged"] = all(
        out["published"].get(c + "_holds_everywhere") ==
        out["deployable"].get(c + "_holds_everywhere")
        for c in ("grows_with_load", "grows_with_recoverable_share",
                  "inert_at_slack", "inert_at_zero_recoverable_in_band"))
    return out


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=10)
    args = ap.parse_args(argv)

    grid = M.published_grid()
    seeds = list(range(301, 301 + grid["n_seeds"]))
    with open(M._PUB_SUMMARY) as fh:
        pub_summary = json.load(fh)

    map_recs, map_missing = load_part(M.tasks_map(grid, n_eval=args.n_eval))
    ctrl_recs, ctrl_missing = load_part(M.tasks_ctrl(grid, n_eval=args.n_eval))
    print("[load] map: %d records, %d missing | ctrl: %d records, %d missing"
          % (len(map_recs), len(map_missing), len(ctrl_recs), len(ctrl_missing)))

    def by_cell(recs):
        g = {}
        for t, r in recs:
            g.setdefault(cell_key(r["campus"], r["u"], r["beta"]), []).append((t, r))
        return g

    gmap, gctrl = by_cell(map_recs), by_cell(ctrl_recs)

    # ---- proof of which policy ran, aggregated -------------------------- #
    proof = {"map": {"cells": 0, "n_make_supervisor": 0, "classes": {},
                     "has_plus2_calls": 0, "from_cache": 0},
             "ctrl": {"cells": 0, "n_make_supervisor": 0, "classes": {},
                      "has_plus2_calls": 0, "from_cache": 0},
             "gate": {"cells": 0, "n_make_supervisor": 0, "classes": {},
                      "has_plus2_calls": 0, "from_cache": 0}}
    gate_recs, _gm = load_part(M.tasks_gate())
    for name, recs in (("map", map_recs), ("ctrl", ctrl_recs),
                       ("gate", gate_recs)):
        for _t, r in recs:
            pf = r.get("policy_proof", {})
            proof[name]["cells"] += 1
            proof[name]["n_make_supervisor"] += pf.get("n_make_supervisor", 0)
            proof[name]["has_plus2_calls"] += pf.get("has_plus2_calls", 0)
            proof[name]["from_cache"] += int(bool(pf.get("from_cache")))
            for c, n in (pf.get("classes") or {}).items():
                proof[name]["classes"][c] = proof[name]["classes"].get(c, 0) + n
    # Cache-proof evidence, read off the records themselves.
    proof["map"]["cells_with_undetermined_rate"] = sum(
        1 for _t, r in map_recs if np.isfinite(r["routing"]["m0_sup_undetermined"]))
    proof["map"]["cells_with_band"] = sum(1 for _t, r in map_recs
                                          if r.get("band") is not None)
    proof["map"]["run_config_policies"] = sorted(
        {r["run_config"]["policy"] for _t, r in map_recs})
    proof["ctrl"]["run_config_policies"] = sorted(
        {r["run_config"]["policy"] for _t, r in ctrl_recs})
    proof["gate"]["run_config_policies"] = sorted(
        {r["run_config"]["policy"] for _t, r in gate_recs})
    proof["note"] = ("has_plus2 is the published policy's undeployable clause: it "
                     "reads the realized latent shift of the pending queue. It "
                     "must be 0 on every deployable cell and large on the gate.")
    with open(os.path.join(_OUT, "policy_proof.json"), "w") as fh:
        json.dump(proof, fh, indent=1)

    # ---- per-cell table -------------------------------------------------- #
    cells = {}
    accuracy = {}
    pooled_check_max = 0.0
    for campus in grid["campuses"]:
        for u in grid["u_levels"]:
            for beta in grid["betas"]:
                ck = cell_key(campus, u, beta)
                pub = pub_summary["cells"].get(ck)
                row = {"campus": campus, "u": u, "beta": beta,
                       "rho": grid["rho"], "published": pub}
                recs = gmap.get(ck)
                if recs and len(recs) == grid["n_seeds"]:
                    dep = cell_stats(recs)
                    row["deployable"] = dep
                    # figure-formula cross-check
                    for a, b, key in (("m0_sup", "rule_sup", "m0sup_over_rulesup_pct"),
                                      ("m0_alone", "rule", "m0_over_rule_pct")):
                        pooled_check_max = max(
                            pooled_check_max,
                            abs(pooled_pct(recs, a, b) - dep[key]))
                    row["delta_m0sup_over_rulesup_pp"] = \
                        dep["m0sup_over_rulesup_pct"] - pub["m0sup_over_rulesup_pct"]
                    row["delta_m0_over_rule_pp"] = \
                        dep["m0_over_rule_pct"] - pub["m0_over_rule_pct"]
                    acc = [policy_free_check(campus, u, beta, grid["rho"],
                                             r["seed"], args.n_eval, r)
                           for _t, r in recs]
                    accuracy[ck] = acc
                else:
                    row["deployable"] = None
                    row["not_run"] = True
                    row["n_seeds_present"] = len(recs) if recs else 0
                crecs = gctrl.get(ck)
                if crecs and len(crecs) == grid["n_seeds"]:
                    ctl = cell_stats(crecs)
                    row["targeted_split"] = ctl
                    # Decomposition of (deployable - published) into the two
                    # coupled changes: `split` is the conformal fold split with
                    # the published policy held fixed, `policy` is the routing
                    # policy with the split held fixed. The latter is the price
                    # of deployability at this cell.
                    row["delta_split_m0sup_over_rulesup_pp"] = \
                        ctl["m0sup_over_rulesup_pct"] - pub["m0sup_over_rulesup_pct"]
                    row["delta_split_m0_over_rule_pp"] = \
                        ctl["m0_over_rule_pct"] - pub["m0_over_rule_pct"]
                    if row["deployable"] is not None:
                        row["delta_policy_m0sup_over_rulesup_pp"] = \
                            row["deployable"]["m0sup_over_rulesup_pct"] \
                            - ctl["m0sup_over_rulesup_pct"]
                        row["delta_policy_m0_over_rule_pp"] = \
                            row["deployable"]["m0_over_rule_pct"] \
                            - ctl["m0_over_rule_pct"]
                else:
                    row["targeted_split"] = None
                cells[ck] = row

    # Data-accuracy verdict.
    acc_flat = [a for v in accuracy.values() for a in v]
    acc_ok = all(a.get("rule_bit_identical") and a.get("oracle_bit_identical")
                 and a.get("inst_ids_identical") for a in acc_flat)
    print("[accuracy] policy-free deciders bit-identical to the committed map on "
          "%d cell-seeds: %s" % (len(acc_flat), "YES" if acc_ok else "NO"))
    print("[accuracy] figure-formula vs summary-formula max disagreement: %.3e pp"
          % pooled_check_max)

    run_cells = [k for k, v in cells.items() if v["deployable"] is not None]
    not_run = [k for k, v in cells.items() if v["deployable"] is None]
    qual = qualitative(cells, grid)

    summary = {
        "config": {"channel": grid["channel"], "rho": grid["rho"],
                   "n_seeds": grid["n_seeds"], "seeds": seeds,
                   "n_eval": args.n_eval,
                   "review_policy": "stability (deployable; conformal band on "
                                    "override-derived weak labels only)",
                   "published_review_policy": "targeted (oracle-informed; reads "
                                              "the realized latent shift)",
                   "scoring": "TWT*(w*,d*) full_class_shift",
                   "comparator": "results/y3_p4/e3_map_summary.json",
                   "plotted_contrast": "m0sup_over_rulesup_pct",
                   "second_contrast": "m0_over_rule_pct"},
        "coverage": {"cells_total": grid["n_cells"], "cells_run": len(run_cells),
                     "cells_not_run": not_run},
        "data_accuracy": {"policy_free_bit_identical": bool(acc_ok),
                          "n_cell_seeds_checked": len(acc_flat),
                          "figure_vs_summary_formula_max_pp": pooled_check_max},
        "qualitative": qual,
        "cells": cells,
    }
    with open(os.path.join(_OUT, "map_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("[map] wrote %s" % os.path.join(_OUT, "map_summary.json"))

    # ---- config diff ------------------------------------------------------ #
    diffs = {}
    for campus in grid["campuses"]:
        for u in grid["u_levels"]:
            for beta in grid["betas"]:
                ck = cell_key(campus, u, beta)
                mine = W1._base_task(campus=campus, regime="storm2", u=u,
                                     beta=beta, rho=grid["rho"], seed=301,
                                     n_eval=args.n_eval, arm="stability",
                                     part="map", policy="stability",
                                     split_fit=True)
                pubt = M.published_task(campus, u, beta, grid["rho"], 301,
                                        args.n_eval)
                diffs[ck] = config_diff(mine, pubt)
    keys = {json.dumps({"differs": sorted(v["differs"]),
                        "only_deployable": sorted(v["only_in_deployable"]),
                        "only_published": sorted(v["only_in_published"])},
                       sort_keys=True) for v in diffs.values()}
    diff_out = {"note": "Resolved task configuration of every deployable map "
                        "cell against the committed y3_p4 map cell it replaces. "
                        "The seed field is shown at 301; seeds 301-303 are "
                        "identical between the two runs.",
                "distinct_diff_shapes": len(keys),
                "shape": json.loads(sorted(keys)[0]) if keys else None,
                "example_cell": diffs.get("c9_u100_b1.00"),
                "per_cell": diffs}
    with open(os.path.join(_OUT, "config_diff.json"), "w") as fh:
        json.dump(diff_out, fh, indent=1, default=str)
    print("[diff] %d distinct diff shapes over %d cells (1 means every cell "
          "differs in exactly the same fields)" % (len(keys), len(diffs)))

    # ---- markdown table --------------------------------------------------- #
    lines = []
    lines.append("| cell | util | beta | published M0+SUP vs RULE+SUP | "
                 "deployable | diff (pp) | published M0 vs RULE | deployable | "
                 "diff (pp) | review frac | undetermined |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for campus in grid["campuses"]:
        for u in grid["u_levels"]:
            for beta in grid["betas"]:
                ck = cell_key(campus, u, beta)
                v = cells[ck]
                pub = v["published"]
                d = v["deployable"]
                if d is None:
                    lines.append("| C%d u%d | %.2f | %.2f | %+.2f%% | "
                                 "NOT RUN (published policy) | -- | %+.2f%% | "
                                 "NOT RUN (published policy) | -- | -- | -- |"
                                 % (campus, u, pub["util_pool"], beta,
                                    pub["m0sup_over_rulesup_pct"],
                                    pub["m0_over_rule_pct"]))
                    continue
                lines.append("| C%d u%d | %.2f | %.2f | %+.2f%% | %+.2f%% | "
                             "%+.2f | %+.2f%% | %+.2f%% | %+.2f | %.3f | %.3f |"
                             % (campus, u, pub["util_pool"], beta,
                                pub["m0sup_over_rulesup_pct"],
                                d["m0sup_over_rulesup_pct"],
                                v["delta_m0sup_over_rulesup_pp"],
                                pub["m0_over_rule_pct"], d["m0_over_rule_pct"],
                                v["delta_m0_over_rule_pp"],
                                d["m0_sup_revfrac_mean"],
                                d["m0_sup_undetermined"]))
    with open(os.path.join(_OUT, "map_table.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- CSV in the shape the figure script reads ------------------------- #
    csv_path = os.path.join(_OUT, "e3_map_deployable.csv")
    cols = ["campus", "regime", "u", "beta", "rho", "seed", "inst_id", "n_wos",
            "util_pool", "rule", "m0_alone", "oracle", "rule_sup", "m0_sup",
            "m0_sup_revfrac", "m0_sup_orr"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for _t, r in sorted(map_recs, key=lambda tr: (tr[1]["campus"],
                                                      tr[1]["u"], tr[1]["beta"],
                                                      tr[1]["seed"])):
            up = pub_summary["cells"][cell_key(r["campus"], r["u"],
                                               r["beta"])]["util_pool"]
            for i, iid in enumerate(r["inst_ids"]):
                row = {"campus": r["campus"], "regime": r["regime"], "u": r["u"],
                       "beta": r["beta"], "rho": r["rho"], "seed": r["seed"],
                       "inst_id": iid, "n_wos": r["n_wos"], "util_pool": up}
                for k in DECIDERS:
                    row[k] = "%.6f" % r["per"][k][i]
                for k in ("m0_sup_revfrac", "m0_sup_orr"):
                    v = r.get(k) or []
                    row[k] = ("%.4f" % v[i]) if i < len(v) else ""
                w.writerow(row)
    print("[csv] wrote %s" % csv_path)

    write_macros(summary, os.path.join(_OUT, "macros_w1b.tex"))
    return summary


# --------------------------------------------------------------------------- #
# Macros                                                                       #
# --------------------------------------------------------------------------- #
# The published regime-map macro of each quantity, so every deployable macro can
# name what it replaces. Macro names carry no digits: LaTeX rejects them.
_MACRO_SPEC = [
    # (deployable macro, published macro, cell, field, decimals, note).
    # `decimals` copies the published macro's own precision so the twin reads
    # consistently with the sentence around it.
    ("GainSlackDep", "GainSlack", "c9_u70_b1.00", "m0sup_over_rulesup_pct", 1,
     "C9 slack capacity, high recoverable share"),
    ("GainBusyDep", "GainBusy", "c9_u90_b1.00", "m0sup_over_rulesup_pct", 1,
     "C9 busy"),
    ("GainSatDep", "GainSat", "c9_u100_b1.00", "m0sup_over_rulesup_pct", 1,
     "C9 saturation (the headline load)"),
    ("GainBetaZeroDep", "GainBetaZero", "c9_u100_b0.00",
     "m0sup_over_rulesup_pct", 1, "C9 saturation, nothing recoverable"),
    ("BetaGainLowDep", "BetaGainLow", "c9_u100_b0.50",
     "m0sup_over_rulesup_pct", 1, "C9 saturation, half recoverable"),
    ("BetaGainHighDep", "BetaGainHigh", "c9_u100_b1.00",
     "m0sup_over_rulesup_pct", 1, "C9 saturation, fully recoverable"),
    ("GainPeakCnineDep", "GainPeakCnine", "c9_u130_b1.00",
     "m0sup_over_rulesup_pct", 0, "C9 overload tail"),
    ("RegretMaxDep", "RegretMax", "c10_u130_b1.00", "m0sup_over_rulesup_pct", 0,
     "C10 overload tail, the map's peak"),
    ("BetaZeroMapOverloadCnineDep", "BetaZeroMapOverloadCnine",
     "c9_u130_b0.00", "m0sup_over_rulesup_pct", 1, "C9 overload, beta = 0"),
    ("BetaZeroMapOverloadCtenDep", "BetaZeroMapOverloadCten",
     "c10_u130_b0.00", "m0sup_over_rulesup_pct", 1, "C10 overload, beta = 0"),
    ("BetaZeroBandCnineDep", "BetaZeroBandCnine", "c9_u100_b0.00",
     "m0_over_rule_pct", 1,
     "C9 in the realistic-load band, beta = 0, layer alone"),
    ("BetaZeroBandCtenDep", "BetaZeroBandCten", "c10_u100_b0.00",
     "m0_over_rule_pct", 1, "C10 in the band, beta = 0, layer alone"),
    ("BetaZeroBandCtenSupDep", "BetaZeroBandCtenSup", "c10_u100_b0.00",
     "m0sup_over_rulesup_pct", 1, "C10 in the band, beta = 0, in the loop"),
    ("BetaZeroOverloadCnineDep", "BetaZeroOverloadCnine", "c9_u130_b0.00",
     "m0_over_rule_pct", 0, "C9 overload, beta = 0, layer alone"),
    ("BetaZeroOverloadCtenDep", "BetaZeroOverloadCten", "c10_u130_b0.00",
     "m0_over_rule_pct", 0, "C10 overload, beta = 0, layer alone"),
]


def _pct(x, dec=1):
    """A percentage as LaTeX. A negative sign is set as a math minus, not a
    hyphen, which is what the class expects."""
    s = "%.*f" % (dec, abs(x))
    return ("$-$%s\\%%" % s) if float("%.*f" % (dec, x)) < 0 else "%s\\%%" % s


def write_macros(summary, path):
    cells = summary["cells"]
    L = []
    A = L.append
    A("% =========================================================================")
    A("% W1b -- the regime map (Figure F3) under the DEPLOYABLE review policy.")
    A("%")
    A("% Every number below is the same cell, the same instances, the same seeds")
    A("% and the same contrast as the published map, re-measured with the review")
    A("% routing decided by the decision-stability test under a conformal band")
    A("% calibrated on override-derived weak labels alone. The published map used")
    A("% the oracle-informed policy, whose review criterion reads the realized")
    A("% latent shift of the pending queue and which no site can run.")
    A("%")
    A("% Source: results/y3_w1b/map_summary.json, field cells.<cell>.deployable.*")
    A("% Comparator: results/y3_p4/e3_map_summary.json, field cells.<cell>.*")
    A("% Runner: scripts/y3_w1b_map.py --part map   Analysis: scripts/y3_w1b_analyze.py")
    A("% Grid: campuses C9 and C10, utilisation 70/90/100/110/130, recoverable")
    A("%% share 0/0.5/1.0, review budget rho = %.2f, %d seeds (%d-%d), %d "
      "held-out instances per cell-seed."
      % (summary["config"]["rho"], summary["config"]["n_seeds"],
         summary["config"]["seeds"][0], summary["config"]["seeds"][-1],
         summary["config"]["n_eval"]))
    A("% Macro names carry no digits by construction: LaTeX rejects them.")
    A("% =========================================================================")
    A("")
    A("% ---- Cell values: the plotted contrast (correction layer with the")
    A("% supervisor against the tuned rule with the same supervisor) unless the")
    A("% comment says 'layer alone', which is the layer alone against the rule")
    A("% alone. Each line names the published macro it is the deployable twin of.")
    for dep, pub, ck, field, dec, note in _MACRO_SPEC:
        v = cells.get(ck, {})
        d = v.get("deployable")
        p = v.get("published")
        if d is None or p is None:
            A("%% \\%s NOT AVAILABLE: cell %s was NOT re-run and remains on the "
              "published oracle-informed policy at \\%s" % (dep, ck, pub))
            continue
        A("\\newcommand{\\%s}{%s} %% %s; %s = %.4f%%, published \\%s = %.4f%% "
          "(%+.2f pp); map_summary.json:cells.%s.deployable.%s"
          % (dep, _pct(d[field], dec), note, field, d[field], pub, p[field],
             d[field] - p[field], ck, field))
    A("")
    A("% ---- What changed, over the whole map. Signed difference = deployable")
    A("% minus published, in percentage points of reduction.")
    run_keys = [k for k, v in cells.items() if v["deployable"] is not None]
    run = [cells[k] for k in run_keys]
    if not run:
        A("% (no cell has been run yet; nothing further to define)")
        with open(path, "w") as fh:
            fh.write("\n".join(L) + "\n")
        print("[macros] wrote %s (%d lines, no cells run)" % (path, len(L)))
        return
    for tag, key, field in (("Sup", "delta_m0sup_over_rulesup_pp",
                             "m0sup_over_rulesup_pct"),
                            ("Alone", "delta_m0_over_rule_pp",
                             "m0_over_rule_pct")):
        d = np.asarray([v[key] for v in run], float)
        A("\\newcommand{\\MapDeltaMean%s}{%s} %% mean signed difference over the "
          "%d re-run cells, contrast %s; map_summary.json:cells.*.%s"
          % (tag, ("$-$%.2f" % abs(d.mean())) if d.mean() < 0
             else ("%.2f" % d.mean()), len(run), field, key))
        A("\\newcommand{\\MapDeltaAbsMean%s}{%.2f} %% mean ABSOLUTE difference, "
          "same %d cells; map_summary.json:cells.*.%s"
          % (tag, np.abs(d).mean(), len(run), key))
        i = int(np.argmax(np.abs(d)))
        ck = run_keys[i]
        A("\\newcommand{\\MapDeltaMaxAbs%s}{%.2f} %% largest absolute difference "
          "of any cell (%s); map_summary.json:cells.%s.%s"
          % (tag, abs(d[i]), ck, ck, key))
    A("")
    A("% ---- The same two differences restricted to the REALISTIC-LOAD BAND")
    A("% (pooled utilisation 0.90-1.00, the operating region the figure boxes and")
    A("% the region every claim in the results section rests on), and to campus C9")
    A("% inside it, which is the primary campus.")
    band_keys = [k for k in run_keys if "_u90_" in k or "_u100_" in k]
    for scope, ks in (("Band", band_keys),
                      ("BandCnine", [k for k in band_keys if k.startswith("c9_")])):
        for tag, key, field in (("Sup", "delta_m0sup_over_rulesup_pp",
                                 "m0sup_over_rulesup_pct"),
                                ("Alone", "delta_m0_over_rule_pp",
                                 "m0_over_rule_pct")):
            d = np.asarray([cells[k][key] for k in ks], float)
            A("\\newcommand{\\MapDeltaMean%s%s}{%s} %% mean signed difference "
              "over the %d %s cells, contrast %s; map_summary.json:cells.*.%s"
              % (tag, scope, ("$-$%.2f" % abs(d.mean())) if d.mean() < 0
                 else ("%.2f" % d.mean()), len(ks),
                 "realistic-load-band" if scope == "Band"
                 else "campus-9 realistic-load-band", field, key))
            A("\\newcommand{\\MapDeltaAbsMean%s%s}{%.2f} %% mean ABSOLUTE "
              "difference, same %d cells; map_summary.json:cells.*.%s"
              % (tag, scope, np.abs(d).mean(), len(ks), key))
            i = int(np.argmax(np.abs(d)))
            A("\\newcommand{\\MapDeltaMaxAbs%s%s}{%.2f} %% largest absolute "
              "difference in that set (%s); map_summary.json:cells.%s.%s"
              % (tag, scope, abs(d[i]), ks[i], ks[i], key))
    A("")
    A("% ---- The deployable policy's own operating numbers, which the published")
    A("% policy does not produce at all. Seed-averaged over the map's cells.")
    hl = cells.get("c9_u100_b1.00", {}).get("deployable")
    if hl is not None:
        A("\\newcommand{\\MapReviewFracHead}{%.3f} %% realised reviewed fraction "
          "of reviewable decisions at the headline map cell c9_u100_b1.00; "
          "map_summary.json:cells.c9_u100_b1.00.deployable.m0_sup_revfrac_mean"
          % hl["m0_sup_revfrac_mean"])
        A("\\newcommand{\\MapUndeterminedHead}{%s} %% share of multi-candidate "
          "decisions the stability test could not certify at the same cell; "
          "map_summary.json:cells.c9_u100_b1.00.deployable.m0_sup_undetermined"
          % _pct(100.0 * hl["m0_sup_undetermined"]))
        A("\\newcommand{\\MapAutomationHead}{%s} %% share of all decisions "
          "dispatched without review at the same cell; "
          "map_summary.json:cells.c9_u100_b1.00.deployable.m0_sup_cov_all"
          % _pct(100.0 * hl["m0_sup_cov_all"]))
    rf = np.asarray([v["deployable"]["m0_sup_revfrac_mean"] for v in run], float)
    un = np.asarray([v["deployable"]["m0_sup_undetermined"] for v in run], float)
    A("\\newcommand{\\MapReviewFracLo}{%.3f} %% smallest realised reviewed "
      "fraction over the %d re-run cells; map_summary.json:cells.*.deployable."
      "m0_sup_revfrac_mean" % (rf.min(), len(run)))
    A("\\newcommand{\\MapReviewFracHi}{%.3f} %% largest, same cells" % rf.max())
    A("\\newcommand{\\MapUndeterminedLo}{%s} %% smallest undetermined share over "
      "the same cells; map_summary.json:cells.*.deployable.m0_sup_undetermined"
      % _pct(100.0 * un.min()))
    A("\\newcommand{\\MapUndeterminedHi}{%s} %% largest, same cells"
      % _pct(100.0 * un.max()))
    A("")
    A("% ---- Price of deployability at the map's cells, where the split-protocol")
    A("% control was run (campus 9 only). (deployable - published) decomposes as")
    A("% (policy effect, split held fixed) + (split effect, policy held fixed).")
    ctl_keys = [k for k in run_keys
                if cells[k].get("delta_policy_m0sup_over_rulesup_pp") is not None]
    if ctl_keys:
        ctl_band = [k for k in ctl_keys if "_u90_" in k or "_u100_" in k]
        for scope, ks in (("", ctl_keys), ("Band", ctl_band)):
            for tag, key in (("Sup", "delta_policy_m0sup_over_rulesup_pp"),
                             ("Alone", "delta_policy_m0_over_rule_pp"),
                             ("SplitSup", "delta_split_m0sup_over_rulesup_pp"),
                             ("SplitAlone", "delta_split_m0_over_rule_pp")):
                d = np.asarray([cells[k][key] for k in ks], float)
                A("\\newcommand{\\MapPrice%s%s}{%s} %% mean over the %d campus-9 "
                  "%scells with a split-protocol control; POSITIVE favours the "
                  "deployable policy; map_summary.json:cells.*.%s"
                  % (tag, scope, ("$-$%.2f" % abs(d.mean())) if d.mean() < 0
                     else ("%.2f" % d.mean()), len(ks),
                     "realistic-load-band " if scope else "", key))
    else:
        A("% (the split-protocol control has not been run)")
    A("")
    A("% ---- Coverage of the re-run, for the sentence that says what is on which")
    A("% policy. If cells_not_run is non-empty, those cells remain on the")
    A("% published (oracle-informed) policy and must be labelled as such.")
    A("\\newcommand{\\MapCellsDeployable}{%d} %% cells re-run under the "
      "deployable policy; map_summary.json:coverage.cells_run"
      % summary["coverage"]["cells_run"])
    A("\\newcommand{\\MapCellsTotal}{%d} %% cells in the published map; "
      "map_summary.json:coverage.cells_total"
      % summary["coverage"]["cells_total"])
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("[macros] wrote %s (%d lines)" % (path, len(L)))


if __name__ == "__main__":
    main()
