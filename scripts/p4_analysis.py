#!/usr/bin/env python
"""P4 Gate-B analysis: turn results/p4_dyneval/results.csv into the Gate B
summary (md + csv) and a LaTeX-ready table fragment.

Protocol (the training spec §5 + the AMENDED Gate B protocol, docs/decision_log.md
2026-07-05):

* Gate B is judged on BOTH (i) replay test @ default capacity and (ii) the
  contended regimes (replay-tight, crew_multiplier in {0.6, 0.8}, PLUS the
  generator storm cells).  Both are reported regardless of outcome.
* "Beats a PDR" = LOWER mean WWT than that PDR AND paired Wilcoxon p < 0.05
  (ties are not wins).  The gate PDR set is the 5 deterministic rules
  {edd, wspt, atc, pfifo, mor} ('random' is a sanity baseline, not gated).
* PASS for a regime = the policy beats >= 3/5 PDRs CONSISTENTLY in all 3 seeds
  (each seed individually satisfies the beat criterion for that PDR).
* Win/tie/loss counts use tie tolerance eps = 1.0 weighted units (Gate A
  decision: CP-SAT centi-grid rounding inflates exact ties).
* E4 held-out campuses {1, 2} are EXCLUDED from every verdict (they still
  appear in the per-campus tables, marked).
* PRIMARY verdict comes from the regime where methods separate (expected (ii));
  regime (i) is reported as the capacity-adequate-regime finding either way.
* rollcp2 runs on a per-cell subsample only, so in every mean-WWT table its
  raw subsample mean is REPLACED by a same-instance pairing: the 'rollcp2'
  column is its mean over the paired subsample ids and 'edd@roll' is the EDD
  mean restricted to exactly those ids (same n) -- the 9 base methods keep
  their full-cell means (juxtaposing the n=8 subsample mean with full-cell
  means was a composition artifact).  Per regime a compact 'rollcp2 vs
  best-PDR (same ids)' paired summary (mean diff, Wilcoxon p, W/T/L) is
  reported in the md and appended to gateB_summary.csv as
  scope_type='rollcp2-vs-best-pdr' rows (seed='rollcp2'; the mean_rl column
  holds rollcp2's mean; schema additive, no renames).

Outputs (results/p4_dyneval/)
-----------------------------
  gateB_summary.md   human-readable report: verdict lines, mean-WWT tables per
                     regime x campus x size, per-seed + pooled comparison stats.
  gateB_summary.csv  the comparison rows (scope, seed incl. 'pooled', pdr,
                     n_pairs, means, Wilcoxon p, win/tie/loss, beats flag).
  tab_gateB.tex      booktabs tabular of mean WWT by method x regime on the
                     verdict campuses (plain numbers, 2 decimals, no \\num{}).

Usage
-----
    PYTHONPATH=src python scripts/p4_analysis.py [--in DIR] [--csv PATH]
        [--rl-tag TAG] [--eps 1.0]

  --in DIR    results root: reads DIR/results.csv (unless --csv overrides) and
              writes the gateB_* outputs there (default results/p4_dyneval).
  --rl-tag T  method-column prefix of the policy to gate (default 'rl' ->
              rl301..rl303); use e.g. 'v2rl' to score a v2-tagged eval dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = _ROOT / "results" / "p4_dyneval"
DEFAULT_CSV = OUT_DIR / "results.csv"

GATE_PDRS = ["edd", "wspt", "atc", "pfifo", "mor"]   # the 5 gated PDRs
ALL_PDRS = GATE_PDRS + ["random"]
RL_SEEDS = [301, 302, 303]
# RL method naming is reconfigurable (--rl-tag); _configure_rl() rederives the
# name lists so the Gate B comparisons pick the tagged policy columns.
RL_TAG = "rl"
RL_METHODS = ["%s%d" % (RL_TAG, s) for s in RL_SEEDS]
ROLLCP = "rollcp2"
METHOD_COLS = ALL_PDRS + RL_METHODS + [ROLLCP]


def _rl_method(seed):
    return "%s%d" % (RL_TAG, seed)


def _configure_rl(tag, seeds=None):
    """Set the RL method tag (+ optional seed list) and rederive the name lists.

    ``seeds`` overrides the gated seed set (default {301,302,303}); the
    all-seeds verdict rule and every per-seed comparison then range over it, so
    the same script scores 3 seeds, the 10-seed MLP pool, or the 3 attn seeds."""
    global RL_TAG, RL_SEEDS, RL_METHODS, METHOD_COLS
    RL_TAG = str(tag)
    if seeds is not None:
        RL_SEEDS = [int(s) for s in seeds]
    RL_METHODS = ["%s%d" % (RL_TAG, s) for s in RL_SEEDS]
    METHOD_COLS = ALL_PDRS + RL_METHODS + [ROLLCP]

VERDICT_CAMPUSES = {5, 9, 10, 12}       # E4 held-out {1,2} excluded
HELDOUT_CAMPUSES = {1, 2}
EPS_DEFAULT = 1.0
ALPHA = 0.05
N_BEAT_REQUIRED = 3

COMP_FIELDS = [
    "scope_type", "regime", "campus", "size", "crew_multiplier",
    "arrival_multiplier", "seed", "pdr", "n_pairs", "mean_rl", "mean_pdr",
    "delta", "wilcoxon_p", "wins", "ties", "losses", "beats",
]


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def _wilcoxon_p(diffs: np.ndarray) -> float:
    """Two-sided paired Wilcoxon signed-rank p on the paired differences.

    All-zero (or empty) difference vectors carry no evidence -> p = 1.0.
    """
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0 or np.all(diffs == 0.0):
        return 1.0
    try:
        return float(wilcoxon(diffs).pvalue)
    except ValueError:
        return 1.0


def _compare(sub: pd.DataFrame, rl_method: str, pdr: str, eps: float):
    """Paired comparison rl_method vs pdr on one scope subset.

    Returns None if no pairs; else a dict of the COMP row statistics.
    Pairs are joined on instance-config ``id`` (both sides feasible).
    """
    a = sub[(sub["method"] == rl_method) & (sub["feasible"] == 1)]
    b = sub[(sub["method"] == pdr) & (sub["feasible"] == 1)]
    if a.empty or b.empty:
        return None
    j = a.set_index("id")["wwt"].to_frame("rl").join(
        b.set_index("id")["wwt"].to_frame("pdr"), how="inner")
    if j.empty:
        return None
    d = (j["rl"] - j["pdr"]).to_numpy(dtype=float)
    return {
        "n_pairs": int(len(j)),
        "mean_rl": float(j["rl"].mean()),
        "mean_pdr": float(j["pdr"].mean()),
        "delta": float(d.mean()),
        "wilcoxon_p": _wilcoxon_p(d),
        "wins": int((d < -eps).sum()),
        "ties": int((np.abs(d) <= eps).sum()),
        "losses": int((d > eps).sum()),
        "_diffs": d,
    }


def _beats(stat) -> bool:
    return (stat is not None and stat["mean_rl"] < stat["mean_pdr"]
            and stat["wilcoxon_p"] < ALPHA)


def _paired_roll_means(sub: pd.DataFrame):
    """(rollcp2_mean, edd_mean, n) over the ids rollcp2 ran on.

    Pairs on instance-config ``id`` with BOTH the rollcp2 and edd rows feasible,
    so the two means cover exactly the same instances (same n) -- the
    like-with-like comparison shown in the mean-WWT tables."""
    r = sub[(sub["method"] == ROLLCP) & (sub["feasible"] == 1)]
    e = sub[(sub["method"] == "edd") & (sub["feasible"] == 1)]
    if r.empty or e.empty:
        return None, None, 0
    j = r.set_index("id")["wwt"].to_frame("roll").join(
        e.set_index("id")["wwt"].to_frame("edd"), how="inner")
    if j.empty:
        return None, None, 0
    return float(j["roll"].mean()), float(j["edd"].mean()), int(len(j))


def roll_vs_best_pdr(sub: pd.DataFrame, eps: float):
    """rollcp2 vs the best gated PDR, paired on rollcp2's own instance ids.

    'best' = the gated PDR with the lowest mean WWT over the ids paired with
    rollcp2 (both sides feasible).  Returns None if rollcp2 has no feasible
    rows in ``sub``; else a stat dict with pdr, n_pairs, mean_roll, mean_pdr,
    delta, wilcoxon_p and W/T/L at tolerance ``eps`` (wins = rollcp2 lower)."""
    r = sub[(sub["method"] == ROLLCP) & (sub["feasible"] == 1)]
    if r.empty:
        return None
    best = None
    for pdr in GATE_PDRS:
        e = sub[(sub["method"] == pdr) & (sub["feasible"] == 1)]
        if e.empty:
            continue
        j = r.set_index("id")["wwt"].to_frame("roll").join(
            e.set_index("id")["wwt"].to_frame("pdr"), how="inner")
        if j.empty:
            continue
        mean_pdr = float(j["pdr"].mean())
        if best is None or mean_pdr < best["mean_pdr"]:
            d = (j["roll"] - j["pdr"]).to_numpy(dtype=float)
            best = {
                "pdr": pdr, "n_pairs": int(len(j)),
                "mean_roll": float(j["roll"].mean()), "mean_pdr": mean_pdr,
                "delta": float(d.mean()), "wilcoxon_p": _wilcoxon_p(d),
                "wins": int((d < -eps).sum()),
                "ties": int((np.abs(d) <= eps).sum()),
                "losses": int((d > eps).sum()),
            }
    return best


def scope_comparisons(sub: pd.DataFrame, eps: float):
    """Per-seed AND pooled comparisons of the policy vs each gated PDR.

    Returns {pdr: {seed(int)|'pooled': stat-dict-or-None}}.
    Pooled = the three seeds' paired differences concatenated (supplementary;
    the verdict uses per-seed consistency).
    """
    out = {}
    for pdr in GATE_PDRS:
        per = {}
        pooled_diffs, pooled_rl, pooled_pdr = [], [], []
        for s in RL_SEEDS:
            st = _compare(sub, _rl_method(s), pdr, eps)
            per[s] = st
            if st is not None:
                pooled_diffs.append(st["_diffs"])
                pooled_rl.append(st["mean_rl"] * st["n_pairs"])
                pooled_pdr.append(st["mean_pdr"] * st["n_pairs"])
        if pooled_diffs:
            d = np.concatenate(pooled_diffs)
            n = int(d.size)
            per["pooled"] = {
                "n_pairs": n,
                "mean_rl": float(sum(pooled_rl) / n),
                "mean_pdr": float(sum(pooled_pdr) / n),
                "delta": float(d.mean()),
                "wilcoxon_p": _wilcoxon_p(d),
                "wins": int((d < -eps).sum()),
                "ties": int((np.abs(d) <= eps).sum()),
                "losses": int((d > eps).sum()),
            }
        else:
            per["pooled"] = None
        out[pdr] = per
    return out


def verdict_from(comps):
    """(n_beaten, {pdr: consistent_bool}) under the all-3-seeds rule."""
    consistent = {}
    for pdr, per in comps.items():
        consistent[pdr] = all(_beats(per.get(s)) for s in RL_SEEDS)
    return sum(consistent.values()), consistent


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return ("%%.%df" % nd) % x


def _fmt_p(p):
    if p is None:
        return "-"
    return "%.4f" % p if p >= 1e-4 else "%.1e" % p


def mean_table(sub: pd.DataFrame):
    """(method -> mean WWT over its feasible rows, method -> n) for a scope."""
    means, ns = {}, {}
    for meth in METHOD_COLS:
        rows = sub[(sub["method"] == meth) & (sub["feasible"] == 1)]
        ns[meth] = int(len(rows))
        means[meth] = float(rows["wwt"].mean()) if len(rows) else None
    return means, ns


def md_mean_block(lines, df, group_cols, title):
    """Append a mean-WWT-per-method markdown table grouped by group_cols.

    Base methods show full-cell means.  rollcp2 (when present) shows its mean
    over the paired subsample ids, next to an extra 'edd@roll' column = EDD
    restricted to exactly those ids (same n, annotated inline)."""
    lines.append("### %s" % title)
    lines.append("")
    present = [m for m in METHOD_COLS
               if not df[(df["method"] == m)].empty]
    has_roll = ROLLCP in present
    hdr_methods = present + (["edd@roll"] if has_roll else [])
    header = "| " + " | ".join(group_cols) + " | n | " + \
             " | ".join(hdr_methods) + " |"
    sep = "|" + "---|" * (len(group_cols) + 1 + len(hdr_methods))
    lines.append(header)
    lines.append(sep)
    for key, sub in df.groupby(group_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        means, ns = mean_table(sub)
        n_base = max((n for m, n in ns.items() if m != ROLLCP), default=0)
        roll_m, eddroll_m, n_roll = _paired_roll_means(sub) if has_roll \
            else (None, None, 0)
        cells = []
        for m in present:
            if m == ROLLCP:
                v = _fmt(roll_m)
                if roll_m is not None:
                    v += " (n=%d)" % n_roll
            else:
                v = _fmt(means[m])
            cells.append(v)
        if has_roll:
            v = _fmt(eddroll_m)
            if eddroll_m is not None:
                v += " (n=%d)" % n_roll
            cells.append(v)
        lines.append("| " + " | ".join(str(k) for k in key) +
                     " | %d | " % n_base + " | ".join(cells) + " |")
    lines.append("")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="P4 Gate-B analysis.")
    ap.add_argument("--in", dest="in_dir", default=str(OUT_DIR),
                    help="results root: read <dir>/results.csv and write the "
                         "gateB_* outputs there (default results/p4_dyneval)")
    ap.add_argument("--csv", default=None,
                    help="explicit results csv (overrides <in>/results.csv)")
    ap.add_argument("--rl-tag", default="rl",
                    help="method-column prefix of the gated policy "
                         "(default 'rl' -> rl301..rl303)")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated RL seed list to gate "
                         "(default 301,302,303; e.g. 301,...,310 for the "
                         "10-seed MLP pool)")
    ap.add_argument("--eps", type=float, default=EPS_DEFAULT,
                    help="win/tie/loss tie tolerance (weighted units)")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] \
        if args.seeds else None
    _configure_rl(args.rl_tag, seeds)
    out_dir = Path(args.in_dir)
    csv_path = Path(args.csv) if args.csv else out_dir / "results.csv"
    if not csv_path.exists():
        sys.exit("results csv not found: %s (run scripts/p4_dyneval.py first)"
                 % csv_path)
    df = pd.read_csv(csv_path)
    eps = float(args.eps)

    n_rows = len(df)
    n_infeas = int((df["feasible"] != 1).sum())

    # ---- comparison rows (csv) + verdicts ----------------------------------
    comp_rows = []

    def emit(scope_type, regime, campus, size, cm, am, comps):
        for pdr, per in comps.items():
            for s in RL_SEEDS + ["pooled"]:
                st = per.get(s)
                if st is None:
                    continue
                comp_rows.append({
                    "scope_type": scope_type, "regime": regime,
                    "campus": campus, "size": size, "crew_multiplier": cm,
                    "arrival_multiplier": am, "seed": s, "pdr": pdr,
                    "n_pairs": st["n_pairs"], "mean_rl": st["mean_rl"],
                    "mean_pdr": st["mean_pdr"], "delta": st["delta"],
                    "wilcoxon_p": st["wilcoxon_p"], "wins": st["wins"],
                    "ties": st["ties"], "losses": st["losses"],
                    "beats": int(_beats(st)),
                })

    # cell-level comparisons (regime x campus x size x cm x am)
    for key, sub in df.groupby(["regime", "campus", "size", "crew_multiplier",
                                "arrival_multiplier"], sort=True):
        regime, campus, size, cm, am = key
        emit("cell", regime, campus, size, cm, am, scope_comparisons(sub, eps))

    # regime x campus x size (pools cm/am inside the regime; skip for
    # replay-default where it equals the cell scope)
    for key, sub in df.groupby(["regime", "campus", "size"], sort=True):
        regime, campus, size = key
        if regime == "replay-default":
            continue
        emit("regime-campus-size", regime, campus, size, "ALL", "ALL",
             scope_comparisons(sub, eps))

    # ---- verdict scopes ------------------------------------------------------
    train = df[df["campus"].isin(VERDICT_CAMPUSES)]
    scope_i = train[train["regime"] == "replay-default"]
    scope_ii = train[train["regime"].isin(["replay-tight", "storm"])]

    comps_i = scope_comparisons(scope_i, eps) if not scope_i.empty else {}
    comps_ii = scope_comparisons(scope_ii, eps) if not scope_ii.empty else {}
    if comps_i:
        emit("verdict-default(i)", "replay-default", "ALL-train", "ALL",
             "ALL", "ALL", comps_i)
    if comps_ii:
        emit("verdict-contended(ii)", "replay-tight+storm", "ALL-train", "ALL",
             "ALL", "ALL", comps_ii)

    nb_i, cons_i = verdict_from(comps_i) if comps_i else (None, {})
    nb_ii, cons_ii = verdict_from(comps_ii) if comps_ii else (None, {})

    # ---- rollcp2 vs best PDR on its own subsample ids, per regime ----------
    # (appended LAST so all pre-existing csv rows keep their positions; the
    # mean_rl column holds rollcp2's mean in these rows, seed='rollcp2')
    regime_order = ["replay-default", "replay-tight", "storm", "pmmix"]
    roll_stats = {}
    for regime in regime_order:
        rsub = df[df["regime"] == regime]
        if rsub.empty:
            continue
        st = roll_vs_best_pdr(rsub, eps)
        if st is None:
            continue
        roll_stats[regime] = st
        comp_rows.append({
            "scope_type": "rollcp2-vs-best-pdr", "regime": regime,
            "campus": "ALL", "size": "ALL", "crew_multiplier": "ALL",
            "arrival_multiplier": "ALL", "seed": "rollcp2", "pdr": st["pdr"],
            "n_pairs": st["n_pairs"], "mean_rl": st["mean_roll"],
            "mean_pdr": st["mean_pdr"], "delta": st["delta"],
            "wilcoxon_p": st["wilcoxon_p"], "wins": st["wins"],
            "ties": st["ties"], "losses": st["losses"],
            "beats": int(st["mean_roll"] < st["mean_pdr"]
                         and st["wilcoxon_p"] < ALPHA),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    comp_df = pd.DataFrame(comp_rows, columns=COMP_FIELDS)
    comp_csv = out_dir / "gateB_summary.csv"
    comp_df.to_csv(comp_csv, index=False)

    # ---- E2 intensity curve (storm2 utilization sweep) ----------------------
    # F3 plotting input: one row per (campus, u_target, method) over the storm2
    # feasible rows, carrying mean WWT / mean breach-share and the mean realized
    # utilization.  Emitted only when storm2 rows are present; purely additive
    # (no verdict logic reads it).
    if "regime" in df.columns and (df["regime"] == "storm2").any():
        s2 = df[(df["regime"] == "storm2") & (df["feasible"] == 1)]
        e2_rows = []
        if not s2.empty and "u_target" in s2.columns:
            for (campus, u_t, meth), grp in s2.groupby(
                    ["campus", "u_target", "method"], sort=True):
                ur = grp["u_realized"].dropna() if "u_realized" in grp.columns \
                    else pd.Series(dtype=float)
                e2_rows.append({
                    "regime": "storm2", "campus": campus, "u_target": u_t,
                    "u_realized_mean": (float(ur.mean()) if len(ur) else None),
                    "method": meth,
                    "mean_wwt": float(grp["wwt"].mean()),
                    "mean_breach_share": float(grp["breach_share"].mean()),
                    "n": int(len(grp)),
                })
        e2_df = pd.DataFrame(e2_rows, columns=[
            "regime", "campus", "u_target", "u_realized_mean", "method",
            "mean_wwt", "mean_breach_share", "n"])
        e2_path = out_dir / "e2_curve.csv"
        e2_df.to_csv(e2_path, index=False)
        print("Wrote %s (%d rows; storm2 intensity curve)"
              % (e2_path, len(e2_df)))

    # ---- markdown report ----------------------------------------------------
    lines = []
    lines.append("# Gate B summary — dynamic evaluation (P4)")
    lines.append("")
    lines.append("Source: `%s` (%d rows; %d infeasible rows excluded from all "
                 "statistics)." % (csv_path.name, n_rows, n_infeas))
    lines.append("Protocol: amended Gate B (docs/decision_log.md 2026-07-05). "
                 "beats = lower mean WWT AND paired Wilcoxon p<%.2f; "
                 "PASS needs >=%d/5 PDRs beaten consistently in ALL %d seeds "
                 "(%s); win/tie/loss tie tolerance eps=%.1f; verdict campuses "
                 "%s (E4 held-out %s excluded)."
                 % (ALPHA, N_BEAT_REQUIRED, len(RL_SEEDS),
                    ",".join(str(s) for s in RL_SEEDS), eps,
                    sorted(VERDICT_CAMPUSES), sorted(HELDOUT_CAMPUSES)))
    lines.append("")

    # -- verdicts --
    lines.append("## GATE B VERDICT")
    lines.append("")

    def verdict_lines(tag, comps, nb, cons, scope_df):
        if not comps or nb is None:
            lines.append("* %s: NO DATA in the results csv (regime not run "
                         "yet)." % tag)
            return
        n_inst = scope_df[scope_df["method"] == "edd"]["id"].nunique()
        beaten = sorted([p for p, c in cons.items() if c])
        verdict = "PASS" if nb >= N_BEAT_REQUIRED else "FAIL"
        lines.append("* **%s** (n=%d instance-configs): policy beats "
                     "**%d/5** PDRs consistently in all %d seeds "
                     "(%s) -> **GATE B %s** for this regime."
                     % (tag, n_inst, nb, len(RL_SEEDS),
                        ("{" + ", ".join(beaten) + "}") if beaten else "none",
                        verdict))
        for pdr in GATE_PDRS:
            per = comps[pdr]
            seeds_txt = []
            for s in RL_SEEDS:
                st = per.get(s)
                if st is None:
                    seeds_txt.append("s%d: n/a" % s)
                else:
                    seeds_txt.append(
                        "s%d: %s (dWWT %+0.2f, p %s)"
                        % (s, "BEAT" if _beats(st) else "no",
                           st["mean_rl"] - st["mean_pdr"],
                           _fmt_p(st["wilcoxon_p"])))
            pl = per.get("pooled")
            pooled_txt = ("pooled: dWWT %+0.2f, p %s, W/T/L %d/%d/%d"
                          % (pl["delta"], _fmt_p(pl["wilcoxon_p"]),
                             pl["wins"], pl["ties"], pl["losses"])) \
                if pl else "pooled: n/a"
            lines.append("  * vs **%s** — %s; %s%s"
                         % (pdr, "; ".join(seeds_txt), pooled_txt,
                            "  [consistent beat]" if cons[pdr] else ""))

    verdict_lines("regime (i) replay-default", comps_i, nb_i, cons_i, scope_i)
    lines.append("")
    verdict_lines("regime (ii) contended (replay-tight m in {0.6,0.8} + storm)",
                  comps_ii, nb_ii, cons_ii, scope_ii)
    lines.append("")
    lines.append("PRIMARY verdict = regime (ii) (per the amendment, the verdict "
                 "regime is where methods separate; regime (i) is reported as "
                 "the capacity-adequate-regime finding either way).")
    lines.append("")

    # -- mean WWT tables --
    lines.append("## Mean WWT per method")
    lines.append("")
    lines.append("rollcp2 runs on the first-8-per-cell subsample. To compare "
                 "like with like, its column is the mean over the paired "
                 "subsample ids and the extra edd@roll column is EDD "
                 "restricted to exactly those ids (same n, shown inline); all "
                 "other columns are full-cell means. Campuses %s are "
                 "E4 held-out: shown for completeness, excluded from verdicts."
                 % sorted(HELDOUT_CAMPUSES))
    lines.append("")
    for regime in regime_order:
        rsub = df[df["regime"] == regime]
        if rsub.empty:
            continue
        if regime == "replay-tight":
            md_mean_block(lines, rsub,
                          ["campus", "size", "crew_multiplier"],
                          "Regime %s (per campus x size x m)" % regime)
        elif regime == "storm":
            md_mean_block(lines, rsub,
                          ["campus", "size", "arrival_multiplier",
                           "crew_multiplier"],
                          "Regime %s (per campus x size x cell)" % regime)
        elif regime == "pmmix":
            gcols = ["campus", "size"]
            if "pm_share_override" in rsub.columns:
                gcols.append("pm_share_override")
            gcols.append("crew_multiplier")
            md_mean_block(lines, rsub, gcols,
                          "Regime %s (per campus x size x cell)" % regime)
        else:
            md_mean_block(lines, rsub, ["campus", "size"],
                          "Regime %s (per campus x size)" % regime)
        st = roll_stats.get(regime)
        if st is not None:
            lines.append("**rollcp2 vs best-PDR (same ids)** — best PDR = "
                         "%s; n=%d; mean WWT %.2f (rollcp2) vs %.2f (%s); "
                         "mean diff %+.2f; Wilcoxon p %s; W/T/L %d/%d/%d "
                         "(eps=%.1f; wins = rollcp2 lower)."
                         % (st["pdr"], st["n_pairs"], st["mean_roll"],
                            st["mean_pdr"], st["pdr"], st["delta"],
                            _fmt_p(st["wilcoxon_p"]), st["wins"], st["ties"],
                            st["losses"], eps))
            lines.append("")

    # -- latency --
    lines.append("## Decision latency (mean ms per decision, feasible rows)")
    lines.append("")
    lat = df[df["feasible"] == 1].groupby("method")["mean_ms_per_decision"] \
        .mean().reindex(METHOD_COLS).dropna()
    lines.append("| method | mean ms/decision |")
    lines.append("|---|---|")
    for meth, v in lat.items():
        note = " (per replan; includes the 2 s CP-SAT budget)" \
            if meth == ROLLCP else ""
        lines.append("| %s | %.3f%s |" % (meth, v, note))
    lines.append("")
    lines.append("Full comparison statistics: `gateB_summary.csv` "
                 "(scope_type in {cell, regime-campus-size, verdict-*, "
                 "rollcp2-vs-best-pdr}; "
                 "seed 'pooled' = 3 seeds' paired diffs concatenated, "
                 "supplementary to the per-seed verdict rule; in "
                 "rollcp2-vs-best-pdr rows seed='rollcp2' and the mean_rl "
                 "column holds rollcp2's mean).")
    lines.append("")

    md_path = out_dir / "gateB_summary.md"
    md_path.write_text("\n".join(lines))

    # ---- LaTeX table ---------------------------------------------------------
    tex_cols = []          # (header, subset)
    t = df[df["campus"].isin(VERDICT_CAMPUSES)]
    tex_cols.append(("Replay default",
                     t[t["regime"] == "replay-default"]))
    tex_cols.append(("Replay $m{=}0.8$",
                     t[(t["regime"] == "replay-tight")
                       & (t["crew_multiplier"] == 0.8)]))
    tex_cols.append(("Replay $m{=}0.6$",
                     t[(t["regime"] == "replay-tight")
                       & (t["crew_multiplier"] == 0.6)]))
    tex_cols.append(("Storm", t[t["regime"] == "storm"]))

    label = {"edd": "EDD", "wspt": "WSPT", "atc": "ATC", "pfifo": "pFIFO",
             "mor": "MOR", "random": "Random",
             ROLLCP: "Rolling CP-SAT (2\\,s)$^{\\dagger}$"}
    for s in RL_SEEDS:
        label[_rl_method(s)] = "Policy (seed %d)" % s

    def cell_mean(sub, meth):
        rows = sub[(sub["method"] == meth) & (sub["feasible"] == 1)]
        return float(rows["wwt"].mean()) if len(rows) else None

    tex = []
    tex.append("% Gate B: mean WWT by method x regime, verdict campuses "
               "{5,9,10,12}, sizes 150+400 pooled.")
    tex.append("% Generated by scripts/p4_analysis.py from "
               "results/p4_dyneval/results.csv.")
    tex.append("% $^{\\dagger}$ rolling CP-SAT and the paired 'EDD (same ids)' "
               "reference row are evaluated on the first-8-per-cell subsample "
               "(identical instance ids).")
    tex.append("\\begin{tabular}{l%s}" % ("r" * len(tex_cols)))
    tex.append("\\toprule")
    tex.append("Method & " + " & ".join(h for h, _ in tex_cols) + " \\\\")
    tex.append("\\midrule")
    for meth in ALL_PDRS:
        vals = [cell_mean(sub, meth) for _, sub in tex_cols]
        tex.append("%s & %s \\\\" % (label[meth],
                                     " & ".join(_fmt(v) for v in vals)))
    tex.append("\\midrule")
    rl_matrix = []
    for meth in RL_METHODS:
        vals = [cell_mean(sub, meth) for _, sub in tex_cols]
        rl_matrix.append(vals)
        tex.append("%s & %s \\\\" % (label[meth],
                                     " & ".join(_fmt(v) for v in vals)))
    rl_mean = []
    for j in range(len(tex_cols)):
        col = [row[j] for row in rl_matrix if row[j] is not None]
        rl_mean.append(float(np.mean(col)) if col else None)
    tex.append("Policy (mean of %d seeds) & %s \\\\"
               % (len(RL_SEEDS), " & ".join(_fmt(v) for v in rl_mean)))
    tex.append("\\midrule")
    roll_vals, eddroll_vals = [], []
    for _, sub in tex_cols:
        rm, em, _n = _paired_roll_means(sub)
        roll_vals.append(rm)
        eddroll_vals.append(em)
    tex.append("%s & %s \\\\" % (label[ROLLCP],
                                 " & ".join(_fmt(v) for v in roll_vals)))
    tex.append("EDD (same ids)$^{\\dagger}$ & %s \\\\"
               % " & ".join(_fmt(v) for v in eddroll_vals))
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex_path = out_dir / "tab_gateB.tex"
    tex_path.write_text("\n".join(tex) + "\n")

    print("Wrote %s (%d comparison rows)" % (comp_csv, len(comp_df)))
    print("Wrote %s" % md_path)
    print("Wrote %s" % tex_path)
    if nb_i is not None:
        print("Gate B regime (i) replay-default : beats %d/5 -> %s"
              % (nb_i, "PASS" if nb_i >= N_BEAT_REQUIRED else "FAIL"))
    if nb_ii is not None:
        print("Gate B regime (ii) contended     : beats %d/5 -> %s"
              % (nb_ii, "PASS" if nb_ii >= N_BEAT_REQUIRED else "FAIL"))


if __name__ == "__main__":
    main()
