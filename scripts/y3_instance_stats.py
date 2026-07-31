#!/usr/bin/env python
"""Y3 Phase-7 deliverable 2: the BENCHMARK INSTANCE-STATISTICS table.

One row per campus used in the paper (the roster of Table "Campus roster" in
setup.tex: C9, C10, C12, C1, C2, C5), describing the instances every reported
number for that campus is scored on, before any result is quoted.

Columns, all computed here from the instance files themselves:
  instances used, work orders per instance (mean), distinct trades, crew size,
  realised pooled utilisation, median eligible orders at a dispatch decision,
  share of decisions with a single feasible candidate, and the recorded
  priority-class mix.

Outputs
-------
results/y3_p7/instance_stats.json  full per-campus statistics, the evaluated
                                   instance ids, and every cross-check.
results/y3_p7/instance_stats.csv   one row per campus.
results/y3_p7/instance_stats.tex   ready-to-paste LaTeX: the \\newcommand block
                                   for macros.tex and the table itself,
                                   generated from the same numbers.

The evaluated slice per campus is taken from the driver that produced the
paper's numbers, not restated:
  C9 / C10 / C12 / C5   scripts/y3_p4_m0grid.py -- released storm2 high-load
                  track at u=100, evaluation on files[20:30] (16 train + 4 probe
                  + 10 held out).
  C1 / C2         scripts/y3_p6_transfer.py -- replay track, size 150,
                  crew-scaled to pooled utilisation ~1.0 and filtered to the
                  util band, evaluation on the 21st-30th qualifying instances.
                  Crew size and utilisation are those of the CREW-SCALED
                  instance, which is what the transfer evaluation runs on.

Every reconstructed slice is then checked, id by id, against the instance ids
recorded in the committed result files (results/y3_p4/cache for the storm2
campuses, results/y3_p6/m0_contention_raw.csv for C1/C2), so the table cannot
describe instances other than the ones the published numbers were scored on.

Queue depth and the single-candidate share are measured by running the tuned
Apparent Tardiness Cost rule (k=2, the deployed baseline) on each evaluation
instance and recording the number of eligible orders at every dispatch decision,
the same instrumentation as scripts/y3_verc_task2.py. They are properties of the
instance and the rule alone: no overlay, no estimator, no seed.

CROSS-CHECK against published macros. macros.tex already publishes queue depths
and a single-candidate share (\\QueueCnine, \\QueueCten, \\QueueCtwelve,
\\ForcedCtwelve) computed by scripts/y3_verc_task2.py over the FIRST 12 storm2
u100 instances, and utilisations (\\utilsat, \\utilbusy) recorded in
results/y3_p4/m0_gate_summary.json. This script recomputes both protocols and
reports every comparison in instance_stats.json under "cross_checks"; a
disagreement is printed as a MISMATCH line and never silently overwritten.

Compute: CPU, single numeric thread, no training. Pin it to four cores:

    cd <repo> && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-3 \\
        python scripts/y3_instance_stats.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import json
import sys
import time
from collections import Counter

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from fmwos import pdrs                                           # noqa: E402
from fmwos.env import DispatchEnv                                # noqa: E402

import y3_p4_m0grid as G                                         # noqa: E402
import y3_p6_transfer as T                                       # noqa: E402

_OUT = os.path.join(_ROOT, "results", "y3_p7")
_GATE = os.path.join(_ROOT, "results", "y3_p4", "m0_gate_summary.json")
_VERC = os.path.join(_ROOT, "results", "y3_verc", "task2_c12_diag.json")
_P4CACHE = os.path.join(_ROOT, "results", "y3_p4", "cache")
_P6RAW = os.path.join(_ROOT, "results", "y3_p6", "m0_contention_raw.csv")
_HARVEST = os.path.join(_ROOT, "results", "y3_p5", "harvest",
                        "primary_multiseed_summary.json")

N_TRAIN, N_PROBE, N_EVAL = 16, 4, 10   # verified against the committed cache
U_HEADLINE = 100

# Campus roster, in the order of the paper's campus-roster table.
ROSTER = [
    (9,  "storm2", "Primary contention campus; the headline cell"),
    (10, "storm2", "Larger second campus; scaled-up confirmation"),
    (12, "storm2", "Boundary: forced ordering"),
    (1,  "replay", "Held-out transfer target"),
    (2,  "replay", "Held-out transfer target"),
    (5,  "storm2", "Off-map, small-denominator corroboration"),
]

# Published macros this script cross-checks (macros.tex, with their provenance).
PUBLISHED = {
    "QueueCnine": {"value": 15, "campus": 9, "field": "median_q"},
    "QueueCten": {"value": 65, "campus": 10, "field": "median_q"},
    "QueueCtwelve": {"value": 1, "campus": 12, "field": "median_q"},
    "ForcedCtwelve": {"value": 52.6, "campus": 12, "field": "frac_forced_pct"},
    "utilsat": {"value": 1.00, "campus": 9, "field": "util_pool"},
}
VERC_PROTOCOL_N = 12          # y3_verc_task2.py used the first 12 instances


def _load(p):
    with open(p) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Instance pools                                                              #
# --------------------------------------------------------------------------- #
def published_eval_ids(campus, track):
    """The instance ids the committed result files record for this campus's
    reported cell (the anti-contamination check on the slice)."""
    if track == "storm2":
        for f in sorted(os.listdir(_P4CACHE)):
            d = _load(os.path.join(_P4CACHE, f))
            if (d["campus"] == campus and d["regime"] == "storm2"
                    and d.get("u") == U_HEADLINE and d["beta"] == 1.0
                    and d["rho"] == 0.25 and d["seed"] == 301):
                return d["inst_ids"], "results/y3_p4/cache/%s" % f
        raise RuntimeError("no committed cache cell for campus %d" % campus)
    ids = []
    with open(_P6RAW) as fh:
        for r in csv.DictReader(fh):
            if int(r["campus"]) == campus and float(r["beta"]) == 1.0 \
                    and int(r["seed"]) == 301:
                ids.append(r["inst_id"])
    if not ids:
        raise RuntimeError("no committed transfer rows for campus %d" % campus)
    return ids, "results/y3_p6/m0_contention_raw.csv"


def eval_pool(campus, track):
    """(label, [(instance_for_stats, instance_that_is_dispatched)]) for the
    slice the paper evaluates this campus on."""
    if track == "storm2":
        files = G.locate_files(campus, "storm2", u=U_HEADLINE)
        need = N_TRAIN + N_PROBE + N_EVAL
        assert len(files) >= need, "c%02d: %d files, need %d" % (campus, len(files), need)
        sel = files[N_TRAIN + N_PROBE:need]
        insts = [_load(p) for p in sel]
        label = "storm2 u%d, files[%d:%d]" % (U_HEADLINE, N_TRAIN + N_PROBE, need)
        return label, [(i, i) for i in insts]
    # replay track, crew-scaled to util ~1 (the transfer protocol)
    picks = T.select_scaled(campus, T.N_TRAIN + T.N_PROBE + T.N_EVAL)
    assert len(picks) >= T.N_TRAIN + T.N_PROBE + T.N_EVAL, \
        "c%02d: only %d qualifying instances" % (campus, len(picks))
    sel = picks[T.N_TRAIN + T.N_PROBE:T.N_TRAIN + T.N_PROBE + T.N_EVAL]
    label = ("replay size %d crew-scaled to util~%.1f, qualifying[%d:%d]"
             % (T.SIZE, T.TARGET_UTIL, T.N_TRAIN + T.N_PROBE,
                T.N_TRAIN + T.N_PROBE + T.N_EVAL))
    return label, [(orig, run) for orig, run, _u in sel]


# --------------------------------------------------------------------------- #
# Per-instance measurements                                                   #
# --------------------------------------------------------------------------- #
def queue_trace(inst):
    """Eligible-order count at every dispatch decision under the tuned ATC rule
    (k=2). Same instrumentation as scripts/y3_verc_task2.py."""
    q = []
    base = pdrs.get_rule("atc")

    def pick(queue, t, rng):
        q.append(len(queue))
        return base(queue, t, rng)

    DispatchEnv(inst).run_policy(pick, method="atc", seed=0)
    return q


def pooled_util(inst):
    """Sum of processing time over crew x horizon; the same quantity as
    y3_p4_m0grid._utilization (storm2, horizon 80 bh) and
    y3_p6_transfer.pooled_util (replay, meta window_bh)."""
    win = float(inst["meta"]["window_bh"])
    p = sum(float(w["p_bh"]) for w in inst["work_orders"])
    k = len(inst["technicians"])
    return p / (k * win) if k * win > 0 else float("nan")


def measure(pairs, want_queue=True):
    """Aggregate the statistics over a list of (stats_instance, run_instance)."""
    n_wos, n_trades, crew, util = [], [], [], []
    cls = Counter()
    qs = []
    forced = dec_n = 0
    for stats_inst, run_inst in pairs:
        n_wos.append(len(stats_inst["work_orders"]))
        n_trades.append(len({w["trade"] for w in stats_inst["work_orders"]}))
        crew.append(len(run_inst["technicians"]))
        util.append(pooled_util(run_inst))
        for w in stats_inst["work_orders"]:
            cls[int(w["priority"])] += 1
        if want_queue:
            tr = queue_trace(run_inst)
            qs.extend(tr)
            dec_n += len(tr)
            forced += sum(1 for q in tr if q <= 1)
    tot = sum(cls.values())
    out = {
        "n_instances": len(pairs),
        "work_orders_mean": float(np.mean(n_wos)),
        "work_orders_min": int(min(n_wos)), "work_orders_max": int(max(n_wos)),
        "trades_mean": float(np.mean(n_trades)),
        "trades_min": int(min(n_trades)), "trades_max": int(max(n_trades)),
        "crew_mean": float(np.mean(crew)),
        "crew_min": int(min(crew)), "crew_max": int(max(crew)),
        "util_pool_mean": float(np.mean(util)),
        "util_pool_min": float(min(util)), "util_pool_max": float(max(util)),
        "class_counts": {str(c): int(cls.get(c, 0)) for c in (1, 2, 3, 4)},
        "class_share_pct": {str(c): 100.0 * cls.get(c, 0) / tot for c in (1, 2, 3, 4)},
        "n_orders_total": int(tot),
    }
    if want_queue:
        out.update({
            "n_decisions": int(dec_n),
            "queue_median": float(np.median(qs)),
            "queue_mean": float(np.mean(qs)),
            "queue_p95": float(np.percentile(qs, 95)),
            "single_candidate_share": forced / dec_n if dec_n else float("nan"),
            "single_candidate_share_pct": 100.0 * forced / dec_n if dec_n else float("nan"),
        })
    return out


# --------------------------------------------------------------------------- #
# Ready-to-paste LaTeX (generated here so the table cannot drift from the data) #
# --------------------------------------------------------------------------- #
_WORD = {9: "nine", 10: "ten", 12: "twelve", 1: "one", 2: "two", 5: "five"}
_TRACK_LABEL = {"storm2": "High-load, saturated", "replay": "Replay, crew-scaled"}


def _th(x, dp=0):
    return format(float(x), ",.%df" % dp).replace(",", "{,}")


def write_tex(path, detail):
    macros = []
    for campus, track, _role in ROSTER:
        st = detail["C%d" % campus]
        w = _WORD[campus]
        macros.append(("QueueC" + w, "%.0f" % st["queue_median"],
                       "results/y3_p7/instance_stats.json:campuses.C%d."
                       "queue_median (held-out evaluation slice)" % campus))
        macros.append(("ForcedC" + w,
                       "%.1f\\%%" % st["single_candidate_share_pct"],
                       "results/y3_p7/instance_stats.json:campuses.C%d."
                       "single_candidate_share_pct" % campus))
    L = ["% ------------------------------------------------------------------"
         "---------",
         "% GENERATED by scripts/y3_instance_stats.py -- do not hand-edit; "
         "re-run the script.",
         "% Instance-statistics table, held-out evaluation slice per campus.",
         "% NOTE: QueueCnine / QueueCten / ForcedCtwelve REPLACE the values in "
         "macros.tex,",
         "%       which were measured on the FIRST 12 instances of each track "
         "instead of the",
         "%       held-out slice (old: 15 / 65 / 52.6\\%). See instance_stats."
         "json:cross_checks.",
         "% ------------------------------------------------------------------"
         "---------",
         "",
         "%% --- macros for macros.tex ---"]
    for name, val, src in macros:
        L.append("\\newcommand{\\%s}{%s} %% %s" % (name, val, src))
    L += ["", r"\begin{table}[pos=htbp]",
          r"""\caption{The benchmark instances behind every reported number, one row per
campus. Statistics are computed on the held-out evaluation instances of the cell
each campus is reported at: the released high-load track at saturation for C9,
C10, C12 and C5, and the replay track crew-scaled to the same utilisation for
the two held-out transfer campuses C1 and C2, whose crew size is therefore a
mean over instances. Pooled utilisation is total processing time divided by crew
size times horizon. Median queue is the number of eligible orders facing the
dispatcher at a decision, and the single-candidate share is the fraction of
decisions at which only one order is eligible, so the dispatcher has no choice;
both are measured by running the tuned rule on these same instances. Class
shares are rounded, and C5 carries no recorded class-1 or class-2 orders at all.
\textbf{Takeaway:} the six campuses span the full range of the condition the
method needs, from the deep queues of C9 and C10, where the dispatcher has real
freedom to reorder work, to C12 and C2, which force the pick at roughly half of
all decisions.}""",
          r"\label{tab:instances}", r"\centering", r"\footnotesize",
          r"\begin{tabular}{@{} l l r r c r c r r c @{}}", r"\toprule",
          r"       & Evaluation & Instances & Orders per &        &      & "
          r"Pooled      & Median & Single-   & Class mix \\",
          r"Campus & track      & ($n$)     & instance   & Trades & Crew & "
          r"utilisation & queue  & candidate & 1/2/3/4 (\%) \\",
          r"\midrule"]
    for campus, track, _role in ROSTER:
        st = detail["C%d" % campus]
        w = _WORD[campus]
        trades = ("%d" % st["trades_min"] if st["trades_min"] == st["trades_max"]
                  else "%d--%d" % (st["trades_min"], st["trades_max"]))
        crew = ("%d" % st["crew_min"] if st["crew_min"] == st["crew_max"]
                else "%.1f" % st["crew_mean"])
        mix = "/".join("%.0f" % st["class_share_pct"][str(c)] for c in (1, 2, 3, 4))
        L.append(r"C%d & %s & %d & %s & %s & %s & %.2f & \QueueC%s & \ForcedC%s "
                 r"& %s \\"
                 % (campus, _TRACK_LABEL[track], st["n_instances"],
                    _th(st["work_orders_mean"]), trades, crew,
                    st["util_pool_mean"], w, w, mix))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(L))


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
CSV_COLS = ["campus", "role", "cell", "n_instances", "work_orders_mean",
            "trades", "crew", "util_pool_mean", "queue_median",
            "single_candidate_share_pct", "class1_pct", "class2_pct",
            "class3_pct", "class4_pct", "n_decisions", "queue_mean",
            "queue_p95", "n_orders_total"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-crosscheck", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(_OUT, exist_ok=True)
    t0 = time.time()

    rows, detail, slice_checks = [], {}, []
    for campus, track, role in ROSTER:
        t1 = time.time()
        label, pairs = eval_pool(campus, track)
        ids = [a["meta"]["id"] for a, _b in pairs]
        pub_ids, pub_src = published_eval_ids(campus, track)
        same = (ids == pub_ids)
        slice_checks.append({"campus": "C%d" % campus, "source": pub_src,
                             "n_reconstructed": len(ids), "n_published": len(pub_ids),
                             "identical": bool(same),
                             "published_ids": pub_ids, "reconstructed_ids": ids})
        assert same, ("C%d: reconstructed slice differs from the committed one "
                      "(%s)" % (campus, pub_src))
        st = measure(pairs)
        st["campus"] = campus
        st["track"] = track
        st["role"] = role
        st["cell"] = label
        st["instance_ids"] = ids
        st["instance_ids_source"] = pub_src
        detail["C%d" % campus] = st
        rows.append({
            "campus": "C%d" % campus, "role": role, "cell": label,
            "n_instances": st["n_instances"],
            "work_orders_mean": "%.1f" % st["work_orders_mean"],
            "trades": ("%d" % st["trades_min"]) if st["trades_min"] == st["trades_max"]
                      else "%.1f" % st["trades_mean"],
            "crew": ("%d" % st["crew_min"]) if st["crew_min"] == st["crew_max"]
                    else "%.1f" % st["crew_mean"],
            "util_pool_mean": "%.3f" % st["util_pool_mean"],
            "queue_median": "%.0f" % st["queue_median"],
            "single_candidate_share_pct": "%.1f" % st["single_candidate_share_pct"],
            "class1_pct": "%.1f" % st["class_share_pct"]["1"],
            "class2_pct": "%.1f" % st["class_share_pct"]["2"],
            "class3_pct": "%.1f" % st["class_share_pct"]["3"],
            "class4_pct": "%.1f" % st["class_share_pct"]["4"],
            "n_decisions": st["n_decisions"], "queue_mean": "%.1f" % st["queue_mean"],
            "queue_p95": "%.0f" % st["queue_p95"],
            "n_orders_total": st["n_orders_total"],
        })
        print("[C%d] %-58s n=%d wos=%.0f trades=%d crew=%d util=%.3f "
              "medq=%.0f forced=%.1f%% classes %s (%.0fs)"
              % (campus, label, st["n_instances"], st["work_orders_mean"],
                 st["trades_max"], st["crew_max"], st["util_pool_mean"],
                 st["queue_median"], st["single_candidate_share_pct"],
                 "/".join("%.1f" % st["class_share_pct"][str(c)] for c in (1, 2, 3, 4)),
                 time.time() - t1), flush=True)

    # ------------------------------------------------------------------ #
    # Cross-checks against the published macros                          #
    # ------------------------------------------------------------------ #
    checks = []
    if not args.skip_crosscheck:
        verc = {d["campus"]: d for d in _load(_VERC)}
        gate = _load(_GATE)["cells"]
        for campus in (9, 10, 12):
            files = G.locate_files(campus, "storm2", u=U_HEADLINE)[:VERC_PROTOCOL_N]
            pairs = [(i, i) for i in (_load(p) for p in files)]
            st = measure(pairs)
            pub = verc[campus]
            checks.append({
                "what": "median eligible orders at a decision, campus C%d" % campus,
                "protocol": "first %d storm2 u%d instances (scripts/y3_verc_task2.py)"
                            % (VERC_PROTOCOL_N, U_HEADLINE),
                "published_source": "results/y3_verc/task2_c12_diag.json:median_q",
                "published": pub["median_q"], "recomputed": st["queue_median"],
                "agrees": bool(abs(pub["median_q"] - st["queue_median"]) < 1e-9),
                "evaluated_slice_value": detail["C%d" % campus]["queue_median"],
            })
            checks.append({
                "what": "single-candidate share, campus C%d" % campus,
                "protocol": "first %d storm2 u%d instances (scripts/y3_verc_task2.py)"
                            % (VERC_PROTOCOL_N, U_HEADLINE),
                "published_source": "results/y3_verc/task2_c12_diag.json:frac_forced",
                "published": 100.0 * pub["frac_forced"],
                "recomputed": st["single_candidate_share_pct"],
                "agrees": bool(abs(100.0 * pub["frac_forced"]
                                   - st["single_candidate_share_pct"]) < 0.05),
                "evaluated_slice_value":
                    detail["C%d" % campus]["single_candidate_share_pct"],
            })
        for campus in (9, 10):
            ck = "c%d_storm2_u%d_b1.00_r0.25" % (campus, U_HEADLINE)
            checks.append({
                "what": "realised pooled utilisation, campus C%d held-out slice" % campus,
                "protocol": "held-out evaluation instances (the grid's own util_pool)",
                "published_source": "results/y3_p4/m0_gate_summary.json:%s:util_pool" % ck,
                "published": gate[ck]["util_pool"],
                "recomputed": detail["C%d" % campus]["util_pool_mean"],
                "agrees": bool(abs(gate[ck]["util_pool"]
                                   - detail["C%d" % campus]["util_pool_mean"]) < 1e-6),
            })
        pub_ids = _load(_HARVEST)["eval_inst_ids"]
        checks.append({
            "what": "C9 held-out instance ids vs the published headline ladder",
            "protocol": "id-by-id",
            "published_source": "results/y3_p5/harvest/primary_multiseed_summary.json"
                                ":eval_inst_ids",
            "published": len(pub_ids), "recomputed": len(detail["C9"]["instance_ids"]),
            "agrees": bool(pub_ids == detail["C9"]["instance_ids"]),
        })
        # macros.tex printed values
        for macro, spec in PUBLISHED.items():
            c = spec["campus"]
            if spec["field"] == "median_q":
                mine = detail["C%d" % c]["queue_median"]
                printed = "%.0f" % mine
                ok = float(printed) == float(spec["value"])
            elif spec["field"] == "frac_forced_pct":
                mine = detail["C%d" % c]["single_candidate_share_pct"]
                printed = "%.1f" % mine
                ok = abs(float(printed) - spec["value"]) < 1e-9
            else:
                mine = detail["C%d" % c]["util_pool_mean"]
                printed = "%.2f" % mine
                ok = abs(float(printed) - spec["value"]) < 1e-9
            checks.append({
                "what": "macros.tex \\%s" % macro,
                "protocol": "recomputed on the EVALUATED slice of C%d" % c,
                "published_source": "paper/macros.tex \\%s" % macro,
                "published": spec["value"], "recomputed": mine,
                "recomputed_printed": printed, "agrees": bool(ok),
            })

    for ch in checks:
        tag = "ok      " if ch["agrees"] else "MISMATCH"
        print("[check %s] %-62s published %s vs recomputed %s"
              % (tag, ch["what"], ch["published"],
                 ch.get("recomputed_printed", ch["recomputed"])), flush=True)

    out = {
        "protocol": {
            "slice": "the evaluation instances each campus's reported numbers are "
                     "scored on (see per-campus 'cell')",
            "queue_depth": "eligible orders at every dispatch decision under the "
                           "tuned ATC rule (k=2); no overlay, no estimator, no seed",
            "utilisation": "sum of processing time / (crew x instance window_bh)",
            "classes": "recorded priority classes 1-4, pooled over the orders of "
                       "the evaluated instances",
            "sources": ["scripts/y3_p4_m0grid.py (storm2 slices)",
                        "scripts/y3_p6_transfer.py (C1/C2 crew-scaled replay slices)",
                        "scripts/y3_verc_task2.py (queue instrumentation)"],
        },
        "campuses": detail,
        "evaluated_slice_verification": slice_checks,
        "cross_checks": checks,
        "all_cross_checks_agree": all(c["agrees"] for c in checks),
    }
    with open(os.path.join(_OUT, "instance_stats.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(_OUT, "instance_stats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    write_tex(os.path.join(_OUT, "instance_stats.tex"), detail)
    print("[y3_p7] wrote instance_stats.{json,csv} (%.0fs); cross-checks %s"
          % (time.time() - t0, "ALL AGREE" if out["all_cross_checks_agree"]
             else "HAVE MISMATCHES -- see instance_stats.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
