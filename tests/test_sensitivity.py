"""E5 sensitivity tests — plain python script (no pytest).

Run:  PYTHONPATH=src python tests/test_sensitivity.py

Covers:
  1. fmwos.sensitivity.scale_sla on a hand-built instance: due windows scaled
     about release by f, releases/priorities/weights/p_bh untouched, meta
     annotated + id suffixed, source instance not mutated; scale_crew re-export.
  2. Runner smoke: scripts/p4_sensitivity.py --limit 3 --workers 2 to a scratch
     --out; assert the CSV has exactly the spec columns, all rows feasible, and
     the expected methods/conditions are present.
  3. Analysis smoke: scripts/p5_sensitivity_analysis.py --in <scratch> runs and
     writes sensitivity_summary.md + tab_sensitivity.tex.

Prints 'ALL SENSITIVITY TESTS PASSED' and deletes the scratch outputs.
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fmwos import sensitivity  # noqa: E402

TOL = 1e-9
EXPECTED_FIELDS = [
    "id", "base_id", "campus", "size", "condition", "sla_multiplier",
    "crew_multiplier", "method", "seed", "feasible", "wwt", "makespan",
    "mean_flow", "breach_share", "breach_p1", "breach_p2", "breach_p3",
    "breach_p4", "wall_seconds",
]
EXPECTED_METHODS = {"edd", "wspt", "atc", "pfifo", "mor", "random",
                    "rl301", "rl302", "rl303"}


def _hand_instance():
    """Tiny two-trade instance with distinct release/due/priority/weight."""
    return {
        "meta": {"id": "hand_0001", "campus": 5, "track": "replay",
                 "size_class": 3},
        "trades": ["D20", "E10"],
        "technicians": [{"id": "T0", "trade": "D20"},
                        {"id": "T1", "trade": "D20"},
                        {"id": "T2", "trade": "E10"}],
        "work_orders": [
            {"id": "W1", "trade": "D20", "p_bh": 3.0, "release_bh": 0.0,
             "due_bh": 8.0, "priority": 1, "weight": 8.0, "is_pm": False},
            {"id": "W2", "trade": "D20", "p_bh": 2.0, "release_bh": 5.0,
             "due_bh": 29.0, "priority": 2, "weight": 4.0, "is_pm": False},
            {"id": "W3", "trade": "E10", "p_bh": 1.5, "release_bh": 10.0,
             "due_bh": 90.0, "priority": 3, "weight": 2.0, "is_pm": True},
        ],
    }


def test_scale_sla(failures):
    for f in (0.5, 1.5, 1.0):
        base = _hand_instance()
        orig_wos = {w["id"]: dict(w) for w in base["work_orders"]}
        out = sensitivity.scale_sla(base, f)

        # source not mutated (deep copy)
        for w in base["work_orders"]:
            if w["due_bh"] != orig_wos[w["id"]]["due_bh"]:
                failures.append("scale_sla mutated the source instance")
                break

        by_id = {w["id"]: w for w in out["work_orders"]}
        for wid, ow in orig_wos.items():
            nw = by_id[wid]
            # due scaled about release by f
            want_due = ow["release_bh"] + f * (ow["due_bh"] - ow["release_bh"])
            if abs(nw["due_bh"] - want_due) > TOL:
                failures.append("f=%s WO %s due_bh %.6f != expected %.6f"
                                % (f, wid, nw["due_bh"], want_due))
            # release, priority, weight, p_bh, trade untouched
            for k in ("release_bh", "priority", "weight", "p_bh", "trade"):
                if nw[k] != ow[k]:
                    failures.append("f=%s WO %s field %s changed %r -> %r"
                                    % (f, wid, k, ow[k], nw[k]))

        # meta annotation + id suffix
        if out["meta"].get("sla_multiplier") != f:
            failures.append("f=%s meta.sla_multiplier=%r != %r"
                            % (f, out["meta"].get("sla_multiplier"), f))
        want_id = "hand_0001_sla%s" % f
        if out["meta"].get("id") != want_id:
            failures.append("f=%s meta.id=%r != %r"
                            % (f, out["meta"].get("id"), want_id))

        # technicians untouched (SLA transform is WO-only)
        if out["technicians"] != base["technicians"]:
            failures.append("f=%s scale_sla altered technicians" % f)

    # tighter deadlines really shrink the window; looser widen it
    base = _hand_instance()
    tight = sensitivity.scale_sla(base, 0.5)
    loose = sensitivity.scale_sla(base, 1.5)
    tb = {w["id"]: w for w in tight["work_orders"]}
    lb = {w["id"]: w for w in loose["work_orders"]}
    for w in base["work_orders"]:
        if not (tb[w["id"]]["due_bh"] < w["due_bh"] < lb[w["id"]]["due_bh"]
                or w["due_bh"] == w["release_bh"]):
            failures.append("WO %s window did not shrink/grow as expected"
                            % w["id"])

    # scale_crew re-exported and callable (capacity knob, no new code)
    if not callable(getattr(sensitivity, "scale_crew", None)):
        failures.append("sensitivity.scale_crew re-export missing")
    else:
        crewed = sensitivity.scale_crew(_hand_instance(), 0.75)
        if crewed["meta"].get("crew_multiplier") != 0.75:
            failures.append("scale_crew re-export did not set crew_multiplier")


def test_runner_and_analysis(failures, scratch):
    env = dict(os.environ, PYTHONPATH=os.path.join(_ROOT, "src"))
    # 1. runner smoke ------------------------------------------------------
    r = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "scripts", "p4_sensitivity.py"),
         "--limit", "3", "--workers", "2", "--out", scratch],
        cwd=_ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        failures.append("runner exited %d\nSTDOUT:\n%s\nSTDERR:\n%s"
                        % (r.returncode, r.stdout, r.stderr[-2000:]))
        return None
    csv_path = os.path.join(scratch, "results.csv")
    if not os.path.exists(csv_path):
        failures.append("runner did not write results.csv")
        return None

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames)
        rows = list(reader)
    if cols != EXPECTED_FIELDS:
        failures.append("csv columns mismatch:\n got %s\n want %s"
                        % (cols, EXPECTED_FIELDS))
    if len(rows) != 3 * len(EXPECTED_METHODS):
        failures.append("expected %d rows (3 configs x %d methods), got %d"
                        % (3 * len(EXPECTED_METHODS), len(EXPECTED_METHODS),
                           len(rows)))
    infeas = [row for row in rows if row["feasible"] != "1"]
    if infeas:
        failures.append("%d infeasible smoke rows (expected 0); e.g. %s/%s"
                        % (len(infeas), infeas[0]["id"], infeas[0]["method"]))
    methods = {row["method"] for row in rows}
    if methods != EXPECTED_METHODS:
        failures.append("method set mismatch: %s" % (methods,))
    conds = {row["condition"] for row in rows}
    if not conds <= {"baseline", "sla0.5", "sla1.5", "crew0.75", "crew1.25"}:
        failures.append("unexpected conditions: %s" % (conds,))
    if "baseline" not in conds:
        failures.append("baseline condition missing from smoke rows")
    # base_id present + id suffixing consistent with condition
    for row in rows:
        if not row["base_id"]:
            failures.append("empty base_id in a smoke row")
            break
        if row["condition"] == "baseline" and row["id"] != row["base_id"]:
            failures.append("baseline id %s != base_id %s"
                            % (row["id"], row["base_id"]))
        if row["condition"] == "sla0.5" and not row["id"].endswith("_sla0.5"):
            failures.append("sla0.5 id %s lacks suffix" % row["id"])

    # 2. analysis smoke ----------------------------------------------------
    a = subprocess.run(
        [sys.executable,
         os.path.join(_ROOT, "scripts", "p5_sensitivity_analysis.py"),
         "--in", scratch],
        cwd=_ROOT, env=env, capture_output=True, text=True)
    if a.returncode != 0:
        failures.append("analysis exited %d\nSTDOUT:\n%s\nSTDERR:\n%s"
                        % (a.returncode, a.stdout, a.stderr[-2000:]))
    for name in ("sensitivity_summary.md", "tab_sensitivity.tex"):
        p = os.path.join(scratch, name)
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            failures.append("analysis did not write non-empty %s" % name)
    md = os.path.join(scratch, "sensitivity_summary.md")
    if os.path.exists(md):
        with open(md) as fh:
            text = fh.read()
        for needle in ("Ranking robustness", "Kendall tau",
                       "Mean WWT per method", "breach-share"):
            if needle not in text:
                failures.append("summary md missing section %r" % needle)
    return rows


def _print_excerpt(rows):
    print("\nSmoke results excerpt (first 9 rows):")
    print("-" * 96)
    print("%-26s %-9s %-7s %-6s %10s %6s"
          % ("id", "condition", "method", "feas", "wwt", "size"))
    print("-" * 96)
    for row in (rows or [])[:9]:
        print("%-26s %-9s %-7s %-6s %10.3f %6s"
              % (row["id"][:26], row["condition"], row["method"],
                 row["feasible"], float(row["wwt"]), row["size"]))
    print("-" * 96)


def main():
    failures = []
    scratch = tempfile.mkdtemp(prefix="e5_smoke_")
    rows = None
    try:
        test_scale_sla(failures)
        rows = test_runner_and_analysis(failures, scratch)
        _print_excerpt(rows)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        print("cleaned scratch: %s" % scratch)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("\nALL SENSITIVITY TESTS PASSED")


if __name__ == "__main__":
    main()
