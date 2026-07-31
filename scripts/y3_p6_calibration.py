#!/usr/bin/env python
"""Paper Y3, Phase P6, TASK 2 -- oracle calibration against real FMUCD signals.

Question (proposal Sec. validity / Threats to Validity): does FMUCD record any
real signal that proxies supervisor OVERRIDE behaviour (reassignments, reopened /
re-prioritised tickets, priority-vs-actual-completion mismatch) that could anchor
plausible ranges for the supervisor parameters beta (recoverable-information
share) and epsilon (override noise)? If not, say so honestly.

Finding (no fabrication; see notes/phase6_transfer.md for the full write-up):

  * FMUCD is a work-order LABOUR-LINE LEDGER, not an event log. The columns the
    pipeline ingests (fmwos.io.USECOLS) are: UniversityID, Country,
    State/Province, BuildingID, Size, Type, SystemCode, SystemDescription,
    SubsystemCode, WOID, WOPriority, WOStartDate, WOEndDate, WODuration, PPM/UPM,
    LaborCost, TotalCost, LaborHours. There is NO status / reopen / reassignment /
    re-prioritisation / assignee / edit-history field. The "reassignments or
    reopened / re-prioritised tickets" the proposal hoped for DO NOT EXIST in the
    corpus, so there is no direct override record to calibrate against.

  * The ONE indirect proxy that exists is the priority-vs-realised-completion
    mismatch, which Y1 already exploited to build the v2 priority mapping: per
    campus it set each numeric priority scale's DIRECTION by the Spearman
    correlation between the recorded-priority rank and the median realised
    corrective completion duration (WOEndDate - WOStartDate). This descriptor
    documents that the recorded priority is an UNRELIABLE urgency label (the
    premise of the whole paper), but it is a confounded aggregate proxy
    (completion time reflects urgency AND crew load AND job size), not an override
    record, and it cannot point-identify beta or epsilon.

This script does NOT touch the raw FMUCD corpus (absent from the repo:
data/raw/ holds only the packaged instances archive; FMUCD.csv is download-
separate per the README). It reads the COMMITTED calibration artifact
results/p1_calib/priority_mapping.csv (identical byte-for-byte to the Y1 repo's)
and distils the recorded-priority reliability descriptor per campus into
results/y3_p6/priority_reliability.csv. That descriptor is the extent of the real
FMUCD signal available to bracket the oracle; the honest verdict (labelled
controlled proxy; beta / epsilon swept; relative comparisons on identical
instances) is argued from it in the notes.

Run:  python scripts/y3_p6_calibration.py
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAP = os.path.join(_ROOT, "results", "p1_calib", "priority_mapping.csv")
_OUT = os.path.join(_ROOT, "results", "y3_p6", "priority_reliability.csv")

# Overlay class -> SLA (business hours) and the SLA in DAYS (8 bh = 1 work day),
# to compare the recorded classes' realised completion against their contractual
# lead time. Locked (overlay.SLA_OF_CLASS).
SLA_BH = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}
SLA_DAYS = {c: v / 8.0 for c, v in SLA_BH.items()}


def main():
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    rows = []
    with open(_MAP) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    # per campus: rule composition, spearman rho/direction (r5c campuses only),
    # and per-mapped-class row-weighted mean of the median CM completion duration.
    campuses = sorted({r["campus"] for r in rows}, key=int)
    out_rows = []
    for c in campuses:
        sub = [r for r in rows if r["campus"] == c]
        total = sum(int(r["rows"]) for r in sub)
        rho = next((r["spearman_rho"] for r in sub if r["rule"] == "r5c"
                    and r["spearman_rho"]), "")
        direction = next((r["direction"] for r in sub if r["rule"] == "r5c"
                          and r["direction"]), "")
        rule_rows = defaultdict(int)
        for r in sub:
            rule_rows[r["rule"]] += int(r["rows"])
        # row-weighted mean median-duration per mapped class over CM rows that
        # carry a duration (PM rows have none). This exposes whether the recorded
        # classes actually order realised completion.
        cls_num = defaultdict(float)
        cls_den = defaultdict(int)
        for r in sub:
            md = r["median_cm_duration_days"]
            if r["is_pm_split"] == "cm" and md not in ("", "None", None):
                k = int(r["mapped_class"])
                cls_num[k] += float(md) * int(r["rows"])
                cls_den[k] += int(r["rows"])
        cls_dur = {k: (cls_num[k] / cls_den[k]) for k in cls_den}
        # monotone iff class-median duration is non-decreasing in the class index
        seq = [cls_dur[k] for k in sorted(cls_dur)]
        monotone = all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))
        # has a usable numeric urgency scale at all?
        has_numeric_scale = rule_rows.get("r5c", 0) > 0
        pm_share = rule_rows.get("r5a", 0) / total if total else 0.0
        default3_share = rule_rows.get("r5d", 0) / total if total else 0.0

        out_rows.append({
            "campus": c,
            "n_rows": total,
            "has_numeric_priority_scale": int(has_numeric_scale),
            "spearman_rho_rank_vs_completion": rho or "n/a",
            "scale_direction": direction or "n/a",
            "pm_share_r5a": round(pm_share, 4),
            "default3_share_r5d": round(default3_share, 4),
            "keyword_rows_r5b": rule_rows.get("r5b", 0),
            "cm_class1_med_days": round(cls_dur.get(1, float("nan")), 2) if 1 in cls_dur else "n/a",
            "cm_class2_med_days": round(cls_dur.get(2, float("nan")), 2) if 2 in cls_dur else "n/a",
            "cm_class3_med_days": round(cls_dur.get(3, float("nan")), 2) if 3 in cls_dur else "n/a",
            "cm_class4_med_days": round(cls_dur.get(4, float("nan")), 2) if 4 in cls_dur else "n/a",
            "class_duration_monotone": int(monotone),
        })

    cols = ["campus", "n_rows", "has_numeric_priority_scale",
            "spearman_rho_rank_vs_completion", "scale_direction",
            "pm_share_r5a", "default3_share_r5d", "keyword_rows_r5b",
            "cm_class1_med_days", "cm_class2_med_days", "cm_class3_med_days",
            "cm_class4_med_days", "class_duration_monotone"]
    with open(_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print("[write] %s" % _OUT)
    for r in out_rows:
        print("  c%2s scale=%s rho=%s dir=%s | class med-days(P1..P4)=%s/%s/%s/%s monotone=%d"
              % (r["campus"], r["has_numeric_priority_scale"],
                 r["spearman_rho_rank_vs_completion"], r["scale_direction"],
                 r["cm_class1_med_days"], r["cm_class2_med_days"],
                 r["cm_class3_med_days"], r["cm_class4_med_days"],
                 r["class_duration_monotone"]))
    print("\nSLA(days) by class (contractual lead time, for reference): "
          + ", ".join("P%d=%.1f" % (c, SLA_DAYS[c]) for c in (1, 2, 3, 4)))


if __name__ == "__main__":
    main()
