"""P0 exploratory profiling of FMUCD.

Outputs JSON + CSV summaries under results/p0_profile/:
  - overview.json: row counts, loader round-trip check, cleaning audit
  - per_campus.csv: rows, buildings, date span, PM share, priority mix per campus
  - trades.csv: trade taxonomy (top-level system codes) with counts and hours
  - priority.csv: raw priority value distribution per campus
  - labor_hours.json: LaborHours distribution stats (for R4 cap + generator)
  - arrivals.csv: monthly CM/PM arrival counts per campus (for calibration)

Run: python scripts/p0_profile.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmwos import io  # noqa: E402

RAW = ROOT / "data" / "raw" / "FMUCD.csv"
OUT = ROOT / "results" / "p0_profile"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    df = io.load_raw(RAW)
    t_load = time.time() - t0

    overview: dict = {
        "rows_raw": int(len(df)),
        "load_seconds": round(t_load, 1),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "na_share": {c: round(float(df[c].isna().mean()), 4) for c in df.columns},
    }

    clean, audit = io.clean(df)
    overview["cleaning_audit"] = audit

    # Per-campus summary.
    g = clean.groupby("UniversityID", observed=True)
    per_campus = pd.DataFrame(
        {
            "rows": g.size(),
            "buildings": g["BuildingID"].nunique(),
            "trades": g["trade"].nunique(),
            "first_wo": g["WOStartDate"].min(),
            "last_wo": g["WOStartDate"].max(),
            "pm_share": g["is_pm"].mean().round(4),
            "median_labor_h": g["LaborHours"].median(),
            "country": g["Country"].first(),
        }
    )
    per_campus.to_csv(OUT / "per_campus.csv")

    # Trade taxonomy.
    tg = clean.groupby("trade", observed=True)
    trades = pd.DataFrame(
        {
            "rows": tg.size(),
            "total_labor_h": tg["LaborHours"].sum().round(0),
            "median_labor_h": tg["LaborHours"].median(),
            "description": tg["SystemDescription"].first(),
            "campuses": tg["UniversityID"].nunique(),
        }
    ).sort_values("rows", ascending=False)
    trades.to_csv(OUT / "trades.csv")

    # Raw priority distribution per campus (semantics check before P1 mapping).
    prio = (
        clean.groupby(["UniversityID", "WOPriority"], observed=True, dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    prio.to_csv(OUT / "priority.csv", index=False)

    # LaborHours distribution (generator calibration + R4 evidence).
    lh = clean["LaborHours"]
    overview["labor_hours"] = {
        "mean": float(lh.mean()),
        "std": float(lh.std()),
        "quantiles": {
            str(q): float(lh.quantile(q))
            for q in [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 1.0]
        },
    }

    # Monthly arrivals per campus and PM/CM (calibration input).
    clean["month"] = clean["WOStartDate"].dt.to_period("M").astype(str)
    arr = (
        clean.groupby(["UniversityID", "month", "is_pm"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    arr.to_csv(OUT / "arrivals.csv", index=False)

    # WODuration vs LaborHours sanity: duration appears to be calendar days.
    dur_days = (clean["WOEndDate"] - clean["WOStartDate"]).dt.days
    both = clean["WODuration"].notna() & dur_days.notna()
    overview["woduration_equals_calendar_days_share"] = float(
        (clean.loc[both, "WODuration"].round() == dur_days[both]).mean()
    )

    with open(OUT / "overview.json", "w") as f:
        json.dump(overview, f, indent=2, default=str)

    print(json.dumps(overview["cleaning_audit"], indent=2))
    print(f"rows_raw={overview['rows_raw']}, load={t_load:.0f}s")
    print(per_campus.to_string())


if __name__ == "__main__":
    main()
