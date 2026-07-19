"""P1 end-to-end: calibration tables + replay instance corpus (sampling v2).

Pipeline (single pass, deterministic; one RNG seeded 301):
  1. load + clean FMUCD (fmwos.io).
  2. calibration -> results/p1_calib/{priority_mapping.csv, capacity.csv}.
  3. replay sampling v2 (the interface spec "Replay sampling protocol (v2)"): for
     each campus in {1,2,5,9,10,12} x size in {50,150,400} x split in
     {train (anchors <= 2017-12-31), test (anchors >= 2018-01-01)}:
       candidate anchors = every weekday 08:00 across the campus's date range,
       shuffled (RNG 301); an instance = the FIRST `size` WOs released at/after
       the anchor (window_bh = last release_bh, min 1.0); greedy acceptance
       with NON-OVERLAPPING [t0, t0+window] ranges within the cell; skip
       anchors with < `size` WOs remaining; train windows must end before
       2018-01-01; cap 100 accepted per cell.
  4. write instance JSONs + data/processed/instances/index.csv (replay rows
     only; scripts/p2_generator.py appends the generator rows), print summaries.

Run: python scripts/p1_instances.py
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

from fmwos import calib, instances, io  # noqa: E402
from fmwos import timeaxis as ta        # noqa: E402

RAW = ROOT / "data" / "raw" / "FMUCD.csv"
CALIB_OUT = ROOT / "results" / "p1_calib"
INST_ROOT = ROOT / "data" / "processed" / "instances"
SEED = 301
TRAIN_END = pd.Timestamp("2017-12-31")
TEST_START = pd.Timestamp("2018-01-01")
MAX_PER_CELL = 100
# First business moment of the test period, on the absolute bh axis: train
# windows must end strictly before this coordinate (no boundary crossing).
B_TRAIN_END_ABS = ta.abs_bh(pd.Timestamp("2018-01-01 08:00:00"))


def candidate_anchors(first: pd.Timestamp, last: pd.Timestamp) -> list[pd.Timestamp]:
    """Every weekday 08:00 anchor in [first, last] (calendar range)."""
    days = pd.date_range(first.normalize(), last.normalize(), freq="D")
    days = days[days.weekday < 5]
    return [d + pd.Timedelta(hours=8) for d in days]


def overlaps_any(intervals: list[tuple[float, float]], lo: float, hi: float) -> bool:
    """Closed-interval overlap test (touching endpoints count as overlap)."""
    return any(lo <= a_hi and a_lo <= hi for a_lo, a_hi in intervals)


def main() -> None:
    t_start = time.time()
    print("loading + cleaning FMUCD ...", flush=True)
    raw = io.load_raw(RAW)
    clean, audit = io.clean(raw)
    del raw
    print(f"  clean rows: {len(clean):,} ({time.time() - t_start:.0f}s)", flush=True)

    # ---- calibration ------------------------------------------------------- #
    mapping, capacity, tmap, trade_m = calib.write_calibration(clean, CALIB_OUT)
    priority = calib.priority_class_series(clean, mapping)
    print(f"calibration written to {CALIB_OUT}", flush=True)

    # ---- priority class shares + sanity gate (v2: no campus > 25% P1) ------ #
    pc = pd.DataFrame({"campus": clean["UniversityID"].astype("int64").to_numpy(),
                       "priority": priority.to_numpy()})
    print("\n=== priority class shares per campus (v2 mapping, pm+cm) ===")
    print(f"{'campus':>6} {'P1':>7} {'P2':>7} {'P3':>7} {'P4':>7} {'n_rows':>9}")
    p1_violations = []
    for campus in instances.CAMPUS_SET:
        s = pc.loc[pc["campus"] == campus, "priority"]
        shares = s.value_counts(normalize=True)
        p1 = float(shares.get(1, 0.0))
        print(f"{campus:6d} "
              f"{p1:7.3f} {shares.get(2, 0.0):7.3f} "
              f"{shares.get(3, 0.0):7.3f} {shares.get(4, 0.0):7.3f} {len(s):9d}")
        if p1 > 0.25:
            p1_violations.append((campus, p1))
    if p1_violations:
        raise SystemExit(
            f"SANITY GATE FAILED: campus(es) with >25% P1 after v2 mapping: "
            f"{p1_violations}. Stopping before instance generation.")

    # ---- instance corpus (sampling v2: first-N-releases) ------------------- #
    INST_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    index_rows: list[dict] = []
    cell_counts: dict[tuple, int] = {}
    cell_candidates: dict[tuple, int] = {}
    cell_windows: dict[tuple, list[float]] = {}

    for campus in instances.CAMPUS_SET:
        prep = instances.prepare_campus(clean, campus, trade_m, priority)
        trades, techs = instances.technicians_for_campus(capacity, campus)
        camp_ts = clean.loc[clean["UniversityID"].astype("int64") == campus,
                            "WOStartDate"]
        first, last = camp_ts.min(), camp_ts.max()

        anchors = candidate_anchors(first, last)
        anchor_abs = ta.abs_bh_series(pd.Series(anchors))
        cands = list(zip(anchors, anchor_abs))
        idx = np.arange(len(cands))
        rng.shuffle(idx)                       # single RNG, deterministic order
        cands = [cands[i] for i in idx]
        train_cands = [c for c in cands if c[0] <= TRAIN_END]
        test_cands = [c for c in cands if c[0] >= TEST_START]

        for size in instances.SIZES:
            counter = 0  # id counter is shared by train+test within (campus,size)
            for split, split_cands in (("train", train_cands), ("test", test_cands)):
                cell = (campus, size, split)
                cell_candidates[cell] = len(split_cands)
                accepted_ranges: list[tuple[float, float]] = []
                windows: list[float] = []
                for t0, t0_abs in split_cands:
                    pw = instances.probe_window(prep, float(t0_abs), size)
                    if pw is None:
                        continue           # < size WOs remaining in the stream
                    lo, hi, window_bh = pw
                    w_lo, w_hi = float(t0_abs), float(t0_abs) + window_bh
                    if split == "train" and w_hi >= B_TRAIN_END_ABS:
                        continue           # window crosses the split boundary
                    if overlaps_any(accepted_ranges, w_lo, w_hi):
                        continue           # overlaps an accepted window in cell
                    inst_id = f"c{campus:02d}_replay_{size}_{counter:04d}"
                    inst = instances.build_instance(
                        prep, t0, float(t0_abs), lo, hi, window_bh, size,
                        inst_id, campus, trades, techs)
                    rel_dir = Path(f"c{campus:02d}") / "replay" / str(size)
                    out_dir = INST_ROOT / rel_dir
                    out_dir.mkdir(parents=True, exist_ok=True)
                    with open(out_dir / f"{inst_id}.json", "w") as f:
                        json.dump(inst, f, separators=(",", ":"))
                    index_rows.append({
                        "id": inst_id,
                        "campus": campus,
                        "track": "replay",
                        "size_class": size,
                        "split": split,
                        "n_wos": size,
                        "window_start": t0.isoformat(),
                        "window_bh": round(float(window_bh), 4),
                        "path": str(rel_dir / f"{inst_id}.json"),
                    })
                    accepted_ranges.append((w_lo, w_hi))
                    windows.append(window_bh)
                    counter += 1
                    if len(accepted_ranges) >= MAX_PER_CELL:
                        break
                cell_counts[cell] = len(accepted_ranges)
                cell_windows[cell] = windows
            parts = [f"  campus {campus:2d} size {size:3d}:"]
            for split in ("train", "test"):
                w = cell_windows[(campus, size, split)]
                med = f"{np.median(w):8.2f}bh" if w else "     n/a"
                parts.append(f"{split}={cell_counts[(campus, size, split)]:3d}"
                             f" (med_win={med})")
            print("  ".join(parts), flush=True)

    index = pd.DataFrame(index_rows, columns=[
        "id", "campus", "track", "size_class", "split", "n_wos",
        "window_start", "window_bh", "path"])
    index.to_csv(INST_ROOT / "index.csv", index=False)

    # ---- summaries --------------------------------------------------------- #
    print("\n=== per-cell accepted instance counts (candidates in parens) ===")
    hdr = f"{'campus':>6} {'size':>5} {'split':>6} {'accepted':>9} {'cands':>7}"
    print(hdr)
    for campus in instances.CAMPUS_SET:
        for size in instances.SIZES:
            for split in ("train", "test"):
                cell = (campus, size, split)
                print(f"{campus:6d} {size:5d} {split:>6} "
                      f"{cell_counts.get(cell, 0):9d} {cell_candidates.get(cell, 0):7d}")
    print(f"total instances written: {len(index)}")

    print("\n=== capacity summary (campus x n_trades, total crew) ===")
    print(f"{'campus':>6} {'n_trades':>9} {'total_crew':>11} trades(crew)")
    for campus in instances.CAMPUS_SET:
        sub = capacity[capacity["campus"] == campus].sort_values("trade")
        pretty = ", ".join(f"{r.trade}:{int(r.crew)}" for r in sub.itertuples())
        print(f"{campus:6d} {len(sub):9d} {int(sub['crew'].sum()):11d}  {pretty}")

    print(f"\ndone in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
