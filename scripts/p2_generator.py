"""P2 end-to-end: fit the [C] generator on train years and ship the default
generator test corpus.

Pipeline (deterministic):
  1. load + clean FMUCD (fmwos.io).
  2. for each campus in {1,2,5,9,10,12}: fit_params -> pretty-printed parameter
     pack results/p2_generator/params_c<campus>.json (a paper artifact).
  3. generate the DEFAULT generator test set: campus x size {50,150,400}, 100
     instances each, seeds 20000+i, default knobs, written under
     data/processed/instances/c<campus>/generator/<size>/ and APPENDED to
     data/processed/instances/index.csv (track='generator', split='test').
  4. print the per-campus fitted summary and a REALISM CHECK comparing generator
     instances against replay TEST instances (>30% relative deltas flagged).

Run: python scripts/p2_generator.py
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

from fmwos import generator, io          # noqa: E402

RAW = ROOT / "data" / "raw" / "FMUCD.csv"
PARAMS_OUT = ROOT / "results" / "p2_generator"
INST_ROOT = ROOT / "data" / "processed" / "instances"
INDEX_CSV = INST_ROOT / "index.csv"

CAMPUSES = [1, 2, 5, 9, 10, 12]
SIZES = [50, 150, 400]
N_PER_CELL = 100
SEED_BASE = 20000
FLAG_REL = 0.30   # flag any metric off by > 30% relative


# --------------------------------------------------------------------------- #
# Summaries                                                                    #
# --------------------------------------------------------------------------- #
def campus_priority_mix(params: dict) -> tuple[dict, float]:
    """Campus-level (expected) priority-class shares + overall pm_share."""
    counts = np.zeros(4, dtype="float64")
    n_total = 0.0
    pm_weighted = 0.0
    for tp in params["trades"].values():
        n = float(tp["n_train"])
        if n <= 0:
            continue
        n_total += n
        pm = float(tp["pm_share"])
        pm_weighted += n * pm
        # PM rows -> class 4; CM rows -> cm distribution
        counts[3] += n * pm
        cm = tp["cm_priority_dist"]
        for c in (1, 2, 3, 4):
            counts[c - 1] += n * (1.0 - pm) * float(cm[str(c)])
    mix = (counts / counts.sum()) if counts.sum() > 0 else counts
    pm_share = pm_weighted / n_total if n_total > 0 else 0.0
    return {c: float(mix[c - 1]) for c in (1, 2, 3, 4)}, float(pm_share)


def print_fit_summary(campus: int, params: dict) -> None:
    mix, pm_share = campus_priority_mix(params)
    print(f"\n=== campus {campus} fitted summary "
          f"(n_train={params['n_train_total']:,}, "
          f"labor_cap={params['labor_cap']:.1f}h, "
          f"span={params['train_span_bh']:.0f}bh) ===")
    print(f"  overall pm_share={pm_share:.3f}  "
          f"priority mix P1..P4="
          f"{mix[1]:.3f}/{mix[2]:.3f}/{mix[3]:.3f}/{mix[4]:.3f}")
    ranked = sorted(params["trades"].items(),
                    key=lambda kv: kv[1]["arrival_rate_per_bh"], reverse=True)
    print(f"  {'trade':>6} {'rate/bh':>8} {'pm_shr':>7} {'crew':>5} "
          f"{'CM P1/P2/P3/P4':>22}  {'logn(mu,sig)':>14}")
    for tr, tp in ranked[:5]:
        cm = tp["cm_priority_dist"]
        print(f"  {tr:>6} {tp['arrival_rate_per_bh']:8.4f} "
              f"{tp['pm_share']:7.3f} {params['capacity'].get(tr, 0):5d} "
              f"{cm['1']:.2f}/{cm['2']:.2f}/{cm['3']:.2f}/{cm['4']:.2f}"
              f"{'':>6}  ({tp['logn_mu']:5.2f},{tp['logn_sigma']:5.2f})")


# --------------------------------------------------------------------------- #
# Realism check                                                                #
# --------------------------------------------------------------------------- #
def instance_metrics(insts: list[dict]) -> dict | None:
    """Pool metrics over a list of instance dicts."""
    if not insts:
        return None
    jobs_per_bh = []
    util = []
    p_bh, pm, prio = [], [], []
    for inst in insts:
        wos = inst["work_orders"]
        n = len(wos)
        wbh = float(inst["meta"]["window_bh"]) or 1.0
        jobs_per_bh.append(n / wbh)
        tot_crew = max(1, len(inst["technicians"]))
        workload = sum(float(w["p_bh"]) for w in wos)
        util.append(workload / (tot_crew * 40.0))
        for w in wos:
            p_bh.append(float(w["p_bh"]))
            pm.append(1.0 if w["is_pm"] else 0.0)
            prio.append(int(w["priority"]))
    prio = np.array(prio)
    shares = {c: float((prio == c).mean()) if len(prio) else 0.0
              for c in (1, 2, 3, 4)}
    return {
        "n_inst": len(insts),
        "mean_jobs_per_bh": float(np.mean(jobs_per_bh)),
        "median_p_bh": float(np.median(p_bh)) if p_bh else 0.0,
        "pm_share": float(np.mean(pm)) if pm else 0.0,
        "prio": shares,
        "util_proxy": float(np.mean(util)),
    }


def load_replay_test(index: pd.DataFrame, campus: int, size: int) -> list[dict]:
    sub = index[(index["campus"] == campus) & (index["size_class"] == size)
                & (index["track"] == "replay") & (index["split"] == "test")]
    out = []
    for p in sub["path"]:
        fp = INST_ROOT / p
        if fp.exists():
            with open(fp) as f:
                out.append(json.load(f))
    return out


def _rel(gen: float, rep: float) -> float:
    if rep == 0.0:
        return 0.0 if gen == 0.0 else float("inf")
    return (gen - rep) / rep


def realism_row(campus, size, g, r):
    """Return (printable rows, flags) comparing generator g vs replay r."""
    if r is None or g is None:
        return [f"  c{campus:2d} s{size:<3d}  (no replay test instances)"], []
    lines, flags = [], []

    def fmt(label, gv, rv, is_share=False):
        rel = _rel(gv, rv)
        flag = " *FLAG*" if (abs(rel) > FLAG_REL) else ""
        if flag:
            flags.append((campus, size, label, gv, rv, rel))
        rels = "inf" if rel == float("inf") else f"{rel:+.1%}"
        return (f"  c{campus:2d} s{size:<3d} {label:<16} "
                f"gen={gv:8.4f}  replay={rv:8.4f}  rel={rels:>8}{flag}")

    lines.append(fmt("jobs_per_bh", g["mean_jobs_per_bh"], r["mean_jobs_per_bh"]))
    lines.append(fmt("median_p_bh", g["median_p_bh"], r["median_p_bh"]))
    lines.append(fmt("pm_share", g["pm_share"], r["pm_share"]))
    for c in (1, 2, 3, 4):
        lines.append(fmt(f"prio_P{c}_share", g["prio"][c], r["prio"][c]))
    lines.append(fmt("util_proxy", g["util_proxy"], r["util_proxy"]))
    return lines, flags


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    print("loading + cleaning FMUCD ...", flush=True)
    raw = io.load_raw(RAW)
    clean, _ = io.clean(raw)
    del raw
    print(f"  clean rows: {len(clean):,} ({time.time() - t0:.0f}s)", flush=True)

    PARAMS_OUT.mkdir(parents=True, exist_ok=True)

    # ---- fit + write parameter packs -------------------------------------- #
    packs: dict[int, dict] = {}
    for campus in CAMPUSES:
        params = generator.fit_params(clean, campus)
        packs[campus] = params
        with open(PARAMS_OUT / f"params_c{campus}.json", "w") as f:
            json.dump(params, f, indent=2, sort_keys=True)
        print_fit_summary(campus, params)
    print(f"\nparameter packs written to {PARAMS_OUT}")

    # ---- generate default test corpus ------------------------------------- #
    index_rows: list[dict] = []
    print("\n=== generating default generator test corpus ===", flush=True)
    for campus in CAMPUSES:
        params = packs[campus]
        for size in SIZES:
            out_dir = INST_ROOT / f"c{campus:02d}" / "generator" / str(size)
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(N_PER_CELL):
                seed = SEED_BASE + i
                inst = generator.generate(params, size=size, seed=seed)
                inst_id = f"c{campus:02d}_gen_{size}_{i:04d}"
                inst["meta"]["id"] = inst_id
                rel_path = f"c{campus:02d}/generator/{size}/{inst_id}.json"
                with open(INST_ROOT / rel_path, "w") as f:
                    json.dump(inst, f, separators=(",", ":"))
                index_rows.append({
                    "id": inst_id,
                    "campus": campus,
                    "track": "generator",
                    "size_class": size,
                    "split": "test",
                    "n_wos": len(inst["work_orders"]),
                    "window_start": "synthetic",
                    "window_bh": inst["meta"]["window_bh"],
                    "path": rel_path,
                })
            print(f"  campus {campus:2d} size {size:3d}: {N_PER_CELL} instances",
                  flush=True)

    # ---- append to index.csv (idempotent: drop prior generator rows) ------ #
    cols = ["id", "campus", "track", "size_class", "split", "n_wos",
            "window_start", "window_bh", "path"]
    existing = pd.read_csv(INDEX_CSV)
    existing = existing[existing["track"] != "generator"]
    new = pd.DataFrame(index_rows, columns=cols)
    combined = pd.concat([existing, new], ignore_index=True)
    combined.to_csv(INDEX_CSV, index=False)
    print(f"appended {len(new)} generator rows to {INDEX_CSV} "
          f"(total {len(combined)})")

    # ---- realism check ---------------------------------------------------- #
    index = pd.read_csv(INDEX_CSV)
    gen_by_cell: dict[tuple[int, int], list[dict]] = {}
    for row in index_rows:
        gen_by_cell.setdefault((row["campus"], row["size_class"]), []).append(
            json.load(open(INST_ROOT / row["path"])))

    print("\n=== REALISM CHECK: generator vs replay TEST "
          f"(flag if |rel delta| > {FLAG_REL:.0%}) ===")
    all_flags = []
    for campus in CAMPUSES:
        for size in SIZES:
            g = instance_metrics(gen_by_cell.get((campus, size), []))
            r = instance_metrics(load_replay_test(index, campus, size))
            lines, flags = realism_row(campus, size, g, r)
            for ln in lines:
                print(ln)
            all_flags.extend(flags)
            print()

    print(f"=== flagged cells ({len(all_flags)}) ===")
    if not all_flags:
        print("  none")
    for campus, size, label, gv, rv, rel in all_flags:
        rels = "inf" if rel == float("inf") else f"{rel:+.1%}"
        print(f"  c{campus:2d} s{size:<3d} {label:<16} "
              f"gen={gv:.4f} replay={rv:.4f} rel={rels}")

    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
