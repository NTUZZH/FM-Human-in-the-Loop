# P9 run plan: corpus-anchored recoverable-share cells

Written before the run. Driver: `scripts/y3_realistic_cell.py`. Outputs: this
directory only. Nothing under `paper/`, `src/` or the existing `scripts/` is
edited, and nothing is committed.

## Why this run exists

The manuscript's headline reduction is measured at a recoverable share of
`beta = 1.00`, where the supervisor's hidden urgency is by construction a
deterministic function of the observable order features the estimator receives.
The companion corpus analysis (`results/y3_calib/`, from
`scripts/y3_calib_overrides.py`) estimates the real-data analogue of that share
at `beta_hat = 0.1056` (campus 1), `0.2038` (campus 9) and `0.2727` (campus 10),
median `0.2038`, and states it as a lower bound. The published regime map
(`results/y3_p4/e3_map_summary.json`) has rows only at `beta` in
`{0.00, 0.50, 1.00}`, so no measured cell sits anywhere near the value the corpus
supports. A referee will read the headline as an artefact of the most favourable
setting in the design unless the paper can point at a measured cell at the
corpus-anchored share.

Campus 9 is the campus the headline cell uses, and its own corpus estimate is
`0.2038`, so `beta = 0.20` is not a generic low value: it is C9's own number.

## The four pre-launch questions

### 1. Purpose: what question does each run answer, and where does the number land

| Run | Question | Destination |
|---|---|---|
| R0 reproduction gate | Does this script's code path reproduce the published `\MzeroGain = 45.4%` at the headline cell (`c9`, storm2, `u=100`, `beta=1.00`, `rho=0.25`, seeds 301-310)? | Gates everything below. Reported as the provenance sentence for the new table, in the style of `results/y3_p8/repro_check.json`. |
| A: `beta=0.20`, `eps=0` | What reduction does the correction layer deliver at the corpus-anchored recoverable share, alone and in the loop? | A new row in the regime-map table, and the sentence that locates the headline against real data. Macros `\RealBeta*`. |
| B: `beta=0.20`, `eps=0.25` | Does that reduction survive a supervisor who makes mistakes on a quarter of its reviews (half missed overrides, half random overrides)? | Second row of the same table; the realistic-operating-point claim and the limitations paragraph. Macros `\RealBetaEps*`. |
| C: `beta=0.25`, `eps=0` | Is the value at A a point or a small range? | Third row; lets the paper write "over the corpus-supported range 0.20 to 0.25" instead of quoting one number. Macros `\RealBetaHi*`. |

Every cell is at campus 9, storm2 `w80`, `u = 100` (saturation), `rho = 0.25`,
targeted review, `theta = 1.0`, channel `full_class_shift`, family `F-NL`,
`master_seed = 12345`, evaluated on `files[20:30]` of the c9/storm2/w80/u100 pool:
the same ten held-out instances the paper uses.

Deciders scored: RULE (tuned rule), RULE+SUP (rule with the supervisor),
M0 (correction layer alone), M0+SUP (correction layer with the supervisor),
ORACLE (myopic full-information reference). The end-to-end learner M1 needs
training and is out of scope; nothing is trained beyond the per-cell M0 shift
estimator, which is the correction layer itself.

Seeds: 301-310 (ten), declared primary here, before the run. The five-seed subset
301-305 is reported alongside as a consistency check only, not as an alternative
headline. Declaring both up front is what stops the seed count from becoming a
choice made after seeing the numbers.

### 2. Ideal result, and what the opposite would mean

The expectation, written down before the run: **the gain at `beta = 0.20` will be
far below 45.4%, and it will sit between the two published neighbours on this
campus, which are `-1.1%` at `beta = 0.00` and `+26.4%` at `beta = 0.50` for the
layer alone (3 seeds, `results/y3_p4/e3_map_summary.json`).** Straight-line
interpolation between those two puts the layer alone near 10%; anywhere from
slightly negative to the mid-teens is consistent with what is already known.

A small number here is the successful outcome of this run, not a failure. It is
the number that lets the paper state its headline honestly and locate it against
real data. Concretely:

* **If the layer alone lands in the single digits or low teens** (the expected
  case): the paper reports it, keeps the `beta = 1.00` cell as the upper end of a
  measured range rather than as *the* result, and leads on the in-loop numbers and
  the share of the rule-to-reference gap closed.
* **If it lands at or below zero**: the honest conclusion is that at the
  corpus-supported recoverable share the correction layer alone does not beat the
  tuned rule on this campus, and the paper must say exactly that, moving its claim
  to the in-loop configuration (M0+SUP versus RULE+SUP) if that survives, and
  stating the scope limit plainly if it does not.
* **If it lands at or above the `beta = 0.50` value**: that contradicts the
  published regime map's monotonicity in `beta` and I would not report it as a
  result. I would first re-read the configuration diff for an unintended
  difference, and report the discrepancy rather than the number.

Both branches lead to different sentences in the manuscript, so the run is worth
making. No tuning, no seed selection, no search for a more flattering
configuration: the three cells listed above are the whole run, and every one of
them is reported.

### 3. Contamination risks, and how each is closed

1. **Stale or shared cache.** `scripts/y3_p4_m0grid.evaluate_cell` caches by a
   SHA-1 over the resolved task, into `results/y3_p4/cache/`. This run redirects
   that cache to `results/y3_p9/cache/`, set inside every worker before the call,
   so (a) the reproduction genuinely recomputes rather than reading the published
   record back, and (b) nothing is written into the published cache directory. The
   task signature already includes `beta` and `eps`, so even a shared cache could
   not have collided; the redirect makes it moot.
2. **Sweeping into an existing analysis.** No script in `scripts/` or `src/`
   references `results/y3_p9` (checked by grep), and the published harvesters
   (`y3_harvest_primary.py`, `y3_harvest_final.py`, `y3_practitioner_metrics.py`,
   `y3_case_table.py`, `y3_w1_sweep.py`) glob `results/y3_p4/cache` only. Writing
   into a fresh directory therefore cannot change any published number.
3. **Partially written result files.** Every JSON/CSV output is written to a
   temporary file and renamed into place. The aggregation asserts the expected
   record count per cell (10 seeds x 10 instances x 5 deciders) and refuses to
   summarise a short harvest.
4. **Split leakage.** `evaluate_cell` asserts the held-out set is disjoint from
   the train and probe pools. This run additionally asserts the ten held-out
   instance ids are exactly the published `eval_inst_ids` of
   `results/y3_p5/harvest/primary_multiseed_summary.json`.
5. **Floating-point drift from thread count.** The pipeline reproduces bit-exactly
   only with one numeric thread per process. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
   `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` are
   hard-set to `1` before numpy/torch import, `torch.set_num_threads(1)` runs in
   every worker, and each worker asserts `torch.get_num_threads() == 1` before it
   computes anything. Parallelism comes from four separate worker processes.
6. **Device contention.** The machine is shared. The run is pinned with
   `taskset -c 20-23` to at most four workers; cores 0-19 belong to other agents.
   No wall-clock timing is measured or reported, because under contention a timing
   is not a measurement.
7. **Silent configuration drift.** Every cell's resolved configuration is dumped
   and diffed field by field against the resolved configuration of the published
   headline cell. The run asserts the difference set is exactly `{beta}` for cells
   A and C and exactly `{beta, eps}` for cell B, and aborts otherwise.
8. **Cross-cell consistency.** RULE and ORACLE do not depend on `eps` (RULE runs
   with no supervisor; ORACLE uses the supervisor object only for its
   preferred-pick logic, with no review protocol). Cells A and B share `beta`, so
   their RULE and ORACLE per-instance values must be bit-identical. This is
   asserted, and it is a direct check that `eps` entered the run where it should
   and nowhere else.

### 4. Data accuracy

* The instance pool is `data/processed/instances/c09/storm2/w80/c09_storm2_w80_u100_*.json`,
  30 files, sorted; train `[0:16]`, probe `[16:20]`, held-out `[20:30]`. The count
  is asserted and the SHA-256 of each of the ten held-out files is recorded.
* The held-out ids are asserted equal to the published set
  (`c09_storm2_w80_u100_0020` ... `_0029`).
* The overlay coefficient file `results/y3_p1/overlay_coeffs/F-NL_seed12345.json`
  (SHA-256 `21b3ef67c107d2c5fbbbe6a1ee28354851c6928cec64aa66749cf2f1c1b9413b`) is
  shared by every cell; only `beta` changes what the overlay draws from it.
* The corpus anchor is read from `results/y3_calib/predictability.csv`, rows with
  `population = cm` and `variant = trade_w30` (the primary label declared in
  `results/y3_calib/summary.json`), campuses with `status = headline` in
  `results/y3_calib/campus_disposition.csv`, namely 1, 9 and 10. Values are
  re-read by the script and recorded, not retyped.
* Statistics follow the manuscript's convention exactly: seed-average each
  decider's per-instance true weighted tardiness, then a two-sided paired Wilcoxon
  signed-rank test (`zero_method='pratt'`) over the ten held-out instances, with
  win/tie/loss counted as the test being strictly lower.

## Deliverables this run produces

`results/y3_p9/`: `RUN_PLAN.md` (this file), `repro_check.json`,
`config_diff.json`, `data_checks.json`, `cells.csv`, `cell_summary.json`,
`results_table.md`, `macros_snippet.tex`, `run.log`.
