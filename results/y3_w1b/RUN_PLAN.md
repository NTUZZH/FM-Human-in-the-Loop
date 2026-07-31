# W1b run plan: the regime map under the deployable review policy

Written before the long run was launched. Four questions per run: purpose and
destination, expected result and what the opposite would mean, contamination
risks checked, data-accuracy checks performed. Then the configuration diff
against the published cell each result replaces.

## What is being closed, and why it needs runs at all

W1 replaced the manuscript's review-routing policy. The published policy
(`Supervisor` with `mechanism="targeted"`) decides which dispatch decisions the
supervisor reviews partly through `_has_plus2`, which reads the realized latent
shift of the pending queue, so no real site can run it. The replacement routes on
a decision-stability test under a conformal band calibrated only on
override-derived weak labels, and is computable from observables alone.

W1 re-measured the headline cell and the eight-cell contention grid under the
replacement. It did **not** re-run the E3 regime map, which is a headline figure
(Figure F3) and is still on the old policy. A manuscript cannot show a headline
table on one routing policy and a headline figure on another without saying so,
and the submission gate requires the oracle-informed policy to appear only as a
clearly labelled upper reference. This work package produces the map under the
deployable policy so the figure and the table describe the same protocol.

## The grid, read rather than assumed

The published map's grid is parsed at run time from
`results/y3_p4/e3_map_summary.json` by `published_grid()` and asserted against
the task list this run builds, so the reproduction cannot silently drift:

| field | value | where it was read |
|---|---|---|
| campuses | 9 and 10 | cell keys of `e3_map_summary.json` |
| utilisation levels | u = 70, 90, 100, 110, 130 | cell keys |
| recoverable shares | beta = 0.00, 0.50, 1.00 | cell keys |
| review budget | rho = 0.25 | `config.rho` |
| seeds | 3 (301-303) | `cells[*].n_seeds`, unique across all 30 cells |
| cells | 30 | `len(cells)` |
| channel | `full_class_shift` | `config.channel` |
| held-out instances | 10, on **both** campuses | counted in `results/y3_p4/e3_map.csv` |

The last row is a correction to the obvious assumption. `y3_p4_m0grid.tasks_B`
takes an `--n-eval-c10` argument that defaults to 8, but the committed
`e3_map.csv` carries ten rows per C10 cell-seed and every committed C10 cache
record has `n_eval = 10`, so the published map was run at `--n-eval-c10 10`.
This run uses ten on both campuses to match.

Everything else is the committed configuration of `scripts/y3_p4_m0grid.py`,
field for field: overlay channel `full_class_shift`, family F-NL, master seed
12345, epsilon 0, theta 1.0, instance split `files[0:16]` train / `[16:20]`
probe / `[20:30]` eval, 8 DAgger iterations, `ShiftEstimator(hidden=32)`,
scoring TWT*(w*, d*) by the independent validator.

## Run 1 -- reproduction gate (headline cell, ten seeds, published protocol)

* **Purpose.** Prove this code path reproduces a published number before any
  conclusion is drawn from it. Destination: the reproduction sentence of the W1b
  report; nothing in the manuscript quotes it directly.
* **Expected result.** Per-instance TWT* identical, to the bit, to the committed
  `results/y3_p4/cache` records for all five deciders on all ten seeds, and hence
  a ten-seed reduction of the correction layer over the tuned rule of 45.3620%,
  which is the published `\MzeroGain` = 45.4%. **If it differs at all**, the
  wrapper has drifted and every later number in this package is void; stop.
* **Contamination checked.** The result cache is a fresh directory
  (`results/y3_w1b/cache`), so no record of W1's or of the published grid can be
  read into this run; records are written to a temp file and `os.replace`d, so a
  partial file cannot be swept into an analysis; no checkpoint is resumed or
  overwritten, since the estimator is fitted from scratch inside each cell; the
  CSV is truncated before a part runs, so a re-run cannot double-append.
* **Data accuracy checked.** The ten held-out instance ids are compared, id for
  id, against the ids in the committed record for the same seed.
* **Config diff against the comparator.** The comparator is the committed
  `y3_p4` headline record. Differences: none. `policy="targeted"` with
  `split_fit=False` makes `run_m0_routed` issue the same calls in the same order
  as `augmented_rule.run_m0`.

## Run 2 -- the deployable regime map (30 cells, 3 seeds, C9 and C10)

* **Purpose.** The manuscript's regime map, re-measured under a policy a site can
  actually run. Destination: Figure F3 and the macros it feeds
  (`\GainSlack`, `\GainBusy`, `\GainSat`, `\GainBetaZero`, `\BetaGainLow`,
  `\BetaGainHigh`, `\GainPeakCnine`, `\RegretMax`, `\BetaZeroMapOverloadCnine`,
  `\BetaZeroMapOverloadCten`, `\BetaZeroBandCnine`, `\BetaZeroBandCten`,
  `\BetaZeroBandCtenSup`), plus the regime-map paragraph of the results section.
* **Contrasts reported.** Both, because the manuscript now names both: the
  correction layer with the supervisor against the tuned rule with the same
  supervisor (`m0sup_over_rulesup_pct`, which is what the figure colours), and
  the layer alone against the rule alone (`m0_over_rule_pct`).
* **Expected result.** W1's evidence is that the two policies perform
  indistinguishably (mean price of deployability -0.4 percentage points over the
  eight-cell grid, no cell surviving Holm correction), so the map should keep its
  three qualitative claims: the reduction grows with load, grows with the
  recoverable share, and collapses where either is absent. **If instead a cell
  moves materially**, the map's reading changes and the figure must be redrawn
  around the deployable numbers with the shift stated; that is a publishable
  result either way. **If the map collapsed at high beta and high load**, the
  published figure would have depended on oracle-informed review placement, which
  would force the headline claim to be restated around the labelled upper
  reference. Either outcome changes what the paper says, so the run is worth its
  cost.
* **The outcome that would result from a mistake, and how it is excluded.** The
  expected result -- a map much like the published one -- is also exactly what
  running the OLD policy by accident would produce. Three checks make that
  falsifiable rather than assumed, and they are recorded per cell:
  1. `routing.make_supervisor` is wrapped so every supervisor built anywhere in
     the run has its class asserted against the policy the task asked for, in the
     DAgger training loop and in the held-out evaluation alike;
  2. `Supervisor._has_plus2`, the undeployable clause itself, is wrapped with a
     call counter that must read **zero** on every deployable cell;
  3. the record carries the undetermined rate of the stability test and the
     calibrated band, neither of which the old policy produces at all.
  A smoke run on `c9 u70 beta 0` returned 148 constructions, all
  `StabilityRoutingSupervisor`, and 0 calls to `_has_plus2`.
* **Contamination checked.** As run 1. In addition, the cache signature includes
  `policy`, `split_fit`, `cal_frac`, `alpha` and `band_mode`, so no two arms can
  share a record; and the deployable map, the split-protocol control and the gate
  write to three separate CSVs.
* **Data accuracy checked.** Thirty instance files exist at every
  (campus, utilisation) of the grid, and `n_train + n_probe + n_eval = 30` uses
  all of them. Because the tuned rule and the omniscient reference do not consult
  the supervisor at all, their per-instance TWT* must be bit-identical to the
  published record for the same cell-seed; the analysis asserts this on every
  cell, which proves same instances, same overlay draw, same scoring and same
  seed handling rather than assuming them. The smoke run confirmed it on
  `c9 u70 beta 0`.
* **Config diff against the comparator.** Against the committed `y3_p4` map task
  for the same cell: `mech="targeted"` becomes `policy="stability"`, and
  `split_fit` becomes True, which brings `cal_frac=0.3`, `alpha=0.1` and
  `band_mode="global"` into existence. These two are not separable and are not
  being presented as one: the stability test needs a band whose residuals are
  out-of-sample, and the fold split is what makes them so. It costs the
  deployable arm about 30% of its training labels, which run 3 measures. Every
  other field is identical, and the analysis diffs the two resolved task
  dictionaries key by key rather than trusting this paragraph.

## Run 3 -- split-protocol control (campus 9 only, 15 cells, 3 seeds)

* **Purpose.** Separate the two coupled differences of run 2. This arm holds the
  review policy at the published one and turns only the conformal fold split on,
  so a per-cell difference between the deployable map and the published map can
  be attributed to review placement or to the 30% of training labels the split
  costs. Destination: one sentence of the W1b report and, if the difference is
  material, a sentence of the manuscript's method section.
* **Expected result.** Small, unsigned differences from the published map,
  consistent with W1's finding that the published gain does not depend on review
  placement at this budget. **If the split alone moved the map**, the deployable
  map's differences would be a data-volume effect rather than a policy effect and
  would have to be reported as such.
* **Scope, stated rather than implied.** Campus 9 only. A C10 cell-seed costs
  roughly fifteen times a C9 one, and this arm is a diagnostic, not a headline;
  spending ten core-hours on it would displace the map itself. C10 therefore
  carries the deployable map and the published map only.
* **Contamination and data accuracy.** As run 2.
* **Config diff.** Against run 2's map task for the same cell: `policy` only.

## Cost, and what would be cut first if it did not fit

Estimated single-core cost, from the committed `y3_p4` cache times inflated by
the ~1.2 the stability test adds: 0.7 core-hours for the 45 C9 map cell-seeds,
10.5 for the 45 C10 ones, 0.6 for the control and 0.1 for the gate, so about 12
core-hours in total. That fits inside the ten pinned cores, so **the full map
runs and nothing is cut**. Had it not fitted, the order of cuts would have been:
first drop to the published map's own three seeds (already the case), then run C9
in full and C10 at the realistic-load band only, listing every unrun cell in the
report and leaving it visibly on the old policy's number.

## Compute discipline

One numeric thread per process. This pipeline reproduces bit-exactly at one
thread only; more changes the floating-point reduction order and moves the
headline by over a percentage point. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` are set to 1 at the top of the
runner, before numpy and torch are imported and before `y3_w1_sweep` is imported
(whose own module-level `setdefault` would otherwise install four), and
`torch.set_num_threads(1)` is called in the parent and again inside every worker.
Parallelism comes from separate processes: nine workers, one thread each, under
`taskset -c 0-9`. Cores 10 to 23 belong to other agents, and the machine also
carries an unpinned training job, so **no wall-clock figure from this run is
reported as a measurement of anything**; the cost table above exists only to
order the task queue longest-first so the pool packs.

## After the runs

`scripts/y3_e0_anchor.py` is re-run on the final code: with the supervisor
disabled and the deadline-aware gate closed, the environment must still reduce
bit-for-bit to the public benchmark's dispatcher. W1b adds two files and edits
none, so the anchor is expected to pass unchanged; it is re-run because the plan
requires it after every code change, not because a break is suspected.

---

# What actually happened

Appended after the runs, against the expectations stated above.

**Run 1, reproduction gate: PASS, exactly as expected.** Per-instance TWT* is
identical to the bit to the committed `results/y3_p4/cache` records across all
fifty comparisons (five deciders on ten seeds, max absolute difference
0.000e+00), the held-out instance ids match seed for seed, and the ten-seed
reduction of the correction layer over the tuned rule reproduces at 45.36203547%
against the published 45.36203547%, a difference of exactly zero. `\MzeroGain`
rounds to 45.4% from both. Log: `gate_run.log`; record: `gate_check.json`.

**Run 2, the deployable regime map: complete, all 30 cells, nothing cut.** Every
cell of the published grid was re-run under the deployable policy at the
published three seeds and ten held-out instances, so no cell remains on the old
policy. The three proofs of which policy ran are unambiguous
(`policy_proof.json`): 13,320 supervisors were constructed across the 90
cell-seeds, every one of them a `StabilityRoutingSupervisor`, and
`Supervisor._has_plus2`, the clause that reads the realized latent shift, was
called **zero** times. The counter is not dead code: the same instrumentation
recorded 2,472,929 calls on the gate and 10,382,078 on the control, both of
which run the published policy. All 90 records carry a calibrated band and an
undetermined rate, neither of which the published policy produces.

The data-accuracy check passed on all 90 cell-seeds: the tuned rule and the
omniscient reference, which never consult the supervisor, are bit-identical to
the committed record for the same cell-seed, which establishes same instances,
same overlay draw, same scoring and same seed handling rather than assuming
them. The figure's pooled-mean formula and the summary's mean-of-seed-means
formula agree to 4e-14 percentage points, so the two ways of reading the map are
the same number.

The map's qualitative conclusions are unchanged, which was the expected outcome.
The reduction still grows with load at every recoverable share above zero and on
both campuses, still grows with the recoverable share at every one of the ten
(campus, utilisation) pairs, and is still inert at slack capacity on both
campuses. The two places where the published map already broke monotonicity, the
$\beta=0$ rows, are the two places the deployable map breaks it as well, so the
agreement is exact rather than approximate.

The magnitudes did move, and more than W1's headline evidence suggested they
would. Averaged over all 30 cells the plotted contrast falls by 3.43 percentage
points (mean absolute movement 4.62, largest 21.06 at `c10_u100_b0.50`), while
the layer-alone contrast falls by only 0.68 (mean absolute 2.26). Restricted to
campus 9 inside the realistic-load band, which is the region every claim in the
results section rests on, the plotted contrast moves by 0.06 points on average
(mean absolute 0.90, largest 2.14) and the alone contrast by 0.27 (mean absolute
1.47). The headline map cell moves in the deployable policy's favour: 30.80% to
32.41% on the plotted contrast and 44.27% to 48.29% on the alone contrast, the
latter agreeing with W1's independently measured ten-seed 48.15%.

The reason the plotted contrast moves more than the alone contrast is
structural, not an artefact: the review policy appears in both the test arm and
the comparator of `m0sup_over_rulesup_pct`, so changing it moves the denominator
too. It does: the deployable policy leaves RULE+SUP better on 17 of the 30 cells,
by up to 30% of true weighted tardiness on the large campus, because routing
review to genuinely close calls helps a fixed rule more than the oracle clause
did. Where the comparator improves faster than the test arm, the ratio compresses
even though both arms are better in absolute terms. The layer-alone contrast has
no supervisor on either side and is correspondingly stable.

**Run 3, split-protocol control: as expected, the fold split is not the cause.**
Over the 15 campus-9 cells the conformal fold split alone, with the published
policy held fixed, moves the plotted contrast by +0.41 percentage points on
average (+0.20 inside the band) and the alone contrast by +0.53 (+0.76 inside the
band). The remainder is the routing policy: -2.62 points over the 15 cells and
-0.26 inside the band on the plotted contrast, -1.58 and -0.49 on the alone
contrast. So the map's movement is a policy effect, not the 30% of training
labels the split costs, and inside the operating band the policy effect is a
fraction of a point. Campus 10 carries no control, as planned.

**Anchor.** `scripts/y3_e0_anchor.py` re-run on the final code: E0 ANCHOR PASS
(48 campus x size x rule triples exact, policy replay deterministic, M1 gate=0
bit-exact, shaped-reward telescoping OK). Log:
`results/y3_w1b/e0_anchor_after_w1b.log`. `tests/test_routing.py` was also re-run
and passes in full, including its bit-exactness check that
`run_m0_routed(policy='targeted', split_fit=False)` reproduces
`augmented_rule.run_m0`.

**Compute, stated rather than implied.** Every job ran pinned to cores 0-9 with
nine single-threaded workers, alongside two other agents' unpinned jobs; the
observed load average was between 40 and 50 on 24 cores and the workers held
about 85% of a core each. The map run was interrupted once by the harness at
15 of 90 cells and relaunched detached; because every cell-seed is cached
atomically by its configuration signature, the completed cells replayed from
cache and no cell was computed twice or half-written. No wall-clock figure from
any of these runs is reported as a measurement of anything.

**Nothing failed and nothing was left uncomputed.** Every cell of the published
grid was re-run; `map_summary.json:coverage.cells_not_run` is empty.
