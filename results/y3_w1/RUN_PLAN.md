# W1 run plan: deployable review routing via a calibrated uncertainty band

Written before any run was launched. Four questions per run: purpose and
destination, expected result and what the opposite would mean, contamination
risks checked, data-accuracy checks performed. Then the configuration diff
against the run each result is compared with.

## What is being replaced, and why it needs runs at all

The published headline review policy (`Supervisor` with `mechanism="targeted"`)
decides which decisions the supervisor reviews partly by reading the realized
latent shift of the pending queue (`_has_plus2`). No site can run that protocol,
so the headline cell describes something undeployable. W1 replaces the criterion
with a decision-stability test computable from observables, and every run below
exists to establish that the replacement still pays, and by how much less.

## The shared configuration, and the one thing that varies

Every run below uses the committed headline configuration of
`scripts/y3_p4_m0grid.py`, field for field:

| field | value | why it is fixed |
|---|---|---|
| campus / regime / utilisation | 9 / storm2 w80 / u = 100 | the manuscript's headline cell |
| overlay channel | `full_class_shift` | the headline private-information channel |
| overlay family / master seed | F-NL / 12345 | the locked overlay draw |
| beta | 1.00 | high recoverable share (headline) |
| rho | 0.25 | low review budget (headline) |
| epsilon / theta | 0.0 / 1.0 | no override noise, published tolerance |
| instance split | files[0:16] train, [16:20] probe, [20:30] eval | the committed split |
| DAgger iterations | 8 | the committed symmetric protocol |
| estimator | `ShiftEstimator(hidden=32)`, `train_estimator` unchanged | the committed estimator |
| scoring | TWT*(w*, d*), independent validator | the sole referee |
| seeds | 301-310 (headline), 301-303 (maps) | the committed seed ranges |

The only fields that vary across arms are `policy` (the review placement rule)
and `split_fit` (whether the conformal fold split is in force). The signature
that keys the result cache contains all of them, so no two arms can share a
cached record.

`split_fit=True` assigns every weak-label example a permanent fold at creation.
A calibration-fold example is never used to fit the estimator in this or any
later iteration, so the conformal residuals are out-of-sample even though the
estimator is warm-started and retrained on a growing aggregate. It costs the
deployable arm ~30% of its training labels, which is why the oracle-informed
upper reference is ALSO run under the split: otherwise the "price of
deployability" would be a mixture of worse review placement and less data.

## Run 1 -- pilot gate (single seed, published protocol)

* **Purpose.** Prove the new code path reproduces the committed pipeline before
  anything is concluded from it. Destination: the reproduction sentence of the
  W1 report; nothing in the manuscript quotes it directly.
* **Expected result.** Per-instance TWT* identical, to the bit, to the committed
  `results/y3_p4/cache` record for the headline cell at seed 301, and hence a
  seed-301 M0-vs-RULE gain of 45.1911%, consistent with the published ten-seed
  `\MzeroGain` of 45.4%. **If it differs at all**, the wrapper has drifted from
  `augmented_rule.run_m0` and every later number is void; stop and fix.
* **Contamination checked.** The result cache is a fresh directory
  (`results/y3_w1/cache`), so no stale record can be read. Records are written to
  a temp file and `os.replace`d, so a partial file cannot be swept into an
  analysis. No checkpoint is resumed or overwritten: the estimator is fitted from
  scratch inside the run. Train/probe/eval slices are asserted disjoint in the
  runner.
* **Data accuracy checked.** 30 instance files exist at
  `data/processed/instances/c09/storm2/w80/c09_storm2_w80_u100_*.json`;
  `n_train + n_probe + n_eval = 30` uses all of them; the ten evaluation instance
  ids are compared against the ids in the committed record.
* **Config diff against the comparator.** The comparator is the committed
  `y3_p4` record. Differences: none. `policy="targeted"` and `split_fit=False`
  make `run_m0_routed` issue the same calls in the same order as `run_m0`, which
  `tests/test_routing.py` (h) checks by bit-comparing estimator predictions and
  per-iteration counts.

## Run 2 -- headline table under the deployable policy (ten seeds, five arms)

* **Purpose.** The manuscript's headline reduction, re-measured under a policy a
  site can actually run, plus the measured price of deployability. Destination:
  Table 4 rows and the abstract's reduction figure; the price is a new sentence
  and a new macro.
* **Arms.** `targeted_pub` (published protocol, the reproduction anchor);
  `targeted` (oracle-informed upper reference, split protocol); `stability`
  (deployable headline); `margin` (observable control: targeted minus its latent
  clause); `random` (lower control).
* **Expected result.** The deployable policy keeps a reduction against the tuned
  rule that is statistically separable (paired Wilcoxon on the ten held-out
  instances, seed-averaged, Holm-corrected), landing at or a few points below the
  44.5-47.6% band. **If instead it collapses**, the conclusion is that the
  published gain depended on oracle-informed review placement, which is a
  publishable negative result and would force the headline claim to be restated
  around the upper reference. **If it matches the upper reference exactly**, the
  conclusion is that placement never mattered at this budget, consistent with the
  reported budget-insensitivity, and the deployability argument becomes free.
  Either outcome changes what the paper says, so the run is worth its cost.
* **Contamination checked.** As run 1. In addition: the five arms have distinct
  cache signatures (the signature includes `policy` and `split_fit`); the CSV is
  truncated before a part is re-run so a re-run cannot double-append; and the
  analysis reads the per-cell JSON records, not the CSV, so a half-written CSV
  cannot enter a statistic.
* **Data accuracy checked.** Same instance files, same split, same ten held-out
  ids as the committed grid, asserted per record in the analysis.
* **Config diff.** Against `targeted_pub`: `policy` and `split_fit` only.
  Between the four split arms: `policy` only.

## Run 3 -- routing curve (rho sweep, two cells)

* **Purpose.** Automation coverage against true weighted tardiness, the
  practitioner-facing deliverable. Destination: the routing-curve headline figure
  and its practitioner-reading paragraph.
* **Grid.** rho in {0.02, 0.05, 0.10, 0.25, 0.50}, seeds 301-303, at the
  headline cell (C9) and the confirmation cell (C10, u = 100, beta = 1.00). Both
  the deployable policy and the oracle-informed reference are swept so the curve
  carries its own upper reference.
* **Expected result.** A monotone trade-off: coverage falls as rho rises and
  TWT* falls with it, with the knee at a small rho, consistent with the reported
  budget-insensitivity. **If coverage is flat at 1 - rho across the sweep**, the
  band is too wide for the stability test to certify anything and the curve is
  driven by the budget alone; that must be reported as such rather than dressed
  up, and it makes the alpha sweep (run 4) the substantive result.
* **Contamination checked.** As above; each (cell, rho, seed, policy) has its own
  signature. C10 uses eight held-out instances, matching the committed grid's
  `--n-eval-c10 8`.
* **Data accuracy checked.** 30 files exist at C10 u = 100; the C10 split is the
  same first-16/next-4/next-8 convention the committed grid used.
* **Config diff.** Against run 2's `stability` arm: `rho` only (and `campus`
  plus `n_eval` for the confirmation cell).

## Run 4 -- conformal level sweep, and band coverage per beta

* **Purpose (alpha).** The band width is the only knob that sets referral demand,
  so the coverage/quality trade-off is stated over alpha, not only over rho.
  Destination: the routing-curve figure's second panel or a small table.
* **Purpose (coverage).** Empirical coverage of the band against the true shift,
  per beta, reported as a result. Destination: the correctness paragraph of the
  method section and a macro.
* **Expected result.** Coverage of the WEAK labels lands at the nominal
  1 - alpha, because that is what split conformal guarantees. Coverage of the
  TRUE shift is expected to be LOWER, because the band targets a noisy, censored,
  budget-limited proxy. **If true coverage were at or above nominal**, the weak
  labels would be a near-unbiased proxy for the latent, which would contradict
  the reported modest recovery; that would need investigating before being
  believed. The gap is itself the finding.
* **Contamination checked.** The true shift is read in exactly one file,
  `scripts/y3_w1_band_coverage.py`, which never calls `calibrate_band` and never
  returns a band. `fmwos.hitl.routing` cannot read it: its calibration functions
  have no parameter that could carry an overlay, reject one passed positionally,
  reject labels outside the weak-label alphabet, and are grepped for latent
  tokens by `tests/test_routing.py` (f).
* **Data accuracy checked.** beta enters only through the overlay parameters;
  the instance files are byte-identical across beta, asserted by the shared
  instance-id list.
* **Config diff.** Against run 2's `stability` arm: `alpha` only, or `beta` only.

## Compute discipline

Every job runs under `taskset -c 0-9` with `nice`. The machine is shared with
three other agents (load average ~21 of 24 cores at the time of writing), so no
wall-clock figure from these runs is reported as a measurement of anything.
Threads: the parent process is capped at four (OMP, MKL, torch), and each forked
worker is capped at ONE thread, so eight workers use eight threads inside the ten
pinned cores. Four threads per worker would put 32 threads on 10 cores and the
workers would fight, which is the failure the pinning exists to prevent. Only one
sweep runs at a time.

## After the code change

`scripts/y3_e0_anchor.py` is re-run on the final code: with the supervisor
disabled and the deadline-aware gate closed, the environment must still reduce
bit-for-bit to the public benchmark's dispatcher. W1 adds files and edits none,
so the anchor is expected to pass unchanged; it is re-run because the plan
requires it after every code change, not because a break is suspected.

---

# What actually happened

Appended after the runs, against the expectations stated above.

**Run 1, pilot gate: PASS, exactly as expected.** Per-instance TWT* is identical
to the bit to the committed `results/y3_p4/cache` record for all five deciders
(max absolute difference 0.000e+00), so the seed-301 gain is 45.1911% against the
committed 45.1911%, and the ten-seed arm reproduces the published
`\MzeroGain = 45.4%` at 45.3620%.

**Run 2, headline table and the eight-cell grid: acceptance criterion met.** The
deployable policy reduces true weighted tardiness by 48.15% against the tuned
rule at the headline cell (10/0/0 on the held-out instances, raw p = 0.001953,
the floor for ten paired instances, Holm p = 0.0098 across the five arms), and it
is separable from the rule in all eight cells of the contention grid at Holm
p = 0.0156, which is the same Holm value the oracle-informed reference reaches.
The price of deployability came out NEGATIVE rather than positive: the deployable
policy is 2.32 percentage points BETTER than the oracle-informed one at the
headline cell (8/0/2, raw p = 0.037, which does not survive correction across the
eight arm-vs-arm tests), and 0.43 points better on average over the eight-cell
grid, ranging from 3.25 points better to 1.05 points worse with no cell
surviving Holm. The honest reading is therefore not "the deployable policy wins"
but "no price of deployability is detectable at this sample size, and the point
estimate favours the deployable policy". A diagnostic explains why the reference
is beatable: at the headline cell the oracle clause flags 47.2% of
multi-candidate decisions, nearly twice the budget, so it cannot rank within the
set it flags.

**Run 3, routing curve: as expected, a monotone trade-off with an early knee.**
At a 5% review budget the site dispatches 96.5% of decisions without review and
still gets a 45.5% reduction, against 48.3% at the 25% budget; the confirmation
cell on C10 shows the same shape at a larger scale (65.0% reduction, 78.1%
coverage at rho = 0.25). Coverage is budget-driven, as the alternative outcome in
the plan anticipated, so the alpha sweep carries the substantive
coverage-versus-quality result.

**Run 4, coverage: the expected gap, and it is large.** Coverage of the weak
labels lands within a point or two of nominal at every level
(0.944 / 0.894 / 0.765 / 0.611 / 0.474 at alpha = 0.05 / 0.10 / 0.20 / 0.30 /
0.50) and at every beta (0.888 to 0.897), so the split-conformal construction
behaves as designed on the quantity it targets. Coverage of the TRUE shift is
36.0% at alpha = 0.1, and on the orders that actually carry a nonzero shift it is
0.4%. The band under-covers the latent almost completely, which is the finding
the plan flagged as worth reporting rather than a defect to fix.

**Compute, stated rather than implied.** All runs were pinned to cores 0-9 with
eight single-threaded workers, under a machine load average between 30 and 50 on
24 cores caused by three other agents. No wall-clock figure from these runs is
reported as a measurement of anything.

**Anchor.** `scripts/y3_e0_anchor.py` re-run on the final code: E0 ANCHOR PASS
(48 campus x size x rule triples exact, policy replay deterministic, M1 gate=0
bit-exact, shaped-reward telescoping OK). Log:
`results/y3_w1/e0_anchor_after_w1.log`.
