# W7 run plan — leave-one-campus-out transfer of the correction layer

**Recording note, stated plainly.** The four pre-launch questions below were
settled before the first fold was launched, but they were recorded in the
working session rather than in this file, and this file was written after the
runs completed. The two additions made *after* seeing the first results are
marked as such, so a reader can separate what was planned from what was
diagnosed. Nothing below was tuned toward a preferred outcome, and both
directions of the result are reported in the manuscript.

## 1. Purpose

Does one fitted correction layer ship to a site it never saw? The published
transfer experiment (`results/y3_p6/`, `scripts/y3_p6_transfer.py`) answered
this for two campuses held out by design. W7 generalises it: every campus is
held out in turn, the estimator is fitted on the others, and it is scored
against a natively fitted layer on the same held-out instances.

The number lands in the manuscript's transfer subsection and its table, and it
decides whether the deliverable is a layer or a per-site fitting recipe.

## 2. Ideal result, and what the opposite would mean

Expected: retention near one on the contended held-out campuses, and inertness
on the forced-queue campuses for *both* arms, so that the Payoff Condition
rather than transfer governs where the layer pays.

If instead retention collapses on a contended campus, that is a real limit on
the deliverable, and the manuscript says so: the layer would have to be refitted
per site, which is cheap but is a different claim from shipping one layer.

**Outcome: the second, in part.** With a source pool drawn from every other
campus, retention is 0.96 on held-out C1 and 0.09 on held-out C9, and C9 is the
only contrast in the table where transfer is separably worse than native fitting
(p = 0.039). Both readings are reported.

## 3. Contamination

- **Stale checkpoints.** None. Every estimator is fitted fresh inside the fold;
  nothing is resumed or overwritten.
- **Partial files.** Each fold is written to a `.part` file and renamed
  atomically, so a killed run leaves no half-written fold for the analysis to
  sweep in. Completed folds are skipped on relaunch by cache key.
- **Device contention.** The machine carried other jobs (load average about 14
  of 24 cores). Workers are pinned with `taskset` and every numeric library is
  hard-capped to one thread, asserted at start-up. **No wall-clock figure from
  this run is reported as a measurement of anything.**
- **Leakage.** The held-out campus contributes no instance to the transfer
  pool, so transfer-side leakage is structurally impossible. For the native arm,
  the runner asserts that neither training pool shares an instance id with the
  evaluation slice, and aborts otherwise.

## 4. Data accuracy

The runner imports every locked constant from `y3_p6_transfer` and asserts it
against an explicit table (`assert_locked_config`), so it cannot drift from the
experiment it extends: family F-NL, master seed 12345, eps 0, theta 1.0,
targeted review, full-class-shift channel, rho 0.25, 8 outer rounds, replay
track at size 150 crew-scaled to utilisation about 1.0 inside the band
[0.85, 1.20], 16 training / 4 probe / 10 evaluation instances.

The **only** deliberate difference is the composition of the transfer training
pool. Its *size* is held at the comparator's 16 instances (4+3+3+3+3 over five
campuses rather than 4x4 over four), because training-set size is a confound and
the question is which campuses, not how many instances.

Two checks confirm the runner reproduces the comparator's setting: the estimator
parameter count is asserted at 1761 on every fit, and the held-out C1 rule
baseline and native gain reproduce the published `results/y3_p6/` values.

## Added after seeing the first results, and labelled as such

1. **Contended-source diagnostic.** The first three-seed pass showed transfer
   failing on C9 with a pool containing three campuses that are themselves inert
   on this track. Restricting the pool to the contended campuses, at the same
   16-instance budget, tests whether the failure is dilution of the training
   signal rather than a failure of transfer. It is reported as a separate,
   labelled block of the manuscript table, never merged into the main folds.
2. **Seed extension from 3 to 10.** The three-seed pass carried seed spreads of
   6 to 9 percentage points, which is not enough power for a finding that
   contradicts a claim already in the manuscript. Seeds 301-310 were run for
   every fold at beta = 1.0. This adds power and does not change the cell.

## Files

- `folds/c<NN>_b<beta>_s<seed>[_contendedsrc].json` — one fold: the per-instance
  true weighted tardiness for rule, transfer, native and reference, the recovery
  metrics for both arms, the training campuses and their quota, and the asserted
  parameter count.
- `loco_summary.csv` — seed-aggregated gains, retention, and the paired
  instance-level Wilcoxon contrasts reported in the manuscript.
