# P9b run plan: the corpus-anchored cells under the DEPLOYABLE review policy

Written before any run was launched. Driver: `scripts/y3_p9b_deployable_cells.py`,
which calls `scripts/y3_w1_sweep.evaluate_cell` verbatim (its cache redirected)
and reuses `scripts/y3_realistic_cell.summarize_cell` verbatim for the
statistics. Outputs go to this directory only. Nothing under `paper/`, `src/` or
the existing `scripts/` is edited, and nothing is committed.

## Why this run exists

Two packages each produced a headline number, and the abstract quotes them side
by side:

* `results/y3_p9/` measured the correction layer at the recoverable share the
  real corpus supports (`beta = 0.20` and `beta = 0.25`), and the abstract leads
  with that number (`\RealBetaGain` = 12.6%).
* `results/y3_w1/` replaced the review policy. The published policy
  (`Supervisor` with `mechanism="targeted"`) decides which decisions the
  supervisor reviews partly by reading the realized latent shift of the pending
  queue, so no site can run it; the replacement routes on a decision-stability
  test under a split-conformal band calibrated only on override-derived weak
  labels, and is computable from observables. The deployable policy is now the
  paper's protocol, and its headline number is `\RoutedGain` = 48.2% against the
  published policy's `\MzeroGain` = 45.4%.

The three `y3_p9` cells were run under the OLD policy. The abstract therefore
quotes a corpus-anchored number measured under a protocol the paper no longer
claims, next to a full-recoverability number measured under the protocol it does
claim. This run re-measures exactly those three cells under the deployable
policy so the two figures come from one protocol.

## The four pre-launch questions

### 1. Purpose: what question does each run answer, and where does the number land

| Run | Question | Destination |
|---|---|---|
| R0 reproduction gate | Does this script's code path reproduce the published `\MzeroGain = 45.4%` at the headline cell? | Gates everything below. Reported as the provenance sentence of the new macro block. |
| R1 per-cell mirror | Does this code path reproduce `results/y3_p9/` bit-for-bit at the three cells when the review policy is set back to the published one? | The empirical configuration diff: it proves the ONLY thing that changed is the review policy. Reported, not quoted. |
| A' `beta=0.20`, `eps=0` | What reduction does the correction layer deliver at the corpus-anchored recoverable share under the policy a site can actually run? | Replaces `\RealBeta*` in the abstract and the regime-map table. New macros `\RealBetaRouted*`. |
| B' `beta=0.20`, `eps=0.25` | Does it survive a supervisor who errs on a quarter of its reviews? | Second row; the realistic-operating-point sentence. `\RealBetaEpsRouted*`. |
| C' `beta=0.25`, `eps=0` | Is the value at A' a point or a small range? | Third row, so the paper can state a corpus-supported range. `\RealBetaHiRouted*`. |
| D control (`targeted`, split protocol) | Of the difference between `y3_p9` and this run, how much is the routing rule and how much is the conformal fold split the routing rule needs? | A diagnostic paragraph, not a quoted number. Labelled upper reference at these three cells. |

Every cell is at campus 9, storm2 `w80`, `u = 100` (saturation), `rho = 0.25`,
`theta = 1.0`, channel `full_class_shift`, family `F-NL`, `master_seed = 12345`,
seeds 301-310, evaluated on `files[20:30]` of the c9/storm2/w80/u100 pool: the
same ten held-out instances `y3_p9` used. Deciders scored: RULE, RULE+SUP, M0,
M0+SUP, ORACLE. Statistics: seed-average each decider's per-instance true
weighted tardiness, then a two-sided paired Wilcoxon signed-rank test
(`zero_method='pratt'`) over the ten held-out instances, W/T/L counted as the
test being strictly lower. This is `y3_p9`'s own `summarize_cell`, imported and
called, not re-derived.

### 2. Ideal result, and what the opposite would mean

The expectation, written down before the run. On the eight-cell contention grid
of `results/y3_w1/` the two policies were statistically indistinguishable, with
the point estimate slightly favouring the deployable one (mean price of
deployability `-0.4` percentage points, per-cell range `-3.3` to `+1.1`, no cell
surviving Holm), and at the headline cell the deployable policy came out 2.3
points BETTER. **These three cells should therefore land close to
`results/y3_p9/` and may come out a little higher.**

* **If the corpus-anchored gain stays positive and separable** (the expected
  case): the abstract's claim survives, and the numbers it quotes come from one
  protocol instead of two. `\RealBetaGain` is replaced by its deployable
  counterpart and nothing else in the argument moves.
* **If it comes out LOWER, or loses separability at `beta = 0.20`**: that is an
  equally acceptable outcome and is reported as measured. The consequence is
  concrete: the abstract must quote the lower figure, and if the layer alone
  stops being separable from the tuned rule at the corpus-anchored share, the
  claim moves to the in-loop configuration (M0+SUP against RULE+SUP), with the
  scope limit stated plainly.
* **If it comes out far higher than the `y3_p9` value** (several points beyond
  the `-3.3` to `+1.1` band the eight-cell grid showed): that is outside what the
  W1 grid supports, and I would re-read the configuration diff for an unintended
  difference and report the discrepancy rather than the number.

Both branches change what the abstract says, so the run is worth its cost. No
tuning, no seed selection, no search for a better configuration: the three cells
listed above are the whole run and every one of them is reported.

### 3. Contamination risks, and how each is closed

1. **Stale or shared cache.** `y3_w1_sweep.evaluate_cell` caches by a SHA-1 over
   the resolved task into `results/y3_w1/cache/`. This run redirects that module
   global to `results/y3_p9b/cache/` inside every worker, so (a) the
   reproduction gate and the per-cell mirror genuinely recompute instead of
   reading a published record back, and (b) nothing is written into the
   published W1 cache. The published `results/y3_p9/cache` and
   `results/y3_p4/cache` are opened read-only, for comparison.
2. **Sweeping into an existing analysis.** No script under `scripts/` or `src/`
   references `results/y3_p9b` (it does not exist before this run); the published
   harvesters glob `results/y3_p4/cache`, `results/y3_p9/cache` or
   `results/y3_w1/cache` only. Writing into a fresh directory cannot change any
   published number. Verified by grep before the run.
3. **Partially written result files.** Every JSON/CSV output is written to a
   temporary file and renamed into place. The aggregation asserts the expected
   record count per cell (10 seeds x 10 instances x 5 deciders) and refuses to
   summarise a short harvest.
4. **Split leakage.** `evaluate_cell` asserts the held-out set is disjoint from
   the train and probe pools. This run additionally asserts the ten held-out
   instance ids equal the published `eval_inst_ids` of
   `results/y3_p5/harvest/primary_multiseed_summary.json`, and equal the ids in
   the `results/y3_p9/` records it is compared against.
5. **Latent leakage into the routing rule.** The routing rule must be computable
   from observables. `fmwos.hitl.routing` enforces this structurally
   (`_forbid_latent`, the weak-label alphabet assertion, and the source-level
   grep in `tests/test_routing.py`); this run does not weaken it, and the test
   file is run before the sweep as a gate.
6. **Floating-point drift from thread count.** This pipeline reproduces
   bit-exactly only with one numeric thread per process. `OMP_NUM_THREADS`,
   `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and
   `VECLIB_MAXIMUM_THREADS` are hard-set to `1` before numpy/torch import (hard,
   not `setdefault`, because `y3_w1_sweep` would otherwise default them to 4),
   `torch.set_num_threads(1)` runs in every worker, and each worker asserts
   `torch.get_num_threads() == 1` and the three environment caps before it
   computes anything. Parallelism comes from separate worker processes.
7. **Device contention.** The machine is shared with three other agents (load
   average around 37 of 24 cores when this plan was written). The run is pinned
   with `taskset -c 10-19` to eight single-threaded workers; cores 0-9 and 20-23
   belong to other agents. No wall-clock timing is measured or reported, because
   under contention a timing is not a measurement.
8. **Silent configuration drift.** Every cell's resolved configuration is dumped
   and diffed field by field against the resolved configuration of its `y3_p9`
   counterpart. The run asserts that every shared field is identical and that the
   difference is exactly the review-policy block, and aborts otherwise. The
   empirical form of the same check is stronger and is also asserted: RULE and
   ORACLE do not depend on the review policy at all, so their per-instance TWT*
   must be bit-identical to the `y3_p9` records, and the per-cell mirror arm must
   reproduce all five deciders bit-for-bit.
9. **Which policy actually ran.** `fmwos.hitl.routing.make_supervisor` is wrapped
   in a pass-through that asserts the returned object is a
   `StabilityRoutingSupervisor` whenever the task's policy is `stability`, counts
   the constructions by class, and records the count in the output. The record's
   own telemetry carries the second proof: the stability test reports the share
   of decisions it marks UNDETERMINED, and the old policy produces no such share
   (the field is `nan`), so its presence is the evidence.

### 4. Data accuracy

* The instance pool is `data/processed/instances/c09/storm2/w80/c09_storm2_w80_u100_*.json`,
  30 files, sorted; train `[0:16]`, probe `[16:20]`, held-out `[20:30]`. The count
  is asserted and the SHA-256 of each of the ten held-out files is recorded and
  asserted equal to the values in `results/y3_p9/data_checks.json`.
* The held-out ids are asserted equal to the published set
  (`c09_storm2_w80_u100_0020` ... `_0029`).
* The overlay coefficient file `results/y3_p1/overlay_coeffs/F-NL_seed12345.json`
  (SHA-256 `21b3ef67c107d2c5fbbbe6a1ee28354851c6928cec64aa66749cf2f1c1b9413b`) is
  shared by every cell and asserted equal to the value `y3_p9` recorded; only
  `beta` changes what the overlay draws from it.
* The tuned rule, the independent validator (`true_objective.score_true`), the
  five deciders and the held-out slice are the same objects `y3_p9` used, because
  the two runners construct them with identical calls; the per-cell mirror arm
  turns that into a bit-exact check rather than a claim.

## The one intended difference, stated exactly

`y3_p9` resolved `mech = "targeted"` and fitted the estimator on the whole
weak-label aggregate. This run resolves `policy = "stability"` with
`split_fit = True`, `cal_frac = 0.3`, `alpha = 0.1`, `band_mode = "global"`.

The fold split is not a second free choice: the conformal band the stability
test needs must be calibrated on examples the estimator has never been fitted on,
so the split is part of the deployable protocol rather than an independent knob,
and W1 ran its oracle-informed reference under the same split for exactly this
reason. Because the split nevertheless costs the deployable arm about 30% of its
training labels, this run also carries the `targeted` policy under the same split
at the three cells (run D), so the difference against `y3_p9` can be decomposed
into "the routing rule" and "the fold split" rather than reported as one lump.
Run D is a diagnostic; it is not quoted in the manuscript.

## Deliverables this run produces

`results/y3_p9b/`: `RUN_PLAN.md` (this file), `repro_check.json`,
`mirror_check.json`, `config_diff.json`, `data_checks.json`, `policy_proof.json`,
`cells.csv`, `cell_summary.json`, `comparison_table.md`, `macros_snippet.tex`,
`run.log`.

---

# What actually happened

Appended after the runs, against the expectations stated above.

**All four gates passed.**

* *Reproduction.* Through this code path the published headline cell recomputes
  `\MzeroGain` at 45.3620354692% against the published 45.3620354692%, a
  difference of 0.00e+00 percentage points, with 500 of 500 per-instance TWT*
  values equal to `results/y3_p4/cache` to the bit
  (`results/y3_p9b/repro_check.json`).
* *Per-cell mirror.* With the review policy set back to the published one, this
  code path reproduces `results/y3_p9/cache` bit-for-bit at all three cells:
  1500 of 1500 per-instance values, max absolute difference 0.0
  (`results/y3_p9b/mirror_check.json`). That is the empirical configuration
  diff, and it is exact.
* *Configuration diff.* No shared field differs at any cell; the difference is
  exactly the review-policy block (`results/y3_p9b/config_diff.json`).
  Independently, RULE and ORACLE, which depend on no review policy, are
  bit-identical to the `y3_p9` records in both arms.
* *Which policy ran.* The wrapped constructor built 4,440
  `StabilityRoutingSupervisor` objects and zero plain `Supervisor` objects in
  the deployable arm (3,960 with a calibrated band, 480 at the first-iteration
  cold start), and the mirror-image counts in the reference arm. The records
  carry a finite undetermined share, which only the stability test produces
  (`results/y3_p9b/policy_proof.json`).

**The measured outcome is the LOWER branch of the plan, and it is reported as
measured.** The correction layer alone falls from 12.63% to 9.53% at
`beta = 0.20`, from 6.31% to 3.97% at `beta = 0.20` with an erring supervisor,
and from 15.00% to 13.43% at `beta = 0.25`: 1.6 to 3.1 percentage points lower
than under the published policy at every cell. That is the opposite sign to the
headline cell, where the deployable policy was 2.3 points better.

**Against the same-split reference the price of deployability is larger here
than anywhere the W1 grid measured it, and that is a genuine extension rather
than a contradiction.** In W1's sign convention (oracle-informed minus
deployable, positive meaning the deployable policy is worse) the price at these
cells is `+4.00`, `+1.94` and `+2.46` points for the layer alone, against a
per-cell range of `-3.25` to `+1.05` over the eight-cell contention grid. The
grid, however, covers only `beta = 0.75` and `beta = 1.00`; no cell in it sits
anywhere near a recoverable share of 0.20 to 0.25. The reading is therefore that
the price of deployability grows as the recoverable share falls, which the grid
could not have shown, and not that the grid was wrong.

**Separability survives at both `eps = 0` cells and is lost for the layer alone
at the noisy-supervisor cell.** Against the tuned rule the layer alone is
separable at `beta = 0.20` (8/0/2, raw p = 0.0098) and at `beta = 0.25` (8/0/2,
raw p = 0.0098), and both survive Holm across the three cells at 0.029; at
`beta = 0.20` with `eps = 0.25` it does not (7/0/3, raw p = 0.084). In the loop,
against the rule under the same supervisor, the noisy-supervisor cell is
separable on the raw p (8/0/2, p = 0.027) and so is `beta = 0.25` (8/0/2,
p = 0.037), while `beta = 0.20` at `eps = 0` misses at p = 0.064 (7/0/3); Holm
across the three in-loop tests puts all three at 0.082, so none of them survives
correction as a family. `results/y3_p9/` reported raw per-cell p-values and did
not Holm-correct the three cells, so the raw values are the like-for-like
comparison and the Holm figures are stated as extra information. Ten paired
instances put
the two-sided floor at p = 0.001953, so these are low-power tests and the honest
reading is that the effect is smaller under the deployable policy rather than
that it has gone.

**The loss is the routing rule, not the fold split.** The oracle-informed policy
run under the SAME conformal fold split lands within about one point of the
published measurement (fold-split effect `-0.6` to `+0.9` points), so the 30% of
training labels the split costs is not what moved the number; the routing rule
accounts for `-1.3` to `-4.0` points.

**Why the routing rule loses here, when it won at full recoverability.** At
these recoverable shares the band is wide relative to the signal it must
certify, so the stability test cannot discriminate: 82.4% of multi-candidate
decisions are undetermined at `beta = 0.20`, 82.0% at `beta = 0.25`, and 97.6%
once the supervisor errs on a quarter of its reviews, where the calibrated
half-width triples from about 0.31 to 1.03 class-shift units. Against a review
budget of 25% the policy is therefore ranking inside an undetermined set three
to four times larger than the budget, which is the same failure mode the W1
diagnostic identified in the oracle-informed policy at the headline cell (its
latent clause flagged 47.2% of decisions against the same 25% budget and so
could not rank within the set it flagged). The realised review fraction is at
budget in every cell (0.245 to 0.246, against 0.247 to 0.248 under the published
policy), so the difference is placement and not spend.

**Compute, stated rather than implied.** All runs were pinned to cores 10-19
with eight single-threaded workers, under a machine load average between 37 and
57 on 24 cores caused by three other agents. No wall-clock figure from these
runs is reported as a measurement of anything.

**What could not be computed.** Nothing in the plan was skipped. One limit is
worth naming: the `targeted_split` arm is a diagnostic run at these three cells
only, not a re-measurement of the eight-cell grid, so it decomposes the
difference here and does not restate W1's grid-wide price of deployability.
