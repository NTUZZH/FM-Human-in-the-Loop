# W2 run plan — queue-conditioned choice-model estimator

Written **before** any run was launched. Every run below states its purpose and
destination, the expected result and what the opposite outcome would mean, the
contamination risks checked, and the data-accuracy checks actually performed.
Both directions of the acceptance decision are publishable outcomes, so no run
is tuned toward a preferred result.

## The cell every run uses

The headline cell of the published M0 gate, taken verbatim from
`scripts/y3_p4_m0grid.py` (`_base_task` + `tasks_A`):

| field | value | source |
|---|---|---|
| campus | 9 | `tasks_A` primary scope |
| regime / track | `storm2`, window `w80`, `u = 100` (saturation) | `tasks_A` |
| beta | 1.0 | `tasks_A` |
| rho | 0.25 | `tasks_A` |
| epsilon | 0.0 | `EPS` |
| theta | 1.0 | `THETA` |
| mechanism | `targeted` | `MECH` |
| channel | `full_class_shift` | `CHANNEL` |
| family / master seed | `F-NL` / 12345 | `FAMILY`, `MASTER_SEED` |
| instance split | `files[:16]` train, `files[16:20]` probe, `files[20:30]` eval | `evaluate_cell` |
| outer DAgger iterations | 8 | `m0_iters` |
| override / confirm weight | 5.0 / 1.0 | `run_m0` defaults |

Published headline: `\MzeroGain` = 45.4% (`paper/macros.tex` line 128, comment
`m0_alone pct_below_rule 45.3620`), the 10-seed (301–310) mean of the
seed-averaged per-instance reduction of `m0_alone` against `rule` in
`results/y3_p4/m0_gate_summary.json`, cell `c9_storm2_u100_b1.00_r0.25`. The
seed-301 value recomputed from `results/y3_p4/m0_gate.csv` is **45.1911%**; the
pilot targets that, not the 10-seed mean.

## The rungs

| rung | id | objective | estimator input | fitting protocol |
|---|---|---|---|---|
| (i) | `mse_published` | weighted squared error | per-order features | all 16 train instances, fixed 40 epochs (the shipped `run_m0`, called verbatim) |
| (i-es) | `mse_es` | weighted squared error | per-order features | 12/4 instance split, early stopping |
| (ii) | `choice` | conditional logit over Q | per-order features | 12/4 instance split, early stopping |
| (iii) | `choice_queue` | conditional logit over Q | per-order features + Deep-Sets pool(Q) | 12/4 instance split, early stopping |

Rung (i-es) is **not** in the original plan; it is added because without it the
(i) → (ii) difference confounds the objective with the fitting protocol, and the
package's whole point is to attribute a difference to the objective.

Robustness rows on top: `choice_tol` / `choice_queue_tol` (tolerance-aware
confirmation term instead of the over-reading full-choice term) and
`choice_queue_k64` (choice set capped at the plan's K ≤ 64).

---

## R0 — pipeline-fidelity pilot

**Purpose and destination.** Prove that the W2 code path reproduces the
published M0 pipeline before anything is compared against it. Nothing from this
run is quoted in the manuscript; it is the gate that licenses R1. Its number
lands in the W2 report's fidelity line only.

**Expected result.** Rung (i) run through `scripts/y3_w2_pilot.py` reproduces the
seed-301 per-instance TWT\* of `m0_alone` in `results/y3_p4/m0_gate.csv`
**exactly** (max absolute difference 0 across the 10 held-out instances), hence
45.1911%. **If it does not**, the difference is a bug in the W2 harness or an
environment drift (torch/numpy version, thread count), and R1 is not launched
until it is explained; a ladder built on a path that cannot reproduce the
incumbent measures the harness, not the estimator.

**Contamination risks checked.**
- *Stale checkpoints*: none exist — the M0 estimator is fitted from scratch inside
  every run and no W2 script reads or writes `results/y3_p4/cache/`.
- *Partially written result files*: every W2 result is written to `path.tmp` and
  `os.replace`d, so a killed run cannot leave a half-file for the analysis to
  sweep in.
- *Train/test leakage*: `evaluate_cell`'s assertion that eval files are disjoint
  from train+probe is re-asserted in `build_cell`; the choice model's own
  train/validation split is by INSTANCE, never by decision, because the same
  work order recurs across decisions and across DAgger iterations, so a
  decision-level split would put the same order's features on both sides.
- *Contended machine*: the box is shared (load average ≈ 23 on 24 cores, a
  `p2_train_mappo.py` job at ~1400% CPU). Every W2 job runs under
  `taskset -c 10-19` with `OMP_NUM_THREADS=MKL_NUM_THREADS=4`. **Wall-clock is
  therefore reported as an upper bound only, never as a measurement**, and no
  timing claim is made about anything except the per-decision deployment cost,
  which is reported with the contention stated.
- *Thread-count drift*: the published run used `torch.set_num_threads(1)`. The
  pilot runs at 1 thread to match the published configuration exactly, and
  separately at 4 threads, so any thread-induced numerical drift is measured
  rather than assumed.

**Data-accuracy checks performed.**
- The 30 instance files at `c09/storm2/w80/u100` are located with the same glob
  and the same `sorted()` order as `y3_p4_m0grid.locate_files`; the resolved
  train / probe / eval file lists are written into the result file.
- The resolved configuration is diffed field-by-field against the published
  `_base_task(campus=9, u=100, beta=1.0, rho=0.25, seed=301)` dictionary, and the
  run aborts on any difference.
- The comparison target is recomputed from `results/y3_p4/m0_gate.csv` rather
  than copied from the macro, and the recomputed 10-seed mean is checked against
  `\MzeroGain` = 45.3620.

---

## R1 — the four-rung ladder

**Purpose and destination.** The package's primary experiment. Lands in the new
Table 5 row ("choice-model queue-conditioned vs squared-error per-order") and in
every place the manuscript reports recovery quality, which C0/C7 require to
carry Kendall tau. Also decides W2's acceptance criterion, and therefore whether
the abstract presents the choice model as one of the three headline algorithms
or demotes it to a modelling finding.

**Seeds.** 301–305, the paper's ablation seed range (`\seedAblHi` = 305,
`\seedsAbl` = 5), on the 10 held-out instances of the headline cell.

**Statistics.** The paper's own test: seed-average each decider's per-instance
TWT\*, then a two-sided paired Wilcoxon signed-rank test (`zero_method='pratt'`)
over the 10 held-out instances, with Holm correction within the family of
contrasts reported here. Identical to `y3_p4_m0grid._contrast`.

**Metrics.** Pearson r and sign accuracy on non-zero-shift orders (the shipped
recovery probe, on the 4 probe instances); Kendall tau-b between the corrected
ranking and the true ranking at decision points (on the 10 eval instances, at a
COMMON reference trajectory — the plain ATC rollout — so the metric compares
rankings and not trajectories); held-out choice-model log-likelihood on reviewed
decisions of RULE+SUP episodes over the 10 eval instances, under one common
conditional-logit functional with the scale calibrated per variant on the
validation instances only; and the deployed reduction in true weighted tardiness
against the tuned rule, scored by the existing independent validator
(`hitl.true_objective.score_true`), which is the sole referee.

**Expected result.** The likelihood correction (ii) improves sign accuracy and
Kendall tau over (i-es) at equal or better TWT\*, and the queue conditioning
(iii) adds a smaller further gain. **If the opposite happens** — either rung
fails to improve, or improves recovery while losing TWT\* — the finding is
reported as "context conditioning does not pay at this override budget", the
choice likelihood is retained on per-order features if (ii) alone succeeds, and
the manuscript demotes the component per C0's claim discipline. Both directions
are successful outcomes of the package; neither is tuned toward.

**Contamination risks checked.** As R0, plus:
- *Silent configuration drift between rungs*: the resolved configuration of every
  rung is diffed against rung (i)'s, and the run aborts unless the ONLY differing
  field is `variant`.
- *Silent capacity drift*: the parameter count of every rung is asserted before
  training (1761 for the per-order estimator, 3041 for the queue-conditioned one,
  +1 for the choice scale), so a variant that quietly grew cannot be compared.
- *Objective leaking into the metric*: the held-out choice log-likelihood is
  computed with ONE functional for all four rungs, including the squared-error
  rungs which have no likelihood of their own; each rung's scale is calibrated on
  the validation instances, never on the test set.
- *Latent leaking into the estimator*: enforced structurally, not promised — one
  feature constructor (`choice_estimator.instance_tables`) feeds everything, and
  `tests/test_choice_estimator.py::test_instance_tables_read_only_observable_fields`
  records every key the feature path touches and asserts the set is a subset of
  {id, trade, p_bh, release_bh, priority, due_bh, weight}. The overlay latent is
  read only inside the two clearly-marked evaluation functions.
- *One sweep at a time*: W2 never runs two of its own sweeps concurrently.

**Data-accuracy checks performed.** As R0, plus the assertion that the four rungs
see the same instance objects, the same overlay, the same supervisor
configuration and the same RNG stream for the DAgger instance order
(`np.random.default_rng(seed)`, matching `run_m0`).

---

## R2 — robustness rows (gated on R1 finishing)

**Purpose and destination.** Two stated simplifications, each measured rather
than argued: (a) treating a confirmation as a full choice over-reads it, since it
certifies only a near-tie within the override tolerance theta — measured by
`choice_tol` / `choice_queue_tol`; (b) the plan's K ≤ 64 choice-set width, which
does not bind here because the measured feasible set reaches 176 with 30% of
reviewed decisions above 64 — measured by `choice_queue_k64`. Lands in the
report and, if it changes a conclusion, in the manuscript as a robustness
sentence.

**Expected result.** Neither changes the sign of R1's conclusion. **If one does**,
the tolerance-aware variant becomes the reported form and the simplification is
no longer "stated openly" but corrected.

**Seeds.** 301–303, three seeds, because these rows qualify a conclusion rather
than establish one.

---

## R3 — reduction-anchor re-verification

**Purpose and destination.** Part D item 6 of the submission plan: the
bit-for-bit reduction anchor is re-verified after every code change. With the
supervisor disabled and the deadline-aware gate closed, the environment must
reduce exactly to the public benchmark's dispatcher. Lands in the W2 report and
in the submission gate's "anchor re-verified on final code" checkbox.

**Expected result.** `scripts/y3_e0_anchor.py` exits 0. **If it does not**, the W2
code has touched shared state and every W2 number is void until it is fixed.

**Note.** W2 adds files only; it edits nothing under `src/fmwos/` or `scripts/`,
so the anchor cannot change unless an import side-effect exists — which is
exactly what this run rules out.

---

# Addendum, written after the runs

Everything below is a record of what actually happened, including the runs that
were not in the plan above and the three places the plan was deviated from.

## R0a — thread-count sensitivity (added; not in the original plan)

**Purpose and destination.** The plan pinned the pilot to `torch.set_num_threads(1)`
to match the published run and said any thread-induced drift would be *measured
rather than assumed*. This run measures it. Its number lands in the W2 report's
reproducibility note and in the ladder's `--threads` default.

**Result.** The identical pipeline at 4 threads gives 46.7514% instead of
45.1911% at seed 301, a **+1.560 percentage-point** shift, with a maximum
per-instance TWT\* difference of 396.75. Changing only the intra-op thread count
changes the floating-point reduction order inside the estimator's matmuls, which
changes the fitted weights, which changes dispatch decisions. The shift is the
size of the published seed spread (`\MzeroGainStd` = 1.7).

**Consequence.** Every W2 run is pinned to one torch thread. This is a deliberate
deviation from the plan's `torch.set_num_threads(4)`: that plan's own overriding
requirement is that the only difference from the comparator be the estimator, and
at 4 threads the thread count is a second difference. One thread was also the
faster setting for these models (48.2 s against 49.2 s).

## R1a — objective diagnostic on a FIXED decision stream (added)

**Purpose and destination.** The first end-to-end smoke run showed the choice
rungs fitting the choice well and recovering the latent badly. That has two
incompatible explanations, and the package's conclusion depends on which is
true: the OBJECTIVE is wrong for this problem, or the OPTIMISER setting is. A
negative result published on the second explanation would be a false negative.
This run separates them and, in the same pass, selects each objective's learning
rate. Its numbers land in the report's diagnostic table and in
`y3_w2_lib.LR_BY_OBJECTIVE`; none is quoted as a headline.

**Design.** One decision stream, generated by the SHIPPED squared-error decider
over three DAgger iterations (19,869 reviewed decisions, 957 overrides), is
shared by every row, so each row differs only in the objective, the inputs and
the optimiser setting, never in the trajectory it was fitted on. 26 rows:
learning rate x objective x per-order/queue; batch 128 against 512; an
overrides-only ablation that deletes the confirmation term entirely; the
tolerance-aware confirmation at three tolerances; and a frozen choice scale.

**Selection rule, and why it matters.** Hyperparameters are selected on the
VALIDATION objective alone. They are never selected on Pearson r, sign accuracy
or Kendall tau, because those read the simulator's true shift; selecting on them
would reintroduce the oracle through the back door, exactly the failure W1's
calibration section is written to avoid. One consequence is reported rather than
hidden: batch 128 gives better latent recovery (Pearson 0.442, sign accuracy
0.825) than the selected batch 512 (0.330, 0.761), but slightly worse validation
loss, so batch 512 is what the deployable criterion picks.

**Expected result and the opposite.** If the optimiser were the problem, some
setting would recover the latent as well as the squared-error fit. **The opposite
happened**: no setting did, and the overrides-only row isolated the mechanism
(see the report), so the negative result is about the objective.

**Contamination risks checked.** Same as R1. Additionally, the stream is built
once and reused, so no row can be advantaged by a better trajectory; and the
overrides-only rows are strict subsets of the same stream, not a re-collection.

## Deviations from the plan above

1. **Threads pinned to 1, not 4** (R0a).
2. **`K_MAX` defaults to 512, not the plan's 64.** The measured feasible set at a
   reviewed decision has mean 50, median 31 and maximum 176, with 30% of reviewed
   decisions wider than 64 (`probe.json`). The Deep-Sets encoder is
   parameter-free and size-agnostic, so 64 is not an architectural bound, and the
   deployed decider pools over the whole queue, so capping only during training
   would create a train/deploy mismatch on 30% of decisions. The plan's cap is run
   as an explicit robustness row (`choice_queue_k64`) instead, and it does not
   change the conclusion.
3. **A fourth rung, `mse_es`, was added.** Without it the (i) -> (ii) difference
   would confound the objective with the fitting protocol.

## Compute note

The box is shared with three other agents. During the ladder the load average was
37 to 41 on 24 cores, with an unpinned 51-thread training job at ~1,400% CPU.
Every W2 job ran single-threaded under `taskset` inside cores 10-19, one process
per seed. **No wall-clock figure from any of these runs is a measurement.** The
only timing reported is the per-decision deployment cost, and it is reported as a
RATIO between two deciders timed in the same process, interleaved, on the same
instances, so the contention applies equally to both.
