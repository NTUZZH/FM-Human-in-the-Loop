# W3 run plan — censoring-aware ordinal shift likelihood

Written **before** the comparison runs were launched (R0, the fidelity pilot,
had already passed and is reported below because it gates everything else).
Every run states its purpose and destination, the expected result and what the
opposite outcome would mean, the contamination risks checked, and the
data-accuracy checks actually performed. Both directions of the acceptance
decision are publishable outcomes, so no run is tuned toward a preferred result.

## What the package changes, and what it deliberately does not

The true priority class is `c* = clip(c - s, 1, 4)`, so the scale censors the
evidence at both ends: a recorded class-1 order can only be revealed as less
urgent, a recorded class-4 order only as more urgent. The shipped estimator fits
a weighted squared error to point labels in {+1, -1, 0}, so on a boundary class
it fits a target the scale cannot express (a -1 on a class-4 order asserts a
true class of 5).

W3 replaces the objective with a **two-limit Tobit likelihood** on the effective
shift `t = c - c* = clip(s, c-4, c-1)` — the quantity the deployed correction
actually applies, since `c_hat = clip(c - s_hat, 1, 4)`. An observation inside
the expressible range contributes a Gaussian density at a point on the shift
scale; an observation at or beyond a limit contributes the one-sided probability
`log Phi((L - mu)/sigma)` ("at most this urgent") or `log Phi((mu - U)/sigma)`
("at least this urgent").

**Why Tobit and not an ordinal logit with collapsed boundary categories.** Both
were admissible. Tobit keeps the output semantics the deployed decider already
consumes (one continuous shift per order), whereas an ordinal logit outputs a
class distribution and needs a decoder before the ATC index can use it, which
would add a second difference to the comparison. Tobit also *nests* the
incumbent: at `sigma = 1` with nothing censored the two objectives are the same
arithmetic, so the comparison is controlled by construction rather than by
argument. And it holds the network at the incumbent's parameter count exactly.

**The level stays anchored (the W2 lesson, deliberately not repeated).** W2
replaced the squared error with a ranking objective and it failed on deployment,
because a ranking objective is invariant to the *level* of the shift and
confirmations were what pinned that level. Every term in the Tobit likelihood is
a density at a point, or a probability of a half-line, **on the shift scale
itself**; no term is invariant to adding a constant to `mu`. Uncensored
observations still contribute the incumbent's unbounded quadratic pull toward a
fixed point. `tobit_imp` is carried as the maximally anchor-preserving variant:
it censors only the two structurally impossible labels and leaves every
attainable point label, including every boundary-class confirmation, as a point
anchor.

## The rungs

| rung | id | objective | sigma | censoring | deployment |
|---|---|---|---|---|---|
| (i) | `mse_published` | weighted squared error | — | — | plug-in (shipped `run_m0`, called verbatim) |
| (i-b) | `mse_reexpr` | weighted squared error | 1 fixed | none | plug-in (re-expressed loop; the bit-exactness control) |
| (ii) | `tobit` | two-limit Tobit | 1 fixed | strict (an observation at a limit is censored) | plug-in |
| (iii) | `tobit_imp` | two-limit Tobit | 1 fixed | impossible labels only | plug-in |
| (iv) | `tobit_exp` | two-limit Tobit | 1 fixed | strict | `E[clip(s, L, U)]` |
| (v) | `tobit_sig` | two-limit Tobit | fitted | strict | plug-in |
| ref | `classmean_oracle` | — | — | — | the TRUE population mean effective shift of each recorded class |

Rung (i-b) is not in the original plan; it is added because without it a difference
between (i) and (ii) could be the re-expressed outer loop rather than the
likelihood. Rung (iv) is separated from (ii) because the likelihood and the
decision rule are separable claims and each must be able to succeed or fail on
its own: `E[clip(s, L, U)]` differs from `clip(E[s], L, U)` exactly at the
boundary classes, which is the same place the censoring bites.
`classmean_oracle` is an EVAL-ONLY upper reference, never a deliverable: it
reads the simulator's true classes to answer one question the package cannot
answer otherwise — how much of a `beta = 0` reduction a class-level constant can
buy when it is exactly right.

Parameter counts are asserted before the first gradient step: the estimator must
have 1761 parameters, the incumbent's, and the total must be 1761 (sigma fixed,
a buffer) or 1762 (sigma fitted). A mismatch aborts.

---

## R0 — pipeline-fidelity pilot (RUN, PASSED, gates everything else)

**Purpose and destination.** Prove that the W3 code path reproduces the
published M0 pipeline before anything is compared against it. Nothing from this
run is quoted in the manuscript; it is the gate that licenses R1 onward.

**Expected result.** Rung (i) through `scripts/y3_w3_pilot.py` reproduces the
seed-301 per-instance TWT\* of `m0_alone` in `results/y3_p4/m0_gate.csv`
exactly, hence 45.1911%, and the recomputed 10-seed mean equals `\MzeroGain`
= 45.3620. Rung (i-b) reproduces rung (i) to the last bit. **If either fails**,
the harness is measuring itself and no W3 number is launched.

**Result.** Published macro `\MzeroGain` = 45.4% (comment: 45.3620). Recomputed
10-seed mean from `m0_gate.csv`: 45.3620%. Seed-301 published: 45.1911%; mine
45.1911%; difference +0.000000 pp; max |dTWT\*| 9.09e-13 on `m0_alone` and
1.82e-12 on `rule` (both below the CSV's own six-decimal precision). Re-expressed
loop against the shipped one: **max |dTWT\*| = 0, exactly**. PASS.

**Contamination risks checked.**
- *Stale checkpoints*: none exist; the estimator is fitted from scratch inside
  every run. No W3 script reads or writes `results/y3_p4/cache/`.
- *Partially written result files*: every W3 result is written to `path.tmp` and
  `os.replace`d, so a killed run cannot leave a half-file for the analysis to
  sweep in.
- *Train/test leakage*: `build_cell` re-asserts that the eval files are disjoint
  from train+probe, and the runner asserts that the resolved held-out instance
  ids equal the published run's `inst_id` column.
- *Contended machine*: the box is shared. Every W3 job runs under
  `taskset -c 20-23` with `OMP_NUM_THREADS=MKL_NUM_THREADS=1` and
  `torch.set_num_threads(1)`, at most four worker processes, and cores 0-19
  belong to other agents. **No wall-clock figure from any W3 run is reported as
  a measurement.**
- *Thread-count drift*: W2 measured a +1.56 pp shift in the headline from moving
  the estimator from 1 to 4 intra-op threads. Every W3 process is pinned to one.

**Data-accuracy checks performed.**
- The 30 instance files are located with the same glob and the same `sorted()`
  order as `y3_p4_m0grid.locate_files`; the resolved train / probe / eval file
  lists are written into every result file.
- The resolved configuration is diffed field-by-field against the published
  `_base_task(campus=9, u=100, beta=1.0, rho=0.25, seed=301)` dictionary, and the
  run aborts on any difference (16 fields, no difference).
- The comparison target is recomputed from `results/y3_p4/m0_gate.csv`, not
  copied from the macro, and the recomputed 10-seed mean is checked against
  `\MzeroGain`.

---

## R1 — the beta = 0 bias, on the published provenance protocol

**Purpose and destination.** The number the manuscript quotes as
`\BetaZeroHatSMean` = 0.04 comes from `scripts/y3_beta0_check.py`, which trains
at campus 9, u130, beta 0, seed 301 with a REDUCED protocol (12 training
instances, 5 DAgger iterations) rather than the regime map's 16/8. R1 reproduces
that exact protocol so the before/after bias comparison is against the published
number and not against a differently-fitted one. Lands in the before/after bias
table and in the replacement macros for `\BetaZeroHatSMean`.

**Expected result.** The incumbent reproduces mean `hat_s` = 0.0408 and Pearson
-0.0122. The censored rungs move the mean fitted shift materially toward zero,
because the labels the censoring removes are exactly the ones that cannot be
true: on a recorded class-4 order (81% of the population here) a -1 label and a
0 confirmation both become the one-sided statement "s <= 0", which no longer
pulls `mu` toward an impossible target. **If the mean does not move**, the offset
has another source than the boundary labels, and the diagnostic that decides
between them is already built into the run: `mean_applied_shift` (the correction
after the deployment clip) and the per-recorded-class breakdown are reported for
every rung alongside the raw mean, together with `classmean_oracle`, which says
how much a class-level constant can buy at all. That case is reported as a
diagnostic finding, not as a failure, and the manuscript's paragraph is not
rewritten.

**Seeds.** 301 (the published provenance seed), extended to 301-303 for the
e3-protocol figures in R2.

**Contamination risks checked.** As R0, plus: the reduced protocol (12/5) is
declared explicitly in the resolved configuration of every R1 result file, so it
can never be silently pooled with an R2 (16/8) result. The two protocols write
to different files.

**Data-accuracy checks performed.** The instance pool at `c09/storm2/w80/u130`
is located with the published glob; `n_train=12`, `n_probe=4`, `n_eval=10` is
transcribed from `y3_beta0_check.train_hat_s`'s call site, and the run asserts
it.

---

## R2 — the beta = 0 regime-map cells on the primary campus

**Purpose and destination.** The two cells the manuscript quotes on campus 9:
the extreme-overload cell (u130, `\BetaZeroOverloadCnine` = 15%) and the
realistic-band cell (u100, `\BetaZeroBandCnine` = -1.1%). Lands in the
before/after overload table and in task C4 (the rewrite of the Section 6.6
anomaly paragraph).

**Protocol.** The regime map's, transcribed from `y3_p4_m0grid._base_task` and
`tasks_B`: `n_train=16, n_probe=4, n_eval=10, m0_iters=8, rho=0.25`, seeds
301-303, pooled over the 30 (seed x instance) rows exactly as
`y3_beta0_check.pooled_twt` pools them.

**Expected result.** The incumbent reproduces 14.67% at u130 and -1.08% at u100
from `results/y3_p4/e3_map.csv` (recomputed, not copied). If the censored rungs
remove the fitted offset, the overload reduction collapses toward zero, and the
anomaly is explained *and* removed. **If instead it grows**, the offset is a
correct population-level correction that the censored model states more
accurately than the incumbent does — which is also a result, and the honest
reading is then that the beta = 0 column is not an artifact but a real
class-level correction with no per-order content. `classmean_oracle` separates
the two readings: it is the ceiling a class-level constant can reach.

**Contamination risks checked.** As R0, plus: every rung on a cell is diffed
against rung (i)'s resolved configuration and the run aborts unless the only
differing fields are `variant`, `censor_mode`, `deploy` and `fit_sigma`. Each
(cell, variant, seed) writes its own file, so a killed sweep cannot merge a
partial cell into a pooled number.

**Data-accuracy checks performed.** The resolved held-out instance ids are
asserted equal to the `inst_id` column of `results/y3_p4/e3_map.csv` for the
same (campus, u, beta, seed), and the recomputed incumbent per-instance TWT\* is
compared against that file's `rule` and `m0_alone` columns.

---

## R3 — the beta = 0 regime-map cells on campus 10 (gated on R2)

**Purpose and destination.** `\BetaZeroOverloadCten` = 33% and
`\BetaZeroBandCten` = 12.9%, the cells where the offset is largest. Campus 10
instances carry 12,155 work orders against campus 9's 2,955, so these runs cost
roughly five times as much; they are gated on R2 finishing so that only one W3
sweep is ever on the pinned cores at a time.

**Expected result and the opposite.** As R2. The C10 band cell matters most for
the manuscript, because it is the one place a Payoff Condition term does not
bind: at beta = 0 inside the realistic band the layer is still worth 12.9% on
C10. If the censored likelihood removes that, the manuscript loses a
qualification it currently makes; if it keeps it, the qualification is
strengthened and given a mechanism.

---

## R4 — the headline cell: is the realistic band unharmed?

**Purpose and destination.** A censoring model that fixes the boundary bias but
costs deployed performance is a negative result and must be reported as one.
This run re-runs the headline cell (campus 9, storm2 u100, beta 1.0, rho 0.25)
under every rung, seeds 301-305 (the paper's ablation seed range), and reports
the reduction over the tuned rule with the paper's own test.

**Statistics.** The paper's: seed-average each decider's per-instance TWT\*,
then a two-sided paired Wilcoxon signed-rank test (`zero_method='pratt'`) over
the 10 held-out instances, Holm-corrected within the family of contrasts
reported here. Identical to `y3_p4_m0grid._contrast`.

**Metrics.** Pearson r and sign accuracy on non-zero-shift orders (the shipped
recovery probe, on the 4 probe instances); Kendall tau-b between the corrected
and the true ranking at decision points, computed on the 10 held-out instances
at a COMMON reference trajectory (the plain ATC rollout) so the metric compares
rankings and not trajectories; and the deployed reduction in true weighted
tardiness against the tuned rule, scored by the existing independent validator
(`hitl.true_objective.score_true`), which is the sole referee.

**Expected result.** The headline reduction is unchanged within noise (the
published seed spread `\MzeroGainStd` is 1.8 pp, and W2 measured a 1.1 pp
seed sd for this rung), because at beta = 1 most of the recoverable signal is
per-order and the censoring only re-reads boundary labels. **If it falls
materially**, the censored likelihood buys an unbiased boundary at the cost of
deployed performance, the package's recommendation becomes "report as a variant,
do not replace", and the loss is reported as measured.

---

## R5 — reduction-anchor re-verification

**Purpose and destination.** Part D item 6 of the submission plan: the
bit-for-bit reduction anchor is re-verified after every code change. With the
supervisor disabled and the deadline-aware gate closed, the environment must
reduce exactly to the public benchmark's dispatcher. Lands in the W3 report and
in the submission gate's "anchor re-verified on final code" checkbox.

**Expected result.** `scripts/y3_e0_anchor.py` exits 0. **If it does not**, the
W3 code has touched shared state and every W3 number is void until it is fixed.

**Note.** W3 adds files only; it edits nothing under `paper/`, nothing existing
under `src/` or `scripts/`, so the anchor cannot change unless an import
side-effect exists — which is exactly what this run rules out.

**Result.** `scripts/y3_e0_anchor.py` exits 0 on the final W3 code
(`results/y3_w3/e0_anchor.log`): the RULE three-way exactness holds on all 48
(campus x size x rule) triples, policy replay is deterministic on 4 campuses,
the M1 latent head is bit-exact at gate 0, and `tests/test_env.py` passes.
**E0 ANCHOR: PASS.** `tests/test_censored.py` (15 tests) passes, as do
`test_m0_estimator`, `test_latent_head`, `test_no_leak` and
`test_choice_estimator` (43 tests together). Eleven shipped test files
(`test_env`, `test_overlay`, `test_supervisor`, ...) fail to *collect* under a
bare `pytest tests/` because they take a script-style `failures` fixture and are
written to be run as programs; this predates W3, those files are untouched, and
the anchor runs `test_env.py` the intended way, where it passes.

---

# Addendum, written after the runs

Everything below is a record of what actually happened, including runs not in
the plan above and every place the plan was deviated from.

## Deviations from the plan

1. **`tobit_sig` (fitted sigma) was dropped from the campus-10 sweeps (R3) and
   from the headline sweep (R4).** It is carried at full seed depth only on the
   campus-9 beta = 0 cells, where it answers the question it exists to answer
   (does the fixed scale drive the result?). Extending it everywhere would have
   put a second W3 sweep on the pinned cores, which the contamination note
   forbids. Its campus-9 result is reported in full.
2. **`classmean_oracle` is reported on the beta = 0 cells only.** At beta = 1 a
   class-level constant is not the relevant ceiling, and the run would answer no
   question.
3. **The campus-10 realistic-band cell (`c10_u100_b0`) was cut to four rungs
   (`mse_published`, `tobit`, `tobit_imp`, `classmean_oracle`) and run with
   Kendall tau switched off.** A campus-10 instance carries 12,155 work orders
   against campus 9's 2,955, and the rank metric alone costs about ten minutes
   per rung there. Kendall tau is recorded as *not computed* on that cell rather
   than silently defaulted; the before/after TWT\* comparison, which is what the
   cell exists for, is unaffected.
4. **The mechanism diagnostic (R6b) ran with one worker alongside the
   campus-10 sweep**, five processes on the four pinned cores rather than four.
   Every job is single-threaded with fixed seeds, so oversubscription changes
   wall-clock and nothing else; no timing from any W3 run is reported.
5. **`mse_reexpr` was run at seed 301 only** on the R2 cells, after the pilot had
   already established the bit-exact reduction at the headline cell. It
   reproduces `mse_published` with max |dTWT\*| = 0 at every cell where it ran.

## R6 — the constant-correction sweep (added; not in the plan above)

**Purpose and destination.** R1 and R2 produced a result the plan did not
anticipate: the exactly-correct class-level constant (`classmean_oracle`) buys
essentially nothing at the beta = 0 overload cell, while the incumbent's much
smaller offset buys 14.7%. That is incompatible with the manuscript's stated
mechanism, so the mechanism had to be measured rather than argued. This run
sweeps a constant `hat_s` -- uniform, class-4-only, and scaled multiples of the
true class-mean effective shift -- through the deployed decider on the held-out
instances, fitting nothing. It lands in the report's mechanism paragraph and in
task C4. `scripts/y3_w3_constsweep.py`.

## R6b — the fitted-map decomposition (added; not in the plan above)

**Purpose and destination.** R6 showed that no constant does anything, which
locates the beta = 0 reduction in the *variation* of the fitted map rather than
its level, but does not say whether that variation is information-free. This run
takes the FITTED incumbent map and degrades it one way at a time without
refitting: replaced by its class means, replaced by its global mean, centred,
shuffled across orders, and replaced by a moment-matched Gaussian. It decides
which sentence the manuscript is allowed to write about the beta = 0 column.
`scripts/y3_w3_mechanism.py`. One permutation seed (7) is used for the shuffled
and Gaussian arms; the effects are large enough (+14.7% against -4.6% and
-13.0%) that a single draw settles the direction, and this is stated rather than
hidden.

## R7 — reproducing the published `\BetaZeroHatSMean` (added)

**Purpose and destination.** The W3 harness reproduced every
`results/y3_p4` number exactly but returned mean `hat_s` = 0.019 where
`paper/macros.tex` quotes 0.041. The cause was found rather than assumed:
`scripts/y3_beta0_check.py` sets `torch.set_num_threads(4)` at import, while
every `results/y3_p4` number was produced at one thread, and W2 had already
measured that the intra-op thread count moves the estimator's fit. This run
refits the published protocol at 1 and at 4 threads. At 4 threads it reproduces
0.040787 exactly. `scripts/y3_w3_threadcheck.py`.

## Compute note

The box is shared with other agents; the load average ran between 15 and 41 on
24 cores throughout, with an unpinned training job of another agent at roughly
1,300% CPU. Every W3 job ran single-threaded under `taskset -c 20-23`, with at
most four worker processes (five during R6b, see deviation 4). **No wall-clock
figure from any of these runs is a measurement**, and none is reported as one.
