# W10 / W11 run plan: is the correction layer's recipe load-bearing?

Runner: `scripts/y3_w10_w11_variants.py`
Output: `results/y3_w10_w11/` (`folds/`, `cache/<variant>/`, `preflight.json`,
`config_diff.json`, `summary.json`, `*.log`)

Written before the pilot fold was launched. The pilot timing in section 5 was
filled in from the pilot, which ran after this file was committed and before the
rest of the folds.

Two experiments, one runner, because they ask the same kind of question about the
same two cells: **which parts of the shipped correction layer actually carry the
result, and which are just the settings we happened to pick?**

The shipped layer is a 1,761-parameter neural estimator fitted to weak labels
harvested from supervisor overrides, in which an override counts five times as
much as a confirmation, and whose predictions re-score a tuned dispatching rule
under a conformal band and stability-based review routing.

* **W11** varies the override up-weight over {1, 2, 5, 10}. Five is the shipped
  setting, and the method section currently discloses it as a setting we chose
  rather than tuned, with no sensitivity sweep behind it.
* **W10** replaces the neural estimator with gradient-boosted regression trees
  under fixed, pre-stated hyperparameters, and changes nothing else.

Cells (world 12345, training seeds 301, 302, 303, review budget rho = 0.25,
deployable stability routing, `split_fit=True`):

| Cell | Campus | Utilisation | Recoverable share beta | Supervisor noise eps | Why |
|---|---|---|---|---|---|
| A | 9 | 100 | 1.00 | 0 | headline ceiling |
| B | 9 | 100 | 0.20 | 0 | corpus-anchored share |

The incumbents are the already-computed `results/y3_w9/folds/{A,B}_*world12345*`
folds. **They are not rerun.** For the override weight 5 arm of W11 the
incumbent is instead *reproduced* on one fold and asserted bit-equal, which is
the verification that the weight-5 column of the sweep and the incumbent are the
same computation (see section 3).

---

## 1. Purpose, and where the numbers land

| Question | Experiment | Where the number lands |
|---|---|---|
| Does the result depend on how heavily an override outweighs a confirmation? | W11 | One sentence in the method, replacing the current "chosen rather than tuned, with no sensitivity sweep" disclosure; one variant row (four weights x two cells) in the robustness table |
| Does the result depend on the neural estimator specifically? | W10 | One or two rows of the same table, plus one clause in the discussion of what the correction layer is |

Deciders come out of one evaluation call per fold: the tuned rule (RULE), the
correction layer alone (M0), the myopic full-information reference
(ORACLE-GREEDY), and the two in-loop arms (RULE+SUP, M0+SUP). The reported
headline per fold is the percentage reduction in true weighted tardiness of M0
against RULE on the ten held-out instances, plus the two recovery numbers the
paper already reports for the estimator (Pearson correlation between the
estimated and the true class shift, and sign accuracy on orders whose true shift
is nonzero).

Neither experiment produces a wall-clock claim. Nothing in this run is reported
as a timing measurement.

## 2. Expected result, and what the opposite would mean

**W11, override up-weight.** Expected: the reduction against the tuned rule is
close to flat over weights 2, 5 and 10, and possibly lower at weight 1, where a
confirmation counts as much as an override and the large mass of confirmations
(zero-labelled examples) pulls every prediction toward zero. If that is what
comes back, the disclosure is upgraded from "we did not tune it" to "we did not
tune it, and it does not matter over an eightfold range", which is strictly
better for the reviewer.

If instead the reduction moves strongly with the weight, the 5x choice was
load-bearing, and the paper must say so: the sweep then goes into the paper as a
sensitivity result with the shipped value identified as one point on it, and the
method sentence has to concede that the setting matters and was not selected on
held-out data. That outcome is publishable and would be reported; it is the
reason the sweep is worth running.

A third possibility is that weight 1 collapses the layer entirely (no gain, or a
loss). That would be the most informative outcome of the three, because it names
the mechanism: the value comes from the override signal, and confirmations are
ballast that has to be down-weighted for the estimator to see anything.

**W10, gradient-boosted trees.** Expected: parity, or near-parity, with the
neural estimator. The correction layer's claim is about the *pipeline* (weak
labels from overrides, a DAgger loop, a conformal band on absolute errors, and
stability routing), not about a particular function approximator, so an
independent estimator family reaching a similar reduction says the idea is
estimator-agnostic and the specific network is an implementation detail.

If the trees fall well short, the claim narrows and must be narrowed in the
paper: the pipeline as shipped depends on the smooth, low-capacity neural
estimator, and the replaceability sentence is deleted rather than softened. If
the trees do substantially *better*, that is also reported as-is; it would say
the shipped estimator is leaving something on the table, and it becomes a stated
limitation rather than a silent one.

Both directions have a home in the paper, so the run answers its question either
way.

## 3. Contamination

- **No checkpoint is reused anywhere.** Every fold fits its estimator from
  scratch inside `routing.run_m0_routed`; there is no saved model in this
  pipeline to resume from.
- **No published cache is read or written.** `y3_w1_sweep.evaluate_cell` is
  called verbatim with its module-level `_CACHE` redirected to
  `results/y3_w10_w11/cache/<variant>/`, one directory per variant. The runner
  asserts at start-up that the redirected path is neither `results/y3_w1/cache`
  nor `results/y3_w9/cache`.
- **The variant is inside the cache key.** The override weight and the estimator
  family are *not* fields of `y3_w1_sweep._SIG_KEYS`, so four override weights at
  the same cell and seed produce the *same* signature. Two independent defences:
  the per-variant cache directory above, and a variant-extended fold key
  (`sha1(cell signature + variant descriptor)`) in every fold filename. The
  runner asserts the four weights yield four distinct fold keys before running
  anything.
- **Stale folds cannot be mixed in.** A fold file carries its variant-extended
  key in its name. A file for the same (variant, cell, seed) under a different
  key means the configuration changed since it was written, and the run aborts
  rather than mixing them.
- **Atomic writes.** Each fold is written to `.part` and `os.replace`d, so an
  interrupted run leaves only complete files and resumes by skipping them.
- **The incumbent is not rerun and not overwritten.** `results/y3_w9/` is opened
  read-only, for the incumbent figures and for the bit-equality gate below.
- **Weight-5 bit-compatibility is verified, not assumed.** The runner recomputes
  cell A, seed 301 at override weight 5 through its own wrapped code path and
  requires every one of the fifty per-instance true-weighted-tardiness values
  (five deciders x ten held-out instances) to equal the committed W9 fold
  exactly. A single differing value aborts. Only after that gate passes are the
  W9 numbers used as the weight-5 column; the remaining five weight-5 folds are
  then read from W9 rather than recomputed.
- **Machine.** Cores 0-9, one numeric thread per process, hard-set before numpy,
  torch or sklearn is imported, and re-asserted inside every worker (including
  `sklearn.utils._openmp_helpers._openmp_effective_n_threads() == 1`). The
  machine is idle, but no wall-clock number from this run is reported as a
  measurement of anything.

## 4. Data accuracy

- **Locked constants are imported, not retyped.** The runner imports the locked
  configuration tables from `scripts/y3_w9_overlay_worlds.py` and calls its
  `assert_locked_config()`, so the resolved default task, the cache-signature key
  list and the overlay constants are checked field by field against the same
  table the incumbent was produced under. Any drift aborts.
- **Same instances.** Both cells resolve to the same 30-instance pool as the
  incumbent, split `files[0:16]` train, `files[16:20]` probe, `files[20:30]`
  held-out. The runner asserts the held-out slice is disjoint from train and
  probe, digests each held-out file, and asserts the ten held-out ids equal the
  published set recorded in
  `results/y3_p5/harvest/primary_multiseed_summary.json`.
- **Same hidden world.** The digest of `results/y3_p1/overlay_coeffs/F-NL_seed12345.json`
  is asserted equal to the value recorded in `results/y3_p9/data_checks.json`.
- **Config diff against the incumbent = the one intended change.** For every
  variant the runner builds the fully resolved configuration (all 21 task fields
  plus the estimator-fitting settings: estimator family, architecture or
  hyperparameters, override weight, confirmation weight, DAgger iterations,
  calibration fraction, conformal level, band mode, review policy) and diffs it
  against the incumbent's. It aborts unless the diff is exactly
  `{override_weight}` for W11 and exactly the estimator block for W10. The diff
  is written to `config_diff.json`.
- **The weight is threaded, and proved to be.** `routing.weak_labels_from_log` is
  wrapped in a pass-through that records the override weight it was actually
  called with and the distinct sample weights it returned. A fold aborts unless
  every call received the requested weight and the returned weights lie in
  {requested weight, 1.0}.
- **The estimator cannot drift.** For W11, `routing.run_m0_routed` is wrapped in
  a pass-through asserting the fitted estimator has exactly 1,761 parameters on
  every fit, as in W9. For W10 the same wrapper instead asserts the fitted object
  is the tree estimator and records the number of trees actually fitted.
- **Review placement cannot drift.** `routing.make_supervisor` is wrapped to
  assert a `StabilityRoutingSupervisor` was built and `policy == "stability"`
  was requested, in training and in evaluation. A fold whose record carries no
  conformal band aborts.

## 5. What is deliberately fixed, and the two things that cannot be identical

**W10 hyperparameters, stated before any result was seen and never revisited.**
`sklearn.ensemble.HistGradientBoostingRegressor` with `max_iter=200`,
`learning_rate=0.1`, `max_depth=None`, `max_leaf_nodes=31`,
`min_samples_leaf=20`, `l2_regularization=0.0`, `early_stopping=False`,
`random_state=seed+iteration`, fitted with `sample_weight` carrying exactly the
weak-label weights the neural estimator receives. Everything else in these are
scikit-learn 1.9.0 defaults. `early_stopping=False` is chosen so that the fit
uses every proper-training example (as the neural estimator does), takes no
internal validation split, and is not a disguised model-selection loop. **There
is no tuning loop of any kind, and no hyperparameter was changed after a result
was seen.**

Two things about W10 cannot be made identical to the shipped layer, and are
reported rather than hidden:

1. **Cold start.** The neural estimator begins each run at random initialisation,
   so its first DAgger iteration runs a slightly perturbed rule. An unfitted tree
   ensemble has no such state; the wrapper predicts exactly zero until the first
   fit, so the first iteration is the plain tuned rule. This is the natural
   analogue but it is not the same starting point.
2. **Warm start.** The neural estimator is warm-started: each iteration continues
   training the same weights on the grown aggregate. The tree ensemble is refit
   from scratch on the grown aggregate each iteration. Continuing to add trees to
   an ensemble built on a different dataset is not the standard use of the
   method, so refitting is the faithful reading of the protocol's "retrained on
   the full aggregate each iteration", but it is a difference.

`scikit-learn==1.9.0` is appended to the pinned pip list in `environment.yml`,
because it becomes a dependency of a reported variant.

**Schedule.** 18 W11 folds (weights 1, 2, 10 x two cells x three seeds), 1 W11
weight-5 reproduction gate fold, and 6 W10 folds: 25 folds in total. One pilot
fold per experiment runs first, in the foreground, and its wall time is printed
before the rest are launched.

Pilot timing, filled in after the pilots ran. Weight-5 reproduction-gate fold
(cell A, seed 301) 38 s, gate passed bit-exactly. W11 pilot (cell A, seed 301,
override weight 1) 36 s. W10 pilot (cell A, seed 301, trees) 40 s. The remaining
22 folds ran on six workers pinned to cores 0-9 and finished in 154 s. These are
scheduling estimates, not measurements; nothing here is reported as a timing
result.
