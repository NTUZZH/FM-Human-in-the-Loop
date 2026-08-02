# W9 run plan: the key cells across ten independent supervisor-overlay worlds

Runner: `scripts/y3_w9_overlay_worlds.py`
Output: `results/y3_w9/` (`folds/`, `cache/`, `preflight.json`, `sweep.log`)
Written before the sweep was launched. Pilot timing and the schedule estimate
were filled in from the pilot fold, which ran before the sweep.

---

## 1. Purpose, and where the numbers land

Every cell published so far draws the hidden supervisor function `f` and the
per-order latent noise `z` from a single overlay master seed, 12345. The ten
seeds the manuscript reports vary training and evaluation randomness *inside*
that one simulated world. Nothing published so far separates "the correction
layer works" from "the correction layer works against this one draw of the
hidden function", which is the largest remaining generalizability risk in the
paper.

This run repeats the key cells over ten independent overlay worlds (master seeds
12345 and 20001..20009), three training seeds each, under the shipped deployable
review protocol.

Destination of the numbers:

| Question | Cells | Where it lands |
|---|---|---|
| Does the headline reduction survive a new hidden function? | A | Headline claim, restated as a mean over worlds with a between-world interval; the robustness table and its figure |
| Does it survive at the corpus-anchored recoverable share, and in between? | B, C | The anchored-share claim and the share ladder |
| Is the layer inert where the theory says it must be? | D, E | The boundary claim |
| Is the beta = 0 anomaly a single-world artifact, a persistent mechanism, or noise? | F, G | The anomaly paragraph and its verdict |
| Does a supervisor that errs change the picture? | H | The supervisor-noise robustness claim |

Deciders per fold: the tuned rule (RULE), the correction layer alone (M0), the
myopic full-information reference (ORACLE-GREEDY), and both in-loop arms
(RULE+SUP, M0+SUP). All five come out of one evaluation call, so the in-loop
arms cost nothing extra.

The end-to-end learner is **not** rerun across worlds. That limitation is stated
in the manuscript rather than hidden: the cross-world evidence covers the
correction layer, the tuned rule and the reference, and does not cover the
end-to-end policy.

## 2. Expected result, and what the opposite would mean

**Headline cells (A, B, C).** Expected: the reduction against the tuned rule
stays clearly positive in every world, with a between-world spread that is
visible but does not cross zero, and a mean close to the published
single-world figure. If instead the mean over worlds fell well below the
published figure, or some worlds turned negative, the conclusion would be that
the published number is a favourable draw of the hidden function; the claim
would then be restated as a distribution over worlds, with the published cell
identified as one draw within it. Either outcome is reportable, and the second
one is exactly what this run exists to detect. It is the published number that
would change, not the method.

**Inert-boundary and anomaly cells (D, E, F, G).** At beta = 0 the per-order
latent is pure noise, so nothing about an individual order is recoverable and
the layer should do nothing. A nonzero effect was nevertheless measured on
campus 10 and in overload columns in the published world. Three outcomes, all
publishable:

1. *Single-realization artifact.* The mean effect over ten worlds converges to
   zero and the published world sits in the tail. Reported as such; the
   boundary claim strengthens.
2. *Persistent mechanism.* The effect keeps the same sign in most or all
   worlds. This is the clip-asymmetry explanation already recorded in
   `results/y3_p5/beta0_check.json`: `c* = clip(c - s, 1, 4)` bounds the
   extreme recorded classes on one side only, which leaves a class-level
   constant bias present at every beta, including zero, and under overload even
   a constant learned correction reorders enough to matter. Cross-world
   persistence would confirm that account and turn an anomaly into a stated,
   explained boundary behaviour.
3. *Noise.* Mean near zero with large between-world variance. Reported as not
   separable at this sample size (ten worlds, three seeds).

Because all three outcomes have a home in the paper, the run is worth its
compute whichever way it goes.

**Noisy-supervisor cell (H).** Expected: the same qualitative picture as cell B
with a smaller reduction. The opposite, a reduction that survives error better
across worlds than within one, would say the published cell understates the
layer's robustness to supervisor error.

## 3. Contamination checks

- **No published cache is read or written.** `y3_w1_sweep.evaluate_cell` is
  called verbatim with its module-level `_CACHE` redirected to
  `results/y3_w9/cache`, so every fold genuinely recomputes.
- **The master seed is inside the cache key.** The runner asserts
  `y3_w1_sweep._SIG_KEYS` field for field against an explicit list and aborts if
  `master_seed` is absent. Without that assertion, ten worlds could silently
  collide on one cache entry and return the published world ten times.
- **Stale folds cannot be mixed in.** A fold file is named with its cache
  signature. A file for the same (cell, world, seed) under a *different*
  signature means the configuration changed since it was written, and the run
  aborts rather than mixing the two.
- **Atomic writes.** Each fold is written to `.part` and `os.replace`d, so a
  killed run leaves complete files only, and the sweep resumes by skipping them.
- **No overlay-coefficient write race.** `overlay.get_coeffs` records a new
  world's coefficients with a plain write. All ten worlds are therefore built
  serially in the parent, before any worker starts, and each is rebuilt from
  scratch and compared against its record. Ten distinct file digests are
  asserted, and every new world is asserted to differ from the published one.
- **The published world is still the published world.** The digest of
  `results/y3_p1/overlay_coeffs/F-NL_seed12345.json` is asserted equal both to
  the value recorded in `results/y3_p9/data_checks.json` and to the constant in
  the runner.
- **Reproduction gate.** Cell A at master seed 12345 is the manuscript's
  deployable headline cell, so its cache signature coincides with a committed
  `results/y3_w1/cache` record. Those three folds are recomputed here and every
  per-instance true weighted tardiness must equal the committed record exactly;
  a single differing value aborts the sweep. This is the proof that the W9 code
  path is the published protocol and not a lookalike.
- **Review placement cannot drift.** `routing.make_supervisor` is wrapped in a
  pass-through that asserts a `StabilityRoutingSupervisor` was built and that
  the requested policy is `stability`, in training and in evaluation. A fold
  whose record carries no conformal band aborts.
- **Model cannot drift.** `routing.run_m0_routed` is wrapped in a pass-through
  that asserts the fitted estimator has 1761 parameters on every fit.
- **Only two things vary.** Every field of the resolved task is compared against
  the locked default of `y3_w1_sweep._base_task()`; only campus, utilisation,
  beta, eps, training seed and overlay master seed may differ.

## 4. Data-accuracy checks

- Four instance pools are used: c09/storm2/w80 at u = 100 and u = 130, and
  c10/storm2/w80 at u = 100 and u = 130. Each is asserted to hold exactly 30
  files, split `files[0:16]` train, `files[16:20]` probe, `files[20:30]`
  held out, the same slicing the published cells use. Overlap between the
  held-out slice and train or probe aborts the run.
- The c9 u100 held-out identifiers are asserted equal to the published set in
  `results/y3_p5/harvest/primary_multiseed_summary.json`
  (`c09_storm2_w80_u100_0020` ... `_0029`).
- The SHA-256 of every held-out instance file, of every overlay coefficient
  file, and the per-instance order counts are recorded in
  `results/y3_w9/preflight.json` before the sweep starts.
- Instance sizes, recorded rather than assumed: 2253 orders per c9 u100
  instance, 2955 for c9 u130, 9155 for c10 u100, 12155 for c10 u130. The campus
  10 cells are the expensive ones and are scheduled accordingly.

### What an "independent world" is, verified rather than assumed

Read from `src/fmwos/hitl/overlay.py` and confirmed by direct measurement. The
master seed feeds three separate draws:

- `stable_seed("lin", master_seed)` draws the linear coefficients of the hidden
  function `f`;
- `stable_seed("nl", master_seed)` draws F-NL's four sparse interactions;
- `stable_seed("z", master_seed, instance_id)` draws the per-order latent noise
  `z`.

So a new master seed changes **both** the hidden function and the latent noise;
the noise is not a function of order identity alone. Measured on one c9 u100
held-out instance, world 12345 against world 20001: at beta = 0 the two latent
draws correlate at 0.0001 and 74% of class shifts differ; at beta = 1 they
correlate at -0.26 and 78% differ.

One qualification, stated because it changes what "independent" means. The
feature basis and its standardization are computed over the fixed
training-campus order population and are therefore identical in every world.
Two worlds' hidden functions are two random coefficient vectors in the same
20-dimensional basis, so they are uncorrelated in expectation but not
orthogonal by construction; a realized pair correlates at order
1/sqrt(dimension), which is what the -0.26 above reflects. Averaging over ten
worlds averages over exactly that draw. What this design does *not* vary is the
functional form of the supervisor's private information: every world is an F-NL
function of the same campus-agnostic features. Cross-world robustness therefore
means robustness to the draw of the hidden function, not to a change in what
kind of function it is.

## 5. Compute discipline

- Ten single-threaded worker processes, pinned with `taskset -c 0-11` on a
  24-core machine that is otherwise idle.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` are hard-set to 1 before
  numpy or torch is imported, not `setdefault`ed: `y3_w1_sweep` setdefaults them
  to 4, and four threads per worker would put forty threads on twelve cores and
  would also change the floating-point reduction order, breaking the bit-exact
  reproduction gate. Every worker re-asserts all five variables and
  `torch.get_num_threads() == 1`.
- The sweep runs detached under `nohup` so it survives the launching session.
- **No wall-clock figure from this run is a measurement of anything.** Times are
  recorded only to schedule the sweep. The field in each fold file is named
  `wall_s_not_a_measurement` for that reason.

## 6. Scale, timing and scope

240 folds: 8 cells x 10 worlds x 3 seeds. The published world's folds are
**not** reused from existing results. Only cell A at world 12345 has a committed
counterpart at all; recomputing those three folds costs minutes and buys the
reproduction gate in section 3, and the other seven cells have no committed
counterpart under this protocol at world 12345.

Two folds were run before the sweep to size it, both single-threaded on the
otherwise idle machine, and both are kept as real folds of the sweep:

- pilot fold, cell A (c9 u100), world 20001, seed 301: **37.0 s**;
- cost probe, cell G (c10 u130), world 20001, seed 301: **1458 s**.

Campus 10 costs about forty times campus 9 per fold, which is what the order
counts predict: 12155 orders per c10 u130 instance against 2253 for c9 u100,
and cost grows faster than linearly because the correction layer trains on
sixteen instances of that size for eight rounds. Scaling the two measured folds
by instance size gives the schedule:

| Cells | Folds | Per fold | Core-hours |
|---|---|---|---|
| A, B, C, D, H (c9 u100, 2253 orders) | 150 | ~40 s | 1.7 |
| E (c9 u130, 2955 orders) | 30 | ~55 s | 0.5 |
| F (c10 u100, 9155 orders) | 30 | ~1100 s | 9.2 |
| G (c10 u130, 12155 orders) | 30 | ~1460 s | 12.2 |
| **Total** | **240** | | **~24** |

At ten workers that is roughly **2.5 to 3 hours** of wall time, the tail set by
the sixty campus 10 folds. This is comfortably inside the 20-hour ceiling, so
**no scope reduction is taken**: all eight cells run over all ten worlds with
three seeds each. Had the estimate exceeded 20 hours, the agreed order of
reduction was to drop cell G first, then restrict the campus 10 cells to worlds
20001..20006.

These figures schedule the sweep and are nothing else. They were taken one fold
at a time on an idle machine, but the sweep itself runs ten workers on twelve
cores, so its per-fold times are contended and are not comparable to them.

## 7. The one respect in which this is not the published protocol

Cells A, B, C and H are measured under exactly the protocol that produced the
manuscript's deployable numbers, and the reproduction gate proves it for cell A.

The beta = 0 anomaly, however, was first reported from the E3 regime map in
`results/y3_p4/e3_map.csv`, which used the *published* review placement
(`mechanism="targeted"`, fitting on the whole weak-label aggregate), not the
shipped deployable routing. W9 re-measures cells D, E, F and G under the
deployable routing, because a robustness sweep must run the protocol the paper
ships, not a lookalike. The instance pools, slices,
training seeds and held-out counts are identical to E3, so the only difference
is review placement, and review placement reaches the layer-alone arm only
through which weak labels its training generated: at evaluation time the M0
arm consults no supervisor at all. The cross-world anomaly result must
therefore be reported as measured under the deployable protocol, and its
comparison with the E3 number as a comparison across two review policies, not
as a like-for-like reproduction. A published-protocol arm was not run, to keep
the master seed and the cell parameters the only varied inputs.

One further difference from a previously published campus 10 run: the W1
routing curve evaluated campus 10 on eight held-out instances to save time. W9
uses ten everywhere, matching the E3 map and the campus 9 headline, so the
campus 10 folds in `results/y3_w1/cache` are not comparable to these and are
not reused.
