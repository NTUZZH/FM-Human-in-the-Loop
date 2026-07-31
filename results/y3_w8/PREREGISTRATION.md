# Pre-registration: W8 practitioner urgency-pairs pilot

**Registered 2026-07-31, before any response exists.** No practitioner has been
invited, no response has been collected, and no statistic in this document has
been computed on response data. The only numbers below that come from data are
properties of the instrument itself, which was built from the public corpus and
frozen before this document was written.

**Instrument frozen at:**

| artefact | SHA-256 (first 16) |
|---|---|
| `pilot/y3_w8_manifest.csv` | `7b8e1fcefce96851` |
| `pilot/y3_w8_pilot.html` | `e9f6b55c46cc7143` |
| `pilot/y3_w8_items.json` | `5f478855c097c387` |

Rebuilding the instrument changes these hashes and invalidates this registration.
If the instrument must be rebuilt, record the reason in section 12 and re-register.

The analysis that implements this document is `scripts/y3_w8_pilot_analyse.py`. It
was written and tested against synthetic fixtures (`results/y3_w8/selftest/`, all
machine-generated, none of it data) before this registration, so no analytic
choice can be made after seeing a real response.

---

## 1. What the pilot establishes

The manuscript models a supervisor who holds an urgency the recorded priority
class does not capture, treats that urgency as partly a function of observable
order attributes (a share the paper calls $\beta$), and observes it through noisy
corrections (a noise level $\epsilon$). The pilot grounds two things:

1. **Practitioners agree with each other about relative urgency more than
   chance.** This is what makes a hidden but shared urgency a coherent notion at
   all, and it bounds the noise level.
2. **Their judgements are partly predictable from the observable order attributes
   the estimator uses.** This is the real-data analogue of $\beta > 0$.

## 2. What the pilot cannot claim

> The pilot asks practitioners to rank pairs of real work orders drawn from the
> same corpus the study models; it does not place a practitioner inside the
> dispatch loop. It therefore grounds two premises of the supervisor model, that
> an urgency ordering exists which practitioners share beyond the recorded
> priority class and that it is partly predictable from the observable order
> attributes the estimator reads, and it grounds nothing else. It does not
> validate the correction loop, the dispatching results, or any reported
> reduction in true weighted tardiness, every one of which is measured against a
> simulated supervisor.

This sentence travels with the numbers wherever they are quoted, including in any
response to reviewers.

## 3. The instrument, as frozen

**Corpus and population.** FMUCD (Mendeley Data, CC BY-NC), cleaned by the
paper's own pipeline (`fmwos.io.load_raw` then `fmwos.io.clean`: rules R1, R2, R3,
R7, R4, R6), classed by the paper's v2 priority mapping (`fmwos.calib`, rules R5a
to R5d) restricted to the six campuses `{1, 2, 5, 9, 10, 12}` the study uses, with
the same MISC trade merge and the same business-hour time axis. 3,731,442 raw rows
become 1,454,039 work orders; 1,449,262 of those are on the six campuses;
1,325,805 additionally carry a readable description, a component, and a system,
and form the item pool.

**Items.** 50 unique pairs, 56 presented items. Each item shows two real orders
sitting in the same campus's queue at the same moment, and asks which should be
started first, plus a three-point confidence and an optional one-line reason.

**Fields shown**: the free-text description, the system and component, a building
descriptor where the corpus records one, the recorded priority class with its
target response time, days waited, and the estimated labour content.

**Fields withheld**: realised duration, cost, and close-out date, which are
outcomes rather than inputs to a dispatch decision.

**One substitution, recorded here.** FMUCD's labour hours are recorded at
close-out, but the simulation treats the processing time as known at dispatch and
the estimator reads job size as one of its three inputs. The item therefore shows
labour hours labelled as an estimate. This matches the simulation's information
set and is stated rather than hidden.

**Sampling rule.** Seed 908. Per campus, every weekday-08:00 anchor in the
campus's span is a candidate dispatch moment, shuffled once; at an anchor the
backlog is every eligible order released in the preceding 120 business hours (15
business days), anchors with fewer than 10 backlog orders are skipped, and up to
25 pairs are drawn per accepted anchor. Waiting time is the anchor minus the
release, so both members of a pair are genuinely queued together and their waits
are real (realised range 0.0 to 15.0 business days). The per-stratum quota is then
filled greedily under three constraints: an order appears in at most one pair, a
normalised description appears in at most one pair, and the campus with the fewest
pairs so far is preferred at each draw.

**Strata**, assigned with priority S3 > S1 > S2 > S4, realised counts as targeted:

| stratum | what it is | target | realised |
|---|---|---|---|
| S1 | recorded classes equal | 18 | **18** |
| S2 | one class apart, record and attributes agree | 14 | **14** |
| S3 | record and attributes disagree | 12 | **12** |
| S4 | two or more classes apart, agreeing | 6 | **6** |

"What the attributes suggest" is an ordinary least-squares fit of the recorded
class on the paper's own campus-agnostic features (merged trade one-hot,
log1p labour hours, release day-of-week one-hot) with a per-campus intercept,
fitted on all 1,325,805 pool orders ($R^2 = 0.191$). A pair is S3 when the
recorded classes differ and the fitted ordering runs the other way by at least
0.25 class units. The fit reads no response and no latent quantity.

**Realised composition.** Pairs per campus 9/9/8/8/8/8 for campuses 1/2/5/9/10/12.
Recorded classes across the 112 shown orders: 8 P1, 20 P2, 37 P3, 47 P4. 18 items
carry a building descriptor (campuses 1 and 12 are the only two of the six for
which FMUCD records one).

**Repeats and counterbalancing.** Six pairs are shown twice, sides swapped and at
least 15 items apart. Across the 50 unique items the canonical first order sits on
the left in exactly 25, and among the 32 class-differing items the more urgent
recorded class sits on the left in 16, so neither the pair's internal labelling nor
the recorded class can be read off the side.

## 4. Participants and sample size

**Target 5 practitioners; minimum 3; maximum 8.** Anyone who dispatches or plans
maintenance work qualifies. Each rater sees all 56 items.

The pairs, not the raters, are the statistical sample: the design draws 50 pairs
by a stated rule from a corpus of 1.3 million orders, and the bootstrap intervals
resample pairs. Raters are a small convenience sample whose number sets how
precisely the consensus is estimated, not how many independent observations exist.
This is why the plan tops out at 8 rather than pursuing a larger panel.

**Stopping rule.** The analysis is run **once**, after the earlier of: 8 completed
responses, or four weeks from the first invitation. No interim analysis is run,
and no recruitment decision is made on the basis of results. If fewer than two
responses arrive, the pilot is reported as not run and the manuscript's existing
limitation paragraphs stand unchanged.

**Handling.** Responses are anonymous. Only a participant code, a self-described
role, and years of experience are recorded. The code-to-person mapping stays with
the author and never enters the repository.

**Two disclosures.** First, the six repeated pairs are not announced to
participants, because announcing them would defeat the test-retest check they
exist for; this is standard practice for a reliability measure and adds no risk,
since a participant who notices a repeat has simply answered the same question
twice. Second, the study asks practitioners for professional judgement on public,
already-published records and collects no personal or sensitive data, but whether
it needs institutional ethics review or an exemption is a question for the
author's institution and must be settled **before the first invitation is sent**.

**Any correspondence about the study says only what the recruitment note says.**
No participant is told what the analysis expects to find, and no participant sees
another participant's answers, because either would create the agreement the study
is trying to measure.

## 5. Hypotheses and the exact statistics

Responses are recoded, before any statistic, from left/right to the identity of
the chosen order, so presentation position cannot enter any test. Agreement and
predictability use the **first** presentation of each pair only; the second
presentations are used only for the instrument check.

### H1. Practitioners share an urgency ordering

- **Statistic**: Krippendorff's $\alpha$ with the nominal difference function over
  the 50 unique pairs.
- **Why this statistic**: the design is fully crossed, but a rater may skip items,
  the rater count is small and may differ across items, and $\alpha$ is defined for
  any number of raters without discarding incomplete items. Fleiss' $\kappa$, mean
  pairwise Cohen's $\kappa$, and mean pairwise raw agreement are reported alongside
  as conventional references. Because the design counterbalances sides, the
  marginal split is close to even and $\alpha$ is not distorted by the skewed-margin
  behaviour that afflicts $\kappa$ on lopsided binary data.
- **Interval**: 95% percentile bootstrap over pairs, 10,000 resamples. Pairs are
  the units the sampling rule drew; resampling three to eight raters would give an
  interval with no usable coverage, so raters are not resampled.
- **Test**: permutation, 10,000 resamples, each rater's answers shuffled across
  pairs independently, one-sided.
- **Supported if** the Holm-adjusted $p < 0.05$.

### H1b. The class is not the whole story (secondary, not in the primary family)

- **Statistic**: $\alpha$ restricted to the 18 pairs whose recorded classes are
  equal, with a 95% bootstrap CI over those pairs.
- **Supported if** the interval excludes 0.
- Secondary because 18 pairs give a wide interval by construction. H3 tests the
  same premise from the other direction with more pairs, and carries the claim.

### H2. Observable attributes predict the consensus, out of sample

- **Target**: the majority judgement per pair, coded 1 when the pair's canonical
  first order is chosen. Pairs whose majority ties are dropped and counted.
- **Features**, entered as the A-minus-B difference, and exactly the three
  observable inputs the estimator reads: a corpus-level urgency prior for the
  merged trade (mean of $5 - c$ over all pool orders of that trade, computed at
  build time from the corpus and never from the responses), $\log(1+\text{labour
  hours})$ as job size, and days waited.
- **Why a scalar trade encoding**: a full 14-level trade one-hot would put roughly
  16 parameters against 50 pairs. The scalar keeps the model at three parameters.
  The one-hot version is reported as a robustness row and labelled high-variance.
- **Model**: L2-penalised logistic regression, penalty fixed at $\lambda = 1$ on
  features standardised inside each training fold, intercept unpenalised. The
  penalty is **pre-registered rather than tuned**, because at 50 pairs an inner
  tuning loop is noisier than the choice it makes.
- **Protocol**: 20 repeats of stratified 5-fold cross-validation; out-of-fold
  probabilities averaged over repeats.
- **Statistic**: area under the ROC curve of the pooled out-of-fold predictions,
  with a 95% percentile bootstrap CI over pairs. Accuracy and Brier score reported
  alongside.
- **Comparators**: M0 constant (a predictor with no information has an area of 0.5
  by definition; its cross-validated base rate supplies Brier and accuracy), M1 the
  recorded class alone, M3 class plus attributes.
- **Test**: permutation on the label, 500 resamples, full protocol at 5 repeats,
  one-sided.
- **Supported if** the Holm-adjusted $p < 0.05$. The bootstrap lower bound
  exceeding 0.5 is reported alongside but is not the decision rule.

### H3. The recorded class does not settle the order of service

- **Sample**: the class-differing pairs (S2, S3, S4) carrying an untied majority.
- **Statistic**: the rate at which the majority follows the recorded class, that
  is, chooses the order whose recorded class is more urgent, with a 95% Wilson
  interval.
- **Test**: exact binomial, one-sided less, against the pre-registered null
  $\pi_0 = 0.90$, which is the rate a record that effectively settles the question
  would produce.
- **Supported if** the Holm-adjusted $p < 0.05$.
- **Also reported, not tested**: the exact binomial against 0.5 (does the class
  carry any signal, where the expected answer is yes); the rate within each of S2,
  S3, S4 separately, since S3 is where the record and the attributes were
  deliberately set against each other; and, on the 18 class-silent pairs, the share
  carrying a majority of at least two thirds, which cannot come from the class.

### Multiplicity

Holm step-down across the three primary hypotheses **{H1, H2, H3}**, matching the
family convention of the manuscript's statistics subsection. H1b is secondary and
reported with an interval rather than a $p$-value. The instrument check below is
deliberately outside the family: it is a check on the questionnaire, not a claim,
and including a small, predictably imprecise test would tax the three real claims
through the correction for nothing.

### Instrument check (gate, not a hypothesis)

- **Statistic**: pooled within-rater consistency over the 6 repeated pairs, that
  is, the share of rater-repeat trials answered identically, with an exact
  Clopper-Pearson 95% interval and an exact binomial $p$ against 0.5. Cohen's
  $\kappa$ between first and second presentations is reported alongside.
- **Gate**: passed when the pooled rate is at least 0.60 **and** the exact interval
  excludes 0.5.
- **Consequence of failing**: the manuscript must state that the instrument did not
  demonstrate stable within-rater judgement at this sample size, and must attach
  that caveat wherever H1 to H3 are quoted. The numbers are still reported.
- Six repeats give 18 to 48 pooled trials across the planned rater range. That is a
  coarse instrument and the gate is set accordingly; a per-rater rate moves in steps
  of one sixth and is descriptive only.

## 6. Translation into the model's parameters

Reported as a translation under stated assumptions, never as a measurement of
$\epsilon$ or $\beta$. The pilot measures how practitioners rank pairs of real work
orders; $\epsilon$ and $\beta$ are properties of a simulated supervisor acting
inside a dispatch loop. The two are analogues.

**Noise level $\epsilon$.** Assume one consensus ordering per pair and that each
practitioner reports it independently with a constant error rate $q$. Then two
practitioners agree with probability $(1-q)^2 + q^2$, so the observed mean pairwise
agreement $P$ identifies

$$q = \tfrac{1}{2}\bigl(1 - \sqrt{2P - 1}\bigr), \qquad P \ge 0.5 .$$

The model's $\epsilon$ is the probability that a reviewed decision yields a
corrupted correction. On a two-alternative item a corrupted correction is wrong
with probability $1/2$ under the random-pick branch and with probability 1 in the
worst case, which brackets $\epsilon \in [q,\, 2q]$. Genuine differences of
professional judgement are counted as error by the single-consensus model, so $q$
over-states any one practitioner's error against their own standard and the band is
an upper reading. The result is compared against the swept grid $\{0, 0.10, 0.25\}$.
If $P < 0.5$ the model does not identify $q$ and nothing is reported.

**Recoverable share $\beta$.** Under the paper's latent
$\xi = \sqrt{\beta} f(x) + \sqrt{1-\beta}\, z$, an oracle scoring pairs by $f(x)$
attains a known area under the ROC curve at each $\beta$. That map is computed by
seeded Monte Carlo (400,000 draws, seed fixed in the script) and inverted. Two
corrections act in opposite directions and both are reported:

- the pilot's model is a three-feature linear proxy for a function nobody has
  written down, fitted on about 50 pairs, so it under-fits and pushes the reading
  **down**;
- symmetric error in the majority label attenuates the area by the exact factor
  $(1 - 2e)$, where $e$ is the probability that a majority of practitioners errs
  (computed from $q$ and the rater count, with ties excluded), and dividing it out
  pushes the reading **up**.

The output is the range between the two, described as a range of plausible values,
not as a confidence interval and not as a bound. Where the proxy happens to be well
specified the upper endpoint can sit above the true share; the self-test confirms
this, which is why the range and not the endpoint is reported. The result is
compared against the study's headline band $[0.75, 1.00]$ and its wider sweep.

## 7. Power

Computed by simulation before registration, under a generative model in which the
consensus ordering is $\text{sign}(\sqrt{\beta} s + \sqrt{1-\beta} z)$ with $s$ a
fixed linear function of the three attribute differences on the actual 50 pairs,
and each rater flipping the consensus with probability $q$ independently per item.
That model is a stand-in for practitioner behaviour, not a measurement of it, so
the table below is a design aid and not a prediction. 200 simulations per cell;
per-hypothesis $\alpha = 0.05$ uncorrected, so realised power after Holm is
somewhat lower. Full output: `results/y3_w8/selftest/power.json`.

| raters | $\beta$ | $q$ | mean $\alpha$ | power H1 | mean AUC | power H2 | power, instrument check |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.40 | 0.20 | 0.358 | 0.77 | 0.723 | 0.64 | 0.23 |
| 3 | 0.40 | 0.20 | 0.360 | **1.00** | 0.693 | 0.66 | 0.45 |
| 5 | 0.40 | 0.20 | 0.358 | **1.00** | 0.737 | **0.84** | 0.58 |
| 8 | 0.40 | 0.20 | 0.359 | **1.00** | 0.765 | **0.86** | 0.76 |
| 5 | 0.10 | 0.20 | 0.355 | 1.00 | 0.554 | 0.20 | 0.71 |
| 5 | 0.20 | 0.20 | 0.354 | 1.00 | 0.627 | 0.41 | 0.69 |
| 5 | 0.60 | 0.20 | 0.357 | 1.00 | 0.815 | 0.98 | 0.64 |
| 5 | 0.40 | 0.10 | 0.643 | 1.00 | 0.779 | 0.93 | 0.99 |
| 5 | 0.40 | 0.30 | 0.162 | 0.91 | 0.667 | 0.59 | 0.23 |

Three things follow, and each is committed to here rather than argued after the
fact.

**H1 is comfortably powered and H1 is where the design's strength lies.** Three
raters already give power 1.00 against a moderately noisy consensus, and even at a
per-rater error rate of 0.30, where $\alpha$ falls to about 0.16, power is 0.91.
Fifty pairs is enough because the pairs, not the raters, carry the test.

**H2 is powered against a substantial recoverable share and not against a small
one.** At five raters, power is 0.84 when the share is 0.40 but only 0.41 at 0.20
and 0.20 at 0.10. **A null result on H2 therefore does not mean the recoverable
share is zero; it means it is below roughly 0.3 at this sample size, and the
manuscript must say so in exactly those terms rather than reporting an absence of
evidence as evidence of absence.** Recruiting five rather than three matters most
here: it lifts H2 power from 0.66 to 0.84, whereas H1 is already saturated.

**The instrument check is the weakest test in the design, deliberately.** Power
runs from 0.23 to 0.76 across the planned range, which is why it is a gate on the
estimate and its interval rather than a member of the multiplicity family. The
simulation is also pessimistic about it: a simulated rater re-answers a repeat with
a fresh independent error draw, so the modelled consistency is
$(1-q)^2 + q^2$, whereas a real practitioner's error is partly systematic and will
repeat with it. Read the simulated figure as a floor.

## 8. What each possible outcome means

| outcome | reading | what goes in the manuscript |
|---|---|---|
| H1 supported | a shared urgency ordering exists beyond chance among practitioners | the premise is grounded; report $\alpha$, its interval, and the S1 restriction |
| H1 not supported | practitioners do not agree at this sample size; a shared hidden urgency is not demonstrated | report the figure and state the premise remains an assumption; delete the translation sentences |
| H2 supported | the consensus is partly a function of the observable attributes: $\beta > 0$ has a real-data analogue | report the area, its interval, and the comparison against the class-only model |
| H2 not supported | practitioners agree on something the three coded attributes do not capture | report it as pointing at the free text (the W6 direction), not against the premise; $\beta$ translation is suppressed |
| H3 supported | the recorded class carries information but does not settle which job is served first | the direct empirical statement of the paper's core premise |
| H3 not supported | on these campuses the record is a better guide than the study assumes | report it beside the Payoff Condition as a bound on the headroom the layer has there |
| instrument gate fails | the questionnaire did not elicit a stable judgement at this size | attach the caveat wherever H1 to H3 are quoted |

**Every measured number goes into the paper whichever way it falls.** Emphasis and
order may change; a number may not.

## 9. Degradation by sample size

The analysis runs and reports honestly at any size, saying what cannot be computed
rather than printing a meaningless number.

| responses | what is computed | what is not |
|---|---|---|
| 0 | the design summary | everything else |
| 1 | within-rater consistency; predictability against that one rater's judgement, labelled as such | all inter-rater statistics; the $\epsilon$ translation; the de-attenuated $\beta$ |
| 2 | every statistic | the "majority" exists only where the two agree, so H3 and H2 are conditioned on unanimous pairs and read high; the script says so in its output |
| 3 to 8 | every statistic | nothing; intervals narrow with the rater count |

At an even rater count a pair can split evenly. Ties are dropped from H2 and H3 and
the dropped count is reported.

## 10. Everything that is reported regardless of outcome

$\alpha$ with its interval and permutation $p$; $\alpha$ per stratum; Fleiss'
$\kappa$; mean pairwise Cohen's $\kappa$; mean pairwise agreement; within-rater
consistency pooled and per rater; the majority-versus-class rate overall and per
stratum, and per rater; the decisiveness of the majority on class-silent pairs; the
areas, accuracies and Brier scores of all four models plus the one-hot robustness
row; the counts dropped for a tied majority; and both translations with their
assumptions.

## 11. Operating procedure

0. Settle the institutional ethics question (review or exemption) before inviting
   anyone. Nothing else in this list happens first.
1. Recruit 3 to 8 practitioners with the note in `pilot/RECRUITMENT_NOTE.md`.
   Assign each a code (R1, R2, ...) and keep the mapping out of the repository.
2. Send `pilot/y3_w8_pilot.html` as an attachment. Never send
   `pilot/y3_w8_manifest.csv`; it is the answer key.
3. Collect the returned CSVs into `pilot/responses/`. Filenames do not matter; the
   `rater_id` column does.
4. Run `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c 20-23 python
   scripts/y3_w8_pilot_analyse.py`. This is the single, final run.
5. Read `results/y3_w8/pilot_analysis.md`, fill the macro block from
   `pilot/METHODS_PARAGRAPH.md`, and paste the subsection.
6. Deposit the instrument, this registration, and the analysis output with the
   released code.

## 12. Deviations

Any change to this document after the first response arrives is recorded here with
its date and its reason, and reported in the manuscript. The list is empty.

| date | change | reason |
|---|---|---|
| (none) | | |
