# Dated decision log (public)

Curated technical log of every protocol deviation and non-obvious design
decision, in chronological order. Format: date — decision — reason/evidence.
The evaluation gates and their pass/fail criteria are stated in
`docs/protocol.md`; entries here timestamp the amendments and corrections the
manuscript discloses.

- 2026-07-04 — **Business-hour time axis** (8 bh/day, 5 d/wk concatenated;
  SLA windows converted accordingly). Jobs longer than one shift would
  otherwise require preemption at shift boundaries, and calendar modelling
  would roughly triple solver/environment state; standard practice in the
  parallel-machine tardiness literature. The E5 SLA sweep (±50%) covers the
  conversion's slack.
- 2026-07-04 — **Campus set**: replay restricted to the 6 timestamp-complete
  campuses (1, 2, 5, 9, 10, 12); transfer study trains on {5, 9, 10, 12} and
  holds out {1, 2}. Data reality: the remaining campuses lack usable start
  timestamps.
- 2026-07-04 — **Travel excluded from the v1 main protocol** (travel = 0);
  the 0.25 bh per-building-switch cost moves to an E5 sensitivity
  re-simulation. Sequence-dependent setups would force routing structure
  into the exact solver and blur comparability, and building identifiers are
  missing on several campuses.
- 2026-07-04 — **Priority mapping v2** (supersedes the naive numeric-
  ascending rule). Evidence: several nominal "priority" values are ~100%
  preventive (planned-work categories mixed into the field); realised
  completion durations invert the numeric order on campus 12 and contradict
  "1 = emergency" on campus 10; text-labelled campuses validate duration as
  an urgency proxy. New rule: preventive→P4; corrective keyword→class;
  corrective numeric scales keep their order with direction set by the
  Spearman sign against median corrective completion duration, then
  rank-quartiles to classes; rare/missing→P3.
- 2026-07-04 — **Replay sampling v2 (first-N releases)**. A generator realism
  check exposed selection bias in fixed-window sampling (only atypical
  low-volume days passed the acceptance filter on high-variance campuses;
  e.g. campus-5 replay PM share 0.02 against a true 0.45). v2 takes the
  first N releases from each anchor with non-overlapping windows; exact
  sizes, bias removed. Static gate re-run on rebuilt instances.
- 2026-07-04 — **R7 duplicate labour-line aggregation** changes the dataset
  scale: 1,454,039 unique work orders from 1,906,865 post-filter rows;
  labour cap p99.5 = 90.86 h computed post-aggregation.
- 2026-07-05 — **Gate A verdict: "trivially easy" branch fires** (see
  protocol.md). CP-SAT proves optimality within 60 s on 100% of a 90-instance
  pilot at every size; due-date rules within 0.2% of best-known there. Study
  weight shifts to the dynamic track; the static track is kept as a
  reference/certificate layer. Win/tie/loss tallies adopt a tie tolerance of
  1.0 weighted unit (the solver's centi-hour grid perturbs ties by ~0.04).
- 2026-07-05 — **Gate B protocol amendment (dated before any contended-arm
  numbers existed)**. Trigger: PPO development curves are flat at default
  capacity (~409–411 across seeds), consistent with Gate A's finding that
  capacity-adequate episodes barely discriminate policies. Amendment: Gate B
  is judged on both the original capacity-adequate arm and a contended arm
  (crew multipliers {0.6, 0.8} on replay tests plus generator storm cells);
  "beats a rule" = lower mean TWT with paired Wilcoxon p < 0.05, ties are
  not wins; both arms are reported regardless of outcome.
- 2026-07-05 — Held-out campuses {1, 2} excluded from all Gate-B verdicts and
  from policy checkpoint selection; they appear only in the transfer study.
- 2026-07-05 — **Curriculum v2 + checkpoint selection change**: the
  default-capacity development metric plateaus for every trained variant and
  cannot discriminate checkpoints; v2 rebalances training toward contended
  regimes and selects checkpoints on a tight-capacity development set
  (crew multiplier 0.6). Disclosure: v2 was trained after aggregate results
  on test regimes existed (a leakage risk); mitigations: held-out campuses
  untouched, fresh evaluation seeds, and both v1 and v2 reported. In the
  event v2 was no better than v1.
- 2026-07-05 — **Rolling CP-SAT correction (two stages)**. (a) An apparent
  2× blow-up on default-capacity cells was an analysis artifact (an n=8
  subsample mean juxtaposed against full-cell means); the analysis was
  changed to same-instance comparisons. (b) The real tight-cell failures
  trace to burst releases under an arrival-only replan trigger: 12–15
  budget-capped replans, after which one stale plan executes uncorrected.
  A larger budget does not help (too few replans, not slow solves). Fix: a
  periodic trigger (arrival OR every 4 bh with non-empty queue) plus a
  lexicographic flow-time tiebreak. Both the pathology and the fix are
  reported.
- 2026-07-05 — **Rolling diagnostic re-run for the released figure**: fresh
  instrumented runs of the pathological cell show the burst-instance
  magnitudes vary across runs (budget-capped solves are not deterministic;
  e.g. 15,471 historical vs 10,324 re-run on one instance) while the
  collapse-and-recovery pattern is stable; the manuscript cites the released
  diagnostic's values and says so. An earlier "rolling beats EDD on a
  spread-arrival instance (420 vs 460)" observation did not reproduce
  against the final scored results and was removed; the supported statement
  is that failure severity scales inversely with replan count.
- 2026-07-06 — **Terminology**: objective renamed TWT (total weighted
  tardiness) throughout; "pre-registered" replaced by "pre-specified"
  (protocol lives in this repository's dated log, not an external registry).
- 2026-07-06 — **Campus-2 overload diagnosed as a calibration artifact**:
  its 69 training weeks sit inside the database's population ramp (median
  weekly volume 49 → 177 → 1,708 bh across 2016/2017/test), so p95 crew
  sizing under-provisions it; held-out campus 1 (221 stationary training
  weeks) calibrates correctly. Campus-2 cells are kept as an
  extreme-overload stress test and flagged as such in the manuscript.
- 2026-07-06 — **Revision experiments added**: travel-overhead re-simulation
  (0.25/0.50 bh per switch; ranking unchanged, τ=1.00 in every
  discriminative cell), priority-weight-vector sweep ((4,3,2,1) and
  (27,9,3,1); pooled τ=1.00), candidate-cap ablation (64→256), MLP seed set
  enlarged from 3 to 10, and an attention policy class trained as a second
  learner. Outcomes: the ten-seed pool reproduces both Gate-B arms exactly;
  the attention class is no stronger (it fails even the default arm's
  consistency requirement) and costs 0.94 ms per decision; the candidate-cap
  ablation shows ~89% exact ties and no verdict flips that survive
  netting; and the tight-capacity development floors of the two
  architectures coincide (426.73 vs 426.89), supporting the problem-set
  ceiling reading. The decision map's primary colouring moved to the
  full-cell four-way comparison (22 of 27 ties; ATC outright on the
  calibration-artifact campus 2 and the two tightest training cells).
