# P8 practitioner metrics: run plan

Written before any computation. Two referee objections are closed by re-scoring
schedules that the published pipeline already produces. **No training run of any
kind is launched**: the M0 estimator refit is the same deterministic in-harness
fit the published `y3_p4_m0grid` path performs, the M1 checkpoints are loaded
frozen from `train_log/y3_sweep/`, and nothing is written to `train_log/`.

Script: `scripts/y3_practitioner_metrics.py` (new).
Outputs: `results/y3_p8/` only.
Nothing under `paper/`, `src/`, or the existing `scripts/*.py` is modified.

---

## Computation 1 — reproduction gate (`--part repro`)

**Purpose.** Prove that my code path reproduces the published headline
`\MzeroGain` = 45.4% before any new number is trusted. Lands in: the gate that
licenses everything below; reported in the reply to the referee as the first
line of the response.

**What is computed.** The headline cell (campus 9, storm2, u=100, beta=1.0,
rho=0.25, eps=0, theta=1.0, targeted, F-NL, master_seed=12345,
channel=full_class_shift), seeds 301-310, held-out instances files[20:30],
through my own module: refit the M0 estimator per seed, run RULE / M0 / ORACLE /
RULE+SUP / M0+SUP, score with `fmwos.hitl.true_objective.score_true`, average
per instance then over seeds, and form
`100*(RULE - M0)/RULE`.

**Expected result.** 45.3620354692... exactly, and every one of the 500
per-instance TWT* values (5 deciders x 10 seeds x 10 instances) equal
bit-for-bit to the committed `results/y3_p4/cache/*.json` values.

**If the opposite happens.** Any mismatch, at any tolerance above 0, means my
re-scoring path is not the published path, so every attainment and
preventive/corrective number derived from it would describe a different
experiment. In that case I stop, report the mismatch, and compute nothing
downstream. There is no partial-credit outcome here.

**Contamination risks checked.**
- *Thread count.* A previous agent established that this pipeline is bit-exact
  only at one numeric thread per process; `torch.set_num_threads(4)` changes the
  estimator's floating-point reduction order and moves the headline by several
  percentage points. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
  and `NUMEXPR_NUM_THREADS` are set to 1 in the module body *before* numpy/torch
  are imported, and `torch.set_num_threads(1)` is called at the top of every
  worker function so a forked or spawned worker cannot inherit a different value.
  Parallelism comes only from separate worker processes.
- *Core contention.* Cores 0-19 belong to other agents; every run is launched
  under `taskset -c 20-23` with at most 4 workers. No wall-clock number is
  reported from this machine.
- *Cache write-back.* My module never writes into `results/y3_p4/cache/`. It
  reads those files only as the comparison target. All new artefacts go to
  `results/y3_p8/`.
- *Stale partial output.* Each part writes to a temporary file and renames, so a
  half-written CSV cannot be swept into the aggregation. The aggregation reads
  only files it wrote in the same invocation.
- *Split leakage.* The eval pool is asserted disjoint from the estimator's
  train+probe pool (files[:20]), the same assertion the published harness makes.

**Data-accuracy checks.**
- Instance ids of the eval pool asserted equal to the ten ids recorded in
  `results/y3_p5/harvest/primary_multiseed_summary.json:eval_inst_ids`
  (`c09_storm2_w80_u100_0020` ... `_0029`).
- Cell dictionary asserted field-by-field against the `cell` block of the same
  file.
- M1 checkpoints: parameter count asserted at 14276, `gate == 1.0`,
  `use_deadline_head == True`, `correction_mode == full_class_shift`, exactly the
  assertions the published harvest makes.
- Per-instance work-order counts recorded, and the per-class denominators
  asserted identical across deciders and seeds (the latent draw depends only on
  instance id, master seed, beta and family, never on the decider seed).

---

## Computation 2 — per-true-class attainment (`--part attain`)

**Purpose.** Answer Objection 1: campus maintenance contracts are written as
percentage attainment inside a service window per priority class, not as
weighted lateness, and a method can cut weighted tardiness while *reducing*
attainment for the most urgent class. Lands in: a new results table
`tab:attainment` and one paragraph of Results; the honest reading in Discussion.

**What is computed.** For every decider (RULE, RULE+SUP, M0, M0+SUP, M1,
ORACLE) and every reported cell, the share of orders whose completion time
satisfies `C_j <= d*_j` (true deadline), broken down by true class `c*_j`
(1 to 4), pooled over the ten held-out instances within a seed and then averaged
over seeds, with the seed standard deviation. The overall attainment is
cross-checked against `1 - breaches/n_scored` from `score_true`, which already
computes it and discards it.

Cells: the headline cell above over all ten seeds; the busy-load cell (u=90,
beta=1.0, rho=0.25) and the low-recoverable-share cell (u=100, beta=0.75,
rho=0.25) over seeds 301-303, which is the seed set and the pair of cells the
manuscript reports as its non-headline contention checks
(`\RegimeHtwoBusy` / `\RegimeHoneBusy` / `\RegimeHtwoBeta` / `\RegimeHoneBeta`).

**Expected result.** The referee's concern is directional and, from the shape of
the objective, plausible: class 1 carries weight 8 and an 8-business-hour SLA, so
a few class-1 orders left very late cost less than many class-2/3 orders
rescued. I therefore expect class-1 attainment under M0 to be at or below RULE
while classes 2-4 rise, and I expect the overall attainment to rise. That is a
prediction, not an assumption: the numbers are reported as they come.

**If the opposite happens** (class-1 attainment rises too), the objection is
answered outright and the table becomes a positive result: the method improves
both the contract metric and the objective it optimises. Either outcome changes
what the manuscript says, so the computation is worth running.

**Contamination risks checked.** As Computation 1, plus: attainment is computed
from the same schedule object that produced the TWT* number in the same worker
call, so a decider's attainment and its weighted tardiness cannot come from
different rollouts.

**Data-accuracy checks.** Per-class denominators printed per cell and asserted
constant across deciders and seeds; class labels taken from
`overlay.apply(inst)["c_star"]`, never re-derived; the boundary test uses the
same `_BREACH_TOL = 1e-9` as the validator and `true_objective`.

---

## Computation 3 — preventive/corrective composition (`--part composition`)

**Purpose.** Answer Objection 2, part 1: the claim that campus C9 is about 82%
preventive-maintenance work, all mapped to recorded class 4, and that the paper
never says so. Lands in: one sentence in the setup/benchmark section and the
referee reply.

**What is computed.** For each campus the manuscript evaluates on (5, 9, 10, 12),
over exactly the instances that campus is *evaluated* on (the union of the
held-out eval pools recorded in the committed `results/y3_p4/cache/*.json`
records for that campus), the preventive share `is_pm`, the recorded-class
distribution, and the recorded-class distribution split by preventive versus
corrective. The headline cell's own ten instances are reported separately.
The campus-level figure is compared against `pm_share_r5a` in
`results/y3_p6/priority_reliability.csv`, which is computed over the raw FMUCD
rows rather than over the generated instances, so the two are related but not
required to be equal.

**Expected result.** C9 preventive share close to 0.81 (the `pm_share_r5a`
value), essentially all preventive orders in recorded class 4, and a recorded
class-4 share therefore above 0.8. If the instance-level share came back far
from the raw-row share, that would mean the generator resamples the priority mix
and the referee's inference from the raw table would not carry to the
experiments; either way the manuscript must state the composition it actually
evaluates on.

**Contamination risks checked.** The composition is read from the instance JSON
files on disk, which are the same files the evaluation loads, so there is no
possibility of describing a different population than the one scored. No RNG is
involved.

**Data-accuracy checks.** Instance ids listed in the output CSV; counts of work
orders per campus reported; recorded class read from `work_orders[*].priority`
and the preventive flag from `work_orders[*].is_pm`, the fields the calibration
rule R5a in `src/fmwos/calib.py` writes.

---

## Computation 4 — exact preventive/corrective decomposition (`--part attain`, same pass)

**Purpose.** Answer Objection 2, parts 2 and 3: split the headline reduction in
true weighted tardiness, and the attainment change, into the preventive and the
corrective part. Lands in: the same Results paragraph and the honest reading.

**What is computed.** True weighted tardiness is a sum of per-order terms,
`TWT* = sum_j w*(c*_j) * max(0, C_j - d*_j)`, plus an access penalty that is off
at every cell reported here. Partitioning the orders into preventive and
corrective therefore gives an exact decomposition, not an approximation:

    TWT*(RULE) - TWT*(M0) = [PM part] + [CM part].

The code asserts `|(PM part + CM part) - total| <= 1e-6` per instance, per seed,
and on the cell aggregate, and asserts that the access penalty is exactly 0 so
the partition is complete. The same partition is applied to the attainment
counts, where the numerators and denominators are integers and the additivity is
exact by construction.

**Expected result.** If C9 is ~81% preventive and preventive work is all
recorded class 4, then the correction's promotions act largely on preventive
orders, and the preventive part of the reduction may dominate. That is a finding
the manuscript must state, because the paper's motivating example is a leaking
pipe, which is corrective work.

**If the opposite happens** (the corrective part dominates, or is proportionally
larger than the corrective share of the work), the motivating framing survives
intact and the paper gains a stronger claim than it currently makes.

**Contamination risks checked.** The decomposition is computed from the same
per-order records as the totals in the same worker call; the totals are then
re-checked against `score_true`'s scalar `TWT_true` for the same schedule, so a
bookkeeping error in the split cannot pass silently.

**Data-accuracy checks.** `is_pm` read per order id from the instance; every
order in a schedule matched to an instance order; the count of scored orders
asserted equal to `score_true`'s `n_scored`.

---

## What is explicitly not done

- No training. No PPO run, no DAgger run, no checkpoint written.
- No wall-clock timing is reported: the machine is shared and any timing measured
  on it is not a measurement.
- Schedules are not read from `results/y3_p4/cache/`, because that cache stores
  only per-instance TWT* scalars, not the schedules themselves. The schedules are
  therefore regenerated by the published code path, and the cache is used as the
  bit-exactness target for the regenerated values. Which numbers are reused and
  which are recomputed is stated explicitly in the report.

---

# Outcome (written AFTER the run; the plan above is unchanged)

The predictions above are left exactly as written before the run. What actually
came back:

**Reproduction gate: PASS, exactly.** All 500 per-instance TWT* values (5 cheap
deciders x 10 seeds x 10 instances) equal the committed
`results/y3_p4/cache` values bit-for-bit (`max|diff| = 0.0`), and `\MzeroGain`
came back at 45.36203546923573%, identical to the published value to every digit
(`results/y3_p8/repro_check.json`).

**Objection 1: the predicted direction is right in sign, but the effect is not
stable and is not the method's doing.** At the headline cell, class-1 attainment
under M0 is 84.51% against the rule's 84.57%, a change of $-0.06$ percentage
points, or 1.1 orders in 1750 per ten-instance pool; across the ten seeds M0 is
above the rule on five and below on five, so the sign of the change does not
hold. Class-2 attainment rises from 90.47% to 97.57% and overall attainment from
96.52% to 98.22%, both on 10/10 seeds. The myopic full-information reference
gives up the same $-0.06$ points on class 1, which places the class-1 shortfall
in the crew's capacity rather than in what the dispatcher knows. Classes 3 and 4
are at 100% for every decider at all three cells. At the two other contention
cells the class-1 change under M0 is positive ($+0.15$ at the busy load,
$+0.03$ at the lower recoverable share).

**Objection 2: the referee's figure is confirmed and the decomposition is
stark.** Campus C9 is 81.16% preventive over the 70 instances it is evaluated
on and 80.85% at the headline cell; every preventive order on every campus maps
to recorded class 4, and on C9 recorded class 4 is exactly the preventive work.
The exact decomposition attributes 99.59% of the headline reduction to
preventive orders and 0.41% to corrective ones, with the largest additivity
residual over every decider and cell at $1.1\times10^{-12}$ per-instance
weighted business hours. The same structure holds for the full-information
reference (100.58% preventive, $-0.58$% corrective), so it is a property of what
the benchmark's latent makes recoverable, not of the correction layer.

**One prediction was wrong in an informative way.** The plan expected class-1
attainment to fall while classes 2 to 4 rose. Classes 3 and 4 turn out to be
saturated at 100% for every decider, so the entire attainment story lives in
classes 1 and 2, and true class 1 contains no preventive work at all: preventive
work is recorded class 4 and the class shift is clipped at two steps, so it can
reach true class 2 at best.
