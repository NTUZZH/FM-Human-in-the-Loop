# FM-Human-in-the-Loop — learning maintenance work-order dispatch from supervisor overrides

Companion repository for the manuscript *"A Correction Layer that Learns Hidden
Urgency from Supervisor Overrides for Maintenance Work-Order Dispatching"*
(submitted). It releases the supervisor overlay, the correction-layer (M0)
pipeline, the learned-dispatcher (M1) and belief (M2) variants, trained
checkpoints, evaluation scripts, and the per-method scored results behind every
reported number, together with the underlying dispatch benchmark so the study
reproduces end to end.

The base benchmark is described in *"When Does Learned Dispatching Beat
Priority Rules? An Open Benchmark for Technician-Constrained Maintenance
Work-Order Scheduling in Building Portfolios"* (preprint:
http://dx.doi.org/10.2139/ssrn.7095162); its instances are reproducible from
the open FMUCD corpus alone.

The base benchmark provides, for reuse and verification:

- **Benchmark instances** (`data/instances.tar.zst`): 3,186 real-data replay
  instances and 1,800 calibrated generator instances (4,986 total) built from
  the public FMUCD work-order database, in a documented JSON schema.
- **The instance generator** with per-campus fitted parameter packs
  (`src/fmwos/generator.py`, `results/p2_generator/`).
- **The independent feasibility validator** (`src/fmwos/validator.py`) that
  scores every method and shares no code with any scheduler.
- **All methods under test**: six dispatching rules, exact and rolling CP-SAT,
  a genetic algorithm, and the PPO-trained dispatcher (MLP and attention
  variants), plus training code.
- **Scored results** for every experiment in the paper (`results/`), and the
  diagnostic re-simulations (travel, weight-vector, candidate-cap sweeps).
- **The pre-specified evaluation protocol** (`docs/protocol.md`): the two
  decision gates, their pass/fail criteria, and the dated amendment history.

## Data source and licence

Raw data: FMUCD (Facility Management Unified Classification Database),
Mendeley Data, DOI [10.17632/cb8d2nsjss.1](https://doi.org/10.17632/cb8d2nsjss.1),
CC BY-NC 4.0. The exact distribution file used has SHA-256
`4464648252c4bdca2a6deba9d467e94aec7568d675f51e06d6d343b3c09f006a`.
Everything in this repository is released under **CC BY-NC 4.0**, inherited
from FMUCD; commercial users must license FMUCD independently.

## Reproduce

```bash
conda env create -f environment.yml && conda activate fmwos
# 1. download FMUCD to data/raw/FMUCD.csv (SHA-256 above must match)
python scripts/p0_profile.py                    # cleaning audit + profiling
python scripts/p1_instances.py                  # calibration + replay track
python scripts/p2_generator.py                  # generator track
PYTHONPATH=src python scripts/p2_e1.py          # E1 static (sharded, resumable)
PYTHONPATH=src python -m fmwos.train --seed 301 --curriculum v2  # PPO
PYTHONPATH=src python scripts/p4_dyneval.py --with-pmmix --with-storm2 \
    --storm-arrivals 1.25,1.5,2.0,3.0           # dynamic evaluation
PYTHONPATH=src python scripts/p4_analysis.py    # Gate-B tables
python scripts/p5_figures.py                    # paper figures
```

Unpack the released instances instead of rebuilding them:

```bash
mkdir -p data/processed && tar -C data/processed --zstd -xf data/instances.tar.zst
```

Tests (plain python): `PYTHONPATH=src python tests/<file>.py`.

## Layout

- `src/fmwos/` — io/cleaning, calibration, instances, generator, validator,
  dispatching rules, CP-SAT (static + rolling), GA, environment, lower
  bound, policies (MLP + attention), PPO training.
- `scripts/` — one entry point per experiment; `r2_*.py` are the revision
  diagnostics (travel, weights, candidate cap).
- `results/` — every number in the paper traces to a file here.
- `docs/` — pre-specified protocol and the public decision log.

## Human-in-the-loop correction layer (follow-on study)

This repository also carries the code, trained models, and scored results for a
follow-on study on recovering a hidden urgency signal from supervisor overrides.
The manuscript is under review; citation details will be added on publication.

**What the overlay is.** A work order's recorded priority class is a coarse proxy
for how urgent the job really is. The supervisor overlay
(`src/fmwos/hitl/overlay.py`) adds a seeded, reproducible latent urgency to each
order and uses it to define the *true* objective a schedule is graded on: a
weighted tardiness computed with corrected weights and corrected deadlines. An
independent validator (`src/fmwos/hitl/true_objective.py`) scores every method
against that true objective and shares no code with any scheduler.

**The review-and-override loop.** A supervisor (`src/fmwos/hitl/supervisor.py`)
inspects a budgeted fraction (the review budget, rho) of dispatch decisions. On a
reviewed decision it may replace the base rule's pick with the one the latent
urgency prefers. Every decision is logged, so the override log records exactly
where recorded priority and true urgency disagree.

**The methods.** M0, M1, and M2 all recover the same latent urgency; they differ
in how.

- **M0**, the correction layer (`src/fmwos/hitl/augmented_rule.py`): fits an
  urgency estimator to the override log, then re-scores the base dispatching rule
  with corrected weights. No reinforcement learning. This is the deployable
  headline method.
- **M1**, the learned dispatcher (`src/fmwos/hitl/latent_head.py`,
  `src/fmwos/hitl/intervention.py`): a PPO policy with an urgency head, trained by
  intervention (DAgger) on the same overrides.
- **M2**, the belief variant (`src/fmwos/hitl/belief.py`): a Bayesian estimator
  that also steers the review budget toward the decisions it is least sure about.
- **PI-0**: a blind reinforcement-learning control that never sees the overrides,
  used to check that M1's gain comes from learning the overrides rather than from
  more training.

With no supervisor attached, the loop is byte-identical to the Y1 dispatcher; the
`y3_e0_anchor.py` script asserts that equivalence.

**Reproduce.**

```bash
conda env create -f environment.yml && conda activate fmwos
mkdir -p data/processed && tar -C data/processed --zstd -xf data/instances.tar.zst

# 1. tests and the E0 equivalence anchor
PYTHONPATH=src python tests/test_overlay.py        # and the other Y3 test_*.py
PYTHONPATH=src python scripts/y3_e0_anchor.py

# 2. train the learner sweep (M1 with the urgency head, plus the PI-0 control)
python scripts/y3_sweep_train.py --verify-only     # pre-launch config check
python scripts/y3_sweep_train.py --concurrency 2   # 10 seeds, held-out cell

# 3. score the method ladder (M0 needs no training; it is fit during evaluation)
PYTHONPATH=src python scripts/y3_p4_m0grid.py --part all --workers 8  # M0 grid
PYTHONPATH=src python scripts/y3_harvest_primary.py --workers 5       # 10-seed ladder
PYTHONPATH=src python scripts/y3_harvest_final.py --workers 5         # attribution

# 4. figures
python scripts/y3_figs_f2.py && python scripts/y3_figs_f3.py
python scripts/y3_figs_f4.py && python scripts/y3_figs_f5.py
```

**Where the follow-on results live.**

- `src/fmwos/hitl/`: the overlay, supervisor, true-objective validator, and the
  M0/M1/M2 methods. `src/fmwos/env.py` gains a flag-gated supervised episode
  driver and `src/fmwos/pdrs.py` a read-only confidence margin; both are additive.
- `results/y3_p1/`: overlay coefficients and the recoverable-share check.
- `results/y3_p4/`: the M0 grid, the regime map, and the boundary cells.
- `results/y3_p5/`: the multi-seed ladder (`harvest/`), the ablations, the review
  and label-noise sweeps (`gaps/`), the M2 variant, and the insurance checks.
- `results/y3_p6/`: transfer to held-out campuses and the FMUCD corpus checks.
- `results/y3_checkpoints/`: the trained models (final weights, config, and
  training metrics) for the sweep, the pilots, and the insurance runs.

## Citation

Citation entry will be added upon publication.
