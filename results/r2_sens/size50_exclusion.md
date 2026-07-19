# B9 -- size-50 exclusion justification

A natural question: why does the dynamic verdict table (`results/p4_dyneval/tab_gateB.tex`) pool
sizes 150+400 and leave out size 50?  Answer: size-50 instances are largely
non-discriminative -- on most of them *every* online method already attains zero
weighted tardiness, so they carry almost no information about which dispatcher is
better.

## Data-availability note (important)

`results/p4_dyneval/results.csv` contains **no size-50 rows** -- the dynamic
evaluation was scoped to sizes {150, 400} (see `REPLAY_SIZES` in
`scripts/p4_dyneval.py`).  So the size-50 share cannot be taken from the dynamic
CSV; the only results file that carries size 50 is the **static** benchmark
`results/e1_static/results.csv` (sizes 50/150/400, the six online PDRs plus
offline solvers).  The three numbers below are therefore computed from
`e1_static` over the six online dispatching rules (edd, wspt, atc, pfifo, mor,
random); the dynamic replay-default 150/400 shares are given as a consistency
cross-check.

## Share of instances where ALL online methods attain zero WWT

"config" = one static instance id; counted only when all six online rules are
feasible; "all zero" = every one of the six has WWT <= 1e-9.

Verdict campuses {5, 9, 10, 12} (the scope of tab_gateB.tex):

| size | all-online-zero share |
|---|---|
| **50**  | **66.5%** (532/800) |
| **150** | **50.8%** (406/800) |
| **400** | **38.5%** (294/763) |

All six campuses (1,2,5,9,10,12):

| size | all-online-zero share |
|---|---|
| 50  | 60.5% (726/1200) |
| 150 | 43.8% (526/1200) |
| 400 | 31.5% (347/1101) |

Dynamic cross-check (`p4_dyneval`, replay-default, online = 6 PDR + 3 RL, verdict
campuses; no size 50 exists there):

| size | all-online-zero share |
|---|---|
| 150 | 43.5% (174/400) |
| 400 | 32.0% (116/363) |

## Verdict

Two-thirds of size-50 instances (66.5%) are solved to zero tardiness by every
online method, versus roughly half at size 150 and ~39% at size 400.  Size-50 is
the least discriminative size, so excluding it (and pooling the more contended
150+400 instances) concentrates the verdict on the cases that actually separate
the methods.
