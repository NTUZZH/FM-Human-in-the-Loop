# Pre-specified evaluation protocol (public artifact)

This document states the two evaluation gates that were fixed in the project
protocol before the corresponding results existed, and the one dated
amendment. It is the artifact referenced by the manuscript's "Pre-specified
evaluation gates" paragraph (Experimental setup) and the data-availability
statement. The repository's commit history timestamps this file and the
result files it governs.

## Gate A — static track (fixed at project start)

**Criterion.** Solve the static instances exactly with CP-SAT under a 60 s
budget at data-calibrated capacity. If the solver reaches optimal or
near-optimal schedules at all realistic instance sizes (i.e., the static
problem is trivially easy at calibrated capacity), then shift the study's
weight to the dynamic track, and say so in the paper.

**Outcome.** Branch taken: CP-SAT certified optimality on 100% of generator
instances and 93-100% of replay instances within 60 s; the best dispatching
rules land within single-digit weighted-tardiness units of the certified
optima. The paper reports the static track as a reference and moves the
discriminative analysis to the dynamic track.

## Gate B — learned dispatcher (fixed before policy training)

**Criterion.** The learned policy passes if it beats at least 3 of the 5
ranked dispatching rules (EDD, WSPT, ATC, pFIFO, MOR) by paired Wilcoxon
signed-rank tests at alpha = 0.05, consistently across all training seeds,
on dynamic instances.

**Pre-committed branch on failure.** Re-scope the paper to "benchmark +
rigorous classical study + negative learning result"; report the failure as
a finding; do not redefine success.

**Amendment (dated).** After the v1 policy's results on the original
capacity-adequate arm (replay-default) existed, and before any contended-arm
numbers existed, a contended arm (replay-tight crew multipliers {0.6, 0.8}
plus storm cells) was added to the protocol, with the same pass criterion,
and designated the primary verdict because it is the regime where
dispatching matters. The amendment, its timing, and the reason are recorded
in the dated decision log entries of the development history and disclosed
in the manuscript (threats-to-validity section).

**Outcome.** Capacity-adequate arm: pass (beats 3/5). Contended arm: fail
(beats 1/5, MOR only). The pre-committed branch was followed: the negative
result is reported in full, for the original three seeds and for the
enlarged seed set and the stronger policy class added during revision.

## Locked defaults

Objective weights w = (8, 4, 2, 1); SLA windows P1/P2/P3/P4 = 8/24/80/171.4
business hours; business calendar 8 h x 5 d; crew sizing p95 weekly trade
hours / 40 h; CP-SAT budgets 60/300 s static, 2 s rolling; GA population 100,
60 s budget; PPO hyperparameters as listed in the manuscript appendix. Any
deviation is logged with its reason in the released decision log.
