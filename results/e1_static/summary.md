# E1 static benchmark — summary (2026-07-05, results.csv @ 31,509 rows)

3,501 test instances (1,701 replay + 1,800 generator) x 9 methods; ALL
validator-feasible; 0 errors; 1h15m55s wall (8 workers).

## CP-SAT certificates
- Proof rate <=60 s: generator 100% (all sizes); replay 100%/96.0%/93.0%
  for sizes 50/150/400. cpsat300 adds only 3 proofs -> stubborn hard tail
  in large replay instances.
- Mean solve wall 1.31 s (p95 1.4 s) at workers=2.

## Method quality (mean abs gap to best-known, weighted-tardiness units)
All instances / nonzero-best-known instances only:
- cpsat60 1.14 / 2.12 (grid rounding + unproven tail)
- cpsat300 - / 0.19
- ga 2.32 / 4.30  (wall ~4.1 s; beats cpsat60 outright on 8 hard instances)
- atc 14.1 / 26.2  (best PDR)
- wspt 23.1 / 42.6
- pfifo 38.7 / 71.7
- edd 39.3 / 73.0  (EDD ~= pfifo by construction: due = release + SLA(priority))
- random 108.7 / 201.2
- mor 318.6 / 589.9 (LPT-flavour is the wrong instinct for tardiness)

## Win/tie/loss vs cpsat60 (eps = 1.0 weighted unit)
edd 3/3311/187 | wspt 2/3205/294 | atc 2/3301/198 | pfifo 3/3311/187 |
mor 2/3281/218 | random 2/3230/269 | ga 8/3434/59

## Triviality gradient (share of instances with best-known WWT == 0)
generator: 67.3%/50.5%/37.0% (sizes 50/150/400)
replay:    53.7%/37.8%/26.9%
-> at calibrated capacity roughly half of static snapshots have zero
attainable tardiness; the other half separates methods mildly; the
discriminative regimes are dynamic/contended (Gate A verdict).

## Decision latency (mean per instance)
PDRs ~0.5 ms | cpsat60 1.31 s | GA 4.1 s.

Paper artifacts fed: T2 (method x track x size table), F2 (gap distributions),
latency table, "hard tail" discussion.
