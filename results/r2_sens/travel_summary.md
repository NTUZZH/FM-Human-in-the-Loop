# A3 travel-overhead re-simulation summary

Source: `travel.csv`. Six PDRs (edd, wspt, atc, pfifo, mor, random) on the E5 base set (campus {5,9,10,12} x size {150,400} x first-30 replay-test = 240 instances). A technician starting an order whose building differs from its previous order's building pays an extra 0.25 (and 0.50) bh; first order and any null-building order pay nothing. TWT is computed in-script (with travel end-start = overhead+p_bh, so the independent validator, which enforces end-start==p_bh, would reject the shifted schedules). Guard: on overhead=0 the validator was run on every schedule and its WWT equals the in-script TWT (1440/1440 rows, 0 failures), pinning the in-script objective to the referee.

## Building coverage

| campus | instances | WOs | missing building | with building |
|---|---|---|---|---|
| 5 | 60 | 16500 | 22 | 99.9% |
| 9 | 60 | 16500 | 16500 | 0.0% |
| 10 | 60 | 16500 | 3 | 100.0% |
| 12 | 60 | 16500 | 16500 | 0.0% |

Buildings exist on campuses 5 and 10 (~99.9%% covered). Campuses 9 and 12 have NO building ids (100%% null) so travel is a no-op there -- a schema fact (the interface spec lists 9/10/12 as null, but campus 10 actually carries building ids). Over the whole 240-instance set 50.0%% of orders have a null building; over the covered campuses (5,10) only 0.08%% are null.

## Ranking robustness (Kendall tau-b, no-travel vs travel)

Per-cell tau on the mean-TWT-per-method vectors. '-' = degenerate cell: the baseline ranking is fully tied (capacity-adequate campus 5 runs at TWT < 8 with all six rules ~equal), so there is no ordering for travel to preserve or break.

| overhead | c5/150 | c5/400 | c9/150 | c9/400 | c10/150 | c10/400 | c12/150 | c12/400 | mean cell |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | - | - | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.50 | - | - | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**Verdict.** On every non-degenerate cell -- campus 10 (both sizes, the only discriminative building-covered cell) and the no-op campuses 9/12 -- Kendall tau-b = 1.00 at both 0.25 and 0.50 bh/switch: travel does NOT reorder the dispatching-rule leaderboard. (A scale-free cross-cell average-rank pooled tau, the tab_sensitivity.tex style, comes out 0.57/0.71 here, but that number is dominated by tie-breaking noise in the near-degenerate campus-5 cells and is not a substantive leaderboard change.)

## Mean TWT inflation per method

Building-covered campuses (5, 10) -- where the knob can bite:

| overhead | edd | wspt | atc | pfifo | mor | random |
|---|---|---|---|---|---|---|
| 0.25 | +0.8% | +0.9% | +0.8% | +0.8% | +1.3% | +5.0% |
| 0.50 | +1.6% | +2.0% | +1.6% | +1.6% | +2.6% | +7.2% |

All four cells (incl. the no-op 9/12), for completeness:

| overhead | edd | wspt | atc | pfifo | mor | random |
|---|---|---|---|---|---|---|
| 0.25 | +0.1% | +0.1% | +0.1% | +0.1% | +0.1% | +0.3% |
| 0.50 | +0.1% | +0.1% | +0.1% | +0.1% | +0.2% | +0.5% |

Travel inflates absolute TWT by a few percent on the covered campuses (pooled largest: random +5.0% at 0.25 bh, random +7.2% at 0.50 bh; per single campus the largest is MOR +6.9% on campus 5 at 0.25 bh) but leaves the ranking intact.
