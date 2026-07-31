# P9 results: corpus-anchored recoverable-share cells

Campus 9, storm2 w80, utilisation 1.00, review budget 0.25, targeted review,
ten held-out instances (`c09_storm2_w80_u100_0020` ... `_0029`), seeds 301-310.
Reductions are in true weighted tardiness TWT*(w*,d*); positive = lower is better.
Tests are seed-averaged per-instance two-sided paired Wilcoxon signed-rank
(pratt), W/T/L counted as the test being strictly lower.

Reproduction gate: published MzeroGain = 45.3620354692%, recomputed = 45.3620354692%, difference = 0.00e+00 pp, 500/500 per-instance values bit-exact.

## Main table (10 seeds)

| Cell | beta | eps | M0 vs RULE | p (W/T/L) | M0+SUP vs RULE | p (W/T/L) | M0+SUP vs RULE+SUP | p (W/T/L) | RULE+SUP vs RULE | ORACLE vs RULE | gap closed M0 | gap closed M0+SUP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.20 | 0.00 | +12.63% | 0.0059 (9/0/1) | +30.46% | 0.0020 (10/0/0) | +9.08% | 0.0020 (10/0/0) | +23.52% | +36.23% | 34.9% | 84.1% |
| B | 0.20 | 0.25 | +6.31% | 0.0059 (9/0/1) | +25.84% | 0.0020 (10/0/0) | +5.36% | 0.0039 (9/0/1) | +21.64% | +36.23% | 17.4% | 71.3% |
| C | 0.25 | 0.00 | +15.00% | 0.0039 (9/0/1) | +31.27% | 0.0020 (10/0/0) | +7.67% | 0.0020 (10/0/0) | +25.56% | +36.16% | 41.5% | 86.5% |

## Absolute ladder, mean TWT* over seeds (10 seeds)

| Cell | Tuned rule (RULE) | Rule + supervisor (RULE+SUP) | Correction layer (M0) | Correction layer + supervisor (M0+SUP) | Myopic full-information reference (ORACLE) |
|---|---|---|---|---|---|
| A (b=0.20, eps=0.00) | 2242.3 | 1715.0 | 1959.0 | 1559.2 | 1429.9 |
| B (b=0.20, eps=0.25) | 2242.3 | 1757.1 | 2100.9 | 1663.0 | 1429.9 |
| C (b=0.25, eps=0.00) | 2287.6 | 1703.0 | 1944.4 | 1572.4 | 1460.4 |

## Recovery quality of the fitted estimator (final DAgger iteration)

| Cell | beta | eps | Pearson r | sign accuracy | zero-baseline accuracy | training override rate |
|---|---|---|---|---|---|---|
| A | 0.20 | 0.00 | 0.1429 | 0.6377 | 0.3627 | 0.0477 |
| B | 0.20 | 0.25 | 0.0693 | 0.5776 | 0.3627 | 0.1722 |
| C | 0.25 | 0.00 | 0.1784 | 0.6544 | 0.3617 | 0.0450 |

## Five-seed subset (301-305), consistency check only

| Cell | M0 vs RULE | M0+SUP vs RULE+SUP |
|---|---|---|
| A | +12.68% | +9.46% |
| B | +7.59% | +6.06% |
| C | +13.80% | +6.92% |

## Published neighbours on the same campus and load (context)

| beta | M0 alone vs RULE | seeds | source |
|---|---|---|---|
| 0.00 | -1.08% | 3 | results/y3_p4/e3_map_summary.json |
| 0.20 (eps 0.00) | +12.63% | 10 | this run |
| 0.20 (eps 0.25) | +6.31% | 10 | this run |
| 0.25 (eps 0.00) | +15.00% | 10 | this run |
| 0.50 | +26.35% | 3 | results/y3_p4/e3_map_summary.json |
| 1.00 | +45.36% | 10 | results/y3_p5/harvest/primary_multiseed_summary.json |
