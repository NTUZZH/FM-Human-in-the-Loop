# P9b: the corpus-anchored cells under the DEPLOYABLE review policy

Campus 9, storm2 w80, utilisation 1.00, review budget 0.25, ten held-out
instances (`c09_storm2_w80_u100_0020` ... `_0029`), seeds 301-310.
Reductions are in true weighted tardiness TWT*(w*,d*); positive = lower is
better. Tests are seed-averaged per-instance two-sided paired Wilcoxon
signed-rank (pratt), W/T/L counted as the test being strictly lower, and
are computed by `y3_realistic_cell.summarize_cell`, the same function that
produced `results/y3_p9/cell_summary.json`.

Gates. Reproduction: published MzeroGain = 45.3620354692%, recomputed = 45.3620354692%, difference = 0.00e+00 pp, 500/500 per-instance values bit-exact against results/y3_p4/cache. Per-cell mirror against results/y3_p9/cache: PASS. Configuration diff: PASS. Policy proof: PASS.

## Side by side: published review policy vs deployable review policy

`published` = results/y3_p9 (oracle-informed `targeted` routing, whole
weak-label aggregate). `deployable` = this run (stability routing under a
split-conformal band, proper-training fold only). Difference is
deployable minus published, in percentage points.

| Cell | beta | eps | contrast | published | p (W/T/L) | deployable | p (W/T/L) | difference |
|---|---|---|---|---|---|---|---|---|
| A | 0.20 | 0.00 | M0 vs RULE | +12.63% | 0.0059 (9/0/1) | +9.53% | 0.0098 (8/0/2) | -3.11 pp |
| A | 0.20 | 0.00 | M0+SUP vs RULE+SUP | +9.08% | 0.0020 (10/0/0) | +7.15% | 0.0645 (7/0/3) | -1.93 pp |
| A | 0.20 | 0.00 | M0+SUP vs RULE | +30.46% | 0.0020 (10/0/0) | +25.72% | 0.0020 (10/0/0) | -4.74 pp |
| A | 0.20 | 0.00 | RULE+SUP vs RULE | +23.52% | 0.0020 (10/0/0) | +20.00% | 0.0020 (10/0/0) | -3.52 pp |
| A | 0.20 | 0.00 | ORACLE vs RULE | +36.23% | 0.0020 (10/0/0) | +36.23% | 0.0020 (10/0/0) | +0.00 pp |
| B | 0.20 | 0.25 | M0 vs RULE | +6.31% | 0.0059 (9/0/1) | +3.97% | 0.0840 (7/0/3) | -2.34 pp |
| B | 0.20 | 0.25 | M0+SUP vs RULE+SUP | +5.36% | 0.0039 (9/0/1) | +3.54% | 0.0273 (8/0/2) | -1.82 pp |
| B | 0.20 | 0.25 | M0+SUP vs RULE | +25.84% | 0.0020 (10/0/0) | +21.10% | 0.0020 (10/0/0) | -4.74 pp |
| B | 0.20 | 0.25 | RULE+SUP vs RULE | +21.64% | 0.0020 (10/0/0) | +18.21% | 0.0020 (10/0/0) | -3.43 pp |
| B | 0.20 | 0.25 | ORACLE vs RULE | +36.23% | 0.0020 (10/0/0) | +36.23% | 0.0020 (10/0/0) | +0.00 pp |
| C | 0.25 | 0.00 | M0 vs RULE | +15.00% | 0.0039 (9/0/1) | +13.43% | 0.0098 (8/0/2) | -1.57 pp |
| C | 0.25 | 0.00 | M0+SUP vs RULE+SUP | +7.67% | 0.0020 (10/0/0) | +6.97% | 0.0371 (8/0/2) | -0.70 pp |
| C | 0.25 | 0.00 | M0+SUP vs RULE | +31.27% | 0.0020 (10/0/0) | +27.02% | 0.0039 (9/0/1) | -4.24 pp |
| C | 0.25 | 0.00 | RULE+SUP vs RULE | +25.56% | 0.0020 (10/0/0) | +21.56% | 0.0020 (10/0/0) | -4.00 pp |
| C | 0.25 | 0.00 | ORACLE vs RULE | +36.16% | 0.0020 (10/0/0) | +36.16% | 0.0020 (10/0/0) | +0.00 pp |

## Direct paired test of the two review policies, same instances

The two arms score the same ten held-out instances under the same ten
seeds, so the two policies can be compared directly rather than through
their separate contrasts against the rule. `deployable lower` is positive
when the deployable policy gives lower true weighted tardiness; W/T/L
counts the deployable policy as the test arm. Ten paired instances put the
two-sided floor at p = 0.001953.

| Cell | beta | eps | decider | published TWT* | deployable TWT* | deployable lower | W/T/L | p |
|---|---|---|---|---|---|---|---|---|
| A | 0.20 | 0.00 | Correction layer (M0) | 1959.0 | 2028.7 | -3.56% | 2/0/8 | 0.0098 |
| A | 0.20 | 0.00 | Correction layer + supervisor (M0+SUP) | 1559.2 | 1665.6 | -6.82% | 0/0/10 | 0.0020 |
| A | 0.20 | 0.00 | Rule + supervisor (RULE+SUP) | 1715.0 | 1793.9 | -4.60% | 5/0/5 | 0.5566 |
| B | 0.20 | 0.25 | Correction layer (M0) | 2100.9 | 2153.3 | -2.49% | 2/0/8 | 0.0488 |
| B | 0.20 | 0.25 | Correction layer + supervisor (M0+SUP) | 1663.0 | 1769.2 | -6.39% | 3/0/7 | 0.0645 |
| B | 0.20 | 0.25 | Rule + supervisor (RULE+SUP) | 1757.1 | 1834.1 | -4.38% | 5/0/5 | 0.2754 |
| C | 0.25 | 0.00 | Correction layer (M0) | 1944.4 | 1980.4 | -1.85% | 3/0/7 | 0.1055 |
| C | 0.25 | 0.00 | Correction layer + supervisor (M0+SUP) | 1572.4 | 1669.4 | -6.17% | 1/0/9 | 0.0039 |
| C | 0.25 | 0.00 | Rule + supervisor (RULE+SUP) | 1703.0 | 1794.5 | -5.37% | 4/0/6 | 0.2754 |

## Share of the rule-to-reference gap closed

| Cell | beta | eps | published M0 | deployable M0 | difference | published M0+SUP | deployable M0+SUP | difference |
|---|---|---|---|---|---|---|---|---|
| A | 0.20 | 0.00 | 34.9% | 26.3% | -8.6 pp | 84.1% | 71.0% | -13.1 pp |
| B | 0.20 | 0.25 | 17.4% | 11.0% | -6.4 pp | 71.3% | 58.2% | -13.1 pp |
| C | 0.25 | 0.00 | 41.5% | 37.1% | -4.4 pp | 86.5% | 74.7% | -11.7 pp |

## Absolute ladder, mean TWT* over seeds

| Cell | policy | Tuned rule (RULE) | Rule + supervisor (RULE+SUP) | Correction layer (M0) | Correction layer + supervisor (M0+SUP) | Myopic full-information reference (ORACLE) |
|---|---|---|---|---|---|---|
| A (b=0.20, eps=0.00) | published | 2242.3 | 1715.0 | 1959.0 | 1559.2 | 1429.9 |
| A (b=0.20, eps=0.00) | deployable | 2242.3 | 1793.9 | 2028.7 | 1665.6 | 1429.9 |
| A (b=0.20, eps=0.00) | reference (split) | 2242.3 | 1715.0 | 1938.9 | 1544.9 | 1429.9 |
| B (b=0.20, eps=0.25) | published | 2242.3 | 1757.1 | 2100.9 | 1663.0 | 1429.9 |
| B (b=0.20, eps=0.25) | deployable | 2242.3 | 1834.1 | 2153.3 | 1769.2 | 1429.9 |
| B (b=0.20, eps=0.25) | reference (split) | 2242.3 | 1757.1 | 2109.7 | 1673.1 | 1429.9 |
| C (b=0.25, eps=0.00) | published | 2287.6 | 1703.0 | 1944.4 | 1572.4 | 1460.4 |
| C (b=0.25, eps=0.00) | deployable | 2287.6 | 1794.5 | 1980.4 | 1669.4 | 1460.4 |
| C (b=0.25, eps=0.00) | reference (split) | 2287.6 | 1703.0 | 1924.2 | 1556.8 | 1460.4 |

## Which policy ran: routing telemetry

The published policy produces no undetermined share at all, because it has
no stability test; the field's presence is the proof that the deployable
policy ran.

| Cell | beta | eps | reviewed fraction (M0+SUP) | reviewed fraction (RULE+SUP) | undetermined share (M0+SUP) | band half-width q | calibration examples |
|---|---|---|---|---|---|---|---|
| A | 0.20 | 0.00 | 0.2462 | 0.2485 | 0.8241 | 0.3179 | 16278 |
| B | 0.20 | 0.25 | 0.2450 | 0.2474 | 0.9756 | 1.0311 | 18196 |
| C | 0.25 | 0.00 | 0.2463 | 0.2482 | 0.8197 | 0.3067 | 16225 |

For comparison, the published-policy reviewed fractions at the same cells (results/y3_p9/cell_summary.json:supervisor_budget):

| Cell | reviewed fraction (M0+SUP) | reviewed fraction (RULE+SUP) | undetermined share |
|---|---|---|---|
| A | 0.2469 | 0.2468 | absent |
| B | 0.2480 | 0.2475 | absent |
| C | 0.2468 | 0.2467 | absent |

## Decomposition: routing rule vs conformal fold split (diagnostic)

The deployable policy needs a calibration fold the estimator is never
fitted on, which costs it about 30% of its training labels. The
`reference (split)` arm is the oracle-informed policy under the SAME fold
split, so the difference against the published column is the fold split
and the difference between the deployable and reference columns is the
routing rule. This is a diagnostic; it is not quoted in the manuscript.

| Cell | contrast | published | reference (split) | deployable | fold-split effect | routing-rule effect |
|---|---|---|---|---|---|---|
| A | M0 vs RULE | +12.63% | +13.53% | +9.53% | +0.90 pp | -4.00 pp |
| A | M0+SUP vs RULE+SUP | +9.08% | +9.92% | +7.15% | +0.83 pp | -2.76 pp |
| B | M0 vs RULE | +6.31% | +5.91% | +3.97% | -0.39 pp | -1.94 pp |
| B | M0+SUP vs RULE+SUP | +5.36% | +4.78% | +3.54% | -0.57 pp | -1.25 pp |
| C | M0 vs RULE | +15.00% | +15.89% | +13.43% | +0.88 pp | -2.46 pp |
| C | M0+SUP vs RULE+SUP | +7.67% | +8.58% | +6.97% | +0.91 pp | -1.61 pp |

The routing-rule effect is the negative of W1's price of deployability
(oracle-informed minus deployable, positive meaning the deployable policy
is worse). Here that price is +4.00, +1.94, +2.46 points for the layer
alone. The eight-cell contention grid of results/y3_w1 reported the same
quantity between -3.25 and +1.05 points, but every cell in that grid sits
at a recoverable share of 0.75 or 1.00. The price of deployability is
therefore larger at these low recoverable shares than anywhere the grid
measured it, which the grid could not have shown.

## Recovery quality of the fitted estimator (final DAgger iteration)

| Cell | beta | eps | Pearson r (published) | Pearson r (deployable) | sign acc (published) | sign acc (deployable) | zero-baseline acc |
|---|---|---|---|---|---|---|---|
| A | 0.20 | 0.00 | 0.1429 | 0.1907 | 0.6377 | 0.6219 | 0.3627 |
| B | 0.20 | 0.25 | 0.0693 | 0.0915 | 0.5776 | 0.5707 | 0.3627 |
| C | 0.25 | 0.00 | 0.1784 | 0.2480 | 0.6544 | 0.6424 | 0.3617 |

## Five-seed subset (301-305), consistency check only

| Cell | M0 vs RULE | M0+SUP vs RULE+SUP |
|---|---|---|
| A | +10.37% | +6.57% |
| B | +4.14% | +1.94% |
| C | +13.88% | +7.94% |

## Where the abstract's two numbers now stand, under one protocol

| Recoverable share | published review policy | deployable review policy |
|---|---|---|
| beta = 0.20, eps = 0.00 | +12.63% | +9.53% |
| beta = 0.20, eps = 0.25 | +6.31% | +3.97% |
| beta = 0.25, eps = 0.00 | +15.00% | +13.43% |
| beta = 1.00, eps = 0.00 | +45.36% | +48.15% |
