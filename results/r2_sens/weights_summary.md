# A4 priority-weight-vector sweep summary

Source: `weights.csv`. Six PDRs on the E5 base set (campus {5,9,10,12} x size {150,400} x first-30 replay-test = 240 instances). Tardiness weights mapped by priority (P1,P2,P3,P4): baseline (8,4,2,1), flat (4,3,2,1), steep (27,9,3,1). The vector enters both the WSPT/ATC scores (dispatch decisions change) and the objective. All schedules travel=0 -> feasible -> scored by the independent validator.

## Ranking robustness (Kendall tau-b, baseline weights vs sweep)

Per-cell tau on the mean-TWT-per-method vectors; '-' = degenerate cell (baseline ranking fully tied, capacity-adequate campus 5). Pooled tau is the scale-free cross-cell average-rank tau (tab_sensitivity.tex methodology).

| weights | c5/150 | c5/400 | c9/150 | c9/400 | c10/150 | c10/400 | c12/150 | c12/400 | mean cell | pooled |
|---|---|---|---|---|---|---|---|---|---|---|
| flat | - | - | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| steep | - | - | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.86 | 0.98 | 1.00 |

### Verdict -- does the leaderboard survive?

Baseline best-ranked method (lowest pooled average rank): **edd**.
* **flat**: pooled tau = 1.00; best method now **edd** (unchanged).
* **steep**: pooled tau = 1.00; best method now **edd** (unchanged).
