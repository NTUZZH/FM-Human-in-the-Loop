# B5 candidate-cap ablation summary

Source: `cap256.csv`. v2 policies (seed 301/302/303) on the contended cells (replay-tight m in {0.6,0.8} + storm2 u>=1.0, campuses {5,9,10,12}). Cap raised 64 -> 256 at EVAL only (no retraining); features are cap-independent (feature 11 and the context divide by the full-queue work, not the candidate slice), so raising the cap only exposes more candidates. Caveat: the policy was trained at cap=64, so cap=256 is a mild train/eval distribution shift. Both caps rerun through THIS script for an apples-to-apples pairing.

## Paired cap64 vs cap256 (per v2 seed, over 1886 contended configs)

diff = TWT(cap256) - TWT(cap64); negative => cap256 helps. Wilcoxon signed-rank over the non-tied pairs.

| seed | n | n_tied | mean diff | median diff | wins(256<64) | losses | Wilcoxon p | mean TWT c64 | mean TWT c256 |
|---|---|---|---|---|---|---|---|---|---|
| rl301 | 1886 | 1662 | +1.5367 | +0.0000 | 91 | 133 | 1.23e-05 | 1035.450 | 1036.987 |
| rl302 | 1886 | 1690 | +0.0941 | +0.0000 | 79 | 117 | 0.0212 | 1013.006 | 1013.100 |
| rl303 | 1886 | 1684 | +49.2136 | +0.0000 | 43 | 159 | 7.65e-22 | 1215.286 | 1264.500 |

## Verdict-flip: RL vs best-PDR on each config

For each (config, seed): is RL's TWT <= best-PDR TWT (RL wins)? Counts at cap=64 vs cap=256; 'flips' = configs that switch from RL-loses at cap64 to RL-wins at cap256.

| seed | RL-wins @cap64 | RL-wins @cap256 | flips (lose->win) | un-flips (win->lose) |
|---|---|---|---|---|
| rl301 | 1622/1886 | 1608/1886 | 0 | 14 |
| rl302 | 1644/1886 | 1642/1886 | 5 | 7 |
| rl303 | 1498/1886 | 1496/1886 | 0 | 2 |

## Mean cap256-cap64 TWT diff by regime (pooled over seeds)

| regime | n_configs | mean diff | mean |diff| |
|---|---|---|---|
| replay-tight | 1526 | +0.0083 | 0.0083 |
| storm2 | 360 | +88.7544 | 93.5140 |
