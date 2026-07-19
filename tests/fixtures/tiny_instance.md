# tiny_instance.json — hand-derived optimum

8 work orders, 2 trades, 3 technicians. All values are exact multiples of 0.01
(integers plus the 171.4 SLA constant), so CP-SAT's centi-bh rounding is lossless
and the model objective, the reported schedule's WWT and the validator agree
exactly. Every `due_bh = release_bh + SLA(priority)` with the locked SLA windows
(P1 8 bh, P2 24 bh, P3 80 bh, P4 171.4 bh) and weights (P1 8, P2 4, P3 2, P4 1).

Because a technician serves exactly one trade and eligibility is exact
trade-match, the two trades are **independent** single-/parallel-machine
subproblems, and the instance optimum is the sum of the two trade optima.

## Trade D20 — 2 identical technicians (T0, T1)

| WO   | p | release | due | prio | w |
|------|---|---------|-----|------|---|
| WO01 | 6 | 0       | 8   | 1    | 8 |
| WO02 | 6 | 0       | 8   | 1    | 8 |
| WO03 | 6 | 0       | 8   | 1    | 8 |
| WO04 | 2 | 0       | 171.4 | 4  | 1 |

Three P1 jobs of length 6 on two machines: by pigeonhole at least two of them
share a machine, so on that machine the *second* one completes no earlier than
`6 + 6 = 12`, i.e. it is tardy by `12 - 8 = 4` → cost `8 * 4 = 32`. The third P1
runs alone on the other machine, completing at 6 (on time). WO04 (loose P4)
slots into idle time (e.g. after the single P1) and is never tardy. So

**WWT(D20) = 32**, and 32 is a tight lower bound (independent of WO04).
Achieved by e.g. T0: WO01[0–6], WO02[6–12]; T1: WO03[0–6], WO04[6–8].

## Trade E10 — 1 technician (T2)

| WO   | p | release | due   | prio | w |
|------|---|---------|-------|------|---|
| WO05 | 8 | 0       | 171.4 | 4    | 1 |
| WO06 | 4 | 1       | 9     | 1    | 8 |
| WO07 | 3 | 2       | 26    | 2    | 4 |
| WO08 | 2 | 2       | 173.4 | 4    | 1 |

This is the classic **non-delay trap**. At bh 0 the only released E10 job is the
long, low-priority WO05 (p=8). WO06 (the urgent P1, due 9) releases at bh 1.

* CP-SAT (may insert idle) schedules WO06 early — e.g. WO08[0–2], WO06[2–6]
  (≤ due 9), WO07[6–9] (≤ 26), WO05[9–17] (≤ 171.4). Everything on time →
  **WWT(E10) = 0** (a lower bound, since tardiness ≥ 0).
* A dispatching rule builds a **non-delay** schedule: it may not idle T2 at bh 0
  while WO05 is waiting, so it must start WO05[0–8]. WO06 is then forced to
  [8–12], tardy by `12 - 9 = 3` → cost `8 * 3 = 24`; WO07[12–15] and WO08[15–17]
  stay on time. Every deterministic rule (edd, wspt, atc, pfifo, mor) picks WO06
  first once T2 frees at bh 8 (it is the most urgent, highest-ratio, highest-
  priority and longest of the three waiting jobs), so all of them score
  **WWT(E10) = 24**. 'random' is ≥ 24 (it may delay WO06 further).

## Instance optimum and expected rule table

| method        | WWT | note |
|---------------|-----|------|
| **CP-SAT**    | **32** | optimum = 32 (D20) + 0 (E10); status OPTIMAL |
| edd           | 56  | 32 + 24 |
| wspt          | 56  | 32 + 24 |
| atc           | 56  | 32 + 24 |
| pfifo         | 56  | 32 + 24 |
| mor           | 56  | 32 + 24 |
| random (301)  | ≥ 56 | D20 = 32 always; E10 ≥ 24 |

**Hand-derived optimal WWT = 32.** The 32→56 gap between the exact optimum and
every dispatching rule is the non-delay suboptimality on trade E10 — the reason
the benchmark keeps CP-SAT as a strong baseline against the rules (and, later,
against the learned dispatcher).
