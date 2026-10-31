# Self-improvement (TRAIN only, Goodhart-guarded)

Backends `{'crust': 'sim', 'llm': 'mock'}`; analyst source above; strict guard: **True**; shadow held-out: **True** (diagnostic only).

| round | w_M | w_C | lessons | train hits@25 | train MRR | held-out R@25 (shadow) |
|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 10/28 | 0.104 | 0.28 |
| 1 | 0.5 | 1.0 | 0 | 11/28 | 0.121 | 0.39 |
| 2 | 0.5 | 1.0 | 0 | 11/28 | 0.121 | 0.39 |
| 3 | 0.5 | 1.0 | 0 | 11/28 | 0.121 | 0.39 |

## Decisions

- round 1: `weight` w_C  -> **ACCEPTED** (missed hires sit high on connection signals (mean pool percentile 0.76))
- round 1: `rubric_sentence`  Candidates one role family away from the target (for example backend, full stack and platform engineering) are plausible hires; do not score them below 3 on title alone. -> **REVERTED** (missed hires look weak on literal title/fit (mean pool percentile 0.32) but were hired)
- round 2: `rubric_sentence`  Candidates one role family away from the target (for example backend, full stack and platform engineering) are plausible hires; do not score them below 3 on title alone. -> **REVERTED** (missed hires look weak on literal title/fit (mean pool percentile 0.40) but were hired)
- round 3: `rubric_sentence`  Candidates one role family away from the target (for example backend, full stack and platform engineering) are plausible hires; do not score them below 3 on title alone. -> **REVERTED** (missed hires look weak on literal title/fit (mean pool percentile 0.40) but were hired)
