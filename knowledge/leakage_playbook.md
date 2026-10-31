# Leakage playbook - every way the answer can sneak in, and its fix

Leakage is any path by which the answer (the real hire) reaches the inputs. The project is credible only if every path is closed and proven closed.

| # | Leak | How it sneaks in | Fix |
|---|---|---|---|
| 1 | Post-T jobs | the hire's profile now shows the target company as current employer | drop roles starting after T; roles that ended after T become ongoing at T |
| 2 | Free text | headline/summary says "Engineer at Target" | never copy headline/summary; recompute the title from masked roles |
| 3 | Retrieval filters | an API title filter matches the hire through the post-T title | re-apply every filter locally on the masked profile (LocalFilter) |
| 4 | Semantic ranking | API ranks by the full current profile | neutral `crustdata_person_id` order only; rank on masked data |
| 5 | Fuzzy dates | start stored as year only (2025-01-01) | infer precision; if the interval straddles T drop the role; ambiguous ends become ongoing |
| 6 | Undated fields | skills and location reflect today | keep but flag; robustness run without skills/location/descriptions; location_at_T prefers dated role location |
| 7 | Company size | current headcount used as headcount at T | no as-of-T series exists: headcount trend is live-only, off in backtests |
| 8 | Connection features | counting target employees who joined after T | roster = start <= T and end empty or after T |
| 9 | LLM memory | the model recognises famous people | send person_id + masked fields only; no names, headlines or URLs |
| 10 | Tuning on test | weights/prompts tuned on reported cases | by-company split; freeze weights before the one held-out run |
| 11 | Brief leaks the hire | Route B brief copied from the hire's profile | brief uses only company, normalised title, seniority, location |
| 12 | Retrieval bias (ours) | API filters evaluate TODAY's profile, so the hire is over-retrieved via the post-T role | unqualified role paths only; coarse API location; enumerate the universe; report pool recall separately |

## Canary tests (prove it)

1. **Leak-on vs leak-off**: the same ranker on unmasked then masked profiles. The drop in recall is the size of the leak you closed.
2. **Shuffled answers**: reassign hires randomly across cases; recall must fall to the random baseline (k / pool size).
3. **Spot audit**: print masked hire profiles; the target company must not appear in any ongoing or post-T role or in free text.
4. **Feature AUC scan**: no single feature may separate hires from the pool with AUC near 1.0 on masked data.
5. **Time shift**: shifting T earlier must not increase recall.
