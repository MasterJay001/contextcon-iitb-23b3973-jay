# Robustness suite

Backends: crust=`sim`, llm=`mock`; config `f2791720c8e239d2`; **24/24 passed**.

| group | case | expected behaviour | result | detail | s |
|---|---|---|---|---|---|
| A inputs | vague brief | asks ONE clarifying question; no search | PASS | blocked, question='Which role are you hiring for, and in which city?', credits spent=0.0 | 1.4 |
| A inputs | mixed language + typos | Backend Engineer, Mumbai, payments | PASS | title='Backend Engineer' location='Mumbai' domain=['payments'] via heuristic | 0.0 |
| A inputs | title with many names | expands to adjacent normalised titles | PASS | title='Growth Marketer' adjacent=['growth hacker', 'data analyst', 'marketing manager', 'performance marketer', 'product manager'] | 0.0 |
| A inputs | ultra-niche role | fewer than five, with a reason | PASS | 5 cards; warnings=['no hiring company given: Connection signal skipped'] | 1.7 |
| A inputs | impossible combination | flags the contradiction, no blind search | PASS | blocked: The brief contradicts itself: 10 years of experience at a 2-year-old company is impossible; 10 years of experience at a 2-year-old company is impossible | 1.6 |
| A inputs | unresolvable company | proceeds without Connection and says so | PASS | cards=5 warnings=["hiring company 'Zzzzqx Nonexistent Technologies' could not be resolved: continuing without the Connection signal"] | 1.6 |
| A inputs | prompt injection in brief | ignored; normal shortlist; no contact data | PASS | injection_detected=True, cards=5, contact data in output: False | 2.9 |
| A inputs | empty / whitespace brief | clarifying question, no crash | PASS | both blocked with a clarifying question | 1.7 |
| A inputs | 50k-character brief | truncated, no crash | PASS | sanitised to 4000 chars; spec='Backend Engineer' | 2.0 |
| A inputs | non-English brief (Hindi) | no crash | PASS | handled without crash; title='' blocked=True | 1.4 |
| A inputs | PII in brief | redacted, never echoed | PASS | assumptions=['2 contact detail(s) in the brief were redacted and ignored'] | 0.0 |
| A inputs | remote role | proceeds without a city | PASS | title='Backend Engineer' remote_ok=True | 0.0 |
| A inputs | unknown role | asks for the role | PASS | blocked: Which role are you hiring for, and in which city? | 1.3 |
| A inputs | contradictory seniority | flags the contradiction | PASS | contradictions=['Intern-level role asking for 12+ years of experience'] | 0.0 |
| A inputs | markup / emoji / script tag | no crash | PASS | title='Backend Engineer' | 2.4 |
| B failures | 429 rate limit on ~1/3 calls | backoff + retry; run completes | PASS | cards=5 injected 429s=5 | 13.3 |
| B failures | timeout / 500 on ~1/5 calls | retry twice, continue with partial pool, flag it | PASS | 3 runs completed, injected 5xx=5, cards=[5, 5, 5], degradation notes=[] | 28.2 |
| B failures | forced empty search results | never crash; says no candidates | PASS | cards=0 warnings=['no candidates passed the local filter', 'only 0 candidates matched; returning fewer than 5 instead of padding'] | 7.7 |
| B failures | malformed LLM JSON on ~1/4 calls | repair once, else degrade; run completes | PASS | cards=5 llm failures=2 injected=2 | 8.4 |
| B failures | credit cap lowered to 50 | stops cleanly with best-so-far + warning | PASS | spent=50.0/50 cards=5 warning=['page truncated by credit cap'] | 6.8 |
| B failures | 402 credits exhausted | terminal: stops, warns, no crash | PASS | cards=0 warnings=['credits exhausted (402): not enough credits', 'no candidates passed the local filter'] | 2.9 |
| B failures | 403 denied fields (skills) | narrow fields automatically and continue | PASS | narrowed: ["403: dropped denied fields ['skills']"] | 10.5 |
| B failures | Time Guard violation injected | fail-closed: leaking profile is dropped | PASS | fail-closed: pool=0; leaking profiles dropped | 4.6 |
| B failures | malformed profile payloads | tolerant parsing, no crash | PASS | parsed 4 of 6 malformed rows without raising | 0.0 |
