# Results (heldout_final)

> Backends: **crust=sim, llm=mock**. **SYNTHETIC: simulator + mock LLM. These numbers verify the harness, not Crustdata or Claude.**

Split: `heldout`, cases: **18**, config hash `f2791720c8e239d2`, run at 2026-10-31T13:49:36. Intervals are 95% bootstrap over cases (1,000 resamples).

| Variant | Pool recall | Recall@10 | Recall@25 | Recall@50 | MRR |
|---|---|---|---|---|---|
| A0 keyword (BM25 on masked text) | 78% [56%-94%] | 6% [0%-17%] | 6% [0%-17%] | 11% [0%-28%] | 0.012 [0.004-0.023] |
| A1 Fit only | 78% [56%-94%] | 6% [0%-17%] | 11% [0%-28%] | 11% [0%-28%] | 0.017 [0.004-0.039] |
| A2 Fit + Movability | 78% [56%-94%] | 0% [0%-0%] | 11% [0%-28%] | 11% [0%-28%] | 0.014 [0.004-0.026] |
| A3 Fit + Connection | 78% [56%-94%] | 11% [0%-28%] | 39% [17%-61%] | 50% [28%-72%] | 0.051 [0.020-0.090] |
| A4 Full (F+M+C) | 78% [56%-94%] | 17% [0%-33%] | 39% [17%-61%] | 50% [28%-72%] | 0.055 [0.022-0.097] |
| A5 learned ranker | 78% [56%-94%] | 33% [11%-56%] | 61% [39%-83%] | 67% [44%-89%] | 0.132 [0.045-0.255] |
| A4 + Critic | 78% [56%-94%] | 17% [0%-33%] | 39% [17%-61%] | 50% [28%-72%] | 0.055 [0.022-0.098] |
| random baseline | 78% | 3% | 7% | 14% | 0.017 |

## A vs B, paired by case (the headline is A4 vs A1)

| A | B | metric | mean diff [95% CI] | wins/losses/ties | sign-flip p | McNemar p | verdict |
|---|---|---|---|---|---|---|---|
| A4 | A1 | recall@25 | +0.278 [+0.000, +0.556] | 7/2/9 | 0.182 (Holm 0.727) | 0.180 | no clear gain (interval includes zero) |
| A4 | A1 | recall@10 | +0.111 [-0.111, +0.333] | 3/1/14 | 0.625 | 0.625 | no clear gain (interval includes zero) |
| A4 | A1 | mrr | +0.038 [-0.003, +0.085] | 12/2/4 | 0.122 | - | no clear gain (interval includes zero) |
| A2 | A1 | recall@25 | +0.000 [+0.000, +0.000] | 0/0/18 | 1.000 (Holm 1.000) | 1.000 | no clear gain (interval includes zero) |
| A2 | A1 | recall@10 | -0.056 [-0.167, +0.000] | 0/1/17 | 1.000 | 1.000 | no clear gain (interval includes zero) |
| A2 | A1 | mrr | -0.004 [-0.016, +0.005] | 7/5/6 | 0.908 | - | no clear gain (interval includes zero) |
| A3 | A1 | recall@25 | +0.278 [-0.001, +0.556] | 7/2/9 | 0.182 (Holm 0.727) | 0.180 | no clear gain (interval includes zero) |
| A3 | A1 | recall@10 | +0.056 [-0.111, +0.222] | 2/1/15 | 1.000 | 1.000 | no clear gain (interval includes zero) |
| A3 | A1 | mrr | +0.034 [-0.004, +0.078] | 12/2/4 | 0.151 | - | no clear gain (interval includes zero) |
| A4 | A0 | recall@25 | +0.333 [+0.056, +0.611] | 7/1/10 | 0.073 (Holm 0.364) | 0.070 | clear gain |
| A4 | A0 | recall@10 | +0.111 [-0.111, +0.333] | 3/1/14 | 0.633 | 0.625 | no clear gain (interval includes zero) |
| A4 | A0 | mrr | +0.043 [+0.006, +0.088] | 12/2/4 | 0.033 | - | clear gain |
| A5 | A1 | recall@25 | +0.500 [+0.167, +0.778] | 11/2/5 | 0.023 (Holm 0.138) | 0.022 | clear gain |
| A5 | A1 | recall@10 | +0.278 [+0.000, +0.556] | 6/1/11 | 0.125 | 0.125 | no clear gain (interval includes zero) |
| A5 | A1 | mrr | +0.115 [+0.024, +0.242] | 12/2/4 | 0.016 | - | clear gain |
| A4C | A4 | recall@25 | +0.000 [+0.000, +0.000] | 0/0/18 | 1.000 (Holm 1.000) | 1.000 | no clear gain (interval includes zero) |
| A4C | A4 | recall@10 | +0.000 [+0.000, +0.000] | 0/0/18 | 1.000 | 1.000 | no clear gain (interval includes zero) |
| A4C | A4 | mrr | +0.000 [+0.000, +0.000] | 1/0/17 | 1.000 | - | no clear gain (interval includes zero) |

## Where the system fails (A4 misses beyond rank 25)

| reason | count |
|---|---|
| NOT_IN_POOL | 4 |
| GT_NOISE | 1 |
| RANKED_LOW_FIT | 4 |
| RANKED_LOW_SIGNAL | 2 |

## power_note

```json
{
  "n_cases": 18,
  "power_to_detect_+0.15_recall@25_(independent-pairs, conservative)": 0.074
}
```

## credits_and_time

```json
{
  "credits_estimated_total": 340.9,
  "credits_saved_by_cache": 2390.6,
  "per_case_seconds_mean": 3.07,
  "llm": {
    "calls": 0,
    "cache_hits": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "usd": 0.0,
    "failures": 0,
    "by_role": {}
  },
  "crust": {
    "requests": 78,
    "cache_hits": 60,
    "cache_misses": 18,
    "credits_estimated": 340.86,
    "credits_saved_by_cache": 2390.6,
    "faults": {}
  }
}
```

## pool_stats

```json
{
  "mean_pool": 331.7,
  "mean_universe": 831.3,
  "cases_with_multiple_hires": 4,
  "share_roles_dropped_ambiguous": 0.0193
}
```
