# Results (train)

> Backends: **crust=sim, llm=mock**. **SYNTHETIC: simulator + mock LLM. These numbers verify the harness, not Crustdata or Claude.**

Split: `train`, cases: **28**, config hash `f2791720c8e239d2`, run at 2026-10-31T13:47:37. Intervals are 95% bootstrap over cases (1,000 resamples).

| Variant | Pool recall | Recall@10 | Recall@25 | Recall@50 | MRR |
|---|---|---|---|---|---|
| A0 keyword (BM25 on masked text) | 79% [61%-93%] | 4% [0%-11%] | 7% [0%-18%] | 14% [4%-29%] | 0.046 [0.007-0.120] |
| A1 Fit only | 79% [61%-93%] | 7% [0%-18%] | 25% [11%-39%] | 25% [11%-39%] | 0.033 [0.012-0.061] |
| A2 Fit + Movability | 79% [61%-93%] | 11% [0%-21%] | 25% [11%-39%] | 25% [11%-39%] | 0.058 [0.013-0.132] |
| A3 Fit + Connection | 79% [61%-93%] | 29% [11%-46%] | 46% [29%-68%] | 64% [46%-82%] | 0.114 [0.052-0.201] |
| A4 Full (F+M+C) | 79% [61%-93%] | 25% [11%-43%] | 46% [29%-68%] | 64% [46%-82%] | 0.115 [0.051-0.203] |
| A5 learned ranker | 79% [61%-93%] | 43% [25%-61%] | 61% [39%-79%] | 71% [54%-89%] | 0.163 [0.089-0.253] |
| A4 + Critic | 79% [61%-93%] | 25% [11%-43%] | 46% [29%-68%] | 64% [46%-82%] | 0.116 [0.051-0.204] |
| random baseline | 79% | 3% | 8% | 16% | 0.019 |

## A vs B, paired by case (the headline is A4 vs A1)

| A | B | metric | mean diff [95% CI] | wins/losses/ties | sign-flip p | McNemar p | verdict |
|---|---|---|---|---|---|---|---|
| A4 | A1 | recall@25 | +0.214 [+0.071, +0.393] | 7/1/20 | 0.069 (Holm 0.276) | 0.070 | clear gain |
| A4 | A1 | recall@10 | +0.179 [+0.000, +0.357] | 6/1/21 | 0.124 | 0.125 | no clear gain (interval includes zero) |
| A4 | A1 | mrr | +0.082 [+0.021, +0.160] | 20/2/6 | 0.016 | - | clear gain |
| A2 | A1 | recall@25 | +0.000 [+0.000, +0.000] | 0/0/28 | 1.000 (Holm 1.000) | 1.000 | no clear gain (interval includes zero) |
| A2 | A1 | recall@10 | +0.036 [+0.000, +0.107] | 1/0/27 | 1.000 | 1.000 | no clear gain (interval includes zero) |
| A2 | A1 | mrr | +0.025 [-0.000, +0.074] | 12/7/9 | 0.218 | - | no clear gain (interval includes zero) |
| A3 | A1 | recall@25 | +0.214 [+0.071, +0.394] | 7/1/20 | 0.069 (Holm 0.276) | 0.070 | clear gain |
| A3 | A1 | recall@10 | +0.214 [+0.036, +0.393] | 7/1/20 | 0.073 | 0.070 | clear gain |
| A3 | A1 | mrr | +0.081 [+0.020, +0.155] | 20/2/6 | 0.015 | - | clear gain |
| A4 | A0 | recall@25 | +0.393 [+0.179, +0.607] | 12/1/15 | 0.004 (Holm 0.022) | 0.003 | clear gain |
| A4 | A0 | recall@10 | +0.214 [+0.036, +0.393] | 7/1/20 | 0.071 | 0.070 | clear gain |
| A4 | A0 | mrr | +0.069 [-0.044, +0.175] | 21/1/6 | 0.254 | - | no clear gain (interval includes zero) |
| A5 | A1 | recall@25 | +0.357 [+0.143, +0.571] | 11/1/16 | 0.006 (Holm 0.029) | 0.006 | clear gain |
| A5 | A1 | recall@10 | +0.357 [+0.179, +0.537] | 10/0/18 | 0.002 | 0.002 | clear gain |
| A5 | A1 | mrr | +0.130 [+0.055, +0.215] | 19/3/6 | 0.002 | - | clear gain |
| A4C | A4 | recall@25 | +0.000 [+0.000, +0.000] | 0/0/28 | 1.000 (Holm 1.000) | 1.000 | no clear gain (interval includes zero) |
| A4C | A4 | recall@10 | +0.000 [+0.000, +0.000] | 0/0/28 | 1.000 | 1.000 | no clear gain (interval includes zero) |
| A4C | A4 | mrr | +0.000 [+0.000, +0.001] | 5/0/23 | 0.064 | - | clear gain |

## Where the system fails (A4 misses beyond rank 25)

| reason | count |
|---|---|
| NOT_IN_POOL | 6 |
| GT_NOISE | 1 |
| RANKED_LOW_FIT | 6 |
| RANKED_LOW_SIGNAL | 2 |

## weight_grid_train_only

```json
{
  "grid": [
    {
      "w_M": 0.0,
      "w_C": 0.0,
      "recall@25": 0.25,
      "mrr": 0.03299787561174909,
      "recall@10": 0.07142857142857142
    },
    {
      "w_M": 0.0,
      "w_C": 0.25,
      "recall@25": 0.2857142857142857,
      "mrr": 0.0794153065004312,
      "recall@10": 0.21428571428571427
    },
    {
      "w_M": 0.0,
      "w_C": 0.5,
      "recall@25": 0.35714285714285715,
      "mrr": 0.10227890017212218,
      "recall@10": 0.25
    },
    {
      "w_M": 0.0,
      "w_C": 1.0,
      "recall@25": 0.4642857142857143,
      "mrr": 0.11362448795909538,
      "recall@10": 0.2857142857142857
    },
    {
      "w_M": 0.25,
      "w_C": 0.0,
      "recall@25": 0.25,
      "mrr": 0.05848827547698889,
      "recall@10": 0.10714285714285714
    },
    {
      "w_M": 0.25,
      "w_C": 0.25,
      "recall@25": 0.21428571428571427,
      "mrr": 0.08635520418465958,
      "recall@10": 0.21428571428571427
    },
    {
      "w_M": 0.25,
      "w_C": 0.5,
      "recall@25": 0.35714285714285715,
      "mrr": 0.10401084182128915,
      "recall@10": 0.25
    },
    {
      "w_M": 0.25,
      "w_C": 1.0,
      "recall@25": 0.4642857142857143,
      "mrr": 0.11539162381168899,
      "recall@10": 0.25
    },
    {
      "w_M": 0.5,
      "w_C": 0.0,
      "recall@25": 0.17857142857142858,
      "mrr": 0.0392518400623608,
      "recall@10": 0.10714285714285714
    },
    {
      "w_M": 0.5,
      "w_C": 0.25,
      "recall@25": 0.21428571428571427,
      "mrr": 0.10832461555987073,
      "recall@10": 0.21428571428571427
    },
    {
      "w_M": 0.5,
      "w_C": 0.5,
      "recall@25": 0.35714285714285715,
      "mrr": 0.1042754706664523,
      "recall@10": 0.21428571428571427
    },
    {
      "w_M": 0.5,
      "w_C": 1.0,
      "recall@25": 0.39285714285714285,
      "mrr": 0.12088530698760633,
      "recall@10": 0.25
    },
    {
      "w_M": 1.0,
      "w_C": 0.0,
      "recall@25": 0.10714285714285714,
      "mrr": 0.027536203023738666,
      "recall@10": 0.10714285714285714
    },
    {
      "w_M": 1.0,
      "w_C": 0.25,
      "recall@25": 0.25,
      "mrr": 0.08769621951343899,
      "recall@10": 0.21428571428571427
    },
    {
      "w_M": 1.0,
      "w_C": 0.5,
      "recall@25": 0.35714285714285715,
      "mrr": 0.10307708774222492,
      "recall@10": 0.21428571428571427
    },
    {
      "w_M": 1.0,
      "w_C": 1.0,
      "recall@25": 0.42857142857142855,
      "mrr": 0.10265582906309689,
      "recall@10": 0.2857142857142857
    }
  ],
  "best": {
    "w_M": 0.25,
    "w_C": 1.0,
    "recall@25": 0.4642857142857143,
    "mrr": 0.11539162381168899,
    "recall@10": 0.25
  }
}
```

## learned_ranker

```json
{
  "lambda": 0.03,
  "cv_mean_rr": {
    "0.003": 0.19096010377792527,
    "0.03": 0.19937996175140763,
    "0.3": 0.18780696998042504,
    "3.0": 0.1891102907030347
  },
  "weights": {
    "fit.title_match": 0.012,
    "fit.seniority": 0.098,
    "fit.skills": -0.092,
    "fit.location": 0.281,
    "fit.domain": 0.04,
    "fit.text_sim": 0.055,
    "fit.llm_fit": 0.029,
    "move.p_move": 0.165,
    "move.personal": -0.018,
    "move.no_recent_promo": -0.045,
    "move.stalled": 0.035,
    "move.between_roles": -0.13,
    "conn.ex_colleague": 0.822,
    "conn.closeness": 1.336,
    "conn.feeder": 0.053,
    "conn.ppr": 0.604
  }
}
```

## power_note

```json
{
  "n_cases": 28,
  "power_to_detect_+0.15_recall@25_(independent-pairs, conservative)": 0.147
}
```

## credits_and_time

```json
{
  "credits_estimated_total": 0.0,
  "credits_saved_by_cache": 0.0,
  "per_case_seconds_mean": 3.52,
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
    "requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "credits_estimated": 0.0,
    "credits_saved_by_cache": 0.0,
    "faults": {}
  }
}
```

## pool_stats

```json
{
  "mean_pool": 330.2,
  "mean_universe": 768.1,
  "cases_with_multiple_hires": 8,
  "share_roles_dropped_ambiguous": 0.0188
}
```
