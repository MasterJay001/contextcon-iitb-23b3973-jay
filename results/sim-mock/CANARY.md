# Canary results

Backends: `{'crust': 'sim', 'llm': 'mock'}`  split `train`  cases 28

## 1. Leak-on vs leak-off (same deterministic ranker)

| arm | variant | pool recall | R@10 | R@25 | R@50 |
|---|---|---|---|---|---|
| leak on | A1 | 100% | 39% | 54% | 61% |
| masked | A1 | 79% | 7% | 25% | 25% |
| leak on | A4 | 100% | 89% | 100% | 100% |
| masked | A4 | 79% | 21% | 36% | 54% |

Leak-on Recall@25 54% vs masked 25% (drop +29%). The leak is real and is now closed.

## 2. Shuffled answers

```json
{
  "observed_recall@25": 0.35714285714285715,
  "random_pseudo_hires_mean_recall@25": 0.10616071428571427,
  "random_sd": 0.05843127558796517,
  "analytic_baseline_recall@25": 0.104030153852354,
  "p_observed_ge_null": 0.0,
  "evaluator_ok": true,
  "cross_case_swap_recall@25": 0.07142857142857142,
  "verdict": "Evaluator is unbiased: random pseudo-hires hit the analytic baseline."
}
```

## 3. Spot audit

38 real hires machine-checked; failures: 0. See SPOT_AUDIT.md for five rendered profiles.

## 4. Feature AUC scan (hire vs pool, masked)

no single feature separates hires from the pool (AUC <= 0.97)

| feature | mean AUC |
|---|---|
| fit.title_match | 0.52 |
| fit.seniority | 0.50 |
| fit.skills | 0.46 |
| fit.location | 0.57 |
| fit.domain | 0.52 |
| fit.text_sim | 0.55 |
| fit.llm_fit | 0.49 |
| move.p_move | 0.64 |
| move.personal | 0.48 |
| move.no_recent_promo | 0.51 |
| move.stalled | 0.49 |
| move.between_roles | 0.47 |
| conn.ex_colleague | 0.86 |
| conn.closeness | 0.89 |
| conn.feeder | 0.53 |
| conn.ppr | 0.86 |

## 5. Without undated fields (skills, profile location, role descriptions)

| arm | R@25 | MRR |
|---|---|---|
| A1_with_skills_location_descriptions | 25% | 0.033 |
| A1_without | 14% | 0.022 |
| A4_with_skills_location_descriptions | 36% | 0.104 |
| A4_without | 54% | 0.124 |
