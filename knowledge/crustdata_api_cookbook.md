# Crustdata API cookbook (version 2025-11-01) - for the Sourcer

All calls: `Authorization: Bearer <key>`, `x-api-version: 2025-11-01`, JSON body. Base URL `https://api.crustdata.com`.

## Endpoints we use

| Endpoint | Use | Cost | Limit |
|---|---|---|---|
| `POST /person/search` | candidate pools, roster of the hiring company, ground-truth hires | 0.03 credits/result; 0.23 when `experience`/`education` are returned (default when `fields` is omitted) | 30 req/min |
| `POST /job/search` | past openings that become test cases (`metadata.date_added` = posting date) | 0.03/result; `limit: 0` + aggregations = free counts | 30 req/min |
| `POST /company/identify` | resolve a company name/domain/URL to `crustdata_company_id` | free, max 25 values, ONE identifier type per request | 30 req/min |
| `POST /company/enrich` | headcount and trailing growth (`headcount.growth_percent.{mom,qoq,six_months,yoy,two_years}`) | 2 credits/company | 15 req/min |
| `GET /account/credits` | balance (`account.credits`) | free | 300 req/min |
| `POST /person/search/autocomplete` | valid filter values (`field`, `query`, `limit<=100`) | cheap | 300 req/min |

There is no standalone title-normalisation endpoint: `basic_profile.normalized_title` is inline on profiles and describes the CURRENT title.

## Filter syntax

`filters` is one condition or a group: `{"op": "and"|"or", "conditions": [...]}` (groups nest).
A condition is `{"field": "...", "type": "<operator>", "value": ...}`.

Operators: `=`, `!=`, `in`, `not_in` (full values, arrays), `(.)` all words in any order (typo tolerant), `[.]` exact phrase,
`(!)` exclude phrase, `>`, `=>` (>=), `<`, `=<` (<=), `is_null`, `is_not_null`, `geo_distance`, `geo_exclude`, `geo_polygon`.
`|` inside a `(.)` value is NOT "or": use an `or` group.

Role paths: `experience.employment_details.current.*`, `.past.*`, or unqualified (any role, current or past).
A profile matches when ANY ONE role satisfies a condition. Two conditions on different fields may be satisfied by different roles.

Useful filter fields: `crustdata_person_id`, `basic_profile.location.{city,state,country,full_location}`,
`experience.employment_details.{title,company_id,start_date,end_date,seniority_level,function_category}`,
`experience.employment_details.current.company_professional_network_profile_url` (employer URLs only work with `current.` or `past.`),
`skills.professional_network_skills` (gated per account), `years_of_experience_raw`, `recently_changed_jobs`.

## Pagination and fields

`limit` 1..1000 (default 20). Send `sorts` for stable paging; follow `next_cursor` until `null`. Always pass `fields` in production
(omitting it returns experience + education for every row = 0.23 credits each). `crustdata_person_id` is always returned.
A 403 `permission_error` lists `denied_fields` / `permitted_fields`: resend with a narrower `fields`. 402 `insufficient_credits` is terminal.
429 = rate limit: wait `x-ratelimit-reset` seconds (plus jitter), cap attempts.

## Dates

Role dates are `YYYY-MM-DD` strings with no precision flag. A role "in 2025" may be `2025-01-01`; a month may be `YYYY-MM-01`.
Treat day-1 dates as month-precision and Jan-1 as year-precision (the Time Guard does).

## Data traps (each is a leak class)

1. **Profiles are today's snapshot.** Title, headline, summary, location, skills reflect NOW. Rebuild as-of T from the dated roles.
2. **Semantic/relevance ranking leaks.** `search.mode` semantic/hybrid ranks by the whole current profile (including the post-T job).
   Never use it to order candidates. Order retrieval by the neutral `crustdata_person_id`.
3. **`past.*` filters select people who have since moved.** For "held this title at T" use the unqualified path, never `past.`.
4. **Job records carry company data at index time**, not posting time. Do not use job-record headcount as point-in-time.
5. **Company enrichment has no history.** Only trailing growth windows relative to today: usable live, not in a backtest.
6. **`recently_changed_jobs`, `years_at_company*`, `normalized_title`, `headline`, `summary`** describe today: never feed them to a backtest agent.

## Retrieval recipes

- *Cardinality probe*: same filters with `limit: 1`, `fields: ["crustdata_person_id"]` returns `total_count` for ~0.03 credits.
  Narrow until the universe is enumerable (`<= pool_cap`), broaden if it is tiny.
- *Title expansion*: OR-group of `(.)` matches on the unqualified title path over the target title, its aliases and adjacent families
  (backend engineer -> software engineer, full stack engineer, platform engineer ...).
- *Location*: use the whole metro cluster (Bengaluru = Bangalore = Bengaluru Urban). In backtests keep API location coarse (country),
  because today's location of a hire may already be the job's city; re-apply the city on the masked role location.
- *Roster of a company*: `experience.employment_details.company_id in [id]` (any role). Mask to T, keep people employed at T.
- *Hires after T*: `experience.employment_details.current.company_id = X` AND `current.start_date => T` (ground truth only).
