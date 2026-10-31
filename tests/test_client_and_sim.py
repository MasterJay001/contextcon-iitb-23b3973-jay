"""Client resilience (rule 10) and simulator fidelity to the documented API."""
from __future__ import annotations

import datetime as dt

import pytest

from crust.client import (CreditCapReached, CrustBadRequest, CrustClient, CrustCreditsExhausted, CrustPermissionError,
                          CrustRateLimited, CrustUnavailable, FaultInjector, RateLimiter)
from crust.companies import CompaniesAPI
from crust.jobs import JobsAPI
from crust.people import AND, PeopleAPI, cond, company_filter, location_filter, title_filter
from crust.sim import SimTransport, eval_filter, resolve
from crust.sim_world import SimWorld


def _client(cfg, world, faults=None, cache=None, credits=15000.0, **kw):
    tr = SimTransport(world, credits=credits)
    c = CrustClient(cfg, tr, api_key="k", use_cache=cache is not None, cache_dir=cache, faults=faults or FaultInjector({}),
                    sleep=lambda s: None, **kw)
    return c, tr


def test_search_paginates_in_neutral_order_and_totals(cfg, world):
    c, tr = _client(cfg, world)
    api = PeopleAPI(c, cfg)
    flt = AND(title_filter(["software engineer", "backend engineer"]), location_filter(["India"]))
    res = api.search(flt, fields=["crustdata_person_id", "experience"], max_results=250, page_size=100)
    ids = [p.person_id for p in res.profiles]
    assert len(ids) == 250 and ids == sorted(ids) and res.pages == 3 and res.total_count > 250 and res.truncated


def test_cache_makes_reruns_free(cfg, world, tmp_path):
    c, tr = _client(cfg, world, cache=tmp_path)
    api = PeopleAPI(c, cfg)
    flt = title_filter(["backend engineer"])
    api.search(flt, fields=["crustdata_person_id", "experience"], max_results=30)
    spent, n_req = c.meter.estimated, len(tr.requests)
    r2 = api.search(flt, fields=["crustdata_person_id", "experience"], max_results=30)
    assert len(r2.profiles) == 30 and c.meter.estimated == spent and len(tr.requests) == n_req and c.cache_hits >= 1


def test_credit_cap_returns_best_so_far(cfg, world):
    c, _ = _client(cfg, world)
    c.meter.cap_case = 5.0                      # ~21 heavy profiles
    api = PeopleAPI(c, cfg)
    res = api.search(title_filter(["software engineer"]), fields=["crustdata_person_id", "experience"], max_results=400, page_size=100)
    assert 0 < len(res.profiles) <= 22 and res.truncated and any("credit cap" in w for w in res.warnings)
    assert c.meter.case_spent <= 5.0 + 1e-9


def test_credit_cap_zero_budget_stops_cleanly(cfg, world):
    c, _ = _client(cfg, world)
    c.meter.cap_case = 0.0
    res = PeopleAPI(c, cfg).search(title_filter(["software engineer"]), fields=["crustdata_person_id", "experience"], max_results=50)
    assert res.profiles == [] and res.truncated


def test_429_is_retried_until_success(cfg, world):
    c, _ = _client(cfg, world, faults=FaultInjector({"429": 0.5}, seed=3))
    api = PeopleAPI(c, cfg)
    for _ in range(6):
        assert api.search(title_filter(["backend engineer"]), fields=["crustdata_person_id"], max_results=5).profiles
    assert c.faults.counts.get("429", 0) > 0


def test_persistent_429_raises_rate_limited(cfg, world):
    c, _ = _client(cfg, world, faults=FaultInjector({"429": 1.0}))
    with pytest.raises(CrustRateLimited):
        PeopleAPI(c, cfg).search(title_filter(["backend engineer"]), fields=["crustdata_person_id"], max_results=5)


def test_5xx_retries_twice_then_unavailable(cfg, world):
    c, tr = _client(cfg, world, faults=FaultInjector({"5xx": 1.0}))
    with pytest.raises(CrustUnavailable):
        PeopleAPI(c, cfg).search(title_filter(["backend engineer"]), fields=["crustdata_person_id"], max_results=5)
    assert len(tr.requests) == 0 and c.faults.counts["5xx"] == 3          # 1 try + 2 retries, then degrade


def test_intermittent_5xx_recovers(cfg, world):
    c, _ = _client(cfg, world, faults=FaultInjector({"5xx": 0.3}, seed=5))
    ok = 0
    for i in range(10):
        try:
            PeopleAPI(c, cfg).search(title_filter(["backend engineer", f"engineer {i}"]), fields=["crustdata_person_id"], max_results=3)
            ok += 1
        except CrustUnavailable:
            pass
    assert ok >= 7


def test_empty_results_never_crash(cfg, world):
    c, _ = _client(cfg, world, faults=FaultInjector({"empty": 1.0}))
    res = PeopleAPI(c, cfg).search(title_filter(["backend engineer"]), fields=["crustdata_person_id"], max_results=10)
    assert res.profiles == [] and res.total_count in (0, None)


def test_403_denied_fields_are_narrowed_automatically(cfg):
    w = SimWorld(seed=3, n_people=300, n_target=6, n_other=10, deny_skills=True)
    c, _ = _client(cfg, w)
    res = PeopleAPI(c, cfg).search(title_filter(["engineer"]), fields=["crustdata_person_id", "experience", "skills"], max_results=20)
    assert res.profiles and "skills" not in res.fields_used and any("403" in x for x in res.warnings)
    assert all(p.skills == [] for p in res.profiles)


def test_400_bad_field_dropped_then_run_continues(cfg, world):
    c, _ = _client(cfg, world)
    res = PeopleAPI(c, cfg).search(title_filter(["engineer"]), fields=["crustdata_person_id", "experience", "not_a_field"], max_results=5)
    assert res.profiles and "not_a_field" not in res.fields_used


def test_402_is_terminal(cfg, world):
    c, _ = _client(cfg, world, credits=0.1)
    with pytest.raises(CrustCreditsExhausted):
        PeopleAPI(c, cfg).search(title_filter(["engineer"]), fields=["crustdata_person_id", "experience"], max_results=50)


def test_auth_and_version_errors(cfg, world):
    c = CrustClient(cfg, SimTransport(world), api_key="", use_cache=False, faults=FaultInjector({}), sleep=lambda s: None)
    c.api_version = "1999-01-01"
    with pytest.raises(CrustBadRequest):
        c.post("/person/search", {"filters": cond("crustdata_person_id", "=", 1)}, limiter="person_search")


def test_rate_limiter_spreads_requests():
    now = [0.0]
    slept = []
    rl = RateLimiter({"k": 30}, sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)), clock=lambda: now[0])
    for _ in range(20):
        rl.acquire("k")
    assert sum(slept) >= (20 - 5) * 2.0 - 0.5     # burst of 5 then one request every 2 s


def test_fault_injector_is_deterministic():
    a, b = FaultInjector.from_string("429:0.3", seed=9), FaultInjector.from_string("429:0.3", seed=9)
    assert [a.hit("429") for _ in range(50)] == [b.hit("429") for _ in range(50)]


# ---------------------------------------------------------------- simulator fidelity
def test_sim_filter_semantics(world):
    p = next(iter(world._payloads.values()))
    title = resolve(p, "experience.employment_details.title")
    assert title and any(t for t in title)
    cur = resolve(p, "experience.employment_details.current.title")
    past = resolve(p, "experience.employment_details.past.title")
    assert sorted(cur + past) == sorted(title)
    assert eval_filter(p, cond("experience.employment_details.title", "(.)", title[0].lower())) is True
    assert eval_filter(p, cond("experience.employment_details.title", "(!)", title[0])) is False
    assert eval_filter(p, AND(cond("crustdata_person_id", "=", p["crustdata_person_id"]), cond("crustdata_person_id", "=>", 0)))
    assert not eval_filter(p, cond("crustdata_person_id", "in", [-1, -2]))


def test_sim_rejects_bad_requests(cfg, world):
    c, tr = _client(cfg, world)
    with pytest.raises(CrustBadRequest):
        c.post("/person/search", {"filters": title_filter(["x"]), "limit": 5000, "fields": ["crustdata_person_id"]}, limiter="person_search")
    with pytest.raises(CrustBadRequest):
        c.post("/person/search", {"limit": 5}, limiter="person_search")
    with pytest.raises(CrustBadRequest):
        c.post("/person/search", {"filters": title_filter(["x"]), "limit": 5, "sorts": [{"field": "basic_profile.nope", "order": "asc"}]}, limiter="person_search")


def test_jobs_and_companies_api(cfg, world):
    c, _ = _client(cfg, world)
    jobs = JobsAPI(c, cfg).search(date_from=dt.date(2025, 4, 1), date_to=dt.date(2025, 10, 1), max_results=500)
    assert len(jobs.jobs) == len(world.postings) and all(dt.date(2025, 4, 1) <= j["date_added"] < dt.date(2025, 10, 1) for j in jobs.jobs)
    comp = world.companies[world.postings[0].company_id]
    ident = CompaniesAPI(c, cfg).resolve_name(comp.name)
    assert ident and ident.company_id == comp.cid
    assert CompaniesAPI(c, cfg).resolve_name("No Such Company Xyz") is None
    g = CompaniesAPI(c, cfg).growth([comp.cid])
    assert comp.cid in g and "yoy" in g[comp.cid]


def test_sim_world_is_deterministic():
    a, b = SimWorld(seed=11, n_people=200, n_target=5, n_other=8), SimWorld(seed=11, n_people=200, n_target=5, n_other=8)
    assert a.truth() == b.truth() and a.person_payload(1_000_001) == b.person_payload(1_000_001)


def test_hires_start_after_T_and_show_target_in_present_profile(world):
    for post in world.postings:
        for pid in post.hires:
            p = world.people[pid]
            role = next(r for r in p.roles if r.company_id == post.company_id and r.start > post.T)
            assert role.start > post.T
            assert world.role_at(p, post.T) is None or world.role_at(p, post.T).company_id != post.company_id
