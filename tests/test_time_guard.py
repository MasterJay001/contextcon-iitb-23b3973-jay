"""The Time Guard decides whether our number is real, so it gets the harshest tests: explicit cases for
every leak class, property-based fuzzing of the invariants, and a sweep over every planted hire."""
from __future__ import annotations

import datetime as dt

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agents.time_guard import (LeakError, LocalFilter, MaskConfig, TargetRef, TimeGuard, assert_clean, mask_profile)
from core.dates import FuzzyDate
from core.models import JobSpec, MaskedProfile, RawProfile, Role
from tests.conftest import T0, make_profile, make_role

SNAP = dt.date(2026, 10, 31)
TARGET = TargetRef(company_id=999, company_name="Acme Payments Pvt Ltd", company_url="https://www.linkedin.com/company/acme-payments")


def _mask(profile, T=T0, **kw):
    return mask_profile(profile, T, target=kw.pop("target", TARGET), snapshot=SNAP, **kw)


# ---------------------------------------------------------------- leak #1 post-T jobs
def test_post_T_role_dropped_and_old_job_ongoing():
    p = make_profile(1, [
        make_role(0, "Senior Backend Engineer", "Acme Payments", "2025-08-20", cid=999),       # the hire's new job
        make_role(1, "Software Engineer", "OldCo", "2019-03-15", "2025-08-19", cid=5),
    ])
    mp = _mask(p)
    assert [r.company_name for r in mp.roles] == ["OldCo"]
    assert mp.roles[0].end is None                      # ended after T -> ongoing at T
    assert mp.employer_at_T == "OldCo" and mp.mask.dropped_post_T == 1 and mp.mask.ended_after_T_to_ongoing == 1
    assert not any(r.company_id == 999 for r in mp.roles)


def test_title_recomputed_from_masked_roles_not_current_title():
    p = make_profile(2, [
        make_role(0, "VP Engineering", "Acme Payments", "2025-09-01", cid=999),
        make_role(1, "Software Engineer", "OldCo", "2019-03-15", "2025-08-30", cid=5),
    ], headline="VP Engineering at Acme Payments", current_title="VP Engineering")
    mp = _mask(p)
    assert mp.title_at_T == "Software Engineer" and mp.norm.seniority == 2
    assert "VP" not in mp.model_dump_json()


# ---------------------------------------------------------------- leak #2 free text
def test_free_text_fields_never_copied():
    p = make_profile(3, [make_role(0, "Engineer", "OldCo", "2019-03-15", cid=5)],
                     name="Jane Doe", headline="Engineer at Acme Payments", summary="Joined Acme Payments in 2026",
                     current_title="Engineer at Acme")
    mp = _mask(p)
    blob = mp.model_dump_json()
    for forbidden in ("Jane", "Doe", "Joined Acme", "Engineer at Acme", "Acme Payments in 2026"):
        assert forbidden not in blob
    assert "headline" in mp.mask.free_text_dropped          # recorded as dropped, never copied


def test_target_mentions_scrubbed_from_descriptions_and_future_sentences_dropped():
    p = make_profile(4, [make_role(0, "Engineer", "OldCo", "2019-03-15", cid=5,
                                   desc="Built APIs. Integrated with Acme Payments gateway. Scaled the team in 2026.")])
    mp = _mask(p)
    d = mp.roles[0].description
    assert "Acme" not in d and "2026" not in d and "Built APIs" in d
    assert mp.mask.scrubbed_company_mentions >= 1 and mp.mask.scrubbed_future_sentences == 1


# ---------------------------------------------------------------- leak #5 fuzzy dates
def test_year_precision_in_T_year_is_dropped():
    p = make_profile(5, [make_role(0, "Engineer", "NewCo", "2025-01-01", cid=7), make_role(1, "Engineer", "OldCo", "2020-06-17", "2025-01-01", cid=5)])
    mp = _mask(p)
    assert mp.mask.dropped_ambiguous == 1
    assert all(r.company_name != "NewCo" for r in mp.roles)


def test_month_straddling_T_is_dropped_but_earlier_month_kept():
    straddle = make_profile(6, [make_role(0, "Engineer", "NewCo", "2025-05-01", cid=7)])
    earlier = make_profile(7, [make_role(0, "Engineer", "NewCo", "2025-04-01", cid=7)])
    assert _mask(straddle).roles == [] and _mask(straddle).mask.dropped_ambiguous == 1
    assert len(_mask(earlier).roles) == 1


def test_ambiguous_end_hides_possibly_post_T_departure():
    p = make_profile(8, [make_role(0, "Engineer", "OldCo", "2019-03-15", "2025-05-01", cid=5)])
    mp = _mask(p)
    assert mp.roles[0].end is None and mp.mask.ambiguous_end_to_ongoing == 1
    q = make_profile(9, [make_role(0, "Engineer", "OldCo", "2019-03-15", "2025-04-15", cid=5)])
    assert _mask(q).roles[0].end is not None


def test_missing_end_inferred_from_next_start():
    p = make_profile(10, [
        make_role(0, "Engineer", "B", "2023-02-14", cid=6),
        make_role(1, "Engineer", "A", "2020-03-12", None, cid=5, from_current=False),   # past role, no end
    ])
    mp = _mask(p, T=dt.date(2024, 1, 10))
    a = next(r for r in mp.roles if r.company_name == "A")
    assert a.end is not None and a.end_inferred
    assert mp.mask.missing_end_inferred == 1


def test_invalid_dates_end_before_start_dropped():
    p = make_profile(11, [make_role(0, "Engineer", "OldCo", "2020-03-12", "2019-01-17", cid=5)])
    assert _mask(p).mask.dropped_invalid == 1


def test_undated_role_dropped():
    p = make_profile(12, [make_role(0, "Engineer", "OldCo", None, cid=5)])
    mp = _mask(p)
    assert mp.roles == [] and mp.mask.dropped_undated == 1


# ---------------------------------------------------------------- live mode (T = today)
def test_live_mode_trusts_dates_and_keeps_year_precision_roles():
    p = make_profile(13, [make_role(0, "Engineer", "NewCo", "2026-01-01", cid=7)])
    mp = mask_profile(p, SNAP, snapshot=SNAP)             # T == snapshot -> trust reported dates
    assert len(mp.roles) == 1 and mp.mask.trust_reported_dates
    assert len(_mask(p, T=dt.date(2026, 3, 1)).roles) == 0   # same date is ambiguous in a backtest


# ---------------------------------------------------------------- education / skills / location policy
def test_education_in_progress_trimmed_and_future_dropped():
    from core.models import School
    p = make_profile(14, [make_role(0, "Engineer", "OldCo", "2022-07-01", cid=5)],
                     education=[School(school="A", start_year=2021, end_year=2026), School(school="B", start_year=2027, end_year=2029)])
    mp = _mask(p)
    assert [s.school for s in mp.education] == ["A"] and mp.education[0].end_year is None


def test_strict_mode_removes_undated_fields():
    p = make_profile(15, [make_role(0, "Engineer", "OldCo", "2019-03-15", cid=5, desc="Wrote Go services", loc="Pune, India")],
                     skills=["Go"], location_raw="Mumbai, India")
    loose, strict = _mask(p), _mask(p, mcfg=MaskConfig().strict())
    assert loose.skills == ["Go"] and "skills" in loose.mask.leaky_fields
    assert strict.skills == [] and strict.location == "" and strict.roles[0].description is None


def test_location_at_T_prefers_dated_role_location():
    p = make_profile(16, [make_role(0, "Engineer", "OldCo", "2019-03-15", cid=5, loc="Pune, India")], location_raw="Bengaluru, India")
    assert _mask(p).location_at_T == "Pune, India"


# ---------------------------------------------------------------- fail-closed post-condition
def test_assert_clean_raises_on_leaky_profile():
    leaky = MaskedProfile(person_id=1, T=T0, roles=[Role(idx=0, title="x", company_name="y", start=FuzzyDate(date=dt.date(2025, 9, 1), precision="day"))])
    with pytest.raises(LeakError):
        assert_clean(leaky)


def test_guard_drops_instead_of_passing_a_violation(monkeypatch):
    import agents.time_guard as tg

    def bad_mask(profile, T, **kw):
        return MaskedProfile(person_id=profile.person_id, T=T, roles=[Role(idx=0, title="x", start=FuzzyDate(date=dt.date(2030, 1, 1)))])

    monkeypatch.setattr(tg, "mask_profile", bad_mask)
    guard = TimeGuard(T0, snapshot=SNAP)
    assert guard.admit([make_profile(1, [])]) == [] and guard.violations == 1


# ---------------------------------------------------------------- local re-filter (leak #3)
def _spec(title="Backend Engineer", seniority="Senior", **kw):
    return JobSpec(norm_title=title, seniority=seniority, location="Bengaluru", **kw)


def test_local_filter_excludes_current_employees_and_wrong_families(cfg):
    flt = LocalFilter.from_spec(_spec(), cfg, TARGET)
    emp = _mask(make_profile(20, [make_role(0, "Senior Backend Engineer", "Acme Payments Pvt Ltd", "2022-01-14", cid=999)]))
    ok = _mask(make_profile(21, [make_role(0, "Senior Backend Engineer", "OtherCo", "2022-01-14", cid=5)]))
    pm = _mask(make_profile(22, [make_role(0, "Senior Product Manager", "OtherCo", "2022-01-14", cid=5)]))
    jr = _mask(make_profile(23, [make_role(0, "Software Engineer Intern", "OtherCo", "2022-01-14", cid=5)]))
    assert flt.passes(emp) == (False, "current_employee")
    assert flt.passes(ok) == (True, "ok")
    assert flt.passes(pm) == (False, "title_family")
    assert flt.passes(jr)[1] == "seniority"


def test_local_filter_drops_hire_whose_title_only_matched_after_T(cfg):
    """The classic leak: the API title filter matches the hire through the post-T title."""
    hire = make_profile(24, [
        make_role(0, "Senior Backend Engineer", "Acme Payments Pvt Ltd", "2025-08-20", cid=999),
        make_role(1, "Marketing Manager", "OldCo", "2019-03-15", "2025-08-19", cid=5),
    ])
    mp = _mask(hire)
    assert LocalFilter.from_spec(_spec(), cfg, TARGET).passes(mp)[1] in {"title_family", "seniority"}


# ---------------------------------------------------------------- property-based fuzzing
_dates = st.dates(min_value=dt.date(2005, 1, 1), max_value=dt.date(2026, 10, 1))


@st.composite
def _roles(draw):
    n = draw(st.integers(0, 6))
    out = []
    for i in range(n):
        s = draw(_dates)
        mode = draw(st.sampled_from(["day", "month", "year"]))
        s = {"day": s, "month": s.replace(day=1), "year": s.replace(month=1, day=1)}[mode]
        e = draw(st.one_of(st.none(), _dates))
        if e is not None and draw(st.booleans()):
            e = e.replace(day=1)
        out.append(make_role(i, draw(st.sampled_from(["Software Engineer", "Backend Developer", "Product Manager", "Acme Payments Engineer"])),
                             draw(st.sampled_from(["A", "B", "Acme Payments", "C"])), s.isoformat(), e.isoformat() if e else None,
                             cid=draw(st.sampled_from([1, 2, 999, 3])), from_current=(e is None)))
    return out


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(roles=_roles(), T=st.dates(min_value=dt.date(2012, 1, 1), max_value=dt.date(2026, 6, 1)))
def test_property_masked_output_never_contains_post_T_information(roles, T):
    mp = mask_profile(make_profile(1, roles), T, target=TARGET, snapshot=SNAP)
    assert_clean(mp)                                              # fail-closed invariants
    for r in mp.roles:
        assert r.start.hi <= T
        assert r.end is None or r.end.date <= T
    assert mp.mask.kept + mp.mask.dropped_post_T + mp.mask.dropped_ambiguous + mp.mask.dropped_undated + mp.mask.dropped_invalid <= len(roles)
    assert [r.start.date for r in mp.roles] == sorted((r.start.date for r in mp.roles), reverse=True)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(roles=_roles(), T1=st.dates(min_value=dt.date(2012, 1, 1), max_value=dt.date(2025, 1, 1)), delta=st.integers(0, 700))
def test_property_monotone_in_T(roles, T1, delta):
    """Anything knowable at T1 is still knowable at a later T2 (roles kept at T1 are kept at T2)."""
    T2 = T1 + dt.timedelta(days=delta)
    key = lambda r: (r.company_id, r.title, r.start.date)  # noqa: E731
    k1 = {key(r) for r in mask_profile(make_profile(1, roles), T1, snapshot=SNAP).roles}
    k2 = {key(r) for r in mask_profile(make_profile(1, roles), T2, snapshot=SNAP).roles}
    assert k1 <= k2


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(roles=_roles(), T=st.dates(min_value=dt.date(2012, 1, 1), max_value=dt.date(2026, 6, 1)))
def test_property_masking_is_idempotent(roles, T):
    mp = mask_profile(make_profile(1, roles), T, snapshot=SNAP)
    again = mask_profile(RawProfile(person_id=1, roles=[r.model_copy(update={"from_current": r.end is None}) for r in mp.roles]), T, snapshot=SNAP)
    sig = lambda m: [(r.company_id, r.title, r.start.date, r.end.date if r.end else None) for r in m.roles]  # noqa: E731
    assert sig(mp) == sig(again)


# ---------------------------------------------------------------- the sim world: every planted hire is closed
def test_every_planted_hire_is_masked_clean(world, cfg):
    """Spot audit at scale: the hire's present-day profile shows the target company; masked, it must not."""
    from crust.sim import SimTransport
    from crust.client import CrustClient, FaultInjector
    from crust.people import PeopleAPI

    client = CrustClient(cfg, SimTransport(world), api_key="k", use_cache=False, faults=FaultInjector({}), sleep=lambda s: None)
    api = PeopleAPI(client, cfg)
    n_checked = 0
    for post in world.postings:
        comp = world.companies[post.company_id]
        tgt = TargetRef(comp.cid, comp.name, comp.url)
        hires = api.fetch_by_ids(post.hires, ["crustdata_person_id", "basic_profile", "experience", "education", "skills"]).profiles
        for raw in hires:
            assert any(r.company_id == comp.cid for r in raw.roles), "world must show the leak before masking"
            mp = mask_profile(raw, post.T, target=tgt, snapshot=SNAP)
            assert_clean(mp)
            ongoing_at_target = [r for r in mp.roles if r.end is None and tgt.matches_role(r)]
            after_T = [r for r in mp.roles if tgt.matches_role(r) and r.start.date >= post.T]
            assert not ongoing_at_target and not after_T, f"leak for hire {raw.person_id} of posting {post.job_id}"
            assert comp.name not in mp.model_dump_json().replace(comp.name + " ", "", 0) or not any(
                comp.name in (r.description or "") for r in mp.roles)
            n_checked += 1
    assert n_checked >= len(world.postings) // 2
