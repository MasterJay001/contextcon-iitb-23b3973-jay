"""Defensive parsing of Crustdata payloads into our typed models.

The documented People Search profile looks like:

    {"crustdata_person_id": 14540,
     "basic_profile": {"name", "headline", "current_title", "location": {city,state,country,raw}, "normalized_title": {...}},
     "experience": {"employment_details": {"current": [role...], "past": [role...]}, "recently_changed_jobs": bool},
     "education": {"schools": [{school, degree, field_of_study, start_year, end_year}]},
     "skills": {"professional_network_skills": ["Python", ...]},
     "social_handles": {"professional_network_identifier": {"profile_url": ...}}}

role = {name, title, description, start_date "YYYY-MM-DD", end_date, crustdata_company_id,
        company_professional_network_profile_url, seniority_level, function_category, location:{raw},
        company_headcount_range, company_industries[], company_type, company_headquarters_country, ...}

We accept reasonable variations (flat role lists, string locations, `company_name` for `name`) and
never raise on a malformed row: a bad role is skipped and counted, a bad profile returns None.
"""
from __future__ import annotations

from typing import Any

from core.dates import parse_date
from core.events import log_event
from core.models import RawProfile, Role, School


def _s(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _i(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _location_str(loc: Any) -> str | None:
    if isinstance(loc, str):
        return _s(loc)
    if isinstance(loc, dict):
        return _s(loc.get("raw")) or _s(loc.get("full_location")) or ", ".join(
            x for x in (_s(loc.get("city")), _s(loc.get("state")), _s(loc.get("country"))) if x
        ) or None
    return None


def parse_role(r: dict, idx: int, src: str, from_current: bool) -> Role | None:
    if not isinstance(r, dict):
        return None
    try:
        inds = r.get("company_industries")
        return Role(
            idx=idx,
            src=src,
            company_name=_s(r.get("name")) or _s(r.get("company_name")),
            company_id=_i(r.get("crustdata_company_id") if r.get("crustdata_company_id") is not None else r.get("company_id")),
            company_url=_s(r.get("company_professional_network_profile_url")),
            title=_s(r.get("title")),
            description=_s(r.get("description")),
            start=parse_date(r.get("start_date")),
            end=parse_date(r.get("end_date")),
            seniority_level=_s(r.get("seniority_level")),
            function_category=_s(r.get("function_category")),
            location=_location_str(r.get("location")),
            employment_type=_s(r.get("employment_type")),
            company_headcount_range=_s(r.get("company_headcount_range")),
            company_industries=[str(x) for x in inds] if isinstance(inds, list) else [],
            company_type=_s(r.get("company_type")),
            company_hq_country=_s(r.get("company_headquarters_country")),
            from_current=from_current,
        )
    except Exception as exc:  # pydantic validation or odd types
        log_event("role_parse_error", src=src, error=str(exc)[:120])
        return None


def parse_profile(p: dict) -> RawProfile | None:
    if not isinstance(p, dict):
        return None
    pid = _i(p.get("crustdata_person_id"))
    if pid is None:
        log_event("profile_missing_id")
        return None
    bp = p.get("basic_profile") if isinstance(p.get("basic_profile"), dict) else {}
    loc = bp.get("location") if isinstance(bp.get("location"), (dict, str)) else None
    loc_d = loc if isinstance(loc, dict) else {}

    roles: list[Role] = []
    exp = p.get("experience") if isinstance(p.get("experience"), dict) else {}
    ed = exp.get("employment_details")
    idx = 0
    if isinstance(ed, dict):
        for key, cur in (("current", True), ("past", False)):
            for i, r in enumerate(ed.get(key) or []):
                role = parse_role(r, idx, f"experience.employment_details.{key}[{i}]", cur)
                idx += 1
                if role is not None:
                    roles.append(role)
    elif isinstance(ed, list):  # tolerate a flat list
        for i, r in enumerate(ed):
            role = parse_role(r, idx, f"experience.employment_details[{i}]", r.get("end_date") in (None, "") if isinstance(r, dict) else False)
            idx += 1
            if role is not None:
                roles.append(role)

    schools: list[School] = []
    edu = p.get("education") if isinstance(p.get("education"), dict) else {}
    for s in edu.get("schools") or []:
        if isinstance(s, dict):
            schools.append(School(school=_s(s.get("school")), degree=_s(s.get("degree")), field_of_study=_s(s.get("field_of_study")),
                                  start_year=_i(s.get("start_year")), end_year=_i(s.get("end_year"))))

    sk = p.get("skills")
    skills: list[str] = []
    if isinstance(sk, dict):
        skills = [str(x) for x in (sk.get("professional_network_skills") or []) if x]
    elif isinstance(sk, list):
        skills = [str(x) for x in sk if x]

    sh = p.get("social_handles") if isinstance(p.get("social_handles"), dict) else {}
    pn = sh.get("professional_network_identifier") if isinstance(sh.get("professional_network_identifier"), dict) else {}

    yoe = p.get("years_of_experience_raw")
    if yoe is None and isinstance(exp, dict):
        yoe = exp.get("years_of_experience_raw")
    return RawProfile(
        person_id=pid,
        name=_s(bp.get("name")),
        headline=_s(bp.get("headline")),
        summary=_s(bp.get("summary")),
        current_title=_s(bp.get("current_title")),
        profile_url=_s(pn.get("profile_url")),
        location_raw=_location_str(loc),
        city=_s(loc_d.get("city")),
        state=_s(loc_d.get("state")),
        country=_s(loc_d.get("country")),
        roles=roles,
        education=schools,
        skills=skills,
        years_of_experience_raw=float(yoe) if isinstance(yoe, (int, float)) else None,
        recently_changed_jobs=exp.get("recently_changed_jobs") if isinstance(exp.get("recently_changed_jobs"), bool) else None,
    )


def parse_profiles(payload: Any) -> list[RawProfile]:
    items = payload.get("profiles", []) if isinstance(payload, dict) else []
    out: list[RawProfile] = []
    for it in items:
        rp = parse_profile(it)
        if rp is not None:
            out.append(rp)
    return out


# ------------------------------------------------------------------------------- jobs
def parse_job(j: dict) -> dict | None:
    if not isinstance(j, dict):
        return None
    jd = j.get("job_details") or {}
    comp = (j.get("company") or {}).get("basic_info") or {}
    loc = j.get("location") or {}
    meta = j.get("metadata") or {}
    added = parse_date(meta.get("date_added"))
    jid = _i(j.get("crustdata_job_id") if j.get("crustdata_job_id") is not None else jd.get("job_id"))
    if jid is None or added is None:
        return None
    hc = ((j.get("company") or {}).get("headcount") or {})
    return {
        "job_id": jid,
        "title": _s(jd.get("title")),
        "category": _s(jd.get("category")),
        "workplace_type": _s(jd.get("workplace_type")),
        "employment_type": _s(jd.get("employment_type")),
        "url": _s(jd.get("url")),
        "company_id": _i(comp.get("crustdata_company_id")),
        "company_name": _s(comp.get("name")),
        "company_domain": _s(comp.get("primary_domain")),
        "company_headcount_now": _i(hc.get("total")),   # index-time value; NOT point-in-time (data trap)
        "company_headcount_range_now": _s(hc.get("range")),
        "location_raw": _s(loc.get("raw")),
        "city": _s(loc.get("city")),
        "state": _s(loc.get("state")),
        "country": _s(loc.get("country")),
        "description": _s(((j.get("content") or {}).get("description"))),
        "date_added": added.date,
    }


def parse_jobs(payload: Any) -> list[dict]:
    items = payload.get("job_listings", []) if isinstance(payload, dict) else []
    return [j for j in (parse_job(x) for x in items) if j is not None]
