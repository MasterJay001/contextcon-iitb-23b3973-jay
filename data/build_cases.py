"""Build the backtest test set: past openings with known hires.

Route A (preferred, from real postings forward)
  1. pull postings with metadata.date_added in [from, to]   (default 2025-04-01 .. 2025-09-30: >1 year for hires to show)
  2. keep companies with 50-5,000 employees (giants hire hundreds into one title: "the hire" would be ambiguous)
  3. for each posting find people whose role at that company STARTED after T (0-180 days), then verify locally:
        start interval strictly after T and within the window; same normalised department and seniority within +-1;
        same city/metro as the posting (or a remote posting)
  4. keep postings with 1-3 matched hires (0 = no answer, >3 = ambiguous / mass hiring)
  5. keep postings whose retrieval UNIVERSE (people holding the title, coarse location) is small enough to enumerate
     completely, so ranking differences are not retrieval luck (sourcer.max_universe_for_full_enumeration)
Route B (fallback, from hires backward): T = start - 60 days; the brief uses ONLY company, normalised title, seniority and
job location (leak #11).

Outputs: data/cases.jsonl (NO hires), eval_private/hires.jsonl (answer key), results/case_funnel.json,
results/case_quality_gate.md (10 random cases to hand-check: if >2 of 10 look wrong, tighten the title rule).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from core.config import ROOT, data_path, load_config, results_dir, today
from core.events import log_event
from core.geo import geo
from core.models import Case, seniority_label
from core.skills import find_skills
from crust.client import CrustBadRequest, CrustError
from crust.people import AND, OR, cond, location_filter, title_filter
from crust.titles import family_distance, normalize_title, taxonomy

_YEARS = re.compile(r"(\d+)\+?\s*(?:years|yrs)", re.IGNORECASE)


def _brief_from_job(job: dict, norm) -> str:
    city = job.get("city") or job.get("location_raw") or ""
    wt = (job.get("workplace_type") or "").lower()
    skills = sorted(find_skills(job.get("description")))
    yrs = _YEARS.search(job.get("description") or "")
    parts = [f"{job['title']}.", f"{city}{', ' + wt if wt else ''}."]
    if yrs:
        parts.append(f"{yrs.group(1)}+ yrs experience.")
    if skills:
        parts.append("Skills: " + ", ".join(skills[:6]) + ".")
    return " ".join(p for p in parts if p.strip(". "))


def split_by_company(cases: list[Case], seed: int, train_frac: float = 0.6) -> None:
    """60% train / 40% held-out BY COMPANY: no company in both (else the self-improving loop could memorise a
    company's feeder pattern and look smarter than it is)."""
    cos = sorted({c.company_id for c in cases}, key=lambda x: hashlib.sha256(f"{seed}-{x}".encode()).hexdigest())
    n_train = int(round(len(cos) * train_frac))
    train = set(cos[:n_train])
    for c in cases:
        c.split = "train" if c.company_id in train else "heldout"


class HireFinder:
    def __init__(self, ctx, window_days: int, tol: int):
        self.ctx, self.window, self.tol = ctx, window_days, tol
        self._url_cache: dict[int, str | None] = {}

    def candidates(self, company_id: int, T: dt.date, titles: list[str]):
        people = self.ctx.people
        hi = (T + dt.timedelta(days=self.window + 31)).isoformat()
        flt = AND(
            cond("experience.employment_details.current.company_id", "in", [company_id]),
            cond("experience.employment_details.current.start_date", "=>", T.isoformat()),
            cond("experience.employment_details.current.start_date", "=<", hi),
            OR(*[cond("experience.employment_details.current.title", "(.)", t) for t in titles[:10]]),
        )
        try:
            return people.search(flt, fields=["crustdata_person_id", "basic_profile.location", "experience"], max_results=12)
        except CrustBadRequest as exc:
            log_event("hire_filter_rejected", message=str(exc)[:200])
            raise

    def verify(self, rp, company_id: int, T: dt.date, job_norm, job: dict):
        """Return a hire record if this profile is a genuine hire for the posting, else None."""
        for r in rp.roles:
            if r.company_id != company_id or r.start is None:
                continue
            if not r.from_current:
                continue
            if not (r.start.lo > T and r.start.date <= T + dt.timedelta(days=self.window)):
                continue                                  # not definitely after T / outside the window
            hn = normalize_title(r.title, r.seniority_level)
            if family_distance(hn.family, job_norm.family) > 3 and hn.department != job_norm.department:
                continue
            if abs(hn.seniority - job_norm.seniority) > self.tol:
                continue
            loc_ok = (job.get("workplace_type") or "").lower() == "remote" or geo().same_metro_or_state(r.location or rp.location_raw, job.get("location_raw") or job.get("city"))
            if not loc_ok:
                continue
            return {"person_id": rp.person_id, "start_date": r.start.date.isoformat(), "start_precision": r.start.precision, "title": r.title,
                    "family": hn.family, "seniority": hn.seniority, "location": r.location or rp.location_raw}
        return None


def build_route_a(ctx, args) -> tuple[list[Case], dict[str, list[dict]], dict]:
    cfg = ctx.cfg
    rng = random.Random(int(cfg["run"]["seed"]))
    funnel: dict[str, int] = {}
    jobs_api = __import__("crust.jobs", fromlist=["JobsAPI"]).JobsAPI(ctx.crust, cfg)
    posted = jobs_api.search(date_from=args.date_from, date_to=args.date_to + dt.timedelta(days=1), category=args.category, max_results=args.limit_postings)
    funnel["postings_pulled"] = len(posted.jobs)
    jobs = [j for j in posted.jobs if j.get("company_id") is not None]
    jobs = [j for j in jobs if (j.get("company_headcount_now") is None or args.min_size <= j["company_headcount_now"] <= args.max_size)]
    funnel["size_50_5000"] = len(jobs)
    rng.shuffle(jobs)
    finder = HireFinder(ctx, int(cfg["eval"]["hire_window_days"]), int(cfg["eval"]["seniority_tolerance"]))
    cases: list[Case] = []
    hires: dict[str, list[dict]] = {}
    per_company: dict[int, int] = {}
    spent0 = ctx.crust.meter.estimated
    funnel.update(family_known=0, has_candidates=0, verified_1_to_3=0, universe_ok=0)
    for job in jobs:
        if len(cases) >= args.n:
            break
        if ctx.crust.meter.estimated - spent0 > args.credit_cap:
            funnel["stopped_credit_cap"] = 1
            break
        norm = normalize_title(job["title"])
        if norm.family == "unknown":
            continue
        funnel["family_known"] += 1
        if per_company.get(job["company_id"], 0) >= 2:
            continue
        titles = [job["title"], *taxonomy().titles_for(norm.family, 5)]
        try:
            res = finder.candidates(job["company_id"], job["date_added"], titles)
        except CrustError:
            continue
        if not res.profiles:
            continue
        funnel["has_candidates"] += 1
        matched = [h for h in (finder.verify(rp, job["company_id"], job["date_added"], norm, job) for rp in res.profiles) if h]
        if not 1 <= len(matched) <= 3:
            continue
        funnel["verified_1_to_3"] += 1
        strict_titles = [norm.matched_title, *taxonomy().titles_for(norm.family, 6)]
        place = geo().parse(job.get("location_raw") or job.get("city"))
        universe = ctx.people.count(AND(title_filter(strict_titles), location_filter([place.country.title()] if place.country else [])))
        if universe is not None and universe > args.max_universe:
            funnel["universe_too_large"] = funnel.get("universe_too_large", 0) + 1
            continue
        funnel["universe_ok"] += 1
        ident = None
        try:
            m = ctx.companies.identify(ids=[job["company_id"]])
            ident = m[0] if m and m[0].resolved else None
        except (CrustError, ValueError):
            pass
        cid = f"c{len(cases) + 1:03d}"
        case = Case(case_id=cid, company_id=job["company_id"], company_url=ident.profile_url if ident else None,
                    company_name=(ident.name if ident else None) or job.get("company_name"), T=job["date_added"],
                    brief=_brief_from_job(job, norm), target_norm_title=norm.matched_title, target_seniority=seniority_label(norm.seniority),
                    location=job.get("city") or job.get("location_raw"), route="A", universe_size=universe,
                    meta={"posting_title": job["title"], "job_id": job["job_id"], "posting_url": job.get("url")})
        cases.append(case)
        hires[cid] = matched
        per_company[job["company_id"]] = per_company.get(job["company_id"], 0) + 1
    funnel["final_cases"] = len(cases)
    funnel["credits_spent_building"] = round(ctx.crust.meter.estimated - spent0, 2)
    return cases, hires, funnel


def build_route_b(ctx, args) -> tuple[list[Case], dict[str, list[dict]], dict]:
    """From hires backward: people who started in 2025 with a target title; T = start - 60 days (stated assumption)."""
    cfg = ctx.cfg
    rng = random.Random(int(cfg["run"]["seed"]))
    titles = [t.strip() for t in args.titles.split(",") if t.strip()]
    funnel = {"mode": "B", "titles": titles}
    flt = AND(cond("experience.employment_details.current.start_date", "=>", args.date_from.isoformat()),
              cond("experience.employment_details.current.start_date", "=<", args.date_to.isoformat()),
              OR(*[cond("experience.employment_details.current.title", "(.)", t) for t in titles]),
              location_filter(["India"]))
    res = ctx.people.search(flt, fields=["crustdata_person_id", "basic_profile.location", "experience"], max_results=args.limit_postings)
    cands = []
    for rp in res.profiles:
        for r in rp.roles:
            if r.from_current and r.company_id is not None and r.start is not None and r.start.precision != "year" and args.date_from <= r.start.date <= args.date_to:
                nt = normalize_title(r.title, r.seniority_level)
                if nt.family != "unknown":
                    cands.append((rp, r, nt))
                    break
    funnel["hire_candidates"] = len(cands)
    rng.shuffle(cands)
    cases, hires, by_co = [], {}, {}
    for rp, r, nt in cands:
        if len(cases) >= args.n:
            break
        by_co.setdefault(r.company_id, []).append((rp, r, nt))
    for cid_, items in by_co.items():
        if len(cases) >= args.n:
            break
        if not 1 <= len(items) <= 3:
            continue
        rp, r, nt = items[0]
        T = r.start.date - dt.timedelta(days=60)
        place = r.location or rp.location_raw or ""
        brief = f"{nt.matched_title}, {seniority_label(nt.seniority)}. {place}."          # company/title/seniority/location ONLY (leak #11)
        cid = f"c{len(cases) + 1:03d}"
        cases.append(Case(case_id=cid, company_id=cid_, company_name=r.company_name, company_url=r.company_url, T=T, brief=brief,
                          target_norm_title=nt.matched_title, target_seniority=seniority_label(nt.seniority), location=place or None, route="B",
                          meta={"assumed_time_to_hire_days": 60}))
        hires[cid] = [{"person_id": p.person_id, "start_date": rr.start.date.isoformat(), "start_precision": rr.start.precision, "title": rr.title,
                       "family": n.family, "seniority": n.seniority, "location": rr.location} for p, rr, n in items]
    funnel["final_cases"] = len(cases)
    return cases, hires, funnel


def write_outputs(cfg: dict, cases: list[Case], hires: dict[str, list[dict]], funnel: dict, backends: dict) -> None:
    cases_p, hires_p = data_path(cfg, "cases", backends["crust"]), data_path(cfg, "hires", backends["crust"])
    cases_p.parent.mkdir(parents=True, exist_ok=True)
    hires_p.parent.mkdir(parents=True, exist_ok=True)
    with open(cases_p, "w", encoding="utf-8") as fh:                       # NO hires in here
        for c in cases:
            fh.write(c.model_dump_json() + "\n")
    with open(hires_p, "w", encoding="utf-8") as fh:                       # the answer key
        for c in cases:
            fh.write(json.dumps({"case_id": c.case_id, "hires": hires[c.case_id]}) + "\n")
    res_dir = results_dir(cfg, backends["crust"], backends["llm"])
    funnel = {**funnel, "backends": backends, "n_train": sum(c.split == "train" for c in cases), "n_heldout": sum(c.split == "heldout" for c in cases),
              "multi_hire_cases": sum(len(hires[c.case_id]) > 1 for c in cases)}
    (res_dir / "case_funnel.json").write_text(json.dumps(funnel, indent=2, default=str), encoding="utf-8")
    rng = random.Random(1)
    sample = rng.sample(cases, min(10, len(cases)))
    lines = ["# Quality gate: hand-check these 10 random cases", "",
             "Open each hire's profile and confirm the role really matches the posting. If more than 2 of 10 look wrong, tighten the title rule "
             "(`eval.seniority_tolerance`, department check in `data/build_cases.py`) before building anything else.", "",
             f"Backends that produced this file: `{backends}`", "",
             "| case | company | T | posting title | hire title (start) | location | verdict (fill in) |", "|---|---|---|---|---|---|---|"]
    for c in sample:
        h = hires[c.case_id][0]
        lines.append(f"| {c.case_id} | {c.company_name} | {c.T} | {c.meta.get('posting_title', c.target_norm_title)} | {h['title']} ({h['start_date']}) | {c.location} | |")
    (res_dir / "case_quality_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from agents.orchestrator import build_context

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--route", choices=["A", "B"], default="A")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--from", dest="date_from", type=dt.date.fromisoformat, default=dt.date(2025, 4, 1))
    ap.add_argument("--to", dest="date_to", type=dt.date.fromisoformat, default=dt.date(2025, 9, 30))
    ap.add_argument("--min-size", type=int, default=50)
    ap.add_argument("--max-size", type=int, default=5000)
    ap.add_argument("--max-universe", type=int, default=None)
    ap.add_argument("--limit-postings", type=int, default=600)
    ap.add_argument("--category", default=None, help="job category filter, e.g. Engineering")
    ap.add_argument("--titles", default="software engineer,backend engineer", help="Route B target titles (comma separated)")
    ap.add_argument("--credit-cap", type=float, default=1500.0, help="credits allowed for case building")
    ap.add_argument("--crust-backend", choices=["live", "sim"], default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.max_universe is None:
        args.max_universe = int(cfg["sourcer"]["max_universe_for_full_enumeration"])
    ctx = build_context(cfg, mode="backtest", crust_backend=args.crust_backend, llm_backend="mock")
    try:
        cases, hires, funnel = (build_route_a if args.route == "A" else build_route_b)(ctx, args)
    except CrustError as exc:
        print(f"Could not build cases: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    split_by_company(cases, int(cfg["run"]["seed"]))
    write_outputs(cfg, cases, hires, funnel, ctx.backends())
    print(json.dumps(funnel, indent=2, default=str))
    print(f"\nwrote {len(cases)} cases ({sum(c.split == 'train' for c in cases)} train / {sum(c.split == 'heldout' for c in cases)} held-out) "
          f"-> {data_path(cfg, 'cases', ctx.crust_backend)} ; answer key -> {data_path(cfg, 'hires', ctx.crust_backend)}")
    print("Next: hand-check results/case_quality_gate.md (Gate 2). Backends:", ctx.backends())
    ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
