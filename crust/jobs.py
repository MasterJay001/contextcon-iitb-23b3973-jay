"""Job Postings Search wrapper: past openings become test cases.

`metadata.date_added` is documented as the posting date (an ISO datetime, no timezone). Company
fields on a job record (headcount etc.) are captured at INDEX time, not posting time, so they are
never used as point-in-time values (data trap, doc sec. 4).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from core.events import log_event
from crust.client import CrustClient
from crust.people import AND, cond
from crust.schema import parse_jobs

JOB_FIELDS = [
    "crustdata_job_id", "job_details.title", "job_details.category", "job_details.workplace_type",
    "job_details.employment_type", "job_details.url", "company.basic_info", "company.headcount",
    "location", "content.description", "metadata.date_added",
]


@dataclass
class JobSearchResult:
    jobs: list[dict] = field(default_factory=list)
    total_count: int | None = None
    pages: int = 0
    credits: float = 0.0


class JobsAPI:
    def __init__(self, client: CrustClient, cfg: dict):
        self.client = client
        self.cfg = cfg

    def search(self, *, date_from: dt.date, date_to: dt.date, title_terms: list[str] | None = None,
               company_ids: list[int] | None = None, category: str | None = None,
               country: str | None = None, max_results: int = 200) -> JobSearchResult:
        conds = [
            cond("metadata.date_added", "=>", date_from.isoformat()),
            cond("metadata.date_added", "<", date_to.isoformat()),
        ]
        if title_terms:
            conds.append({"op": "or", "conditions": [cond("job_details.title", "(.)", t) for t in title_terms]})
        if company_ids:
            conds.append(cond("company.basic_info.company_id", "in", company_ids))
        if category:
            conds.append(cond("job_details.category", "=", category))
        if country:
            conds.append(cond("location.country", "=", country))
        res = JobSearchResult()
        cursor = None
        page = min(200, max_results)
        while len(res.jobs) < max_results:
            body = {
                "filters": AND(*conds), "fields": JOB_FIELDS, "limit": min(page, max_results - len(res.jobs)),
                "sorts": [{"field": "metadata.date_added", "order": "asc"}],   # documented deterministic sort
            }
            if cursor:
                body["cursor"] = cursor
            r = self.client.post("/job/search", body, limiter="job_search")
            res.pages += 1
            res.credits += r.credits
            payload = r.json if isinstance(r.json, dict) else {}
            if payload.get("total_count") is not None:
                res.total_count = int(payload["total_count"])
            res.jobs += parse_jobs(payload)
            cursor = payload.get("next_cursor")
            if not cursor:
                break
        log_event("job_search", n=len(res.jobs), total=res.total_count, pages=res.pages)
        return res
