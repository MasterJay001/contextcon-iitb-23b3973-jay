"""People Search wrapper: filter builders, neutral-order pagination, credit-aware paging and
adaptive field narrowing (a 403 `denied_fields` or a 400 naming a bad field never kills a run).

Rule 4: retrieval is sorted by the neutral `crustdata_person_id`. We never sort or rank by anything
the hire event could influence (start dates, relevance, semantic scores).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.events import log_event
from core.models import RawProfile
from crust.client import (CreditCapReached, CrustBadRequest, CrustClient, CrustError, CrustPermissionError)
from crust.schema import parse_profiles

NEUTRAL_SORT = [{"field": "crustdata_person_id", "order": "asc"}]


# ------------------------------------------------------------------------------- filter DSL
def cond(field_: str, type_: str, value: Any) -> dict:
    return {"field": field_, "type": type_, "value": value}


def AND(*conds: dict) -> dict:
    conds = tuple(c for c in conds if c)
    return conds[0] if len(conds) == 1 else {"op": "and", "conditions": list(conds)}


def OR(*conds: dict) -> dict:
    conds = tuple(c for c in conds if c)
    return conds[0] if len(conds) == 1 else {"op": "or", "conditions": list(conds)}


def company_filter(company_id: int | None, company_url: str | None, scope: str = "any", mode: str = "id") -> dict | None:
    """Anyone with a role at the company. scope: any | current | past."""
    if mode == "id" and company_id is not None:
        base = "experience.employment_details" + ("" if scope == "any" else f".{scope}")
        return cond(f"{base}.company_id", "in", [int(company_id)])
    if company_url:
        cur = cond("experience.employment_details.current.company_professional_network_profile_url", "in", [company_url])
        past = cond("experience.employment_details.past.company_professional_network_profile_url", "in", [company_url])
        return {"current": cur, "past": past}.get(scope) or OR(cur, past)
    return None


def title_filter(titles: list[str], scope: str = "any") -> dict | None:
    """OR over title phrases using the all-words operator `(.)` on ANY role (current or past).

    Unqualified role paths are state-neutral: they do not condition on whether the person has
    since left a job (a `past.*` filter would silently select people who later moved = leak)."""
    base = "experience.employment_details" + ("" if scope == "any" else f".{scope}") + ".title"
    uniq = []
    for t in titles:
        t = (t or "").strip()
        if t and t.lower() not in [u.lower() for u in uniq]:
            uniq.append(t)
    return OR(*[cond(base, "(.)", t) for t in uniq[:12]]) if uniq else None


def location_filter(places: list[str]) -> dict | None:
    uniq = [p for p in dict.fromkeys(x.strip() for x in places if x and x.strip())]
    return OR(*[cond("basic_profile.location.full_location", "(.)", p) for p in uniq[:8]]) if uniq else None


def describe(filters: dict | None, depth: int = 0) -> str:
    if not filters:
        return "<none>"
    if "op" in filters:
        inner = f" {filters['op'].upper()} ".join(describe(c, depth + 1) for c in filters["conditions"])
        return f"({inner})" if depth else inner
    v = filters.get("value")
    if isinstance(v, list) and len(v) > 3:
        v = f"[{len(v)} values]"
    return f"{filters['field'].rsplit('.', 1)[-1]} {filters['type']} {v}"


# ------------------------------------------------------------------------------- results
@dataclass
class SearchResult:
    profiles: list[RawProfile] = field(default_factory=list)
    total_count: int | None = None
    pages: int = 0
    truncated: bool = False          # more results existed than we fetched
    credits: float = 0.0
    warnings: list[str] = field(default_factory=list)
    fields_used: list[str] = field(default_factory=list)


def _overlaps(a: str, b: str) -> bool:
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


class PeopleAPI:
    def __init__(self, client: CrustClient, cfg: dict):
        self.client = client
        self.cfg = cfg
        self.page_size = int(cfg["crust"]["page_size"])
        self.per_result = dict(cfg["crust"]["credits_per_result"])
        self.company_mode = str(cfg["crust"].get("company_filter_mode", "id"))

    # ---- low level ----------------------------------------------------------------------
    def _per_result(self, fields: list[str]) -> float:
        heavy = (not fields) or any(f.split(".")[0] in {"experience", "education"} for f in fields)
        return float(self.per_result["experience_education" if heavy else "base"])

    def _page(self, filters: dict, fields: list[str], limit: int, cursor: str | None,
              sorts: list[dict] | None, exclude_urls: list[str] | None, warnings: list[str]) -> tuple[dict, list[str], float]:
        fields = list(fields)
        for _attempt in range(5):
            body: dict[str, Any] = {"filters": filters, "fields": fields, "limit": int(limit), "sorts": sorts or NEUTRAL_SORT}
            if cursor:
                body["cursor"] = cursor
            if exclude_urls:
                body["post_processing"] = {"exclude_profiles": exclude_urls}
            try:
                r = self.client.post("/person/search", body, limiter="person_search")
                return r.json, fields, r.credits
            except CrustPermissionError as exc:
                denied = exc.denied_fields
                keep = [f for f in fields if not any(_overlaps(f, d) for d in denied)] if denied else fields
                if denied and keep != fields and "crustdata_person_id" in keep:
                    msg = f"403: dropped denied fields {sorted(set(fields) - set(keep))}"
                    warnings.append(msg)
                    log_event("fields_narrowed", reason="403", dropped=sorted(set(fields) - set(keep)))
                    fields = keep
                    continue
                raise
            except CrustBadRequest as exc:
                text = str(exc).lower()
                bad = [f for f in fields if f.lower() in text and f != "crustdata_person_id"]
                if bad:
                    msg = f"400: dropped rejected fields {bad}"
                    warnings.append(msg)
                    log_event("fields_narrowed", reason="400", dropped=bad, message=str(exc)[:200])
                    fields = [f for f in fields if f not in bad]
                    continue
                raise
        raise CrustBadRequest("could not satisfy field permissions after 5 attempts")

    # ---- public -------------------------------------------------------------------------
    def count(self, filters: dict) -> int | None:
        """Cardinality probe: limit=1, id only (~0.03 credits). None if the API omits total_count."""
        res = self.search(filters, fields=["crustdata_person_id"], max_results=1)
        return res.total_count

    def search(self, filters: dict, *, fields: list[str], max_results: int, page_size: int | None = None,
               sorts: list[dict] | None = None, exclude_urls: list[str] | None = None) -> SearchResult:
        out = SearchResult(fields_used=list(fields))
        seen: set[int] = set()
        cursor: str | None = None
        per = self._per_result(fields)
        meter = self.client.meter
        page_cap = int(page_size or self.page_size)
        while len(out.profiles) < max_results:
            want = min(page_cap, max_results - len(out.profiles), 1000)
            afford = meter.affordable_results(per)
            if afford <= 0:
                out.warnings.append(f"credit cap reached after {len(out.profiles)} profiles (best-so-far returned)")
                out.truncated = True
                log_event("credit_cap_reached", fetched=len(out.profiles), spent=round(meter.estimated, 2))
                break
            limit = max(1, min(want, afford))
            payload, used, credits = self._page(filters, fields, limit, cursor, sorts, exclude_urls, out.warnings)
            out.fields_used, out.credits, out.pages = used, out.credits + credits, out.pages + 1
            if isinstance(payload, dict) and payload.get("total_count") is not None:
                out.total_count = int(payload["total_count"])
            for rp in parse_profiles(payload):
                if rp.person_id not in seen:
                    seen.add(rp.person_id)
                    out.profiles.append(rp)
            cursor = payload.get("next_cursor") if isinstance(payload, dict) else None
            if not cursor:
                break
            if limit < want:  # budget clamp, stop paging
                out.truncated = True
                out.warnings.append("page truncated by credit cap")
                break
        else:
            out.truncated = bool(cursor)
        if out.total_count is not None and len(out.profiles) < out.total_count and max_results > 1:
            out.truncated = True
        return out

    def fetch_by_ids(self, ids: list[int], fields: list[str]) -> SearchResult:
        ids = [int(i) for i in ids]
        merged = SearchResult(fields_used=list(fields))
        for i in range(0, len(ids), 200):
            chunk = ids[i : i + 200]
            part = self.search(cond("crustdata_person_id", "in", chunk), fields=fields, max_results=len(chunk))
            merged.profiles += part.profiles
            merged.credits += part.credits
            merged.warnings += part.warnings
            merged.pages += part.pages
        merged.total_count = len(merged.profiles)
        return merged

    def company_employees(self, company_id: int | None, company_url: str | None, *, fields: list[str], cap: int) -> SearchResult:
        """Everyone with ANY role at the company (as of today). The Connection agent masks them to T
        and keeps only people employed at T - never the post-T joiners (leak #8)."""
        flt = company_filter(company_id, company_url, "any", self.company_mode)
        if flt is None:
            return SearchResult(warnings=["no company identifier: connection signal unavailable"])
        try:
            return self.search(flt, fields=fields, max_results=cap)
        except CrustBadRequest as exc:
            if self.company_mode == "id" and company_url:
                log_event("company_filter_fallback", message=str(exc)[:160])
                self.company_mode = "url"
                return self.search(company_filter(company_id, company_url, "any", "url"), fields=fields, max_results=cap)
            raise
