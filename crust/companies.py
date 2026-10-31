"""Company Identify (free) and Company Enrichment (2 credits/company).

Documented facts we rely on:
  * /company/identify: exactly ONE identifier array per request (<= 25 values), free, returns
    [{matched_on, match_type, matches:[{confidence_score, company_data:{crustdata_company_id, basic_info}}]}].
  * /company/enrich:  same request shape plus `fields`; `headcount.total` and trailing
    `headcount.growth_percent.{mom,qoq,six_months,yoy,two_years}`. There is NO historical
    time series, so `employer_headcount_trend` is a LIVE-ONLY feature (leak #7): in a backtest the
    trailing windows describe today, not T, so the feature is switched off there.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.events import log_event
from crust.client import CrustClient, CrustError


@dataclass
class CompanyMatch:
    company_id: int | None
    name: str | None
    domain: str | None
    profile_url: str | None
    employee_range: str | None
    confidence: float
    matched_on: str = ""

    @property
    def resolved(self) -> bool:
        return self.company_id is not None


class CompaniesAPI:
    def __init__(self, client: CrustClient, cfg: dict):
        self.client = client
        self.cfg = cfg

    def identify(self, *, names: list[str] | None = None, domains: list[str] | None = None,
                 urls: list[str] | None = None, ids: list[int] | None = None) -> list[CompanyMatch]:
        kinds = {"names": names, "domains": domains, "professional_network_profile_urls": urls, "crustdata_company_ids": ids}
        given = {k: v for k, v in kinds.items() if v}
        if len(given) != 1:
            raise ValueError("exactly one identifier type per request")
        key, values = next(iter(given.items()))
        out: list[CompanyMatch] = []
        for i in range(0, len(values), 25):
            body: dict[str, Any] = {key: values[i : i + 25]}
            r = self.client.post("/company/identify", body, limiter="company_identify")
            items = r.json if isinstance(r.json, list) else []
            for item in items:
                matches = (item or {}).get("matches") or []
                if not matches:
                    out.append(CompanyMatch(None, None, None, None, None, 0.0, str(item.get("matched_on", ""))))
                    continue
                best = max(matches, key=lambda m: float(m.get("confidence_score") or 0.0))
                cd = best.get("company_data") or {}
                bi = cd.get("basic_info") or {}
                out.append(CompanyMatch(
                    company_id=cd.get("crustdata_company_id"), name=bi.get("name"), domain=bi.get("primary_domain"),
                    profile_url=bi.get("professional_network_url"), employee_range=bi.get("employee_count_range"),
                    confidence=float(best.get("confidence_score") or 0.0), matched_on=str(item.get("matched_on", "")),
                ))
        return out

    def resolve_name(self, name: str) -> CompanyMatch | None:
        """Best match for a company name; None if unresolved (caller proceeds without Connection)."""
        try:
            res = self.identify(names=[name])
        except (CrustError, ValueError) as exc:
            log_event("company_identify_failed", name=name, error=str(exc)[:160])
            return None
        return res[0] if res and res[0].resolved else None

    def growth(self, company_ids: list[int]) -> dict[int, dict]:
        """Trailing headcount growth percentages (LIVE ONLY). Returns {company_id: {six_months, yoy, total}}."""
        out: dict[int, dict] = {}
        for i in range(0, len(company_ids), 25):
            chunk = company_ids[i : i + 25]
            try:
                r = self.client.post(
                    "/company/enrich", {"crustdata_company_ids": chunk, "fields": ["basic_info", "headcount"]},
                    limiter="company_enrich", ttl_hours=6,
                )
            except CrustError as exc:
                log_event("company_enrich_failed", error=str(exc)[:160])
                continue
            for item in r.json if isinstance(r.json, list) else []:
                for m in (item or {}).get("matches") or []:
                    cd = m.get("company_data") or {}
                    hc = cd.get("headcount") or {}
                    gp = hc.get("growth_percent") or {}
                    cid = cd.get("crustdata_company_id")
                    if cid is not None:
                        out[int(cid)] = {"total": hc.get("total"), "six_months": gp.get("six_months"), "yoy": gp.get("yoy")}
                    break
        return out
