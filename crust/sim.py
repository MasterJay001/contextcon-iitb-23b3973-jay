"""Offline Crustdata stand-in: same routes, filter operators, pagination, field projection, credit
costs and error shapes as the documented API (2025-11-01), served from a `SimWorld`.

Documented semantics reproduced here: operators `= != in not_in (.) [.] (!) > => < =< is_null
is_not_null`; `and`/`or` groups; a condition on `experience.employment_details.*` matches when ANY
one role satisfies it (and an unqualified path means current OR past); `limit` 1..1000 (0..1000 for
jobs); opaque cursors; sorts must be supported fields; 0.03 credits/result (0.23 with experience or
education); 402 when credits run out; 403 `permission_error` with `denied_fields` for gated
sections; 400 for unknown fields with the valid list in the message.
"""
from __future__ import annotations

import base64
import json
import re
import time
from datetime import timedelta
from typing import Any

from crust.client import TransportResponse
from crust.sim_world import SimWorld

_TOKEN = re.compile(r"[a-z0-9+#]+")
_ROLE = "experience.employment_details"
_ROLE_ALIASES = {"company_id": "crustdata_company_id", "company_name": "name"}
_TOP_ALIASES = {"recently_changed_jobs": "experience.recently_changed_jobs", "years_of_experience": "years_of_experience_raw"}
_VALID_SORTS = {"crustdata_person_id", "basic_profile.name", "metadata.updated_at", "experience.employment_details.start_date",
                "experience.employment_details.company_id", "metadata.date_added", "years_of_experience_raw"}
_POS = {"=", "in", "(.)", "[.]", ">", "=>", "<", "=<"}
_NEG = {"!=": "=", "not_in": "in", "(!)": "(.)"}


def _err(status: int, type_: str, message: str, metadata: dict | None = None) -> TransportResponse:
    return TransportResponse(status, {}, {"error": {"type": type_, "message": message, "metadata": metadata or {}}}, 0.0)


def _flatten(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        out: list = []
        for x in v:
            out += _flatten(x)
        return out
    return [v]


def _get(obj: Any, dotted: str) -> list:
    cur = [obj]
    for key in dotted.split("."):
        nxt = []
        for o in cur:
            if isinstance(o, dict) and key in o:
                nxt += _flatten(o[key]) if not isinstance(o[key], dict) else [o[key]]
            elif isinstance(o, list):
                for item in o:
                    if isinstance(item, dict) and key in item:
                        nxt += _flatten(item[key]) if not isinstance(item[key], dict) else [item[key]]
        cur = nxt
    return [x for x in cur if x is not None]


def resolve(person: dict, path: str) -> list:
    path = _TOP_ALIASES.get(path, path)
    if path == _ROLE or path.startswith(_ROLE + "."):
        rest = path[len(_ROLE):].lstrip(".").split(".") if path != _ROLE else []
        ed = (person.get("experience") or {}).get("employment_details") or {}
        if rest and rest[0] in ("current", "past"):
            roles, rest = list(ed.get(rest[0]) or []), rest[1:]
        else:
            roles = list(ed.get("current") or []) + list(ed.get("past") or [])
        key = ".".join(rest)
        key = _ROLE_ALIASES.get(key, key)
        if not key:
            return roles
        out: list = []
        for r in roles:
            out += _get(r, key)
        return out
    return _get(person, path)


def _tokens(s: Any) -> list[str]:
    return _TOKEN.findall(str(s).lower())


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def _cmp(op: str, v: Any, target: Any) -> bool:
    if op == "=":
        return str(v).strip().lower() == str(target).strip().lower()
    if op == "in":
        return str(v).strip().lower() in {str(t).strip().lower() for t in _flatten(target)}
    if op == "(.)":
        have = {_stem(t) for t in _tokens(v)}
        return all(_stem(t) in have for t in _tokens(target))
    if op == "[.]":
        return " ".join(_tokens(target)) in " ".join(_tokens(v))
    if op in {">", "=>", "<", "=<"}:
        sv, st = str(v), str(target)
        if re.match(r"^\d{4}-\d{2}-\d{2}", st):
            a, b = sv[:10], st[:10]
        else:
            try:
                a, b = float(v), float(target)
            except (TypeError, ValueError):
                return False
        return {">": a > b, "=>": a >= b, "<": a < b, "=<": a <= b}[op]
    return False


def eval_cond(person: dict, c: dict) -> bool:
    op, vals = c.get("type"), resolve(person, str(c.get("field")))
    target = c.get("value")
    if op == "is_null":
        return len(vals) == 0
    if op == "is_not_null":
        return len(vals) > 0
    if op in _NEG:
        return not any(_cmp(_NEG[op], v, target) for v in vals)
    if op in _POS:
        return any(_cmp(op, v, target) for v in vals)
    raise ValueError(f"unsupported operator '{op}'")


def eval_filter(person: dict, node: dict) -> bool:
    if "op" in node:
        results = (eval_filter(person, ch) for ch in node.get("conditions", []))
        return all(results) if node["op"] in ("and", "all_of") else any(results)
    return eval_cond(person, node)


def _project(person: dict, fields: list[str] | None) -> dict:
    if not fields:
        return {k: v for k, v in person.items() if k in {"crustdata_person_id", "basic_profile", "experience", "education"}}
    out: dict = {"crustdata_person_id": person["crustdata_person_id"]}
    for f in fields:
        parts, src, dst = f.split("."), person, out
        ok = True
        for i, p in enumerate(parts):
            if not isinstance(src, dict) or p not in src:
                ok = False
                break
            if i == len(parts) - 1:
                dst[p] = src[p]
            else:
                dst = dst.setdefault(p, {})
                src = src[p]
        if not ok:
            continue
    return out


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"o": offset}).encode()).decode()


def _decode_cursor(c: str | None) -> int:
    if not c:
        return 0
    try:
        return int(json.loads(base64.urlsafe_b64decode(c.encode()).decode())["o"])
    except Exception:
        return -1


class SimTransport:
    """Drop-in `Transport` for `CrustClient`."""

    def __init__(self, world: SimWorld, *, credits: float = 15000.0, api_version: str = "2025-11-01", latency_s: float = 0.0):
        self.world = world
        self.balance = float(credits)
        self.api_version = api_version
        self.latency_s = latency_s
        self.requests: list[tuple[str, str]] = []
        self.rows_returned = 0
        self._sample = next(iter(world._payloads.values()))
        self._persons = list(world._payloads.values())
        self._jobs = [world.job_payload(p) for p in world.postings]

    # ---- plumbing ----------------------------------------------------------------------
    def request(self, method: str, path: str, *, headers: dict, json_body: Any | None, timeout: float) -> TransportResponse:
        self.requests.append((method, path))
        if self.latency_s:
            time.sleep(self.latency_s)
        if headers.get("x-api-version") != self.api_version:
            return _err(400, "invalid_request", f"x-api-version must be {self.api_version}")
        if not str(headers.get("Authorization", "")).startswith("Bearer "):
            return _err(401, "unauthorized", "missing bearer token")
        path = path.rstrip("/")
        try:
            if (method, path) == ("GET", "/account/credits"):
                return TransportResponse(200, {}, {"account": {"credits": round(self.balance, 2), "recurring_credits": 15000}}, 0.0)
            if (method, path) == ("POST", "/person/search"):
                return self._person_search(json_body or {})
            if (method, path) == ("POST", "/person/search/autocomplete"):
                return self._autocomplete(json_body or {})
            if (method, path) == ("POST", "/job/search"):
                return self._job_search(json_body or {})
            if (method, path) in {("POST", "/company/identify"), ("POST", "/company/enrich")}:
                return self._company(path, json_body or {})
        except ValueError as exc:
            return _err(400, "invalid_request", str(exc))
        return _err(404, "not_found", f"no route {method} {path}")

    def _charge(self, credits: float) -> TransportResponse | None:
        if credits > self.balance:
            return _err(402, "insufficient_credits", "not enough credits")
        self.balance -= credits
        return None

    # ---- people ------------------------------------------------------------------------
    def _valid_field(self, f: str) -> bool:
        cur: Any = self._sample
        for p in f.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return False
            cur = cur[p]
        return True

    def _person_search(self, body: dict) -> TransportResponse:
        filters = body.get("filters")
        if not isinstance(filters, dict) or not filters:
            return _err(400, "invalid_request", "filters is required")
        limit = body.get("limit", body.get("count", 20))
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            return _err(400, "invalid_request", "limit must be between 1 and 1000")
        fields = body.get("fields")
        if fields is not None:
            bad = [f for f in fields if not self._valid_field(f)]
            if bad:
                valid = sorted(self._sample.keys())
                return _err(400, "invalid_request", f"unknown field(s) {bad}. Valid top-level fields: {valid}")
            if self.world.deny_skills and any(f.split(".")[0] == "skills" for f in fields):
                return _err(403, "permission_error", "field not enabled for this key",
                            {"denied_fields": ["skills"], "permitted_fields": [k for k in self._sample if k != "skills"]})
        sorts = body.get("sorts") or []
        for s in sorts:
            if s.get("field") not in _VALID_SORTS:
                return _err(400, "invalid_request", f"cannot sort by {s.get('field')}")
        offset = _decode_cursor(body.get("cursor"))
        if offset < 0:
            return _err(400, "invalid_request", "invalid cursor")
        excl = set((body.get("post_processing") or {}).get("exclude_profiles") or [])
        try:
            matched = [p for p in self._persons if eval_filter(p, filters)]
        except ValueError as exc:
            return _err(400, "invalid_request", str(exc))
        if excl:
            matched = [p for p in matched if (p["social_handles"]["professional_network_identifier"]["profile_url"] not in excl)]
        for s in reversed(sorts):
            rev = s.get("order", "asc") == "desc"
            fld = s["field"]
            matched.sort(key=lambda p: (min(map(str, resolve(p, fld))) if resolve(p, fld) else ""), reverse=rev)
        if not sorts:
            matched.sort(key=lambda p: p["crustdata_person_id"])
        total = len(matched)
        page = matched[offset: offset + limit]
        heavy = (not fields) or any(f.split(".")[0] in {"experience", "education"} for f in fields)
        cost = len(page) * (0.23 if heavy else 0.03)
        denied = self._charge(cost)
        if denied:
            return denied
        self.rows_returned += len(page)
        nxt = _encode_cursor(offset + limit) if offset + limit < total else None
        return TransportResponse(200, {}, {"profiles": [_project(p, fields) for p in page], "next_cursor": nxt,
                                           "total_count": total, "total_count_relation": None, "remarks": []}, 0.0)

    def _autocomplete(self, body: dict) -> TransportResponse:
        field = str(body.get("field", ""))
        q = str(body.get("query", "")).lower()
        vals: list[str] = []
        for p in self._persons[:600]:
            vals += [str(v) for v in resolve(p, field)]
        counts: dict[str, int] = {}
        for v in vals:
            if q in v.lower():
                counts[v] = counts.get(v, 0) + 1
        top = sorted(counts, key=lambda k: (-counts[k], k))[: int(body.get("limit", 20))]
        return TransportResponse(200, {}, {"suggestions": [{"value": v} for v in top]}, 0.0)

    # ---- jobs --------------------------------------------------------------------------
    def _job_search(self, body: dict) -> TransportResponse:
        limit = body.get("limit", 20)
        if not isinstance(limit, int) or not 0 <= limit <= 1000:
            return _err(400, "invalid_request", "limit must be between 0 and 1000")
        filters = body.get("filters")
        rows = [j for j in self._jobs if (not filters or eval_filter_job(j, filters))]
        for s in reversed(body.get("sorts") or []):
            if s.get("field") == "company.basic_info.company_id":
                return _err(400, "invalid_request", "cannot sort by company_id; sort on metadata.date_added")
            rows.sort(key=lambda j: j["metadata"]["date_added"], reverse=s.get("order") == "desc")
        offset = _decode_cursor(body.get("cursor"))
        page = rows[offset: offset + limit] if limit else []
        denied = self._charge(0.03 * len(page))
        if denied:
            return denied
        nxt = _encode_cursor(offset + limit) if limit and offset + limit < len(rows) else None
        return TransportResponse(200, {}, {"job_listings": page, "next_cursor": nxt, "total_count": len(rows)}, 0.0)

    # ---- companies ---------------------------------------------------------------------
    def _company(self, path: str, body: dict) -> TransportResponse:
        keys = [k for k in ("names", "domains", "professional_network_profile_urls", "crustdata_company_ids") if body.get(k)]
        if len(keys) != 1:
            return _err(400, "invalid_request", "exactly one identifier type per request")
        key = keys[0]
        values = body[key]
        if len(values) > 25:
            return _err(400, "invalid_request", "max 25 values per request")
        out = []
        for v in values:
            hit = None
            for c in self.world.companies.values():
                if key == "names" and (str(v).strip().lower() == c.name.lower() or str(v).strip().lower() in c.name.lower()):
                    hit = c
                elif key == "domains" and str(v).lower() == f"{c.slug}.com":
                    hit = c
                elif key == "professional_network_profile_urls" and v == c.url:
                    hit = c
                elif key == "crustdata_company_ids" and int(v) == c.cid:
                    hit = c
                if hit:
                    break
            if not hit:
                out.append({"matched_on": v, "match_type": key, "matches": []})
                continue
            data: dict = {"crustdata_company_id": hit.cid, "basic_info": {
                "name": hit.name, "primary_domain": f"{hit.slug}.com", "professional_network_url": hit.url,
                "professional_network_id": str(hit.cid), "employee_count_range": hit.size_range, "industries": [hit.industry]}}
            if path.endswith("enrich"):
                cost = self._charge(2.0)
                if cost:
                    return cost
                now = self.world.now
                n_now = len(self.world.employed_at(hit.cid, now))
                n_6 = len(self.world.employed_at(hit.cid, now - timedelta(days=183)))
                n_12 = len(self.world.employed_at(hit.cid, now - timedelta(days=365)))
                g = lambda a, b: round((a / b - 1) * 100, 2) if b else None  # noqa: E731
                data["headcount"] = {"total": hit.size, "growth_percent": {"six_months": g(n_now, n_6), "yoy": g(n_now, n_12)}}
            out.append({"matched_on": v, "match_type": key, "matches": [{"confidence_score": 0.99, "company_data": data}]})
        return TransportResponse(200, {}, out, 0.0)


def eval_filter_job(job: dict, node: dict) -> bool:
    if "op" in node:
        results = (eval_filter_job(job, ch) for ch in node.get("conditions", []))
        return all(results) if node["op"] in ("and", "all_of") else any(results)
    f = node.get("field", "")
    f = f.replace("company.basic_info.company_id", "company.basic_info.crustdata_company_id")
    op, vals = node.get("type"), _get(job, f)
    target = node.get("value")
    if op in _NEG:
        return not any(_cmp(_NEG[op], v, target) for v in vals)
    return any(_cmp(op, v, target) for v in vals)
