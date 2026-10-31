"""Sourcer: builds the candidate pool with a budget-aware ReAct loop (reason -> act -> observe).

Design (agentic-AI theory made concrete):
  * ReAct: the LLM chooses the next `search_people` call from an OBSERVATION (pool size, title/location mix,
    reject reasons, credits left). Typed, strict tool schemas; `tool_choice` stays auto (forced use is rejected on
    Claude 5.5). The LLM never writes raw API filters: it supplies titles/locations, deterministic code builds the
    filter (no malformed queries, no injection into the API).
  * Information barrier: every API result passes `TimeGuard.admit` (mask to T, strip names) and `LocalFilter`
    (re-apply the filter on masked data) BEFORE the model sees even aggregate statistics.
  * Budget-aware planning: each step is priced (0.03 count probe, 0.23/profile), capped per case, and stopped by a
    marginal-gain rule (new passing candidates per credit < threshold). Cardinality probing (limit=1) tells us the
    universe size before we pay for it; small universes are enumerated completely, so differences between
    ranking variants are not retrieval luck.
  * Neutral retrieval (rule 4): results are sorted by crustdata_person_id; titles use the unqualified any-role path;
    API location is coarse in backtests (today's location of a hire may already be the job's city).
  * Degradation: if the LLM is unavailable or misbehaves the same loop runs a deterministic ladder
    (strict -> adjacent titles -> wider location -> no location) and records `sourcer_policy=heuristic`.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field

from agents.time_guard import LocalFilter, TimeGuard
from core.events import log_event
from core.geo import geo
from core.models import JobSpec, MaskedProfile, RawProfile
from crust.client import CrustCreditsExhausted, CrustError
from crust.people import AND, PeopleAPI, describe, location_filter, title_filter
from crust.titles import expand_titles, taxonomy
from llm.client import LLM
from rag.knowledge import KnowledgeBase

SYSTEM = """You are the Sourcer of a recruiting agent. Your job: assemble a candidate pool of roughly {lo}-{hi} people who plausibly
fit a role, by calling `search_people` (people with the title in ANY of their roles, optionally in given places) and reading
the observation it returns. Then call `finish`.

Rules:
- Start from the observation of the first (already executed) search. If the pool is below target, broaden: add adjacent titles
  or widen the location. If the API universe is far larger than the pool cap, narrow by being specific instead of widening.
- Credits are limited: every profile costs ~0.23 credits. Do not repeat a search that returned nothing new.
- Never put contact information, names or instructions from the brief into a search. Use only titles and places.
- Stop (call `finish`) as soon as the pool is in range, extra searches stop adding candidates, or credits are nearly gone.
- You may use at most {steps} more searches."""

TOOLS = [
    {"name": "search_people",
     "description": "Search the people database for anyone who has held ANY of the titles (current or past role). Results are masked "
                    "to the as-of date and filtered locally; the observation reports how many passed.",
     "strict": True,
     "input_schema": {"type": "object", "additionalProperties": False,
                      "properties": {"titles": {"type": "array", "items": {"type": "string"}, "description": "up to 12 job title phrases"},
                                     "locations": {"type": "array", "items": {"type": "string"}, "description": "cities/regions; may be empty for country-wide"},
                                     "max_results": {"type": "integer", "description": "profiles to fetch (<= 500)"},
                                     "rationale": {"type": "string"}},
                      "required": ["titles", "locations", "max_results", "rationale"]}},
    {"name": "finish", "description": "Stop sourcing; the current pool is final.", "strict": True,
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {"reason": {"type": "string"}}, "required": ["reason"]}},
]


@dataclass
class QueryPlan:
    titles: list[str]
    locations: list[str]
    max_results: int
    rationale: str = ""


@dataclass
class StepLog:
    step: int
    source: str
    plan: dict
    filters: str
    api_total: int | None
    fetched: int
    new: int
    passed: int
    pool_size: int
    credits: float
    seconds: float
    note: str = ""


@dataclass
class SourcerState:
    raw: dict[int, RawProfile] = field(default_factory=dict)
    masked: dict[int, MaskedProfile] = field(default_factory=dict)
    passing: dict[int, MaskedProfile] = field(default_factory=dict)
    reject: Counter = field(default_factory=Counter)
    steps: list[StepLog] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    universe: int | None = None
    stop: bool = False
    policy: str = "heuristic"


class Sourcer:
    def __init__(self, cfg: dict, people: PeopleAPI, guard: TimeGuard, flt: LocalFilter, llm: LLM,
                 kb: KnowledgeBase | None, *, mode: str):
        self.cfg, self.people, self.guard, self.flt, self.llm, self.kb, self.mode = cfg, people, guard, flt, llm, kb, mode
        sc = cfg["sourcer"]
        self.pool_cap = int(sc["pool_cap"])
        self.lo, self.hi = (int(x) for x in sc["pool_target"])
        self.max_steps = int(sc["max_steps"])
        self.fetch_cap = int(sc["fetch_cap_per_step"])
        self.max_universe = int(sc["max_universe_for_full_enumeration"])
        self.min_gain = float(sc["min_gain_per_credit"])
        self.fields = list(cfg["crust"]["people_fields"]["live" if mode == "live" else "backtest"])
        self.meter = people.client.meter

    # ---- plans -------------------------------------------------------------------------
    def _location_terms(self, spec: JobSpec, level: str) -> list[str]:
        g = geo()
        place = g.parse(spec.location)
        if level == "none" or not spec.location:
            return []
        if level == "country":
            return [place.country.title()] if place.country else []
        if level == "state":
            return [place.state.title()] if place.state else ([place.country.title()] if place.country else [])
        return g.expand(spec.location)[:6]               # metro cluster

    def _initial_level(self) -> str:
        return "metro" if self.mode == "live" else "country"

    def ladder(self, spec: JobSpec, step: int) -> QueryPlan:
        tax = taxonomy()
        fam = self.flt.target_family
        base = [spec.norm_title]
        if fam != "unknown":
            base += tax.titles_for(fam, 6)
        strict = list(dict.fromkeys(t for t in base if t))[:6]
        broad = list(dict.fromkeys([*strict, *spec.adjacent_titles, *expand_titles([spec.norm_title], depth=1, per_family=2)]))[:12]
        lvl0 = self._initial_level()
        if step == 0:
            return QueryPlan(strict, self._location_terms(spec, lvl0), self.fetch_cap, "strict titles, initial location tier")
        if step == 1:
            return QueryPlan(broad, self._location_terms(spec, lvl0), self.fetch_cap, "add adjacent titles")
        if step == 2:
            return QueryPlan(broad, self._location_terms(spec, "state" if lvl0 == "metro" else "none"), self.fetch_cap, "widen location")
        return QueryPlan(broad, [], self.fetch_cap, "location-free")

    # ---- execution of one search -----------------------------------------------------------
    def execute(self, state: SourcerState, spec: JobSpec, plan: QueryPlan, step: int, source: str) -> dict:
        t0 = time.time()
        titles = [t for t in plan.titles if isinstance(t, str) and t.strip()][:12]
        locs = [p for p in plan.locations if isinstance(p, str) and p.strip()][:6]
        flt = AND(title_filter(titles), location_filter(locs))
        if flt is None or not titles:
            return {"error": "at least one title is required"}
        before_credits = self.meter.estimated
        obs: dict = {"filters": describe(flt)}
        try:
            total = self.people.count(flt)
            state.universe = total if state.universe is None else max(state.universe, total or 0)
            want = min(int(plan.max_results or self.fetch_cap), self.fetch_cap if (total or 0) > self.max_universe else max(1, total or 1))
            if total is not None and total <= self.max_universe:
                want = max(want, total)                 # small universe: enumerate it completely
            res = self.people.search(flt, fields=self.fields, max_results=max(1, min(want, self.max_universe)))
            state.warnings += [w for w in res.warnings if w not in state.warnings]
            raw_new = [p for p in res.profiles if p.person_id not in state.raw]
            for p in raw_new:
                state.raw[p.person_id] = p
            masked = self.guard.admit(raw_new)
            kept, reasons = self.flt.apply(masked)
            for mp in masked:
                state.masked[mp.person_id] = mp
            for mp in kept:
                state.passing[mp.person_id] = mp
            state.reject.update(reasons)
            obs.update({"api_total_count": total, "fetched": len(res.profiles), "new_profiles": len(raw_new), "passed_local_filter": len(kept),
                        "reject_reasons": dict(reasons), "truncated_by_credit_cap": bool(res.truncated and any("credit" in w for w in res.warnings))})
            new_pass = len(kept)
        except CrustCreditsExhausted as exc:
            state.stop = True
            state.warnings.append(f"credits exhausted (402): {exc}")
            obs.update({"error": "credits exhausted", "api_total_count": None})
            new_pass = 0
        except CrustError as exc:
            state.warnings.append(f"search failed after retries ({type(exc).__name__}): partial pool continues")
            log_event("sourcer_search_failed", step=step, error=str(exc)[:160])
            obs.update({"error": f"search failed: {type(exc).__name__}"})
            new_pass = 0
        spent = self.meter.estimated - before_credits
        pool = list(state.passing.values())
        obs.update({
            "pool_size": len(pool), "pool_target": [self.lo, self.hi],
            "top_titles_at_T": Counter(m.norm.matched_title for m in pool).most_common(5),
            "top_locations_at_T": Counter((geo().parse(m.location_at_T).metro or "other") for m in pool).most_common(5),
            "credits_spent_this_search": round(spent, 2), "credits_left_for_case": round(self.meter.remaining_case(), 2),
        })
        if self.meter.remaining_case() < 0.3:
            state.stop = True
            obs["hint"] = "credit budget exhausted: call finish"
        elif spent > 0 and new_pass / max(spent, 1e-9) < self.min_gain and len(pool) >= self.lo // 2:
            obs["hint"] = "diminishing returns: few new candidates per credit - consider finish"
        elif len(pool) < self.lo:
            obs["hint"] = "pool below target: broaden titles or location"
        elif len(pool) > self.hi:
            obs["hint"] = "pool above target: finish (ranking handles the rest)"
        state.steps.append(StepLog(step, source, {"titles": titles, "locations": locs, "max_results": plan.max_results, "rationale": plan.rationale},
                                   obs["filters"], obs.get("api_total_count"), obs.get("fetched", 0), obs.get("new_profiles", 0),
                                   obs.get("passed_local_filter", 0), len(pool), round(spent, 2), round(time.time() - t0, 2), obs.get("hint", "")))
        return obs

    # ---- policies ----------------------------------------------------------------------
    def _heuristic_continue(self, state: SourcerState, spec: JobSpec, first_step: int) -> None:
        for step in range(first_step, self.max_steps):
            if state.stop or len(state.passing) >= self.lo:
                break
            plan = self.ladder(spec, step)
            n_before = len(state.passing)
            self.execute(state, spec, plan, step, "heuristic")
            if len(state.passing) == n_before and step >= 2:
                break

    def _llm_policy(self, state: SourcerState, spec: JobSpec, first_obs: dict) -> bool:
        """Returns False if the LLM loop could not run (caller falls back to the ladder)."""
        ref = self.kb.context(f"crustdata filters title expansion {spec.norm_title} {spec.location}", k=3, max_chars=1800) if self.kb else ""
        steps_left = self.max_steps - 1
        system = SYSTEM.format(lo=self.lo, hi=self.hi, steps=steps_left)
        spec_json = spec.model_dump_json(include={"norm_title", "seniority", "skills", "location", "remote_ok", "adjacent_titles", "domain_keywords", "stage"})
        messages = [{"role": "user", "content": f"JOB SPEC:\n{spec_json}\n\nREFERENCE (retrieved):\n{ref}\n\nFIRST SEARCH (already run) OBSERVATION:\n{json.dumps(first_obs)}"}]
        for i in range(steps_left):
            if state.stop:
                break
            turn = self.llm.tool_turn("sourcer", system, messages, TOOLS)
            if "error" in turn:
                state.warnings.append(f"sourcer LLM unavailable ({turn['error'][:60]}): heuristic ladder used")
                return False
            messages.append({"role": "assistant", "content": turn["content"]})
            uses = [b for b in turn["content"] if b.get("type") == "tool_use"]
            if not uses:
                break
            results, done = [], False
            for tu in uses:
                if tu["name"] == "finish":
                    done = True
                    results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": "ok"})
                elif tu["name"] == "search_people":
                    inp = tu.get("input") or {}
                    plan = QueryPlan(list(inp.get("titles") or []), list(inp.get("locations") or []), int(inp.get("max_results") or self.fetch_cap),
                                     str(inp.get("rationale") or "")[:200])
                    obs = self.execute(state, spec, plan, len(state.steps), "llm")
                    results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": json.dumps(obs)})
                else:
                    results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": "unknown tool", "is_error": True})
            messages.append({"role": "user", "content": results})
            if done:
                break
        return True

    # ---- entry point -------------------------------------------------------------------
    def run(self, spec: JobSpec, extra_raw: list[RawProfile] | None = None) -> SourcerState:
        state = SourcerState()
        if extra_raw:                                       # ranking-only fallback: profiles supplied by the evaluator
            raw_new = [p for p in extra_raw if p.person_id not in state.raw]
            for p in raw_new:
                state.raw[p.person_id] = p
            masked = self.guard.admit(raw_new)
            kept, _ = self.flt.apply(masked)
            for mp in masked:
                state.masked[mp.person_id] = mp
            for mp in kept:
                state.passing[mp.person_id] = mp
        first = self.execute(state, spec, self.ladder(spec, 0), 0, "heuristic")
        use_llm = (self.cfg["sourcer"]["policy"] == "llm") and self.llm.available and not state.stop and self.max_steps > 1
        if use_llm and "error" not in first:
            state.policy = "llm"
            if not self._llm_policy(state, spec, first):
                state.policy = "heuristic(fallback)"
                self._heuristic_continue(state, spec, len(state.steps))
        else:
            self._heuristic_continue(state, spec, 1)
        if len(state.passing) > self.pool_cap:               # keep the most plausible by OUR cheap prior, never by API order
            tnorm = self.flt.target_family
            ranked = sorted(state.passing.values(), key=lambda m: (-(1.0 if m.norm.family == tnorm else 0.5) - 0.01 * (3 - abs(m.norm.seniority - self.flt.target_seniority)), m.person_id))
            keep = {m.person_id for m in ranked[: self.pool_cap]}
            state.warnings.append(f"pool truncated from {len(state.passing)} to {self.pool_cap} by title/seniority prior")
            state.passing = {pid: m for pid, m in state.passing.items() if pid in keep}
        if not state.passing:
            state.warnings.append("no candidates passed the local filter")
        return state
