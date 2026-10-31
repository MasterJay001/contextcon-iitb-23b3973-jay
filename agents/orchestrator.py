"""Orchestrator-worker pipeline. Live mode and backtest mode run THIS code; only `T` and the backend differ
(live: T = today, trusted dates; backtest: T = posting date, masked history). The scoreboard therefore measures the exact
system the judges watch run.

    Brief Interpreter -> Sourcer (ReAct, via Time Guard) -> Fit | Movability | Connection (parallel-capable) -> Ranker
                      -> Critic -> shortlist (MMR) -> cards + outreach drafts (queued for approval)

`build_artifacts` does everything expensive once per case and caches it; `run_case` ranks (free) under any weights.
The evaluator is NOT in this file and never sees an agent's internals: it only receives the ranked list.
"""
from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from agents.artifacts import CaseArtifacts, artifact_key, load_artifacts, save_artifacts
from agents.cards import build_card
from agents.connection import ConnectionIndex
from agents.critic import Critic, CriticResult
from agents.fit import DEFAULT_RUBRIC, FitScorer, spec_query
from agents.interpreter import Interpreter
from agents.movability import MovabilityModel
from agents.outreach import OutreachQueue, draft_for, render_text
from agents.ranker import mmr_select, rank
from agents.sourcer import Sourcer
from agents.time_guard import LocalFilter, TargetRef, TimeGuard
from core.config import ROOT, config_hash, resolve_crust_backend, resolve_llm_backend, today
from core.events import log_event
from core.models import Case, CaseResult, JobSpec, MaskedProfile, RawProfile, Scored, Weights, seniority_label
from core.render import render_masked
from crust.client import CrustClient, CrustError, FaultInjector
from crust.companies import CompaniesAPI
from crust.people import PeopleAPI
from crust.sim import SimTransport
from crust.sim_world import SimWorld
from llm.client import LLM
from rag.index import BM25, Doc
from rag.knowledge import KnowledgeBase


@functools.lru_cache(maxsize=4)
def _world(seed: int, n_people: int, n_target: int, n_other: int, effect: str, betas: tuple[float, float, float] | None) -> SimWorld:
    return SimWorld(seed=seed, n_people=n_people, n_target=n_target, n_other=n_other, effect=effect, betas=betas)


def sim_world_from_cfg(cfg: dict) -> SimWorld:
    s = cfg["sim"]
    betas = tuple(float(x) for x in s["betas"]) if s.get("betas") else None
    if betas is not None and str(s["effect"]) == "null":
        betas = (betas[0], 0.0, 0.0)
    return _world(int(s["seed"]), int(s["n_people"]), int(s["n_target"]), int(s["n_other"]), str(s["effect"]), betas)


@dataclass
class Context:
    cfg: dict
    crust: CrustClient
    people: PeopleAPI
    companies: CompaniesAPI
    llm: LLM
    kb: KnowledgeBase
    mode: str
    crust_backend: str
    llm_backend: str
    world: SimWorld | None = None

    def backends(self) -> dict[str, str]:
        return {"crust": self.crust_backend, "llm": self.llm_backend}

    def close(self) -> None:
        self.crust.close()
        self.llm.close()


def build_context(cfg: dict, *, mode: str = "backtest", crust_backend: str | None = None, llm_backend: str | None = None,
                  faults: FaultInjector | None = None, use_cache: bool = True, cache_ns: str = "") -> Context:
    cb = crust_backend or resolve_crust_backend(cfg)
    lb = llm_backend or resolve_llm_backend(cfg)
    faults = faults if faults is not None else FaultInjector.from_env()
    world = None
    cache_root = ROOT / str(cfg["paths"]["cache"])
    if cb == "sim":
        world = sim_world_from_cfg(cfg)
        client = CrustClient(cfg, SimTransport(world, credits=1e9), api_key="sim", mode=mode, use_cache=use_cache,
                             cache_dir=cache_root / f"http-sim{cache_ns}", faults=faults)
    else:
        client = CrustClient(cfg, None, mode=mode, use_cache=use_cache, cache_dir=cache_root / f"http{cache_ns}", faults=faults)
    llm = LLM(cfg, lb, use_cache=use_cache, cache_dir=cache_root / f"llm{cache_ns}", faults=faults)
    kb = KnowledgeBase(cfg).load()
    return Context(cfg, client, PeopleAPI(client, cfg), CompaniesAPI(client, cfg), llm, kb, mode, cb, lb, world)


def cfg_fingerprint(cfg: dict) -> str:
    keys = ["sourcer", "mask", "fit", "movability", "connection", "crust"]
    sub = {k: cfg[k] for k in keys}
    sub["crust"] = {k: v for k, v in cfg["crust"].items() if k in {"people_fields", "page_size"}}
    sub["llm"] = {"models": cfg["llm"]["models"], "fit_batch_size": cfg["llm"]["fit_batch_size"]}
    return config_hash(sub)


def rubric_text(cfg: dict, lessons: str = "") -> str:
    return DEFAULT_RUBRIC + (f"\n\nLessons learned from earlier misses (general rules):\n{lessons.strip()}" if lessons.strip() else "")


def load_lessons(cfg: dict) -> str:
    p = ROOT / str(cfg["paths"]["lessons"])
    if not p.exists():
        return ""
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8-sig").splitlines() if ln.strip().startswith("- ")]
    return "\n".join(lines)


# ------------------------------------------------------------------------------- build
def build_artifacts(case: Case, ctx: Context, *, rubric: str | None = None, extra_raw: list[RawProfile] | None = None,
                    use_cache: bool = True, strict_mask: bool = False, force: bool = False, leak_on: bool = False) -> CaseArtifacts:
    """`leak_on=True` is the CANARY ONLY: the naive system that works from today's profiles (no masking, T_eff = today,
    current employees not excluded, roster includes post-T joiners). Its recall is the size of the leak we closed."""
    cfg = ctx.cfg
    rubric = rubric if rubric is not None else rubric_text(cfg, load_lessons(cfg))
    T_eff = today(cfg) if leak_on else case.T
    if ctx.crust.faults.active or (ctx.llm.faults is not None and ctx.llm.faults.active):
        use_cache = False          # fault-injected runs must never read or poison the artifact cache
    key = artifact_key(case.case_id, case.T.isoformat(), case.brief, hashlib.sha256(rubric.encode()).hexdigest()[:10], cfg_fingerprint(cfg),
                       ctx.crust_backend, ctx.llm_backend, ctx.mode, strict_mask, leak_on, sorted(p.person_id for p in (extra_raw or [])),
                       cfg["sim"] if ctx.crust_backend == "sim" else "")
    if use_cache and not force:
        hit = load_artifacts(cfg, key)
        if hit is not None:
            return hit
    t_all = time.time()
    stage: dict[str, float] = {}
    ctx.crust.meter.begin_case()
    warnings: list[str] = []

    # 1. brief -> spec
    t0 = time.time()
    interp = Interpreter(cfg, ctx.llm, ctx.kb)
    spec = interp.interpret(case.brief)
    if case.target_norm_title:                      # backtest cases carry the posting's own normalised fields
        spec.norm_title = case.target_norm_title
        spec.seniority = case.target_seniority or spec.seniority
        if case.location:
            spec.location = case.location
            spec.expanded_locations = interp_expand(spec.location)
    stage["interpret"] = time.time() - t0

    # 2. hiring company
    target = TargetRef(case.company_id, case.company_name or spec.company_name, case.company_url)
    if target.company_id is None and (target.company_name or spec.company_name):
        match = ctx.companies.resolve_name(target.company_name or spec.company_name or "")
        if match:
            target = TargetRef(match.company_id, match.name, match.profile_url)
        else:
            warnings.append(f"hiring company '{target.company_name or spec.company_name}' could not be resolved: continuing without the Connection signal")
    spec.company_name = spec.company_name or target.company_name

    guard = TimeGuard(T_eff, target=None if leak_on else target, cfg=cfg, strict=strict_mask)
    flt = LocalFilter.from_spec(spec, cfg, None if leak_on else target)

    # 3. sourcing
    t0 = time.time()
    sourcer = Sourcer(cfg, ctx.people, guard, flt, ctx.llm, ctx.kb, mode=ctx.mode)
    state = sourcer.run(spec, extra_raw=extra_raw)
    stage["source"] = time.time() - t0
    warnings += [w for w in state.warnings if w not in warnings]
    pool = sorted(state.passing.values(), key=lambda m: m.person_id)
    ctx.llm.forbid([p.name for p in state.raw.values()])

    # 4. roster (connection)
    t0 = time.time()
    roster: list[MaskedProfile] = []
    if target.company_id is not None or target.company_url:
        try:
            rfields = list(cfg["crust"]["people_fields"]["roster"])
            if cfg["connection"]["use_school_overlap"] and "education" not in rfields:
                rfields.append("education")
            res = ctx.people.company_employees(target.company_id, target.company_url, fields=rfields, cap=int(cfg["connection"]["roster_cap"]))
            masked = guard.admit(res.profiles)
            roster = [m for m in masked if guard.employed_at_T(m, target.company_id, target.company_name, target.company_url)]
            warnings += [w for w in res.warnings if w not in warnings]
            if res.truncated:
                warnings.append(f"roster truncated to {len(res.profiles)} of {res.total_count} indexed employees (connection features use a neutral sample)")
        except CrustError as exc:
            warnings.append(f"roster unavailable ({type(exc).__name__}): continuing without the Connection signal")
            log_event("roster_failed", error=str(exc)[:160])
    else:
        warnings.append("no hiring company given: Connection signal skipped")
    stage["roster"] = time.time() - t0

    art = CaseArtifacts(case_id=case.case_id, T=case.T, brief=case.brief, spec=spec, pool=pool)
    art.raw = {pid: state.raw[pid] for pid in (m.person_id for m in pool) if pid in state.raw}
    art.n_retrieved = len(state.raw)
    art.api_total_count = state.universe
    art.steps = [s.__dict__ for s in state.steps]
    art.reject_reasons = dict(state.reject)
    art.roster_size = len(roster)
    art.roster_ids = [m.person_id for m in roster]
    art.conn_available = bool(roster)
    if leak_on:
        art.flags.append("LEAK_ON_CANARY")
    if state.policy:
        art.flags.append(f"sourcer_policy={state.policy}")
    if extra_raw:
        art.flags.append("ranking_only_injected_profiles")

    if pool:
        # 5. fit
        t0 = time.time()
        fit = FitScorer(cfg, ctx.llm, ctx.kb)
        art.fit_feats, art.fit_details, rendered = fit.score_pool(pool, spec, case.brief, rubric)
        stage["fit"] = time.time() - t0

        # 6. movability (fit the hazard on pool + roster spells)
        t0 = time.time()
        mov = MovabilityModel(cfg)
        art.survival = mov.fit([*pool, *roster]).summary()
        growth: dict[int, dict] = {}
        if ctx.mode == "live" and case.T >= today(cfg) and pool:
            top_by_fit = sorted(pool, key=lambda m: (-art.fit_feats[m.person_id].get("title_match", 0), m.person_id))[:30]
            ids = sorted({m.employer_id_at_T for m in top_by_fit if m.employer_id_at_T is not None})
            if ids:
                growth = ctx.companies.growth(ids)
                if len(growth) < len(ids):
                    warnings.append(f"employer headcount trend available for {len(growth)} of {len(ids)} employers (credit cap / enrichment failures)")
        for mp in pool:
            hg = growth.get(mp.employer_id_at_T) if mp.employer_id_at_T is not None else None
            art.move_feats[mp.person_id], art.move_diag[mp.person_id] = mov.features(mp, hg)
        stage["movability"] = time.time() - t0

        # 7. connection
        t0 = time.time()
        idx = ConnectionIndex(cfg, roster, T_eff, target)
        art.feeders = idx.feeder_report()
        if idx.available:
            idx.compute_ppr(pool)
        for mp in pool:
            art.conn_feats[mp.person_id], art.conn_diag[mp.person_id] = idx.features(mp)
        if idx.available:
            idx.normalise_ppr(art.conn_feats)
        stage["connection"] = time.time() - t0

        # 8. A0 keyword baseline on masked text
        bm = BM25().fit([Doc(str(m.person_id), rendered[m.person_id].plain) for m in pool])
        sc = bm.scores(spec_query(spec, case.brief))
        art.bm25 = {m.person_id: float(s) for m, s in zip(pool, sc)}
    art.mask_totals = dict(guard.totals)
    art.credits = ctx.crust.meter.case_spent
    art.warnings = warnings
    stage["total"] = time.time() - t_all
    art.stage_seconds = {k: round(v, 3) for k, v in stage.items()}
    art.backends = ctx.backends()
    art.llm_usage = ctx.llm.usage.summary()
    if use_cache:
        save_artifacts(cfg, key, art)
    return art


def interp_expand(location: str) -> list[str]:
    from core.geo import geo

    return geo().expand(location)


# ------------------------------------------------------------------------------- run
def run_case(case: Case, ctx: Context, weights: Weights | None = None, *, rubric: str | None = None, variant: str = "A4",
             use_critic: bool = False, art: CaseArtifacts | None = None, extra_raw: list[RawProfile] | None = None,
             learned=None, use_school: bool = False, artifact_cache: bool = True) -> tuple[CaseResult, CaseArtifacts, CriticResult | None]:
    cfg = ctx.cfg
    weights = weights or Weights.from_cfg(cfg)
    art = art or build_artifacts(case, ctx, rubric=rubric, extra_raw=extra_raw, use_cache=artifact_cache)
    ranked = rank(art, weights, variant, use_school=use_school, learned=learned)
    crit: CriticResult | None = None
    if use_critic and ranked:
        crit = Critic(cfg, ctx.llm).review(art, ranked)
        ranked = rank(art, weights, variant, use_school=use_school, learned=learned, demoted=set(crit.demoted),
                      demote_penalty=float(cfg["critic"]["demote_penalty"]))
    res = CaseResult(case_id=case.case_id, mode=ctx.mode, T=case.T, jobspec=art.spec, ranked=ranked, pool_ids=art.pool_ids,
                     n_retrieved=art.n_retrieved, n_masked=len(art.pool), credits_spent=art.credits, seconds=art.stage_seconds.get("total", 0.0),
                     warnings=list(art.warnings), backends=art.backends, stage_seconds=art.stage_seconds, mask_totals=art.mask_totals)
    return res, art, crit


# ------------------------------------------------------------------------------- live service
@dataclass
class LiveOutput:
    result: CaseResult
    art: CaseArtifacts
    critic: CriticResult | None
    blocked: str | None = None
    queue_ids: list[str] = field(default_factory=list)


def live_case(brief: str, cfg: dict, company: str | None = None) -> Case:
    cid = "live-" + hashlib.sha256(f"{brief}|{company}".encode()).hexdigest()[:8]
    return Case(case_id=cid, T=today(cfg), brief=brief, company_name=company, route="live", split="live")


def run_live(brief: str, ctx: Context, *, company: str | None = None, weights: Weights | None = None, assume: bool = False,
             with_outreach: bool = True, artifact_cache: bool = True) -> LiveOutput:
    """The customer-facing service: brief in -> five cards with sourced claims + drafted outreach queued for approval."""
    cfg = ctx.cfg
    t0 = time.time()
    case = live_case(brief, cfg, company)
    interp = Interpreter(cfg, ctx.llm, ctx.kb)
    spec = interp.interpret(brief)
    block = Interpreter.blocking(spec)
    if block and (not assume or not spec.norm_title):          # --assume can override a contradiction, never a missing role
        res = CaseResult(case_id=case.case_id, mode="live", T=case.T, jobspec=spec, warnings=[f"NEEDS CLARIFICATION: {block}"], backends=ctx.backends())
        empty = CaseArtifacts(case.case_id, case.T, brief, spec, [])
        return LiveOutput(res, empty, None, blocked=block)
    case.target_norm_title = spec.norm_title
    case.target_seniority = spec.seniority
    case.location = spec.location or None
    case.company_name = company or spec.company_name
    res, art, crit = run_case(case, ctx, weights, variant="A4", use_critic=True, artifact_cache=artifact_cache)
    k = int(cfg["live"]["k"])
    pool = {m.person_id: m for m in art.pool}
    shortlist = mmr_select(res.ranked, pool, k, float(cfg["live"]["mmr_lambda"]))
    # thin results: do not pad with weak candidates
    strong = [s for s in shortlist if s.person_id not in (crit.demoted if crit else {}) or len(shortlist) <= 2]
    if len(strong) < len(shortlist):
        res.warnings.append(f"{len(shortlist) - len(strong)} candidate(s) removed by the Critic; fewer than {k} strong candidates exist")
        shortlist = strong
    elif len(art.pool) < k:
        res.warnings.append(f"only {len(art.pool)} candidates matched; returning fewer than {k} instead of padding")
    colleague_names: dict[int, str] = {}
    ids = sorted({p.colleague_id for s in shortlist for p in (art.conn_diag.get(s.person_id).paths if art.conn_diag.get(s.person_id) else [])})
    if ids and ctx.mode == "live":
        try:
            for rp in ctx.people.fetch_by_ids(ids, ["crustdata_person_id", "basic_profile.name"]).profiles:
                if rp.name:
                    colleague_names[rp.person_id] = rp.name
        except CrustError:
            pass
    cards = [build_card(i, s, art, crit, colleague_names=colleague_names) for i, s in enumerate(shortlist, 1)]
    out = LiveOutput(res, art, crit)
    if with_outreach and cards:
        queue = OutreachQueue(cfg)
        for card, s in zip(cards, shortlist):
            o = draft_for(ctx.llm, art, s, int(cfg["outreach"]["max_words"]))
            card.outreach = o
            first_colleague = next(iter(colleague_names.values()), None) if colleague_names else None
            out.queue_ids.append(queue.enqueue(case.case_id, s.person_id, card.rank, o, render_text(o, card.name, first_colleague)))
    res.cards = cards
    res.seconds = time.time() - t0
    res.stage_seconds = {**art.stage_seconds, "end_to_end": round(res.seconds, 2)}
    return out
