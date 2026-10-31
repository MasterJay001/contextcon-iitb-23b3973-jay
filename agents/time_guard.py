"""Time Guard - deterministic code, not an LLM. The only door through which a profile reaches an agent.

    RawProfile (what the API says TODAY)  --mask_profile(T)-->  MaskedProfile (what was knowable on T)

It closes leak classes 1,2,3,5,8,9 of the design doc (see README "Leakage"):

 #1 post-T jobs      roles that start after T are dropped; roles that ended after T become ongoing at T
 #2 free text        headline / summary / current_title / normalized_title / recently_changed_jobs /
                     years_at_company* are never copied; the current title is recomputed from the masked roles
 #3 retrieval        `LocalFilter` re-applies every filter on the masked profile (drop anyone who fails)
 #5 fuzzy dates      a role whose start interval straddles T is dropped (conservative); ambiguous ends
                     become "ongoing" (hiding a possibly-post-T departure)
 #8 connection       employed-at-T test used for the roster (start <= T and ongoing at T)
 #9 LLM memory       the masked profile has no name, headline, URL or any identifier but `person_id`

Fail-closed: after masking, `assert_clean` re-verifies the invariants. A profile that violates them is
dropped and logged (`leak_guard_violation`) rather than passed on.

Live mode uses the same code with T = today: nothing starts after today, dates are trusted
(there is no later snapshot to be ambiguous against), and only anonymisation + hygiene remain.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from core.config import today as cfg_today
from core.dates import FuzzyDate, months_between, relation_to
from core.events import log_event
from core.geo import geo
from core.models import JobSpec, MaskedProfile, MaskReport, RawProfile, Role, School, seniority_index
from core.text import company_variants, jaccard, normalize, scrub_future_mentions, scrub_mentions, tokenize
from crust.titles import expand_titles, normalize_title, taxonomy


class LeakError(AssertionError):
    """Raised by `assert_clean` when a masked profile still contains post-T information."""


# ------------------------------------------------------------------------------- config / target
@dataclass
class MaskConfig:
    ambiguity: str = "conservative"            # conservative | trust_reported
    include_descriptions: bool = True
    include_profile_skills: bool = True
    include_profile_location: bool = True
    scrub_target_mentions: bool = True
    infer_missing_end: bool = True

    @classmethod
    def from_cfg(cls, cfg: dict) -> "MaskConfig":
        m = cfg["mask"]
        return cls(
            ambiguity=str(m["ambiguity"]), include_descriptions=bool(m["include_descriptions"]),
            include_profile_skills=bool(m["include_profile_skills"]), include_profile_location=bool(m["include_profile_location"]),
            scrub_target_mentions=bool(m["scrub_target_mentions"]), infer_missing_end=bool(m["infer_missing_end_from_next_start"]),
        )

    def strict(self) -> "MaskConfig":
        """Robustness variant (leak #6): remove every undated / editable field."""
        return MaskConfig(self.ambiguity, include_descriptions=False, include_profile_skills=False,
                          include_profile_location=False, scrub_target_mentions=self.scrub_target_mentions,
                          infer_missing_end=self.infer_missing_end)


@dataclass
class TargetRef:
    """The hiring company. Used to scrub free text and to exclude current employees."""
    company_id: int | None = None
    company_name: str | None = None
    company_url: str | None = None

    def variants(self) -> list[str]:
        return company_variants(self.company_name)

    def matches_role(self, r: Role) -> bool:
        if self.company_id is not None and r.company_id is not None:
            return int(self.company_id) == int(r.company_id)
        if self.company_url and r.company_url:
            return self.company_url.rstrip("/").lower() == r.company_url.rstrip("/").lower()
        n = normalize(r.company_name)
        return bool(n) and any(n == v or n.startswith(v + " ") for v in self.variants())


# ------------------------------------------------------------------------------- helpers
def _dedupe(roles: list[Role]) -> list[Role]:
    seen: set[tuple] = set()
    out: list[Role] = []
    for r in roles:
        key = (r.company_id or normalize(r.company_name), normalize(r.title), r.start.label() if r.start else None)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _infer_ends(roles: list[Role], rep: MaskReport) -> list[Role]:
    """A past role with no end date ended (at the latest) when the next role began. Without this a
    forgotten end date would make an old job look ongoing at T."""
    dated = sorted((r for r in roles if r.start is not None), key=lambda r: (r.start.date, r.idx))
    out = {r.idx: r for r in roles}
    for i, r in enumerate(dated):
        if r.end is not None or r.from_current:
            continue
        nxt = next((n for n in dated[i + 1:] if n.start.date > r.start.date), None)
        if nxt is not None:
            out[r.idx] = r.model_copy(update={"end": nxt.start, "end_inferred": True})
            rep.missing_end_inferred += 1
    return [out[r.idx] for r in roles]


def _top_role(kept: list[Role]) -> tuple[Role | None, bool]:
    ongoing = [r for r in kept if r.end is None]
    if ongoing:
        return min(ongoing, key=lambda r: (-r.start.date.toordinal(), r.idx)), True
    if kept:
        return max(kept, key=lambda r: (r.end.date.toordinal() if r.end else 0, r.start.date.toordinal())), False
    return None, False


# ------------------------------------------------------------------------------- the mask
def mask_profile(profile: RawProfile, T: dt.date, *, target: TargetRef | None = None,
                 mcfg: MaskConfig | None = None, snapshot: dt.date | None = None) -> MaskedProfile:
    """Rebuild `profile` as it looked on T. Pure and deterministic. See module docstring."""
    mcfg = mcfg or MaskConfig()
    snapshot = snapshot or dt.date.today()
    trust = mcfg.ambiguity == "trust_reported" or T >= snapshot
    rep = MaskReport(n_roles_raw=len(profile.roles), trust_reported_dates=trust)
    variants = target.variants() if (target and mcfg.scrub_target_mentions) else []

    roles = _dedupe(profile.roles)
    if mcfg.infer_missing_end:
        roles = _infer_ends(roles, rep)

    kept: list[Role] = []
    for r in roles:
        if r.start is None:
            rep.dropped_undated += 1
            continue
        rel = relation_to(r.start, T, trust=trust)
        if rel == "after":
            rep.dropped_post_T += 1
            continue
        if rel == "ambiguous":
            rep.dropped_ambiguous += 1
            continue
        end = r.end
        if end is not None:
            erel = relation_to(end, T, trust=trust)
            if erel == "after":
                end = None
                rep.ended_after_T_to_ongoing += 1
            elif erel == "ambiguous":
                end = None
                rep.ambiguous_end_to_ongoing += 1
            elif end.date < r.start.date:
                rep.dropped_invalid += 1
                continue
        desc = r.description if mcfg.include_descriptions else None
        if desc:
            desc, n_future = scrub_future_mentions(desc, T.year)
            rep.scrubbed_future_sentences += n_future
        title = r.title
        if variants:
            desc, n1 = scrub_mentions(desc, variants) if desc else (desc, 0)
            title, n2 = scrub_mentions(title, variants) if title else (title, 0)
            rep.scrubbed_company_mentions += n1 + n2
        kept.append(r.model_copy(update={"end": end, "description": desc or None, "title": title}))

    kept.sort(key=lambda r: (-r.start.date.toordinal(), r.idx))
    rep.kept = len(kept)
    rep.free_text_dropped = ["headline", "summary", "current_title", "normalized_title", "recently_changed_jobs", "years_at_company"]

    edu: list[School] = []
    for s in profile.education:
        if s.start_year is not None and s.start_year > T.year:
            rep.education_trimmed += 1
            continue
        if s.end_year is not None and s.end_year > T.year:
            s = s.model_copy(update={"end_year": None})
            rep.education_trimmed += 1
        edu.append(s)

    skills = list(profile.skills) if mcfg.include_profile_skills else []
    prof_loc = (profile.location_raw or "") if mcfg.include_profile_location else ""
    top, ongoing = _top_role(kept)
    if skills:
        rep.leaky_fields.append("skills")
    if prof_loc:
        rep.leaky_fields.append("profile_location")
    if any(r.description for r in kept):
        rep.leaky_fields.append("role_descriptions")

    norm = normalize_title(top.title, top.seniority_level) if top else normalize_title(None)
    loc_at_T = (top.location if top and top.location else prof_loc) or ""
    first_start = min((r.start.date for r in kept), default=None)
    yrs = months_between(first_start, T) / 12.0 if first_start else 0.0
    gap = months_between(top.end.date, T) if (top and not ongoing and top.end) else None
    mp = MaskedProfile(
        person_id=profile.person_id, T=T, roles=kept, education=edu, skills=skills, location=prof_loc, location_at_T=loc_at_T,
        title_at_T=(top.title or "") if top else "", employer_at_T=(top.company_name or "") if (top and ongoing) else "",
        employer_id_at_T=top.company_id if (top and ongoing) else None, norm=norm, years_experience_at_T=max(0.0, round(yrs, 2)),
        between_roles_months=round(gap, 2) if gap is not None else None, mask=rep,
    )
    return mp


def assert_clean(mp: MaskedProfile, *, trust: bool = False) -> None:
    """Post-conditions. Raises LeakError if any role still carries post-T information."""
    T = mp.T
    for r in mp.roles:
        if r.start is None:
            raise LeakError(f"{mp.person_id}: undated role survived masking")
        # a kept role's WHOLE start interval must lie on or before T (trust mode: the reported date)
        start_bound = r.start.date if trust else r.start.hi
        if start_bound > T:
            raise LeakError(f"{mp.person_id}: role starts {r.start.label()} (> T={T} or ambiguous)")
        if r.end is not None and r.end.date > T:
            raise LeakError(f"{mp.person_id}: role ends {r.end.date} > T={T}")
    if not mp.roles and (mp.title_at_T or mp.employer_at_T):
        raise LeakError(f"{mp.person_id}: title/employer without roles")


# ------------------------------------------------------------------------------- facade
class TimeGuard:
    """Owns T, the target company and the mask config for one run; every tool result passes through
    `admit()` before any agent sees it. Keeps aggregate statistics for the report."""

    def __init__(self, T: dt.date, *, target: TargetRef | None = None, cfg: dict | None = None,
                 mcfg: MaskConfig | None = None, snapshot: dt.date | None = None, strict: bool = False):
        self.T = T
        self.target = target
        self.snapshot = snapshot or (cfg_today(cfg) if cfg is not None else dt.date.today())
        base = mcfg or (MaskConfig.from_cfg(cfg) if cfg is not None else MaskConfig())
        self.mcfg = base.strict() if strict else base
        self.trust = self.mcfg.ambiguity == "trust_reported" or T >= self.snapshot
        self.totals: dict[str, int] = {}
        self.violations = 0

    def admit(self, raw: list[RawProfile]) -> list[MaskedProfile]:
        out: list[MaskedProfile] = []
        for p in raw:
            try:
                mp = mask_profile(p, self.T, target=self.target, mcfg=self.mcfg, snapshot=self.snapshot)
                assert_clean(mp, trust=self.trust)
            except LeakError as exc:
                self.violations += 1
                log_event("leak_guard_violation", person_id=p.person_id, detail=str(exc)[:160])
                continue
            for k, v in mp.mask.model_dump().items():
                if isinstance(v, int) and not isinstance(v, bool):
                    self.totals[k] = self.totals.get(k, 0) + v
            out.append(mp)
        self.totals["profiles"] = self.totals.get("profiles", 0) + len(out)
        return out

    def employed_at_T(self, mp: MaskedProfile, company_id: int | None, company_name: str | None = None,
                      company_url: str | None = None) -> bool:
        ref = TargetRef(company_id, company_name, company_url)
        return any(ref.matches_role(r) for r in mp.roles if r.end is None)


# ------------------------------------------------------------------------------- local re-filter (leak #3)
class LocalFilter(BaseModel):
    families: list[str] = Field(default_factory=list)
    target_family: str = "unknown"
    target_title_tokens: list[str] = Field(default_factory=list)
    target_seniority: int = 2
    max_seniority_gap: int = 2
    job_location: str = ""
    remote_ok: bool = False
    location_mode: str = "soft"            # soft | hard
    exclude_company_ids: list[int] = Field(default_factory=list)
    exclude_company_names: list[str] = Field(default_factory=list)
    require_title_family: bool = True

    @classmethod
    def from_spec(cls, spec: JobSpec, cfg: dict, target: TargetRef | None = None) -> "LocalFilter":
        tf = normalize_title(spec.norm_title).family
        fams: list[str] = []
        for t in [spec.norm_title, *spec.adjacent_titles]:
            f = normalize_title(t).family
            if f != "unknown" and f not in fams:
                fams.append(f)
        if tf != "unknown":
            for nb in taxonomy().adjacent(tf, 1):
                if nb not in fams:
                    fams.append(nb)
        sc = cfg["sourcer"]["local_filter"]
        return cls(
            families=fams, target_family=tf, target_title_tokens=tokenize(spec.norm_title),
            target_seniority=seniority_index(spec.seniority), max_seniority_gap=int(sc["max_seniority_gap"]),
            job_location=spec.location, remote_ok=spec.remote_ok, location_mode=str(sc["location_mode"]),
            exclude_company_ids=[int(target.company_id)] if (target and target.company_id is not None) else [],
            exclude_company_names=target.variants() if target else [], require_title_family=bool(sc["require_title_family"]),
        )

    def passes(self, mp: MaskedProfile) -> tuple[bool, str]:
        if not mp.roles:
            return False, "no_history"
        for r in mp.roles:
            if r.end is not None:
                continue
            if r.company_id is not None and r.company_id in self.exclude_company_ids:
                return False, "current_employee"
            n = normalize(r.company_name)
            if n and any(n == v or n.startswith(v + " ") for v in self.exclude_company_names):
                return False, "current_employee"
        if self.require_title_family:
            if self.families:
                if mp.norm.family not in self.families:
                    return False, "title_family"
            elif jaccard(tokenize(mp.title_at_T), self.target_title_tokens) <= 0:
                return False, "title_family"
        if abs(mp.norm.seniority - self.target_seniority) > self.max_seniority_gap:
            return False, "seniority"
        if self.location_mode == "hard" and not self.remote_ok and self.job_location:
            if geo().score(mp.location_at_T or mp.location, self.job_location) <= 0:
                return False, "location"
        return True, "ok"

    def apply(self, pool: list[MaskedProfile]) -> tuple[list[MaskedProfile], dict[str, int]]:
        kept: list[MaskedProfile] = []
        reasons: dict[str, int] = {}
        for mp in pool:
            ok, why = self.passes(mp)
            if ok:
                kept.append(mp)
            else:
                reasons[why] = reasons.get(why, 0) + 1
        return kept, reasons
