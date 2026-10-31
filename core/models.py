"""Typed contracts shared by every module (the doc's section-8 interfaces, extended).

Design note: `RawProfile` is what the API says *today*. `MaskedProfile` is what was knowable on
the as-of date T. Agents only ever see `MaskedProfile` (rule 2).
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field

from core.dates import FuzzyDate

SENIORITY_LABELS = ["Intern", "Junior", "Mid", "Senior", "Lead", "Principal", "Director", "VP", "C-Level"]
_SENIORITY_ALIASES = {
    "intern": 0, "internship": 0, "trainee": 0, "entry": 1, "entry level": 1, "junior": 1, "associate": 1,
    "mid": 2, "mid-level": 2, "mid level": 2, "intermediate": 2, "regular": 2,
    "senior": 3, "sr": 3, "lead": 4, "staff": 4, "manager": 4, "principal": 5, "senior manager": 5,
    "director": 6, "head": 6, "vp": 7, "vice president": 7, "c-level": 8, "c level": 8, "cxo": 8,
    "founder": 8, "owner": 8, "executive": 8, "partner": 7,
}


def seniority_index(label: str | int | None, default: int = 2) -> int:
    if label is None:
        return default
    if isinstance(label, int):
        return max(0, min(8, label))
    key = str(label).strip().lower()
    if key in _SENIORITY_ALIASES:
        return _SENIORITY_ALIASES[key]
    for name, idx in _SENIORITY_ALIASES.items():
        if name in key:
            return idx
    return default


def seniority_label(idx: int) -> str:
    return SENIORITY_LABELS[max(0, min(8, int(idx)))]


# ------------------------------------------------------------------------------ profile
class Role(BaseModel):
    idx: int = 0                          # index in the raw profile (stable citation handle)
    src: str = ""                         # e.g. experience.employment_details.past[2]
    company_name: str | None = None
    company_id: int | None = None
    company_url: str | None = None
    title: str | None = None
    description: str | None = None
    start: FuzzyDate | None = None
    end: FuzzyDate | None = None          # None = ongoing
    end_inferred: bool = False            # end taken from the next role's start
    seniority_level: str | None = None    # API role-level label (derived from the role's own title)
    function_category: str | None = None
    location: str | None = None
    employment_type: str | None = None
    company_headcount_range: str | None = None
    company_industries: list[str] = Field(default_factory=list)
    company_type: str | None = None
    company_hq_country: str | None = None
    from_current: bool = False

    def start_date(self) -> dt.date | None:
        return self.start.date if self.start else None

    def end_date(self) -> dt.date | None:
        return self.end.date if self.end else None


class School(BaseModel):
    school: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    start_year: int | None = None
    end_year: int | None = None


class RawProfile(BaseModel):
    person_id: int
    name: str | None = None
    headline: str | None = None
    summary: str | None = None
    current_title: str | None = None
    profile_url: str | None = None
    location_raw: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    roles: list[Role] = Field(default_factory=list)
    education: list[School] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    years_of_experience_raw: float | None = None
    recently_changed_jobs: bool | None = None


class NormTitle(BaseModel):
    """Local mirror of Crustdata's `basic_profile.normalized_title`, plus seniority and family."""
    raw: str = ""
    family: str = "unknown"               # taxonomy id, e.g. swe_backend
    matched_title: str = ""
    department: str = ""
    sub_department: str = ""
    seniority: int = 2
    similarity: float = 0.0
    confident: bool = False
    modifiers: list[str] = Field(default_factory=list)


class MaskReport(BaseModel):
    n_roles_raw: int = 0
    kept: int = 0
    dropped_post_T: int = 0
    dropped_ambiguous: int = 0
    dropped_undated: int = 0
    dropped_invalid: int = 0
    ended_after_T_to_ongoing: int = 0
    ambiguous_end_to_ongoing: int = 0
    missing_end_inferred: int = 0
    scrubbed_company_mentions: int = 0
    scrubbed_future_sentences: int = 0
    education_trimmed: int = 0
    free_text_dropped: list[str] = Field(default_factory=list)
    leaky_fields: list[str] = Field(default_factory=list)
    trust_reported_dates: bool = False


class MaskedProfile(BaseModel):
    person_id: int
    T: dt.date
    roles: list[Role] = Field(default_factory=list)      # newest first, masked to T
    education: list[School] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)       # undated -> flagged leaky
    location: str = ""                                    # undated -> flagged leaky
    location_at_T: str = ""                               # from the dated role when available
    title_at_T: str = ""
    employer_at_T: str = ""
    employer_id_at_T: int | None = None
    norm: NormTitle = Field(default_factory=NormTitle)
    years_experience_at_T: float = 0.0
    between_roles_months: float | None = None             # gap at T if no ongoing role
    mask: MaskReport = Field(default_factory=MaskReport)

    def ongoing_roles(self) -> list[Role]:
        return [r for r in self.roles if r.end is None]


# ------------------------------------------------------------------------------ brief / case
class JobSpec(BaseModel):
    norm_title: str
    seniority: str = "Mid"
    skills: list[str] = Field(default_factory=list)
    location: str = ""
    remote_ok: bool = False
    adjacent_titles: list[str] = Field(default_factory=list)
    # extensions
    company_name: str | None = None
    industries: list[str] = Field(default_factory=list)
    domain_keywords: list[str] = Field(default_factory=list)      # e.g. payments, fintech
    stage: str | None = None                                       # startup | scaleup | enterprise
    min_years: float | None = None
    max_years: float | None = None
    expanded_locations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    clarifying_question: str | None = None
    injection_detected: bool = False
    interpreter: str = "heuristic"                                 # heuristic | llm


class Case(BaseModel):
    case_id: str
    company_id: int | None = None
    company_url: str | None = None
    company_name: str | None = None
    T: dt.date
    brief: str
    target_norm_title: str = ""
    target_seniority: str = "Mid"
    location: str | None = None
    route: str = "A"
    split: str = "train"
    universe_size: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Weights(BaseModel):
    w_F: float = 1.0
    w_M: float = 0.5
    w_C: float = 0.5
    fit_inner: dict[str, float] = Field(default_factory=dict)
    move_inner: dict[str, float] = Field(default_factory=dict)
    conn_inner: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Weights":
        return cls(
            w_F=float(cfg["weights"]["w_F"]),
            w_M=float(cfg["weights"]["w_M"]),
            w_C=float(cfg["weights"]["w_C"]),
            fit_inner=dict(cfg["fit"]["inner_weights"]),
            move_inner=dict(cfg["movability"]["inner_weights"]),
            conn_inner=dict(cfg["connection"]["inner_weights"]),
        )


# ------------------------------------------------------------------------------ outputs
class Source(BaseModel):
    person_id: int
    path: str                              # Crustdata field path, e.g. experience.employment_details.past[2].title
    value: str = ""


class Claim(BaseModel):
    text: str
    source: Source
    quote: str | None = None
    verified: bool = False


class Scored(BaseModel):
    person_id: int
    F: float = 0.0
    M: float = 0.0
    C: float = 0.0
    score: float = 0.0
    evidence: list[str] = Field(default_factory=list)      # quotes from MaskedProfile fields only
    features: dict[str, float] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    demoted: bool = False


class Outreach(BaseModel):
    subject: str
    body: str                                # may contain {{CANDIDATE_FIRST_NAME}} / {{COLLEAGUE_NAME}}
    facts_used: list[str] = Field(default_factory=list)
    requires_approval: bool = True
    status: str = "queued"                   # queued | approved | sent | rejected


class Card(BaseModel):
    rank: int
    person_id: int
    name: str | None = None
    current_role: str = ""
    employer: str = ""
    profile_url: str | None = None
    why_fit: list[Claim] = Field(default_factory=list)
    why_now: list[Claim] = Field(default_factory=list)
    path_in: list[Claim] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    outreach: Outreach | None = None


class CaseResult(BaseModel):
    case_id: str
    mode: str                                 # backtest | live
    T: dt.date
    jobspec: JobSpec | None = None
    ranked: list[Scored] = Field(default_factory=list)
    pool_ids: list[int] = Field(default_factory=list)
    n_retrieved: int = 0
    n_masked: int = 0
    credits_spent: float = 0.0
    seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    backends: dict[str, str] = Field(default_factory=dict)
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    mask_totals: dict[str, int] = Field(default_factory=dict)
    cards: list[Card] = Field(default_factory=list)
