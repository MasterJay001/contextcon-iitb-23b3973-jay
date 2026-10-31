"""Failure Analyst (Reflexion-style): reads TRAIN misses and proposes at most two general edits per round.

Theory: Reflexion (Shinn et al.) turns failures into verbal lessons stored as episodic memory that conditions the next attempt.
Here the "memory" is `lessons.md` (rubric sentences retrieved into the Fit prompt) and a coarse weight change on w_M / w_C. A Goodhart guard
(eval/self_improve.py) accepts an edit only if the TRAIN metric genuinely improves; the held-out split is never consulted.

This module holds only the schema, prompt, heuristic fallback and the lint. It never reads the answer key: the loop (which lives in
eval/) hands it a pre-built `MissReport`, and it is called only for TRAIN cases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from core.text import contains_phrase
from llm.client import LLM

SYSTEM = """You are the Failure Analyst of a recruiting agent. You are shown cases where the REAL hire was ranked low, together with the people
ranked above them and the scores each family produced (F = fit, M = movability, C = connection, all in [0,1]).
Compare the hire with those ranked above. Propose AT MOST TWO edits that would have helped and would help on other roles too:
 - kind "rubric_sentence": ONE general sentence to add to the fit rubric (e.g. about adjacent roles, seniority, tenure or evidence);
 - kind "weight": change w_M or w_C to a value in {0, 0.25, 0.5, 1.0}.
Rules: edits must be general. NEVER name a specific company, person, school or city; never mention ids. Each edit needs a one-line reason."""

ALLOWED_WEIGHT_VALUES = (0.0, 0.25, 0.5, 1.0)


class Edit(BaseModel):
    kind: str = Field(description="rubric_sentence or weight")
    text: str = Field(description="the sentence for rubric_sentence; empty for weight")
    param: str = Field(description="w_M or w_C for weight edits; empty otherwise")
    value: float = Field(description="new value for weight edits; 0 otherwise")
    reason: str


class AnalystOut(BaseModel):
    edits: list[Edit]


@dataclass
class MissReport:
    case_id: str
    spec_summary: str
    hire_rank: int
    pool_size: int
    hire_text: str
    hire_scores: dict[str, float]
    above: list[tuple[str, dict[str, float]]] = field(default_factory=list)     # (rendering, scores)
    hire_percentiles: dict[str, float] = field(default_factory=dict)             # feature -> percentile of the hire within the pool


def build_prompt(misses: list[MissReport], current: dict) -> str:
    parts = [f"CURRENT STATE: w_M={current['w_M']}, w_C={current['w_C']}; lessons so far: {len(current.get('lessons', []))}"]
    for m in misses:
        parts.append(f"\n=== CASE (hire ranked {m.hire_rank} of {m.pool_size}) ===\nROLE: {m.spec_summary}\n"
                     f"HIRE as of the posting date (scores {m.hire_scores}):\n{m.hire_text}\nHIRE feature percentiles in the pool: {m.hire_percentiles}")
        for txt, sc in m.above[:2]:
            parts.append(f"RANKED ABOVE (scores {sc}):\n{txt}")
    return "\n".join(parts)


def heuristic_edits(misses: list[MissReport], current: dict) -> AnalystOut:
    """Deterministic diagnosis when no LLM is available: if hires systematically sit high on a signal family the system is
    under-weighting (or low on one it over-weights), propose the coarse weight change; otherwise a general rubric sentence."""
    edits: list[Edit] = []
    if not misses:
        return AnalystOut(edits=[])

    def mean_pct(prefix: str) -> float:
        vals = [v for m in misses for k, v in m.hire_percentiles.items() if k.startswith(prefix)]
        return sum(vals) / len(vals) if vals else 0.5

    c_pct, m_pct = mean_pct("conn."), mean_pct("move.")
    steps = list(ALLOWED_WEIGHT_VALUES)

    def bump(w: float, up: bool) -> float:
        i = min(range(len(steps)), key=lambda j: abs(steps[j] - w))
        return steps[min(len(steps) - 1, i + 1)] if up else steps[max(0, i - 1)]

    if c_pct >= 0.65 and current["w_C"] < 1.0:
        edits.append(Edit(kind="weight", text="", param="w_C", value=bump(current["w_C"], True),
                          reason=f"missed hires sit high on connection signals (mean pool percentile {c_pct:.2f})"))
    if m_pct >= 0.62 and current["w_M"] < 1.0 and len(edits) < 2:
        edits.append(Edit(kind="weight", text="", param="w_M", value=bump(current["w_M"], True),
                          reason=f"missed hires sit high on movability signals (mean pool percentile {m_pct:.2f})"))
    if len(edits) < 2:
        fit_pct = mean_pct("fit.")
        if fit_pct < 0.45:
            edits.append(Edit(kind="rubric_sentence", param="", value=0.0,
                              text="Candidates one role family away from the target (for example backend, full stack and platform engineering) are plausible hires; do not score them below 3 on title alone.",
                              reason=f"missed hires look weak on literal title/fit (mean pool percentile {fit_pct:.2f}) but were hired"))
    return AnalystOut(edits=edits[:2])


def propose(llm: LLM, misses: list[MissReport], current: dict) -> tuple[AnalystOut, str]:
    res = llm.structured("analyst", SYSTEM, build_prompt(misses, current), AnalystOut, mock=lambda: heuristic_edits(misses, current))
    if res.ok:
        return res.parsed, res.source
    return heuristic_edits(misses, current), "heuristic(fallback)"


# ------------------------------------------------------------------------------- lint
@dataclass
class LintResult:
    ok: bool
    reason: str = ""


def lint_edit(e: Edit, entities: set[str]) -> LintResult:
    """Reject any edit that names a specific company, person or school (or looks like an id). Rules must be general."""
    if e.kind == "weight":
        if e.param not in {"w_M", "w_C"}:
            return LintResult(False, f"unknown weight parameter '{e.param}'")
        if round(e.value, 2) not in ALLOWED_WEIGHT_VALUES:
            return LintResult(False, f"weight value {e.value} not in {ALLOWED_WEIGHT_VALUES}")
        return LintResult(True)
    if e.kind != "rubric_sentence":
        return LintResult(False, f"unknown edit kind '{e.kind}'")
    text = (e.text or "").strip()
    if not 20 <= len(text) <= 260:
        return LintResult(False, "rubric sentence must be 20-260 characters")
    if re.search(r"\d{5,}", text) or re.search(r"https?://|@", text):
        return LintResult(False, "contains an id, URL or contact-like token")
    for ent in entities:
        if len(ent) >= 4 and contains_phrase(text, ent):
            return LintResult(False, f"names a specific entity: '{ent}'")
    return LintResult(True)
