"""Critic: the verifier. Two layers, cheapest first.

 1. Deterministic (always runs, no LLM): re-verifies every evidence quote against the masked record (rule 6), checks
    hard contradictions with the JobSpec (title family, seniority gap >= 3, experience far below the stated minimum),
    flags thin/invalid histories and a one-dimensional list (3+ of the top 5 from one employer).
 2. LLM (when available): reads the top-N renderings + evidence and flags semantic contradictions the rules cannot
    see (e.g. an "engineer" whose roles are all sales). It must cite candidate ids; unknown ids are ignored.

Demotion is a multiplicative score cut (critic.demote_penalty), never deletion, and every demotion carries a reason that
is shown on the card ("Unverified" / "Critic" lines). The Critic runs only on the top-N: it is a verifier, not a ranker.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from agents.artifacts import CaseArtifacts
from core.models import JobSpec, Scored, seniority_index
from core.render import render_masked
from llm.client import LLM

SYSTEM = """You are the Critic of a recruiting agent: a strict verifier. You receive a job spec and the top candidates (anonymised,
as of a past date) with the evidence the scorer cited. Check ONLY what is on the page:
- flag a candidate when the history contradicts the spec or the scorer's evidence (wrong profession, evidence that is not about
  the person's work, seniority far from the target, a claim that rests only on the undated skills list);
- flag list-level problems (all candidates from one employer, one narrow profile);
- do not invent facts; do not demote for missing information alone.
Return only ids from the list. Keep reasons under 20 words."""


class DemoteItem(BaseModel):
    person_id: int
    reason: str


class CriticOut(BaseModel):
    demote: list[DemoteItem]
    list_issues: list[str] = Field(default_factory=list)


@dataclass
class CriticResult:
    demoted: dict[int, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    unverified: dict[int, list[str]] = field(default_factory=dict)
    source: str = "deterministic"


def deterministic_checks(art: CaseArtifacts, top: list[Scored], spec: JobSpec) -> CriticResult:
    res = CriticResult()
    by_id = {m.person_id: m for m in art.pool}
    tsen = seniority_index(spec.seniority)
    employers: Counter = Counter()
    for s in top[:5]:
        cur = next((r for r in by_id[s.person_id].roles if r.end is None), None)
        if cur is not None:
            employers[cur.company_id or (cur.company_name or "")] += 1
    for s in top:
        mp = by_id[s.person_id]
        rendered = render_masked(mp)
        det = art.fit_details.get(s.person_id)
        bad = []
        if det is not None:
            bad = [q for ref, q in det.evidence if not rendered.verify(q, ref)]
            if det.skills_only_listed:
                res.unverified.setdefault(s.person_id, []).append(f"{', '.join(det.skills_only_listed)} appear only in the undated skills list, not in any role")
        if bad:
            res.unverified.setdefault(s.person_id, []).append(f"{len(bad)} cited quote(s) not found in the record")
            res.demoted[s.person_id] = "cited evidence not found in the record"
        if abs(mp.norm.seniority - tsen) >= 3:
            res.demoted.setdefault(s.person_id, f"seniority far from target ({mp.norm.seniority} vs {tsen})")
        if spec.min_years and mp.years_experience_at_T + 3.0 < spec.min_years:
            res.demoted.setdefault(s.person_id, f"{mp.years_experience_at_T:.0f} years of experience at T vs {spec.min_years:.0f}+ required")
        if mp.mask.dropped_invalid:
            res.unverified.setdefault(s.person_id, []).append("profile has contradictory dates; treated with caution")
        if len(mp.roles) <= 1:
            res.unverified.setdefault(s.person_id, []).append("thin history (one dated role)")
    if employers and employers.most_common(1)[0][1] >= 3:
        res.issues.append("3+ of the top 5 currently work at the same employer: list is one-dimensional")
    return res


class Critic:
    def __init__(self, cfg: dict, llm: LLM):
        self.cfg, self.llm = cfg, llm
        self.top_n = int(cfg["critic"]["top_n"])

    def review(self, art: CaseArtifacts, ranked: list[Scored]) -> CriticResult:
        top = ranked[: self.top_n]
        res = deterministic_checks(art, top, art.spec)
        if not self.llm.available or not top:
            return res
        by_id = {m.person_id: m for m in art.pool}
        blocks = []
        for s in top:
            r = render_masked(by_id[s.person_id])
            det = art.fit_details.get(s.person_id)
            ev = "; ".join(f"[{ref}] \"{q}\"" for ref, q in (det.evidence if det else [])) or "none"
            blocks.append(f"{r.text}\n  SCORER EVIDENCE: {ev}\n  SCORER GAPS: {'; '.join(det.gaps) if det else ''}")
        spec_json = art.spec.model_dump_json(include={"norm_title", "seniority", "skills", "location", "remote_ok", "domain_keywords", "stage", "min_years"})
        user = f"JOB SPEC:\n{spec_json}\n\nTOP {len(top)} CANDIDATES:\n\n" + "\n\n".join(blocks)
        out = self.llm.structured("critic", SYSTEM, user, CriticOut, mock=lambda: CriticOut(demote=[]))
        if out.ok:
            valid = {s.person_id for s in top}
            for d in out.parsed.demote:
                if d.person_id in valid:
                    res.demoted.setdefault(d.person_id, d.reason[:140])
            res.issues += [i for i in out.parsed.list_issues if i][:3]
            res.source = "deterministic+llm"
        else:
            res.issues.append("LLM critic unavailable: deterministic checks only")
        return res
