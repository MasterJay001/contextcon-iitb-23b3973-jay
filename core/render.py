"""Canonical text rendering of a MaskedProfile.

The same text is (a) shown to the LLM, (b) indexed for retrieval and (c) the ground truth that evidence quotes
are verified against - so a quote the model "remembers" but that is not in the record cannot survive (rule 6).
Handles: [R0].. roles (newest first), [E0].. education, [S] skills list (undated).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.models import MaskedProfile
from core.text import normalize, truncate


@dataclass
class Rendered:
    person_id: int
    text: str
    refs: dict[str, str] = field(default_factory=dict)
    plain: str = ""                    # retrieval text (no markup)

    def verify(self, quote: str, ref: str | None = None) -> bool:
        """True iff `quote` occurs (whitespace/case-insensitive) in the referenced part, or anywhere if ref unknown."""
        q = normalize(quote)
        if len(q) < 3:
            return False
        if ref and ref in self.refs:
            return q in normalize(self.refs[ref])
        return q in normalize(" ".join(self.refs.values()))

    def source_path(self, ref: str, mp: MaskedProfile) -> str:
        if ref.startswith("R") and ref[1:].isdigit() and int(ref[1:]) < len(mp.roles):
            return mp.roles[int(ref[1:])].src
        if ref == "S":
            return "skills.professional_network_skills"
        if ref.startswith("E"):
            return "education.schools"
        return ""


def render_masked(mp: MaskedProfile, *, max_desc: int = 260, max_roles: int = 8) -> Rendered:
    refs: dict[str, str] = {}
    lines = [f"CANDIDATE {mp.person_id} (as of {mp.T.isoformat()})"]
    plain: list[str] = []
    for i, r in enumerate(mp.roles[:max_roles]):
        end = r.end.label() if r.end else "ongoing at T"
        start = r.start.label() if r.start else "?"
        meta = "; ".join(x for x in [", ".join(r.company_industries[:2]) if r.company_industries else "", f"size {r.company_headcount_range}" if r.company_headcount_range else ""] if x)
        desc = truncate(r.description, max_desc)
        body = f"{r.title or 'untitled'} | {r.company_name or 'unknown company'} | {start} -> {end}"
        if r.location:
            body += f" | {r.location}"
        if meta:
            body += f" | {meta}"
        refs[f"R{i}"] = " ".join(x for x in [r.title, r.company_name, r.location, meta, r.description] if x)
        lines.append(f"[R{i}] {body}" + (f"\n     {desc}" if desc else ""))
        plain.append(refs[f"R{i}"])
    for j, s in enumerate(mp.education[:3]):
        txt = ", ".join(x for x in [s.degree, s.field_of_study, s.school] if x)
        yrs = f" ({s.start_year or '?'}-{s.end_year or 'in progress'})" if (s.start_year or s.end_year) else ""
        refs[f"E{j}"] = txt
        lines.append(f"[E{j}] {txt}{yrs}")
        plain.append(txt)
    if mp.skills:
        refs["S"] = ", ".join(mp.skills[:30])
        lines.append(f"[S] Skills (undated list as shown on the profile): {refs['S']}")
        plain.append(refs["S"])
    return Rendered(mp.person_id, "\n".join(lines), refs, " . ".join(plain))
