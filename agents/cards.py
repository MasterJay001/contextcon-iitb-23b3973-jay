"""Candidate cards: every claim carries its source (person id + Crustdata field path) and a verified flag (rule 6).

Why they fit      verified quotes from dated roles (the exact words, the role's field path)
Why now           movability evidence derived from dated roles (tenure vs the fitted hazard, personal norm, promotions)
Path in           ex-colleagues inside the hiring company at T, with employer and overlap dates
Unverified        what the agent could NOT confirm, stated plainly
"""
from __future__ import annotations

from agents.artifacts import CaseArtifacts
from agents.critic import CriticResult
from core.models import Card, Claim, MaskedProfile, Scored, Source
from core.render import render_masked
from core.text import normalize


def _field_for(quote: str, role) -> str:
    q = normalize(quote)
    if role.title and q in normalize(role.title):
        return "title"
    if role.description and q in normalize(role.description):
        return "description"
    if role.company_name and q in normalize(role.company_name):
        return "name"
    return "title"


def _role_line(r) -> str:
    end = r.end.label() if r.end else "present"
    return f"{r.title or 'role'} at {r.company_name or 'company'} ({r.start.label() if r.start else '?'} to {end})"


def build_card(rank: int, s: Scored, art: CaseArtifacts, critic: CriticResult | None, *, colleague_names: dict[int, str] | None = None) -> Card:
    mp: MaskedProfile = next(m for m in art.pool if m.person_id == s.person_id)
    raw = art.raw.get(s.person_id)
    rendered = render_masked(mp)
    det = art.fit_details.get(s.person_id)
    top = next((r for r in mp.roles if r.end is None), mp.roles[0] if mp.roles else None)

    why_fit: list[Claim] = []
    seen: set[str] = set()
    for ref, quote in (det.evidence if det else []):
        if ref.startswith("R") and ref[1:].isdigit() and int(ref[1:]) < len(mp.roles):
            role = mp.roles[int(ref[1:])]
            key = f"{ref}:{normalize(quote)}"
            if key in seen:
                continue
            seen.add(key)
            ok = rendered.verify(quote, ref)
            why_fit.append(Claim(text=f"{_role_line(role)}: \"{quote}\"", quote=quote, verified=ok,
                                 source=Source(person_id=s.person_id, path=f"{role.src}.{_field_for(quote, role)}", value=quote)))
        elif ref == "S":
            ok = rendered.verify(quote, "S")
            why_fit.append(Claim(text=f"Lists the skill \"{quote}\" (undated list: weaker evidence)", quote=quote, verified=ok,
                                 source=Source(person_id=s.person_id, path="skills.professional_network_skills", value=quote)))
        if len(why_fit) >= 3:
            break
    if not why_fit and top is not None and top.title:
        why_fit.append(Claim(text=_role_line(top), quote=top.title, verified=True,
                             source=Source(person_id=s.person_id, path=f"{top.src}.title", value=top.title)))

    why_now: list[Claim] = []
    md = art.move_diag.get(s.person_id)
    cur = top if (top is not None and top.end is None) else None
    sv = art.survival or {}
    if md is not None and cur is not None and md.months_at_employer is not None:
        txt = f"{md.months_at_employer:.0f} months at {cur.company_name or 'current employer'}"
        if sv.get("median_tenure_at_exit_months"):
            txt += f" (people in this pool typically leave after ~{sv['median_tenure_at_exit_months']:.0f} months)"
        if md.p_move is not None:
            txt += f"; model estimate {md.p_move:.0%} chance of moving within 6 months"
        why_now.append(Claim(text=txt, verified=True, source=Source(person_id=s.person_id, path=f"{cur.src}.start_date", value=cur.start.label() if cur.start else "")))
        if md.frailty is not None and md.n_past_spells >= 2 and md.frailty > 1.2:
            why_now.append(Claim(text=f"Habitual mover: {md.n_past_spells} earlier employers with short stays (personal factor x{md.frailty:.1f})", verified=True,
                                 source=Source(person_id=s.person_id, path="experience.employment_details.past", value=str(md.n_past_spells))))
    elif mp.between_roles_months is not None:
        why_now.append(Claim(text=f"Between roles for ~{mp.between_roles_months:.0f} months at the as-of date", verified=True,
                             source=Source(person_id=s.person_id, path=f"{mp.roles[0].src}.end_date" if mp.roles else "experience", value="")))
    mf = art.move_feats.get(s.person_id, {})
    if mf.get("headcount") is not None and mf["headcount"] > 0.6 and cur is not None:
        why_now.append(Claim(text=f"{cur.company_name or 'Employer'} headcount is shrinking (Company Enrichment growth window)", verified=True,
                             source=Source(person_id=s.person_id, path=f"{cur.src}.crustdata_company_id", value=str(cur.company_id or ""))))

    path_in: list[Claim] = []
    cd = art.conn_diag.get(s.person_id)
    if cd is not None:
        for p in cd.paths:
            who = (colleague_names or {}).get(p.colleague_id) or f"person {p.colleague_id}"
            role = mp.roles[p.cand_role_ref] if p.cand_role_ref < len(mp.roles) else None
            path_in.append(Claim(text=f"{who} (now at the hiring company) overlapped with them at {p.employer}, {p.start} to {p.end} ({p.overlap_months:.0f} months)",
                                 verified=True, source=Source(person_id=s.person_id, path=f"{role.src}.name" if role else "experience", value=f"colleague_person_id={p.colleague_id}")))

    unverified: list[str] = []
    if det:
        unverified += det.gaps[:3]
        unverified += [f for f in det.flags if "unverified" in f or "capped" in f or "heuristic" in f]
    if critic:
        unverified += critic.unverified.get(s.person_id, [])
        if s.person_id in critic.demoted:
            unverified.append(f"Critic: {critic.demoted[s.person_id]}")
    if not art.conn_available:
        unverified.append("no connection signal (hiring company unresolved or no roster)")
    unverified = list(dict.fromkeys(u for u in unverified if u))

    return Card(
        rank=rank, person_id=s.person_id, name=raw.name if raw else None,
        current_role=(top.title or "") if top else "", employer=(top.company_name or "") if top else "",
        profile_url=raw.profile_url if raw else None, why_fit=why_fit, why_now=why_now, path_in=path_in, unverified=unverified,
        scores={"F": round(s.F, 3), "M": round(s.M, 3), "C": round(s.C, 3), "score": round(s.score, 3)},
    )
