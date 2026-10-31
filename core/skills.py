"""Skill canonicalisation and matching (knowledge/skills_synonyms.json).

`find_skills(text)` finds canonical skills mentioned in free text (role descriptions, titles).
`canon_list(items)` canonicalises an explicit skills list. Ambiguous short words ("go") only match when
they come from an explicit skills list or as the unambiguous form ("golang") in prose.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from core.config import ROOT
from core.text import normalize

_AMBIGUOUS_IN_PROSE = {"go", "swift", "rust", "spark", "express", "node", "react", "ci/cd"}


@lru_cache(maxsize=1)
def _tables():
    with open(ROOT / "knowledge" / "skills_synonyms.json", "r", encoding="utf-8-sig") as fh:
        raw = json.load(fh)
    phrase_to_canon: dict[str, str] = {}
    for canon, syns in raw.items():
        c = normalize(canon)
        phrase_to_canon[c] = c
        for s in syns:
            phrase_to_canon.setdefault(normalize(s), c)
    prose = {p: c for p, c in phrase_to_canon.items() if p not in _AMBIGUOUS_IN_PROSE or p == c and c in {"rust", "spark", "swift", "react"}}
    # "go" is only unambiguous as golang in prose
    prose.pop("go", None)
    pat = re.compile(r"(?<![a-z0-9+#.])(" + "|".join(re.escape(p) for p in sorted(prose, key=len, reverse=True)) + r")(?![a-z0-9+#])")
    return phrase_to_canon, prose, pat


def canonical(term: str | None) -> str:
    t = normalize(term)
    return _tables()[0].get(t, t)


def canon_list(items: list[str] | None) -> set[str]:
    return {canonical(x) for x in (items or []) if x}


def find_skills(text: str | None) -> set[str]:
    if not text:
        return set()
    _, prose, pat = _tables()
    return {prose[m.group(1)] for m in pat.finditer(normalize(text))}


def overlap(brief_skills: list[str], list_skills: list[str], text_blobs: list[str]) -> tuple[float, list[str], list[str]]:
    """Share of brief skills evidenced, and where: dated text (roles) counts fully, the undated list counts 0.7.

    Returns (score in [0,1], matched_in_roles, matched_only_in_skill_list).
    """
    want = canon_list(brief_skills)
    if not want:
        return 0.0, [], []
    in_roles: set[str] = set()
    for blob in text_blobs:
        in_roles |= find_skills(blob)
    in_list = canon_list(list_skills)
    full = want & in_roles
    partial = (want & in_list) - full
    score = (len(full) + 0.7 * len(partial)) / len(want)
    return min(1.0, score), sorted(full), sorted(partial)
