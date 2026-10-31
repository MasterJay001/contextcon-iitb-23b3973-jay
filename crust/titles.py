"""Local job-title normalisation.

Crustdata exposes title normalisation only *inline* (`basic_profile.normalized_title`, which
describes the profile's CURRENT title - i.e. the post-T job for a hire, a leak) and has no
standalone endpoint. The masking rule says to "recompute the current title and seniority from
the top role of the masked history", so we normalise locally from `knowledge/title_taxonomy.json`
(the same file is indexed for RAG so the Interpreter can look titles up too).

Output mirrors Crustdata's schema {department, sub_department, matched_title, similarity,
confident} and adds `family` (our id), `seniority` (0-8 ladder) and `modifiers`.

Algorithm: clean -> expand abbreviations -> strip seniority tokens -> best alias by token-F1
(with a contiguous-containment bonus) -> keyword fallback -> seniority from modifiers/levels,
interpreting a bare "manager" per family (Product Manager is an IC title, Engineering Manager
is not).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from core.config import ROOT
from core.models import NormTitle, seniority_index
from core.text import jaccard, normalize

_TAXO_PATH = ROOT / "knowledge" / "title_taxonomy.json"

_PAREN = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_CUT = re.compile(r"\s(?:at|@|\|)\s.*$")
_SEP = re.compile(r"[,/&:;]+|\s-\s|\s–\s|\s—\s")
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}
_ROLE_NOUNS = (
    "engineer|developer|analyst|manager|scientist|designer|specialist|consultant|executive|associate|"
    "architect|lead|officer|representative|recruiter|marketer|accountant|sde|swe|programmer|writer"
)
_NUM_LEVEL = re.compile(rf"\b(?:{_ROLE_NOUNS})\s*[-–]?\s*(iv|iii|ii|i|v|[1-5])\b")
_LEVEL_WORD = re.compile(r"\b(?:level|lvl|l)\s*[-–]?\s*([1-5])\b")

_SENIORITY_STRIP = {
    "senior", "sr", "junior", "jr", "principal", "staff", "associate", "intern", "internship", "trainee",
    "graduate", "entry", "fresher", "apprentice", "founding", "i", "ii", "iii", "iv", "v", "1", "2", "3",
    "4", "5", "level", "chief", "distinguished", "fellow", "group", "assistant", "lead",
}

_RX_C = re.compile(r"\b(chief|ceo|cto|cfo|coo|cmo|cpo|cio|ciso|cxo|founder|co founder|cofounder|owner|managing director|managing partner|chairman|president)\b")
_RX_VP = re.compile(r"\b(vp|svp|evp|vice president|vice-president)\b")
_RX_DIR = re.compile(r"\b(director|head of|head|avp|country head|business head)\b")
_RX_PRINCIPAL = re.compile(r"\b(principal|distinguished|fellow|senior manager|sr manager|group manager|senior staff|associate director|group product manager|senior director)\b")
_RX_LEAD = re.compile(r"\b(lead|staff|team lead|tech lead|supervisor)\b")
_RX_SENIOR = re.compile(r"\b(senior|sr)\b")
_RX_JUNIOR = re.compile(r"\b(junior|jr|graduate|entry|fresher|apm)\b")
_RX_ASSOCIATE = re.compile(r"\bassociate\b(?!\s+(?:director|vice|principal|general counsel))")
_RX_INTERN = re.compile(r"\b(intern|internship|trainee|apprentice)\b")


@dataclass(frozen=True)
class Family:
    id: str
    canonical: str
    department: str
    sub_department: str
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]
    adjacent: tuple[str, ...]
    manager_is_lead: bool


def _alias_norm(x: str) -> str:
    """Aliases are compared token-wise, so hyphens/slashes/brackets are just separators."""
    return re.sub(r"\s+", " ", re.sub(r"[()\[\]/-]", " ", normalize(x))).strip()


class Taxonomy:
    def __init__(self, raw: dict):
        self.departments: list[str] = list(raw["departments"])
        self.abbrev: dict[str, str] = dict(raw.get("abbreviations", {}))
        self.families: dict[str, Family] = {}
        for f in raw["families"]:
            self.families[f["id"]] = Family(
                id=f["id"], canonical=f["canonical"], department=f["department"], sub_department=f["sub_department"],
                aliases=tuple(_alias_norm(a) for a in f["aliases"]),
                keywords=tuple(_alias_norm(k) for k in f.get("keywords", [])),
                adjacent=tuple(f.get("adjacent", [])), manager_is_lead=bool(f.get("manager_is_lead", True)),
            )
        # (family_id, alias_text, alias_token_tuple, alias_token_set)
        self.alias_rows: list[tuple[str, str, tuple[str, ...], frozenset[str]]] = []
        self.alias_exact: dict[str, str] = {}
        for fid in sorted(self.families):
            fam = self.families[fid]
            for a in sorted({*fam.aliases, _alias_norm(fam.canonical)}):
                toks = tuple(a.split())
                self.alias_rows.append((fid, a, toks, frozenset(toks)))
                self.alias_exact.setdefault(a, fid)

    def titles_for(self, family_id: str, limit: int = 6) -> list[str]:
        fam = self.families.get(family_id)
        if not fam:
            return []
        out = [_alias_norm(fam.canonical)]
        for a in fam.aliases:
            if a not in out:
                out.append(a)
            if len(out) >= limit:
                break
        return out

    def adjacent(self, family_id: str, depth: int = 1) -> list[str]:
        seen, frontier = {family_id}, [family_id]
        for _ in range(depth):
            nxt = []
            for fid in frontier:
                for nb in self.families.get(fid, Family("", "", "", "", (), (), (), True)).adjacent:
                    if nb in self.families and nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
        seen.discard(family_id)
        return sorted(seen)


@lru_cache(maxsize=1)
def taxonomy() -> Taxonomy:
    with open(_TAXO_PATH, "r", encoding="utf-8") as fh:
        return Taxonomy(json.load(fh))


# ------------------------------------------------------------------------------- cleaning
_NOISE = {"contract", "remote", "hybrid", "onsite", "freelance", "temporary", "fulltime", "parttime"}


def _clean(raw: str | None) -> list[str]:
    s = normalize(raw)
    if not s:
        return []
    s = _CUT.sub("", s)                      # "Engineer at Acme" -> "Engineer"
    s = re.sub(r"[()\[\]]", " ", s)          # keep parenthetical content: "Rust Engineer (Embedded)"
    s = _SEP.sub(" ", s)
    s = re.sub(r"[^a-z0-9+#. ]", " ", s)     # hyphens become spaces: "sde-3", "back-end"
    toks = []
    abbrev = taxonomy().abbrev
    for t in s.split():
        t = t.strip(".")
        if not t or t in _NOISE:
            continue
        toks.extend(abbrev.get(t, t).split())
    return toks


def _numeric_level(text: str) -> int | None:
    m = _NUM_LEVEL.search(text)
    if m:
        tok = m.group(1)
        return _ROMAN.get(tok, int(tok) if tok.isdigit() else None)
    m = _LEVEL_WORD.search(text)
    if m:
        return int(m.group(1))
    m = re.search(r"\bsde\s*[-–]?\s*([1-5])\b", text)
    if m:
        return int(m.group(1))
    return None


def _seniority(text: str, fam: Family | None) -> tuple[int, list[str], bool]:
    mods: list[str] = []
    if "founding" in text.split():
        mods.append("founding")
    manager_is_lead = fam.manager_is_lead if fam else True
    level, explicit = 2, False
    if _RX_C.search(text):
        level, explicit = 8, True
    elif _RX_VP.search(text):
        level, explicit = 7, True
    elif _RX_PRINCIPAL.search(text):
        level, explicit = 5, True
    elif _RX_DIR.search(text):
        level, explicit = 6, True
    elif _RX_LEAD.search(text):
        level, explicit = 4, True
    elif re.search(r"\bmanager\b", text) and manager_is_lead and not _RX_SENIOR.search(text):
        level, explicit = 4, True
    elif _RX_SENIOR.search(text):
        level, explicit = 3, True
    elif _RX_INTERN.search(text):
        level, explicit = 0, True
    elif _RX_JUNIOR.search(text) or _RX_ASSOCIATE.search(text):
        level, explicit = 1, True
    if not explicit:
        n = _numeric_level(text)
        if n is not None:
            level, explicit = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}[n], True
    if "founding" in mods:
        level = max(level, 3)
    if re.search(r"\bsenior manager\b|\bsr manager\b", text):
        level = max(level, 5)
    return level, mods, explicit


def _match_family(core: list[str]) -> tuple[Family | None, float]:
    tax = taxonomy()
    if not core:
        return None, 0.0
    text = " ".join(core)
    if text in tax.alias_exact:
        return tax.families[tax.alias_exact[text]], 1.0
    cset = set(core)
    best: tuple[float, int, str, str] | None = None  # (score, alias_len, -, family id)
    padded = f" {text} "
    for fid, alias, toks, aset in tax.alias_rows:
        inter = len(aset & cset)
        if not inter:
            continue
        p, r = inter / len(cset), inter / len(aset)
        f1 = 2 * p * r / (p + r)
        if f" {alias} " in padded:
            f1 = max(f1, 0.85 + 0.15 * len(toks) / max(len(core), 1))
        key = (f1, len(toks), "", fid)
        if best is None or (key[0], key[1]) > (best[0], best[1]) or (
            (key[0], key[1]) == (best[0], best[1]) and fid < best[3]
        ):
            best = key
    if best and best[0] >= 0.5:
        return tax.families[best[3]], round(min(best[0], 1.0), 4)
    # keyword fallback (weaker)
    kw_best: tuple[int, str] | None = None
    for fid in sorted(tax.families):
        fam = tax.families[fid]
        hits = sum(len(k.split()) for k in fam.keywords if f" {k} " in padded)
        if hits and (kw_best is None or hits > kw_best[0]):
            kw_best = (hits, fid)
    if kw_best:
        return tax.families[kw_best[1]], 0.45
    if best:
        return tax.families[best[3]], round(best[0] * 0.8, 4)
    return None, 0.0


@lru_cache(maxsize=200_000)
def normalize_title(raw: str | None, seniority_hint: str | None = None) -> NormTitle:
    """Normalise a raw title. `seniority_hint` is the API's role-level seniority label, used only
    when the title itself carries no seniority signal."""
    toks = _clean(raw)
    if not toks:
        return NormTitle(raw=raw or "", family="unknown", department="Other", sub_department="Other", seniority=2)
    text = " ".join(toks)
    core = [t for t in toks if t not in _SENIORITY_STRIP]
    fam, sim = _match_family(core)
    if fam is None and core != toks:
        fam, sim = _match_family(toks)
    level, mods, explicit = _seniority(text, fam)
    if not explicit and seniority_hint:
        level = seniority_index(seniority_hint, default=level)
        mods.append("api_seniority_hint")
    if fam is None:
        return NormTitle(raw=raw or "", family="unknown", department="Other", sub_department="Other",
                         seniority=level, similarity=0.0, confident=False, modifiers=mods)
    return NormTitle(
        raw=raw or "", family=fam.id, matched_title=fam.canonical, department=fam.department,
        sub_department=fam.sub_department, seniority=level, similarity=sim, confident=sim >= 0.6, modifiers=mods,
    )


# ------------------------------------------------------------------------------- comparison
def family_distance(a: str, b: str) -> int:
    """0 same, 1 adjacent, 2 same sub-department, 3 same department, 9 unrelated."""
    if a == b:
        return 0 if a != "unknown" else 9
    tax = taxonomy()
    fa, fb = tax.families.get(a), tax.families.get(b)
    if not fa or not fb:
        return 9
    if b in fa.adjacent or a in fb.adjacent:
        return 1
    if fa.sub_department == fb.sub_department:
        return 2
    if fa.department == fb.department:
        return 3
    return 9


_DIST_SCORE = {0: 1.0, 1: 0.7, 2: 0.55, 3: 0.3, 9: 0.0}


def title_similarity(a: NormTitle, b: NormTitle) -> float:
    """0-1 similarity used by the `title_match` fit feature: taxonomy distance blended with a
    small lexical term so 'Payments Engineer' beats 'Backend Engineer' for a payments brief."""
    base = _DIST_SCORE[family_distance(a.family, b.family)]
    lex = jaccard(_clean(a.raw), _clean(b.raw))
    return min(1.0, 0.85 * base + 0.15 * lex)


def resolve_family(text: str) -> tuple[str, float]:
    nt = normalize_title(text)
    return nt.family, nt.similarity


def expand_titles(titles: list[str], depth: int = 1, per_family: int = 4) -> list[str]:
    """Query expansion: input titles -> their family's aliases + adjacent families' canonical titles."""
    tax = taxonomy()
    out: list[str] = []
    fams: list[str] = []
    for t in titles:
        nt = normalize_title(t)
        if nt.family != "unknown" and nt.family not in fams:
            fams.append(nt.family)
        if normalize(t) not in out:
            out.append(normalize(t))
    for fid in list(fams):
        for a in tax.titles_for(fid, per_family):
            if a not in out:
                out.append(a)
        for nb in tax.adjacent(fid, depth):
            canon = _alias_norm(tax.families[nb].canonical)
            if canon not in out:
                out.append(canon)
    return out
