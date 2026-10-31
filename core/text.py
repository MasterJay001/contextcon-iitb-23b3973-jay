"""Small, dependency-free text helpers (tokenising, scrubbing, canonical forms)."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9+#.]+")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it its of on or that the their this to "
    "was were will with we you your our they he she them his her who whom".split()
)
LEGAL_SUFFIXES = (
    "private limited", "pvt ltd", "pvt. ltd.", "pvt limited", "limited", "ltd", "inc", "inc.", "llc",
    "llp", "gmbh", "corp", "corporation", "co.", "company", "technologies", "technology", "labs",
    "solutions", "systems", "software", "india", "private", "pvt",
)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize(s: str | None) -> str:
    """lowercase, accent-free, single-spaced. Keeps + # . so c++ / c# / node.js survive."""
    if not s:
        return ""
    return _WS.sub(" ", strip_accents(str(s)).lower()).strip()


def tokenize(s: str | None, *, drop_stop: bool = True) -> list[str]:
    toks = [t.strip(".") for t in _NON_ALNUM.split(normalize(s)) if t.strip(".")]
    if drop_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def company_variants(name: str | None) -> list[str]:
    """Surface forms of a company name that should all count as a mention."""
    if not name:
        return []
    base = normalize(name)
    variants = {base}
    stripped = base
    changed = True
    while changed:
        changed = False
        for suf in LEGAL_SUFFIXES:
            if stripped.endswith(" " + suf):
                stripped = stripped[: -len(suf) - 1].strip()
                changed = True
    variants.add(stripped)
    variants.add(stripped.replace(" ", ""))
    return sorted((v for v in variants if len(v) >= 3), key=len, reverse=True)


def scrub_mentions(text: str | None, names: Iterable[str], replacement: str = "[HIRING COMPANY]") -> tuple[str, int]:
    """Replace any mention of the hiring company in free text. Returns (text, n_replacements)."""
    if not text:
        return "", 0
    out, total = text, 0
    for name in names:
        if not name:
            continue
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", re.IGNORECASE)
        out, n = pat.subn(replacement, out)
        total += n
    return out, total


def scrub_future_mentions(text: str | None, T_year: int) -> tuple[str, int]:
    """Drop sentences that mention a calendar year after T. Descriptions are free text that may
    have been edited after T ("... scaled the team in 2026"); this is cheap and conservative."""
    if not text:
        return "", 0
    kept, dropped = [], 0
    for sent in re.split(r"(?<=[.!?\n])\s+", text):
        years = [int(m.group(0)) for m in _YEAR.finditer(sent)]
        if any(y > T_year for y in years):
            dropped += 1
        else:
            kept.append(sent)
    return " ".join(kept).strip(), dropped


def contains_phrase(haystack: str, needle: str) -> bool:
    """Case/accents/whitespace-insensitive substring test used to verify evidence quotes."""
    n = normalize(needle)
    return bool(n) and n in normalize(haystack)


def first_name(full: str | None) -> str:
    if not full:
        return ""
    return str(full).strip().split(" ")[0]


def truncate(s: str | None, n: int) -> str:
    if not s:
        return ""
    s = _WS.sub(" ", s).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"
