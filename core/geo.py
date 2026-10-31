"""Location matching on top of knowledge/locations.json.

Profiles carry free-text locations ("Bengaluru, Karnataka, India", "Greater Bangalore Area").
We map them to a metro cluster when an alias appears as a whole token sequence; otherwise fall
back to state/country comparison. Scores: same metro 1.0, same state 0.6, same country 0.3.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from core.config import ROOT
from core.text import normalize


@dataclass(frozen=True)
class Metro:
    id: str
    canonical: str
    state: str
    country: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Place:
    metro: str | None = None
    state: str | None = None
    country: str | None = None
    remote: bool = False
    raw: str = ""


class Geo:
    def __init__(self, raw: dict):
        self.metros: dict[str, Metro] = {
            m["id"]: Metro(m["id"], m["canonical"], normalize(m["state"]), normalize(m["country"]), tuple(normalize(a) for a in m["aliases"]))
            for m in raw["metros"]
        }
        self.country_alias: dict[str, str] = {}
        for canon, al in raw["countries"].items():
            for a in al:
                self.country_alias[normalize(a)] = canon
        self.remote_markers = tuple(normalize(x) for x in raw["remote_markers"])
        self._alias_rows = sorted(((a, mid) for mid, m in self.metros.items() for a in m.aliases), key=lambda x: -len(x[0]))

    def parse(self, text: str | None) -> Place:
        s = normalize(text)
        if not s:
            return Place(raw="")
        padded = f" {re.sub(r'[^a-z0-9 ]+', ' ', s)} "
        padded = re.sub(r"\s+", " ", padded)
        remote = any(f" {m} " in padded for m in self.remote_markers)
        metro = None
        for alias, mid in self._alias_rows:
            if f" {alias} " in padded:
                metro = mid
                break
        country = None
        for alias, canon in self.country_alias.items():
            if len(alias) > 2 and f" {alias} " in padded:
                country = canon
                break
        state = None
        if metro:
            m = self.metros[metro]
            state, country = state or m.state, country or m.country
        return Place(metro=metro, state=state, country=country, remote=remote, raw=s)

    def score(self, candidate_loc: str | None, job_loc: str | None, *, remote_ok: bool = False) -> float:
        c, j = self.parse(candidate_loc), self.parse(job_loc)
        if not j.raw and not j.metro:
            return 0.5  # unspecified job location: neutral
        if c.metro and j.metro and c.metro == j.metro:
            return 1.0
        if c.state and j.state and c.state == j.state:
            return 0.6
        if c.country and j.country and c.country == j.country:
            return 1.0 if (remote_ok or j.remote) else 0.3
        if not c.raw:
            return 0.15
        return 0.0

    def expand(self, text: str | None) -> list[str]:
        """Aliases a recruiter would also search: the whole metro cluster for a city mention."""
        p = self.parse(text)
        if p.metro:
            m = self.metros[p.metro]
            return [m.canonical] + [a for a in m.aliases if a != normalize(m.canonical)][:5]
        return [text] if text else []

    def same_metro_or_state(self, a: str | None, b: str | None) -> bool:
        pa, pb = self.parse(a), self.parse(b)
        return bool((pa.metro and pa.metro == pb.metro) or (pa.state and pa.state == pb.state))


@lru_cache(maxsize=1)
def geo() -> Geo:
    with open(ROOT / "knowledge" / "locations.json", "r", encoding="utf-8") as fh:
        return Geo(json.load(fh))
