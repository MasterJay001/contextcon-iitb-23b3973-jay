"""Connection agent: is there a path in?  Deterministic graph maths, no LLM.

Roster = people employed at the hiring company ON T (role start <= T and ongoing at T after masking). People who
joined after T - including the real hire - are structurally absent (leak #8).

Features (each squashed to [0,1]):
  ex_colleague  distinct roster members who overlapped >= min_overlap_months with the candidate at a past employer
                 1 - exp(-n/2)
  closeness     sum over overlaps of overlap_months * w(size),  w = ln(12)/ln(2+size)  (Adamic-Adar-style inverse-log
                 size weighting: an overlap at a 40-person firm counts ~3x one at a 50,000-person firm)   x/(x+6)
  feeder        roster members whose employer right before joining the target was the candidate's current employer
                 ("companies have hiring pipelines from specific feeders")                                  1 - exp(-n/2)
  ppr           Personalized PageRank mass from the roster seeds over the bipartite person-employer graph
                 (edge weight = months * w(size)); captures multi-hop ties                                  x/(x+median)
  school        roster members sharing a school (weak; riskiest for bias; weight 0 by default, reported with/without)
Company size comes from the role's `company_headcount_range` (a slowly varying snapshot: documented, and the
connection feature is re-run without it in the robustness suite).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from agents.movability import _emp_key
from agents.time_guard import TargetRef
from core.dates import overlap_months
from core.models import MaskedProfile, Role
from core.text import normalize

SIZE_MID = {"1-10": 5, "2-10": 5, "11-50": 30, "51-200": 110, "201-500": 350, "501-1000": 750, "1001-5000": 2500,
            "5001-10000": 7500, "10001+": 30000}


def size_of(r: Role) -> int:
    return SIZE_MID.get(r.company_headcount_range or "", 1000)


def size_weight(size: int, floor: int = 10) -> float:
    return math.log(12.0) / math.log(2.0 + max(size, floor))


@dataclass
class PathInfo:
    colleague_id: int
    employer: str
    overlap_months: float
    start: str
    end: str
    weight: float
    cand_role_ref: int                  # index into the candidate's masked roles (for citations)


@dataclass
class ConnDiag:
    paths: list[PathInfo] = field(default_factory=list)
    n_ex_colleagues: int = 0
    notes: list[str] = field(default_factory=list)


class ConnectionIndex:
    def __init__(self, cfg: dict, roster: list[MaskedProfile], T, target: TargetRef | None):
        self.cfg, self.cc = cfg, cfg["connection"]
        self.T = T
        self.target = target
        self.roster = [m for m in roster]
        self.available = bool(self.roster)
        self.min_overlap = float(self.cc["min_overlap_months"])
        self.floor = int(self.cc["size_floor"])
        # employer -> [(pid, start, end, role)]
        self.by_employer: dict[str, list[tuple[int, object, object, Role]]] = {}
        self.feeders: dict[str, int] = {}
        self.employer_names: dict[str, str] = {}
        self.schools: dict[str, set[int]] = {}
        tkey = None
        for m in self.roster:
            at_target = next((r for r in m.roles if r.end is None and target and target.matches_role(r)), None)
            if at_target is not None:
                tkey = _emp_key(at_target)
            for r in m.roles:
                if r.start is None:
                    continue
                s, e = r.start.date, (r.end.date if r.end else T)
                self.by_employer.setdefault(_emp_key(r), []).append((m.person_id, s, e, r))
            self._register_feeder(m, target)
            for sc in m.education:
                if sc.school:
                    self.schools.setdefault(normalize(sc.school), set()).add(m.person_id)
        self.target_key = tkey
        self._ppr: dict[int, float] = {}

    def _register_feeder(self, m: MaskedProfile, target: TargetRef | None) -> None:
        roles = sorted((r for r in m.roles if r.start is not None), key=lambda r: r.start.date)
        for i, r in enumerate(roles):
            if target and target.matches_role(r) and r.end is None and i > 0:
                prev = roles[i - 1]
                if prev.end is not None and (r.start.date - prev.end.date).days <= 190 and not target.matches_role(prev):
                    k = _emp_key(prev)
                    self.feeders[k] = self.feeders.get(k, 0) + 1
                    self.employer_names.setdefault(k, prev.company_name or k)
                break

    def feeder_report(self, top: int = 5) -> list[tuple[str, int]]:
        """Top employers that sent people to the hiring company before T (a 'fact you cannot Google')."""
        return sorted(((self.employer_names.get(k, k), n) for k, n in self.feeders.items()), key=lambda x: (-x[1], x[0]))[:top]

    # ---- personalised PageRank -------------------------------------------------------------
    def compute_ppr(self, pool: list[MaskedProfile]) -> dict[int, float]:
        people = {m.person_id: m for m in [*self.roster, *pool]}
        pid_list = sorted(people)
        emp_index: dict[str, int] = {}
        rows, cols, vals = [], [], []
        pid_idx = {p: i for i, p in enumerate(pid_list)}
        for p in pid_list:
            for r in people[p].roles:
                if r.start is None:
                    continue
                e = r.end.date if r.end else self.T
                months = max(0.0, (e - r.start.date).days / 30.4375)
                if months < 1:
                    continue
                k = _emp_key(r)
                j = emp_index.setdefault(k, len(pid_list) + len(emp_index))
                w = months * size_weight(size_of(r), self.floor)
                rows += [pid_idx[p], j]
                cols += [j, pid_idx[p]]
                vals += [w, w]
        n = len(pid_list) + len(emp_index)
        if n == 0 or not self.roster:
            return {}
        A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        P = sparse.diags(1.0 / deg) @ A                      # row-stochastic
        seed = np.zeros(n)
        for m in self.roster:
            seed[pid_idx[m.person_id]] = 1.0
        seed /= seed.sum()
        d, r = float(self.cc["ppr_damping"]), seed.copy()
        for _ in range(int(self.cc["ppr_iterations"])):
            r = (1 - d) * seed + d * (P.T @ r)
        out = {p: float(r[pid_idx[p]]) for p in pid_list}
        self._ppr = out
        return out

    # ---- per candidate ---------------------------------------------------------------------
    def features(self, mp: MaskedProfile) -> tuple[dict[str, float | None], ConnDiag]:
        diag = ConnDiag()
        if not self.available:
            diag.notes.append("connection unavailable: no roster for the hiring company at T")
            return {k: None for k in ("ex_colleague", "closeness", "feeder", "ppr", "school")}, diag
        colleagues: dict[int, float] = {}
        closeness = 0.0
        for ci, r in enumerate(mp.roles):
            if r.start is None:
                continue
            key = _emp_key(r)
            s1, e1 = r.start.date, (r.end.date if r.end else self.T)
            for pid, s2, e2, rr in self.by_employer.get(key, []):
                if pid == mp.person_id:
                    continue
                m = overlap_months(s1, e1, s2, e2)
                if m >= self.min_overlap:
                    w = m * size_weight(size_of(r), self.floor)
                    closeness += w
                    colleagues[pid] = colleagues.get(pid, 0.0) + w
                    diag.paths.append(PathInfo(pid, r.company_name or "?", round(m, 1), max(s1, s2).isoformat()[:7], min(e1, e2).isoformat()[:7], w, ci))
        diag.n_ex_colleagues = len(colleagues)
        diag.paths.sort(key=lambda p: (-p.weight, p.colleague_id))
        diag.paths = diag.paths[:3]
        cur = next((r for r in mp.roles if r.end is None), None)
        n_feed = self.feeders.get(_emp_key(cur), 0) if cur is not None else 0
        school_hits = set()
        for sc in mp.education:
            school_hits |= self.schools.get(normalize(sc.school or ""), set())
        school_hits.discard(mp.person_id)
        feats = {
            "ex_colleague": 1.0 - math.exp(-len(colleagues) / 2.0),
            "closeness": closeness / (closeness + 6.0),
            "feeder": 1.0 - math.exp(-n_feed / 2.0),
            "ppr": None,
            "school": 1.0 - math.exp(-len(school_hits) / 3.0),
        }
        if self._ppr:
            feats["ppr"] = self._ppr.get(mp.person_id, 0.0)
        return feats, diag

    def normalise_ppr(self, feats_by_pid: dict[int, dict[str, float | None]]) -> None:
        """Squash raw PPR mass to [0,1] relative to the pool median (x / (x + median))."""
        vals = [f["ppr"] for f in feats_by_pid.values() if f.get("ppr") is not None]
        if not vals:
            return
        med = float(np.median(vals)) or 1e-9
        for f in feats_by_pid.values():
            if f.get("ppr") is not None:
                x = float(f["ppr"])
                f["ppr"] = x / (x + med)


def combine_conn(feats: dict[str, float | None], inner: dict[str, float], use_school: bool) -> float:
    num = den = 0.0
    for k, w in inner.items():
        if k == "school" and not use_school:
            continue
        v = feats.get(k)
        if v is not None and w > 0:
            num += w * v
            den += w
    return num / den if den else 0.0
