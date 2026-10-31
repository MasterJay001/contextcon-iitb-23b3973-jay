"""Ranker: score = w_F * F + w_M * M + w_C * C, with the ablation variants of design doc section 3.4.

    A0  keyword baseline   plain lexical (BM25) match of the brief against the MASKED profile text - what a keyword
                           search tool returns when run as of T (no taxonomy, no features, no LLM)
    A1  Fit only           F                      (stands in for today's tools)
    A2  Fit + Movability   F + w_M * M
    A3  Fit + Connection   F + w_C * C
    A4  Full               F + w_M * M + w_C * C
    A5  learned            logistic pairwise ranker over every raw feature (trained on TRAIN cases only)

Rule 4: nothing here looks at API order. Ties break by (score desc, person_id asc) so reruns are identical.
"""
from __future__ import annotations

import math

import numpy as np

from agents.artifacts import CaseArtifacts
from agents.connection import combine_conn
from agents.fit import combine_fit
from agents.movability import combine_move
from core.models import MaskedProfile, Scored, Weights

VARIANTS = ("A0", "A1", "A2", "A3", "A4", "A5")

FEATURE_NAMES = [
    "fit.title_match", "fit.seniority", "fit.skills", "fit.location", "fit.domain", "fit.text_sim", "fit.llm_fit",
    "move.p_move", "move.personal", "move.no_recent_promo", "move.stalled", "move.between_roles",
    "conn.ex_colleague", "conn.closeness", "conn.feeder", "conn.ppr",
]


def feature_row(art: CaseArtifacts, pid: int) -> dict[str, float]:
    row: dict[str, float] = {}
    for k, v in art.fit_feats.get(pid, {}).items():
        row[f"fit.{k}"] = float(v)
    for k, v in art.move_feats.get(pid, {}).items():
        if v is not None:
            row[f"move.{k}"] = float(v)
    for k, v in art.conn_feats.get(pid, {}).items():
        if v is not None:
            row[f"conn.{k}"] = float(v)
    return row


def feature_matrix(art: CaseArtifacts, ids: list[int] | None = None) -> tuple[list[int], np.ndarray]:
    ids = ids if ids is not None else sorted(art.pool_ids)
    X = np.zeros((len(ids), len(FEATURE_NAMES)))
    for i, pid in enumerate(ids):
        row = feature_row(art, pid)
        for j, name in enumerate(FEATURE_NAMES):
            X[i, j] = row.get(name, 0.0)
    return ids, X


def family_scores(art: CaseArtifacts, pid: int, w: Weights, use_school: bool = False) -> tuple[float, float, float]:
    F = combine_fit(art.fit_feats.get(pid, {}), w.fit_inner)
    M = combine_move(art.move_feats.get(pid, {}), w.move_inner)
    C = combine_conn(art.conn_feats.get(pid, {}), w.conn_inner, use_school) if art.conn_available else 0.0
    return F, M, C


def variant_weights(variant: str, w: Weights) -> tuple[float, float, float]:
    return {"A1": (w.w_F, 0.0, 0.0), "A2": (w.w_F, w.w_M, 0.0), "A3": (w.w_F, 0.0, w.w_C), "A4": (w.w_F, w.w_M, w.w_C)}[variant]


def rank(art: CaseArtifacts, w: Weights, variant: str = "A4", *, use_school: bool = False,
         learned: np.ndarray | None = None, demoted: set[int] | None = None, demote_penalty: float = 0.35) -> list[Scored]:
    """Pure function of (artifacts, weights, variant). Returns candidates best-first."""
    out: list[Scored] = []
    if variant == "A0":
        mx = max(art.bm25.values(), default=0.0) or 1.0
        for pid in art.pool_ids:
            s = art.bm25.get(pid, 0.0) / mx
            out.append(Scored(person_id=pid, F=s, score=s, features=feature_row(art, pid)))
    elif variant == "A5":
        assert learned is not None, "A5 needs learned weights"
        ids, X = feature_matrix(art)
        z = X @ learned
        for pid, s in zip(ids, z):
            F, M, C = family_scores(art, pid, w, use_school)
            out.append(Scored(person_id=pid, F=F, M=M, C=C, score=float(s), features=feature_row(art, pid)))
    else:
        wf, wm, wc = variant_weights(variant, w)
        for pid in art.pool_ids:
            F, M, C = family_scores(art, pid, w, use_school)
            out.append(Scored(person_id=pid, F=F, M=M, C=C, score=wf * F + wm * M + wc * C, features=feature_row(art, pid)))
    if demoted:
        for s in out:
            if s.person_id in demoted:
                s.score *= (1.0 - demote_penalty)
                s.demoted = True
    out.sort(key=lambda s: (-s.score, s.person_id))
    return out


def rank_of(ranked: list[Scored], person_id: int) -> int | None:
    for i, s in enumerate(ranked, 1):
        if s.person_id == person_id:
            return i
    return None


# ------------------------------------------------------------------------------- diversity (live shortlist)
def _div_tokens(mp: MaskedProfile) -> set[str]:
    toks = {f"fam:{mp.norm.family}"}
    for r in mp.roles[:3]:
        if r.company_id is not None:
            toks.add(f"emp:{r.company_id}")
        elif r.company_name:
            toks.add(f"emp:{r.company_name.lower()}")
    for s in mp.skills[:8]:
        toks.add(f"sk:{s.lower()}")
    return toks


def mmr_select(ranked: list[Scored], pool: dict[int, MaskedProfile], k: int, lam: float = 0.8) -> list[Scored]:
    """Maximal Marginal Relevance: argmax  lam * rel(c) - (1-lam) * max_{s in S} sim(c, s).
    rel is the ranker score rescaled to [0,1]; sim is Jaccard over employers/family/skills. Prevents five clones."""
    if not ranked:
        return []
    top = ranked[: max(k * 6, 30)]
    smax, smin = max(s.score for s in top), min(s.score for s in top)
    rel = {s.person_id: ((s.score - smin) / (smax - smin) if smax > smin else 1.0) for s in top}
    toks = {s.person_id: _div_tokens(pool[s.person_id]) for s in top if s.person_id in pool}
    chosen: list[Scored] = []
    rest = list(top)
    while rest and len(chosen) < k:
        best, best_v = None, -math.inf
        for c in rest:
            sim = max((len(toks.get(c.person_id, set()) & toks.get(s.person_id, set())) / max(1, len(toks.get(c.person_id, set()) | toks.get(s.person_id, set())))
                       for s in chosen), default=0.0)
            v = lam * rel[c.person_id] - (1 - lam) * sim
            if v > best_v or (v == best_v and best is not None and c.person_id < best.person_id):
                best, best_v = c, v
        chosen.append(best)
        rest.remove(best)
    return chosen
