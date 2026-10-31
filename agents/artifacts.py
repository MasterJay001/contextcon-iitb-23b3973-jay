"""Everything expensive about one case, computed once and cached: retrieval, masking, the roster, features and
the LLM fit scores. Ranking under any weights/variant is then a pure, free function of these artifacts - which is
what makes the 16-point weight grid, the ablations and the self-improvement loop cost zero extra credits.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from agents.connection import ConnDiag
from agents.fit import FitDetail
from agents.movability import MoveDiag
from core.config import ROOT
from core.models import JobSpec, MaskedProfile, RawProfile


@dataclass
class CaseArtifacts:
    case_id: str
    T: dt.date
    brief: str
    spec: JobSpec
    pool: list[MaskedProfile]
    fit_feats: dict[int, dict[str, float]] = field(default_factory=dict)
    fit_details: dict[int, FitDetail] = field(default_factory=dict)
    move_feats: dict[int, dict[str, float | None]] = field(default_factory=dict)
    move_diag: dict[int, MoveDiag] = field(default_factory=dict)
    survival: dict = field(default_factory=dict)
    conn_feats: dict[int, dict[str, float | None]] = field(default_factory=dict)
    conn_diag: dict[int, ConnDiag] = field(default_factory=dict)
    conn_available: bool = False
    roster_size: int = 0
    bm25: dict[int, float] = field(default_factory=dict)
    raw: dict[int, RawProfile] = field(default_factory=dict)            # today's profiles (cards need names; canary needs leak-on)
    n_retrieved: int = 0
    api_total_count: int | None = None
    steps: list[dict] = field(default_factory=list)
    reject_reasons: dict[str, int] = field(default_factory=dict)
    mask_totals: dict[str, int] = field(default_factory=dict)
    credits: float = 0.0
    warnings: list[str] = field(default_factory=list)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    backends: dict[str, str] = field(default_factory=dict)
    llm_usage: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    feeders: list[tuple[str, int]] = field(default_factory=list)
    roster_ids: list[int] = field(default_factory=list)

    @property
    def pool_ids(self) -> list[int]:
        return [m.person_id for m in self.pool]


def artifact_key(case_id: str, *parts) -> str:
    blob = json.dumps([case_id, *parts], sort_keys=True, default=str).encode()
    return f"{case_id}-{hashlib.sha256(blob).hexdigest()[:12]}"


def artifact_path(cfg: dict, key: str) -> Path:
    d = ROOT / str(cfg["paths"]["cache"]) / "cases"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.pkl"


def save_artifacts(cfg: dict, key: str, art: CaseArtifacts) -> Path:
    p = artifact_path(cfg, key)
    with open(p, "wb") as fh:
        pickle.dump(art, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_artifacts(cfg: dict, key: str) -> CaseArtifacts | None:
    p = artifact_path(cfg, key)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None
