"""A5: learned ranker. Pairwise logistic regression (RankNet / Bradley-Terry) over all 16 raw features.

    minimise   sum_cases  (1/|P_c|) sum_{(p,n) in P_c}  log(1 + exp(-(x_p - x_n) . w))   +   lambda ||w||^2

where P_c pairs each hire p of case c with every other pool member n. Cases are weighted equally (not by pool size).
Only ~20-25 TRAIN cases exist, so the model is small (16 weights), strongly regularised, and lambda is chosen by
leave-one-company-out cross-validation on train (mean reciprocal rank). Held-out cases never touch any of this.

Fine-tuning a language model or an embedding model on ~25 cases would be pure overfitting, so we do not; quantisation is
therefore irrelevant to the ranker. (Dense retrieval uses an already-quantised ONNX encoder when installed; see rag/index.py.)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from agents.artifacts import CaseArtifacts
from agents.ranker import FEATURE_NAMES, feature_matrix
from eval.evaluator import Evaluator

LAMBDAS = [0.003, 0.03, 0.3, 3.0]


@dataclass
class LearnedModel:
    w: np.ndarray
    lam: float
    names: list[str]
    cv: dict[str, float]

    def to_json(self) -> dict:
        return {"w": [float(x) for x in self.w], "lambda": self.lam, "names": self.names, "cv_mean_rr": self.cv}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "LearnedModel":
        d = json.loads(path.read_text(encoding="utf-8"))
        return LearnedModel(np.array(d["w"]), float(d["lambda"]), d["names"], d.get("cv_mean_rr", {}))


def _pairs(art: CaseArtifacts, hire_ids: list[int]) -> np.ndarray | None:
    ids, X = feature_matrix(art)
    pos = [i for i, p in enumerate(ids) if p in set(hire_ids)]
    neg = [i for i in range(len(ids)) if i not in set(pos)]
    if not pos or not neg:
        return None
    D = (X[pos][:, None, :] - X[neg][None, :, :]).reshape(-1, X.shape[1])
    return D


def _fit(D_list: list[np.ndarray], lam: float) -> np.ndarray:
    d = D_list[0].shape[1]

    def f(w):
        loss, grad = 0.0, np.zeros(d)
        for D in D_list:
            m = D @ w
            loss += np.mean(np.logaddexp(0.0, -m)) / len(D_list)
            s = 1.0 / (1.0 + np.exp(np.clip(m, -30, 30)))
            grad += -(D * s[:, None]).mean(axis=0) / len(D_list)
        return loss + lam * float(w @ w), grad + 2 * lam * w

    res = minimize(f, np.zeros(d), jac=True, method="L-BFGS-B")
    return res.x


def _mean_rr(w: np.ndarray, arts: dict[str, CaseArtifacts], evalr: Evaluator, case_ids: list[str]) -> float:
    rrs = []
    for cid in case_ids:
        art = arts[cid]
        ids, X = feature_matrix(art)
        order = np.argsort(-(X @ w), kind="stable")
        pos = {ids[i]: r + 1 for r, i in enumerate(order)}
        ranks = [pos[h] for h in evalr.hire_ids(cid) if h in pos]
        rrs.append(1.0 / min(ranks) if ranks else 0.0)
    return float(np.mean(rrs)) if rrs else 0.0


def train(arts: dict[str, CaseArtifacts], evalr: Evaluator, company_of: dict[str, int]) -> LearnedModel:
    """Train on the given (TRAIN) artifacts. Leave-one-company-out CV chooses lambda."""
    usable = {}
    for cid, art in arts.items():
        D = _pairs(art, evalr.hire_ids(cid))
        if D is not None:
            usable[cid] = D
    cids = sorted(usable)
    if len(cids) < 4:
        return LearnedModel(np.zeros(len(FEATURE_NAMES)), 1.0, FEATURE_NAMES, {})
    companies = sorted({company_of[c] for c in cids})
    cv: dict[str, float] = {}
    for lam in LAMBDAS:
        scores = []
        for co in companies:
            tr = [c for c in cids if company_of[c] != co]
            te = [c for c in cids if company_of[c] == co]
            if not tr or not te:
                continue
            w = _fit([usable[c] for c in tr], lam)
            scores.append(_mean_rr(w, arts, evalr, te) * len(te))
        cv[str(lam)] = float(sum(scores) / max(1, len(cids)))
    best = max(LAMBDAS, key=lambda l: (round(cv[str(l)], 6), l))
    w = _fit([usable[c] for c in cids], best)
    return LearnedModel(w, best, FEATURE_NAMES, cv)
