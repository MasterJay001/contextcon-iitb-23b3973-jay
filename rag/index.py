"""Hybrid retrieval: BM25 (lexical) + dense (semantic) fused with Reciprocal Rank Fusion.

Dense backends, chosen by `fit.rag.backend`:
  * `lsa`   - TF-IDF -> truncated SVD (Latent Semantic Analysis). Pure numpy, no model download,
              deterministic. Captures co-occurrence structure ("payments" ~ "UPI" ~ "ledger") from the corpus itself.
  * `onnx`  - `fastembed` INT8-quantised ONNX sentence embeddings (BAAI bge-small family) when the package and
              model are available. Smaller/faster than fp32 and needs no GPU; falls back to LSA otherwise.
  * `auto`  - onnx if it loads, else lsa.
  * `bm25`  - lexical only.

RRF:  score(d) = sum_r 1 / (k + rank_r(d)),  k = 60 (Cormack et al.). It needs no score calibration between
retrievers, which is why it is the default fusion rule for BM25 + dense.

Rule 4 note: this module ranks *documents we hand it* (knowledge chunks, masked profile texts). It never
orders API results; retrieval from Crustdata stays neutral-sorted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from core.events import log_event
from core.text import tokenize


@dataclass
class Doc:
    id: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class Hit:
    doc: Doc
    score: float
    rank: int


# ------------------------------------------------------------------------------- BM25
class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[Doc] = []
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        self.avgdl = 0.0
        self.lens: np.ndarray = np.zeros(0)

    def fit(self, docs: Iterable[Doc]) -> "BM25":
        self.docs = list(docs)
        self.tf, self.df = [], {}
        for d in self.docs:
            counts: dict[str, int] = {}
            for t in tokenize(d.text):
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1
        self.lens = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avgdl = float(self.lens.mean()) if len(self.lens) else 0.0
        return self

    def idf(self, t: str) -> float:
        n, df = len(self.docs), self.df.get(t, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        if not self.docs:
            return out
        q = tokenize(query)
        for t in set(q):
            if t not in self.df:
                continue
            idf = self.idf(t)
            for i, counts in enumerate(self.tf):
                f = counts.get(t)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avgdl or 1.0))
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out


# ------------------------------------------------------------------------------- dense: LSA
class LSA:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)
        self.Vt: np.ndarray = np.zeros((0, 0))
        self.doc_emb: np.ndarray = np.zeros((0, 0))
        self.name = "lsa"

    def _tfidf(self, toks: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t in toks:
            j = self.vocab.get(t)
            if j is not None:
                v[j] += 1.0
        v = np.where(v > 0, 1.0 + np.log(np.maximum(v, 1e-9)), 0.0) * self.idf
        n = np.linalg.norm(v)
        return v / n if n else v

    def fit(self, docs: list[Doc]) -> "LSA":
        toks = [tokenize(d.text) for d in docs]
        df: dict[str, int] = {}
        for ts in toks:
            for t in set(ts):
                df[t] = df.get(t, 0) + 1
        vocab_terms = sorted(t for t, c in df.items() if c >= 1)
        self.vocab = {t: i for i, t in enumerate(vocab_terms)}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in vocab_terms])
        X = np.vstack([self._tfidf(ts) for ts in toks]) if docs else np.zeros((0, len(self.vocab)))
        if X.size == 0 or min(X.shape) < 2:
            self.Vt, self.doc_emb = np.zeros((0, X.shape[1] if X.size else 0)), np.zeros((len(docs), 0))
            return self
        k = max(1, min(self.dim, min(X.shape) - 1))
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.Vt = Vt[:k]
        emb = U[:, :k] * S[:k]
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        self.doc_emb = emb / np.where(norms == 0, 1.0, norms)
        return self

    def scores(self, query: str) -> np.ndarray:
        if self.doc_emb.size == 0:
            return np.zeros(len(self.doc_emb))
        q = self._tfidf(tokenize(query)) @ self.Vt.T
        n = np.linalg.norm(q)
        return self.doc_emb @ (q / n) if n else np.zeros(self.doc_emb.shape[0])


# ------------------------------------------------------------------------------- dense: quantised ONNX (optional)
class OnnxDense:
    """fastembed (ONNX Runtime) sentence embeddings. Quantised INT8 variants are preferred: ~4x smaller
    weights and faster CPU inference with negligible retrieval loss for short English passages."""

    def __init__(self, model_names: list[str], cache_dir: str | None = None):
        from fastembed import TextEmbedding  # optional dependency

        last: Exception | None = None
        self.model = None
        for name in model_names:
            try:
                self.model = TextEmbedding(model_name=name, cache_dir=cache_dir)
                self.name = f"onnx:{name}"
                break
            except Exception as exc:  # unsupported name / offline
                last = exc
        if self.model is None:
            raise RuntimeError(f"no ONNX embedding model could be loaded: {last}")
        self.doc_emb = np.zeros((0, 0))

    def fit(self, docs: list[Doc]) -> "OnnxDense":
        embs = np.array(list(self.model.embed([d.text for d in docs])), dtype=np.float32)
        self.doc_emb = embs / np.where(np.linalg.norm(embs, axis=1, keepdims=True) == 0, 1.0, np.linalg.norm(embs, axis=1, keepdims=True))
        return self

    def scores(self, query: str) -> np.ndarray:
        q = np.array(list(self.model.embed([query])), dtype=np.float32)[0]
        n = np.linalg.norm(q)
        return self.doc_emb @ (q / n) if n else np.zeros(self.doc_emb.shape[0])


def _make_dense(backend: str, dim: int, cfg: dict | None):
    if backend in {"onnx", "auto"}:
        try:
            names = list((cfg or {}).get("fit", {}).get("rag", {}).get("onnx_models", [])) if cfg else []
            names = names or ["Qdrant/bge-small-en-v1.5-onnx-Q", "BAAI/bge-small-en-v1.5"]
            return OnnxDense(names)
        except Exception as exc:
            if backend == "onnx":
                log_event("rag_dense_fallback", reason=str(exc)[:160])
            else:
                log_event("rag_dense_auto", note="onnx unavailable, using LSA", reason=str(exc)[:120])
    if backend == "bm25":
        return None
    return LSA(dim=dim)


# ------------------------------------------------------------------------------- hybrid
def rrf(rankings: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion over score arrays (higher = better). Ties get distinct stable ranks."""
    n = len(rankings[0]) if rankings else 0
    fused = np.zeros(n)
    for s in rankings:
        order = np.argsort(-s, kind="stable")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        fused += 1.0 / (k + ranks)
    return fused


class HybridIndex:
    def __init__(self, backend: str = "auto", rrf_k: int = 60, dim: int = 128, cfg: dict | None = None):
        self.backend, self.rrf_k, self.dim, self.cfg = backend, rrf_k, dim, cfg
        self.docs: list[Doc] = []
        self.bm25 = BM25()
        self.dense = None
        self.dense_name = "none"

    def fit(self, docs: list[Doc]) -> "HybridIndex":
        self.docs = list(docs)
        self.bm25.fit(self.docs)
        self.dense = _make_dense(self.backend, self.dim, self.cfg) if len(self.docs) >= 2 else None
        if self.dense is not None:
            self.dense.fit(self.docs)
            self.dense_name = getattr(self.dense, "name", "dense")
        return self

    def component_scores(self, query: str) -> dict[str, np.ndarray]:
        out = {"bm25": self.bm25.scores(query)}
        if self.dense is not None:
            out["dense"] = self.dense.scores(query)
        return out

    def scores(self, query: str) -> np.ndarray:
        comps = list(self.component_scores(query).values())
        return rrf(comps, self.rrf_k) if len(comps) > 1 else (comps[0] if comps else np.zeros(0))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        sc = self.scores(query)
        order = np.argsort(-sc, kind="stable")[:k]
        return [Hit(self.docs[i], float(sc[i]), r + 1) for r, i in enumerate(order) if sc[i] > 0]

    def similarity01(self, query: str) -> np.ndarray:
        """Per-document similarity in [0,1]: mean of (BM25 / max BM25) and clipped dense cosine. Used as the
        `text_sim` fit feature (absolute-ish scale, unlike rank-only RRF)."""
        comps = self.component_scores(query)
        parts = []
        b = comps["bm25"]
        parts.append(b / b.max() if b.size and b.max() > 0 else np.zeros_like(b))
        if "dense" in comps:
            parts.append(np.clip(comps["dense"], 0.0, 1.0))
        return np.mean(parts, axis=0) if parts else np.zeros(len(self.docs))
