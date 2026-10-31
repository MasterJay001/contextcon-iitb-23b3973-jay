"""Metrics and statistics. Small samples (30-50 cases) make every number noisy, so each is reported with an interval,
a random baseline, and comparisons use PAIRED resampling (same cases, both variants, resampled together).

    Recall@k = (1/N) sum_i 1[ min_{h in H_i} rank_i(h) <= k ]       MRR = (1/N) sum_i 1 / min_h rank_i(h)   (0 if absent)
    Random baseline for case i (n = pool size, h = #hires in the pool):  P(hit@k) = 1 - C(n-h, k) / C(n, k)   (= k/n for h = 1)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class CaseScore:
    case_id: str
    pool_size: int
    n_hires: int
    ranks: list[int | None]                       # rank of each hire in the ranked list (1-based) or None
    injected: bool = False                        # hire was injected into the pool (ranking-only fallback)
    extra: dict = field(default_factory=dict)

    @property
    def n_in_pool(self) -> int:
        return sum(r is not None for r in self.ranks)

    @property
    def in_pool(self) -> bool:
        return self.n_in_pool > 0

    @property
    def best_rank(self) -> int | None:
        rs = [r for r in self.ranks if r is not None]
        return min(rs) if rs else None

    def recall(self, k: int) -> float:
        b = self.best_rank
        return 1.0 if (b is not None and b <= k) else 0.0

    def rr(self) -> float:
        b = self.best_rank
        return 1.0 / b if b else 0.0

    def ndcg(self, k: int) -> float:
        rel = sorted((r for r in self.ranks if r is not None and r <= k))
        if not rel:
            return 0.0
        dcg = sum(1.0 / math.log2(r + 1) for r in rel)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(len(self.ranks)))
        return dcg / ideal if ideal else 0.0

    def baseline(self, k: int) -> float:
        n, h = self.pool_size, self.n_in_pool
        if h <= 0 or n <= 0:
            return 0.0
        k = min(k, n)
        if n - h < k:
            return 1.0
        return 1.0 - math.comb(n - h, k) / math.comb(n, k)

    def baseline_rr(self) -> float:
        """Expected reciprocal rank of the first of h hires among n randomly ordered candidates."""
        n, h = self.pool_size, self.n_in_pool
        if h <= 0 or n <= 0:
            return 0.0
        # E[1/min rank] = sum_j (1/j) * P(min rank = j) with P(min rank > j) = C(n-j, h)/C(n, h)
        tot, prev = 0.0, 1.0
        for j in range(1, n - h + 2):
            surv = math.comb(n - j, h) / math.comb(n, h) if n - j >= h else 0.0
            tot += (1.0 / j) * (prev - surv)
            prev = surv
        return tot


def mean_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 11, alpha: float = 0.05) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 1000, seed: int = 11) -> dict:
    """Paired comparison of per-case values a (treatment) and b (control): bootstrap CI of the mean difference,
    one-sided P(diff <= 0), and a two-sided sign-flip permutation p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    out = {"n": int(n), "mean_diff": float(d.mean()), "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
           "p_boot_le0": float((boots <= 0).mean()), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}
    nz = d[d != 0]
    if nz.size == 0:
        out["p_signflip"] = 1.0
    else:
        signs = rng.choice([-1.0, 1.0], size=(20000, nz.size))
        perm = np.abs((signs * nz).mean(axis=1))
        out["p_signflip"] = float((perm >= abs(nz.mean()) - 1e-12).mean())
    out["verdict"] = ("clear gain" if out["ci_lo"] > 0 else "clear loss" if out["ci_hi"] < 0 else "no clear gain (interval includes zero)")
    return out


def mcnemar_exact(a_hit: Sequence[float], b_hit: Sequence[float]) -> dict:
    """Exact McNemar on paired binary outcomes (a vs b): only discordant pairs carry information."""
    a, b = np.asarray(a_hit) > 0.5, np.asarray(b_hit) > 0.5
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    p = 1.0 if n == 0 else float(stats.binomtest(n10, n, 0.5).pvalue)
    return {"a_only": n10, "b_only": n01, "p": p}


def holm(pvals: Sequence[float], alpha: float = 0.05) -> list[dict]:
    """Holm-Bonferroni step-down over m comparisons. Returns adjusted p and reject flag in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return [{"p": float(pvals[i]), "p_adj": float(adj[i]), "reject": bool(adj[i] < alpha)} for i in range(m)]


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Mann-Whitney AUC: P(score(pos) > score(neg)) + 0.5 P(tie)."""
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = stats.rankdata(allv)
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def simulated_power(p_base: float, p_alt: float, n_cases: int, n_sims: int = 2000, seed: int = 5) -> float:
    """Power of the paired sign-flip test to detect a recall difference p_alt - p_base with n_cases cases, assuming the
    paired outcomes are independent Bernoulli (conservative: real pairs are positively correlated, which raises power)."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sims):
        a = rng.random(n_cases) < p_alt
        b = rng.random(n_cases) < p_base
        n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
        n = n10 + n01
        if n and stats.binomtest(n10, n, 0.5).pvalue < 0.05 and n10 > n01:
            hits += 1
    return hits / n_sims
