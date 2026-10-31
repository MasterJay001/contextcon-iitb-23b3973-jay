"""Movability agent: would this person change employer around T?  Deterministic maths, no LLM.

Model (all estimated from MASKED data as of T, no labels, so there is nothing to leak or overfit):

 1. Employer spells. Consecutive roles at one employer (gap < merge_gap_months) form one spell, so a promotion
    is not a departure. A spell is an *event* if it ended before T, *right-censored* if still ongoing at T.

 2. Baseline hazard from the pool + roster spells, fitted by censored maximum likelihood with a Gaussian prior
    on log-parameters (MAP shrinkage; strength fixed, so with few spells the prior dominates):
        log-logistic  S(t) = 1/(1+(t/a)^b)   h(t) = (b/a)(t/a)^(b-1)/(1+(t/a)^b)   L(t)=ln(1+(t/a)^b)
        Weibull       S(t) = exp(-(t/l)^k)   L(t) = (t/l)^k
    The log-logistic hazard RISES THEN FALLS (peak t* = a((b-1)/(b+1))^(1/b)): exactly the
    "movement peaks after ~18-36 months" hypothesis, but the peak is *estimated*, not assumed. Kaplan-Meier is
    reported alongside as the non-parametric check.

 3. Person effect (gamma frailty, empirical Bayes). A habitual two-year mover is "due" earlier. With
    n_i = completed past spells and e_i = cumulative baseline hazard they were exposed to (completed spells + the
    current one so far), the posterior mean multiplier is  E[z_i] = (alpha + n_i) / (alpha + e_i).

 4. Probability of leaving within the horizon h given survival to tenure t:
        p_i = 1 - exp( - z_i * [ L(t + h) - L(t) ] )

Other components: personal-norm ratio (shrunk), no-recent-promotion, stalled role (ECDF vs same-seniority peers),
between-roles gap, and employer headcount trend (LIVE ONLY: Company Enrichment has no as-of-T series, leak #7).
M in [0,1] is the weighted mean of the available components.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from core.dates import DAYS_PER_MONTH, months_between
from core.models import MaskedProfile, Role
from core.text import normalize

PRIOR_BETA = 2.0          # log-logistic shape prior (hazard peaks, then falls)


# ------------------------------------------------------------------------------- spells
@dataclass
class Spell:
    key: str
    start: "object"
    end: "object | None"
    n_roles: int
    last_role_start: "object"
    last_role: Role | None = None


def _emp_key(r: Role) -> str:
    return str(r.company_id) if r.company_id is not None else normalize(r.company_name) or f"?{r.idx}"


def employer_spells(mp: MaskedProfile, merge_gap_months: float = 2.0) -> list[Spell]:
    """Chronological employer spells from masked roles (promotions inside one employer are merged)."""
    roles = sorted((r for r in mp.roles if r.start is not None), key=lambda r: (r.start.date, r.idx))
    spells: list[Spell] = []
    for r in roles:
        s, e = r.start.date, (r.end.date if r.end else None)
        if spells and spells[-1].key == _emp_key(r):
            prev = spells[-1]
            prev_end = prev.end
            gap = months_between(prev_end, s) if prev_end is not None else 0.0
            if prev_end is None or gap <= merge_gap_months:
                prev.end = None if e is None else (e if prev_end is None or e > prev_end else prev_end)
                prev.n_roles += 1
                prev.last_role_start, prev.last_role = s, r
                continue
        spells.append(Spell(_emp_key(r), s, e, 1, s, r))
    return spells


def spell_durations(mp: MaskedProfile, merge_gap_months: float = 2.0) -> list[tuple[float, bool]]:
    """[(months, event)] with event=True if the spell ended before T, False if censored at T."""
    out = []
    for sp in employer_spells(mp, merge_gap_months):
        end = sp.end or mp.T
        d = months_between(sp.start, end)
        if 0.5 <= d <= 600:
            out.append((d, sp.end is not None))
    return out


# ------------------------------------------------------------------------------- baseline hazard models
class LogLogistic:
    name = "loglogistic"

    def __init__(self, a: float, b: float):
        self.a, self.b = a, b

    def cum_hazard(self, t):
        t = np.maximum(np.asarray(t, dtype=float), 1e-9)
        return np.log1p((t / self.a) ** self.b)

    def hazard(self, t):
        t = np.maximum(np.asarray(t, dtype=float), 1e-9)
        x = (t / self.a) ** self.b
        return (self.b / self.a) * (t / self.a) ** (self.b - 1) / (1 + x)

    def survival(self, t):
        return np.exp(-self.cum_hazard(t))

    def peak(self) -> float | None:
        return self.a * ((self.b - 1) / (self.b + 1)) ** (1 / self.b) if self.b > 1 else None

    def params(self) -> dict:
        return {"alpha_months": round(self.a, 3), "beta": round(self.b, 3)}


class Weibull:
    name = "weibull"

    def __init__(self, lam: float, k: float):
        self.lam, self.k = lam, k

    def cum_hazard(self, t):
        return (np.maximum(np.asarray(t, dtype=float), 1e-9) / self.lam) ** self.k

    def hazard(self, t):
        t = np.maximum(np.asarray(t, dtype=float), 1e-9)
        return (self.k / self.lam) * (t / self.lam) ** (self.k - 1)

    def survival(self, t):
        return np.exp(-self.cum_hazard(t))

    def peak(self) -> float | None:
        return None                                   # monotone hazard

    def params(self) -> dict:
        return {"lambda_months": round(self.lam, 3), "k": round(self.k, 3)}


def _nll(theta: np.ndarray, t: np.ndarray, d: np.ndarray, family: str, mu: tuple[float, float], sig: tuple[float, float]) -> float:
    la, lb = theta
    a, b = math.exp(la), math.exp(lb)
    x = (t / a) ** b
    if family == "loglogistic":
        log_h = math.log(b / a) + (b - 1) * np.log(t / a) - np.log1p(x)
        cum = np.log1p(x)
    else:
        log_h = math.log(b / a) + (b - 1) * np.log(t / a)
        cum = x
    nll = -(d * log_h).sum() + cum.sum()
    nll += 0.5 * ((la - mu[0]) / sig[0]) ** 2 + 0.5 * ((lb - mu[1]) / sig[1]) ** 2     # MAP prior
    return float(nll) if np.isfinite(nll) else 1e12


@dataclass
class SurvivalFit:
    model: LogLogistic | Weibull
    family: str
    n_spells: int
    n_events: int
    shrunk_to_prior: bool
    aic: float
    km_t: list[float] = field(default_factory=list)
    km_s: list[float] = field(default_factory=list)
    median_tenure_at_exit: float | None = None
    converged: bool = True

    def summary(self) -> dict:
        peak = self.model.peak()
        return {"family": self.family, **self.model.params(), "n_spells": self.n_spells, "n_events": self.n_events,
                "hazard_peak_months": None if peak is None else round(peak, 1), "shrunk_to_prior": self.shrunk_to_prior,
                "aic": round(self.aic, 1), "median_tenure_at_exit_months": None if self.median_tenure_at_exit is None else round(self.median_tenure_at_exit, 1),
                "converged": self.converged}


def kaplan_meier(t: np.ndarray, d: np.ndarray) -> tuple[list[float], list[float]]:
    order = np.argsort(t)
    t, d = t[order], d[order]
    times, surv, s, at_risk = [], [], 1.0, len(t)
    i = 0
    while i < len(t):
        j, deaths = i, 0
        while j < len(t) and t[j] == t[i]:
            deaths += int(d[j])
            j += 1
        if deaths:
            s *= 1.0 - deaths / at_risk
            times.append(float(t[i]))
            surv.append(float(s))
        at_risk -= j - i
        i = j
    return times, surv


def fit_survival(spells: list[tuple[float, bool]], family: str = "loglogistic", prior_peak: float = 27.0,
                 min_events: int = 25) -> SurvivalFit:
    t = np.array([max(d, 0.5) for d, _ in spells], dtype=float)
    e = np.array([1.0 if ev else 0.0 for _, ev in spells])
    n_events = int(e.sum())
    if family == "loglogistic":
        a0 = prior_peak / (((PRIOR_BETA - 1) / (PRIOR_BETA + 1)) ** (1 / PRIOR_BETA))
        mu = (math.log(a0), math.log(PRIOR_BETA))
        sig = (0.6, 0.45)
    else:
        mu = (math.log(48.0), math.log(1.6))
        sig = (0.6, 0.45)
    converged = True
    if len(t) >= 3:
        res = minimize(_nll, x0=np.array(mu), args=(t, e, family, mu, sig), method="L-BFGS-B",
                       bounds=[(math.log(3.0), math.log(400.0)), (math.log(0.6), math.log(6.0))])
        la, lb = res.x
        converged = bool(res.success)
        nll = float(res.fun)
    else:
        la, lb, nll = mu[0], mu[1], 0.0
    a, b = math.exp(la), math.exp(lb)
    model = LogLogistic(a, b) if family == "loglogistic" else Weibull(a, b)
    km_t, km_s = kaplan_meier(t, e)
    exits = t[e == 1]
    return SurvivalFit(model, family, len(t), n_events, n_events < min_events, 2 * 2 + 2 * nll, km_t, km_s,
                       float(np.median(exits)) if len(exits) else None, converged)


# ------------------------------------------------------------------------------- features
@dataclass
class MoveDiag:
    months_in_role: float | None = None
    months_at_employer: float | None = None
    p_move: float | None = None
    frailty: float | None = None
    n_past_spells: int = 0
    notes: list[str] = field(default_factory=list)


class MovabilityModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mc = cfg["movability"]
        self.fit_: SurvivalFit | None = None
        self.peer_months: dict[int, np.ndarray] = {}
        self.pop_median: float = 30.0

    def fit(self, profiles: list[MaskedProfile]) -> SurvivalFit:
        spells: list[tuple[float, bool]] = []
        peers: dict[int, list[float]] = {}
        gap = float(self.mc["merge_gap_months"])
        for mp in profiles:
            spells += spell_durations(mp, gap)
            top = next((r for r in mp.roles if r.end is None), None)
            if top is not None and top.start is not None:
                peers.setdefault(mp.norm.seniority, []).append(max(0.0, months_between(top.start.date, mp.T)))
        self.peer_months = {k: np.sort(np.array(v)) for k, v in peers.items()}
        self.fit_ = fit_survival(spells, str(self.mc["family"]), float(self.mc["prior_peak_months"]), int(self.mc["min_events_for_fit"]))
        done = [d for d, ev in spells if ev]
        self.pop_median = float(np.median(done)) if done else 30.0
        return self.fit_

    # ---- one candidate --------------------------------------------------------------------
    def features(self, mp: MaskedProfile, headcount_growth: dict | None = None) -> tuple[dict[str, float | None], MoveDiag]:
        assert self.fit_ is not None, "call fit() first"
        m = self.fit_.model
        gap_m = float(self.mc["merge_gap_months"])
        h = float(self.mc["horizon_days"]) / DAYS_PER_MONTH
        alpha_f = float(self.mc["frailty_alpha"])
        spells = employer_spells(mp, gap_m)
        diag = MoveDiag()
        cur = spells[-1] if spells and spells[-1].end is None else None
        past = [s for s in spells if s.end is not None]
        diag.n_past_spells = len(past)
        feats: dict[str, float | None] = {}

        if cur is not None:
            t_emp = max(0.5, months_between(cur.start, mp.T))
            diag.months_at_employer = t_emp
            top = cur.last_role
            diag.months_in_role = max(0.0, months_between(cur.last_role_start, mp.T))
            exposure = float(m.cum_hazard(t_emp))
            for s in past:
                exposure += float(m.cum_hazard(max(0.5, months_between(s.start, s.end))))
            n_i = len(past)
            z = (alpha_f + n_i) / (alpha_f + exposure)
            lam = max(0.0, float(m.cum_hazard(t_emp + h) - m.cum_hazard(t_emp)))
            p = 1.0 - math.exp(-z * lam)
            diag.p_move, diag.frailty = p, z
            feats["p_move"] = p / (p + 0.10)
            # personal norm, shrunk toward the population median (kappa = 2 pseudo-spells)
            if past:
                durs = [months_between(s.start, s.end) for s in past]
                med = (len(durs) * float(np.median(durs)) + 2.0 * self.pop_median) / (len(durs) + 2.0)
                ratio = t_emp / max(med, 3.0)
                feats["personal"] = 1.0 / (1.0 + math.exp(-2.0 * (ratio - 1.0)))
            else:
                feats["personal"] = 0.5
                diag.notes.append("no past spells: personal norm unavailable")
            promo = cur.n_roles >= 2 and months_between(cur.last_role_start, mp.T) <= float(self.mc["recent_promotion_months"])
            feats["no_recent_promo"] = 0.0 if promo else 1.0
            peers = self.peer_months.get(mp.norm.seniority)
            if peers is not None and len(peers) >= 8 and diag.months_in_role is not None:
                feats["stalled"] = float(np.searchsorted(peers, diag.months_in_role, side="right") / len(peers))
            else:
                feats["stalled"] = 0.5
            feats["between_roles"] = 0.0
        else:
            # between jobs at T (or no usable history)
            gap = mp.between_roles_months
            feats["between_roles"] = float(math.exp(-gap / 6.0)) if (gap is not None and gap <= 12) else 0.2
            feats["p_move"] = 0.5
            feats["personal"] = 0.5
            feats["no_recent_promo"] = 1.0
            feats["stalled"] = 0.5
            diag.p_move = None
            diag.notes.append("between roles at T" if gap is not None else "no history")
        # headcount trend: LIVE ONLY
        if headcount_growth and headcount_growth.get("six_months") is not None:
            g = float(headcount_growth["six_months"])
            feats["headcount"] = 1.0 / (1.0 + math.exp(g / 8.0))       # shrinking employer -> high
        else:
            feats["headcount"] = None
        return feats, diag


def combine_move(feats: dict[str, float | None], inner: dict[str, float]) -> float:
    num = den = 0.0
    for k, w in inner.items():
        v = feats.get(k)
        if v is not None and w > 0:
            num += w * v
            den += w
    return num / den if den else 0.0
