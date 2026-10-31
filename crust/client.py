"""Crustdata HTTP client: auth, token-bucket rate limits, retries, disk cache, credit metering and
fault injection (rule 5, 8, 10).

Everything that can go wrong with the network is handled *here* so the agents above never see a
raw exception they cannot reason about:

    429  -> wait x-ratelimit-reset (or exponential backoff + jitter), retry, then CrustRateLimited
    5xx / timeout / connection error -> retry (twice), then CrustUnavailable (caller degrades)
    402  -> CrustCreditsExhausted (terminal, per the docs)
    403  -> CrustPermissionError with `denied_fields` so the caller can narrow `fields`
    400  -> CrustBadRequest (message preserved: it lists valid fields)
    401  -> CrustAuthError

A transport is injected, so the same client runs against live Crustdata (`HttpTransport`) or the
offline simulator (`crust.sim.SimTransport`) - the code path above the transport is identical.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NamedTuple, Protocol

from core.config import ROOT
from core.events import log_event


# ------------------------------------------------------------------------------- exceptions
class CrustError(Exception):
    status: int | None = None

    def __init__(self, message: str, *, status: int | None = None, metadata: dict | None = None):
        super().__init__(message)
        self.status = status
        self.metadata = metadata or {}


class CrustBadRequest(CrustError): ...
class CrustAuthError(CrustError): ...
class CrustNotFound(CrustError): ...
class CrustCreditsExhausted(CrustError): ...
class CrustRateLimited(CrustError): ...
class CrustUnavailable(CrustError): ...
class CreditCapReached(CrustError): ...


class CrustPermissionError(CrustError):
    @property
    def denied_fields(self) -> list[str]:
        d = self.metadata.get("denied_fields") or []
        return [str(x) for x in d]


# ------------------------------------------------------------------------------- transports
class TransportResponse(NamedTuple):
    status: int
    headers: dict
    json: Any
    elapsed: float


class TransportTimeout(Exception): ...
class TransportError(Exception): ...


class Transport(Protocol):
    def request(self, method: str, path: str, *, headers: dict, json_body: Any | None, timeout: float) -> TransportResponse: ...


class HttpTransport:
    """Real HTTPS transport (httpx). Imported lazily so offline tests never need the network stack."""

    def __init__(self, base_url: str):
        import httpx

        self._httpx = httpx
        self._client = httpx.Client(base_url=base_url, timeout=60.0)

    def request(self, method: str, path: str, *, headers: dict, json_body: Any | None, timeout: float) -> TransportResponse:
        t0 = time.monotonic()
        try:
            r = self._client.request(method, path, headers=headers, json=json_body, timeout=timeout)
        except self._httpx.TimeoutException as exc:
            raise TransportTimeout(str(exc)) from exc
        except self._httpx.TransportError as exc:
            raise TransportError(str(exc)) from exc
        try:
            body = r.json()
        except ValueError:
            body = {"error": {"type": "non_json_response", "message": r.text[:300]}}
        return TransportResponse(r.status_code, {k.lower(): v for k, v in r.headers.items()}, body, time.monotonic() - t0)


# ------------------------------------------------------------------------------- fault injection
class FaultInjector:
    """Deterministic fault injection for the robustness suite.

    Spec string (env FAULT_INJECT): `429:0.33,5xx:0.2,empty:0.5,llm_json:0.25,credit_cap:50`
    Each key's value is a probability (credit_cap is a number of credits). Decisions come from a
    seeded RNG so a failing run is reproducible.
    """

    def __init__(self, spec: dict[str, float] | None = None, seed: int = 13):
        self.spec = {k: float(v) for k, v in (spec or {}).items()}
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}

    @classmethod
    def from_env(cls) -> "FaultInjector":
        return cls.from_string(os.getenv("FAULT_INJECT", ""), seed=int(os.getenv("FAULT_SEED", "13")))

    @classmethod
    def from_string(cls, s: str, seed: int = 13) -> "FaultInjector":
        spec: dict[str, float] = {}
        for part in (s or "").split(","):
            part = part.strip()
            if not part:
                continue
            k, _, v = part.partition(":")
            try:
                spec[k.strip()] = float(v) if v else 1.0
            except ValueError:
                continue
        return cls(spec, seed)

    @property
    def active(self) -> bool:
        return any(v > 0 for v in self.spec.values())

    def hit(self, kind: str) -> bool:
        p = self.spec.get(kind, 0.0)
        if p <= 0:
            return False
        with self._lock:
            fired = self._rng.random() < p
            if fired:
                self.counts[kind] = self.counts.get(kind, 0) + 1
        return fired


# ------------------------------------------------------------------------------- rate limiting
class RateLimiter:
    """Token bucket per endpoint key. Small burst, even refill (the docs warn against bursting)."""

    def __init__(self, per_minute: dict[str, int], *, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self._rpm = dict(per_minute)
        self._tokens: dict[str, float] = {}
        self._stamp: dict[str, float] = {}
        self._lock = threading.Lock()
        self._sleep, self._clock = sleep, clock

    def acquire(self, key: str) -> float:
        """Block until a token is available; returns seconds waited."""
        rpm = max(1, int(self._rpm.get(key, 30)))
        cap = max(1.0, rpm / 6.0)
        rate = rpm / 60.0
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                tokens = self._tokens.get(key, cap)
                last = self._stamp.get(key, now)
                tokens = min(cap, tokens + (now - last) * rate)
                if tokens >= 1.0:
                    self._tokens[key], self._stamp[key] = tokens - 1.0, now
                    return waited
                self._tokens[key], self._stamp[key] = tokens, now
                need = (1.0 - tokens) / rate
            self._sleep(need)
            waited += need


# ------------------------------------------------------------------------------- credits
@dataclass
class CreditMeter:
    """Estimated spend (documented per-result prices) with hard caps (rule 8).

    `estimated` counts only requests that actually hit the network; cache hits add to `saved`.
    """

    cap_run: float = 6000.0
    cap_case: float = 160.0
    estimated: float = 0.0
    saved: float = 0.0
    case_start: float = 0.0
    balance_start: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def begin_case(self) -> None:
        with self._lock:
            self.case_start = self.estimated

    @property
    def case_spent(self) -> float:
        return self.estimated - self.case_start

    def remaining_case(self) -> float:
        return max(0.0, min(self.cap_case - self.case_spent, self.cap_run - self.estimated))

    def charge(self, credits: float) -> None:
        with self._lock:
            self.estimated += credits

    def credit_saved(self, credits: float) -> None:
        with self._lock:
            self.saved += credits

    def affordable_results(self, per_result: float) -> int:
        if per_result <= 0:
            return 10**9
        return int(self.remaining_case() // per_result)


@dataclass
class CallRecord:
    method: str
    path: str
    status: int
    cached: bool
    credits: float
    seconds: float
    attempts: int


@dataclass
class CrustResponse:
    json: Any
    status: int
    cached: bool
    credits: float
    attempts: int


# ------------------------------------------------------------------------------- client
def _canonical(body: Any) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)


class CrustClient:
    def __init__(
        self,
        cfg: dict,
        transport: Transport | None = None,
        *,
        mode: str = "backtest",
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        faults: FaultInjector | None = None,
        meter: CreditMeter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        c = cfg["crust"]
        self.mode = mode
        self.api_version = str(c["api_version"])
        self.timeout = float(c["timeout_s"])
        self.max_attempts = int(c["max_attempts"])
        self.backoff_base = float(c["backoff_base_s"])
        self.backoff_max = float(c["backoff_max_s"])
        self.faults = faults if faults is not None else FaultInjector.from_env()
        self.meter = meter or CreditMeter(cap_run=float(c["credit_cap_run"]), cap_case=float(c["credit_cap_case"]))
        if self.faults.spec.get("credit_cap"):
            self.meter.cap_run = self.meter.cap_case = float(self.faults.spec["credit_cap"])
        self.limiter = RateLimiter(dict(c["rate_limit_per_min"]), sleep=sleep, clock=clock)
        self._sleep = sleep
        self.api_key = api_key if api_key is not None else os.getenv("CRUSTDATA_API_KEY", "")
        self.transport = transport
        self._base_url = str(c["base_url"])
        self.calls: list[CallRecord] = []
        self.cache_hits = 0
        self.cache_misses = 0
        self._cache = None
        if use_cache:
            import diskcache

            root = Path(cache_dir) if cache_dir else ROOT / str(cfg["paths"]["cache"]) / "http"
            root.mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(str(root))
        ttl = c["cache_ttl_hours"]
        self.default_ttl_hours = float(ttl["live"] if mode == "live" else ttl["backtest"])
        self.per_result = dict(c["credits_per_result"])
        self._rng = random.Random(1234)

    # ---- public API ---------------------------------------------------------------------
    def post(self, path: str, body: dict, *, limiter: str, cache: bool = True, ttl_hours: float | None = None) -> CrustResponse:
        return self._request("POST", path, body, limiter=limiter, cache=cache, ttl_hours=ttl_hours)

    def get(self, path: str, *, limiter: str) -> CrustResponse:
        return self._request("GET", path, None, limiter=limiter, cache=False, ttl_hours=None)

    def credits_balance(self) -> float | None:
        try:
            r = self.get("/account/credits", limiter="credits")
            return float(r.json["account"]["credits"])
        except (CrustError, KeyError, TypeError, ValueError) as exc:
            log_event("credits_balance_unavailable", error=str(exc))
            return None

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()

    # ---- internals ----------------------------------------------------------------------
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "x-api-version": self.api_version,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _key(self, method: str, path: str, body: Any) -> str:
        raw = f"{method} {path} {self.api_version} {_canonical(body)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def estimate_cost(self, path: str, body: Any, payload: Any) -> float:
        if not isinstance(payload, (dict, list)):
            return 0.0
        if path.startswith("/person/search") and not path.endswith("autocomplete"):
            n = len(payload.get("profiles", [])) if isinstance(payload, dict) else 0
            fields = (body or {}).get("fields")
            heavy = (not fields) or any(str(f).split(".")[0] in {"experience", "education"} for f in fields)
            return n * float(self.per_result["experience_education" if heavy else "base"])
        if path.startswith("/job/search"):
            n = len(payload.get("job_listings", [])) if isinstance(payload, dict) else 0
            return n * float(self.per_result["base"])
        if path.startswith("/company/enrich"):
            n = sum(1 for item in payload if isinstance(item, dict) and item.get("matches")) if isinstance(payload, list) else 0
            return 2.0 * n
        return 0.0

    def _raise_for_status(self, status: int, payload: Any) -> None:
        err = (payload or {}).get("error", {}) if isinstance(payload, dict) else {}
        msg = str(err.get("message") or err.get("type") or f"HTTP {status}")
        meta = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
        if status == 400:
            raise CrustBadRequest(msg, status=status, metadata=meta)
        if status == 401:
            raise CrustAuthError(f"{msg} (check CRUSTDATA_API_KEY in .env)", status=status, metadata=meta)
        if status == 402:
            raise CrustCreditsExhausted(msg, status=status, metadata=meta)
        if status == 403:
            raise CrustPermissionError(msg, status=status, metadata=meta)
        if status == 404:
            raise CrustNotFound(msg, status=status, metadata=meta)
        raise CrustError(msg, status=status, metadata=meta)

    def _send_once(self, method: str, path: str, body: Any) -> TransportResponse:
        if self.faults.hit("429"):
            log_event("fault_injected", fault="429", path=path)
            return TransportResponse(429, {"x-ratelimit-reset": "1"}, {"error": {"type": "rate_limit_error", "message": "injected 429"}}, 0.0)
        if self.faults.hit("5xx"):
            log_event("fault_injected", fault="5xx", path=path)
            if self._rng.random() < 0.5:
                raise TransportTimeout("injected timeout")
            return TransportResponse(503, {}, {"error": {"type": "server_error", "message": "injected 503"}}, 0.0)
        if self.transport is None:
            if not self.api_key:
                raise CrustAuthError("CRUSTDATA_API_KEY is not set (put it in .env), or run with backend.crust=sim")
            self.transport = HttpTransport(self._base_url)
        resp = self.transport.request(method, path, headers=self._headers(), json_body=body, timeout=self.timeout)
        if resp.status == 200 and self.faults.hit("empty") and isinstance(resp.json, dict):
            for k in ("profiles", "job_listings"):
                if k in resp.json:
                    log_event("fault_injected", fault="empty", path=path)
                    resp = TransportResponse(200, resp.headers, {**resp.json, k: [], "total_count": 0, "next_cursor": None}, resp.elapsed)
        return resp

    def _request(self, method: str, path: str, body: Any, *, limiter: str, cache: bool, ttl_hours: float | None) -> CrustResponse:
        key = self._key(method, path, body) if (cache and self._cache is not None) else None
        if key is not None:
            hit = self._cache.get(key)
            if hit is not None:
                self.cache_hits += 1
                self.meter.credit_saved(hit.get("credits", 0.0))
                self.calls.append(CallRecord(method, path, 200, True, 0.0, 0.0, 0))
                return CrustResponse(hit["json"], 200, True, 0.0, 0)
        self.cache_misses += 1
        attempts, t0 = 0, time.monotonic()
        last_exc: Exception | None = None
        while attempts < self.max_attempts:
            attempts += 1
            self.limiter.acquire(limiter)
            try:
                resp = self._send_once(method, path, body)
            except (TransportTimeout, TransportError) as exc:
                last_exc = exc
                log_event("crust_retry", path=path, reason=type(exc).__name__, attempt=attempts)
                if attempts >= min(self.max_attempts, 3):
                    break
                self._sleep(self._backoff(attempts))
                continue
            if resp.status == 429:
                reset = resp.headers.get("x-ratelimit-reset") or resp.headers.get("retry-after")
                wait = float(reset) if reset and str(reset).replace(".", "", 1).isdigit() else self._backoff(attempts)
                wait = min(wait + self._rng.random() * 0.5, self.backoff_max)
                log_event("crust_retry", path=path, reason="429", attempt=attempts, wait_s=round(wait, 2))
                last_exc = CrustRateLimited("rate limited", status=429)
                self._sleep(wait)
                continue
            if resp.status >= 500:
                last_exc = CrustUnavailable(f"HTTP {resp.status}", status=resp.status)
                log_event("crust_retry", path=path, reason=str(resp.status), attempt=attempts)
                if attempts >= min(self.max_attempts, 3):
                    break
                self._sleep(self._backoff(attempts))
                continue
            if resp.status != 200:
                self._raise_for_status(resp.status, resp.json)
            credits = self.estimate_cost(path, body, resp.json)
            self.meter.charge(credits)
            if key is not None:
                ttl = (ttl_hours if ttl_hours is not None else self.default_ttl_hours) * 3600.0
                self._cache.set(key, {"json": resp.json, "credits": credits}, expire=ttl)
            self.calls.append(CallRecord(method, path, 200, False, credits, time.monotonic() - t0, attempts))
            return CrustResponse(resp.json, 200, False, credits, attempts)
        self.calls.append(CallRecord(method, path, 599, False, 0.0, time.monotonic() - t0, attempts))
        if isinstance(last_exc, CrustRateLimited):
            raise last_exc
        raise CrustUnavailable(f"{path} unavailable after {attempts} attempts: {last_exc}")

    def _backoff(self, attempt: int) -> float:
        return min(self.backoff_max, self.backoff_base * (2 ** (attempt - 1)) + self._rng.random())

    # ---- diagnostics --------------------------------------------------------------------
    def summary(self) -> dict:
        return {
            "requests": len(self.calls),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "credits_estimated": round(self.meter.estimated, 3),
            "credits_saved_by_cache": round(self.meter.saved, 3),
            "faults": dict(self.faults.counts),
        }
