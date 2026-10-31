"""The only module that talks to the Anthropic API.

Properties the rest of the system relies on:

* **Schema-guaranteed output**: `messages.parse(output_format=<pydantic model>)` (structured outputs); we still
  validate, and on a bad/corrupt payload re-ask ONCE with the validation error, then degrade (return None).
* **Determinism = cache**: `temperature` is rejected (HTTP 400) on Claude 5.x, so reruns are made identical by a
  request-hash disk cache (rule 5), not by temperature 0.
* **No names to the LLM** (rule 3): `forbid()` registers person names; any prompt containing one raises
  `NameLeakError` before a request is made. Prompts are built from MaskedProfile only, so this is belt and braces.
* **Never crash** (rule 10): refusals, truncation, network/rate errors and malformed JSON all end as
  `StructuredResult(parsed=None, error=...)` and an event; callers degrade to deterministic heuristics.
* **Cost is visible**: tokens and USD (config prices) per role.
* **Mock backend**: `llm.backend: mock` runs the pipeline offline with deterministic heuristics supplied by the agents.
  Mock outputs are labelled (`source="mock"`) and must never be presented as Claude results (rule 14).
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
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from core.config import ROOT, resolve_llm_backend
from core.events import log_event

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class NameLeakError(RuntimeError):
    """A person's name was about to be sent to the LLM (rule 3)."""


@dataclass
class StructuredResult:
    parsed: Any = None
    source: str = "none"             # anthropic | cache | mock | none
    error: str | None = None
    stop_reason: str | None = None
    attempts: int = 0
    model: str = ""

    @property
    def ok(self) -> bool:
        return self.parsed is not None


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    usd: float = 0.0
    failures: int = 0
    by_role: dict[str, dict] = field(default_factory=dict)

    def add(self, role: str, in_t: int, out_t: int, cr: int, cw: int, usd: float) -> None:
        self.calls += 1
        self.input_tokens += in_t
        self.output_tokens += out_t
        self.cache_read_tokens += cr
        self.cache_write_tokens += cw
        self.usd += usd
        r = self.by_role.setdefault(role, {"calls": 0, "usd": 0.0, "in": 0, "out": 0})
        r["calls"] += 1
        r["usd"] += usd
        r["in"] += in_t
        r["out"] += out_t

    def summary(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.cache_hits, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "usd": round(self.usd, 4), "failures": self.failures,
                "by_role": {k: {**v, "usd": round(v["usd"], 4)} for k, v in self.by_role.items()}}


class LLM:
    def __init__(self, cfg: dict, backend: str | None = None, *, cache_dir: str | Path | None = None,
                 use_cache: bool = True, faults: Any = None):
        self.cfg = cfg
        self.backend = backend or resolve_llm_backend(cfg)
        lc = cfg["llm"]
        self.models = dict(lc["models"])
        self.effort = dict(lc["effort"])
        self.max_tokens = dict(lc["max_tokens"])
        self.max_attempts = int(lc["max_attempts"])
        self.fallbacks = str(lc.get("fallbacks", "off")).lower() not in {"off", "false", "none", "0", ""}
        self.prices = dict(lc["usd_per_mtok"])
        self.usage = Usage()
        self._lock = threading.Lock()
        self._forbidden: list[str] = []
        self._client_obj = None
        self._cache = None
        self.faults = faults
        self._rng = random.Random(77)
        if use_cache and self.backend == "anthropic":
            import diskcache

            root = Path(cache_dir) if cache_dir else ROOT / str(cfg["paths"]["cache"]) / "llm"
            root.mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(str(root))
        self._ttl = float(lc.get("cache_ttl_hours", 8760)) * 3600.0

    # ---- guards -------------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.backend == "anthropic"

    def forbid(self, names: list[str | None]) -> None:
        """Register person names that must never appear in a prompt (full names only: avoids tripping on
        common first names that are also English words)."""
        with self._lock:
            for n in names:
                if n and len(n.strip()) >= 6 and " " in n.strip():
                    self._forbidden.append(n.strip().lower())

    def _check_prompt(self, *texts: str) -> None:
        blob = " ".join(texts).lower()
        for n in self._forbidden:
            if n in blob:
                log_event("llm_name_leak_blocked")
                raise NameLeakError("a person's name was about to be sent to the LLM")

    # ---- client -------------------------------------------------------------------------
    def _client(self):
        if self._client_obj is None:
            import anthropic

            self._client_obj = anthropic.Anthropic(max_retries=self.max_attempts, timeout=180.0)
        return self._client_obj

    def _key(self, *parts: Any) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def _cost(self, in_t: int, out_t: int, cr: int, cw: int) -> float:
        p = self.prices
        return (in_t * float(p["input"]) + out_t * float(p["output"]) + cr * float(p["cache_read"]) + cw * float(p["cache_write"])) / 1e6

    def _record(self, role: str, resp: Any) -> None:
        u = getattr(resp, "usage", None)
        if u is None:
            return
        in_t = int(getattr(u, "input_tokens", 0) or 0)
        out_t = int(getattr(u, "output_tokens", 0) or 0)
        cr = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cw = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        with self._lock:
            self.usage.add(role, in_t, out_t, cr, cw, self._cost(in_t, out_t, cr, cw))

    # ---- structured call ----------------------------------------------------------------
    def structured(self, role: str, system: str, user: str, schema: type[T], *, mock: Callable[[], T | dict] | None = None,
                   max_tokens: int | None = None, effort: str | None = None) -> StructuredResult:
        """One schema-validated LLM call. Never raises (except NameLeakError, which is a programming error)."""
        self._check_prompt(system, user)
        if self.backend != "anthropic":
            return self._mock(schema, mock)
        model = self.models.get(role, self.models.get("fit", "claude-opus-5-5"))
        effort = effort or self.effort.get(role, "medium")
        max_tokens = int(max_tokens or self.max_tokens.get(role, 8000))
        key = self._key("structured", model, role, system, user, schema.model_json_schema(), effort, max_tokens)
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                try:
                    parsed = schema.model_validate_json(hit)
                    with self._lock:
                        self.usage.cache_hits += 1
                    return StructuredResult(parsed, "cache", model=model)
                except ValidationError:
                    pass
        attempts, last_err, user_msg = 0, "", user
        while attempts < 2:
            attempts += 1
            try:
                resp = self._parse_call(model, system, user_msg, schema, effort, max_tokens)
            except Exception as exc:                       # network, auth, 400... never crash the pipeline
                last_err = f"{type(exc).__name__}: {str(exc)[:200]}"
                log_event("llm_call_failed", role=role, error=last_err)
                break
            self._record(role, resp)
            stop = getattr(resp, "stop_reason", None)
            if stop == "refusal":
                last_err = "refusal"
                log_event("llm_refusal", role=role, details=str(getattr(resp, "stop_details", ""))[:160])
                break
            text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
            text = self._maybe_corrupt(text)
            try:
                parsed = schema.model_validate_json(text)
            except ValidationError as exc:
                last_err = f"invalid JSON for {schema.__name__}: {str(exc)[:160]}"
                log_event("llm_bad_json", role=role, attempt=attempts, stop=stop)
                if stop == "max_tokens":
                    max_tokens = min(max_tokens * 2, 32000)
                user_msg = (user + "\n\n[Your previous reply could not be parsed: " + str(exc)[:300] +
                            "] Reply again with ONLY valid JSON matching the schema.")
                continue
            if self._cache is not None:
                self._cache.set(key, parsed.model_dump_json(), expire=self._ttl)
            return StructuredResult(parsed, "anthropic", stop_reason=stop, attempts=attempts, model=model)
        with self._lock:
            self.usage.failures += 1
        return StructuredResult(None, "none", error=last_err or "unknown", attempts=attempts, model=model)

    def _parse_call(self, model: str, system: str, user: str, schema: type[T], effort: str, max_tokens: int):
        kwargs: dict[str, Any] = dict(
            model=model, max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort}, output_format=schema,
        )
        client = self._client()
        if self.fallbacks:
            try:
                return client.beta.messages.parse(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
            except TypeError as exc:                      # SDK without the parameter: degrade once, permanently
                log_event("llm_fallbacks_unsupported", error=str(exc)[:120])
                self.fallbacks = False
            except Exception as exc:
                import anthropic

                if isinstance(exc, anthropic.BadRequestError):
                    log_event("llm_fallbacks_rejected", error=str(exc)[:160])
                    self.fallbacks = False
                else:
                    raise
        return client.messages.parse(**kwargs)

    def _maybe_corrupt(self, text: str) -> str:
        if self.faults is not None and self.faults.hit("llm_json"):
            log_event("fault_injected", fault="llm_json")
            return text[: max(1, len(text) // 2)]
        return text

    def _mock(self, schema: type[T], mock: Callable[[], T | dict] | None) -> StructuredResult:
        if mock is None:
            return StructuredResult(None, "none", error="mock backend has no heuristic for this call")
        try:
            out = mock()
            if self.faults is not None and self.faults.hit("llm_json"):
                log_event("fault_injected", fault="llm_json")
                raise ValueError("injected malformed mock output")
            parsed = out if isinstance(out, schema) else schema.model_validate(out)
            with self._lock:
                self.usage.calls += 0
            return StructuredResult(parsed, "mock", model="mock")
        except Exception as exc:
            log_event("llm_mock_failed", error=str(exc)[:160])
            with self._lock:
                self.usage.failures += 1
            return StructuredResult(None, "none", error=f"mock failed: {exc}", model="mock")

    # ---- one tool-use turn (Sourcer ReAct loop) -------------------------------------------
    def tool_turn(self, role: str, system: str, messages: list[dict], tools: list[dict], *,
                  effort: str | None = None, max_tokens: int | None = None) -> dict:
        """One assistant turn with tools. Returns {"content": [dict blocks], "stop_reason": str, "source": str}
        or {"error": str}. tool_choice stays `auto` (forced tool use is rejected on Claude 5.5 models)."""
        self._check_prompt(system, json.dumps(messages, default=str))
        if self.backend != "anthropic":
            return {"error": "mock backend has no tool loop"}
        model = self.models.get(role, self.models["fit"])
        effort = effort or self.effort.get(role, "low")
        max_tokens = int(max_tokens or self.max_tokens.get(role, 8000))
        key = self._key("tool_turn", model, role, system, messages, tools, effort, max_tokens)
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                with self._lock:
                    self.usage.cache_hits += 1
                return {**hit, "source": "cache"}
        try:
            resp = self._client().messages.create(
                model=model, max_tokens=max_tokens, system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages, tools=tools, output_config={"effort": effort},
            )
        except Exception as exc:
            log_event("llm_call_failed", role=role, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            with self._lock:
                self.usage.failures += 1
            return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        self._record(role, resp)
        if resp.stop_reason == "refusal":
            log_event("llm_refusal", role=role)
            return {"error": "refusal"}
        out = {"content": [b.model_dump(exclude_none=True) for b in resp.content], "stop_reason": resp.stop_reason, "source": "anthropic"}
        if self._cache is not None:
            self._cache.set(key, {"content": out["content"], "stop_reason": out["stop_reason"]}, expire=self._ttl)
        return out

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
