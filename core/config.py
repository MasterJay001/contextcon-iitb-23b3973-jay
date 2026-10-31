"""Config loading: `config.yaml` is the single source of truth, `.env` supplies secrets."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Secrets live in .env (git-ignored). Existing environment variables win.
load_dotenv(ROOT / ".env", override=False)


class DotDict(dict):
    """dict with attribute access (recursively). Missing keys raise AttributeError so a typo
    in a config key fails loudly instead of silently using a default."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(f"config has no key '{name}'") from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self, default=str))


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return DotDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> DotDict:
    """Load config.yaml, apply env backend overrides, then explicit `overrides` (nested dict)."""
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8-sig") as fh:
        raw = yaml.safe_load(fh) or {}
    env_crust = os.getenv("TTR_CRUST_BACKEND")
    env_llm = os.getenv("TTR_LLM_BACKEND")
    if env_crust:
        raw.setdefault("backend", {})["crust"] = env_crust
    if env_llm:
        raw.setdefault("backend", {})["llm"] = env_llm
    if overrides:
        _deep_update(raw, copy.deepcopy(overrides))
    return _wrap(raw)


def config_hash(cfg: dict) -> str:
    """Stable hash of the whole config: stored next to results so a number can be traced to the
    exact weights/thresholds that produced it."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def today(cfg: dict | None = None) -> dt.date:
    now = None
    if cfg is not None:
        try:
            now = cfg["run"]["now"]
        except (KeyError, TypeError):
            now = None
    if now:
        return dt.date.fromisoformat(str(now))
    return dt.date.today()


def resolve_crust_backend(cfg: dict) -> str:
    choice = str(cfg["backend"]["crust"]).lower()
    if choice in {"live", "sim"}:
        return choice
    return "live" if os.getenv("CRUSTDATA_API_KEY") else "sim"


def anthropic_credentials_available() -> bool:
    """True if an env key/token is set or the `ant` CLI reports an active login profile.

    (The SDK constructor does NOT raise without credentials, so constructing a client proves nothing.)"""
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    ant = shutil.which("ant")
    if not ant:
        return False
    try:
        r = subprocess.run([ant, "auth", "status"], capture_output=True, text=True, timeout=8)
        out = (r.stdout + r.stderr).lower()
        return r.returncode == 0 and "no active" not in out and "not logged" not in out and "no credential" not in out
    except Exception:
        return False


def resolve_llm_backend(cfg: dict) -> str:
    choice = str(cfg["backend"]["llm"]).lower()
    if choice in {"anthropic", "mock"}:
        return choice
    return "anthropic" if anthropic_credentials_available() else "mock"


def abs_path(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p
