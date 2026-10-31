"""Tiny structured event log. Every degradation (retry, fallback, partial pool, repaired JSON...)
is recorded so (a) nothing fails silently and (b) NOTES.md can be written from facts."""
from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_EVENTS: list[dict] = []
_FILE: Path | None = None
_MAX_MEMORY = 5000


def set_event_file(path: str | Path | None) -> None:
    global _FILE
    _FILE = Path(path) if path else None
    if _FILE:
        _FILE.parent.mkdir(parents=True, exist_ok=True)


def log_event(kind: str, **fields: Any) -> dict:
    ev = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "kind": kind, **fields}
    with _LOCK:
        _EVENTS.append(ev)
        if len(_EVENTS) > _MAX_MEMORY:
            del _EVENTS[: _MAX_MEMORY // 5]
        if _FILE is not None:
            try:
                with open(_FILE, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(ev, default=str) + "\n")
            except OSError:
                pass
    return ev


def get_events(kind: str | None = None, since: int = 0) -> list[dict]:
    with _LOCK:
        evs = list(_EVENTS[since:])
    return [e for e in evs if kind is None or e["kind"] == kind]


def event_count() -> int:
    with _LOCK:
        return len(_EVENTS)


def clear_events() -> None:
    with _LOCK:
        _EVENTS.clear()
