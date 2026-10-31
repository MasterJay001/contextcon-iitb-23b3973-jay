"""Outreach: short drafts built ONLY from verified facts, queued for human approval. Nothing is sent automatically.

Rules (CLAUDE.md #3, #9):
  * the model never sees names - it writes {{CANDIDATE_FIRST_NAME}} / {{COLLEAGUE_NAME}}; the renderer fills them from
    the raw store when the card is displayed;
  * facts used must verify verbatim in the masked record; unverified claims never reach a draft;
  * no contact data is produced or used; the recipient of an approved demo send is a TEAM ALLOWLIST address, never the candidate;
  * sends are disabled by default (`outreach.send_enabled: false`) and need explicit approval + allowlist.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import smtplib
from email.message import EmailMessage
from pathlib import Path

from pydantic import BaseModel, Field

from agents.artifacts import CaseArtifacts
from core.config import ROOT
from core.models import JobSpec, Outreach, Scored
from core.render import render_masked
from core.text import first_name
from llm.client import LLM

SYSTEM = """You write a short recruiting outreach message for a human recruiter to approve. Follow the style guide strictly.
You are given ONLY verified facts about the candidate (exact phrases from their history). Use one or two of them. Do not add
facts, achievements, compensation, urgency or guesses about their situation. Use the placeholder {{CANDIDATE_FIRST_NAME}} for the
greeting (you never know real names) and, only if a path-in is provided, {{COLLEAGUE_NAME}} for the colleague.
Plain text, 60-90 words, one clear ask (a 20-minute chat). No emojis, no links, no contact details."""


class DraftOut(BaseModel):
    subject: str
    body: str
    facts_used: list[str] = Field(description="the verified fact phrases you used, copied exactly")


_CONTACT = re.compile(r"(@[A-Za-z0-9_.-]+\.[A-Za-z]{2,})|(https?://)|(\+?\d[\d\s().-]{8,}\d)", re.IGNORECASE)


def _template(spec: JobSpec, facts: list[str], employer: str, path: str | None, company: str | None) -> DraftOut:
    fact = facts[0] if facts else f"your background as a {spec.norm_title}"
    where = f" in {spec.location}" if spec.location else ""
    co = f" at {company}" if company else ""
    body = (f"Hi {{{{CANDIDATE_FIRST_NAME}}}},\n\nI came across your work ({fact}{' at ' + employer if employer and employer not in fact else ''}). "
            f"We are hiring a {spec.norm_title}{where}{co}, and that experience looks very relevant to what the role needs.\n"
            + (f"{path}\n" if path else "") +
            "Would you be open to a 20-minute chat this week?\n\nThanks")
    return DraftOut(subject=f"{spec.norm_title}{co} - quick chat?", body=body, facts_used=facts[:2])


def draft_for(llm: LLM, art: CaseArtifacts, s: Scored, max_words: int) -> Outreach:
    spec = art.spec
    mp = next(m for m in art.pool if m.person_id == s.person_id)
    rendered = render_masked(mp)
    det = art.fit_details.get(s.person_id)
    facts = []
    for ref, q in (det.evidence if det else []):
        if rendered.verify(q, ref) and q not in facts:
            facts.append(q)
    cur = next((r for r in mp.roles if r.end is None), mp.roles[0] if mp.roles else None)
    employer = (cur.company_name or "") if cur else ""
    if not facts and cur is not None and cur.title:
        facts = [cur.title]
    diag = art.conn_diag.get(s.person_id)
    path = None
    if diag and diag.paths:
        p = diag.paths[0]
        path = f"{{{{COLLEAGUE_NAME}}}}, who worked with you at {p.employer} ({p.start}-{p.end}), now works with us."
    user = (f"ROLE: {spec.norm_title} ({spec.seniority}){', ' + spec.location if spec.location else ''}"
            f"{', company: ' + spec.company_name if spec.company_name else ''}\nVERIFIED FACTS (use 1-2):\n" + "\n".join(f"- {f}" for f in facts[:4]) +
            f"\nCANDIDATE CURRENT EMPLOYER: {employer}\nPATH IN: {path or 'none'}\nMAX WORDS: {max_words}")
    res = llm.structured("outreach", SYSTEM, user, DraftOut, mock=lambda: _template(spec, facts, employer, path, spec.company_name))
    d = res.parsed if res.ok else _template(spec, facts, employer, path, spec.company_name)
    bad = (_CONTACT.search(d.body) is not None or len(d.body.split()) > int(max_words * 1.5) or
           any(not rendered.verify(f) for f in d.facts_used if f))
    if bad:                                    # guard failed: fall back to the template (always compliant)
        d = _template(spec, facts, employer, path, spec.company_name)
    return Outreach(subject=d.subject[:90], body=d.body, facts_used=d.facts_used, requires_approval=True, status="queued")


def render_text(o: Outreach, candidate_name: str | None, colleague_name: str | None = None) -> str:
    body = o.body.replace("{{CANDIDATE_FIRST_NAME}}", first_name(candidate_name) or "there")
    body = body.replace("{{COLLEAGUE_NAME}}", colleague_name or "a former colleague")
    return body


# ------------------------------------------------------------------------------- approval queue
class OutreachQueue:
    """Append-only JSONL queue + approve/send state machine. Sending is gated twice: config flag and allowlist."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dir = ROOT / str(cfg["paths"]["outbox"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "queue.jsonl"

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _write(self, rows: list[dict]) -> None:
        self.path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")

    def enqueue(self, run_id: str, person_id: int, rank: int, o: Outreach, rendered_body: str) -> str:
        rows = self._read()
        item_id = f"{run_id}-{person_id}"
        rows = [r for r in rows if r["id"] != item_id]
        rows.append({"id": item_id, "run": run_id, "person_id": person_id, "rank": rank, "subject": o.subject, "body": rendered_body,
                     "status": "queued", "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        self._write(rows)
        return item_id

    def list(self, status: str | None = None) -> list[dict]:
        return [r for r in self._read() if status is None or r["status"] == status]

    def set_status(self, item_id: str, status: str) -> dict | None:
        rows = self._read()
        hit = None
        for r in rows:
            if r["id"] == item_id:
                r["status"], hit = status, r
        self._write(rows)
        return hit

    def allowlist(self) -> list[str]:
        return [a.strip().lower() for a in os.getenv("TTR_SEND_ALLOWLIST", "").split(",") if a.strip()]

    def send(self, item_id: str, to_addr: str) -> tuple[bool, str]:
        """Send an APPROVED item to a team-allowlist address. Returns (ok, message)."""
        if not bool(self.cfg["outreach"]["send_enabled"]) and os.getenv("TTR_ENABLE_SEND") != "1":
            return False, "sends are disabled (outreach.send_enabled: false). Set it to true or TTR_ENABLE_SEND=1 to demo an approved send."
        if to_addr.strip().lower() not in self.allowlist():
            return False, f"{to_addr} is not on the team allowlist (TTR_SEND_ALLOWLIST): candidates are never contacted during the event"
        item = next((r for r in self._read() if r["id"] == item_id), None)
        if item is None or item["status"] != "approved":
            return False, "item is not approved"
        msg = EmailMessage()
        msg["Subject"], msg["To"], msg["From"] = f"[DEMO] {item['subject']}", to_addr, os.getenv("SMTP_FROM", "recruiter-demo@localhost")
        msg.set_content("(Demo send to a teammate - not a real candidate.)\n\n" + item["body"])
        sent_dir = self.dir / "sent"
        sent_dir.mkdir(exist_ok=True)
        host = os.getenv("SMTP_HOST", "")
        try:
            if host:
                with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=20) as s:
                    s.starttls()
                    if os.getenv("SMTP_USER"):
                        s.login(os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASS", ""))
                    s.send_message(msg)
                where = "via SMTP"
            else:
                (sent_dir / f"{item_id}.eml").write_bytes(bytes(msg))
                where = f"written to outbox/sent/{item_id}.eml (no SMTP configured)"
        except Exception as exc:
            return False, f"send failed: {type(exc).__name__}: {exc}"
        self.set_status(item_id, "sent")
        return True, f"sent to {to_addr} {where}"
