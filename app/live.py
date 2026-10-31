"""The service a customer uses: brief in -> five people worth a call, each with sourced reasons and a drafted outreach queued for approval.

    python -m app.live "We need a founding backend engineer in Mumbai who has shipped payments systems at a startup." --company "Acme"
    python -m app.live --queue                         # list drafts waiting for approval
    python -m app.live --approve <id> [--send-to team@example.com]

Same pipeline as the backtest (T = today). Sends are disabled unless outreach.send_enabled / TTR_ENABLE_SEND=1, and only go to
TTR_SEND_ALLOWLIST addresses. Candidates are never contacted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.orchestrator import build_context, run_live
from agents.outreach import OutreachQueue, render_text
from core.config import ROOT, load_config

console = Console()


def bar(x: float, width: int = 12) -> str:
    n = int(round(max(0.0, min(1.0, x)) * width))
    return "█" * n + "░" * (width - n)


def render_cards(out, ctx) -> str:
    md = []
    for c in out.result.cards:
        console.print(Panel.fit(f"[bold]#{c.rank}  {c.name or 'person ' + str(c.person_id)}[/bold]  ·  {c.current_role} @ {c.employer}\n"
                                f"id {c.person_id}  {c.profile_url or ''}\n"
                                f"F {bar(c.scores['F'])} {c.scores['F']:.2f}   M {bar(c.scores['M'])} {c.scores['M']:.2f}   C {bar(c.scores['C'])} {c.scores['C']:.2f}",
                                border_style="blue"))
        md.append(f"### #{c.rank} {c.name or c.person_id}: {c.current_role} @ {c.employer}\n")
        for title, claims in (("Why they fit", c.why_fit), ("Why now", c.why_now), ("Path in", c.path_in)):
            if claims:
                console.print(f"  [bold]{title}[/bold]")
                md.append(f"**{title}**")
                for cl in claims:
                    tick = "✓" if cl.verified else "?"
                    console.print(f"   {tick} {cl.text}  [dim]({cl.source.path})[/dim]")
                    md.append(f"- {cl.text} _(source: person {cl.source.person_id}, `{cl.source.path}`)_")
        if c.unverified:
            console.print("  [bold yellow]Unverified[/bold yellow]")
            md.append("**Unverified**")
            for u in c.unverified:
                console.print(f"   - {u}")
                md.append(f"- {u}")
        if c.outreach:
            colleague = next((p.text.split(" (now")[0] for p in c.path_in), None)
            body = render_text(c.outreach, c.name, colleague)
            console.print(Panel(f"[bold]{c.outreach.subject}[/bold]\n\n{body}\n\n[dim]status: queued for approval (nothing is sent)[/dim]", title="Outreach draft", border_style="green"))
            md.append(f"\n> **{c.outreach.subject}**\n>\n> " + body.replace("\n", "\n> ") + "\n")
    return "\n".join(md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("brief", nargs="?", help="free-text job brief")
    ap.add_argument("--company", default=None, help="hiring company (enables the Connection signal)")
    ap.add_argument("--assume", action="store_true", help="proceed despite a contradiction in the brief")
    ap.add_argument("--no-outreach", action="store_true")
    ap.add_argument("--json", default=None, help="write the full result to this path")
    ap.add_argument("--crust-backend", choices=["live", "sim"], default=None)
    ap.add_argument("--llm-backend", choices=["anthropic", "mock"], default=None)
    ap.add_argument("--queue", action="store_true")
    ap.add_argument("--approve", default=None)
    ap.add_argument("--send-to", default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    queue = OutreachQueue(cfg)
    if args.queue:
        t = Table("id", "rank", "status", "subject")
        for r in queue.list():
            t.add_row(r["id"], str(r["rank"]), r["status"], r["subject"])
        console.print(t)
        return 0
    if args.approve:
        queue.set_status(args.approve, "approved")
        console.print(f"approved {args.approve}")
        if args.send_to:
            ok, msg = queue.send(args.approve, args.send_to)
            console.print(("[green]" if ok else "[red]") + msg)
            return 0 if ok else 1
        return 0
    if not args.brief:
        ap.error("a brief is required")
    ctx = build_context(cfg, mode="live", crust_backend=args.crust_backend, llm_backend=args.llm_backend)
    banner = f"crust={ctx.crust_backend}  llm={ctx.llm_backend}"
    if ctx.crust_backend == "sim" or ctx.llm_backend == "mock":
        banner += "   [bold red]SYNTHETIC DATA / MOCK LLM: for verification only[/bold red]"
    console.print(f"[dim]{banner}[/dim]")
    brief = args.brief
    t0 = time.time()
    out = run_live(brief, ctx, company=args.company, assume=args.assume, with_outreach=not args.no_outreach)
    if out.blocked and sys.stdin.isatty():
        console.print(f"[yellow]Clarification needed:[/yellow] {out.blocked}")
        ans = input("> ").strip()
        if ans:
            out = run_live(f"{brief}. {ans}", ctx, company=args.company, assume=args.assume, with_outreach=not args.no_outreach)
    if out.blocked:
        console.print(f"[yellow]Needs clarification (nothing was searched):[/yellow] {out.blocked}")
        return 2
    spec = out.result.jobspec
    console.print(Panel.fit(f"[bold]{spec.seniority} {spec.norm_title}[/bold] · {spec.location or 'location not stated'}"
                            f"{' · remote OK' if spec.remote_ok else ''}\nskills: {', '.join(spec.skills) or '-'}   domain: {', '.join(spec.domain_keywords) or '-'}\n"
                            f"assumptions: {'; '.join(spec.assumptions) or 'none'}", title=f"Brief understood ({spec.interpreter})"))
    md = render_cards(out, ctx)
    for w in out.result.warnings:
        console.print(f"[yellow]! {w}[/yellow]")
    s = out.result.stage_seconds
    console.print(f"\n[dim]{len(out.result.cards)} candidates in {time.time() - t0:.1f}s (budget 180s) · pool {len(out.art.pool)} · credits {out.result.credits_spent:.0f} · "
                  f"LLM ${ctx.llm.usage.usd:.3f} · stages {s}[/dim]")
    if out.queue_ids:
        console.print(f"[dim]drafts queued: approve one with  python -m app.live --approve {out.queue_ids[0]} --send-to <team address>[/dim]")
    run_dir = ROOT / "results" / "live"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{out.result.case_id}.md").write_text(f"# {brief}\n\n{banner}\n\n{md}\n", encoding="utf-8")
    payload = json.dumps(out.result.model_dump(mode="json", exclude={"ranked"}), indent=2, default=str)
    (run_dir / f"{out.result.case_id}.json").write_text(payload, encoding="utf-8")
    if args.json:
        open(args.json, "w", encoding="utf-8").write(payload)
    ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
