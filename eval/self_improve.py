"""Self-improving ranker with a Goodhart guard (optional piece of the build plan; the most impressive when it works).

    python -m eval.self_improve --rounds 3                 # TRAIN only
    python -m eval.self_improve --rounds 3 --strict-guard  # analyst sees fold A; acceptance also needs fold B not to get worse
    python -m eval.self_improve --rounds 3 --freeze        # write accepted weights into config.yaml, lessons into lessons.md, freeze

Loop (design doc sec. 9):
 1. run all TRAIN cases with the current rubric R and weights W; record the rank of each hire;
 2. the Failure Analyst reads the misses (the hire's masked profile, the people ranked above, their F/M/C) and proposes <= 2 edits;
 3. LINT: reject any edit that names a company, person, school or id - rules must be general;
 4. ACCEPT only if Recall@25 rises by >= 1 case AND MRR does not fall (re-run on train); otherwise REVERT;
 5. stop after N rounds or the time budget; 6. freeze; 7. test on held-out ONCE (eval/run_ablations.py --split heldout --final).

Goodhart's law: a measure that becomes a target stops being a good measure; with ~25 train cases an optimiser memorises. The guard is
(a) the accept rule above, (b) the optional fold split (the analyst only ever sees fold A, acceptance must not hurt fold B), (c) the lint,
and (d) the held-out set that nothing here ever consults. `--shadow-heldout` logs held-out recall per round for the "train vs held-out"
plot; it is DIAGNOSTIC ONLY: no decision reads it, and the official held-out run still happens exactly once.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time

import numpy as np

from agents.artifacts import CaseArtifacts
from agents.failure_analyst import Edit, MissReport, lint_edit, propose
from agents.orchestrator import build_artifacts, build_context, load_lessons, rubric_text
from agents.ranker import family_scores, feature_row, rank
from core.config import ROOT, config_hash, load_config, results_dir
from core.models import Case, Weights
from core.render import render_masked
from eval.evaluator import Evaluator, load_cases
from eval.metrics import CaseScore
from eval.run_ablations import K_REF, patch_config_weights, score_variant


def metrics_of(scores: dict[str, CaseScore]) -> dict:
    cs = list(scores.values())
    return {"hits@25": int(sum(c.recall(K_REF) for c in cs)), "recall@25": float(np.mean([c.recall(K_REF) for c in cs])) if cs else 0.0,
            "mrr": float(np.mean([c.rr() for c in cs])) if cs else 0.0, "n": len(cs)}


def split_folds(cases: list[Case], seed: int) -> tuple[list[Case], list[Case]]:
    cos = sorted({c.company_id for c in cases}, key=lambda x: hashlib.sha256(f"fold-{seed}-{x}".encode()).hexdigest())
    a = set(cos[: max(1, len(cos) // 2)])
    return [c for c in cases if c.company_id in a], [c for c in cases if c.company_id not in a]


def build_misses(arts: dict[str, CaseArtifacts], scores: dict[str, CaseScore], cases: list[Case], evalr: Evaluator, w: Weights,
                 max_cases: int = 6) -> tuple[list[MissReport], set[str]]:
    """Misses = hire in the pool but ranked beyond K_REF (retrieval failures are not fixable by rubric/weights)."""
    reports: list[MissReport] = []
    entities: set[str] = set()
    worst = sorted((c for c in cases if c.case_id in scores and scores[c.case_id].in_pool and (scores[c.case_id].best_rank or 0) > K_REF),
                   key=lambda c: -(scores[c.case_id].best_rank or 0))[:max_cases]
    for case in worst:
        art = arts[case.case_id]
        hid = next(h for h in evalr.hire_ids(case.case_id) if h in art.pool_ids)
        by_id = {m.person_id: m for m in art.pool}
        ranked = rank(art, w, "A4")
        pos = {s.person_id: i + 1 for i, s in enumerate(ranked)}
        F, M, C = family_scores(art, hid, w)
        rows = {pid: feature_row(art, pid) for pid in art.pool_ids}
        pct = {}
        for name, v in rows[hid].items():
            vals = np.array([r.get(name, 0.0) for r in rows.values()])
            pct[name] = round(float((vals < v).mean() + 0.5 * (vals == v).mean()), 2)
        above = []
        for s in ranked[:2]:
            above.append((render_masked(by_id[s.person_id]).text, {"F": round(s.F, 2), "M": round(s.M, 2), "C": round(s.C, 2)}))
        hm = by_id[hid]
        reports.append(MissReport(case.case_id, f"{art.spec.norm_title} ({art.spec.seniority}) in {art.spec.location or 'any location'}", pos[hid], len(art.pool),
                                  render_masked(hm).text, {"F": round(F, 2), "M": round(M, 2), "C": round(C, 2)}, above, pct))
        for m in (hm, *[by_id[s.person_id] for s in ranked[:3]]):
            for r in m.roles:
                if r.company_name:
                    entities.add(r.company_name.lower())
            for sc in m.education:
                if sc.school:
                    entities.add(sc.school.lower())
        if case.company_name:
            entities.add(case.company_name.lower())
        entities.add(str(hid))
    return reports, entities


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Self-improving ranker with a Goodhart guard (TRAIN only).")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--strict-guard", action="store_true")
    ap.add_argument("--shadow-heldout", action="store_true", help="log held-out recall per round (diagnostic only; nothing reads it)")
    ap.add_argument("--max-rubric-evals", type=int, default=4, help="cap on rubric edits evaluated (each re-runs LLM fit on all train cases)")
    ap.add_argument("--freeze", action="store_true", help="write accepted weights to config.yaml, lessons to lessons.md, set freeze.frozen: true")
    ap.add_argument("--crust-backend", choices=["live", "sim"], default=None)
    ap.add_argument("--llm-backend", choices=["anthropic", "mock"], default=None)
    args = ap.parse_args(argv)

    cfg = load_config()
    ctx = build_context(cfg, mode="backtest", crust_backend=args.crust_backend, llm_backend=args.llm_backend)
    rdir = results_dir(cfg, ctx.crust_backend, ctx.llm_backend)
    evalr = Evaluator(cfg, ctx.crust_backend)
    train = load_cases(cfg, "train", ctx.crust_backend)
    held = load_cases(cfg, "heldout", ctx.crust_backend) if args.shadow_heldout else []
    fold_a, fold_b = split_folds(train, int(cfg["run"]["seed"])) if args.strict_guard else (train, [])
    w0 = Weights.from_cfg(cfg)
    state = {"w_M": w0.w_M, "w_C": w0.w_C, "lessons": [ln[2:].strip() for ln in load_lessons(cfg).splitlines() if ln.startswith("- ")]}
    t_start, rubric_evals = time.time(), 0
    history, decisions = [], []

    def arts_for(cases: list[Case], lessons: list[str]) -> dict[str, CaseArtifacts]:
        rub = rubric_text(cfg, "\n".join(f"- {x}" for x in lessons))
        return {c.case_id: build_artifacts(c, ctx, rubric=rub) for c in cases}

    def evaluate(cases: list[Case], arts, st) -> dict[str, CaseScore]:
        w = w0.model_copy(update={"w_M": st["w_M"], "w_C": st["w_C"]})
        return score_variant(arts, cases, evalr, w, "A4")

    print(f"Self-improvement on {len(train)} TRAIN cases ({'strict guard: fold A %d / B %d' % (len(fold_a), len(fold_b)) if args.strict_guard else 'standard guard'}); backends {ctx.backends()}")
    arts = arts_for(train, state["lessons"])
    scores = evaluate(train, arts, state)
    cur = metrics_of(scores)
    shadow = metrics_of(evaluate(held, arts_for(held, state["lessons"]), state)) if held else None
    history.append({"round": 0, "state": dict(state), "train": cur, "heldout_shadow": shadow})
    print(f"round 0: train hits@25={cur['hits@25']}/{cur['n']} MRR={cur['mrr']:.3f}" + (f" | heldout(shadow) R@25={shadow['recall@25']:.2f}" if shadow else ""))

    for rnd in range(1, args.rounds + 1):
        if (time.time() - t_start) / 60 > args.minutes:
            print("time budget reached")
            break
        w = w0.model_copy(update={"w_M": state["w_M"], "w_C": state["w_C"]})
        misses, entities = build_misses(arts, scores, fold_a, evalr, w)
        if not misses:
            print(f"round {rnd}: no rankable misses left on the analyst's cases; stopping")
            break
        out, src = propose(ctx.llm, misses, state)
        print(f"round {rnd}: analyst ({src}) saw {len(misses)} misses and proposed {len(out.edits)} edit(s)")
        for e in out.edits[:2]:
            lint = lint_edit(e, entities)
            rec = {"round": rnd, "edit": e.model_dump(), "lint": lint.reason or "ok", "accepted": False}
            if not lint.ok:
                rec["decision"] = f"REJECTED by lint: {lint.reason}"
                decisions.append(rec)
                print(f"   x {e.kind} {e.param or ''} rejected by lint: {lint.reason}")
                continue
            cand = {"w_M": state["w_M"], "w_C": state["w_C"], "lessons": list(state["lessons"])}
            if e.kind == "weight":
                cand[e.param] = round(float(e.value), 2)
                cand_arts = arts
            else:
                if rubric_evals >= args.max_rubric_evals:
                    rec["decision"] = "SKIPPED: rubric-evaluation budget exhausted"
                    decisions.append(rec)
                    continue
                cand["lessons"].append(e.text.strip())
                cand_arts = arts_for(train, cand["lessons"])
                rubric_evals += 1
            new_scores = evaluate(train, cand_arts, cand)
            new = metrics_of(new_scores)
            ok = new["hits@25"] >= cur["hits@25"] + 1 and new["mrr"] >= cur["mrr"] - 1e-12
            if ok and args.strict_guard and fold_b:
                b_old, b_new = metrics_of({c.case_id: scores[c.case_id] for c in fold_b}), metrics_of({c.case_id: new_scores[c.case_id] for c in fold_b})
                if b_new["hits@25"] < b_old["hits@25"] or b_new["mrr"] < b_old["mrr"] - 1e-12:
                    ok = False
                    rec["guard_fold"] = {"old": b_old, "new": b_new}
            rec.update({"before": cur, "after": new, "accepted": bool(ok), "decision": "ACCEPTED" if ok else "REVERTED"})
            decisions.append(rec)
            print(f"   {'+' if ok else '-'} {e.kind} {e.param or ''}{'=' + str(cand.get(e.param)) if e.param else ''}: hits@25 {cur['hits@25']}->{new['hits@25']}, "
                  f"MRR {cur['mrr']:.3f}->{new['mrr']:.3f}  => {'ACCEPTED' if ok else 'REVERTED'}  ({e.reason[:70]})")
            if ok:
                state, arts, scores, cur = cand, cand_arts, new_scores, new
        shadow = metrics_of(evaluate(held, arts_for(held, state["lessons"]), state)) if held else None
        history.append({"round": rnd, "state": {k: (list(v) if isinstance(v, list) else v) for k, v in state.items()}, "train": cur, "heldout_shadow": shadow})
        print(f"round {rnd} done: train hits@25={cur['hits@25']}/{cur['n']} MRR={cur['mrr']:.3f}" + (f" | heldout(shadow) R@25={shadow['recall@25']:.2f}" if shadow else ""))

    final = {"w_F": 1.0, "w_M": state["w_M"], "w_C": state["w_C"], "lessons": state["lessons"], "train": cur, "config_hash_before": config_hash(cfg),
             "backends": ctx.backends(), "ts": dt.datetime.now().isoformat(timespec="seconds"), "strict_guard": args.strict_guard}
    (rdir / "frozen.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (rdir / "self_improve.json").write_text(json.dumps({"history": history, "decisions": decisions, "final": final}, indent=2, default=str), encoding="utf-8")
    md = ["# Self-improvement (TRAIN only, Goodhart-guarded)", "", f"Backends `{ctx.backends()}`; analyst source above; strict guard: **{args.strict_guard}**; shadow held-out: **{bool(held)}** (diagnostic only).", "",
          "| round | w_M | w_C | lessons | train hits@25 | train MRR | held-out R@25 (shadow) |", "|---|---|---|---|---|---|---|"]
    for h in history:
        md.append(f"| {h['round']} | {h['state']['w_M']} | {h['state']['w_C']} | {len(h['state']['lessons'])} | {h['train']['hits@25']}/{h['train']['n']} | {h['train']['mrr']:.3f} | "
                  f"{('%.2f' % h['heldout_shadow']['recall@25']) if h['heldout_shadow'] else '-'} |")
    md += ["", "## Decisions", ""]
    for d in decisions:
        md.append(f"- round {d['round']}: `{d['edit']['kind']}` {d['edit'].get('param') or ''} {d['edit'].get('text') or ''} -> **{d.get('decision')}** ({d['edit']['reason']})")
    (rdir / "SELF_IMPROVE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if args.freeze:
        patch_config_weights(state["w_M"], state["w_C"], frozen=True)
        lines = ["# Lessons (episodic memory written by the Failure Analyst; accepted edits only)", ""] + [f"- {x}" for x in state["lessons"]]
        (ROOT / str(cfg["paths"]["lessons"])).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("FROZEN: config.yaml weights updated and freeze.frozen=true; lessons.md written. Commit, then run held-out ONCE:  python -m eval.run_ablations --split heldout --final")
    print(f"accepted state: w_M={state['w_M']} w_C={state['w_C']} lessons={len(state['lessons'])}  -> {rdir / 'SELF_IMPROVE.md'}")
    ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
