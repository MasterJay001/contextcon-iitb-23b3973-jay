"""Canary tests: proof that the number is real.

    1. LEAK-ON vs LEAK-OFF   the same deterministic ranker on today's unmasked profiles (the naive system) vs masked profiles.
                             Leak-on recall should be suspiciously high; the drop is the size of the leak we closed.
    2. SHUFFLED ANSWERS      (a) pseudo-hires drawn at random from each pool: mean recall must equal the random baseline
                             (a test of the evaluator itself); (b) hires swapped across cases: recall must be ~0.
    3. SPOT AUDIT            masked profiles of real hires are printed and machine-checked: the hiring company must not appear in
                             any ongoing/post-T role or in any free text; no sentence may mention a year after T.
    4. FEATURE AUC SCAN      no single feature may separate hires from the pool with AUC > 0.97 on masked data.
    5. UNDATED-FIELD RUN     re-run without skills / location / descriptions (leak #6) and report both.

All rankers here are the deterministic ones (mock LLM) so leak-on and leak-off differ ONLY in what the profile reveals.
    python -m eval.canary --split train
"""
from __future__ import annotations

import argparse
import json
import random
import sys

import numpy as np

from agents.orchestrator import build_artifacts, build_context
from agents.ranker import FEATURE_NAMES, feature_row, rank
from agents.time_guard import TargetRef, assert_clean, mask_profile
from core.config import config_hash, load_config, results_dir, today
from core.models import Case, Weights
from core.render import render_masked
from core.text import contains_phrase, normalize
from eval.evaluator import Evaluator, load_cases
from eval.metrics import CaseScore, auc, mean_ci
from eval.run_ablations import prepare_artifacts, score_variant

KS = (10, 25, 50)


def _recalls(scores: dict[str, CaseScore]) -> dict:
    cs = list(scores.values())
    out = {"n": len(cs), "pool_recall": float(np.mean([c.in_pool for c in cs]))}
    for k in KS:
        m, lo, hi = mean_ci([c.recall(k) for c in cs])
        out[f"recall@{k}"] = {"mean": m, "lo": lo, "hi": hi}
    out["mrr"] = float(np.mean([c.rr() for c in cs]))
    return out


def leak_on_vs_off(ctx, cases, evalr, w) -> dict:
    leak_arts = {c.case_id: build_artifacts(c, ctx, leak_on=True) for c in cases}
    mask_arts = {c.case_id: build_artifacts(c, ctx) for c in cases}
    res = {}
    for v in ("A1", "A4"):
        res[f"{v}_leak_on"] = _recalls(score_variant(leak_arts, cases, evalr, w, v))
        res[f"{v}_masked"] = _recalls(score_variant(mask_arts, cases, evalr, w, v))
    base = []
    for c in cases:
        pool = set(mask_arts[c.case_id].pool_ids)
        in_pool = [h for h in evalr.hire_ids(c.case_id) if h in pool]
        base.append(CaseScore(c.case_id, len(pool), len(in_pool), [1] * len(in_pool)).baseline(25))
    res["random_baseline_recall@25"] = float(np.mean(base))
    drop = res["A1_leak_on"]["recall@25"]["mean"] - res["A1_masked"]["recall@25"]["mean"]
    res["verdict"] = (f"Leak-on Recall@25 {res['A1_leak_on']['recall@25']['mean']:.0%} vs masked {res['A1_masked']['recall@25']['mean']:.0%} "
                      f"(drop {drop:+.0%}). " + ("The leak is real and is now closed." if drop > 0.15 else "WARNING: leak-on is not much higher than masked: check the canary setup."))
    return {"results": res, "_mask_arts": mask_arts}


def shuffled_answers(arts, cases, evalr, w, n_rep: int = 400, seed: int = 3) -> dict:
    rng = random.Random(seed)
    real = score_variant(arts, cases, evalr, w, "A4")
    obs25 = float(np.mean([c.recall(25) for c in real.values()]))
    null25 = []
    base_each = []
    for _ in range(n_rep):
        vals = []
        for c in cases:
            art = arts[c.case_id]
            ids = art.pool_ids
            h = min(len(evalr.hire_ids(c.case_id)), len(ids))
            fake = set(rng.sample(ids, h)) if ids and h else set()
            ranked = [s.person_id for s in rank(art, w, "A4")]
            ranks = [ranked.index(p) + 1 for p in fake]
            vals.append(1.0 if ranks and min(ranks) <= 25 else 0.0)
        null25.append(float(np.mean(vals)))
    for c in cases:
        n, h = len(arts[c.case_id].pool_ids), min(len(evalr.hire_ids(c.case_id)), len(arts[c.case_id].pool_ids))
        base_each.append(CaseScore(c.case_id, n, h, [1] * h).baseline(25))
    null_mean, null_sd = float(np.mean(null25)), float(np.std(null25))
    p = float((np.array(null25) >= obs25).mean())
    # (b) cross-case swap: hires of case pi(i) scored against pool i
    swap = evalr.shuffled_mapping([c.case_id for c in cases], seed)
    swapped = []
    for c in cases:
        ranked = [s.person_id for s in rank(arts[c.case_id], w, "A4")]
        pool = set(arts[c.case_id].pool_ids)
        ranks = [ranked.index(h) + 1 for h in swap[c.case_id] if h in pool]
        swapped.append(1.0 if ranks and min(ranks) <= 25 else 0.0)
    return {"observed_recall@25": obs25, "random_pseudo_hires_mean_recall@25": null_mean, "random_sd": null_sd,
            "analytic_baseline_recall@25": float(np.mean(base_each)), "p_observed_ge_null": p,
            "evaluator_ok": abs(null_mean - float(np.mean(base_each))) < max(0.03, 3 * null_sd),
            "cross_case_swap_recall@25": float(np.mean(swapped)),
            "verdict": ("Evaluator is unbiased: random pseudo-hires hit the analytic baseline." if abs(null_mean - float(np.mean(base_each))) < max(0.03, 3 * null_sd)
                        else "WARNING: random pseudo-hires do not match the analytic baseline: the evaluator or the ranker has a position bug.")}


def spot_audit(ctx, cases, evalr, n_show: int = 5, seed: int = 4) -> dict:
    """Mask the REAL hires' present-day profiles and machine-check them; render a few for human reading."""
    rng = random.Random(seed)
    checked, failures, shown = 0, [], []
    fields = list(ctx.cfg["crust"]["people_fields"]["backtest"])
    for case in cases:
        ids = evalr.hire_ids(case.case_id)
        if not ids:
            continue
        raws = ctx.people.fetch_by_ids(ids, fields).profiles
        tgt = TargetRef(case.company_id, case.company_name, case.company_url)
        for raw in raws:
            mp = mask_profile(raw, case.T, target=tgt, snapshot=today(ctx.cfg))
            checked += 1
            problems = []
            try:
                assert_clean(mp)
            except AssertionError as exc:
                problems.append(str(exc))
            for r in mp.roles:
                if tgt.matches_role(r) and (r.end is None or r.start.date >= case.T):
                    problems.append(f"target company in ongoing/post-T role: {r.title}")
                text = " ".join(x for x in [r.title, r.description] if x)
                if case.company_name and any(contains_phrase(text, v) for v in tgt.variants()):
                    problems.append("target company name in free text")
                if any(int(y) > case.T.year for y in __import__("re").findall(r"\b(20\d{2})\b", r.description or "")):
                    problems.append("year after T in description")
            if problems:
                failures.append({"case": case.case_id, "person_id": raw.person_id, "problems": problems})
            if len(shown) < 40:
                shown.append((case, raw, mp))
    rng.shuffle(shown)
    md = ["# Spot audit: masked profiles of real hires", "",
          "Each profile below is a REAL HIRE rebuilt as of the posting date T. The hiring company must not appear in any ongoing or post-T role, nor in any text.", ""]
    for case, raw, mp in shown[:n_show]:
        md += [f"## case {case.case_id}: hire {raw.person_id}  (T = {case.T}, hiring company = {case.company_name})", "", "```", render_masked(mp).text, "```",
               f"mask report: dropped_post_T={mp.mask.dropped_post_T}, ambiguous={mp.mask.dropped_ambiguous}, ended_after_T->ongoing={mp.mask.ended_after_T_to_ongoing}", ""]
    return {"hires_checked": checked, "failures": failures, "clean": not failures, "markdown": "\n".join(md)}


def feature_auc_scan(arts, cases, evalr) -> dict:
    per = {n: [] for n in FEATURE_NAMES}
    for c in cases:
        art = arts[c.case_id]
        hires = [h for h in evalr.hire_ids(c.case_id) if h in set(art.pool_ids)]
        if not hires:
            continue
        rows = {pid: feature_row(art, pid) for pid in art.pool_ids}
        for n in FEATURE_NAMES:
            pos = [rows[h].get(n, 0.0) for h in hires]
            neg = [rows[p].get(n, 0.0) for p in art.pool_ids if p not in hires]
            a = auc(pos, neg)
            if not np.isnan(a):
                per[n].append(a)
    table = {n: {"auc_mean": float(np.mean(v)), "n_cases": len(v)} for n, v in per.items() if v}
    suspicious = {n: t for n, t in table.items() if t["auc_mean"] > 0.97}
    return {"features": table, "suspicious": suspicious, "clean": not suspicious,
            "verdict": "no single feature separates hires from the pool (AUC <= 0.97)" if not suspicious else f"SUSPICIOUS features: {sorted(suspicious)}"}


def undated_field_run(ctx, cases, evalr, w) -> dict:
    loose = {c.case_id: build_artifacts(c, ctx) for c in cases}
    strict = {c.case_id: build_artifacts(c, ctx, strict_mask=True) for c in cases}
    out = {}
    for v in ("A1", "A4"):
        out[f"{v}_with_skills_location_descriptions"] = _recalls(score_variant(loose, cases, evalr, w, v))
        out[f"{v}_without"] = _recalls(score_variant(strict, cases, evalr, w, v))
    out["note"] = "Skills, profile location and role descriptions are undated (they reflect today). If recall barely moves without them, the result does not rest on them."
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Canary tests for leakage.")
    ap.add_argument("--split", choices=["train", "heldout", "all"], default="train",
                    help="canaries never tune anything, but default to train to respect the held-out one-shot rule")
    ap.add_argument("--crust-backend", choices=["live", "sim"], default=None)
    ap.add_argument("--skip-undated", action="store_true")
    ap.add_argument("--max-cases", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = load_config()
    ctx = build_context(cfg, mode="backtest", crust_backend=args.crust_backend, llm_backend="mock")     # deterministic ranker on both arms
    rdir = results_dir(cfg, ctx.crust_backend, "mock")
    evalr = Evaluator(cfg, ctx.crust_backend)
    cases = load_cases(cfg, args.split, ctx.crust_backend)
    if args.max_cases:
        cases = cases[: args.max_cases]
    w = Weights.from_cfg(cfg)
    print(f"Canaries on {len(cases)} {args.split} cases, backends {ctx.backends()}")
    out: dict = {"meta": {"backends": ctx.backends(), "split": args.split, "n_cases": len(cases), "config_hash": config_hash(cfg)}}
    lo = leak_on_vs_off(ctx, cases, evalr, w)
    mask_arts = lo.pop("_mask_arts")
    out["leak_on_vs_off"] = lo["results"]
    print("1.", lo["results"]["verdict"])
    out["shuffled_answers"] = shuffled_answers(mask_arts, cases, evalr, w)
    print("2.", out["shuffled_answers"]["verdict"], f"(cross-case swap recall@25 = {out['shuffled_answers']['cross_case_swap_recall@25']:.0%})")
    sa = spot_audit(ctx, cases, evalr)
    out["spot_audit"] = {k: v for k, v in sa.items() if k != "markdown"}
    (rdir / "SPOT_AUDIT.md").write_text(sa["markdown"], encoding="utf-8")
    print("3.", f"spot audit: {sa['hires_checked']} hires machine-checked, failures: {len(sa['failures'])}", "-> CLEAN" if sa["clean"] else f"-> {sa['failures'][:2]}")
    out["feature_auc_scan"] = feature_auc_scan(mask_arts, cases, evalr)
    print("4.", out["feature_auc_scan"]["verdict"])
    if not args.skip_undated:
        out["undated_fields"] = undated_field_run(ctx, cases, evalr, w)
        u = out["undated_fields"]
        print("5.", f"A4 Recall@25 with undated fields {u['A4_with_skills_location_descriptions']['recall@25']['mean']:.0%} vs without {u['A4_without']['recall@25']['mean']:.0%}")
    ok = out["shuffled_answers"]["evaluator_ok"] and sa["clean"] and out["feature_auc_scan"]["clean"]
    out["all_clear"] = bool(ok)
    (rdir / "canary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    md = ["# Canary results", "", f"Backends: `{ctx.backends()}`  split `{args.split}`  cases {len(cases)}", "",
          "## 1. Leak-on vs leak-off (same deterministic ranker)", "", "| arm | variant | pool recall | R@10 | R@25 | R@50 |", "|---|---|---|---|---|---|"]
    for k, v in out["leak_on_vs_off"].items():
        if isinstance(v, dict) and "recall@25" in v:
            md.append(f"| {k.split('_', 1)[1].replace('_', ' ')} | {k.split('_')[0]} | {v['pool_recall']:.0%} | " + " | ".join(f"{v[f'recall@{x}']['mean']:.0%}" for x in KS) + " |")
    md += ["", out["leak_on_vs_off"]["verdict"], "", "## 2. Shuffled answers", "", "```json", json.dumps(out["shuffled_answers"], indent=2), "```",
           "", "## 3. Spot audit", "", f"{sa['hires_checked']} real hires machine-checked; failures: {len(sa['failures'])}. See SPOT_AUDIT.md for five rendered profiles.", "",
           "## 4. Feature AUC scan (hire vs pool, masked)", "", out["feature_auc_scan"]["verdict"], "", "| feature | mean AUC |", "|---|---|"]
    md += [f"| {n} | {t['auc_mean']:.2f} |" for n, t in out["feature_auc_scan"]["features"].items()]
    if "undated_fields" in out:
        u = out["undated_fields"]
        md += ["", "## 5. Without undated fields (skills, profile location, role descriptions)", "", "| arm | R@25 | MRR |", "|---|---|---|"]
        for k, v in u.items():
            if isinstance(v, dict) and "recall@25" in v:
                md.append(f"| {k} | {v['recall@25']['mean']:.0%} | {v['mrr']:.3f} |")
    (rdir / "CANARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("ALL CLEAR" if ok else "ATTENTION NEEDED", "->", rdir / "CANARY.md")
    ctx.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
