"""Run the ablations A0..A5 (+ A4 with the Critic) and produce THE results table.

    python -m eval.run_ablations --split train                       # development
    python -m eval.run_ablations --split train --tune                # grid-search w_M, w_C on TRAIN only (16 cached combos)
    python -m eval.run_ablations --split heldout --final             # ONE shot, after freezing weights (rule 7)

Rows: A0 keyword baseline, A1 fit-only, A2 F+M, A3 F+C, A4 full, A5 learned, A4+Critic. Columns: pool recall, Recall@10/25/50,
MRR, each with a 95% bootstrap interval, plus the random baseline. The headline is A4 vs A1 by PAIRED bootstrap on per-case
differences; "no clear gain" is reported plainly when the interval includes zero. Multiple comparisons are Holm-corrected.
Results go to results/<crust>-<llm>/ so synthetic runs can never overwrite real ones.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from agents.artifacts import CaseArtifacts
from agents.orchestrator import build_artifacts, build_context, run_case
from agents.ranker import FEATURE_NAMES, rank
from core.config import ROOT, config_hash, load_config, results_dir
from core.models import Case, Weights
from crust.client import CrustError
from crust.titles import family_distance, normalize_title
from eval import learned_ranker
from eval.evaluator import Evaluator, load_cases
from eval.metrics import CaseScore, holm, mcnemar_exact, mean_ci, paired_bootstrap, simulated_power

VARIANT_LABEL = {
    "A0": "A0 keyword (BM25 on masked text)", "A1": "A1 Fit only", "A2": "A2 Fit + Movability", "A3": "A3 Fit + Connection",
    "A4": "A4 Full (F+M+C)", "A5": "A5 learned ranker", "A4C": "A4 + Critic",
}
K_REF = 25


# ------------------------------------------------------------------------------- building / scoring
def prepare_artifacts(ctx, cases: list[Case], evalr: Evaluator, *, inject: bool, strict_mask: bool, rubric: str | None, log=print) -> dict[str, CaseArtifacts]:
    arts: dict[str, CaseArtifacts] = {}
    for i, case in enumerate(cases, 1):
        t0 = time.time()
        extra = None
        if inject:                                  # ranking-only fallback: evaluator hands the hires over as plain profiles
            ids = evalr.hire_ids(case.case_id)
            extra = ctx.people.fetch_by_ids(ids, list(ctx.cfg["crust"]["people_fields"]["backtest"])).profiles
        try:
            arts[case.case_id] = build_artifacts(case, ctx, rubric=rubric, extra_raw=extra, strict_mask=strict_mask)
        except CrustError as exc:
            log(f"  ! {case.case_id}: skipped ({type(exc).__name__}: {str(exc)[:80]})")
            continue
        a = arts[case.case_id]
        log(f"  [{i}/{len(cases)}] {case.case_id} pool={len(a.pool)} roster={a.roster_size} credits={a.credits:.0f} {time.time() - t0:.1f}s"
            + (f" warnings={len(a.warnings)}" if a.warnings else ""))
    return arts


def score_variant(arts: dict[str, CaseArtifacts], cases: list[Case], evalr: Evaluator, w: Weights, variant: str, *, learned=None,
                  use_school: bool = False, inject: bool = False, critic_demoted: dict[str, set[int]] | None = None, cfg: dict | None = None) -> dict[str, CaseScore]:
    out: dict[str, CaseScore] = {}
    for case in cases:
        art = arts.get(case.case_id)
        if art is None:
            continue
        v = "A4" if variant == "A4C" else variant
        dem = (critic_demoted or {}).get(case.case_id) if variant == "A4C" else None
        ranked = rank(art, w, v, use_school=use_school, learned=learned, demoted=dem,
                      demote_penalty=float(cfg["critic"]["demote_penalty"]) if cfg else 0.35)
        out[case.case_id] = evalr.score(case.case_id, [s.person_id for s in ranked], art.pool_ids, injected=inject)
    return out


def summarise(scores: dict[str, dict[str, CaseScore]], ks: list[int], n_boot: int, seed: int) -> list[dict]:
    rows = []
    for v, per_case in scores.items():
        cs = [per_case[c] for c in sorted(per_case)]
        row: dict = {"variant": v, "label": VARIANT_LABEL.get(v, v), "n_cases": len(cs)}
        m, lo, hi = mean_ci([float(c.in_pool) for c in cs], n_boot, seed)
        row["pool_recall"] = {"mean": m, "lo": lo, "hi": hi}
        for k in ks:
            m, lo, hi = mean_ci([c.recall(k) for c in cs], n_boot, seed)
            row[f"recall@{k}"] = {"mean": m, "lo": lo, "hi": hi}
            ing = [c.recall(k) for c in cs if c.in_pool]
            row[f"recall@{k}|pool"] = {"mean": float(np.mean(ing)) if ing else float("nan"), "n": len(ing)}
        m, lo, hi = mean_ci([c.rr() for c in cs], n_boot, seed)
        row["mrr"] = {"mean": m, "lo": lo, "hi": hi}
        rows.append(row)
    ref = next(iter(scores.values()))
    cs = [ref[c] for c in sorted(ref)]
    base = {"variant": "random", "label": "random baseline", "n_cases": len(cs), "pool_recall": {"mean": float(np.mean([c.in_pool for c in cs]))}}
    for k in ks:
        base[f"recall@{k}"] = {"mean": float(np.mean([c.baseline(k) for c in cs]))}
    base["mrr"] = {"mean": float(np.mean([c.baseline_rr() for c in cs]))}
    rows.append(base)
    return rows


def paired_comparisons(scores: dict[str, dict[str, CaseScore]], ks: list[int], n_boot: int, seed: int, alpha: float) -> list[dict]:
    pairs = [("A4", "A1"), ("A2", "A1"), ("A3", "A1"), ("A4", "A0"), ("A5", "A1"), ("A4C", "A4")]
    pairs = [(a, b) for a, b in pairs if a in scores and b in scores]
    out = []
    for a, b in pairs:
        common = sorted(set(scores[a]) & set(scores[b]))
        for metric, getter in [(f"recall@{K_REF}", lambda c: c.recall(K_REF)), ("recall@10", lambda c: c.recall(10)), ("mrr", lambda c: c.rr())]:
            va = [getter(scores[a][c]) for c in common]
            vb = [getter(scores[b][c]) for c in common]
            pb = paired_bootstrap(va, vb, n_boot, seed)
            rec = {"a": a, "b": b, "metric": metric, **pb}
            if metric.startswith("recall"):
                rec["mcnemar"] = mcnemar_exact(va, vb)
            out.append(rec)
    primary = [r for r in out if r["metric"] == f"recall@{K_REF}"]
    for r, h in zip(primary, holm([r["p_signflip"] for r in primary], alpha)):
        r["holm"] = h
    return out


def error_codes(arts: dict[str, CaseArtifacts], s4: dict[str, CaseScore], s1: dict[str, CaseScore] | None, cases: list[Case], evalr: Evaluator,
                w: Weights) -> dict:
    """One reason code per miss (A4 best rank > K_REF): NOT_IN_POOL, GT_NOISE (suspected), RANKED_LOW_FIT, RANKED_LOW_SIGNAL."""
    out: dict[str, list] = {"NOT_IN_POOL": [], "GT_NOISE": [], "RANKED_LOW_FIT": [], "RANKED_LOW_SIGNAL": []}
    for case in cases:
        sc = s4.get(case.case_id)
        if sc is None or (sc.best_rank is not None and sc.best_rank <= K_REF):
            continue
        rec = evalr.hire_records(case.case_id)[0]
        tn = normalize_title(case.target_norm_title)
        if not sc.in_pool:
            code = "NOT_IN_POOL"
        elif family_distance(rec.get("family", "unknown"), tn.family) >= 2 or rec.get("start_precision") == "year":
            code = "GT_NOISE"
        elif s1 is not None and case.case_id in s1 and s1[case.case_id].best_rank is not None and s1[case.case_id].best_rank <= K_REF:
            code = "RANKED_LOW_SIGNAL"
        else:
            art = arts[case.case_id]
            hid = next(h for h in evalr.hire_ids(case.case_id) if h in art.pool_ids)
            from agents.ranker import family_scores

            F = {p: family_scores(art, p, w)[0] for p in art.pool_ids}
            order = sorted(F, key=lambda p: (-F[p], p))
            code = "RANKED_LOW_FIT" if order.index(hid) / max(1, len(order)) > 0.5 else "RANKED_LOW_SIGNAL"
        out[code].append({"case": case.case_id, "best_rank": sc.best_rank, "pool": sc.pool_size, "hire_title": rec.get("title")})
    return {"counts": {k: len(v) for k, v in out.items()}, "cases": out}


def tune_grid(arts: dict[str, CaseArtifacts], cases: list[Case], evalr: Evaluator, base: Weights, grid: list[float]) -> dict:
    """16 combinations of (w_M, w_C) with w_F = 1, TRAIN only, all from cached artifacts (zero credits)."""
    rows = []
    for wm in grid:
        for wc in grid:
            w = base.model_copy(update={"w_M": wm, "w_C": wc})
            sc = score_variant(arts, cases, evalr, w, "A4")
            cs = list(sc.values())
            rows.append({"w_M": wm, "w_C": wc, "recall@25": float(np.mean([c.recall(K_REF) for c in cs])), "mrr": float(np.mean([c.rr() for c in cs])),
                         "recall@10": float(np.mean([c.recall(10) for c in cs]))})
    best = max(rows, key=lambda r: (round(r["recall@25"], 6), round(r["mrr"], 6), -(r["w_M"] + r["w_C"])))
    return {"grid": rows, "best": best}


# ------------------------------------------------------------------------------- reporting
def fmt(x: dict, pct: bool = True) -> str:
    if x is None or "mean" not in x:
        return "-"
    f = (lambda v: f"{100 * v:.0f}%") if pct else (lambda v: f"{v:.3f}")
    if "lo" in x:
        return f"{f(x['mean'])} [{f(x['lo'])}-{f(x['hi'])}]"
    return f(x["mean"])


def markdown_table(rows: list[dict], ks: list[int]) -> str:
    head = ["Variant", "Pool recall"] + [f"Recall@{k}" for k in ks] + ["MRR"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["label"], fmt(r["pool_recall"])] + [fmt(r.get(f"recall@{k}")) for k in ks] + [fmt(r["mrr"], pct=False)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(rdir: Path, tag: str, meta: dict, rows: list[dict], comps: list[dict], errors: dict, ks: list[int], extra: dict) -> None:
    n = rows[0]["n_cases"]
    md = [f"# Results ({tag})", "", f"> Backends: **crust={meta['backends']['crust']}, llm={meta['backends']['llm']}**. "
          + ("**SYNTHETIC: simulator + mock LLM. These numbers verify the harness, not Crustdata or Claude.**" if "sim" in meta["backends"]["crust"] or meta["backends"]["llm"] == "mock" else "Real data and models."),
          "", f"Split: `{meta['split']}`, cases: **{n}**, config hash `{meta['config_hash']}`, run at {meta['ts']}. Intervals are 95% bootstrap over cases (1,000 resamples).", "",
          markdown_table(rows, ks), "", "## A vs B, paired by case (the headline is A4 vs A1)", "",
          "| A | B | metric | mean diff [95% CI] | wins/losses/ties | sign-flip p | McNemar p | verdict |", "|---|---|---|---|---|---|---|---|"]
    for c in comps:
        if c.get("n", 0) == 0:
            continue
        mc = c.get("mcnemar", {}).get("p")
        md.append(f"| {c['a']} | {c['b']} | {c['metric']} | {c['mean_diff']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] | {c['wins']}/{c['losses']}/{c['ties']} | "
                  f"{c['p_signflip']:.3f}{' (Holm ' + format(c['holm']['p_adj'], '.3f') + ')' if 'holm' in c else ''} | {'-' if mc is None else format(mc, '.3f')} | {c['verdict']} |")
    md += ["", "## Where the system fails (A4 misses beyond rank %d)" % K_REF, "", "| reason | count |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in errors["counts"].items()]
    for k, v in extra.items():
        md += ["", f"## {k}", "", "```json", json.dumps(v, indent=2, default=str)[:4000], "```"]
    (rdir / f"RESULTS_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def patch_config_weights(w_m: float, w_c: float, frozen: bool | None = None) -> None:
    import re

    p = ROOT / "config.yaml"
    t = p.read_text(encoding="utf-8-sig")
    t = re.sub(r"(^  w_M:\s*)[0-9.]+", rf"\g<1>{w_m}", t, flags=re.M)
    t = re.sub(r"(^  w_C:\s*)[0-9.]+", rf"\g<1>{w_c}", t, flags=re.M)
    if frozen is not None:
        t = re.sub(r"(^  frozen:\s*)(true|false)", rf"\g<1>{'true' if frozen else 'false'}", t, flags=re.M)
    p.write_text(t, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run ablations A0..A5 and the results table.")
    ap.add_argument("--split", choices=["train", "heldout", "all"], default="train")
    ap.add_argument("--final", action="store_true", help="required for held-out; asserts weights are frozen; runs ONCE")
    ap.add_argument("--allow-rerun", action="store_true", help="override the one-shot held-out ledger (leaves a mark in the report)")
    ap.add_argument("--assume-frozen", action="store_true", help="SIMULATOR ONLY: skip the freeze.frozen check to exercise the protocol on synthetic data")
    ap.add_argument("--weights-file", default=None, help="tuned_weights.json from a TRAIN --tune run (instead of config.yaml weights)")
    ap.add_argument("--variants", default="A0,A1,A2,A3,A4,A5,A4C")
    ap.add_argument("--tune", action="store_true", help="grid-search w_M, w_C on TRAIN (needs --split train)")
    ap.add_argument("--write-config", action="store_true", help="after --tune, write the chosen w_M/w_C into config.yaml")
    ap.add_argument("--strict-mask", action="store_true", help="robustness run without skills/location/descriptions (leak #6)")
    ap.add_argument("--inject-hires", action="store_true", help="ranking-only fallback: add the hire to the pool (reported separately)")
    ap.add_argument("--with-school", action="store_true", help="include the (riskier) school-overlap signal in C")
    ap.add_argument("--crust-backend", choices=["live", "sim"], default=None)
    ap.add_argument("--llm-backend", choices=["anthropic", "mock"], default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    cfg = load_config()
    ctx = build_context(cfg, mode="backtest", crust_backend=args.crust_backend, llm_backend=args.llm_backend)
    rdir = results_dir(cfg, ctx.crust_backend, ctx.llm_backend)
    backends = ctx.backends()
    key = f"{backends['crust']}/{backends['llm']}"
    ledger_p = rdir / "heldout_ledger.json"
    ledger = json.loads(ledger_p.read_text()) if ledger_p.exists() else {}

    if args.split in ("heldout", "all"):                                           # rule 7
        if not args.final:
            print("REFUSED: held-out cases run once, with --final, after weights are frozen (CLAUDE.md rule 7).", file=sys.stderr)
            return 3
        if not bool(cfg["freeze"]["frozen"]) and not (args.assume_frozen and ctx.crust_backend == "sim"):
            print("REFUSED: config.yaml has freeze.frozen: false. Tune on train, then set weights and `freeze.frozen: true` (and commit) first.", file=sys.stderr)
            return 3
        if key in ledger and not args.allow_rerun:
            print(f"REFUSED: held-out already run for {key} at {ledger[key]['ts']} (config {ledger[key]['config_hash']}). Never iterate after seeing held-out.", file=sys.stderr)
            return 3

    evalr = Evaluator(cfg, ctx.crust_backend)
    ks = [int(k) for k in cfg["eval"]["k_values"]]
    n_boot, seed = int(cfg["eval"]["bootstrap_n"]), int(cfg["eval"]["seed"])
    cases = load_cases(cfg, args.split, ctx.crust_backend)
    if not cases:
        print("no cases for this split", file=sys.stderr)
        return 2
    print(f"Backends: {backends}   split={args.split}   cases={len(cases)}   config={config_hash(cfg)}")
    arts = prepare_artifacts(ctx, cases, evalr, inject=args.inject_hires, strict_mask=args.strict_mask, rubric=None)
    cases = [c for c in cases if c.case_id in arts]
    weights = Weights.from_cfg(cfg)
    weights_source = "config.yaml"
    if args.weights_file:
        best = json.loads(Path(args.weights_file).read_text(encoding="utf-8"))["best"]
        weights = weights.model_copy(update={"w_M": float(best["w_M"]), "w_C": float(best["w_C"])})
        weights_source = args.weights_file
    if args.with_school:
        weights = weights.model_copy(update={"conn_inner": {**weights.conn_inner, "school": 0.15}})
    extra: dict = {}

    if args.tune:
        if args.split != "train":
            print("REFUSED: --tune is TRAIN only", file=sys.stderr)
            return 3
        tuned = tune_grid(arts, cases, evalr, weights, [float(x) for x in cfg["weights"]["grid"]])
        weights = weights.model_copy(update={"w_M": tuned["best"]["w_M"], "w_C": tuned["best"]["w_C"]})
        extra["weight_grid_train_only"] = tuned
        (rdir / "tuned_weights.json").write_text(json.dumps(tuned, indent=2), encoding="utf-8")
        print(f"tuned on TRAIN: w_M={weights.w_M} w_C={weights.w_C} (R@25={tuned['best']['recall@25']:.2f}, MRR={tuned['best']['mrr']:.3f})")
        if args.write_config:
            patch_config_weights(weights.w_M, weights.w_C)
            print("wrote w_M/w_C to config.yaml (freeze.frozen unchanged). Review, set freeze.frozen: true, commit, then run held-out ONCE.")

    variants = [v for v in args.variants.split(",") if v]
    learned_vec = None
    if "A5" in variants:
        lp = rdir / "learned_weights.json"
        if args.split in ("train", "all"):
            train_cases = [c for c in cases if c.split == "train"]
            model = learned_ranker.train({c.case_id: arts[c.case_id] for c in train_cases}, evalr, {c.case_id: c.company_id or 0 for c in train_cases})
            model.save(lp)
            learned_vec = model.w
            extra["learned_ranker"] = {"lambda": model.lam, "cv_mean_rr": model.cv, "weights": dict(zip(FEATURE_NAMES, [round(float(x), 3) for x in model.w]))}
        elif lp.exists():
            learned_vec = learned_ranker.LearnedModel.load(lp).w
        else:
            print("A5 skipped: no learned weights (train the ranker on --split train first)")
            variants.remove("A5")

    crit_dem: dict[str, set[int]] = {}
    if "A4C" in variants:
        for case in cases:
            _, _, crit = run_case(case, ctx, weights, variant="A4", use_critic=True, art=arts[case.case_id])
            crit_dem[case.case_id] = set(crit.demoted) if crit else set()
    scores = {v: score_variant(arts, cases, evalr, weights, v, learned=learned_vec, use_school=args.with_school, inject=args.inject_hires,
                               critic_demoted=crit_dem, cfg=cfg) for v in variants}
    rows = summarise(scores, ks, n_boot, seed)
    comps = paired_comparisons(scores, ks, n_boot, seed, float(cfg["eval"]["holm_alpha"]))
    errors = error_codes(arts, scores.get("A4", {}), scores.get("A1"), cases, evalr, weights) if "A4" in scores else {"counts": {}, "cases": {}}
    p_base = float(np.mean([c.recall(K_REF) for c in scores["A1"].values()])) if "A1" in scores else 0.3
    extra["power_note"] = {"n_cases": len(cases), "power_to_detect_+0.15_recall@25_(independent-pairs, conservative)": simulated_power(min(p_base, 0.85), min(p_base + 0.15, 0.99), len(cases))}
    extra["credits_and_time"] = {"credits_estimated_total": round(ctx.crust.meter.estimated, 1), "credits_saved_by_cache": round(ctx.crust.meter.saved, 1),
                                 "per_case_seconds_mean": round(float(np.mean([a.stage_seconds.get("total", 0) for a in arts.values()])), 2),
                                 "llm": ctx.llm.usage.summary(), "crust": ctx.crust.summary()}
    extra["pool_stats"] = {"mean_pool": round(float(np.mean([len(a.pool) for a in arts.values()])), 1),
                           "mean_universe": round(float(np.mean([a.api_total_count or 0 for a in arts.values()])), 1),
                           "cases_with_multiple_hires": sum(1 for c in cases if len(evalr.hire_ids(c.case_id)) > 1),
                           "share_roles_dropped_ambiguous": round(sum(a.mask_totals.get("dropped_ambiguous", 0) for a in arts.values()) / max(1, sum(a.mask_totals.get("n_roles_raw", 0) for a in arts.values())), 4)}
    tag = args.tag or (args.split + ("_strict" if args.strict_mask else "") + ("_injected" if args.inject_hires else "") + ("_final" if args.final else ""))
    meta = {"backends": backends, "split": args.split, "config_hash": config_hash(cfg), "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "weights": {"w_F": weights.w_F, "w_M": weights.w_M, "w_C": weights.w_C}, "weights_source": weights_source,
            "strict_mask": args.strict_mask, "injected": args.inject_hires,
            "final": args.final, "rerun_override": bool(args.allow_rerun)}
    write_report(rdir, tag, meta, rows, comps, errors, ks, extra)
    (rdir / f"ablations_{tag}.json").write_text(json.dumps({"meta": meta, "rows": rows, "comparisons": comps, "errors": errors, "extra": extra,
                                                            "per_case": {v: {c: {"ranks": s.ranks, "pool": s.pool_size} for c, s in per.items()} for v, per in scores.items()}},
                                                           indent=2, default=str), encoding="utf-8")
    print("\n" + markdown_table(rows, ks))
    for c in comps:
        if c["a"] == "A4" and c["b"] == "A1" and c.get("n"):
            print(f"\nA4 vs A1 {c['metric']}: {c['mean_diff']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]  wins/losses/ties {c['wins']}/{c['losses']}/{c['ties']}  -> {c['verdict']}")
    print("errors:", errors["counts"])
    if args.final:
        ledger[key] = {"ts": meta["ts"], "config_hash": meta["config_hash"], "weights": meta["weights"], "n_cases": len(cases)}
        ledger_p.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"\nwrote {rdir / ('RESULTS_' + tag + '.md')}")
    ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
