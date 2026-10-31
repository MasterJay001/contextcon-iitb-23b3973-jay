"""Byproduct facts you cannot Google, computed from the same masked data (doc sec. 3.4 / 14):

  * share of hires who already had an ex-colleague inside the hiring company on T (and the lift over non-hires)
  * median tenure at the moment people moved
  * share of hires who were between roles / recently promoted at T
  * top feeder employers per target company
  * the fitted hazard: where the job-change hazard peaks (tests "movement peaks after ~18-36 months")

    python -m eval.byproducts --split train
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from agents.movability import employer_spells
from agents.orchestrator import build_artifacts, build_context
from core.config import load_config, results_dir
from core.dates import months_between
from eval.evaluator import Evaluator, load_cases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "heldout", "all"], default="train")
    ap.add_argument("--crust-backend", choices=["live", "sim"], default=None)
    ap.add_argument("--llm-backend", choices=["anthropic", "mock"], default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    ctx = build_context(cfg, mode="backtest", crust_backend=args.crust_backend, llm_backend=args.llm_backend)
    evalr = Evaluator(cfg, ctx.crust_backend)
    cases = load_cases(cfg, args.split, ctx.crust_backend)
    hire_conn, other_conn = [], []
    tenure_at_T, tenure_at_move, pool_tenure = [], [], []
    between = promo = n_hires = 0
    feeders: dict[str, dict[str, int]] = {}
    peaks, medians = [], []
    for case in cases:
        art = build_artifacts(case, ctx)
        recs = {int(h["person_id"]): h for h in evalr.hire_records(case.case_id)}
        if art.survival.get("hazard_peak_months") is not None:
            peaks.append(art.survival["hazard_peak_months"])
        if art.survival.get("median_tenure_at_exit_months") is not None:
            medians.append(art.survival["median_tenure_at_exit_months"])
        if art.feeders:
            feeders[case.case_id] = {name: n for name, n in art.feeders}
        for mp in art.pool:
            md = art.move_diag.get(mp.person_id)
            if md and md.months_at_employer is not None:
                pool_tenure.append(md.months_at_employer)
        if not art.conn_available:
            continue
        for mp in art.pool:
            n_ex = art.conn_diag[mp.person_id].n_ex_colleagues
            if mp.person_id in recs:
                n_hires += 1
                hire_conn.append(n_ex >= 1)
                md = art.move_diag.get(mp.person_id)
                if md and md.months_at_employer is not None:
                    tenure_at_T.append(md.months_at_employer)
                    start = np.datetime64(recs[mp.person_id]["start_date"])
                    gap_m = (start - np.datetime64(case.T)).astype("timedelta64[D]").astype(float) / 30.4375
                    tenure_at_move.append(md.months_at_employer + max(0.0, gap_m))
                if mp.between_roles_months is not None:
                    between += 1
                sp = employer_spells(mp)
                if sp and sp[-1].end is None and sp[-1].n_roles >= 2 and months_between(sp[-1].last_role_start, mp.T) <= 6:
                    promo += 1
            else:
                other_conn.append(n_ex >= 1)
    out = {
        "backends": ctx.backends(), "split": args.split, "n_cases": len(cases), "hires_in_pool": n_hires,
        "share_hires_with_ex_colleague_at_company": float(np.mean(hire_conn)) if hire_conn else None,
        "share_non_hires_with_ex_colleague": float(np.mean(other_conn)) if other_conn else None,
        "lift": (float(np.mean(hire_conn)) / max(1e-9, float(np.mean(other_conn)))) if hire_conn and other_conn else None,
        "median_tenure_months_at_T_hires": float(np.median(tenure_at_T)) if tenure_at_T else None,
        "median_tenure_months_at_move_hires": float(np.median(tenure_at_move)) if tenure_at_move else None,
        "median_tenure_months_at_T_pool": float(np.median(pool_tenure)) if pool_tenure else None,
        "share_hires_between_roles_at_T": between / n_hires if n_hires else None,
        "share_hires_recently_promoted": promo / n_hires if n_hires else None,
        "hazard_peak_months_mean_over_cases": float(np.mean(peaks)) if peaks else None,
        "median_tenure_at_exit_months_mean_over_cases": float(np.mean(medians)) if medians else None,
        "top_feeders_by_case": feeders,
    }
    rdir = results_dir(cfg, ctx.crust_backend, ctx.llm_backend)
    (rdir / "byproducts.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = ["# Byproduct facts (from the same masked data)", "", f"Backends `{ctx.backends()}`, split `{args.split}`, {len(cases)} cases, {n_hires} hires in pools.", ""]
    for k, v in out.items():
        if k in {"backends", "top_feeders_by_case"}:
            continue
        md.append(f"- **{k.replace('_', ' ')}**: {('%.2f' % v) if isinstance(v, float) else v}")
    md += ["", "## Top feeder employers (people who joined the hiring company from them before T)", ""]
    for cid, fd in list(feeders.items())[:15]:
        md.append(f"- {cid}: " + ", ".join(f"{n} ({c})" for n, c in fd.items()))
    (rdir / "BYPRODUCTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "top_feeders_by_case"}, indent=2))
    ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
