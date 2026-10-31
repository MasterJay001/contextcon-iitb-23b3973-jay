"""Presentation figures from results/<backend>/*.json (palette: dataviz reference; thin marks, one axis, direct labels).

    python -m scripts.make_figures --backend sim-mock --tag train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLUE, GRAY, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#b8b7b1", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#d9d8d2")
    ax.spines["bottom"].set_color("#d9d8d2")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color="#ecebe6", linewidth=0.8)
    ax.set_axisbelow(True)


def recall_figure(data: dict, out: Path, backend: str) -> None:
    rows = [r for r in data["rows"] if r["variant"] != "random"]
    base = next(r for r in data["rows"] if r["variant"] == "random")["recall@25"]["mean"]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    _style(ax)
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        m = r["recall@25"]
        hot = r["variant"] in ("A4", "A5")
        ax.barh(y, m["mean"], height=0.55, color=BLUE if hot else GRAY)
        ax.plot([m["lo"], m["hi"]], [y, y], color=INK, linewidth=1.2)
        ax.text(min(m["hi"], 0.97) + 0.015, y, f"{m['mean']:.0%}", va="center", color=INK, fontsize=9)
    ax.axvline(base, color=ORANGE, linewidth=1.5)
    ax.text(base + 0.008, len(rows) - 0.4, f"random {base:.0%}", color=ORANGE, fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], color=INK, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    synth = "  [SYNTHETIC: simulator + mock LLM]" if backend.startswith("sim") or backend.endswith("mock") else ""
    ax.set_title(f"Recall@25 of the real hire, by ranking signal{synth}", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def leak_figure(canary: dict, out: Path, backend: str) -> None:
    r = canary["leak_on_vs_off"]
    labels = ["A1 leak-on", "A1 masked", "A4 leak-on", "A4 masked"]
    vals = [r["A1_leak_on"]["recall@25"]["mean"], r["A1_masked"]["recall@25"]["mean"], r["A4_leak_on"]["recall@25"]["mean"], r["A4_masked"]["recall@25"]["mean"]]
    fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor=SURFACE)
    _style(ax)
    ys = [3, 2, 1, 0]
    for y, v, lab in zip(ys, vals, labels):
        ax.barh(y, v, height=0.55, color=ORANGE if "leak-on" in lab else BLUE)
        ax.text(v + 0.012, y, f"{v:.0%}", va="center", color=INK, fontsize=9)
    ax.axvline(r["random_baseline_recall@25"], color=MUTED, linewidth=1, linestyle=":")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, color=INK, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Leak canary: same ranker, unmasked vs masked profiles", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def errors_figure(data: dict, out: Path) -> None:
    c = data["errors"]["counts"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor=SURFACE)
    _style(ax)
    names = list(c)[::-1]
    ax.barh(range(len(names)), [c[n] for n in names], height=0.55, color=BLUE)
    for i, n in enumerate(names):
        ax.text(c[n] + 0.1, i, str(c[n]), va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=INK, fontsize=9)
    ax.set_title("Where the system misses (A4, beyond rank 25)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sim-mock")
    ap.add_argument("--tag", default="train")
    a = ap.parse_args()
    d = ROOT / "results" / a.backend
    fig_dir = d / "figures"
    fig_dir.mkdir(exist_ok=True)
    abl = d / f"ablations_{a.tag}.json"
    if abl.exists():
        data = json.loads(abl.read_text(encoding="utf-8"))
        recall_figure(data, fig_dir / "recall25.png", a.backend)
        errors_figure(data, fig_dir / "errors.png")
    can = d / "canary.json"
    if can.exists():
        leak_figure(json.loads(can.read_text(encoding="utf-8")), fig_dir / "leak_canary.png", a.backend)
    print("figures ->", fig_dir)


if __name__ == "__main__":
    main()
