"""Render charts from reports/<tag>/*.json into docs/.

    python benchmarks/plot.py reports/glove25_n100000
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = {  # fixed entity -> colour mapping (never re-assigned by rank)
    "vecdb HNSW": ("#2a78d6", "-", "o"),
    "FAISS HNSWFlat": ("#eb6834", "--", "s"),
    "vecdb IVF-PQ + rerank100": ("#1baf7a", "-", "^"),
    "vecdb IVF-PQ": ("#eda100", "-", "v"),
    "FAISS IVF-PQ": ("#e87ba4", "--", "D"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def recall_qps(rep_dir: str, out: str) -> None:
    r = json.load(open(os.path.join(rep_dir, "sweeps.json")))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for name, (color, ls, mk) in SERIES.items():
        rows = [s for s in r["sweeps"] if s["method"] == name]
        if not rows:
            continue
        ax.plot([s["recall"] for s in rows], [s["qps"] for s in rows], ls, color=color,
                marker=mk, markersize=5, linewidth=2, label=name)
    exact = [s for s in r["sweeps"] if s["method"].startswith("exact")][0]
    ax.scatter([exact["recall"]], [exact["qps"]], color=INK, marker="*", s=120, zorder=5,
               label="exact brute force (numpy)")
    ax.set_yscale("log")
    ax.set_xlabel("recall@10", color=INK)
    ax.set_ylabel("queries per second (1 thread, log scale)", color=INK)
    ax.set_title(f"GloVe-25 angular, n = {r['n_base']:,}, {r['n_queries']} queries",
                 color=INK, fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def filtered(rep_dir: str, out: str) -> None:
    f = json.load(open(os.path.join(rep_dir, "filtered.json")))
    rows = f["rows"]
    sel = [r["selectivity"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(sel, [r["hnsw_qps"] for r in rows], "-o", color="#2a78d6", linewidth=2,
            markersize=5, label=f"HNSW with filter (ef={f['ef']})")
    ax.plot(sel, [r["exact_qps"] for r in rows], "-s", color="#eb6834", linewidth=2,
            markersize=5, label="brute force over matching points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("filter selectivity (fraction of points that match; random, uncorrelated)",
                  color=INK)
    ax.set_ylabel("queries per second (log)", color=INK)
    ax.set_title(f"Filtered search, n = {f['n_base']:,}", color=INK, fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    rep_dir = sys.argv[1]
    tag = os.path.basename(os.path.normpath(rep_dir))
    os.makedirs("docs", exist_ok=True)
    recall_qps(rep_dir, f"docs/recall_qps_{tag}.png")
    if os.path.exists(os.path.join(rep_dir, "filtered.json")):
        filtered(rep_dir, f"docs/filtered_{tag}.png")
    print("charts written to docs/")


if __name__ == "__main__":
    main()
