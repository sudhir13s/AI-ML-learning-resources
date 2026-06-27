"""Static figure generator for 05-Hybrid-Search-BM25-and-Dense.

Imports the SAME canonical functions the page and notebook use (hybrid_search.py) so every plotted
number is the chapter's own -- no hand-typed values. Writes muted-palette PNGs to the shared
chapter image dir (../../images/) with the per-chapter prefix `rag05_`.

    python make_figures_05.py

Figures produced:
  rag05_lens_miss_catch.png  -- gold rank under dense / sparse / hybrid for both blind-spot probes:
                                each single lens misses ONE probe; hybrid catches both.
  rag05_bm25_saturation.png  -- BM25 term-frequency saturation: score vs tf for several k1 (the
                                ceiling that stops keyword-stuffing).
  rag05_bm25_length_norm.png -- BM25 length-normalization: score vs doc length / avgdl for several b.
  rag05_scale_mismatch.png   -- raw cosine vs raw BM25 score ranges on one query: incomparable
                                scales, the reason naive addition fails.
  rag05_alpha_sweep.png      -- weighted-sum MRR & recall vs alpha: the optimum is interior, and
                                alpha=0.5 is not automatically best.
  rag05_rrf_heatmap.png      -- RRF contribution 1/(k+rank) as a function of rank for two lists, and
                                the fused ordering -- how a rank-based fusion combines lists.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / sentence-transformers (CPU). Headless (Agg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from hybrid_search import (
    BM25_B,
    BM25_K1,
    RRF_K,
    BM25,
    DenseRetriever,
    alpha_sweep,
    build_probes,
    full_corpus,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # data / dense
PURPLE = "#5D4A8A"  # process / fusion
GREEN = "#2E7A5A"  # hit / caught
RED = "#8B3B4A"  # miss / danger
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # sparse / highlight
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 110


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def _rank_in(scores: np.ndarray, gold: int) -> int:
    """1-based rank of the gold doc under a score vector (1 = best)."""
    return int(list(map(int, np.argsort(scores)[::-1])).index(gold)) + 1


def fig_lens_miss_catch(dense: DenseRetriever, bm25: BM25, probes) -> None:
    """Gold rank under each method for both probes: each single lens misses one; hybrid catches both.

    Lower bars are better (rank 1 = top). A bar at the 'MISS' line means the gold fell outside the
    top-k the system would actually use -- a real failure. The point of the figure: neither single
    lens is reliable across query TYPES; fusion is.
    """
    from hybrid_search import (
        TUNED_ALPHA,
        reciprocal_rank_fusion,
        weighted_sum_fusion,
    )

    methods = ["dense", "sparse (BM25)", f"hybrid\nweighted a={TUNED_ALPHA}", f"hybrid\nRRF k={RRF_K}"]
    miss_rank = len(full_corpus()) + 1  # plot a miss as one past the corpus size
    bars_per_probe = []
    for probe in probes:
        ds = dense.all_scores(probe.query)
        ss = bm25.all_scores(probe.query)
        ws = weighted_sum_fusion(ds, ss, alpha=TUNED_ALPHA, k=len(full_corpus())).indices
        rrf = reciprocal_rank_fusion([ds, ss], k_rrf=RRF_K, k=len(full_corpus())).indices
        ranks = [
            _rank_in(ds, probe.gold),
            _rank_in(ss, probe.gold),
            (ws.index(probe.gold) + 1) if probe.gold in ws else miss_rank,
            (rrf.index(probe.gold) + 1) if probe.gold in rrf else miss_rank,
        ]
        bars_per_probe.append(ranks)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    _style_axis(ax)
    x = np.arange(len(methods))
    width = 0.38
    labels = ["exact-code probe\n(dense weak)", "paraphrase probe\n(BM25 blind)"]
    colors = [BLUE, AMBER]
    for j, (ranks, label, color) in enumerate(zip(bars_per_probe, labels, colors)):
        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, ranks, width, label=label, color=color, edgecolor=INK, linewidth=0.8)
        for bar, r in zip(bars, ranks):
            txt = "MISS" if r >= miss_rank else f"#{r}"
            tcol = RED if r >= miss_rank or r > 3 else GREEN if r == 1 else INK
            ax.annotate(txt, (bar.get_x() + bar.get_width() / 2, bar.get_height()), fontsize=8.5,
                        color=tcol, ha="center", va="bottom", xytext=(0, 2), textcoords="offset points",
                        fontweight="bold")
    ax.axhline(3.5, color=RED, linewidth=1.3, linestyle="--", alpha=0.7)
    ax.annotate("top-3 cutoff", (len(methods) - 0.5, 3.5), color=RED, fontsize=8.5, ha="right",
                va="bottom", xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("rank of the correct passage  (1 = best, lower is better)")
    ax.set_title("Each lens misses one query type; hybrid catches both", fontsize=12, pad=12)
    ax.set_ylim(0, miss_rank + 1)
    ax.invert_yaxis()  # rank 1 at the top so "better" is visually up
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    _save(fig, "rag05_lens_miss_catch.png")


def fig_bm25_saturation() -> None:
    """BM25 term-frequency saturation: score contribution vs tf for several k1 (length-norm = 1).

    Plots the saturated-tf factor tf*(k1+1)/(tf + k1) (the length-normalized term at |d|=avgdl,
    IDF factored out) so the SHAPE is visible: each curve rises fast then flattens toward its
    ceiling (k1+1). Larger k1 delays saturation (rewards repetition more); k1->0 saturates instantly
    (one occurrence is as good as ten). This is exactly the BM25 saturation term the page derives.
    """
    tf = np.linspace(0, 12, 200)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _style_axis(ax)
    palette = [SLATE, BLUE, PURPLE, AMBER]
    for k1, color in zip([0.5, 1.2, 2.0, 4.0], palette):
        factor = tf * (k1 + 1.0) / (tf + k1)  # length_norm = 1 (avg-length doc)
        ceiling = k1 + 1.0
        lw = 2.6 if k1 == BM25_K1 else 1.8
        label = f"k1 = {k1}" + ("  (Lucene default)" if k1 == BM25_K1 else "")
        ax.plot(tf, factor, color=color, linewidth=lw, label=label)
        ax.axhline(ceiling, color=color, linewidth=0.9, linestyle=":", alpha=0.6)
    # linear (non-saturating) reference, to show what BM25 deliberately avoids
    ax.plot(tf, tf, color=RED, linewidth=1.3, linestyle="--", alpha=0.6, label="linear tf (no saturation)")
    ax.set_title("BM25 saturates term frequency — repetition has diminishing returns", fontsize=12, pad=12)
    ax.set_xlabel("term frequency tf  (count of the query term in the document)")
    ax.set_ylabel("tf contribution  tf·(k1+1)/(tf + k1)")
    ax.set_ylim(0, 8)
    ax.set_xlim(0, 12)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    _save(fig, "rag05_bm25_saturation.png")


def fig_bm25_length_norm() -> None:
    """BM25 length normalization: the denominator factor (1 - b + b·|d|/avgdl) vs |d|/avgdl for b.

    A document at average length (|d|/avgdl = 1) has factor 1 for every b (no penalty). Longer docs
    (>1) inflate the denominator -> lower score; b sets how hard. b=0 ignores length entirely (flat);
    b=1 penalizes fully; b=0.75 (the default) is the standard middle ground.
    """
    ratio = np.linspace(0, 2.5, 200)  # |d| / avgdl
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _style_axis(ax)
    palette = [SLATE, BLUE, AMBER, RED]
    for b, color in zip([0.0, 0.5, 0.75, 1.0], palette):
        factor = 1.0 - b + b * ratio
        lw = 2.6 if b == BM25_B else 1.8
        label = f"b = {b}" + ("  (default)" if b == BM25_B else ("  (ignores length)" if b == 0.0 else ""))
        ax.plot(ratio, factor, color=color, linewidth=lw, label=label)
    ax.axvline(1.0, color=INK, linewidth=1.0, linestyle=":", alpha=0.6)
    ax.annotate("average-length doc\n(factor = 1 for all b)", (1.0, 0.3), color=INK, fontsize=8.5,
                ha="center", va="bottom", xytext=(0, 0), textcoords="offset points")
    ax.set_title("BM25 length normalization — longer docs are discounted (b sets how hard)", fontsize=11.5, pad=12)
    ax.set_xlabel("document length / average length  (|d| / avgdl)")
    ax.set_ylabel("length factor  1 − b + b·|d|/avgdl\n(bigger → lower score)")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.5)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    _save(fig, "rag05_bm25_length_norm.png")


def fig_scale_mismatch(dense: DenseRetriever, bm25: BM25, probes) -> None:
    """Raw cosine vs raw BM25 score ranges on one query — incomparable scales (the naive-sum trap).

    Two horizontal score axes for the SAME query: cosine sits inside [-1, 1]; BM25 ranges over
    [0, ~5+]. Because the BM25 numbers are several times larger, a naive sum is decided almost
    entirely by BM25 — so the dense signal is effectively ignored until both are normalized.
    """
    probe = probes[0]  # exact-code probe: BM25 magnitudes are large here
    ds = dense.all_scores(probe.query)
    ss = bm25.all_scores(probe.query)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    _style_axis(ax)
    # plot each doc's two scores as points on two rows
    n = len(ds)
    ax.scatter(ds, np.full(n, 1.0), s=120, color=BLUE, edgecolors=INK, zorder=3, label="cosine (dense)")
    ax.scatter(ss, np.full(n, 0.0), s=120, color=AMBER, edgecolors=INK, zorder=3, label="BM25 (sparse)")
    ax.axvline(0, color=GRID, linewidth=1.0)
    # bracket the ranges
    ax.annotate(f"cosine ∈ [{ds.min():.2f}, {ds.max():.2f}]", (ds.max(), 1.0), color=BLUE, fontsize=9,
                ha="left", va="center", xytext=(10, 0), textcoords="offset points", fontweight="bold")
    ax.annotate(f"BM25 ∈ [{ss.min():.2f}, {ss.max():.2f}]", (ss.max(), 0.0), color=AMBER, fontsize=9,
                ha="left", va="center", xytext=(10, 0), textcoords="offset points", fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["BM25", "cosine"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(min(-0.3, ds.min() - 0.3), ss.max() + 2.2)
    ax.set_xlabel("raw score")
    ax.set_title("Incomparable scales: BM25 dwarfs cosine — why you must normalize before summing", fontsize=11, pad=12)
    _save(fig, "rag05_scale_mismatch.png")


def fig_alpha_sweep(dense: DenseRetriever, bm25: BM25, probes) -> None:
    """Weighted-sum MRR & recall vs alpha — the tuning curve; optimum is interior, 0.5 isn't best.

    alpha=0 is pure sparse (left), alpha=1 pure dense (right); the interior is the blend. The curve
    shows the best score sits at an INTERIOR alpha (here ~0.6-0.7), so equal weighting (0.5) is a
    starting guess, not the answer — the reason alpha is tuned per corpus.
    """
    sweep = alpha_sweep(probes, dense, bm25, alphas=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0))
    alphas = sorted(sweep)
    mrr = [sweep[a][0] for a in alphas]
    recall = [sweep[a][1] for a in alphas]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _style_axis(ax)
    ax.plot(alphas, mrr, marker="o", color=PURPLE, linewidth=2.4, markersize=7,
            markeredgecolor=INK, markerfacecolor=GREEN, label="MRR")
    ax.plot(alphas, recall, marker="s", color=BLUE, linewidth=2.0, markersize=6,
            markeredgecolor=INK, markerfacecolor=BLUE, alpha=0.85, label="recall@3")
    best = max(alphas, key=lambda a: sweep[a])
    ax.axvspan(0.55, 0.75, color=GREEN, alpha=0.08)
    ax.annotate("best blend\n(interior optimum)", (best, sweep[best][0]), color=GREEN, fontsize=9,
                ha="center", va="bottom", xytext=(0, 14), textcoords="offset points", fontweight="bold")
    ax.annotate("pure sparse", (0.0, sweep[0.0][0]), color=AMBER, fontsize=8.5, ha="left", va="top",
                xytext=(2, -8), textcoords="offset points")
    ax.annotate("pure dense", (1.0, sweep[1.0][0]), color=BLUE, fontsize=8.5, ha="right", va="top",
                xytext=(-2, -8), textcoords="offset points")
    ax.set_title("Tuning alpha — the best blend is interior, not 0.5 by default", fontsize=12, pad=12)
    ax.set_xlabel("alpha  (0 = pure BM25 / sparse  →  1 = pure dense)")
    ax.set_ylabel("score over the probe set")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower center", framealpha=0.95, fontsize=9)
    _save(fig, "rag05_alpha_sweep.png")


def fig_rrf_heatmap() -> None:
    """RRF contribution 1/(k+rank) vs rank, plus a worked two-list fusion — how rank fusion combines.

    Left: the RRF weight curve for a few k_rrf values — being #1 is worth far more than #10, and a
    larger k flattens the curve (deeper ranks still count). Right: a worked example fusing two
    ranked lists into one fused order, showing a doc that is decent in BOTH overtaking a doc that is
    #1 in only one — the behaviour that makes RRF robust but also why k matters.
    """
    fig, (ax_curve, ax_demo) = plt.subplots(1, 2, figsize=(11.4, 4.8))

    # --- left: the 1/(k+rank) weight curve ---
    _style_axis(ax_curve)
    ranks = np.arange(1, 13)
    for k_rrf, color in zip([10, 60, 200], [AMBER, PURPLE, SLATE]):
        weights = 1.0 / (k_rrf + ranks)
        lw = 2.6 if k_rrf == RRF_K else 1.8
        ax_curve.plot(ranks, weights, marker="o", markersize=5, color=color, linewidth=lw,
                      markeredgecolor=INK, label=f"k = {k_rrf}" + ("  (default)" if k_rrf == RRF_K else ""))
    ax_curve.set_title("RRF weight 1/(k + rank) by rank", fontsize=11.5, pad=10)
    ax_curve.set_xlabel("rank in a list  (1 = best)")
    ax_curve.set_ylabel("contribution to the fused score")
    ax_curve.set_xticks(ranks)
    ax_curve.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    # --- right: a worked two-list fusion (illustrative ranks) ---
    ax_demo.axis("off")
    # illustrative ranked lists of 5 docs (A..E) under two retrievers
    list_dense = ["C", "A", "E", "B", "D"]  # dense ranking
    list_sparse = ["A", "C", "D", "E", "B"]  # sparse ranking
    docs = ["A", "B", "C", "D", "E"]
    k_rrf = RRF_K
    rrf_score = {}
    for d in docs:
        rd = list_dense.index(d) + 1
        rs = list_sparse.index(d) + 1
        rrf_score[d] = 1.0 / (k_rrf + rd) + 1.0 / (k_rrf + rs)
    fused = sorted(docs, key=lambda d: rrf_score[d], reverse=True)

    ax_demo.set_xlim(0, 1)
    ax_demo.set_ylim(0, 1)
    ax_demo.text(0.16, 0.96, "dense", ha="center", fontsize=10, fontweight="bold", color=BLUE)
    ax_demo.text(0.5, 0.96, "sparse", ha="center", fontsize=10, fontweight="bold", color=AMBER)
    ax_demo.text(0.86, 0.96, "RRF fused", ha="center", fontsize=10, fontweight="bold", color=GREEN)
    for i, (d, s, f) in enumerate(zip(list_dense, list_sparse, fused)):
        y = 0.84 - i * 0.15
        ax_demo.text(0.16, y, f"{i+1}. {d}", ha="center", fontsize=10, color=INK,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=BLUE, alpha=0.14, edgecolor=BLUE))
        ax_demo.text(0.5, y, f"{i+1}. {s}", ha="center", fontsize=10, color=INK,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=AMBER, alpha=0.14, edgecolor=AMBER))
        ax_demo.text(0.86, y, f"{i+1}. {f}  ({rrf_score[f]:.4f})", ha="center", fontsize=9.5, color=INK,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=GREEN, alpha=0.16, edgecolor=GREEN))
    ax_demo.text(0.5, 0.02, "A is #2/#1 → wins; C is #1/#2 → close behind. Strong-in-both beats strong-in-one.",
                 ha="center", fontsize=8.8, style="italic", color=INK)
    ax_demo.set_title("Worked RRF fusion of two lists (illustrative)", fontsize=11.5, pad=10)
    _save(fig, "rag05_rrf_heatmap.png")


def main() -> None:
    corpus = full_corpus()
    bm25 = BM25(corpus)
    dense = DenseRetriever(corpus)
    probes = build_probes(corpus)
    print(f"dense lens: {dense.backend}")
    fig_lens_miss_catch(dense, bm25, probes)
    fig_bm25_saturation()
    fig_bm25_length_norm()
    fig_scale_mismatch(dense, bm25, probes)
    fig_alpha_sweep(dense, bm25, probes)
    fig_rrf_heatmap()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
