"""Static figure generator for 06-Re-ranking-Cross-Encoders.

Imports the SAME canonical functions the page and notebook use (reranking.py) so every plotted
number is the chapter's own -- no hand-typed values. Writes muted-palette PNGs to the shared
chapter image dir (../../images/) with the per-chapter prefix `rag06_`.

    python make_figures_06.py

Figures produced:
  rag06_funnel.png        -- the retrieve -> re-rank -> answer funnel: N passages narrow to a
                             top-k pool, re-ranked down to the final few.
  rag06_bi_vs_cross.png   -- bi-encoder (independent dual towers -> cosine, precomputable) vs
                             cross-encoder (one joint tower over [q SEP d] -> scalar), schematic.
  rag06_rank_shuffle.png  -- the rank shuffle: each candidate's bi-encoder rank vs its re-ranked
                             rank; the gold climbs from #4 to #1.
  rag06_ndcg_mrr.png      -- nDCG@k and MRR, bi-encoder vs re-ranked (the measured lift).
  rag06_latency_vs_k.png  -- query-time cost vs pool size: bi-encoder compare is flat-cheap; the
                             cross-encoder pays k joint forward passes (illustrative model).
  rag06_recall_ceiling.png-- retrieve top-3 (gold missed -> unrecoverable) vs top-10 (gold present
                             -> re-ranked to #1): re-ranking reorders, it cannot rescue.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / sentence-transformers (CPU). Headless (Agg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from reranking import (
    CORPUS,
    FINAL_K,
    GOLD_INDEX,
    NDCG_KS,
    QUERY,
    RETRIEVE_K,
    BiEncoderRetriever,
    CrossEncoderReranker,
    ndcg_at_k,
    reciprocal_rank,
    retrieve_then_rerank,
)

# ---- Palette (matches the chapter family's muted Mermaid classDefs) -------------------------
BLUE = "#3A6B96"  # data / bi-encoder
PURPLE = "#5D4A8A"  # process / cross-encoder
GREEN = "#2E7A5A"  # gold / win
RED = "#8B3B4A"  # miss / danger
SLATE = "#4A5B6E"  # neutral / distractor
AMBER = "#7A6528"  # query / highlight
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 110


def _style_axis(ax: plt.Axes) -> None:
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


def fig_funnel() -> None:
    """The two-stage funnel: N passages -> bi-encoder top-k pool -> cross-encoder final few."""
    n = len(CORPUS)
    stages = [
        (f"CORPUS\n{n} passages", n, BLUE),
        (f"RETRIEVE (bi-encoder)\ntop-{RETRIEVE_K}", RETRIEVE_K, PURPLE),
        (f"RE-RANK (cross-encoder)\ntop-{FINAL_K}", FINAL_K, GREEN),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axis("off")
    max_w = 1.0
    y = 0.0
    centers = []
    for label, count, color in stages:
        w = max_w * count / n
        x0 = (max_w - w) / 2
        ax.add_patch(plt.Rectangle((x0, y), w, 0.8, facecolor=color, alpha=0.22, edgecolor=color, linewidth=2))
        ax.text(0.5, y + 0.4, label, ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold")
        centers.append((0.5, y))
        y -= 1.15
    # arrows + the "cheap recall / expensive precision" annotation
    for (cx, cy0), (_, cy1) in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=(cx, cy1 + 0.8), xytext=(cx, cy0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
    ax.text(0.93, -0.18, "cheap, precomputable\n(independent encodings)", ha="left", va="center",
            fontsize=8.5, color=PURPLE, style="italic")
    ax.text(0.93, -1.33, "expensive, query-time only\n(joint encoding, k forward passes)", ha="left",
            va="center", fontsize=8.5, color=GREEN, style="italic")
    ax.set_xlim(-0.05, 1.7)
    ax.set_ylim(-2.5, 0.95)
    ax.set_title("Two-stage retrieval: cheap recall, then expensive precision", fontsize=12, pad=10)
    _save(fig, "rag06_funnel.png")


def fig_bi_vs_cross() -> None:
    """Schematic: bi-encoder (two independent towers -> cosine) vs cross-encoder (one joint tower)."""
    fig, (ax_bi, ax_cross) = plt.subplots(1, 2, figsize=(11.2, 5.2))
    for ax in (ax_bi, ax_cross):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def box(ax, x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.2, edgecolor=color, linewidth=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, color=INK)

    # --- bi-encoder: two towers ---
    ax_bi.set_title("Bi-encoder — encode INDEPENDENTLY (precomputable, fast)", fontsize=10.5, color=BLUE)
    box(ax_bi, 0.05, 0.82, 0.38, 0.12, "query", AMBER)
    box(ax_bi, 0.57, 0.82, 0.38, 0.12, "passage", BLUE)
    box(ax_bi, 0.05, 0.55, 0.38, 0.16, "Encoder", PURPLE)
    box(ax_bi, 0.57, 0.55, 0.38, 0.16, "Encoder\n(same weights)", PURPLE)
    box(ax_bi, 0.05, 0.34, 0.38, 0.12, "q vector", AMBER)
    box(ax_bi, 0.57, 0.34, 0.38, 0.12, "d vector\n(precomputed once)", BLUE)
    box(ax_bi, 0.30, 0.10, 0.40, 0.13, "cosine(q, d) → score", GREEN)
    for x in (0.24, 0.76):
        ax_bi.annotate("", xy=(x, 0.71), xytext=(x, 0.82), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
        ax_bi.annotate("", xy=(x, 0.46), xytext=(x, 0.55), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax_bi.annotate("", xy=(0.45, 0.165), xytext=(0.24, 0.34), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax_bi.annotate("", xy=(0.55, 0.165), xytext=(0.76, 0.34), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax_bi.text(0.5, 0.015, "query & passage never interact", ha="center", fontsize=8.5, style="italic", color=RED)

    # --- cross-encoder: one joint tower ---
    ax_cross.set_title("Cross-encoder — encode TOGETHER (accurate, query-time only)", fontsize=10.5, color=PURPLE)
    box(ax_cross, 0.18, 0.82, 0.64, 0.12, "[CLS] query [SEP] passage", AMBER)
    box(ax_cross, 0.18, 0.52, 0.64, 0.20, "Encoder\n(joint self-attention:\nevery q token ↔ every d token)", PURPLE)
    box(ax_cross, 0.30, 0.28, 0.40, 0.12, "[CLS] vector", SLATE)
    box(ax_cross, 0.32, 0.08, 0.36, 0.12, "head → relevance logit", GREEN)
    ax_cross.annotate("", xy=(0.5, 0.72), xytext=(0.5, 0.82), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax_cross.annotate("", xy=(0.5, 0.40), xytext=(0.5, 0.52), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax_cross.annotate("", xy=(0.5, 0.20), xytext=(0.5, 0.28), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax_cross.text(0.5, 0.015, "one forward pass PER (query, passage) — cannot precompute",
                  ha="center", fontsize=8.5, style="italic", color=RED)
    _save(fig, "rag06_bi_vs_cross.png")


def fig_rank_shuffle(bi_full, reranked) -> None:
    """Each candidate's bi-encoder rank vs its re-ranked rank; the gold climbs to #1."""
    pool = list(reranked.indices)  # the re-ranked candidates
    bi_rank = {doc: bi_full.indices.index(doc) + 1 for doc in pool}
    rr_rank = {doc: reranked.indices.index(doc) + 1 for doc in pool}

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    _style_axis(ax)
    for doc in pool:
        is_gold = doc == GOLD_INDEX
        color = GREEN if is_gold else SLATE
        lw = 2.6 if is_gold else 1.2
        alpha = 0.95 if is_gold else 0.5
        ax.plot([0, 1], [bi_rank[doc], rr_rank[doc]], color=color, linewidth=lw, alpha=alpha,
                marker="o", markersize=8 if is_gold else 5, markeredgecolor=INK, zorder=3 if is_gold else 2)
        if is_gold:
            ax.annotate(f"GOLD  doc[{doc}]", (0, bi_rank[doc]), color=GREEN, fontsize=9.5, fontweight="bold",
                        ha="right", va="center", xytext=(-8, 0), textcoords="offset points")
            ax.annotate("→ #1", (1, rr_rank[doc]), color=GREEN, fontsize=9.5, fontweight="bold",
                        ha="left", va="center", xytext=(8, 0), textcoords="offset points")
    ax.axhline(FINAL_K + 0.5, color=RED, linewidth=1.4, linestyle="--", alpha=0.7)
    ax.annotate(f"top-{FINAL_K} cutoff", (0.5, FINAL_K + 0.5), color=RED, fontsize=8.5, ha="center",
                va="bottom", xytext=(0, 3), textcoords="offset points")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["bi-encoder\nrank", "re-ranked\nrank"], fontsize=10)
    ax.set_ylabel("rank  (1 = best, lower is better)")
    ax.set_ylim(0.3, len(pool) + 0.7)
    ax.invert_yaxis()
    ax.set_title("The re-rank shuffle: the cross-encoder lifts the gold from #4 to #1", fontsize=11.5, pad=10)
    ax.set_xlim(-0.35, 1.35)
    _save(fig, "rag06_rank_shuffle.png")


def fig_ndcg_mrr(bi_full, reranked) -> None:
    """nDCG@k and MRR, bi-encoder vs re-ranked — the measured lift."""
    metrics = [f"nDCG@{k}" for k in NDCG_KS] + ["MRR"]
    before = [ndcg_at_k(bi_full.indices, GOLD_INDEX, k) for k in NDCG_KS] + [reciprocal_rank(bi_full.indices, GOLD_INDEX)]
    after = [ndcg_at_k(reranked.indices, GOLD_INDEX, k) for k in NDCG_KS] + [reciprocal_rank(reranked.indices, GOLD_INDEX)]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    _style_axis(ax)
    x = np.arange(len(metrics))
    w = 0.38
    b1 = ax.bar(x - w / 2, before, w, label="bi-encoder (retrieve only)", color=BLUE, edgecolor=INK, linewidth=0.8)
    b2 = ax.bar(x + w / 2, after, w, label="+ cross-encoder re-rank", color=GREEN, edgecolor=INK, linewidth=0.8)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        fontsize=8.5, color=INK, ha="center", va="bottom", xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("score  (higher is better)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Re-ranking quality lift — nDCG@k and MRR, before vs after", fontsize=12, pad=10)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    _save(fig, "rag06_ndcg_mrr.png")


def fig_latency_vs_k() -> None:
    """Illustrative query-time cost vs pool size: bi-encoder compare flat; cross-encoder linear in k.

    Illustrative: uses a simple cost model (bi-encoder = one query encode + cheap dot products;
    cross-encoder = k transformer forward passes) to show the SHAPE of the tradeoff, not measured
    wall-clock. The point is the slope: cross-encoder cost grows linearly with the pool, which is
    why you keep the re-rank pool small.
    """
    ks = np.arange(1, 201)
    bi_cost = np.full_like(ks, 1.0, dtype=float)  # ~constant: one query encode + cheap compares
    per_pass_ms = 1.0  # illustrative cost of one cross-encoder forward pass (relative units)
    cross_cost = ks * per_pass_ms  # k joint forward passes

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _style_axis(ax)
    ax.plot(ks, bi_cost, color=BLUE, linewidth=2.4, label="bi-encoder compare (≈ constant)")
    ax.plot(ks, cross_cost, color=PURPLE, linewidth=2.4, label="cross-encoder re-rank (k forward passes)")
    for k in (RETRIEVE_K, 50, 100):
        ax.axvline(k, color=GRID, linewidth=1.0, linestyle=":")
        ax.annotate(f"k={k}", (k, cross_cost[k - 1]), color=INK, fontsize=8, ha="left", va="bottom",
                    xytext=(3, 2), textcoords="offset points")
    ax.axvspan(1, RETRIEVE_K, color=GREEN, alpha=0.07)
    ax.annotate("practical\nre-rank pool", (RETRIEVE_K / 2, 150), color=GREEN, fontsize=8.5, ha="center", va="center")
    ax.set_title("Why the re-rank pool stays small — cost grows linearly with k (illustrative)", fontsize=11, pad=10)
    ax.set_xlabel("pool size k (passages scored)")
    ax.set_ylabel("relative query-time cost")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 210)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    _save(fig, "rag06_latency_vs_k.png")


def fig_recall_ceiling(bi) -> None:
    """Retrieve top-3 (gold missed) vs top-10 (gold present) -> re-rank outcome. The recall ceiling."""
    cross = CrossEncoderReranker()
    fig, (ax_narrow, ax_wide) = plt.subplots(1, 2, figsize=(11.0, 4.8))

    for ax, pool_k, title in ((ax_narrow, FINAL_K, f"Retrieve top-{FINAL_K}, then re-rank"),
                              (ax_wide, RETRIEVE_K, f"Retrieve top-{RETRIEVE_K}, then re-rank")):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pool = bi.retrieve(QUERY, k=pool_k).indices
        reranked = cross.rerank(QUERY, pool, CORPUS).indices
        gold_present = GOLD_INDEX in pool
        ax.set_title(title, fontsize=10.5, color=INK)
        # show the pool, gold highlighted (or its absence)
        ax.text(0.5, 0.9, f"pool = {list(pool)}", ha="center", fontsize=9.5, color=INK, family="monospace")
        ax.text(0.5, 0.78, f"gold doc[{GOLD_INDEX}] in pool: {gold_present}", ha="center", fontsize=10,
                color=GREEN if gold_present else RED, fontweight="bold")
        # outcome box
        if gold_present:
            rank = reranked.index(GOLD_INDEX) + 1
            ax.add_patch(plt.Rectangle((0.12, 0.30), 0.76, 0.34, facecolor=GREEN, alpha=0.16, edgecolor=GREEN, linewidth=2))
            ax.text(0.5, 0.47, f"re-ranked → gold at #{rank}\n✓ recovered", ha="center", va="center",
                    fontsize=12, color=GREEN, fontweight="bold")
        else:
            ax.add_patch(plt.Rectangle((0.12, 0.30), 0.76, 0.34, facecolor=RED, alpha=0.16, edgecolor=RED, linewidth=2))
            ax.text(0.5, 0.47, "gold MISSING from pool\n✗ re-ranking cannot rescue it", ha="center",
                    va="center", fontsize=12, color=RED, fontweight="bold")
    fig.suptitle("The recall ceiling: re-ranking REORDERS the pool — it cannot retrieve what stage 1 missed",
                 fontsize=11.5, color=INK, y=1.01)
    _save(fig, "rag06_recall_ceiling.png")


def main() -> None:
    bi = BiEncoderRetriever(CORPUS)
    cross = CrossEncoderReranker()
    print(f"bi-encoder: {bi.backend} | cross-encoder: {cross.backend}")
    bi_full, reranked = retrieve_then_rerank(QUERY, CORPUS, bi, cross)
    fig_funnel()
    fig_bi_vs_cross()
    fig_rank_shuffle(bi_full, reranked)
    fig_ndcg_mrr(bi_full, reranked)
    fig_latency_vs_k()
    fig_recall_ceiling(bi)
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
