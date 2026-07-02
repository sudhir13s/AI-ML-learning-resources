"""Figure generator for 06-Re-ranking-Cross-Encoders — REAL measurements + honest schematics.

Every *measured* figure comes from the SAME real pipeline the page and notebook use (`reranking.py`
over the real BeIR/scifact benchmark): real bi-encoder retrieval, real cross-encoder reranking, real
nDCG@10 / MRR@10 / Recall / Precision against real relevance labels. It reads the cached per-query
orderings (`data/scifact_orderings.npz`) and the summary produced by `python reranking.py`, so it
never repeats the tens of thousands of cross-encoder forward passes. The few *schematic* figures (the
funnel, the bi-vs-cross tower diagram) are clearly labelled illustrative.

Writes muted-palette PNGs to the shared chapter image dir (../../images/) with prefix `rag06_`.

    python reranking.py          # first — produces the cached orderings + summary.json
    python make_figures_06.py    # then — draws the figures from that real run

Figures produced:
  rag06_ndcg_mrr.png       -- REAL aggregate nDCG@10/MRR@10/Recall@10/Precision@10, bi vs reranked.
  rag06_quality_vs_k.png   -- REAL nDCG@10 vs rerank-depth K (the quality curve + the knee).
  rag06_latency_vs_k.png   -- REAL per-query latency: retrieve vs rerank cost as K grows.
  rag06_score_separation.png-- REAL cross-encoder score distribution for gold vs non-gold candidates.
  rag06_rank_shuffle.png   -- REAL hero query: the buried gold the reranker lifts most, bi vs reranked rank.
  rag06_funnel.png         -- SCHEMATIC two-stage funnel (corpus -> retrieve top-K -> rerank top-few).
  rag06_bi_vs_cross.png    -- SCHEMATIC bi-encoder (independent towers) vs cross-encoder (joint tower).

Device-agnostic (only needed for the score-separation figure, which re-scores one query's pool);
matplotlib headless (Agg). Verified on Python 3.12 / matplotlib 3.x / numpy 2.x.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from reranking import (
    DATA_DIR,
    K_SWEEP,
    METRIC_K,
    RETRIEVE_K,
    CrossEncoderReranker,
    hero_query,
    load_orderings,
    load_scifact,
    sweep_rerank_depth,
)

# ---- Palette (matches the chapter family's muted Mermaid classDefs) -----------------------------
BLUE = "#3A6B96"  # data / bi-encoder
PURPLE = "#5D4A8A"  # process / cross-encoder
GREEN = "#2E7A5A"  # gold / win
RED = "#8B3B4A"  # miss / danger / non-gold
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


# ============================ measured figures (from the real pipeline) ==========================
def fig_ndcg_mrr(summary: dict) -> None:
    """REAL aggregate metrics bi vs reranked over all scifact test queries (from summary.json)."""
    pool_metric = f"Recall@{RETRIEVE_K}"
    metrics = ["nDCG@10", "MRR@10", "Recall@10", pool_metric]
    before = [summary["metrics"][m]["bi"] for m in metrics]
    after = [summary["metrics"][m]["reranked"] for m in metrics]

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    _style_axis(ax)
    x = np.arange(len(metrics))
    w = 0.38
    b1 = ax.bar(x - w / 2, before, w, label="bi-encoder (retrieve only)", color=BLUE, edgecolor=INK, linewidth=0.8)
    b2 = ax.bar(x + w / 2, after, w, label="+ cross-encoder re-rank", color=GREEN, edgecolor=INK, linewidth=0.8)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        fontsize=8.5, color=INK, ha="center", va="bottom", xytext=(0, 2), textcoords="offset points")
    # delta annotations
    for xi, m in enumerate(metrics):
        d = summary["metrics"][m]["delta"]
        ax.annotate(f"{d:+.3f}", (xi, max(before[xi], after[xi]) + 0.06), color=GREEN if d > 0 else SLATE,
                    fontsize=8.5, ha="center", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("score (higher is better)")
    ax.set_ylim(0, 1.12)
    n = summary["n_queries"]
    ax.set_title(f"REAL re-ranking lift on BeIR/scifact ({n} queries): ranking metrics rise, "
                 f"pool recall@{RETRIEVE_K} is the flat ceiling",
                 fontsize=10, pad=10)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    _save(fig, "rag06_ndcg_mrr.png")


def fig_quality_vs_k(data, bi_orders, rr_orders) -> None:
    """REAL nDCG@10 vs rerank depth K — the quality curve and where it plateaus (the knee)."""
    sweep = sweep_rerank_depth(data, bi_orders, rr_orders)
    ks = list(sweep.keys())
    bi_ndcg = [sweep[k][0] for k in ks]
    rr_ndcg = [sweep[k][1] for k in ks]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    _style_axis(ax)
    ax.axhline(bi_ndcg[0], color=BLUE, linewidth=1.8, linestyle="--", alpha=0.8,
               label=f"bi-encoder only (nDCG@10 = {bi_ndcg[0]:.3f})")
    ax.plot(ks, rr_ndcg, marker="o", color=GREEN, linewidth=2.4, markersize=7, markeredgecolor=INK,
            label="rerank top-K then keep the tail")
    for k, r in zip(ks, rr_ndcg):
        ax.annotate(f"{r:.3f}", (k, r), fontsize=8, color=INK, ha="center", va="bottom",
                    xytext=(0, 7), textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("rerank depth K (how many top candidates the cross-encoder re-scores) — log scale")
    ax.set_ylabel("nDCG@10 (higher is better)")
    ax.set_title("REAL quality vs rerank depth: most of the lift comes from a shallow pool", fontsize=10.5, pad=10)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    _save(fig, "rag06_quality_vs_k.png")


def fig_latency_vs_k(summary: dict) -> None:
    """REAL per-query latency: the flat-cheap retrieve vs the rerank cost that grows linearly with K.

    The measured retrieve latency and the measured rerank latency at K=RETRIEVE_K give the real
    per-pass cross-encoder cost; we extrapolate the line across K (rerank is linear in the number of
    pairs) and mark the real measured endpoint.
    """
    ret_ms = summary["retrieve_ms"]
    rr_ms_full = summary["rerank_ms"]
    per_pass = rr_ms_full / RETRIEVE_K  # real ms per cross-encoder pair, from the measured full-pool cost
    ks = np.array(K_SWEEP)
    rr_line = ks * per_pass

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    _style_axis(ax)
    ax.axhline(ret_ms, color=BLUE, linewidth=2.2, linestyle="--",
               label=f"stage-1 retrieve (≈ {ret_ms:.2f} ms, ≈ constant)")
    ax.plot(ks, rr_line, marker="o", color=PURPLE, linewidth=2.4, markersize=7, markeredgecolor=INK,
            label=f"stage-2 rerank (~{per_pass:.1f} ms/candidate × K)")
    ax.scatter([RETRIEVE_K], [rr_ms_full], s=120, color=RED, edgecolors=INK, zorder=5,
               label=f"measured at K={RETRIEVE_K}: {rr_ms_full:.0f} ms")
    ax.set_xlabel("rerank depth K (candidates scored by the cross-encoder)")
    ax.set_ylabel(f"per-query latency (ms, {summary['device']})")
    ax.set_title("REAL latency: retrieve is ~flat, rerank grows linearly with K (why the pool stays small)",
                 fontsize=10.5, pad=10)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    _save(fig, "rag06_latency_vs_k.png")


def fig_score_separation(data, bi_orders) -> None:
    """REAL cross-encoder relevance-logit distribution for gold vs non-gold candidates.

    Re-scores every query's retrieved pool with the real cross-encoder and splits the logits by
    whether the candidate is truly relevant (in the gold set). The separation between the two
    distributions IS why reranking works: the cross-encoder scores gold passages far higher.
    """
    cross = CrossEncoderReranker()
    gold_scores, other_scores = [], []
    for qi in range(min(60, data.n_queries)):  # a subset keeps the figure fast but real
        pool = bi_orders[qi]
        logits = cross.scores(data.query_texts[qi], [data.doc_texts[i] for i in pool])
        for idx, s in zip(pool, logits):
            (gold_scores if int(idx) in data.gold[qi] else other_scores).append(float(s))

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    _style_axis(ax)
    bins = np.linspace(min(other_scores + gold_scores), max(other_scores + gold_scores), 45)
    ax.hist(other_scores, bins=bins, color=RED, alpha=0.7, edgecolor=INK, linewidth=0.4,
            label=f"non-relevant candidates (n={len(other_scores):,})", density=True)
    ax.hist(gold_scores, bins=bins, color=GREEN, alpha=0.8, edgecolor=INK, linewidth=0.4,
            label=f"relevant / gold candidates (n={len(gold_scores):,})", density=True)
    ax.axvline(np.median(gold_scores), color=GREEN, linewidth=1.8, linestyle="--")
    ax.axvline(np.median(other_scores), color=RED, linewidth=1.8, linestyle="--")
    ax.set_xlabel("cross-encoder relevance logit (uncalibrated)")
    ax.set_ylabel("density")
    ax.set_title("REAL: the cross-encoder scores gold candidates far higher — why re-ranking works",
                 fontsize=10.5, pad=10)
    ax.legend(loc="upper center", framealpha=0.95, fontsize=8.5)
    _save(fig, "rag06_score_separation.png")


def fig_rank_shuffle(data, bi_orders, rr_orders) -> None:
    """REAL hero query: the buried gold the reranker lifts most, bi-encoder rank vs reranked rank."""
    qi = hero_query(data, bi_orders, rr_orders)
    gold = data.gold[qi]
    bi_ord = [int(x) for x in bi_orders[qi]]
    rr_ord = [int(x) for x in rr_orders[qi]]
    show = 10  # show the top-10 of each side
    pool = rr_ord[:show]
    bi_rank = {d: (bi_ord.index(d) + 1 if d in bi_ord else None) for d in pool}
    rr_rank = {d: rr_ord.index(d) + 1 for d in pool}

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    _style_axis(ax)
    for d in pool:
        if bi_rank[d] is None:
            continue
        is_gold = d in gold
        color = GREEN if is_gold else SLATE
        lw = 2.6 if is_gold else 1.1
        alpha = 0.95 if is_gold else 0.4
        ax.plot([0, 1], [bi_rank[d], rr_rank[d]], color=color, linewidth=lw, alpha=alpha,
                marker="o", markersize=8 if is_gold else 5, markeredgecolor=INK, zorder=3 if is_gold else 2)
        if is_gold:
            ax.annotate(f"gold #{bi_rank[d]}", (0, bi_rank[d]), color=GREEN, fontsize=9.5, fontweight="bold",
                        ha="right", va="center", xytext=(-8, 0), textcoords="offset points")
            ax.annotate(f"→ #{rr_rank[d]}", (1, rr_rank[d]), color=GREEN, fontsize=9.5, fontweight="bold",
                        ha="left", va="center", xytext=(8, 0), textcoords="offset points")
    ax.axhline(METRIC_K + 0.5, color=RED, linewidth=1.2, linestyle=":", alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["bi-encoder\nrank", "re-ranked\nrank"], fontsize=10)
    ax.set_ylabel("rank (1 = best, lower is better)")
    ax.set_ylim(0.3, show + 0.7)
    ax.invert_yaxis()
    q_preview = data.query_texts[qi][:54]
    ax.set_title(f"REAL hero query: the cross-encoder lifts the buried gold\n\"{q_preview}...\"",
                 fontsize=10, pad=10)
    ax.set_xlim(-0.4, 1.4)
    _save(fig, "rag06_rank_shuffle.png")


# ============================ schematic figures (illustrative) ===================================
def fig_funnel() -> None:
    """SCHEMATIC two-stage funnel: corpus -> bi-encoder top-K pool -> cross-encoder final few."""
    stages = [("CORPUS\n5,183 passages", 1.0, BLUE),
              (f"RETRIEVE (bi-encoder)\ntop-{RETRIEVE_K}", 0.42, PURPLE),
              ("RE-RANK (cross-encoder)\ntop-10 → generator", 0.16, GREEN)]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axis("off")
    y = 0.0
    centers = []
    for label, w, color in stages:
        x0 = (1.0 - w) / 2
        ax.add_patch(plt.Rectangle((x0, y), w, 0.8, facecolor=color, alpha=0.22, edgecolor=color, linewidth=2))
        ax.text(0.5, y + 0.4, label, ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold")
        centers.append((0.5, y))
        y -= 1.15
    for (cx, cy0), (_, cy1) in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=(cx, cy1 + 0.8), xytext=(cx, cy0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
    ax.text(0.93, -0.18, "cheap, precomputable\n(independent encodings)", ha="left", va="center",
            fontsize=8.5, color=PURPLE, style="italic")
    ax.text(0.93, -1.33, "expensive, query-time only\n(joint encoding, K forward passes)", ha="left",
            va="center", fontsize=8.5, color=GREEN, style="italic")
    ax.set_xlim(-0.05, 1.7)
    ax.set_ylim(-2.5, 0.95)
    ax.set_title("SCHEMATIC — two-stage retrieval: cheap recall, then expensive precision", fontsize=11.5, pad=10)
    _save(fig, "rag06_funnel.png")


def fig_bi_vs_cross() -> None:
    """SCHEMATIC: bi-encoder (two independent towers -> cosine) vs cross-encoder (one joint tower)."""
    fig, (ax_bi, ax_cross) = plt.subplots(1, 2, figsize=(11.2, 5.2))
    for ax in (ax_bi, ax_cross):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def box(ax, x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.2, edgecolor=color, linewidth=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, color=INK)

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


def main() -> None:
    summary_path = DATA_DIR / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run `python reranking.py` first to produce data/summary.json + orderings.")
    summary = json.loads(summary_path.read_text("utf-8"))
    data = load_scifact()
    orderings = load_orderings()
    if orderings is None:
        raise FileNotFoundError("Cached orderings missing — run `python reranking.py` first.")
    bi_orders, rr_orders = orderings

    # measured figures
    fig_ndcg_mrr(summary)
    fig_quality_vs_k(data, bi_orders, rr_orders)
    fig_latency_vs_k(summary)
    fig_score_separation(data, bi_orders)
    fig_rank_shuffle(data, bi_orders, rr_orders)
    # schematic figures
    fig_funnel()
    fig_bi_vs_cross()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
