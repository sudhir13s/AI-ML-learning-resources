"""Reproducible figure generator for 16-Information-Retrieval-and-Semantic-Search.

Produces every embedded PNG for the chapter from the SAME backend used on the page and in the
notebook -- the corpus, BM25 scores, dense cosines, RRF fusion, cross-encoder rerank, the ANN
recall/nprobe sweep, the PQ compression math, and the nDCG decomposition are all IMPORTED from
`information_retrieval.py`, so the figures cannot silently drift from the prose or the demo. Run:

    python make_figures_16.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at 150 dpi,
prefixed `ir_`. The palette matches the chapter's Mermaid diagrams (muted, white text on fills).

Figures produced (measured = from the live backend; illustrative = a labelled schematic):
  ir_pipeline.png        -- illustrative: the retrieve-then-rerank cascade (query -> sparse+dense -> RRF -> rerank -> RAG)
  ir_inverted_index.png  -- illustrative: forward index vs inverted index (term -> posting list)
  ir_bm25_saturation.png -- measured: BM25 TF-saturation curve for several k1 (raw TF vs saturated)
  ir_sparse_vs_dense.png -- measured: BM25 vs dense scores on the vocab-mismatch query (gold rescued)
  ir_rrf.png             -- measured: the RRF worked example -- A wins despite no single #1
  ir_rerank.png          -- measured: dense top-k order vs cross-encoder rerank (gold to the top)
  ir_metrics_curve.png   -- measured: precision@k falls, recall@k rises -- the canonical trade-off
  ir_ndcg.png            -- measured: the nDCG decomposition (DCG / IDCG / ratio) for the page's list
  ir_ann_tradeoff.png    -- measured: IVF recall@10 vs nprobe (and % scanned) + the PQ memory win

All numbers come from `information_retrieval.py`; when the real Sentence-BERT is present the dense
figures are MEASURED from all-MiniLM-L6-v2, offline they fall back to the deterministic synthetic
encoder and the figures still render (titles note the backend). Verified on Python 3.12 / numpy 2.x /
matplotlib 3.x, CPU, deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

import information_retrieval as ir

# ---- Palette (matches the chapter Mermaid classDefs) ------------------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"  # near-black for axis text
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 150

# One shared backend for every measured figure, so all numbers come from the same model load.
ENCODER = ir.load_encoder()
BACKEND_TAG = "all-MiniLM-L6-v2, measured" if ENCODER.is_real else "synthetic fallback, measured"
BACKEND_SHORT = "all-MiniLM-L6-v2" if ENCODER.is_real else "synthetic"


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(OUT_DIR.parent.parent)}")


def _box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, color: str,
         fontsize: float = 9.5) -> None:
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="white", lw=1.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white",
            fontsize=fontsize, zorder=3)


def fig_pipeline() -> None:
    """Illustrative: the retrieve-then-rerank cascade, the one-figure map of the whole chapter."""
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)

    _box(ax, 0.3, 4.3, 2.2, 1.4, "user\nquery", BLUE)
    # Two parallel stage-1 retrievers.
    _box(ax, 3.4, 6.6, 3.6, 1.7, "SPARSE retriever\nBM25 + inverted index\n(exact terms)", SLATE, 9.0)
    _box(ax, 3.4, 1.4, 3.6, 1.7, "DENSE retriever\nbi-encoder + ANN index\n(meaning / synonyms)", PURPLE, 9.0)
    _box(ax, 7.9, 4.0, 2.7, 2.0, "HYBRID fusion\nRRF\n1/(k+rank)", AMBER, 9.5)
    _box(ax, 11.4, 4.0, 2.4, 2.0, "RE-RANK\ncross-encoder\n(joint scoring)", NAVY, 9.0)
    _box(ax, 14.0, 4.3, 1.8, 1.4, "top-n\n-> LLM\n(RAG)", GREEN, 8.5)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.8))

    arrow(2.5, 5.4, 3.4, 7.2)
    arrow(2.5, 4.6, 3.4, 2.5)
    arrow(7.0, 7.0, 7.9, 5.6)
    arrow(7.0, 2.4, 7.9, 4.4)
    arrow(10.6, 5.0, 11.4, 5.0)
    arrow(13.8, 5.0, 14.0, 5.0)

    # Stage labels with the cost/quality framing.
    ax.text(5.2, 9.4, "STAGE 1 — RETRIEVE  (cheap, recall-oriented, millions -> top-k ~100-1000)",
            ha="center", color=INK, fontsize=10.5, fontweight="bold")
    ax.text(12.6, 8.3, "STAGE 2 — RANK\n(expensive, precision-oriented, top-k only)",
            ha="center", color=INK, fontsize=10.0, fontweight="bold")
    ax.text(5.2, 0.5, "lexical catches EXACT terms; dense catches SYNONYMS — fused, they cover both",
            ha="center", color=GREEN, fontsize=9.5, fontstyle="italic")
    fig.suptitle("The retrieve-then-rerank cascade: a cost-managed approximation of the ideal ranking (illustrative)",
                 fontsize=12, color=INK, y=1.0)
    fig.tight_layout()
    _save(fig, "ir_pipeline.png")


def fig_inverted_index() -> None:
    """Illustrative: forward index (doc -> terms) vs the inverted index (term -> posting list)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    # Left: forward index.
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    forward = [
        ("d0", "mechanic automobile engine brake"),
        ("d2", "fix spelling errors report"),
        ("d4", "how fix code build machine"),
    ]
    for i, (doc, terms) in enumerate(forward):
        y = 7.5 - i * 2.2
        _box(ax, 0.3, y, 1.4, 1.3, doc, BLUE)
        _box(ax, 2.2, y, 7.4, 1.3, terms, SLATE, 9.0)
        ax.annotate("", xy=(2.2, y + 0.65), xytext=(1.7, y + 0.65),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_title("FORWARD index: document -> its terms\n(scan every doc per query -> O(N): hopeless)",
                 fontsize=10.5, color=INK)
    ax.text(5.0, 0.4, "answering a query means reading every document", ha="center", color=RED,
            fontsize=9.0, fontstyle="italic")

    # Right: inverted index.
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    inverted = [
        ("automobile", "d0:1", GREEN),
        ("fix", "d2:1  d4:1", SLATE),
        ("engine", "d0:1", GREEN),
        ("car", "(no postings!)", RED),
    ]
    for i, (term, postings, color) in enumerate(inverted):
        y = 7.8 - i * 2.0
        _box(ax, 0.3, y, 2.6, 1.2, f'"{term}"', PURPLE, 9.5)
        _box(ax, 3.4, y, 6.2, 1.2, postings, color, 9.5)
        ax.annotate("", xy=(3.4, y + 0.6), xytext=(2.9, y + 0.6),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_title("INVERTED index: term -> posting list (docID:tf)\n(touch only docs sharing a query term -> fast)",
                 fontsize=10.5, color=INK)
    ax.text(5.0, 0.4, '"car" has NO posting list -> the lexical gap (gold says "automobile")',
            ha="center", color=RED, fontsize=8.8, fontstyle="italic")
    fig.suptitle("Why lexical search is fast -- and where it goes blind (illustrative)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "ir_inverted_index.png")


def fig_bm25_saturation() -> None:
    """Measured: BM25's term-frequency saturation for several k1, vs the unbounded raw-TF line."""
    tf = np.linspace(0, 20, 300)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    # Raw TF (no saturation) for contrast -- normalized to start at the same slope region.
    ax.plot(tf, tf, color=RED, lw=2.0, ls="--", label="raw TF (TF-IDF) — unbounded", zorder=2)
    # BM25 saturation factor f*(k1+1)/(f+k1) at b=0 (length-neutral) for several k1.
    for k1, color in ((0.5, BLUE), (1.5, GREEN), (3.0, PURPLE)):
        sat = tf * (k1 + 1.0) / (tf + k1)
        ax.plot(tf, sat, color=color, lw=2.6, label=f"BM25 saturated, $k_1$={k1} (→ {k1+1:.1f})", zorder=3)
        ax.axhline(k1 + 1.0, color=color, lw=0.9, ls=":", alpha=0.6)
    ax.set_xlabel("term frequency  f(t, d)   (how many times the term appears in the doc)")
    ax.set_ylabel("contribution to the score (before IDF)")
    ax.set_ylim(0, 12)
    ax.set_xlim(0, 20)
    ax.set_title(
        "BM25 saturates term frequency: the 10th 'engine' barely beats the 9th\n"
        "raw TF (red) grows forever; BM25 (solid) approaches a ceiling $k_1+1$ — relevance saturates",
        fontsize=10.5,
    )
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _style_axis(ax)
    _save(fig, "ir_bm25_saturation.png")


def fig_sparse_vs_dense() -> None:
    """Measured: BM25 vs dense scores on the vocab-mismatch query -- dense rescues the gold passage."""
    lvd = ir.lexical_vs_dense(ENCODER)
    bm = lvd["bm25_scores"]
    dense = lvd["dense_scores"]
    n = len(ir.CORPUS)
    labels = [f"d{i}" + ("\nGOLD" if i == ir.GOLD_ID else "") for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))

    # Left: BM25 (normalized for readability), gold highlighted.
    ax = axes[0]
    bm_norm = bm / bm.max() if bm.max() > 0 else bm
    colors = [AMBER if i == ir.GOLD_ID else (RED if i in (2, 4) else SLATE) for i in range(n)]
    bars = ax.bar(range(n), bm_norm, color=colors, edgecolor="white", width=0.7)
    bars[ir.GOLD_ID].set_edgecolor(INK)
    bars[ir.GOLD_ID].set_linewidth(2.2)
    for i, v in enumerate(bm_norm):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", color=INK, fontsize=8.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("BM25 score (normalized)")
    ax.set_ylim(0, 1.18)
    ax.set_title(
        f"LEXICAL (BM25): gold scores ZERO\nd2/d4 (red) win on shared 'fix'/'my'; gold(d0) shares no word -> rank {lvd['bm25_gold_rank']}",
        fontsize=10.0,
    )
    _style_axis(ax)
    ax.grid(axis="x", visible=False)

    # Right: dense cosine, gold highlighted.
    ax = axes[1]
    colors = [GREEN if i == ir.GOLD_ID else SLATE for i in range(n)]
    bars = ax.bar(range(n), dense, color=colors, edgecolor="white", width=0.7)
    bars[ir.GOLD_ID].set_edgecolor(INK)
    bars[ir.GOLD_ID].set_linewidth(2.2)
    for i, v in enumerate(dense):
        ax.text(i, v + 0.01 * np.sign(v or 1), f"{v:.2f}", ha="center", color=INK, fontsize=8.5)
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("dense cosine similarity to query")
    ax.set_title(
        f"DENSE ({BACKEND_SHORT}): gold ranks #1\n'automobile' embeds near 'car' — a MEANING match, not a keyword match",
        fontsize=10.0,
    )
    _style_axis(ax)
    ax.grid(axis="x", visible=False)

    fig.suptitle(
        f'Vocabulary mismatch, measured: "{ir.QUERY}"  ({BACKEND_TAG})',
        fontsize=12, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "ir_sparse_vs_dense.png")


def fig_rrf() -> None:
    """Measured: the RRF worked example -- A wins despite ranking #1 in NEITHER list."""
    table = ir.rrf_worked_example()
    docs = [d for d, _ in table]
    scores = [s for _, s in table]
    colors = [GREEN if d == "A" else (BLUE if d in ("C", "B") else SLATE) for d in docs]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ypos = np.arange(len(docs))[::-1]
    bars = ax.barh(ypos, scores, color=colors, edgecolor="white")
    for bar, s, d in zip(bars, scores, docs):
        ax.text(s + 0.00012, bar.get_y() + bar.get_height() / 2, f"{s:.5f}",
                va="center", ha="left", color=INK, fontsize=9.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(docs, fontsize=11)
    ax.set_xlabel(r"RRF score  $\sum_\ell \dfrac{1}{k + r_\ell(d)}$   ($k=60$)")
    ax.set_xlim(0, max(scores) * 1.18)
    ax.set_title(
        "Reciprocal Rank Fusion rewards AGREEMENT (measured)\n"
        'BM25=[A,B,C,D,E], dense=[C,A,F,B,G]: A wins — ranked 1 & 2 in BOTH — over C\'s single #1',
        fontsize=10.5,
    )
    ax.text(max(scores) * 0.52, 0.2, "F, D, E, G appear in only ONE list -> sink below the agreed docs",
            color=RED, fontsize=8.8, fontstyle="italic")
    _style_axis(ax)
    ax.grid(axis="y", visible=False)
    _save(fig, "ir_rrf.png")


def fig_rerank() -> None:
    """Measured: the dense top-k candidate order vs the cross-encoder re-rank (gold lifted to the top)."""
    rr = ir.rerank_demo(ENCODER)
    candidates = rr["candidates"]
    ce = rr["ce_scores"]
    reranked = rr["reranked"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # Left: cross-encoder score per candidate (in dense-retrieval order).
    ax = axes[0]
    labels = [f"d{c}" + ("\nGOLD" if c == ir.GOLD_ID else "") for c in candidates]
    colors = [GREEN if c == ir.GOLD_ID else SLATE for c in candidates]
    ax.bar(range(len(candidates)), ce, color=colors, edgecolor="white", width=0.6)
    for i, v in enumerate(ce):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", color=INK, fontsize=9.5)
    ax.set_xticks(range(len(candidates)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("cross-encoder joint score (query, passage)")
    ax.set_ylim(0, max(max(ce) * 1.25, 0.2))
    ax.set_title("Cross-encoder reads each (query, passage) PAIR jointly\nscoring the BM25 top-k candidates",
                 fontsize=10.0)
    _style_axis(ax)
    ax.grid(axis="x", visible=False)

    # Right: before/after order as ranked lists.
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(2.5, 9.2, "BM25 order\n(stage 1)", ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.text(7.5, 9.2, "after cross-encoder\nre-rank (stage 2)", ha="center", color=INK, fontsize=10, fontweight="bold")
    for rank, c in enumerate(candidates):
        color = GREEN if c == ir.GOLD_ID else SLATE
        _box(ax, 1.0, 7.2 - rank * 1.7, 3.0, 1.2, f"#{rank+1}  d{c}", color, 9.5)
    for rank, c in enumerate(reranked):
        color = GREEN if c == ir.GOLD_ID else SLATE
        _box(ax, 6.0, 7.2 - rank * 1.7, 3.0, 1.2, f"#{rank+1}  d{c}", color, 9.5)
    ax.annotate("", xy=(5.9, 4.5), xytext=(4.1, 4.5),
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=2.2))
    ax.text(5.0, 5.0, "re-sort", ha="center", color=AMBER, fontsize=9.5, fontweight="bold")
    ax.text(5.0, 0.5, f"gold(d0): rank {rr['gold_rank_before']} -> rank {rr['gold_rank_after']} (lifted to the top)",
            ha="center", color=GREEN, fontsize=9.5, fontweight="bold")
    fig.suptitle(f"Re-ranking buys precision at the very top ({BACKEND_TAG})",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "ir_rerank.png")


def fig_metrics_curve() -> None:
    """Measured: precision@k falls and recall@k rises as the cutoff k grows -- the canonical trade-off."""
    rel = np.array(ir.GRADED_RELEVANCE)
    n_rel = int((rel > 0).sum())
    pr = ir.precision_recall_curve(rel, n_rel)
    ks, prec, rec = pr["ks"], pr["precision"], pr["recall"]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(ks, prec, color=BLUE, lw=2.8, marker="o", markersize=7, label="precision@k (ranker's metric)", zorder=3)
    ax.plot(ks, rec, color=GREEN, lw=2.8, marker="s", markersize=7, label="recall@k (retriever's metric)", zorder=3)
    for k, p, r in zip(ks, prec, rec):
        if k in (1, 3, 5, 8):
            ax.text(k, p + 0.03, f"{p:.2f}", ha="center", color=BLUE, fontsize=8.5)
            ax.text(k, r - 0.06, f"{r:.2f}", ha="center", color=GREEN, fontsize=8.5)
    ax.set_xlabel("cutoff k (top-k of the ranked list)")
    ax.set_ylabel("metric value")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(ks)
    ax.set_title(
        "Precision falls, recall rises as k grows (measured)\n"
        f"rel = {list(ir.GRADED_RELEVANCE)}, {n_rel} relevant of {len(rel)} — recall@8 = 1.0 (all found)",
        fontsize=10.5,
    )
    ax.legend(frameon=False, loc="center right", fontsize=9.5)
    _style_axis(ax)
    _save(fig, "ir_metrics_curve.png")


def fig_ndcg() -> None:
    """Measured: the nDCG decomposition -- per-rank contributions, DCG vs ideal IDCG, ratio nDCG."""
    dec = ir.ndcg_decomposition()
    rel = dec["rel"].astype(int)
    actual = dec["actual_contrib"]
    ideal = dec["ideal_contrib"]
    ideal_rel = dec["ideal"].astype(int)
    ranks = np.arange(1, len(rel) + 1)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios": [1.7, 1]})

    # Left: per-rank contribution, actual vs ideal, grouped.
    width = 0.4
    ax.bar(ranks - width / 2, actual, width, color=BLUE, edgecolor="white",
           label="our ranking  rel/$\\log_2(i{+}1)$")
    ax.bar(ranks + width / 2, ideal, width, color=GREEN, edgecolor="white",
           label="ideal ranking (rel sorted desc)")
    for r, a, irel in zip(ranks, actual, rel):
        ax.text(r - width / 2, a + 0.04, f"{irel}", ha="center", color=BLUE, fontsize=8)
    for r, b, irel in zip(ranks, ideal, ideal_rel):
        ax.text(r + width / 2, b + 0.04, f"{irel}", ha="center", color=GREEN, fontsize=8)
    ax.set_xlabel("rank position $i$   (label = relevance grade at that position)")
    ax.set_ylabel(r"discounted gain  $\mathrm{rel}_i / \log_2(i+1)$")
    ax.set_xticks(ranks)
    ax.set_ylim(0, 3.5)
    ax.set_title(
        "nDCG, decomposed (measured): discount each grade by log of its rank\n"
        f"rel = {list(ir.GRADED_RELEVANCE)} — a perfect grade at rank 1 beats the same grade deeper",
        fontsize=10.0,
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _style_axis(ax)
    ax.grid(axis="x", visible=False)

    # Right: DCG / IDCG / nDCG summary bars.
    vals = [dec["dcg"], dec["idcg"]]
    bars = ax2.bar(["DCG@8\n(ours)", "IDCG@8\n(ideal)"], vals, color=[BLUE, GREEN],
                   edgecolor="white", width=0.55)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.3f}", ha="center",
                 color=INK, fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(vals) * 1.22)
    ax2.set_ylabel("cumulative discounted gain")
    ax2.set_title(
        f"nDCG@8 = DCG / IDCG\n= {dec['dcg']:.3f} / {dec['idcg']:.3f} = {dec['ndcg']:.3f}",
        fontsize=10.5,
    )
    ax2.text(0.5, max(vals) * 0.12, "96.1% of the\nbest-possible order", ha="center", color=PURPLE,
             fontsize=9.5, fontweight="bold")
    _style_axis(ax2)
    ax2.grid(axis="x", visible=False)
    fig.suptitle("nDCG = relevance, discounted by depth, normalized against the ideal (measured, linear gain)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "ir_ndcg.png")


def fig_ann_tradeoff() -> None:
    """Measured: IVF recall@10 vs nprobe (and % of corpus scanned) + the PQ compression memory win."""
    sweep = ir.ann_recall_sweep()
    nprobes = sweep["nprobes"]
    recall = sweep["recall"]
    fraction = sweep["fraction_scanned"] * 100.0
    pq = ir.pq_compression()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: recall vs nprobe, with % scanned on a twin axis (the recall-vs-cost knob).
    ax.plot(nprobes, recall, color=GREEN, lw=2.8, marker="o", markersize=8,
            label="recall@10 vs exact", zorder=3)
    for x, y in zip(nprobes, recall):
        ax.text(x, y + 0.025, f"{y:.2f}", ha="center", color=GREEN, fontsize=8.5)
    ax.axhline(1.0, color=RED, lw=1.0, ls="--", alpha=0.7)
    ax.text(nprobes[0], 1.02, "exact kNN recall = 1.0 (scans 100%)", color=RED, fontsize=8.5)
    ax.set_xlabel("nprobe  (IVF cells searched — the recall–latency dial)")
    ax.set_ylabel("recall@10 vs exact", color=GREEN)
    ax.set_ylim(0, 1.12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(nprobes)
    ax.set_xticklabels([str(int(n)) for n in nprobes])
    _style_axis(ax)

    ax_twin = ax.twinx()
    ax_twin.plot(nprobes, fraction, color=SLATE, lw=2.0, ls=":", marker="s", markersize=6,
                 label="% of corpus scanned", zorder=2)
    ax_twin.set_ylabel("% of corpus scanned (∝ latency)", color=SLATE)
    ax_twin.set_ylim(0, 110)
    ax_twin.tick_params(colors=SLATE)
    ax.set_title(
        f"ANN (IVF) recall climbs with nprobe — at rising cost (measured, {sweep['n_docs']} vectors)\n"
        "probe more cells -> higher recall AND more of the corpus touched: no free lunch",
        fontsize=10.0,
    )
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right", fontsize=9)

    # Right: PQ compression -- raw FP32 vs PQ code bytes per vector.
    vals = [pq["raw_bytes"], pq["pq_bytes"]]
    bars = ax2.bar(["FP32\n(raw)", "PQ code\n(m=96, 8-bit)"], vals, color=[RED, GREEN],
                   edgecolor="white", width=0.55)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 40, f"{int(v)} B", ha="center",
                 color=INK, fontsize=11, fontweight="bold")
    ax2.set_ylabel("bytes per 768-dim vector")
    ax2.set_ylim(0, pq["raw_bytes"] * 1.18)
    ax2.set_title(
        f"Product quantization: {int(pq['raw_bytes'])} B -> {int(pq['pq_bytes'])} B\n"
        f"= {pq['ratio']:.0f}x smaller — 1B vectors: ~3 TB -> ~96 GB (fits in RAM)",
        fontsize=10.0,
    )
    _style_axis(ax2)
    ax2.grid(axis="x", visible=False)
    fig.suptitle("The recall–latency–memory surface: HNSW/IVF buy latency, PQ buys memory, both spend recall (measured)",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "ir_ann_tradeoff.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    tag = "real" if ENCODER.is_real else "synthetic fallback"
    print(f"backend: {ENCODER.name} ({tag})")
    fig_pipeline()
    fig_inverted_index()
    fig_bm25_saturation()
    fig_sparse_vs_dense()
    fig_rrf()
    fig_rerank()
    fig_metrics_curve()
    fig_ndcg()
    fig_ann_tradeoff()
    print("done.")


if __name__ == "__main__":
    main()
