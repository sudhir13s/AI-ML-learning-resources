"""Reproducible figure generator for 03-Bag-of-Words-and-TF-IDF.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
notebook -- the corpus, the TF-IDF math, the BM25 scorer, and every constant are IMPORTED from
`bow_tfidf.py`, so the figures cannot silently drift from the prose or the demo. Run:

    python make_figures_03.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at 150 dpi,
prefixed `bow_`. The palette matches the chapter's Mermaid diagrams (muted, white text on fills).

Figures produced:
  bow_term_document_matrix.png  -- the raw BoW count matrix as a heatmap (the core representation)
  bow_tf_variants.png           -- raw vs binary vs log-normalized TF across counts 1..50 (saturation)
  bow_idf_curve.png             -- IDF vs document frequency for N=1000 (plain log(N/df) + smoothed)
  bow_tfidf_vs_counts.png       -- which terms get up/down-weighted by TF-IDF relative to raw counts
  bow_tfidf_matrix_heatmap.png  -- the L2-normalized TF-IDF document-term matrix (heatmap)
  bow_cosine_heatmap.png        -- pairwise cosine similarity between the three documents
  bow_bm25_saturation.png       -- BM25 TF saturation vs unbounded raw TF (resists keyword stuffing)
  bow_retrieval_ranking.png     -- query "happy cat": cosine TF-IDF vs BM25 ranking, side by side
  bow_sparsity.png              -- how a BoW matrix's sparsity (% zeros) grows with vocabulary size

Verified on Python 3.12 / numpy 2.x / matplotlib 3.x. Pure NumPy, CPU, deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from bow_tfidf import (
    BM25_EXTRA_DOCS,
    BM25_K1,
    CORPUS,
    IDF_CURVE_N,
    QUERY,
    TF_DEMO_COUNTS,
    bm25_scores,
    bm25_tf_component,
    build_vocabulary,
    cosine_similarity_matrix,
    count_matrix,
    log_normalized_tf,
    smoothed_idf,
    tfidf_matrix,
    tfidf_query_scores,
)

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

DOC_LABELS = ("D1", "D2", "D3")


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


def fig_term_document_matrix() -> None:
    """Raw BoW count matrix as a heatmap — the central object of classic text mining."""
    vocab = build_vocabulary(CORPUS)
    counts = count_matrix(CORPUS, vocab)
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    im = ax.imshow(counts, cmap="Blues", aspect="auto", vmin=0, vmax=counts.max())
    ax.set_xticks(range(len(vocab)))
    ax.set_xticklabels(vocab, rotation=35, ha="right")
    ax.set_yticks(range(len(CORPUS)))
    ax.set_yticklabels(DOC_LABELS)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            val = counts[i, j]
            ax.text(
                j, i, str(val), ha="center", va="center",
                color="white" if val >= 1 else SLATE, fontsize=11, fontweight="bold",
            )
    ax.set_title("Bag-of-Words document-term matrix (raw counts)\n'the' dominates every row; order is gone", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="count")
    ax.set_xlabel("vocabulary term")
    _save(fig, "bow_term_document_matrix.png")


def fig_tf_variants() -> None:
    """Raw vs binary vs log-normalized TF across a wide count range — the saturation property."""
    counts = np.arange(1, 51)
    raw = counts.astype(float)
    binary = np.ones_like(counts, dtype=float)
    log_norm = np.array([log_normalized_tf(c) for c in counts])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(counts, raw, color=RED, lw=2.2, label="raw count  (linear, unbounded)")
    ax.plot(counts, log_norm, color=GREEN, lw=2.4, label=r"log-normalized  $1+\ln f$  (saturates)")
    ax.plot(counts, binary, color=SLATE, lw=2.0, ls="--", label="binary  (presence only)")
    # Mark the demo counts 1, 5, 50 used by the table on the page.
    for c in TF_DEMO_COUNTS:
        ax.scatter([c], [log_normalized_tf(c)], color=GREEN, zorder=5, s=45, edgecolor="white")
    ax.set_ylim(0, 12)
    ax.set_xlabel("raw term count  $f$")
    ax.set_ylabel("term-frequency weight")
    ax.set_title("Term-frequency variants: relevance is concave in count\nraw tf spans 50×; log-normalized compresses it to ~4.9×", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)
    _save(fig, "bow_tf_variants.png")


def fig_idf_curve() -> None:
    """IDF vs document frequency for a collection of N=1000 — plain log(N/df) and smoothed sklearn."""
    n_docs = IDF_CURVE_N
    df = np.arange(1, n_docs + 1)
    plain = np.log(n_docs / df)
    smooth = np.log((1 + n_docs) / (1 + df)) + 1.0
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(df, plain, color=BLUE, lw=2.4, label=r"textbook  $\mathrm{idf}=\log(N/df)$")
    ax.plot(df, smooth, color=PURPLE, lw=2.2, ls="--", label=r"smoothed  $\log\frac{1+N}{1+df}+1$  (sklearn)")
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.annotate(
        "df = N: term in every doc\ncarries 0 information",
        xy=(n_docs, 0), xytext=(n_docs * 0.45, 1.6),
        color=INK, fontsize=9.5,
        arrowprops=dict(arrowstyle="->", color=SLATE),
    )
    ax.set_xlabel("document frequency  $df$  (number of docs containing the term)")
    ax.set_ylabel("IDF weight")
    ax.set_title(f"IDF is the surprisal of 'this term appears'  (N={n_docs})\nrare terms → high weight; ubiquitous terms → ~0", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    _style_axis(ax)
    _save(fig, "bow_idf_curve.png")


def fig_tfidf_vs_counts() -> None:
    """Which terms TF-IDF up- or down-weights relative to their raw count, for document D1."""
    vocab = build_vocabulary(CORPUS)
    counts = count_matrix(CORPUS, vocab)
    idf = smoothed_idf(counts)
    weighted = counts * idf  # un-normalized tf-idf so the up/down-weighting is on a comparable scale
    doc_idx = 0  # D1
    present = counts[doc_idx] > 0
    terms = [t for t, p in zip(vocab, present) if p]
    raw = counts[doc_idx][present].astype(float)
    tfidf = weighted[doc_idx][present]
    order = np.argsort(-tfidf)
    terms = [terms[i] for i in order]
    raw = raw[order]
    tfidf = tfidf[order]
    x = np.arange(len(terms))
    width = 0.4
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(x - width / 2, raw, width, color=SLATE, label="raw count")
    ax.bar(x + width / 2, tfidf, width, color=GREEN, label="TF-IDF weight (un-normalized)")
    ax.set_xticks(x)
    ax.set_xticklabels(terms)
    ax.set_ylabel("weight")
    ax.set_title("D1: raw count ranks 'the' top; TF-IDF lifts distinctive 'mat' above it\nsmoothed idf('the')=1 holds it flat while rarer terms get boosted", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    _style_axis(ax)
    _save(fig, "bow_tfidf_vs_counts.png")


def fig_tfidf_matrix_heatmap() -> None:
    """The L2-normalized TF-IDF document-term matrix as a heatmap."""
    vocab = build_vocabulary(CORPUS)
    tfidf = tfidf_matrix(CORPUS, vocab)
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    im = ax.imshow(tfidf, cmap="Purples", aspect="auto", vmin=0, vmax=tfidf.max())
    ax.set_xticks(range(len(vocab)))
    ax.set_xticklabels(vocab, rotation=35, ha="right")
    ax.set_yticks(range(len(CORPUS)))
    ax.set_yticklabels(DOC_LABELS)
    for i in range(tfidf.shape[0]):
        for j in range(tfidf.shape[1]):
            val = tfidf[i, j]
            if val > 0:
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > tfidf.max() * 0.45 else INK, fontsize=8.5,
                )
    ax.set_title("L2-normalized TF-IDF matrix (each row is a unit vector)\n'the' (count 2, idf 1) still leads; unique terms mat/log/happy/chased rank next", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="tf-idf")
    ax.set_xlabel("vocabulary term")
    _save(fig, "bow_tfidf_matrix_heatmap.png")


def fig_cosine_heatmap() -> None:
    """Pairwise cosine similarity between the three documents."""
    vocab = build_vocabulary(CORPUS)
    tfidf = tfidf_matrix(CORPUS, vocab)
    cos = cosine_similarity_matrix(tfidf)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cos, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(DOC_LABELS)
    ax.set_yticks(range(3))
    ax.set_yticklabels(DOC_LABELS)
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, f"{cos[i, j]:.3f}", ha="center", va="center",
                color="white" if cos[i, j] > 0.55 else INK, fontsize=11, fontweight="bold",
            )
    ax.set_title("Cosine similarity between documents\nD1–D2 (0.618) > D1–D3 (0.455): shared sentence frame", fontsize=10.5)
    _save(fig, "bow_cosine_heatmap.png")


def fig_bm25_saturation() -> None:
    """BM25's TF component (length-off) saturates toward k1+1, vs unbounded raw TF."""
    freqs = np.linspace(0, 50, 200)
    raw = freqs
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(freqs, raw, color=RED, lw=2.0, ls="--", label="raw tf  (linear, unbounded)")
    for k1, color in ((1.2, GREEN), (1.5, BLUE), (2.0, PURPLE)):
        comp = np.array([bm25_tf_component(f, k1=k1) for f in freqs])
        ax.plot(freqs, comp, color=color, lw=2.4, label=rf"BM25 tf  ($k_1={k1}$), ceiling {k1 + 1:.1f}")
    ax.axhline(BM25_K1 + 1, color=BLUE, lw=0.8, ls=":")
    ax.set_ylim(0, 12)
    ax.set_xlabel("raw term count  $f$")
    ax.set_ylabel("TF contribution to score")
    ax.set_title("BM25 saturates term frequency; raw tf does not\nthe 50th occurrence barely beats the 5th → resists keyword stuffing", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)
    _save(fig, "bow_bm25_saturation.png")


def fig_retrieval_ranking() -> None:
    """Query 'happy cat' over the 5-doc corpus: cosine TF-IDF vs BM25 ranking, side by side."""
    full_corpus = CORPUS + BM25_EXTRA_DOCS
    labels = [f"D{i + 1}" for i in range(len(full_corpus))]
    # Cosine TF-IDF with correct fit/transform discipline: IDF learned from the corpus, query projected
    # into that fitted space (matches the page's 0.247 / 0.615 and avoids fitting on the query).
    cos_scores = tfidf_query_scores(full_corpus, QUERY)
    bm25 = bm25_scores(full_corpus, QUERY)
    bm25_norm = bm25 / bm25.max() if bm25.max() > 0 else bm25  # scale to [0,1] for shared axis
    x = np.arange(len(full_corpus))
    width = 0.4
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.bar(x - width / 2, cos_scores, width, color=BLUE, label="cosine TF-IDF")
    ax.bar(x + width / 2, bm25_norm, width, color=GREEN, label="BM25 (normalized to max)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("relevance score")
    ax.set_title("Retrieval for query 'happy cat': both rank D3 > D1 > rest\nBM25 widens the margin — 'happy' is a rare, high-IDF term", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    _style_axis(ax)
    _save(fig, "bow_retrieval_ranking.png")


def fig_sparsity() -> None:
    """How the fraction of zeros in a BoW matrix grows with vocabulary size (curse of dimensionality)."""
    # A document touches ~avg_doc_len distinct terms regardless of how large the vocabulary is, so the
    # density (nonzeros / cells) falls as 1/V. This is illustrative of real corpora, not the toy CORPUS.
    vocab_sizes = np.array([100, 1_000, 10_000, 100_000, 1_000_000])
    avg_distinct_terms = 120  # a typical document's distinct-token count (illustrative)
    density = np.minimum(avg_distinct_terms / vocab_sizes, 1.0)
    sparsity = (1 - density) * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(vocab_sizes, sparsity, color=NAVY, lw=2.4, marker="o", markersize=7, markeredgecolor="white")
    ax.set_xscale("log")
    ax.set_ylim(0, 101)
    for vs, sp in zip(vocab_sizes, sparsity):
        dy = 10 if sp < 50 else -14  # push the low-sparsity label above the point, others below
        ax.annotate(f"{sp:.2f}%", xy=(vs, sp), xytext=(0, dy), textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_xlabel("vocabulary size  $V$  (log scale)")
    ax.set_ylabel("matrix sparsity  (% zero entries)")
    ax.set_title("BoW vectors are overwhelmingly zero\na ~120-term document in a 1M-term vocabulary is 99.99% zeros (illustrative)", fontsize=11)
    _style_axis(ax)
    _save(fig, "bow_sparsity.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    fig_term_document_matrix()
    fig_tf_variants()
    fig_idf_curve()
    fig_tfidf_vs_counts()
    fig_tfidf_matrix_heatmap()
    fig_cosine_heatmap()
    fig_bm25_saturation()
    fig_retrieval_ranking()
    fig_sparsity()
    print("done.")


if __name__ == "__main__":
    main()
