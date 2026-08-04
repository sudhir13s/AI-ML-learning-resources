"""Reproducible figure generator for 15-Topic-Modeling-LDA-NMF.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
notebook -- the planted corpus, the from-scratch Gibbs LDA sampler, the multiplicative-update NMF,
and the coherence functions are all IMPORTED from `topic_modeling.py`, so the figures cannot
silently drift from the prose or the demo. Run:

    python make_figures_15.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at 150 dpi,
prefixed `tm_`. The palette matches the chapter's Mermaid diagrams (muted, white text on fills).

Figures produced:
  tm_dirichlet_simplex.png    -- Dirichlet samples on the 2-simplex for alpha<1, =1, >1 (sparsity dial)
  tm_doc_topic_heatmap.png    -- recovered doc-topic theta: documents cluster by planted topic (block)
  tm_topic_word_heatmap.png   -- recovered topic-word phi: the planted block structure, recovered
  tm_gibbs_convergence.png    -- log-likelihood + doc-topic purity rising over Gibbs sweeps
  tm_nmf_factorization.png    -- V ~= W.H schematic on the planted count matrix (all entries >= 0)
  tm_nmf_convergence.png      -- NMF Frobenius reconstruction error, monotone decrease
  tm_coherence_bars.png       -- coherent (planted) topic vs random word set: NPMI and UMass
  tm_coherence_perplexity.png -- sweep K: held-out perplexity (min) + coherence (peak) both at K=3
  tm_lda_nmf_topwords.png     -- top words per topic, LDA (counts) vs NMF (TF-IDF), readable corpus

Verified on Python 3.12 / numpy 2.x / scikit-learn 1.x / matplotlib 3.x. CPU, deterministic (seed 7).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from topic_modeling import (
    PLANTED_TOPIC_NAMES,
    PLANTED_TOPIC_WORDS,
    READABLE_CORPUS,
    SEED,
    GibbsLDA,
    docs_to_count_matrix,
    make_planted_corpus,
    nmf_multiplicative,
    npmi_coherence,
    sklearn_lda_nmf,
    sweep_k_coherence_perplexity,
    umass_coherence,
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


def _dirichlet_to_xy(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map points on the 3-d probability simplex to 2-d triangle coordinates for plotting."""
    # corners of an equilateral triangle for topics (1,0,0), (0,1,0), (0,0,1)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    xy = samples @ corners
    return xy[:, 0], xy[:, 1]


def fig_dirichlet_simplex() -> None:
    """Dirichlet samples on the K=3 simplex for alpha<1, =1, >1 — the sparsity dial, visualized."""
    rng = np.random.default_rng(SEED)
    alphas = [0.1, 1.0, 5.0]
    titles = [
        r"$\alpha=0.1$  (sparse)" "\ncorners: each doc ≈ one topic",
        r"$\alpha=1$  (uniform)" "\neven spread over the simplex",
        r"$\alpha=5$  (dense)" "\ncenter: each doc ≈ all topics",
    ]
    colors = [GREEN, SLATE, RED]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2], [0.0, 0.0]])
    corner_labels = ["topic 0", "topic 1", "topic 2"]
    corner_xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    for ax, a, title, c in zip(axes, alphas, titles, colors):
        samples = rng.dirichlet([a, a, a], size=600)
        x, y = _dirichlet_to_xy(samples)
        ax.plot(tri[:, 0], tri[:, 1], color=INK, lw=1.2, zorder=2)
        ax.scatter(x, y, s=10, color=c, alpha=0.45, zorder=3, edgecolor="none")
        for (cx, cy), lab in zip(corner_xy, corner_labels):
            dy = -0.07 if cy < 0.1 else 0.04
            ax.text(cx, cy + dy, lab, ha="center", va="top" if cy < 0.1 else "bottom",
                    fontsize=9, color=INK)
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.2, 1.05)
        ax.axis("off")
    fig.suptitle(
        "The Dirichlet prior is a knob on topic-mixture sparsity (K=3 simplex)\n"
        r"small $\alpha$ pushes documents to the corners (about one topic); large $\alpha$ to the center",
        fontsize=12, color=INK, y=1.06,
    )
    _save(fig, "tm_dirichlet_simplex.png")


def _fit_planted_lda() -> tuple[GibbsLDA, list[str], np.ndarray, np.ndarray]:
    """Shared fit: from-scratch Gibbs LDA on the planted corpus. Returns (model, vocab, counts, true)."""
    docs, vocab, true_topic = make_planted_corpus()
    counts = docs_to_count_matrix(docs, len(vocab))
    lda = GibbsLDA(n_topics=3, n_iter=300, seed=SEED).fit(docs, len(vocab))
    return lda, vocab, counts, true_topic


def _planted_topic_order(top_words_per_topic: list[list[str]]) -> list[int]:
    """Map each recovered topic to the planted topic it best matches (by word overlap), so the
    heatmaps line up with the planted topic names regardless of arbitrary topic numbering."""
    order = []
    for planted in PLANTED_TOPIC_WORDS:
        planted_set = set(planted)
        overlaps = [len(planted_set & set(words)) for words in top_words_per_topic]
        order.append(int(np.argmax(overlaps)))
    return order


def fig_topic_word_heatmap() -> None:
    """Recovered topic-word phi as a heatmap: the planted block structure, recovered from scratch."""
    lda, vocab, _, _ = _fit_planted_lda()
    phi = lda.topic_word()  # (3, V)
    order = _planted_topic_order(lda.top_words(vocab, n=6))
    phi = phi[order]
    # order vocabulary by planted topic so the blocks are visually contiguous
    word_order = [vocab.index(w) for words in PLANTED_TOPIC_WORDS for w in words]
    phi = phi[:, word_order]
    words_sorted = [vocab[i] for i in word_order]

    fig, ax = plt.subplots(figsize=(11.0, 3.0))
    im = ax.imshow(phi, cmap="Purples", aspect="auto", vmin=0, vmax=phi.max())
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"topic {k}\n({PLANTED_TOPIC_NAMES[k]})" for k in range(3)])
    ax.set_xticks(range(len(words_sorted)))
    ax.set_xticklabels(words_sorted, rotation=45, ha="right", fontsize=8.5)
    # draw block boundaries every 6 words
    for b in (6, 12):
        ax.axvline(b - 0.5, color=INK, lw=1.2)
    ax.set_title(
        "Recovered topic–word distributions φ (from-scratch Gibbs LDA on the planted corpus)\n"
        "each topic concentrates on its planted 6-word block — the structure was recovered, not given",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, label="P(word | topic)")
    _save(fig, "tm_topic_word_heatmap.png")


def fig_doc_topic_heatmap() -> None:
    """Recovered doc-topic theta as a heatmap: documents cluster cleanly by their planted topic."""
    lda, vocab, _, true_topic = _fit_planted_lda()
    theta = lda.doc_topic()  # (D, 3)
    order = _planted_topic_order(lda.top_words(vocab, n=6))
    theta = theta[:, order]
    # sort documents by planted topic so the block-diagonal is visible
    doc_order = np.argsort(true_topic, kind="stable")
    theta = theta[doc_order]

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    im = ax.imshow(theta, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"topic {k}\n({PLANTED_TOPIC_NAMES[k]})" for k in range(3)])
    ax.set_ylabel("documents (sorted by planted topic)")
    # mark the planted topic-group boundaries on the y axis
    sorted_true = true_topic[doc_order]
    for k in range(1, 3):
        boundary = np.searchsorted(sorted_true, k) - 0.5
        ax.axhline(boundary, color=INK, lw=1.0)
    ax.set_title(
        "Recovered doc–topic mixtures θ (block-diagonal = clean recovery)\n"
        "each document loads almost entirely on its single planted topic",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="P(topic | doc)")
    _save(fig, "tm_doc_topic_heatmap.png")


def fig_gibbs_convergence() -> None:
    """Corpus log-likelihood and doc-topic purity rising as the Gibbs sampler organizes the corpus."""
    docs, vocab, true_topic = make_planted_corpus()
    lda = GibbsLDA(n_topics=3, n_iter=120, seed=SEED)
    iters, lls, purities = lda.fit_with_trace(docs, len(vocab), true_topic)

    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    ax1.plot(iters, lls, color=BLUE, lw=2.4, marker="o", markersize=4, markeredgecolor="white",
             label="corpus log-likelihood")
    ax1.set_xlabel("Gibbs sweep")
    ax1.set_ylabel("corpus log-likelihood", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    _style_axis(ax1)

    ax2 = ax1.twinx()
    ax2.plot(iters, purities, color=GREEN, lw=2.4, marker="s", markersize=4, markeredgecolor="white",
             label="doc-topic purity vs planted truth")
    ax2.set_ylabel("purity vs planted topics", color=GREEN)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    ax2.spines["top"].set_visible(False)

    ax1.set_title(
        "Collapsed-Gibbs LDA self-organizes the corpus\n"
        "log-likelihood climbs and purity reaches 1.0 — topics emerge from random init",
        fontsize=11,
    )
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False, loc="center right")
    _save(fig, "tm_gibbs_convergence.png")


def fig_nmf_factorization() -> None:
    """Schematic V ~= W.H on the planted count matrix — every entry non-negative (additive parts)."""
    docs, vocab, true_topic = make_planted_corpus()
    counts = docs_to_count_matrix(docs, len(vocab))
    # sort docs by planted topic + words by planted block so V shows visible block structure
    doc_order = np.argsort(true_topic, kind="stable")
    word_order = [vocab.index(w) for words in PLANTED_TOPIC_WORDS for w in words]
    V = counts[doc_order][:, word_order]
    W, H, _ = nmf_multiplicative(counts, n_topics=3, n_iter=400, seed=SEED)
    W = W[doc_order]
    H = H[:, word_order]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4),
                             gridspec_kw={"width_ratios": [3.0, 0.8, 3.0]})
    axV, axW, axH = axes
    axV.imshow(V, cmap="Blues", aspect="auto")
    axV.set_title("V  (documents × words)\nnon-negative TF counts", fontsize=10.5)
    axV.set_xlabel("words (planted blocks)")
    axV.set_ylabel("documents")
    axV.set_xticks([])
    axV.set_yticks([])

    axW.imshow(W, cmap="Greens", aspect="auto")
    axW.set_title("W\n(docs × topics)", fontsize=10.5)
    axW.set_xlabel("3 topics")
    axW.set_xticks([])
    axW.set_yticks([])

    axH.imshow(H, cmap="Purples", aspect="auto")
    axH.set_title("H  (topics × words)\nthe 3 recovered topics", fontsize=10.5)
    axH.set_xlabel("words (planted blocks)")
    axH.set_ylabel("topics")
    axH.set_xticks([])
    axH.set_yticks(range(3))

    fig.suptitle(
        r"NMF factorizes the non-negative count matrix $V \approx W H$, all entries $\geq 0$"
        "\n"
        "W = documents in topic space, H = topics in word space; non-negativity forces additive, readable parts",
        fontsize=12, y=1.04,
    )
    _save(fig, "tm_nmf_factorization.png")


def fig_nmf_convergence() -> None:
    """NMF Frobenius reconstruction error decreasing monotonically under multiplicative updates."""
    docs, vocab, _ = make_planted_corpus()
    counts = docs_to_count_matrix(docs, len(vocab))
    # n_iter matches the headline run on the page (106.92 -> 42.58, i.e. 42.6 to one decimal)
    _, _, errors = nmf_multiplicative(counts, n_topics=3, n_iter=400, seed=SEED)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(range(len(errors)), errors, color=PURPLE, lw=2.6)
    ax.scatter([0, len(errors) - 1], [errors[0], errors[-1]], color=PURPLE, s=55,
               zorder=5, edgecolor="white")
    ax.annotate(f"start  {errors[0]:.1f}", xy=(0, errors[0]), xytext=(8, -4),
                textcoords="offset points", color=INK, fontsize=9.5)
    ax.annotate(f"converged  {errors[-1]:.1f}", xy=(len(errors) - 1, errors[-1]),
                xytext=(-30, 14), textcoords="offset points", color=INK, fontsize=9.5)
    ax.set_xlabel("multiplicative-update iteration")
    ax.set_ylabel(r"reconstruction error  $\| V - WH \|_F$")
    ax.set_title(
        "Lee & Seung's multiplicative updates decrease the error monotonically\n"
        "no learning rate, no projection — non-negativity is preserved automatically",
        fontsize=11,
    )
    _style_axis(ax)
    _save(fig, "tm_nmf_convergence.png")


def fig_coherence_bars() -> None:
    """A coherent (planted) topic scores far above a random word set on both NPMI and UMass."""
    docs, vocab, _ = make_planted_corpus()
    counts = docs_to_count_matrix(docs, len(vocab))
    coherent = [vocab.index(w) for w in PLANTED_TOPIC_WORDS[0]]
    rng = np.random.default_rng(SEED)
    random_set = list(rng.choice(len(vocab), size=len(coherent), replace=False))

    metrics = ["NPMI\n(higher better)", "UMass\n(higher better)"]
    coherent_scores = [npmi_coherence(coherent, counts), umass_coherence(coherent, counts)]
    random_scores = [npmi_coherence(random_set, counts), umass_coherence(random_set, counts)]

    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    b1 = ax.bar(x - width / 2, coherent_scores, width, color=GREEN, label="coherent (planted) topic")
    b2 = ax.bar(x + width / 2, random_scores, width, color=RED, label="random word set")
    ax.axhline(0, color=SLATE, lw=0.9)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:+.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4 if h >= 0 else -12), textcoords="offset points",
                        ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("coherence score")
    ax.set_title(
        "Coherence measures what we care about: do a topic's words co-occur?\n"
        "the planted topic out-scores a random word set on both NPMI and UMass",
        fontsize=11,
    )
    ax.legend(frameon=False, loc="lower right")
    _style_axis(ax)
    _save(fig, "tm_coherence_bars.png")


def fig_coherence_perplexity() -> None:
    """Sweep K: held-out perplexity (minimized) and coherence (peak) both favor the true K = 3."""
    docs, vocab, _ = make_planted_corpus()
    counts = docs_to_count_matrix(docs, len(vocab))
    ks, coherences, perplexities = sweep_k_coherence_perplexity(counts, k_values=(2, 3, 4, 5, 6, 8))

    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    ax1.plot(ks, perplexities, color=RED, lw=2.4, marker="o", markersize=6, markeredgecolor="white",
             label="held-out perplexity (lower better)")
    ax1.set_xlabel("number of topics  K")
    ax1.set_ylabel("held-out perplexity", color=RED)
    ax1.tick_params(axis="y", labelcolor=RED)
    best_p = ks[int(np.argmin(perplexities))]
    ax1.axvline(best_p, color=SLATE, lw=1.0, ls=":")
    _style_axis(ax1)

    ax2 = ax1.twinx()
    ax2.plot(ks, coherences, color=GREEN, lw=2.4, marker="s", markersize=6, markeredgecolor="white",
             label="UMass coherence (higher better)")
    ax2.set_ylabel("UMass coherence", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    ax2.spines["top"].set_visible(False)

    ax1.annotate("true K = 3", xy=(3, min(perplexities)), xytext=(3.4, np.mean(perplexities)),
                 color=INK, fontsize=10, arrowprops=dict(arrowstyle="->", color=SLATE))
    ax1.set_title(
        "Choosing K on the planted corpus: perplexity bottoms and coherence peaks near K = 3\n"
        "the true number of themes — but the curves are flatter for coherence, so read the topics too",
        fontsize=11,
    )
    lines = ax1.get_lines()[:1] + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], frameon=True, framealpha=0.9,
               edgecolor="none", loc="center right")
    _save(fig, "tm_coherence_perplexity.png")


def fig_lda_nmf_topwords() -> None:
    """Top words per topic for both methods on the readable corpus — LDA (counts) vs NMF (TF-IDF)."""
    lda_topics, nmf_topics, _ = sklearn_lda_nmf(READABLE_CORPUS, n_topics=3, top_n=6)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 5.2))
    method_rows = [("LDA (counts)", lda_topics, GREEN), ("NMF (TF-IDF)", nmf_topics, PURPLE)]
    names = ["sports", "cooking", "space"]

    def _match_order(topics):
        # order recovered topics to (sports, cooking, space) by keyword presence for stable layout
        keys = [{"team", "goal", "football", "striker"}, {"garlic", "onion", "recipe", "tomato"},
                {"planet", "galaxy", "orbit", "rocket", "telescope"}]
        order = []
        for kset in keys:
            order.append(int(np.argmax([len(kset & set(t)) for t in topics])))
        return order

    for r, (method, topics, color) in enumerate(method_rows):
        order = _match_order(topics)
        for c in range(3):
            ax = axes[r, c]
            words = topics[order[c]]
            y = np.arange(len(words))[::-1]
            ax.barh(y, np.linspace(1.0, 0.45, len(words)), color=color, alpha=0.85)
            for yi, w in zip(y, words):
                ax.text(0.04, yi, w, va="center", ha="left", color="white", fontsize=11,
                        fontweight="bold")
            ax.set_xlim(0, 1.05)
            ax.set_yticks([])
            ax.set_xticks([])
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(f"discovered: {names[c]}", fontsize=10.5, color=INK)
            if c == 0:
                ax.set_ylabel(method, fontsize=11, color=INK)
    fig.suptitle(
        "Both methods recover sports / cooking / space with no labels (readable 18-doc corpus)\n"
        "top-6 words per topic; LDA on raw counts (top), NMF on TF-IDF (bottom)",
        fontsize=12, y=1.02,
    )
    _save(fig, "tm_lda_nmf_topwords.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    fig_dirichlet_simplex()
    fig_topic_word_heatmap()
    fig_doc_topic_heatmap()
    fig_gibbs_convergence()
    fig_nmf_factorization()
    fig_nmf_convergence()
    fig_coherence_bars()
    fig_coherence_perplexity()
    fig_lda_nmf_topwords()
    print("done.")


if __name__ == "__main__":
    main()
