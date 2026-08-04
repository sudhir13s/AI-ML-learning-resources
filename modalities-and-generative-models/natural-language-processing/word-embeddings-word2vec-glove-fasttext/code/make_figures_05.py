"""Reproducible figure generator for 05-Word-Embeddings-Word2Vec-GloVe-FastText.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
notebook -- the toy corpus, the skip-gram model, the freq^0.75 distribution, the by-hand triple,
and the GloVe ice/steam ratios are all IMPORTED from `word_embeddings.py`, so the figures cannot
silently drift from the prose or the demo. The real-GloVe figures (PCA, analogy parallelogram)
load the pretrained `glove-wiki-gigaword-50` vectors via gensim. Run:

    python make_figures_05.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at
150 dpi, prefixed `we_`. The palette matches the chapter's Mermaid diagrams (muted, white text on
fills).

Figures produced:
  we_onehot_vs_dense.png        -- one-hot (orthogonal, vocab-sized) vs a short dense embedding
  we_pca_real.png               -- 2-D PCA of real GloVe-50 vectors: meaning clusters + shared direction
  we_skipgram_vs_cbow.png       -- skip-gram (center->context) vs CBOW (pooled context->center)
  we_softmax_bottleneck.png     -- cost per pair: O(V) full softmax vs O(k) negative sampling
  we_negsampling.png            -- NS as geometry (pull true, push negatives) + raw vs freq^0.75
  we_training_loss.png          -- from-scratch skip-gram+NS loss falling over training steps
  we_cosine_neighbors.png       -- learned cosines: within-cluster HIGH vs cross-cluster LOWER
  we_glove_weighting.png        -- GloVe's f(x) weighting function (zero at 0, rises, then caps)
  we_ice_steam_ratio.png        -- the ice/steam co-occurrence ratio that makes analogies linear
  we_analogy_parallelogram.png  -- king - man + woman ~ queen as a parallelogram (real GloVe)
  we_fasttext_oov.png           -- FastText builds an OOV vector for an unseen word from its n-grams
  we_static_vs_contextual.png   -- one static vector for 'bank' vs context-dependent vectors

Verified on Python 3.12 / numpy 2.x / matplotlib 3.x / gensim 4.x. Deterministic; CPU.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt

from word_embeddings import (
    ICE_STEAM_PROBES,
    NEG_SAMPLING_WORDS,
    build_corpus,
    cbow_windows,
    cosine,
    ice_steam_ratios,
    negative_sampling_distribution,
    skipgram_pairs,
    train_skipgram,
    unit_embeddings,
)
from word_embeddings import build_vocab as build_vocab_fn

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


# ====================================================================================================
# 1) one-hot vs dense
# ====================================================================================================
def fig_onehot_vs_dense() -> None:
    """One-hot (vocab-sized, orthogonal) vs a short dense embedding (graded, comparable)."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.6, 5.4), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.85}
    )

    cell = 0.6  # uniform cell side so both rows render as squares, not slivers

    # Top: a one-hot row of width V (illustrative V=24), a single 1, drawn as unit squares.
    v = 24
    for j in range(v):
        face = NAVY if j == 8 else "#EEF3F8"
        ax_top.add_patch(mpatches.Rectangle((j * cell, 0), cell, cell, facecolor=face, edgecolor="white"))
    ax_top.text(8 * cell + cell / 2, -0.35, "slot 8123", ha="center", va="top", color=INK, fontsize=8)
    ax_top.text(-0.25, cell / 2, "'cat'", ha="right", va="center", color=INK, fontsize=10)
    ax_top.set_xlim(-3, v * cell + 0.2)
    ax_top.set_ylim(-1.2, cell + 1.3)
    ax_top.set_aspect("equal")
    ax_top.axis("off")
    ax_top.text(
        v * cell / 2, cell + 0.35,
        "one-hot: length = whole vocabulary, all zeros but one 1\n"
        "every pair is orthogonal -> cos('cat','kitten') = cos('cat','tuesday') = 0",
        ha="center", va="bottom", color=INK, fontsize=9.5,
    )

    # Bottom: a short dense embedding, graded values, drawn as colored unit squares.
    rng = np.random.default_rng(0)
    dense = rng.uniform(-1, 1, 12)
    cmap = matplotlib.colormaps["RdBu"]
    for j, val in enumerate(dense):
        ax_bot.add_patch(mpatches.Rectangle(
            (j * cell, 0), cell, cell, facecolor=cmap((val + 1) / 2), edgecolor="white"))
        ax_bot.text(j * cell + cell / 2, -0.32, f"d{j}", ha="center", va="top", color=INK, fontsize=7)
    ax_bot.text(-0.25, cell / 2, "'cat'", ha="right", va="center", color=INK, fontsize=10)
    ax_bot.set_xlim(-3, 12 * cell + 0.2)
    ax_bot.set_ylim(-1.6, cell + 1.0)
    ax_bot.set_aspect("equal")
    ax_bot.axis("off")
    ax_bot.text(
        12 * cell / 2, cell + 0.32,
        "dense embedding: ~12-300 dims, every entry a graded feature\n"
        "two words compared by cosine -> similar words get similar vectors",
        ha="center", va="bottom", color=INK, fontsize=9.5,
    )

    fig.suptitle(
        "From identity to meaning: one-hot vs dense embedding",
        fontsize=12.5, fontweight="bold", color=INK, y=0.99,
    )
    _save(fig, "we_onehot_vs_dense.png")


# ====================================================================================================
# Real-GloVe helpers
# ====================================================================================================
def _load_glove():
    import gensim.downloader as api

    return api.load("glove-wiki-gigaword-50")


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project rows of `matrix` to 2-D via PCA (centered, top-2 singular directions)."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


# ====================================================================================================
# 2) PCA of real GloVe vectors
# ====================================================================================================
def fig_pca_real() -> None:
    """2-D PCA of real GloVe-50 vectors: meaning clusters + a shared male->female direction."""
    glove = _load_glove()
    groups = {
        "royalty": (AMBER, ["king", "queen", "prince", "princess"]),
        "animals": (GREEN, ["cat", "dog", "rabbit", "horse"]),
        "capitals": (BLUE, ["paris", "london", "rome", "berlin"]),
        "countries": (RED, ["france", "england", "italy", "germany"]),
    }
    words = [w for _, ws in groups.values() for w in ws]
    vectors = np.stack([glove[w] for w in words])
    coords = _pca_2d(vectors)
    coord_of = {w: coords[i] for i, w in enumerate(words)}

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for _, (color, ws) in groups.items():
        pts = np.stack([coord_of[w] for w in ws])
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=90, edgecolor="white", zorder=4)
        for w in ws:
            x, y = coord_of[w]
            ax.annotate(w, (x, y), xytext=(6, 6), textcoords="offset points",
                        color=color, fontsize=11, fontweight="bold")
    # The shared male->female direction (king->queen, prince->princess).
    for a, b in (("king", "queen"), ("prince", "princess")):
        xa, ya = coord_of[a]
        xb, yb = coord_of[b]
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="->", color=PURPLE, lw=2, ls="--"))
    ax.text(
        0.5, -0.12,
        "dashed purple arrows = the shared male->female direction  (king->queen ~ prince->princess)",
        transform=ax.transAxes, ha="center", color=PURPLE, fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Real GloVe-50 vectors, PCA to 2-D: meaning clusters", fontsize=13, fontweight="bold")
    _style_axis(ax)
    _save(fig, "we_pca_real.png")


# ====================================================================================================
# 3) skip-gram vs CBOW
# ====================================================================================================
def fig_skipgram_vs_cbow() -> None:
    """Skip-gram (center -> each context) vs CBOW (pooled context -> center)."""
    fig, (ax_sg, ax_cb) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    context = ["the", "ruled", "crown"]

    # Skip-gram: one center node points OUT to each context node.
    ax_sg.set_title("Skip-gram: center -> each context word", fontsize=11, color=INK)
    cx, cy = 0.5, 0.5
    ctx_pos = [(0.12, 0.85), (0.12, 0.5), (0.12, 0.15)]
    ax_sg.add_patch(mpatches.FancyBboxPatch(
        (cx - 0.09, cy - 0.07), 0.18, 0.14, boxstyle="round,pad=0.02",
        facecolor=AMBER, edgecolor="none"))
    ax_sg.text(cx, cy, "king", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    for (px, py), word in zip(ctx_pos, context):
        ax_sg.add_patch(mpatches.FancyBboxPatch(
            (px - 0.08, py - 0.06), 0.16, 0.12, boxstyle="round,pad=0.02",
            facecolor=GREEN, edgecolor="none"))
        ax_sg.text(px, py, word, ha="center", va="center", color="white", fontsize=10)
        ax_sg.annotate("", xy=(px + 0.08, py), xytext=(cx - 0.09, cy),
                       arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.8))
    ax_sg.text(0.5, 0.02, "one center -> many predictions  (better for rare words)",
               ha="center", color=INK, fontsize=9)

    # CBOW: context nodes pool into a vector that points to the center.
    ax_cb.set_title("CBOW: averaged context -> center word", fontsize=11, color=INK)
    pool = (0.5, 0.5)
    ctx_pos2 = [(0.12, 0.85), (0.12, 0.5), (0.12, 0.15)]
    for (px, py), word in zip(ctx_pos2, context):
        ax_cb.add_patch(mpatches.FancyBboxPatch(
            (px - 0.08, py - 0.06), 0.16, 0.12, boxstyle="round,pad=0.02",
            facecolor=GREEN, edgecolor="none"))
        ax_cb.text(px, py, word, ha="center", va="center", color="white", fontsize=10)
        ax_cb.annotate("", xy=(pool[0] - 0.06, pool[1]), xytext=(px + 0.08, py),
                       arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.8))
    ax_cb.add_patch(mpatches.Circle(pool, 0.07, facecolor=PURPLE, edgecolor="none"))
    ax_cb.text(pool[0], pool[1], "avg", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax_cb.add_patch(mpatches.FancyBboxPatch(
        (0.78, 0.44), 0.18, 0.14, boxstyle="round,pad=0.02", facecolor=AMBER, edgecolor="none"))
    ax_cb.text(0.87, 0.51, "king", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax_cb.annotate("", xy=(0.78, 0.51), xytext=(pool[0] + 0.07, pool[1]),
                   arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.8))
    ax_cb.text(0.5, 0.02, "pooled context -> one prediction  (faster, better on frequent words)",
               ha="center", color=INK, fontsize=9)

    for ax in (ax_sg, ax_cb):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig.suptitle("Word2Vec's two objectives are mirror images", fontsize=12.5, fontweight="bold", color=INK)
    _save(fig, "we_skipgram_vs_cbow.png")


# ====================================================================================================
# 4) softmax bottleneck: O(V) vs O(k)
# ====================================================================================================
def fig_softmax_bottleneck() -> None:
    """Cost per (center, context) pair: full softmax O(V) vs negative sampling O(k)."""
    vocab_sizes = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
    k = 10
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(vocab_sizes, vocab_sizes, color=RED, lw=2.6, marker="o", markersize=7,
            markeredgecolor="white", label="full softmax: O(V) dot products / pair")
    ax.plot(vocab_sizes, np.full_like(vocab_sizes, k + 1), color=GREEN, lw=2.6, marker="s",
            markersize=7, markeredgecolor="white", label=f"negative sampling: O(k)=k+1={k + 1} / pair")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.annotate(
        "at V=10^6, k=10:\n~91,000x less work / pair",
        xy=(1e6, 1e6), xytext=(2e3, 2e5), color=INK, fontsize=9.5,
        arrowprops=dict(arrowstyle="->", color=SLATE),
    )
    ax.set_xlabel("vocabulary size  $V$  (log scale)")
    ax.set_ylabel("dot products per training pair  (log scale)")
    ax.set_title("The softmax bottleneck and its cure\nfull softmax scales with V; negative sampling is constant in k",
                 fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)
    _save(fig, "we_softmax_bottleneck.png")


# ====================================================================================================
# 5) negative sampling geometry + freq^0.75
# ====================================================================================================
def fig_negsampling() -> None:
    """Left: NS as a tug-of-war. Right: raw unigram vs freq^0.75 sampling probabilities."""
    fig, (ax_geo, ax_dist) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # Left: center v_c pulls true context u_o in, pushes k negatives out.
    ax_geo.set_title("Negative sampling as geometry", fontsize=11, color=INK)
    vc = np.array([0.0, 0.0])
    ax_geo.scatter(*vc, color=AMBER, s=160, edgecolor="white", zorder=5)
    ax_geo.annotate("$v_c$ 'king'", vc, xytext=(8, 8), textcoords="offset points",
                    color=AMBER, fontsize=11, fontweight="bold")
    true_ctx = np.array([0.75, 0.45])
    ax_geo.scatter(*true_ctx, color=GREEN, s=140, edgecolor="white", zorder=5)
    ax_geo.annotate("$u_o$ 'crown'", true_ctx, xytext=(8, 4), textcoords="offset points",
                    color=GREEN, fontsize=10, fontweight="bold")
    ax_geo.annotate("", xy=true_ctx * 0.92, xytext=vc,
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.4))
    negs = np.array([[-0.7, 0.5], [-0.6, -0.6], [0.4, -0.75]])
    for i, ng in enumerate(negs):
        ax_geo.scatter(*ng, color=RED, s=110, edgecolor="white", zorder=5)
        ax_geo.annotate("", xy=ng, xytext=ng * 0.35,
                        arrowprops=dict(arrowstyle="->", color=RED, lw=2.0))
        ax_geo.annotate(f"$u_{{n_{i + 1}}}$", ng, xytext=(6, 6), textcoords="offset points",
                        color=RED, fontsize=9)
    ax_geo.text(0.5, -0.16, "green = pull true context IN   red = push k random negatives OUT",
                transform=ax_geo.transAxes, ha="center", color=INK, fontsize=9)
    ax_geo.set_xlim(-1.1, 1.2)
    ax_geo.set_ylim(-1.1, 1.0)
    ax_geo.set_aspect("equal")
    ax_geo.axis("off")

    # Right: raw vs freq^0.75.
    raw, smoothed = negative_sampling_distribution()
    x = np.arange(len(NEG_SAMPLING_WORDS))
    width = 0.38
    ax_dist.bar(x - width / 2, raw, width, color=SLATE, label="raw unigram  $p(w)$")
    ax_dist.bar(x + width / 2, smoothed, width, color=PURPLE, label=r"smoothed  $p(w)\propto \mathrm{count}^{0.75}$")
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(NEG_SAMPLING_WORDS)
    ax_dist.set_ylabel("sampling probability")
    ax_dist.set_title("The 0.75 power lifts rare words, damps frequent ones\n'the' 0.602->0.524, 'kingdom' 0.006->0.017 (~2.75x)",
                      fontsize=10.5)
    ax_dist.legend(frameon=False, loc="upper right")
    _style_axis(ax_dist)
    _save(fig, "we_negsampling.png")


# ====================================================================================================
# 6) from-scratch skip-gram training loss
# ====================================================================================================
def fig_training_loss() -> None:
    """The from-scratch skip-gram+NS loss falling over training steps (the page's exact run)."""
    _, _, _, loss_history = train_skipgram(device="cpu")
    steps = np.arange(1, len(loss_history) + 1)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(steps, loss_history, color=BLUE, lw=2.4)
    ax.scatter([1, len(loss_history)], [loss_history[0], loss_history[-1]],
               color=BLUE, s=55, edgecolor="white", zorder=5)
    ax.annotate(f"start {loss_history[0]:.2f}", xy=(1, loss_history[0]),
                xytext=(30, -10), textcoords="offset points", color=INK, fontsize=9.5)
    ax.annotate(f"end {loss_history[-1]:.2f}", xy=(len(loss_history), loss_history[-1]),
                xytext=(-70, 18), textcoords="offset points", color=INK, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.set_xlabel("training step")
    ax.set_ylabel("negative-sampling loss")
    ax.set_title("From-scratch skip-gram + negative sampling: the loss falls\nthe model learns to separate true (center, context) pairs from noise",
                 fontsize=11)
    _style_axis(ax)
    _save(fig, "we_training_loss.png")


# ====================================================================================================
# 7) learned cosine neighbors
# ====================================================================================================
def fig_cosine_neighbors() -> None:
    """Learned cosines: within-cluster pairs HIGH, cross-cluster pair LOWER."""
    model, vocab, word_to_index, _ = train_skipgram(device="cpu")
    embeddings = unit_embeddings(model)
    pairs = [("king", "queen"), ("dog", "cat"), ("king", "dog")]
    labels = ["king~queen\n(royalty)", "dog~cat\n(animals)", "king~dog\n(cross-cluster)"]
    values = [cosine(embeddings, word_to_index, a, b) for a, b in pairs]
    colors = [AMBER, GREEN, RED]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:+.3f}",
                ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_ylabel("cosine similarity (learned)")
    ax.set_ylim(0, max(values) + 0.12)
    ax.set_title("The mechanism works: shared context -> higher cosine\nthe ORDERING is the point; tiny corpus -> modest magnitudes",
                 fontsize=11)
    _style_axis(ax)
    _save(fig, "we_cosine_neighbors.png")


# ====================================================================================================
# 8) GloVe weighting function f(x)
# ====================================================================================================
def fig_glove_weighting() -> None:
    """GloVe's f(x) = min((x/x_max)^0.75, 1): zero at 0, rises with count, then caps."""
    x_max = 100.0
    counts = np.linspace(0, 250, 500)
    weight = np.minimum((counts / x_max) ** 0.75, 1.0)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(counts, weight, color=PURPLE, lw=2.6)
    ax.axhline(1.0, color=SLATE, lw=0.8, ls=":")
    ax.axvline(x_max, color=GREEN, lw=1.0, ls="--")
    ax.scatter([0], [0], color=RED, s=60, zorder=5, edgecolor="white")
    ax.annotate("f(0)=0: skip pairs that never co-occur\n(the matrix is sparse -> cheap)",
                xy=(0, 0), xytext=(35, 0.18), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.annotate(r"caps at 1 for $x \geq x_{\max}$" + "\nhyper-frequent pairs can't dominate",
                xy=(x_max, 1.0), xytext=(120, 0.55), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.set_xlabel(r"co-occurrence count  $X_{ij}$")
    ax.set_ylabel(r"weight  $f(X_{ij})$")
    ax.set_title(r"GloVe weighting $f(x)=\min((x/x_{\max})^{0.75},\,1)$" + "\nsame 0.75 frequency-damping instinct as negative sampling",
                 fontsize=11)
    _style_axis(ax)
    _save(fig, "we_glove_weighting.png")


# ====================================================================================================
# 9) GloVe ice/steam ratio
# ====================================================================================================
def fig_ice_steam_ratio() -> None:
    """The co-occurrence RATIO P(.|ice)/P(.|steam) isolates meaning -> linear analogies."""
    ratios = ice_steam_ratios()
    ratio = ratios["ratio"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colors = [BLUE if r > 1.3 else RED if r < 0.77 else SLATE for r in ratio]
    bars = ax.bar(ICE_STEAM_PROBES, ratio, color=colors, edgecolor="white", width=0.6)
    ax.axhline(1.0, color=AMBER, lw=1.4, ls="--")
    ax.set_yscale("log")
    for bar, r in zip(bars, ratio):
        ax.text(bar.get_x() + bar.get_width() / 2, r * 1.08, f"{r:.2f}",
                ha="center", va="bottom", color=INK, fontsize=10.5, fontweight="bold")
    ax.text(0.05, 1.18, "ratio = 1 (dashed)", color=AMBER, fontsize=9, va="bottom", ha="left")
    ax.set_ylabel(r"ratio  $P(j\mid\mathrm{ice})\,/\,P(j\mid\mathrm{steam})$  (log)")
    ax.set_title("The ratio isolates meaning that raw counts blur\n>>1 -> ice (solid);  <<1 -> steam (gas);  ~1 -> both (water) or neither (fashion)",
                 fontsize=10.5)
    blue_patch = mpatches.Patch(color=BLUE, label="belongs to ice")
    red_patch = mpatches.Patch(color=RED, label="belongs to steam")
    slate_patch = mpatches.Patch(color=SLATE, label="ratio ~ 1 (uninformative)")
    ax.legend(handles=[blue_patch, red_patch, slate_patch], frameon=False, loc="upper right", fontsize=8.5)
    _style_axis(ax)
    _save(fig, "we_ice_steam_ratio.png")


# ====================================================================================================
# 10) analogy parallelogram (real GloVe)
# ====================================================================================================
def fig_analogy_parallelogram() -> None:
    """king - man + woman ~ queen as a parallelogram, drawn from real GloVe vectors.

    To draw an HONEST parallelogram we must respect the full-50-D geometry, not PCA only the four
    corners (PCA of just four points distorts the analogy badly). We therefore (1) compute the
    analogy vector king - man + woman in full 50-D, (2) build a 2-D basis from the two relevant
    directions -- the gender axis (woman - man) and the royalty axis (king - man) -- orthonormalized,
    and (3) project all five points onto that plane. In this faithful basis the four words form a
    near-parallelogram and the computed point (king - man + woman) lands CLOSEST to queen at cosine
    ~0.85 -- near it, not exactly on it -- with the residual drawn as a dotted line, matching the text.
    """
    glove = _load_glove()
    vec = {w: np.asarray(glove[w], dtype=float) for w in ("man", "king", "woman", "queen")}
    computed_full = vec["king"] - vec["man"] + vec["woman"]  # the analogy, in full 50-D

    # Project onto the plane spanned by the two SIDES of the parallelogram, anchored at 'man':
    # b1 = king - man (royalty side), b2 = woman - man (gender side), orthonormalized. In this plane
    # the analogy king - man + woman projects EXACTLY to king's + woman's corner (the parallelogram
    # closes), and queen falls right next to it -- the 0.85-cosine full-D fact made visible without
    # distortion. (man, king, woman sit exactly on the plane by construction; only queen carries the
    # tiny out-of-plane residual that the cosine 0.85 reflects.)
    royalty = vec["king"] - vec["man"]
    gender = vec["woman"] - vec["man"]
    e1 = royalty / np.linalg.norm(royalty)
    gender_perp = gender - (gender @ e1) * e1
    e2 = gender_perp / np.linalg.norm(gender_perp)
    origin = vec["man"]

    def project(v: np.ndarray) -> np.ndarray:
        d = v - origin
        return np.array([d @ e1, d @ e2])

    coord_of = {w: project(v) for w, v in vec.items()}
    computed = project(computed_full)

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for w, color in (("man", SLATE), ("king", AMBER), ("woman", SLATE), ("queen", AMBER)):
        x, y = coord_of[w]
        ax.scatter(x, y, color=color, s=120, edgecolor="white", zorder=5)
        ax.annotate(w, (x, y), xytext=(9, 9), textcoords="offset points",
                    color=color, fontsize=12, fontweight="bold")
    # the two near-parallel royalty arrows (man->king, woman->queen)
    ax.annotate("", xy=coord_of["king"], xytext=coord_of["man"],
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=2.2))
    ax.annotate("", xy=coord_of["queen"], xytext=coord_of["woman"],
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=2.2))
    # computed point: king - man + woman, projected from full-D. It is the 4th parallelogram corner;
    # queen is its NEAREST word (cos 0.85) but not identical -- the analogy is approximate, and we
    # draw the small residual honestly rather than pretend the point lands exactly on queen.
    ax.scatter(*computed, facecolor="none", edgecolor=GREEN, s=260, lw=2.6, zorder=6)
    ax.annotate("", xy=coord_of["queen"], xytext=computed,
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.4, ls=":"))
    ax.annotate("king - man + woman\nnearest word: queen (cos 0.85)", computed,
                xytext=(12, 6), textcoords="offset points", color=GREEN, fontsize=9.5, fontweight="bold")
    ax.text(0.5, -0.30,
            "purple arrows = shared royalty direction;  computed point (green ring) is CLOSEST to queen,\n"
            "not identical -- the parallelogram is approximate (dotted = the residual to queen)",
            transform=ax.transAxes, ha="center", color=INK, fontsize=8.5)
    ax.set_xlabel("royalty axis  (king - man direction)")
    ax.set_ylabel("gender axis  (woman - man, orthogonalized)")
    ax.set_title("Analogy as a parallelogram (real GloVe-50 vectors)\nking - man + woman ~ queen",
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    _style_axis(ax)
    _save(fig, "we_analogy_parallelogram.png")


# ====================================================================================================
# 11) FastText OOV
# ====================================================================================================
def _char_ngrams(word: str, min_n: int = 3, max_n: int = 6) -> list[str]:
    """Character n-grams of a boundary-marked word, FastText-style, plus the whole-word token."""
    marked = f"<{word}>"
    grams: list[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(len(marked) - n + 1):
            grams.append(marked[i : i + n])
    grams.append(f"<{word}>")  # the special whole-word token
    return grams


def fig_fasttext_oov() -> None:
    """FastText builds a word vector by SUMMING its character n-gram vectors -> unseen words work.

    Shows the mechanism the OOV win rests on: 'kingdom' and the unseen plural 'kingdoms' share almost
    all their n-grams, so summing those shared pieces gives 'kingdoms' a vector ~identical to 'kingdom'
    (measured cosine 0.9997) even though it was NEVER in the training vocabulary.
    """
    from gensim.models import FastText

    from word_embeddings import fasttext_corpus

    # Measure the real OOV cosine on the SAME FastText corpus the page's Code 3 and the notebook use
    # (8 repeats -> enough signal for clean subword vectors), so the caption number is reproduced.
    ft = FastText(fasttext_corpus(), vector_size=24, window=2, min_count=1,
                  min_n=2, max_n=4, sg=1, epochs=80, seed=0)
    va, vb = ft.wv["kingdom"], ft.wv["kingdoms"]
    oov_cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
    oov_norm = float(np.linalg.norm(vb))

    seen = _char_ngrams("kingdom")
    oov = _char_ngrams("kingdoms")
    shared = [g for g in oov if g in seen]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def draw_row(y: float, label: str, grams: list[str], base_color: str) -> None:
        ax.text(0.01, y, label, ha="left", va="center", color=INK, fontsize=10, fontweight="bold")
        x = 0.30
        for g in grams:
            color = GREEN if g in shared else AMBER
            w = 0.018 * len(g) + 0.025
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y - 0.035), w, 0.07, boxstyle="round,pad=0.004",
                facecolor=color, edgecolor="white"))
            ax.text(x + w / 2, y, g, ha="center", va="center", color="white", fontsize=8)
            x += w + 0.006

    draw_row(0.80, "'kingdom'\n(seen)", _char_ngrams("kingdom"), AMBER)
    draw_row(0.50, "'kingdoms'\n(OOV!)", oov, GREEN)
    ax.annotate("", xy=(0.5, 0.30), xytext=(0.5, 0.42),
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=2))
    ax.text(0.5, 0.20,
            f"sum the n-gram vectors  ->  vector for 'kingdoms'  (norm {oov_norm:.2f}, NOT zero)\n"
            f"{len(shared)} of {len(oov)} n-grams shared with 'kingdom'  ->  cos(kingdom, kingdoms) = {oov_cos:.4f}",
            ha="center", va="center", color=INK, fontsize=10)
    green_patch = mpatches.Patch(color=GREEN, label="n-gram shared with 'kingdom'")
    amber_patch = mpatches.Patch(color=AMBER, label="n-gram unique to one word")
    ax.legend(handles=[green_patch, amber_patch], frameon=False, loc="upper right", fontsize=9)
    ax.set_title("FastText handles a word it NEVER saw: words are sums of n-grams",
                 fontsize=12, fontweight="bold", color=INK)
    _save(fig, "we_fasttext_oov.png")


# ====================================================================================================
# 12) static vs contextual
# ====================================================================================================
def fig_static_vs_contextual() -> None:
    """One static vector for 'bank' (a blurry average) vs context-dependent vectors."""
    fig, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(10.4, 4.4))

    river = np.array([-0.8, 0.6])
    money = np.array([0.85, 0.5])
    avg = (river + money) / 2

    # Static: both sentences map 'bank' to ONE averaged vector.
    ax_s.set_title("Static embedding: one vector for 'bank'", fontsize=11, color=INK)
    ax_s.scatter(*river, color=GREEN, s=40, alpha=0.4)
    ax_s.scatter(*money, color=BLUE, s=40, alpha=0.4)
    ax_s.annotate("river-bank sense", river, xytext=(-10, 12), textcoords="offset points", color=GREEN, fontsize=9)
    ax_s.annotate("money-bank sense", money, xytext=(-10, 12), textcoords="offset points", color=BLUE, fontsize=9)
    ax_s.scatter(*avg, color=RED, s=150, edgecolor="white", zorder=5)
    ax_s.annotate("'bank'\n(one blurry average)", avg, xytext=(-30, -36), textcoords="offset points",
                  color=RED, fontsize=10, fontweight="bold")
    ax_s.annotate("", xy=avg, xytext=river, arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.2, ls=":"))
    ax_s.annotate("", xy=avg, xytext=money, arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.2, ls=":"))

    # Contextual: two distinct vectors, one per sentence.
    ax_c.set_title("Contextual embedding: a vector per sentence", fontsize=11, color=INK)
    ax_c.scatter(*river, color=GREEN, s=150, edgecolor="white", zorder=5)
    ax_c.scatter(*money, color=BLUE, s=150, edgecolor="white", zorder=5)
    ax_c.annotate("'bank' in\n'river bank'", river, xytext=(-12, 14), textcoords="offset points",
                  color=GREEN, fontsize=10, fontweight="bold")
    ax_c.annotate("'bank' in\n'savings bank'", money, xytext=(-12, 14), textcoords="offset points",
                  color=BLUE, fontsize=10, fontweight="bold")

    for ax in (ax_s, ax_c):
        ax.set_xlim(-1.4, 1.5)
        ax.set_ylim(-0.4, 1.1)
        ax.axis("off")
    fig.suptitle("Why static embeddings fail on polysemy", fontsize=12.5, fontweight="bold", color=INK)
    _save(fig, "we_static_vs_contextual.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    fig_onehot_vs_dense()
    fig_pca_real()
    fig_skipgram_vs_cbow()
    fig_softmax_bottleneck()
    fig_negsampling()
    fig_training_loss()
    fig_cosine_neighbors()
    fig_glove_weighting()
    fig_ice_steam_ratio()
    fig_analogy_parallelogram()
    fig_fasttext_oov()
    fig_static_vs_contextual()
    print("done.")


if __name__ == "__main__":
    main()
