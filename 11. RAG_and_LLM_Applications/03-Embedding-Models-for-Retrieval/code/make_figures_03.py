"""Static figure generator for 03-Embedding-Models-for-Retrieval.

Imports the SAME canonical functions the page and notebook use (embedding_models.py) so every
plotted number from our own embedders is the chapter's own — no hand-typed values. Writes
muted-palette PNGs to the shared chapter image dir (../../images/) with the per-chapter prefix
`rag03_`.

    python make_figures_03.py

Figures produced:
  rag03_paraphrase_clusters.png -- the 8 toy sentences in 2D under sparse vs dense embeddings;
                                  sparse scatters paraphrases apart, dense clusters them.
  rag03_cosine_heatmap.png      -- query x passage cosine matrices, sparse vs dense; the dense
                                  diagonal lights up (paraphrases match) while sparse stays flat.
  rag03_contrastive_training.png-- the InfoNCE loss falling and the paraphrase/unrelated similarity
                                  separating over training steps -- contrastive learning in action.
  rag03_sparse_dense_gap.png    -- the paraphrase-vs-unrelated similarity GAP for sparse / dense
                                  (from scratch) / pretrained all-MiniLM -- the headline result.
  rag03_dim_quality_cost.png    -- ILLUSTRATIVE: embedding dimension vs retrieval quality (rising,
                                  diminishing) and memory cost (linear) -- the dimension tradeoff.
  rag03_asymmetric_prefix.png   -- SCHEMATIC: asymmetric search -- query and passage get different
                                  instruction prefixes (E5 'query:'/'passage:') before encoding.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / torch 2.x. Headless (Agg); no display needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from embedding_models import (
    PARAPHRASE_PAIRS,
    bow_tensor,
    build_vocab,
    diagonal_vs_offdiagonal,
    info_nce_loss,
    similarity_matrix,
    sparse_embed,
    train_bi_encoder,
    try_pretrained_demo,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # query / sparse
PURPLE = "#5D4A8A"  # process / dense
GREEN = "#2E7A5A"  # positive / match
RED = "#8B3B4A"  # negative / miss
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # highlight
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


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project (n, d) -> (n, 2) via top-2 principal components (visualization only)."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _vocab_and_texts():
    queries = [q for q, _ in PARAPHRASE_PAIRS]
    passages = [p for _, p in PARAPHRASE_PAIRS]
    vocab = build_vocab(queries + passages)
    return queries, passages, vocab


def fig_paraphrase_clusters() -> None:
    """The 8 sentences in 2D under sparse vs dense; dense pulls paraphrase pairs together."""
    queries, passages, vocab = _vocab_and_texts()
    texts = queries + passages
    pair_id = list(range(len(queries))) * 2  # same id for a query and its paraphrase passage

    model, _ = train_bi_encoder(PARAPHRASE_PAIRS, vocab)
    sparse_vecs = np.array([sparse_embed(t, vocab) for t in texts])
    dense_vecs = np.array(
        [model(bow_tensor(t, vocab).unsqueeze(0)).squeeze(0).detach().numpy() for t in texts]
    )

    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(11.0, 5.0))
    cluster_colors = [BLUE, GREEN, AMBER, PURPLE]
    for ax, vecs, title in ((ax_s, sparse_vecs, "SPARSE (bag-of-words)"),
                            (ax_d, dense_vecs, "DENSE (trained bi-encoder)")):
        _style_axis(ax)
        coords = _pca_2d(vecs)
        for i, (x, y) in enumerate(coords):
            cid = pair_id[i]
            is_query = i < len(queries)
            ax.scatter(x, y, s=200, color=cluster_colors[cid], alpha=0.9,
                       marker="o" if is_query else "s", edgecolors=INK, linewidths=1.0, zorder=3)
        # draw a line connecting each paraphrase pair (query i <-> passage i)
        for i in range(len(queries)):
            qx, qy = coords[i]
            px, py = coords[i + len(queries)]
            ax.plot([qx, px], [qy, py], color=cluster_colors[i], linewidth=1.2, alpha=0.45, zorder=2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("PC 1 (illustrative 2D projection)")
        ax.set_ylabel("PC 2")
    circle = plt.Line2D([], [], marker="o", color=SLATE, linestyle="", markersize=9, label="query")
    square = plt.Line2D([], [], marker="s", color=SLATE, linestyle="", markersize=9, label="paraphrase passage")
    fig.legend(handles=[circle, square], loc="upper center", ncol=2, framealpha=0.95, fontsize=9,
               bbox_to_anchor=(0.5, 0.04))
    fig.suptitle("Same sentences, two embedders: dense pulls paraphrase pairs together",
                 fontsize=13, color=INK, y=1.0)
    _save(fig, "rag03_paraphrase_clusters.png")


def fig_cosine_heatmap() -> None:
    """Query x passage cosine matrices, sparse vs dense; the dense diagonal lights up."""
    queries, passages, vocab = _vocab_and_texts()
    model, _ = train_bi_encoder(PARAPHRASE_PAIRS, vocab)
    sparse_fn = lambda t: sparse_embed(t, vocab)  # noqa: E731
    dense_fn = lambda t: model(bow_tensor(t, vocab).unsqueeze(0)).squeeze(0).detach().numpy()  # noqa: E731
    sparse_sim = similarity_matrix(queries, passages, sparse_fn)
    dense_sim = similarity_matrix(queries, passages, dense_fn)

    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(11.2, 5.0))
    for ax, mat, title in ((ax_s, sparse_sim, "SPARSE cosine"), (ax_d, dense_sim, "DENSE cosine")):
        im = ax.imshow(mat, cmap="BuGn", vmin=-0.3, vmax=1.0, aspect="equal")
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_xlabel("passage index")
        ax.set_ylabel("query index")
        ax.set_xticks(range(len(passages)))
        ax.set_yticks(range(len(queries)))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color=INK if mat[i, j] < 0.6 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Cosine(query, passage): the dense diagonal (paraphrases) lights up; sparse stays flat",
                 fontsize=12.5, color=INK, y=1.0)
    _save(fig, "rag03_cosine_heatmap.png")


def fig_contrastive_training() -> None:
    """InfoNCE loss falling and paraphrase/unrelated similarity separating over training steps."""
    queries, passages, vocab = _vocab_and_texts()
    # re-run training while snapshotting the similarity gap every few steps (re-implement loop here
    # using the canonical loss so the curve is the chapter's own)
    import torch

    from embedding_models import DenseBiEncoder, LEARNING_RATE, SEED, TRAIN_STEPS

    torch.manual_seed(SEED)
    model = DenseBiEncoder(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    q_in = torch.stack([bow_tensor(q, vocab) for q in queries])
    p_in = torch.stack([bow_tensor(p, vocab) for p in passages])
    steps, losses, diags, offs = [], [], [], []
    for step in range(TRAIN_STEPS + 1):
        q_emb, p_emb = model(q_in), model(p_in)
        loss = info_nce_loss(q_emb, p_emb)
        if step % 20 == 0:
            mat = (q_emb @ p_emb.t()).detach().numpy()
            d, o = diagonal_vs_offdiagonal(mat)
            steps.append(step)
            losses.append(float(loss.item()))
            diags.append(d)
            offs.append(o)
        if step < TRAIN_STEPS:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    fig, (ax_loss, ax_sim) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    _style_axis(ax_loss)
    ax_loss.plot(steps, losses, color=PURPLE, linewidth=2.2)
    ax_loss.set_title("InfoNCE loss falls", fontsize=12)
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("contrastive loss")
    _style_axis(ax_sim)
    ax_sim.plot(steps, diags, color=GREEN, linewidth=2.2, label="paraphrase pairs (positives)")
    ax_sim.plot(steps, offs, color=RED, linewidth=2.2, label="unrelated pairs (negatives)")
    ax_sim.fill_between(steps, diags, offs, color=GREEN, alpha=0.08)
    ax_sim.set_title("Positives pulled together, negatives pushed apart", fontsize=11.5)
    ax_sim.set_xlabel("training step")
    ax_sim.set_ylabel("mean cosine similarity")
    ax_sim.legend(loc="center right", framealpha=0.95, fontsize=9)
    fig.suptitle("Contrastive training: the paraphrase gap opens as the loss falls", fontsize=12.5,
                 color=INK, y=1.02)
    _save(fig, "rag03_contrastive_training.png")


def fig_sparse_dense_gap() -> None:
    """The paraphrase-vs-unrelated similarity GAP: sparse vs dense (scratch) vs pretrained."""
    queries, passages, vocab = _vocab_and_texts()
    model, _ = train_bi_encoder(PARAPHRASE_PAIRS, vocab)
    sparse_fn = lambda t: sparse_embed(t, vocab)  # noqa: E731
    dense_fn = lambda t: model(bow_tensor(t, vocab).unsqueeze(0)).squeeze(0).detach().numpy()  # noqa: E731
    sparse_diag, sparse_off = diagonal_vs_offdiagonal(similarity_matrix(queries, passages, sparse_fn))
    dense_diag, dense_off = diagonal_vs_offdiagonal(similarity_matrix(queries, passages, dense_fn))

    labels = ["sparse\n(bag-of-words)", "dense\n(from scratch)"]
    diags = [sparse_diag, dense_diag]
    offs = [sparse_off, dense_off]

    pre = try_pretrained_demo(queries, passages)
    if pre is not None:
        pre_diag, pre_off = diagonal_vs_offdiagonal(pre)
        labels.append("pretrained\n(all-MiniLM, 384d)")
        diags.append(pre_diag)
        offs.append(pre_off)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    _style_axis(ax)
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, diags, width, color=GREEN, edgecolor=INK, linewidth=0.8,
           label="paraphrase pairs (want HIGH)")
    ax.bar(x + width / 2, offs, width, color=RED, edgecolor=INK, linewidth=0.8,
           label="unrelated pairs (want LOW)")
    for i, (d, o) in enumerate(zip(diags, offs)):
        ax.annotate(f"gap\n{d - o:+.2f}", (i, max(d, o) + 0.03), ha="center", va="bottom",
                    fontsize=9, color=INK, fontweight="bold")
    ax.axhline(0, color=GRID, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(min(offs) - 0.08, max(diags) + 0.22)  # headroom so the gap labels clear the title
    ax.set_title("The paraphrase gap: sparse can't separate, dense (learned) can", fontsize=12, pad=16)
    ax.set_ylabel("mean cosine similarity")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    _save(fig, "rag03_sparse_dense_gap.png")


def fig_dim_quality_cost() -> None:
    """ILLUSTRATIVE: embedding dimension vs retrieval quality (diminishing) and memory cost (linear).

    The shape is illustrative — quality rises with dimension but with diminishing returns, while
    memory/compute cost grows linearly. Real anchor points (e.g. MTEB) are model-specific; the
    takeaway is the SHAPE of the tradeoff, not exact values. Marked illustrative on the page.
    """
    dims = np.array([64, 128, 256, 384, 512, 768, 1024, 1536, 3072])
    # a saturating quality curve (diminishing returns) — illustrative normalized retrieval score
    quality = 1.0 - 0.55 * np.exp(-dims / 350.0)
    # memory cost is linear in dimension (bytes per vector ∝ dim) — normalized to the smallest
    cost = dims / dims.min()

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    _style_axis(ax)
    ax.plot(dims, quality, marker="o", color=GREEN, linewidth=2.2, markersize=6,
            markeredgecolor=INK, label="retrieval quality (saturates)")
    ax.set_xlabel("embedding dimension")
    ax.set_ylabel("retrieval quality (normalized)", color=GREEN)
    ax.set_ylim(0.4, 1.02)
    ax2 = ax.twinx()
    ax2.plot(dims, cost, marker="s", color=AMBER, linewidth=2.2, markersize=6,
             markeredgecolor=INK, label="memory / cost (linear)")
    ax2.set_ylabel("relative memory & compute cost", color=AMBER)
    ax2.tick_params(colors=INK, labelsize=9)
    ax2.spines["top"].set_visible(False)
    # annotate the "sweet spot" region where quality has mostly saturated but cost is still moderate
    ax.axvspan(256, 768, color=BLUE, alpha=0.07)
    ax.annotate("common sweet spot\n(384–768 dims)", (512, 0.55), ha="center", color=INK, fontsize=9)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.95, fontsize=9)
    ax.set_title("Embedding dimension: quality saturates, cost keeps rising (illustrative)",
                 fontsize=11.5, pad=12)
    _save(fig, "rag03_dim_quality_cost.png")


def fig_asymmetric_prefix() -> None:
    """SCHEMATIC: asymmetric search — query and passage get different instruction prefixes."""
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def _box(x, y, w, h, text, color, fontsize=9.5, mono=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.16, edgecolor=color,
                                   linewidth=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK,
                family="monospace" if mono else "sans-serif")

    # query path (top) — short text gets the 'query:' prefix
    _box(0.2, 4.4, 2.4, 1.0, "short query\n\"reset password\"", BLUE)
    _box(3.2, 4.4, 2.6, 1.0, 'prepend\n"query: ..."', AMBER, mono=True)
    _box(6.4, 4.4, 1.7, 1.0, "encoder", PURPLE)
    _box(8.4, 4.4, 1.4, 1.0, "q vector", GREEN)
    # passage path (bottom) — long text gets the 'passage:' prefix
    _box(0.2, 0.6, 2.4, 1.0, "long passage\n\"To reset, go to…\"", BLUE)
    _box(3.2, 0.6, 2.6, 1.0, 'prepend\n"passage: ..."', AMBER, mono=True)
    _box(6.4, 0.6, 1.7, 1.0, "encoder\n(SAME weights)", PURPLE, fontsize=8.5)
    _box(8.4, 0.6, 1.4, 1.0, "p vector", GREEN)
    # arrows
    for y in (4.9, 1.1):
        for x0, x1 in ((2.6, 3.2), (5.8, 6.4), (8.1, 8.4)):
            ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    # the comparison
    ax.annotate("", (8.95, 4.3), (8.95, 1.7), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
    ax.text(9.5, 3.0, "cosine", rotation=90, ha="center", va="center", fontsize=9.5, color=RED)
    ax.text(5.0, 5.75, "Asymmetric search: query and passage get DIFFERENT instruction prefixes",
            ha="center", fontsize=12, color=INK, fontweight="bold")
    ax.text(5.0, 3.0, "same model, different prefixes\n→ forget the prefix and similarity scores "
            "silently degrade", ha="center", va="center", fontsize=9, style="italic", color=RED)
    _save(fig, "rag03_asymmetric_prefix.png")


def main() -> None:
    fig_paraphrase_clusters()
    fig_cosine_heatmap()
    fig_contrastive_training()
    fig_sparse_dense_gap()
    fig_dim_quality_cost()
    fig_asymmetric_prefix()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
