"""Static figure generator for 01-RAG-Fundamentals.

Imports the SAME canonical functions the page and notebook use (rag_fundamentals.py) so every
plotted number is the chapter's own — no hand-typed values. Writes muted-palette PNGs to the
shared chapter image dir (../../images/) with the per-chapter prefix `rag01_`.

    python make_figures_01.py

Figures produced:
  rag01_embedding_space.png    -- the corpus + query projected to 2D; the query's top-k passages
                                  highlighted, distractors greyed -- retrieval as nearest-neighbours.
  rag01_similarity_bars.png    -- cosine score of every passage to the private query, sorted, with
                                  the top-k cut line -- why doc[0] wins and where the cutoff lands.
  rag01_recall_vs_k.png        -- retrieval recall@k as k grows across a set of probe questions:
                                  more passages retrieved -> higher chance the answer is included.
  rag01_prompt_anatomy.png     -- the augmented prompt dissected into instruction / context /
                                  question, with the retrieved passages shown as the grounding.
  rag01_lost_in_the_middle.png -- the U-shaped accuracy curve: an LLM uses evidence best at the
                                  start and end of a long context, worst in the middle.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x. Headless (Agg); no display needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from rag_fundamentals import (
    CORPUS,
    PRIVATE_QUESTION,
    TOP_K,
    build_index,
    compute_idf,
    cosine_top_k,
    embed,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # data / corpus
PURPLE = "#5D4A8A"  # process
GREEN = "#2E7A5A"  # retrieved / output
RED = "#8B3B4A"  # miss / danger
SLATE = "#4A5B6E"  # frozen / distractor
AMBER = "#7A6528"  # highlight / query
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
    """Project rows of an (n, d) matrix to 2D via the top-2 principal components (for plotting only).

    Pure-numpy PCA: centre, take the right singular vectors, keep 2. This is a *visualization* of
    the high-dim embedding geometry — the retriever itself never reduces dimensions.
    """
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T  # (n, 2): coordinates in the top-2 PC plane


def fig_embedding_space(idf: dict[str, float], index: np.ndarray) -> None:
    """The corpus + query in 2D; top-k passages highlighted — retrieval as nearest-neighbours."""
    q_vec = embed(PRIVATE_QUESTION, idf)
    top_idx, _ = cosine_top_k(q_vec, index, k=TOP_K)
    # project corpus AND query together so they share one PCA plane
    stacked = np.vstack([index, q_vec[None, :]])
    coords = _pca_2d(stacked)
    doc_xy, q_xy = coords[:-1], coords[-1]
    top_set = set(int(i) for i in top_idx)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    _style_axis(ax)
    for i, (x, y) in enumerate(doc_xy):
        retrieved = i in top_set
        ax.scatter(
            x, y, s=240 if retrieved else 150,
            color=GREEN if retrieved else SLATE, alpha=0.95 if retrieved else 0.55,
            edgecolors=INK, linewidths=1.0, zorder=3,
        )
        # offset the label below-right of the marker so it never sits on top of the query star
        ax.annotate(f"doc[{i}]", (x, y), fontsize=8, color=INK, ha="left", va="top",
                    xytext=(8, -8), textcoords="offset points", zorder=4)
        if retrieved:
            # a faint line from query to each retrieved passage = "this is what got pulled in"
            ax.plot([q_xy[0], x], [q_xy[1], y], color=GREEN, linewidth=1.4, alpha=0.5, zorder=2)
    ax.scatter(*q_xy, s=420, marker="*", color=AMBER, edgecolors=INK, linewidths=1.2, zorder=5)
    ax.annotate("query", q_xy, fontsize=10, color=INK, ha="center", va="bottom",
                xytext=(0, 14), textcoords="offset points", fontweight="bold")
    ax.set_title("Retrieval is nearest-neighbour search in embedding space", fontsize=12, pad=12)
    ax.set_xlabel("principal component 1 (illustrative 2D projection)")
    ax.set_ylabel("principal component 2")
    green_proxy = ax.scatter([], [], s=150, color=GREEN, edgecolors=INK, label=f"top-{TOP_K} retrieved")
    slate_proxy = ax.scatter([], [], s=110, color=SLATE, edgecolors=INK, label="not retrieved")
    star_proxy = ax.scatter([], [], s=220, marker="*", color=AMBER, edgecolors=INK, label="query")
    ax.legend(handles=[star_proxy, green_proxy, slate_proxy], loc="best", framealpha=0.95, fontsize=9)
    _save(fig, "rag01_embedding_space.png")


def fig_similarity_bars(idf: dict[str, float], index: np.ndarray) -> None:
    """Cosine score of every passage to the query, sorted — why doc[0] wins, where the cut lands."""
    q_vec = embed(PRIVATE_QUESTION, idf)
    scores = index @ q_vec  # cosine per passage (rows are unit-norm)
    order = np.argsort(scores)[::-1]  # best first
    sorted_scores = scores[order]
    colors = [GREEN if rank < TOP_K else SLATE for rank in range(len(order))]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    _style_axis(ax)
    labels = [f"doc[{i}]" for i in order]
    bars = ax.bar(labels, sorted_scores, color=colors, edgecolor=INK, linewidth=0.8, width=0.66)
    for bar, score in zip(bars, sorted_scores):
        ax.annotate(f"{score:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    fontsize=8, color=INK, ha="center", va="bottom", xytext=(0, 2),
                    textcoords="offset points")
    # the top-k cutoff: everything left of this line is retrieved
    ax.axvline(TOP_K - 0.5, color=RED, linewidth=1.6, linestyle="--", alpha=0.85)
    ax.annotate(f"top-{TOP_K} cutoff", (TOP_K - 0.5, max(sorted_scores) * 0.92), color=RED,
                fontsize=9, ha="left", va="center", xytext=(6, 0), textcoords="offset points")
    ax.set_title("Cosine similarity to the query — sorted, with the top-k cut", fontsize=12, pad=12)
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(0, max(sorted_scores) * 1.18)
    _save(fig, "rag01_similarity_bars.png")


def fig_recall_vs_k(idf: dict[str, float], index: np.ndarray) -> None:
    """Recall@k across probe questions: more passages retrieved -> higher chance the answer is in.

    The probes are deliberately *paraphrased* — they share few literal words with their gold
    passage (e.g. "liftoff date" vs the passage's "launched on") — so a lexical embedder does NOT
    always rank the right passage first. That is the realistic case: recall@1 is imperfect and
    climbs as k grows, trading precision (more noise in the prompt) for recall. An over-easy corpus
    where recall@1 is already 1.0 would teach nothing, so the curve below is what RAG actually looks
    like before you add better embedders, hybrid search, or re-ranking (later chapters).
    """
    # (paraphrased question, index of its gold passage) — low lexical overlap on purpose
    probes = [
        ("What is the liftoff date of the orbiter from Kourou?", 0),
        ("Which optical sensor does the spacecraft carry?", 1),
        ("Who is the engineer in charge of the programme?", 2),
        ("How long does a single revolution around the planet take?", 3),
        ("Total number of playing tiles on a chess grid?", 6),
        ("Boiling point of liquid at sea level pressure?", 7),
    ]
    ks = np.arange(1, len(CORPUS) + 1)
    recall = []
    for k in ks:
        hits = 0
        for question, gold in probes:
            top_idx, _ = cosine_top_k(embed(question, idf), index, k=int(k))
            hits += int(gold in top_idx)  # 1 if the answering passage made the top-k
        recall.append(hits / len(probes))

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    _style_axis(ax)
    ax.plot(ks, recall, marker="o", color=PURPLE, linewidth=2.2, markersize=7,
            markeredgecolor=INK, markerfacecolor=GREEN)
    for k, r in zip(ks, recall):
        ax.annotate(f"{r:.2f}", (k, r), fontsize=8, color=INK, ha="center", va="bottom",
                    xytext=(0, 7), textcoords="offset points")
    ax.set_title("Retrieval recall@k — the recall/noise knob", fontsize=12, pad=12)
    ax.set_xlabel("k (passages retrieved)")
    ax.set_ylabel("recall@k  (fraction of answers retrieved)")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(ks)
    _save(fig, "rag01_recall_vs_k.png")


def fig_prompt_anatomy(idf: dict[str, float], index: np.ndarray) -> None:
    """The augmented prompt dissected into instruction / context / question blocks."""
    q_vec = embed(PRIVATE_QUESTION, idf)
    top_idx, _ = cosine_top_k(q_vec, index, k=TOP_K)

    def _wrap(text: str, width: int = 64) -> str:
        # soft-wrap so long passages stay inside the box rather than overflowing the right edge
        words, lines, line = text.split(), [], ""
        for word in words:
            if len(line) + len(word) + 1 > width:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)
        return "\n".join(lines)

    passages = [_wrap(CORPUS[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.axis("off")
    blocks = [
        ("INSTRUCTION", "Answer using ONLY the context. If absent, say you don't know.", AMBER),
        ("CONTEXT  (retrieved passages — the grounding)",
         "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages)), GREEN),
        ("QUESTION", PRIVATE_QUESTION, BLUE),
    ]
    y = 0.96
    for title, body, color in blocks:
        n_lines = body.count("\n") + 1
        height = 0.10 + 0.075 * n_lines  # taller box for multi-line context
        ax.add_patch(plt.Rectangle((0.02, y - height), 0.96, height, transform=ax.transAxes,
                                    facecolor=color, alpha=0.16, edgecolor=color, linewidth=1.6))
        ax.text(0.04, y - 0.03, title, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
                color=color, va="top")
        ax.text(0.04, y - 0.075, body, transform=ax.transAxes, fontsize=9, color=INK, va="top",
                family="monospace")
        y -= height + 0.03
    ax.text(0.04, y - 0.005, "↓ this whole block — not just the question — is what the LLM generates from",
            transform=ax.transAxes, fontsize=9.5, style="italic", color=INK, va="top")
    ax.set_title("Anatomy of the augmented prompt", fontsize=13, color=INK, pad=8)
    _save(fig, "rag01_prompt_anatomy.png")


def fig_lost_in_the_middle() -> None:
    """The U-shaped 'lost in the middle' accuracy curve — a documented LLM long-context failure.

    Illustrative shape from Liu et al. 2024 ('Lost in the Middle'): accuracy is highest when the
    relevant passage sits at the start or end of the context and dips when buried in the middle.
    Marked illustrative because the exact numbers are model-specific.
    """
    positions = np.linspace(0, 1, 11)  # relative position of the gold passage in the context
    # a smooth U: high at the ends, lowest in the middle (illustrative, normalized accuracy)
    accuracy = 0.55 + 0.40 * (2 * positions - 1) ** 2

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _style_axis(ax)
    ax.plot(positions, accuracy, marker="o", color=RED, linewidth=2.4, markersize=7,
            markeredgecolor=INK, markerfacecolor=AMBER)
    ax.fill_between(positions, accuracy, 0.5, where=(accuracy < 0.75), color=RED, alpha=0.08)
    ax.annotate("buried in the middle\n→ often ignored", (0.5, accuracy[5]), color=RED, fontsize=9.5,
                ha="center", va="bottom", xytext=(0, 18), textcoords="offset points")
    ax.annotate("start", (0.0, accuracy[0]), color=INK, fontsize=9, ha="left", va="bottom",
                xytext=(4, 6), textcoords="offset points")
    ax.annotate("end", (1.0, accuracy[-1]), color=INK, fontsize=9, ha="right", va="bottom",
                xytext=(-4, 6), textcoords="offset points")
    ax.set_title("Lost in the middle — position of evidence matters (illustrative)", fontsize=12, pad=12)
    ax.set_xlabel("position of the relevant passage within the context  (0 = start, 1 = end)")
    ax.set_ylabel("answer accuracy (normalized)")
    ax.set_ylim(0.45, 1.0)
    _save(fig, "rag01_lost_in_the_middle.png")


def main() -> None:
    idf = compute_idf(CORPUS)
    index = build_index(CORPUS, idf)
    fig_embedding_space(idf, index)
    fig_similarity_bars(idf, index)
    fig_recall_vs_k(idf, index)
    fig_prompt_anatomy(idf, index)
    fig_lost_in_the_middle()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
