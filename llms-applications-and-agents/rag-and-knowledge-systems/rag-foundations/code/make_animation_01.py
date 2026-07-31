"""Animated (GIF) intuition figure for 01-RAG-Fundamentals.

Companion to the static PNGs. Where those show final states, this brings the *mechanism* to life:
a query vector sweeps the embedding space, lights up its nearest passages, those passages flow into
the prompt, and the grounded answer appears — the retrieve -> augment -> generate loop, stepping.

    python make_animation_01.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). The
geometry is the chapter's OWN: the corpus and query embeddings come straight from
rag_fundamentals.py, projected to 2D the same way the static embedding-space figure is — so the
passages that light up are exactly the top-k the retriever returns on the page.

Produced:
  rag01_retrieve_augment_generate.gif -- the query finds its nearest passages, they flow into the
                                          augmented prompt, and the grounded answer is produced.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

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
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
INK = "#1C2530"
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 95
FPS = 12
HOLD_FRAMES = 14  # frames to dwell on the final grounded answer before looping


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project (n, d) -> (n, 2) via top-2 principal components (visualization only)."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def build_animation() -> None:
    idf = compute_idf(CORPUS)
    index = build_index(CORPUS, idf)
    q_vec = embed(PRIVATE_QUESTION, idf)
    top_idx, _ = cosine_top_k(q_vec, index, k=TOP_K)

    coords = _pca_2d(np.vstack([index, q_vec[None, :]]))
    doc_xy, q_xy = coords[:-1], coords[-1]

    fig, (ax_space, ax_flow) = plt.subplots(1, 2, figsize=(11.0, 5.2))
    fig.suptitle("RAG: retrieve the right passages, then answer grounded in them",
                 fontsize=13, color=INK, y=0.98)

    # ---- left panel: the embedding space the query sweeps -------------------------------
    ax_space.set_xlim(doc_xy[:, 0].min() - 0.15, doc_xy[:, 0].max() + 0.15)
    ax_space.set_ylim(doc_xy[:, 1].min() - 0.15, doc_xy[:, 1].max() + 0.15)
    ax_space.set_title("1. retrieve — nearest passages", fontsize=11, color=INK)
    ax_space.set_xticks([])
    ax_space.set_yticks([])
    for spine in ax_space.spines.values():
        spine.set_color(GRID)

    # ---- right panel: the prompt the retrieved passages flow into -----------------------
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    ax_flow.set_title("2. augment + generate — grounded answer", fontsize=11, color=INK)

    # static scatter of all passages (dim until retrieved)
    doc_artists = []
    for i, (x, y) in enumerate(doc_xy):
        artist = ax_space.scatter(x, y, s=150, color=SLATE, alpha=0.5, edgecolors=INK,
                                  linewidths=1.0, zorder=3)
        ax_space.annotate(f"doc[{i}]", (x, y), fontsize=7.5, color=INK, ha="left", va="top",
                          xytext=(7, -7), textcoords="offset points")
        doc_artists.append(artist)
    query_artist = ax_space.scatter(*q_xy, s=380, marker="*", color=AMBER, edgecolors=INK,
                                    linewidths=1.2, zorder=5)
    ax_space.annotate("query", q_xy, fontsize=9.5, color=INK, ha="center", va="bottom",
                      xytext=(0, 12), textcoords="offset points", fontweight="bold")
    beam_lines: list = []  # query->passage connectors, drawn as they light up

    # Order the retrieved passages by descending score so they light up best-first.
    retrieved_order = list(top_idx)
    passage_texts = [CORPUS[i] for i in retrieved_order]
    answer_text = CORPUS[int(top_idx[0])]  # the grounded answer = the #1 passage

    # Frame plan: 1 intro + one frame per retrieved passage + 1 augment + 1 generate + hold.
    n_retrieve_frames = len(retrieved_order)
    total = 1 + n_retrieve_frames + 2 + HOLD_FRAMES

    flow_texts: list = []

    def _wrap(text: str, width: int = 38) -> str:
        words, lines, line = text.split(), [], ""
        for word in words:
            if len(line) + len(word) + 1 > width:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)
        return "\n".join(lines)

    def update(frame: int):
        # phase boundaries
        retrieve_start = 1
        augment_frame = retrieve_start + n_retrieve_frames
        generate_frame = augment_frame + 1

        if frame == 0:  # intro: nothing lit yet
            return [query_artist]

        if retrieve_start <= frame < augment_frame:
            # light up retrieved passages one at a time, best first
            lit_count = frame - retrieve_start + 1
            for rank in range(lit_count):
                doc_i = int(retrieved_order[rank])
                doc_artists[doc_i].set_color(GREEN)
                doc_artists[doc_i].set_alpha(0.95)
                doc_artists[doc_i].set_sizes([260])
                if rank >= len(beam_lines):
                    x, y = doc_xy[doc_i]
                    (line,) = ax_space.plot([q_xy[0], x], [q_xy[1], y], color=GREEN,
                                            linewidth=1.6, alpha=0.55, zorder=2)
                    beam_lines.append(line)
            return doc_artists + beam_lines

        if frame == augment_frame:
            # the retrieved passages flow into the prompt as the context block
            ax_flow.text(0.5, 0.93, "augmented prompt", ha="center", va="top", fontsize=10.5,
                         fontweight="bold", color=PURPLE, transform=ax_flow.transAxes)
            y = 0.82
            for rank, passage in enumerate(passage_texts, start=1):
                txt = ax_flow.text(0.04, y, f"[{rank}] {_wrap(passage)}", ha="left", va="top",
                                   fontsize=8.5, color=INK, family="monospace",
                                   transform=ax_flow.transAxes,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor=GREEN, alpha=0.14,
                                             edgecolor=GREEN))
                flow_texts.append(txt)
                y -= 0.16
            q_txt = ax_flow.text(0.04, y, f"Q: {_wrap(PRIVATE_QUESTION)}", ha="left", va="top",
                                 fontsize=8.5, color=INK, family="monospace",
                                 transform=ax_flow.transAxes,
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor=BLUE, alpha=0.14,
                                           edgecolor=BLUE))
            flow_texts.append(q_txt)
            return flow_texts

        if frame >= generate_frame:
            # the grounded answer appears, drawn from the #1 retrieved passage
            ans = ax_flow.text(0.5, 0.08, f"✓ grounded answer:\n{_wrap(answer_text, 44)}",
                               ha="center", va="bottom", fontsize=9.5, color=INK,
                               fontweight="bold", transform=ax_flow.transAxes,
                               bbox=dict(boxstyle="round,pad=0.4", facecolor=GREEN, alpha=0.22,
                                         edgecolor=GREEN))
            return [ans]

        return [query_artist]

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag01_retrieve_augment_generate.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
