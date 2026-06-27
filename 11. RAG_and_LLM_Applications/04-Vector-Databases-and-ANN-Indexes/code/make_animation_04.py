"""Animated (GIF) intuition figure for 04-Vector-Databases-and-ANN-Indexes.

Companion to the static PNGs. Where those show final states, this brings the *mechanism* to life:
the query lands in its Voronoi cell, and as `nprobe` grows the search lights up more nearby cells —
covering more of the true neighbours but scanning more vectors. You *watch* the recall/speed knob
turn: probe fewer cells (fast, misses neighbours) -> probe more (slower, recovers them).

    python make_animation_04.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). The
geometry and the recall are the chapter's OWN: a 2D IVF built with the canonical build_ivf, and the
true neighbours from the canonical brute_force_topk.

Produced:
  rag04_nprobe_growth.gif -- the probed Voronoi cells expanding with nprobe, the recovered true
                             neighbours filling in, and the recall@10 climbing.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from vector_indexes import brute_force_topk, build_ivf, ivf_search, recall_at_k

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
INK = "#1C2530"
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 95
FPS = 2  # slow: dwell on each nprobe so the expansion is readable
HOLD_FRAMES = 3
NPROBE_FRAMES = (1, 2, 3, 5, 8, 12)  # the nprobe values the animation steps through


def build_animation() -> None:
    # A single broad 2D cloud cut into many cells: each cell is a thin slice, so a query's true
    # neighbours straddle several cells and probing one misses some (recall 0.5 -> 1.0 as nprobe grows).
    # 2D can only ever show a gentle climb (spatial locality keeps neighbours close); the STEEP
    # high-dimensional cliff lives in the static rag04_recall_cliff figure. Here the point you watch
    # is the *probed region expanding* to cover more neighbours.
    rng = np.random.default_rng(3)
    n_cells = 30
    corpus = rng.normal(0, 3.0, (1200, 2)).astype(np.float32)
    query = np.array([3.067, -1.517], dtype=np.float32)  # a boundary query: misses neighbours at nprobe=1
    index = build_ivf(corpus, n_cells=n_cells, seed=3)
    true_nn = brute_force_topk(query, corpus, k=10)
    cell_dist = ((index.centroids - query) ** 2).sum(axis=1)
    cell_order = np.argsort(cell_dist)  # cells in probe order (nearest first)
    cmap = plt.cm.hsv(np.linspace(0, 1, n_cells))  # distinct hues for many (30) cells

    plan = list(NPROBE_FRAMES) + [NPROBE_FRAMES[-1]] * HOLD_FRAMES

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    def update(frame_idx: int):
        nprobe = plan[frame_idx]
        probed = set(int(c) for c in cell_order[:nprobe])
        retrieved, n_scanned = ivf_search(index, query, k=10, nprobe=nprobe)
        recall = recall_at_k(retrieved, true_nn)
        recovered = set(retrieved.tolist()) & set(true_nn.tolist())

        ax.clear()
        ax.set_xlim(corpus[:, 0].min() - 1, corpus[:, 0].max() + 1)
        ax.set_ylim(corpus[:, 1].min() - 1, corpus[:, 1].max() + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        # points: probed cells saturated, others faded
        for cell, ids in index.cells.items():
            if len(ids) == 0:
                continue
            on = cell in probed
            ax.scatter(corpus[ids, 0], corpus[ids, 1], s=12, color=cmap[cell],
                       alpha=0.85 if on else 0.12, edgecolors="none", zorder=2 if on else 1)
        # true neighbours: green ring if recovered, red ring if still missed
        for nn in true_nn:
            recovered_nn = nn in recovered
            ax.scatter(corpus[nn, 0], corpus[nn, 1], s=70, facecolors="none",
                       edgecolors=GREEN if recovered_nn else RED,
                       linewidths=2.0, zorder=5)
        ax.scatter(*query, s=360, marker="*", color=AMBER, edgecolors=INK, linewidths=1.2, zorder=6)
        frac = 100 * n_scanned / len(corpus)
        ax.set_title(f"nprobe = {nprobe}  ·  {len(probed)} of {n_cells} cells scanned "
                     f"({frac:.0f}% of vectors)", fontsize=12, color=INK)
        ax.text(0.5, -0.04,
                f"recall@10 = {recall:.1f}   ·   green = neighbour found, red = still missed",
                transform=ax.transAxes, ha="center", va="top", fontsize=10,
                color=GREEN if recall >= 0.9 else INK)
        return ax.collections

    anim = FuncAnimation(fig, update, frames=len(plan), interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag04_nprobe_growth.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
