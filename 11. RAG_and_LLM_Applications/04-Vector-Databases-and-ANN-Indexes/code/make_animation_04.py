"""Animated (GIF) intuition figure for 04-Vector-Databases-and-ANN-Indexes.

Companion to the static PNGs. Where those show final states, this brings the *mechanism* to life:
the query lands in its Voronoi cell, and as `nprobe` grows the search lights up more nearby cells —
covering more of the true neighbours but scanning more vectors. You *watch* the recall/speed knob
turn: probe fewer cells (fast, misses neighbours) → probe more (slower, recovers them).

This is deliberately a **2D SCHEMATIC** (self-contained numpy k-means, no faiss/torch needed): you
cannot animate 384-D space, so a 2D toy conveys the routing geometry honestly. The *steep* real
recall cliff — measured on the real 384-D FAISS IVF over the real Wikipedia corpus — lives in the
static `rag04_ivf_recall_cliff.png` figure. Here the point you watch is the *probed region
expanding* to cover the query's true neighbours as the knob turns.

    python make_animation_04.py

The GIF is written to ../../images/ via matplotlib's PillowWriter (no ffmpeg needed).

Produced:
  rag04_nprobe_growth.gif — the probed Voronoi cells expanding with nprobe, the recovered true
                            neighbours turning green, and recall@10 climbing to 1.0.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
GREEN = "#2E7A5A"
RED = "#8B3B4A"
AMBER = "#7A6528"
INK = "#1C2530"
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 95
FPS = 2  # slow: dwell on each nprobe so the expansion is readable
HOLD_FRAMES = 3
NPROBE_FRAMES = (1, 2, 3, 5, 8, 12)  # the nprobe values the animation steps through


def _kmeans_2d(corpus: np.ndarray, n_cells: int, iters: int = 12, seed: int = 3):
    """Tiny numpy Lloyd's k-means for the 2D schematic. Returns (centroids, assignments)."""
    rng = np.random.default_rng(seed)
    centroids = corpus[rng.choice(len(corpus), n_cells, replace=False)].copy()
    assignments = np.zeros(len(corpus), dtype=np.int64)
    for _ in range(iters):
        dist = ((corpus[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignments = dist.argmin(axis=1)
        for c in range(n_cells):
            members = assignments == c
            if members.any():
                centroids[c] = corpus[members].mean(axis=0)
    return centroids, assignments


def build_animation() -> None:
    # A single broad 2D cloud cut into many cells: each cell is a thin slice, so a query's true
    # neighbours straddle several cells and probing one misses some (recall climbs as nprobe grows).
    rng = np.random.default_rng(3)
    n_cells = 30
    corpus = rng.normal(0, 3.0, (1200, 2)).astype(np.float32)
    query = np.array([3.067, -1.517], dtype=np.float32)  # a boundary query: misses neighbours at nprobe=1
    centroids, assignments = _kmeans_2d(corpus, n_cells)
    cells = {c: np.where(assignments == c)[0] for c in range(n_cells)}
    true_nn = set(np.argsort(((corpus - query) ** 2).sum(axis=1))[:10].tolist())
    cell_order = np.argsort(((centroids - query) ** 2).sum(axis=1))  # cells in probe order (nearest first)
    base = np.vstack([plt.cm.tab20b(np.linspace(0, 1, 20)), plt.cm.tab20c(np.linspace(0, 1, 20))])
    cmap = base[:n_cells]

    plan = list(NPROBE_FRAMES) + [NPROBE_FRAMES[-1]] * HOLD_FRAMES
    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    def update(frame_idx: int):
        nprobe = plan[frame_idx]
        probed = set(int(c) for c in cell_order[:nprobe])
        candidate_ids = np.concatenate([cells[c] for c in probed]) if probed else np.array([], dtype=int)
        cand_dist = ((corpus[candidate_ids] - query) ** 2).sum(axis=1)
        retrieved = set(candidate_ids[np.argsort(cand_dist)[:10]].tolist()) if len(candidate_ids) else set()
        recovered = retrieved & true_nn
        recall = len(recovered) / len(true_nn)
        n_scanned = len(candidate_ids)

        ax.clear()
        ax.set_xlim(corpus[:, 0].min() - 1, corpus[:, 0].max() + 1)
        ax.set_ylim(corpus[:, 1].min() - 1, corpus[:, 1].max() + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        for cell, ids in cells.items():
            if len(ids) == 0:
                continue
            on = cell in probed
            ax.scatter(corpus[ids, 0], corpus[ids, 1], s=12, color=cmap[cell],
                       alpha=0.85 if on else 0.12, edgecolors="none", zorder=2 if on else 1)
        for nn in true_nn:
            found = nn in recovered
            ax.scatter(corpus[nn, 0], corpus[nn, 1], s=70, facecolors="none",
                       edgecolors=GREEN if found else RED, linewidths=2.0, zorder=5)
        ax.scatter(*query, s=360, marker="*", color=AMBER, edgecolors=INK, linewidths=1.2, zorder=6)
        frac = 100 * n_scanned / len(corpus)
        ax.set_title(f"nprobe = {nprobe}  ·  {len(probed)} of {n_cells} cells scanned "
                     f"({frac:.0f}% of vectors)", fontsize=12, color=INK)
        ax.text(0.5, -0.04,
                f"recall@10 = {recall:.1f}   ·   green = neighbour found, red = still missed  "
                f"(2D schematic; real 384-D cliff is in the static figure)",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
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
