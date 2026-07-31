"""Animated (GIF) intuition figure for 07-Query-Transformation-HyDE-Multi-Query.

Companion to the static PNGs. Where `rag07_asymmetry_2d.png` shows the START and END states, this
brings the HyDE JUMP to life: the raw question sits FAR from its gold answer passage (in
question-space), then the LLM rewrites it as a hypothetical ANSWER and the retrieval probe TRAVELS
across the embedding space to land in the gold's neighbourhood (answer-space). You watch the top-k
retrieval snap from "distractor #1, gold #2" to "gold #1" exactly as the probe crosses over.

    python make_animation_07.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). The two
endpoints -- the raw-question point and the hypothetical-answer point -- are REAL all-MiniLM
embeddings projected to 2D by the same PCA the static figure uses; only the travel BETWEEN them is
interpolated for the animation. The cosine and rank read-outs at each end are the chapter's own
measured numbers.

Produced:
  rag07_hyde_jump.gif -- the retrieval probe morphs from the question to the hypothetical answer and
                         crosses the embedding space into the gold cluster; the retrieved top-k
                         re-orders live, gold climbing from #2 to #1.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter) / sentence-transformers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from query_transformation import (
    DenseRetriever,
    build_hyde_probes,
    cosine_to_gold,
)
from hybrid_search import full_corpus  # noqa: E402  (on path via query_transformation)

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
DPI = 100
FPS = 12


def _pca_2d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-centered top-2 PCA. Returns (projected_points, mean, components) so a probe embedding
    can be projected into the SAME 2D frame as the corpus for the travelling dot."""
    mean = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    comps = vt[:2]
    return centered @ comps.T, mean, comps


def build_animation() -> None:
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    probe = build_hyde_probes(corpus)[1]  # exact-code probe: raw #2 -> HyDE #1, the vivid jump

    texts = list(corpus) + [probe.query, probe.hyde_good]
    vecs = np.asarray(dense._encode(texts))  # noqa: SLF001 -- reuse ch5 encoder; unit-norm rows
    pts, _, _ = _pca_2d(vecs)
    n = len(corpus)
    doc_pts, q_pt, h_pt = pts[:n], pts[n], pts[n + 1]

    # measured read-outs (chapter's own numbers)
    cos_q = cosine_to_gold(dense, corpus, probe.query, probe.gold)
    cos_h = cosine_to_gold(dense, corpus, probe.hyde_good, probe.gold)
    raw_rank = list(dense.search(probe.query, k=n).indices).index(probe.gold) + 1
    hyde_rank = list(dense.search(probe.hyde_good, k=n).indices).index(probe.gold) + 1

    # travel path: ease from the question point to the hypothetical point, hold at each end
    ease = np.concatenate([
        np.full(10, 0.0),                       # hold on the raw question
        0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 26)),  # smooth ease across (0->1)
        np.full(18, 1.0),                       # hold on the hypothetical answer (gold cluster)
    ])

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.10)

    xlo = min(pts[:, 0].min(), q_pt[0], h_pt[0]) - 0.1
    xhi = max(pts[:, 0].max(), q_pt[0], h_pt[0]) + 0.1
    ylo = min(pts[:, 1].min(), q_pt[1], h_pt[1]) - 0.1
    yhi = max(pts[:, 1].max(), q_pt[1], h_pt[1]) + 0.1

    other = [i for i in range(n) if i != probe.gold]

    def update(frame: int):
        ax.clear()
        t = float(ease[frame])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.tick_params(colors=INK, labelsize=8)
        ax.set_xlabel("principal component 1", color=INK, fontsize=9)
        ax.set_ylabel("principal component 2", color=INK, fontsize=9)

        # corpus + gold
        ax.scatter(doc_pts[other, 0], doc_pts[other, 1], s=60, color=SLATE, alpha=0.5,
                   edgecolors=INK, linewidths=0.5, zorder=2)
        ax.scatter(*doc_pts[probe.gold], s=300, color=GREEN, edgecolors=INK, linewidths=1.3,
                   marker="*", zorder=5)
        ax.annotate("gold answer", doc_pts[probe.gold], color=GREEN, fontsize=9, fontweight="bold",
                    xytext=(8, -12), textcoords="offset points")

        # the travelling retrieval probe: question (t=0) -> hypothetical (t=1)
        probe_pt = (1 - t) * q_pt + t * h_pt
        # a fading trail from the question toward the current position
        trail = np.linspace(0, t, 12)
        for tau in trail:
            p = (1 - tau) * q_pt + tau * h_pt
            ax.scatter(*p, s=30, color=AMBER, alpha=0.10 + 0.10 * (tau / max(t, 1e-6)), zorder=3)
        # ghost of the original question (where it started)
        ax.scatter(*q_pt, s=130, color=BLUE, alpha=0.35, edgecolors=INK, linewidths=0.8,
                   marker="o", zorder=4)
        ax.annotate("raw question", q_pt, color=BLUE, fontsize=9, fontweight="bold",
                    xytext=(8, 8), textcoords="offset points", alpha=0.8)
        # the live probe: morphs colour/shape from blue-circle to amber-diamond as it crosses
        marker = "o" if t < 0.5 else "D"
        color = BLUE if t < 0.25 else (AMBER if t < 0.85 else GREEN)
        ax.scatter(*probe_pt, s=190, color=color, edgecolors=INK, linewidths=1.2, marker=marker, zorder=6)

        cos_now = (1 - t) * cos_q + t * cos_h
        rank_now = raw_rank if t < 0.5 else hyde_rank
        # phase label
        if t < 0.02:
            phase, pcol = "RAW QUESTION — embedded in question-space, far from the answer", BLUE
        elif t < 0.98:
            phase, pcol = "HyDE: the query is rewritten as a hypothetical ANSWER, and travels…", AMBER
        else:
            phase, pcol = "HYPOTHETICAL ANSWER — lands in the gold's neighbourhood (answer-space)", GREEN
        fig.suptitle(phase, fontsize=12.5, color=pcol, y=0.965, fontweight="bold")
        ax.set_title(f"cos(probe, gold) ≈ {cos_now:.3f}     gold retrieval rank: #{rank_now}",
                     fontsize=11, color=INK, pad=10)
        # measured-endpoints caption
        fig.text(0.5, 0.02,
                 f"endpoints are REAL all-MiniLM embeddings (PCA-2D): question cos={cos_q:.3f} (gold #{raw_rank})  "
                 f"→  HyDE cos={cos_h:.3f} (gold #{hyde_rank}); the travel between is interpolated",
                 ha="center", fontsize=8, style="italic", color=SLATE)
        return []

    anim = FuncAnimation(fig, update, frames=len(ease), interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag07_hyde_jump.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
