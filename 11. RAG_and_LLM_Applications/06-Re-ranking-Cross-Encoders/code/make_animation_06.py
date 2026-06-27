"""Animated (GIF) intuition figure for 06-Re-ranking-Cross-Encoders.

Companion to the static PNGs. Where those show final states, this brings the re-rank to life: the
bi-encoder's top-k pool starts in its (coarse) order, then the cross-encoder re-scores each
candidate by JOINT encoding and the list re-sorts -- the gold passage, buried at #4 after retrieval,
climbs to #1. The bars are coloured by the cross-encoder relevance logit, so you watch relevance
"light up" the true answer as the order settles.

    python make_animation_06.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). The
scores are the chapter's OWN: the bi-encoder pool and cross-encoder logits come straight from
reranking.py -- so the passage that climbs to #1 is exactly the one the page and notebook report.

Produced:
  rag06_rerank.gif -- the bi-encoder pool re-orders under cross-encoder scores; the gold rises to #1.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow / sentence-transformers (CPU).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from reranking import (
    CORPUS,
    GOLD_INDEX,
    QUERY,
    RETRIEVE_K,
    BiEncoderRetriever,
    CrossEncoderReranker,
)

# ---- Palette (muted, matches the chapter family) -------------------------------------------
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
FPS = 8


def _short(text: str, n: int = 52) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def build_animation() -> None:
    bi = BiEncoderRetriever(CORPUS)
    cross = CrossEncoderReranker()

    pool = list(bi.retrieve(QUERY, k=RETRIEVE_K).indices)  # bi-encoder order (coarse)
    ce_scores = {doc: float(s) for doc, s in zip(pool, cross.scores_for(QUERY, [CORPUS[i] for i in pool]))}
    reranked = sorted(pool, key=lambda d: ce_scores[d], reverse=True)  # cross-encoder order

    # Interpolate positions from the bi-encoder order to the re-ranked order, frame by frame.
    bi_pos = {doc: i for i, doc in enumerate(pool)}
    rr_pos = {doc: i for i, doc in enumerate(reranked)}
    n = len(pool)
    norm = plt.Normalize(min(ce_scores.values()), max(ce_scores.values()))

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    fig.subplots_adjust(left=0.40, right=0.93, top=0.78, bottom=0.05)

    n_move = 24
    hold = 12
    total = 6 + n_move + hold

    def update(frame: int):
        ax.clear()
        # clear figure-level texts (suptitle + captions) so they don't stack across frames
        for txt in list(fig.texts):
            txt.remove()
        if frame < 6:
            t = 0.0  # hold on the bi-encoder order first
            phase = "bi-encoder order (coarse — independent encodings)"
            pcol = BLUE
        elif frame < 6 + n_move:
            t = (frame - 6) / (n_move - 1)
            t = t * t * (3 - 2 * t)  # smoothstep
            phase = "cross-encoder re-scoring… (joint encoding)"
            pcol = PURPLE
        else:
            t = 1.0
            phase = "re-ranked order — gold lifted to #1"
            pcol = GREEN
        # current vertical position of each doc (0 = top)
        positions = {doc: (1 - t) * bi_pos[doc] + t * rr_pos[doc] for doc in pool}
        for doc in pool:
            y = n - positions[doc]  # top of chart = best rank
            is_gold = doc == GOLD_INDEX
            # bar length grows with the cross-encoder score as t advances (relevance "lights up")
            score_frac = norm(ce_scores[doc])
            length = 0.15 + t * 0.8 * score_frac
            color = GREEN if is_gold else SLATE
            ax.barh(y, length, height=0.72, color=color, alpha=0.9 if is_gold else 0.55,
                    edgecolor=INK, linewidth=1.1 if is_gold else 0.6)
            label = _short(CORPUS[doc])
            ax.text(-0.02, y, f"doc[{doc}]  {label}", ha="right", va="center", fontsize=7.6,
                    color=GREEN if is_gold else INK, fontweight="bold" if is_gold else "normal")
            if is_gold:
                rank_now = sorted(pool, key=lambda d: positions[d]).index(doc) + 1
                ax.annotate(f"#{rank_now}", (length, y), color=GREEN, fontsize=10, fontweight="bold",
                            ha="left", va="center", xytext=(4, 0), textcoords="offset points")
        ax.set_xlim(0, 1.15)
        ax.set_ylim(-0.6, n + 0.1)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # title + phase live in the figure margin (not the axes) so they never overlap the bars
        fig.suptitle(f"Re-ranking the bi-encoder's top-{RETRIEVE_K} pool", fontsize=13, color=INK,
                     y=0.975, fontweight="bold")
        fig.text(0.5, 0.915, phase, ha="center", fontsize=10.5, color=pcol, fontweight="bold")
        fig.text(0.5, 0.875, f"query: {QUERY}", ha="center", fontsize=8.8, color=AMBER, style="italic")
        return []

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag06_rerank.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
