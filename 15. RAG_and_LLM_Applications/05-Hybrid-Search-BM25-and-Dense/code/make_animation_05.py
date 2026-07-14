"""Animated (GIF) intuition figure for 05-Hybrid-Search-BM25-and-Dense.

Companion to the static PNGs. Where those show final states, this brings the FUSION dial to life:
an alpha slider sweeps from pure sparse (alpha=0) to pure dense (alpha=1), and TWO fused rankings
re-order live -- one per probe type. You watch each correct passage (left: the exact-code line;
right: the paraphrase line) sit where its winning lens puts it, and see that only the BLENDED middle
(alpha ~0.6) puts BOTH golds at #1 at once -- which neither pure-sparse (alpha=0) nor pure-dense
(alpha=1) manages.

    python make_animation_05.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). The
scores are the chapter's OWN: BM25 + the dense bi-encoder from hybrid_search.py over the chapter
corpus and the two blind-spot probes -- so the rank each passage takes is exactly what the page and
notebook report.

Produced:
  rag05_alpha_fusion.gif -- the alpha dial sweeps; the paraphrase gold snaps from #6 to #1 at the
                            blend, while the exact-code gold holds #1 until pure-dense drops it --
                            the blended middle is the only setting that serves both.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter) / sentence-transformers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from hybrid_search import (
    BM25,
    DenseRetriever,
    build_probes,
    full_corpus,
    min_max_normalize,
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
DPI = 100
FPS = 10


def _short(text: str, n: int = 40) -> str:
    """Trim a passage to a short label for the bar chart."""
    return text if len(text) <= n else text[: n - 1] + "…"


def build_animation() -> None:
    corpus = full_corpus()
    bm25 = BM25(corpus)
    dense = DenseRetriever(corpus)
    code_probe, para_probe = build_probes(corpus)

    # Per-probe normalized lens scores (the chapter's own numbers).
    panels = []
    for probe, gold_color in ((code_probe, BLUE), (para_probe, GREEN)):
        dn = min_max_normalize(dense.all_scores(probe.query))
        sp = min_max_normalize(bm25.all_scores(probe.query))
        panels.append((probe, dn, sp, gold_color))

    labels = [_short(d) for d in corpus]
    n = len(corpus)

    alphas = np.concatenate([
        np.linspace(0.0, 1.0, 28),  # sweep up
        np.full(5, 1.0),            # hold at pure dense (exact-code drops here)
        np.linspace(1.0, 0.6, 10),  # settle back to the good blend
        np.full(14, 0.6),           # hold on the best blend (both golds #1)
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.82, bottom=0.10, wspace=0.62)

    def draw_panel(ax, probe, dn, sp, gold_color, alpha: float) -> None:
        ax.clear()
        scores = alpha * dn + (1 - alpha) * sp
        order = np.argsort(scores)[::-1]
        y = np.arange(n)[::-1]
        colors = [gold_color if int(d) == probe.gold else SLATE for d in order]
        ax.barh(y, scores[order], color=colors, edgecolor=INK, linewidth=0.6, height=0.72, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[int(d)] for d in order], fontsize=7.2)
        gold_rank = list(map(int, order)).index(probe.gold) + 1
        ax.annotate(f"#{gold_rank}", (scores[order][gold_rank - 1], y[gold_rank - 1]), fontsize=10,
                    color=gold_color, va="center", ha="left", xytext=(4, 0),
                    textcoords="offset points", fontweight="bold")
        ax.set_xlim(0, 1.05)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=INK, labelsize=8)
        hit = "✓ at #1" if gold_rank == 1 else f"gold at #{gold_rank}"
        hit_col = GREEN if gold_rank == 1 else RED
        ax.set_title(f"{probe.label.split(' (')[0]}\n{hit}", fontsize=10.5, color=hit_col, pad=8)
        ax.set_xlabel("fused score", fontsize=8.5, color=INK)

    def update(frame: int):
        alpha = float(alphas[frame])
        for ax, (probe, dn, sp, gold_color) in zip(axes, panels):
            draw_panel(ax, probe, dn, sp, gold_color, alpha)
        if alpha <= 0.02:
            regime, rcol = "alpha = 0.00  ·  pure SPARSE (BM25) — paraphrase stranded at #6", AMBER
        elif alpha >= 0.98:
            regime, rcol = "alpha = 1.00  ·  pure DENSE — exact-code drops to #2", BLUE
        elif 0.55 <= alpha <= 0.72:
            regime, rcol = f"alpha = {alpha:.2f}  ·  BLEND — BOTH golds at #1 ✓ (only the middle does this)", GREEN
        else:
            regime, rcol = f"alpha = {alpha:.2f}  ·  blending…", PURPLE
        fig.suptitle(regime, fontsize=13.5, color=rcol, y=0.96, fontweight="bold")
        # the dial caption
        fig.text(0.5, 0.015, "alpha:  0 = pure BM25 (lexical)   →   1 = pure dense (semantic)",
                 ha="center", fontsize=9, color=INK)
        return []

    anim = FuncAnimation(fig, update, frames=len(alphas), interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag05_alpha_fusion.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
