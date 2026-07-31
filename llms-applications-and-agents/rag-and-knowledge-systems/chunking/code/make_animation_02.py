"""Animated (GIF) intuition figure for 02-Document-Chunking-Strategies.

Companion to the static PNGs. Where those show final states, this brings the *mechanism* to life: a
sliding window steps across the document laying down chunks, and you watch a fact that straddles a
boundary get shattered WITHOUT overlap, then rescued WITH overlap — the index-card intuition in
motion (cut mid-thought and the card is useless; overlap re-includes the seam so the idea survives).

    python make_animation_02.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). The token
strip and the fact's position are the chapter's own toy; the two passes (no-overlap then overlap=2)
mirror the overlap figure exactly.

Produced:
  rag02_sliding_window.gif -- a window slides across the document twice: first with no overlap (the
                              fact at tokens 3-4 is split), then with overlap (a chunk re-includes the
                              seam, so the fact lands whole in one chunk).

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

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
FPS = 4  # slow enough to read each window placement
HOLD_FRAMES = 4  # dwell on each pass's final verdict

N_TOKENS = 10  # abstract document tokens
WINDOW = 4  # chunk width in tokens
FACT_TOKENS = (3, 4)  # the fact straddles tokens 3 and 4 (the boundary region)


def _chunk_starts(overlap: int) -> list[int]:
    """Window start positions for a given overlap (step = WINDOW - overlap)."""
    step = WINDOW - overlap
    starts, s = [], 0
    while s < N_TOKENS:
        starts.append(s)
        s += step
    return starts


def _fact_whole_in_any(starts: list[int]) -> bool:
    """Does some window fully contain BOTH fact tokens? (the fact survives chunking)."""
    return any(s <= FACT_TOKENS[0] and FACT_TOKENS[1] <= s + WINDOW - 1 for s in starts)


def build_animation() -> None:
    passes = [(0, "Pass 1 — NO overlap"), (2, "Pass 2 — overlap = 2")]
    # one frame per window placement, per pass, plus a hold on each pass's verdict
    plan: list[tuple[int, str, int]] = []  # (overlap, title, chunks_shown_so_far)
    for overlap, title in passes:
        starts = _chunk_starts(overlap)
        for i in range(len(starts)):
            plan.append((overlap, title, i + 1))
        for _ in range(HOLD_FRAMES):
            plan.append((overlap, title, len(starts)))

    fig, ax = plt.subplots(figsize=(8.8, 3.4))

    def update(frame: int):
        overlap, title, n_shown = plan[frame]
        starts = _chunk_starts(overlap)
        ax.clear()
        ax.set_xlim(-0.5, N_TOKENS + 0.5)
        ax.set_ylim(-2.4, 1.6)
        ax.axis("off")
        ax.set_title(f"Sliding window lays down chunks   ·   {title}", fontsize=12, color=INK)

        # the document token strip
        for t in range(N_TOKENS):
            ax.add_patch(Rectangle((t, 0.5), 0.9, 0.6, facecolor="#EEF1F4", edgecolor=GRID))
            ax.text(t + 0.45, 0.8, str(t), fontsize=8, ha="center", va="center", color=SLATE)
        # the fact spans FACT_TOKENS (the boundary region)
        ax.add_patch(Rectangle((FACT_TOKENS[0], 0.5), FACT_TOKENS[1] - FACT_TOKENS[0] + 0.9, 0.6,
                               facecolor=AMBER, alpha=0.35, edgecolor=AMBER, linewidth=1.5))
        ax.text((FACT_TOKENS[0] + FACT_TOKENS[1] + 0.9) / 2, 1.32, "the fact", fontsize=9,
                ha="center", color=INK)

        # chunks laid down so far
        colors = [BLUE, PURPLE, GREEN, SLATE, AMBER]
        for ci in range(n_shown):
            s = starts[ci]
            width = min(WINDOW, N_TOKENS - s)
            row_y = -0.7 - (ci % 2) * 0.7  # stagger rows so overlapping chunks are both visible
            color = colors[ci % len(colors)]
            holds_fact = s <= FACT_TOKENS[0] and FACT_TOKENS[1] <= s + WINDOW - 1
            ax.add_patch(Rectangle((s, row_y), width, 0.55, facecolor=color,
                                    alpha=0.75 if holds_fact else 0.4,
                                    edgecolor=GREEN if holds_fact else color,
                                    linewidth=2.4 if holds_fact else 1.0))
            tag = f"chunk {ci}" + ("  ✓ holds fact" if holds_fact else "")
            ax.text(s + width / 2, row_y + 0.27, tag, fontsize=8.5, ha="center", va="center",
                    color=INK, fontweight="bold" if holds_fact else "normal")

        # verdict once all chunks for this pass are down
        if n_shown == len(starts):
            survived = _fact_whole_in_any(starts)
            msg = ("✓ the fact lands WHOLE in a chunk — retrievable"
                   if survived else "✗ the fact is SPLIT across chunks — unretrievable")
            ax.text(N_TOKENS / 2, -2.1, msg, fontsize=11, ha="center", color=GREEN if survived else RED,
                    fontweight="bold")
        return ax.patches

    anim = FuncAnimation(fig, update, frames=len(plan), interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag02_sliding_window.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
