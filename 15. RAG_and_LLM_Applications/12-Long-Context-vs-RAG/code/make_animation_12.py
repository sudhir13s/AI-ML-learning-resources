"""Animated (GIF) intuition figure for 12-Long-Context-vs-RAG.

Companion to the static PNGs. It animates the core tradeoff: as the corpus grows, the per-query cost
of STUFFING the whole thing climbs relentlessly (it re-pays for every chunk on every query), while
RAG's cost stays FLAT (it only ever sends the top-k). You watch the stuffing curve rise, cross the
flat RAG line at the crossover, and keep climbing — the moment "just paste it all in" stops paying
off. Every dollar figure is the chapter's own cost arithmetic (long_context_vs_rag.py) — our
measurement.

    python make_animation_12.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed).

Produced:
  rag12_cost_crossover.gif -- the stuffing cost curve rising past the flat RAG line, the crossover lit.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from long_context_vs_rag import (
    RETRIEVE_K,
    cost_crossover_chunks,
    query_cost_usd,
    rag_tokens,
    stuff_tokens,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
AMBER = "#7A6528"
INK = "#1C2530"
GRID = "#D4D9DF"
SLATE_OK = "#4A5B6E"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 100
FPS = 12
PRICE = 3.00


def build_animation() -> None:
    # the corpus-size sweep (log-spaced) and the two cost curves
    chunk_sizes = np.unique(np.round(np.logspace(0, 3.2, 40)).astype(int))  # 1 .. ~1585 chunks
    stuff_costs = np.array([query_cost_usd(stuff_tokens(int(c)), PRICE) for c in chunk_sizes])
    rag_cost = query_cost_usd(rag_tokens(), PRICE)
    crossover = cost_crossover_chunks()

    reveal = len(chunk_sizes)
    hold = 20
    total = reveal + hold

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.13)

    def update(frame: int):
        ax.clear()
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=INK, labelsize=9)

        n = min(frame + 1, reveal)  # how many points of the stuffing curve are revealed
        in_hold = frame >= reveal

        # flat RAG line (always shown in full)
        ax.axhline(rag_cost, color=BLUE, linewidth=2.2, linestyle="-",
                   label=f"RAG: retrieve k={RETRIEVE_K} (flat — {rag_tokens()} tok/query)")
        # stuffing curve, revealed progressively
        ax.plot(chunk_sizes[:n], stuff_costs[:n], "-o", color=RED, linewidth=2.2, markersize=4,
                label="stuff whole corpus (grows every query)")

        # once revealed past the crossover, light it up
        current_chunks = chunk_sizes[n - 1]
        if current_chunks >= crossover or in_hold:
            ax.axvline(crossover, color=AMBER, linewidth=1.4, linestyle="--")
            ax.annotate(f"crossover ≈ {crossover} chunks\nRAG wins beyond here",
                        xy=(crossover, rag_cost), xytext=(crossover * 4, rag_cost * 5),
                        fontsize=8.8, color=AMBER, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))

        # a moving readout of the current cost gap
        cur_stuff = stuff_costs[n - 1]
        gap = cur_stuff / rag_cost
        verdict = "RAG cheaper" if cur_stuff > rag_cost else "stuffing cheaper"
        vcol = GREEN if cur_stuff > rag_cost else SLATE_OK
        ax.text(0.98, 0.06, f"at {current_chunks:,} chunks: stuffing = {gap:,.1f}× RAG  →  {verdict}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.8, color=vcol,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=vcol, alpha=0.9))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, chunk_sizes[-1] * 1.1)
        ax.set_ylim(stuff_costs.min() * 0.6, stuff_costs.max() * 1.6)
        ax.set_xlabel("corpus size (chunks, log scale)")
        ax.set_ylabel("input cost per query, USD (log)")
        ax.set_title("Long-context (stuff) vs RAG: cost per query as the corpus grows",
                     fontsize=12, color=INK, fontweight="bold", pad=10)
        ax.legend(loc="upper left", fontsize=8.6, framealpha=0.9)
        ax.text(0.98, 0.94, "OUR MEASUREMENT (real cost arithmetic)", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.4, color=GREEN, style="italic", fontweight="bold")
        # per-frame progress tick keeps frames distinct so the GIF writer preserves pacing
        prog = (frame + 1) / total
        ax.add_patch(plt.Rectangle((0.90, 0.005), 0.08 * prog, 0.006, transform=ax.transAxes,
                     facecolor=GREEN, edgecolor="none"))
        return []

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag12_cost_crossover.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
