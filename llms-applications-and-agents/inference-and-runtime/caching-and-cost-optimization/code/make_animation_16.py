"""Animated (GIF) intuition figure for 16-Caching-and-Cost-Optimization.

Companion to the static PNGs. It brings the semantic cache to life -- queries streaming in one at a
time: a cold/miss query lights RED, computes, and drops a new entry into the cache store; a repeat or
paraphrase lights GREEN and is served instantly from the cache. As the stream flows, the running
HIT RATE climbs and the cost/latency saved accumulates -- you watch caching pay off.

    python make_animation_16.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). Every
query, its hit/miss verdict, the running hit rate, and the cost/latency saved are REAL caching_cost.py
output over the ch5 all-MiniLM encoder; only the MISS answer text and the exact latency constants are
illustrative stand-ins (stated in the honesty footer).

Produced:
  rag16_cache_stream.gif -- queries stream in; hits served instantly, misses fill the cache, hit rate climbs.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter) / sentence-transformers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from caching_cost import (
    CACHE_THRESHOLD,
    FULL_CALL_MS,
    HIT_MS,
    QUERY_STREAM,
    STREAM_ANSWERS,
    DenseRetriever,
    SemanticCache,
    full_call_cost_usd,
    full_corpus,
    hit_cost_usd,
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
FPS = 12


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def build_animation() -> None:
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    # replay the stream once to record each query's verdict, cosine, and the growing cache size
    cache = SemanticCache(dense, threshold=CACHE_THRESHOLD)
    steps = []  # (query, hit, cosine, cache_size_after)
    for query in QUERY_STREAM:
        lookup = cache.lookup(query)
        hit = lookup.hit
        if not hit:
            cache.store(query, STREAM_ANSWERS[query])
        steps.append((query, hit, lookup.best_cosine, len(cache.entries)))
    n = len(steps)

    intro = 6
    per_query = 14
    hold = 26
    total = intro + n * per_query + hold

    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.04)

    def update(frame: int):
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        shown = min(max((frame - intro) // per_query + 1, 0), n)
        in_hold = frame >= intro + n * per_query

        fig.suptitle("Semantic cache: repeats and paraphrases stream in as HITS, the hit rate climbs",
                     fontsize=12.5, color=INK, y=0.965, fontweight="bold")
        ax.text(0.02, 0.88, "query stream", ha="left", fontsize=9.4, color=INK, fontweight="bold")

        # rows: each revealed query, coloured by hit/miss with its cosine
        hits_so_far = 0
        ys = [0.80 - i * 0.083 for i in range(n)]
        for i, ((query, hit, cos, _size), y) in enumerate(zip(steps, ys)):
            revealed = i < shown
            if revealed:
                face = GREEN if hit else RED
                tag = f"HIT  (cos {cos:.2f})" if hit else ("MISS (cold)" if cos == 0.0 else f"MISS (cos {cos:.2f})")
                if hit:
                    hits_so_far += 1
            else:
                face, tag = SLATE, ""
            alpha = 0.16 if revealed else 0.05
            ax.add_patch(plt.Rectangle((0.03, y - 0.033), 0.66, 0.062, facecolor=face, alpha=alpha,
                         edgecolor=face if revealed else GRID, linewidth=1.2))
            ax.text(0.05, y, _short(query, 52), ha="left", va="center", fontsize=7.4,
                    color=INK if revealed else SLATE)
            if revealed:
                ax.text(0.70, y, tag, ha="left", va="center", fontsize=7.8,
                        color=GREEN if hit else RED, fontweight="bold")

        # the cache store panel (right) -- grows as misses fill it
        cache_size = steps[shown - 1][3] if shown else 0
        ax.text(0.88, 0.88, "cache store", ha="center", fontsize=9.0, color=BLUE, fontweight="bold")
        ax.add_patch(plt.Rectangle((0.80, 0.30), 0.17, 0.54, facecolor=BLUE, alpha=0.07, edgecolor=BLUE, linewidth=1.2))
        for j in range(cache_size):
            ax.add_patch(plt.Rectangle((0.815, 0.78 - j * 0.075), 0.14, 0.055, facecolor=BLUE, alpha=0.5,
                         edgecolor=INK, linewidth=0.8))
            ax.text(0.885, 0.807 - j * 0.075, f"entry {j+1}", ha="center", va="center", fontsize=7.0, color="white", fontweight="bold")

        # running readout: hit rate + cost/latency saved (real numbers)
        hit_rate = hits_so_far / shown if shown else 0.0
        cost_saved = hits_so_far * (full_call_cost_usd() - hit_cost_usd())
        lat_saved = hits_so_far * (FULL_CALL_MS - HIT_MS)
        # place the readout in the empty right-side column so it never overlaps the query rows
        ax.text(0.88, 0.20, f"hit rate: {hits_so_far}/{shown} = {hit_rate:.3f}" if shown else "hit rate: —",
                ha="center", fontsize=11.5, color=GREEN, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=GREEN, alpha=0.10, edgecolor=GREEN))
        ax.text(0.5, 0.115, f"cost saved so far: ${cost_saved:.5f}    latency saved: {lat_saved:.0f} ms",
                ha="center", fontsize=9.6, color=INK, fontweight="bold")
        if in_hold:
            ax.text(0.5, 0.055, f"final: {hits_so_far}/{n} hits — repeats and paraphrases served for ~free",
                    ha="center", fontsize=9.0, color=GREEN, style="italic", fontweight="bold")

        # honesty footer
        ax.text(0.5, -0.005, "hit/miss verdicts, cosines, hit rate and cost/latency saved are REAL "
                "caching_cost output; only the answer text + exact latency constants are illustrative",
                ha="center", va="bottom", fontsize=6.6, color=SLATE, style="italic")
        prog = (frame + 1) / total
        ax.add_patch(plt.Rectangle((0.90, 0.008), 0.07 * prog, 0.006, facecolor=GREEN, edgecolor="none"))
        return []

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag16_cache_stream.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
