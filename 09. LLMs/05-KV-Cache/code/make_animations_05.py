"""Animated (GIF) figure generator for 05-KV-Cache.

Companion to the chapter's static PNGs. Where those show the final state, these bring the
mechanism to life so a reader *watches* the cache fill and the redundant work pile up, step by
step, rather than reading it off a finished plot.

    python make_animations_05.py

GIFs are written to ../../images/ (the shared chapter image dir) and use matplotlib's PillowWriter
(no ffmpeg required). The numbers are the chapter's own:
  * recompute work is the EXACT per-step K/V projection count from kv_cache.py -- no-cache projects
    all t tokens each step (O(n^2) total), the cache projects one (O(n) total);
  * KV-cache memory uses the standard fp16 cache formula 2 (K,V) x L x d x seq x 2 bytes for the
    7B/13B/70B configs, the same curve as kv_memory_growth.png.

Animations produced:
  kv_recompute_waste.gif -- per-step K/V work (no-cache grows with position, cache stays at 1) plus
                            the cumulative O(n^2)-vs-O(n) total, the redundant area filling in.
  kv_memory_growth.gif   -- KV-cache size climbing linearly with context for 7B/13B/70B until it
                            rivals and then exceeds the model weights.

Verified on Python 3.12 / matplotlib 3.x / Pillow (PillowWriter).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 95
FPS = 16


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def make_recompute_waste_gif() -> None:
    """Watch the redundant work pile up: without a cache, step t re-projects K/V for ALL t tokens;
    with a cache it projects exactly one. The per-step gap and the cumulative O(n^2)-vs-O(n) total
    are the same counts kv_cache.py actually performs."""
    n = 24  # decode steps -- small enough to read every step, large enough to show the gap open
    steps = np.arange(1, n + 1)
    work_no = steps.astype(float)          # no-cache: project K/V for all t tokens at step t
    work_yes = np.ones(n, dtype=float)     # cache:    project K/V for the 1 new token
    cum_no = np.cumsum(work_no)            # O(n^2): t(t+1)/2
    cum_yes = np.cumsum(work_yes)          # O(n):   t

    reveal = list(range(1, n + 1))
    frames = reveal + [n] * 16  # hold the final state so the widening gap is readable

    fig, (ax_step, ax_cum) = plt.subplots(1, 2, figsize=(11.4, 4.7))

    # ---- Left: per-step K/V projection work -------------------------------------------
    (line_no,) = ax_step.plot([], [], color=RED, linewidth=2.6,
                              label="no cache — re-project all $t$ tokens")
    (line_yes,) = ax_step.plot([], [], color=GREEN, linewidth=2.6,
                               label="KV cache — project the 1 new token")
    (dot_no,) = ax_step.plot([], [], "o", color=RED, markersize=6, zorder=5)
    (dot_yes,) = ax_step.plot([], [], "o", color=GREEN, markersize=6, zorder=5)
    fill_step: list = []
    ax_step.set_xlim(0, n)
    ax_step.set_ylim(0, n + 1)
    ax_step.set_xlabel("decode step $t$")
    ax_step.set_ylabel("K/V projections this step")
    ax_step.legend(loc="upper left", frameon=False, fontsize=9)
    _style_axis(ax_step)
    ax_step.set_title("Per-step work: the cache stays flat", fontweight="bold")

    # ---- Right: cumulative total work (the O(n^2) vs O(n) divergence) ------------------
    (cline_no,) = ax_cum.plot([], [], color=RED, linewidth=2.6, label="no cache — $O(n^2)$")
    (cline_yes,) = ax_cum.plot([], [], color=GREEN, linewidth=2.6, label="KV cache — $O(n)$")
    (cdot_no,) = ax_cum.plot([], [], "o", color=RED, markersize=6, zorder=5)
    (cdot_yes,) = ax_cum.plot([], [], "o", color=GREEN, markersize=6, zorder=5)
    fill_cum: list = []
    ax_cum.set_xlim(0, n)
    ax_cum.set_ylim(0, cum_no[-1] * 1.05)
    ax_cum.set_xlabel("decode step $t$")
    ax_cum.set_ylabel("cumulative K/V projections")
    ax_cum.legend(loc="upper left", frameon=False, fontsize=9)
    _style_axis(ax_cum)
    ax_cum.set_title("Cumulative work: the gap is pure waste", fontweight="bold")

    counter = fig.text(0.5, 0.015, "", ha="center", color=INK, fontsize=10, fontweight="bold")
    fig.suptitle("Why the KV cache exists: the redundant work it removes",
                 fontweight="bold", fontsize=13)

    def update(k: int):
        x = steps[:k]
        line_no.set_data(x, work_no[:k])
        line_yes.set_data(x, work_yes[:k])
        cline_no.set_data(x, cum_no[:k])
        cline_yes.set_data(x, cum_yes[:k])
        j = k - 1
        dot_no.set_data([steps[j]], [work_no[j]])
        dot_yes.set_data([steps[j]], [work_yes[j]])
        cdot_no.set_data([steps[j]], [cum_no[j]])
        cdot_yes.set_data([steps[j]], [cum_yes[j]])

        for coll in (*fill_step, *fill_cum):
            coll.remove()
        fill_step.clear()
        fill_cum.clear()
        fill_step.append(ax_step.fill_between(x, work_yes[:k], work_no[:k],
                                              color=RED, alpha=0.13, zorder=1))
        fill_cum.append(ax_cum.fill_between(x, cum_yes[:k], cum_no[:k],
                                            color=RED, alpha=0.13, zorder=1))
        saved = cum_no[j] / cum_yes[j]
        counter.set_text(
            f"step {int(steps[j]):>2}/{n}    "
            f"no-cache total {int(cum_no[j]):>3} projections    "
            f"cache total {int(cum_yes[j]):>2}    "
            f"→ {saved:.1f}× less work"
        )
        return line_no, line_yes, cline_no, cline_yes

    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "kv_recompute_waste.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def make_memory_growth_gif() -> None:
    """Watch the KV cache climb with context until it rivals, then exceeds, the model weights.
    Cache bytes (fp16, batch 1) = 2 (K,V) x L x d x seq x 2; weights = params x 2 bytes."""
    gb = 1024**3
    # (label, n_layers, d_model, n_params, colour)
    configs = [
        ("7B  (L=32, d=4096)", 32, 4096, 7e9, GREEN),
        ("13B (L=40, d=5120)", 40, 5120, 13e9, BLUE),
        ("70B (L=80, d=8192)", 80, 8192, 70e9, PURPLE),
    ]
    max_ctx = 65536  # 64k context: long enough that even the 70B cache overtakes its weights
    ctx = np.linspace(0, max_ctx, 130)

    def cache_gb(layers: int, d: int, seq: np.ndarray) -> np.ndarray:
        return 2 * layers * d * seq * 2 / gb  # 2 tensors (K,V) x L x d x seq x 2 bytes (fp16)

    series = [(lbl, cache_gb(L, d, ctx), p * 2 / gb, c) for lbl, L, d, p, c in configs]
    y_top = max(max(s[1][-1] for s in series), max(s[2] for s in series)) * 1.08

    reveal = list(range(2, len(ctx) + 1, 2))
    if reveal[-1] != len(ctx):
        reveal.append(len(ctx))
    frames = reveal + [len(ctx)] * 16

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.set_xlim(0, max_ctx)
    ax.set_ylim(0, y_top)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("memory (GiB), batch = 1, fp16")
    _style_axis(ax)
    ax.set_title("The KV cache grows linearly — and overtakes the weights", fontweight="bold")

    lines, dots = [], []
    for lbl, cache, w_gb, colour in series:
        (ln,) = ax.plot([], [], color=colour, linewidth=2.6, label=f"{lbl} cache")
        (dt,) = ax.plot([], [], "o", color=colour, markersize=6, zorder=6)
        # The model-weights ceiling for this config (dotted), so the crossover is visible.
        ax.axhline(w_gb, color=colour, linewidth=1.2, linestyle=":", alpha=0.85, zorder=2)
        ax.text(max_ctx * 0.985, w_gb + 1.5, f"{lbl.split()[0]} weights ≈ {w_gb:.0f} GiB",
                ha="right", va="bottom", color=colour, fontsize=8.5)
        lines.append(ln)
        dots.append(dt)

    ax.legend(loc="upper left", frameon=False, fontsize=9)
    counter = fig.text(0.5, 0.015, "", ha="center", color=INK, fontsize=10, fontweight="bold")

    def update(k: int):
        x = ctx[:k]
        for (lbl, cache, w_gb, colour), ln, dt in zip(series, lines, dots):
            ln.set_data(x, cache[:k])
            dt.set_data([ctx[k - 1]], [cache[k - 1]])
        cur = ctx[k - 1]
        msg = f"context {int(cur):,} tokens    "
        msg += "    ".join(f"{lbl.split()[0]} cache {cache[k - 1]:.0f} GiB"
                           for lbl, cache, _w, _c in series)
        counter.set_text(msg)
        return tuple(lines)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "kv_memory_growth.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    make_recompute_waste_gif()
    make_memory_growth_gif()
    print("all animations written to", OUT_DIR)


if __name__ == "__main__":
    main()
