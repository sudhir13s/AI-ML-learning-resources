"""Animated (GIF) figure generator for 09-Inference-Optimization-and-Serving.

Brings the batching-timeline intuition to life: the same 6 ragged requests on a 3-slot engine,
played forward in time. A sweeping clock reveals each request as it runs; under STATIC batching the
freed slots sit idle (hatched) until the wave's longest request ends, while under CONTINUOUS
batching a freed slot is backfilled the very next step (D starts the instant A finishes) -- so the
continuous panel visibly finishes sooner. Companion to the static serving_batching_timeline.png.

    python make_animations_09.py

Writes ../../images/serving_batching_timeline.gif (matplotlib FuncAnimation + PillowWriter). The
schedules are the exact simulation output used by the static figure; palette matches the chapter.

Verified on Python 3.12 / matplotlib 3.x / Pillow.
"""

from __future__ import annotations

import _pathsetup  # noqa: F401  (sys.path bootstrap for the moved generator)

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
INK = "#1C2530"

OUT_DIR = Path(__file__).resolve().parent.parent / "09. LLMs" / "images"
DPI = 95
FPS = 18

# slot -> [(label, start, dur, color, is_idle)] -- the exact schedules from the simulation.
STATIC_BARS = {
    0: [("A", 0, 4, BLUE, False), ("idle", 4, 16, SLATE, True), ("D", 20, 60, RED, False)],
    1: [("B", 0, 6, BLUE, False), ("idle", 6, 14, SLATE, True),
        ("E", 20, 5, GREEN, False), ("idle", 25, 55, SLATE, True)],
    2: [("C", 0, 20, PURPLE, False), ("F", 20, 8, GREEN, False), ("idle", 28, 52, SLATE, True)],
}
CONT_BARS = {
    0: [("A", 0, 4, BLUE, False), ("D", 4, 60, RED, False)],
    1: [("B", 0, 6, BLUE, False), ("E", 6, 5, GREEN, False), ("F", 11, 8, GREEN, False),
        ("idle", 19, 45, SLATE, True)],
    2: [("C", 0, 20, PURPLE, False), ("idle", 20, 44, SLATE, True)],
}


def _draw(ax, bars, title, makespan, t) -> None:
    ax.clear()
    for slot, segs in bars.items():
        for label, start, dur, color, is_idle in segs:
            if t <= start:
                continue
            shown = min(dur, t - start)  # reveal the bar only up to the current time
            ax.add_patch(Rectangle((start, slot - 0.4), shown, 0.8,
                                   facecolor=color, alpha=0.28 if is_idle else 0.9,
                                   hatch="//" if is_idle else None,
                                   edgecolor="white", linewidth=1.0))
            if not is_idle and shown >= 4:
                ax.text(start + min(dur, shown) / 2, slot, label, ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
    if t >= makespan:
        ax.axvline(makespan, color=INK, ls="--", lw=1.6)
        ax.text(makespan + 0.6, 0.9, f"done\n{makespan * 10} ms", fontsize=9,
                fontweight="bold", color=INK)
    ax.axvline(min(t, 84), color="#C04030", lw=1.4, alpha=0.7)  # the clock cursor
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["slot 1", "slot 2", "slot 3"])
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(0, 84)
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def make_batching_gif() -> None:
    times = np.arange(0, 86, 2)
    frames = list(times) + [84] * 16  # hold the finished state

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=True)
    fig.suptitle("Same 6 ragged requests (4/6/20/60/5/8 tok), 3 slots: continuous finishes 1.25× sooner",
                 fontsize=12.5, fontweight="bold", color=INK)

    def update(t):
        _draw(ax1, STATIC_BARS, "STATIC — freed slots sit idle until the wave's longest ends (80 steps)", 80, t)
        _draw(ax2, CONT_BARS, "CONTINUOUS — a freed slot is backfilled the next step (64 steps)", 64, t)
        ax2.set_xlabel("Decode steps (10 ms each, modeled)")
        return ()

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "serving_batching_timeline.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_batching_gif()
