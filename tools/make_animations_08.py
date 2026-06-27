"""Animated (GIF) figure generator for 08-Long-Context-Methods.

Companion to make_figures_08.py. Animates the intuition figure (rope_clock_hands) as what its name
promises: actual rotating clock hands. As the token position m advances, each RoPE frequency pair
rotates at its own speed -- the fast pair (i=0) spins a full radian per token, the slow pairs barely
move over the whole window. The angle-vs-position line (the static figure) traces alongside, so the
animation both *shows the hands turn* and reproduces the static plot. Final frame == the static plot.

    python make_animations_08.py

Writes ../images (chapter-local) /rope_clock_hands.gif via matplotlib FuncAnimation + PillowWriter.
Frequencies, base, head dim, train length, and palette are imported from make_figures_08, so the
animation matches the chapter exactly.

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
from matplotlib.patches import Circle

from make_figures_08 import (
    BLUE,
    GRID,
    HEAD_DIM,
    INK,
    PURPLE,
    GREEN,
    SLATE,
    TRAIN_LEN,
    rope_angles,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "09. LLMs" / "08-Long-Context-Methods" / "images"
DPI = 95
FPS = 18

# Three representative frequency pairs: fastest, a middle one, and (near-)slowest.
DIALS = [
    (0, BLUE, "fast pair (i=0)"),
    (1, GREEN, "medium pair (i=1)"),
    (HEAD_DIM // 2 - 1, PURPLE, f"slow pair (i={HEAD_DIM // 2 - 1})"),
]


def _draw_dial(ax: plt.Axes, theta: float, color: str, label: str) -> None:
    ax.clear()
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=SLATE, linewidth=1.4))
    # the swept angle so far (from 0), as a faint wedge of the colour
    ts = np.linspace(0, theta, 60)
    ax.fill(np.concatenate([[0], np.cos(ts)]), np.concatenate([[0], np.sin(ts)]),
            color=color, alpha=0.12, zorder=1)
    ax.plot([0, np.cos(theta)], [0, np.sin(theta)], color=color, linewidth=3, zorder=3)
    ax.plot([np.cos(theta)], [np.sin(theta)], "o", color=color, markersize=7, zorder=4)
    ax.plot([0], [0], "o", color=INK, markersize=4, zorder=5)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{label}\nθ = {theta:5.2f} rad", fontsize=9.5, color=color, fontweight="bold")


def make_clock_hands_gif() -> None:
    positions = np.arange(0, TRAIN_LEN)
    fast = rope_angles(positions, 0)
    slow = rope_angles(positions, HEAD_DIM // 2 - 1)
    m_path = np.linspace(0, TRAIN_LEN - 1, 76)  # smooth position sweep
    frames = list(range(len(m_path))) + [len(m_path) - 1] * 16  # hold the final state

    fig = plt.figure(figsize=(10.4, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.45, wspace=0.25)
    dial_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ax_line = fig.add_subplot(gs[1, :])

    (l_fast,) = ax_line.plot([], [], "-o", color=BLUE, linewidth=2.2, markersize=4,
                             label="fast pair (i=0): angle = position")
    (l_slow,) = ax_line.plot([], [], "-s", color=PURPLE, linewidth=2.2, markersize=4,
                             label=f"slow pair (i={HEAD_DIM // 2 - 1}): barely rotates")
    ax_line.set_xlim(0, TRAIN_LEN - 1)
    ax_line.set_ylim(-0.5, fast.max() + 0.5)
    ax_line.set_xlabel("token position m")
    ax_line.set_ylabel("rotation angle θ (rad)")
    ax_line.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax_line.set_axisbelow(True)
    for side in ("top", "right"):
        ax_line.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_line.spines[side].set_color(SLATE)
    ax_line.tick_params(colors=INK)
    ax_line.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("RoPE clock-hands: each feature pair rotates at its own speed",
                 fontweight="bold", fontsize=13, color=INK)

    def update(fi: int):
        m = float(m_path[fi])
        for ax, (pair, color, label) in zip(dial_axes, DIALS):
            _draw_dial(ax, rope_angles(np.array([m]), pair)[0], color, label)
        k = int(np.floor(m)) + 1
        l_fast.set_data(positions[:k], fast[:k])
        l_slow.set_data(positions[:k], slow[:k])
        return (l_fast, l_slow)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "rope_clock_hands.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_clock_hands_gif()
