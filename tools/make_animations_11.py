"""Animated (GIF) figure generator for 11-Knowledge-Distillation.

Companion to make_figures_11.py. Animates the intuition figure: the teacher's softened class
distribution with the distillation temperature *swept*, so the reader watches the "dark knowledge"
emerge -- at T=1 the cat class dominates and the runner-ups are invisible; as T rises the relative
mass on dog and lynx (the similarity structure the student learns from) lifts into view.

    python make_animations_11.py

Writes ../../images/kd_temperature_softening.gif (matplotlib FuncAnimation + PillowWriter). Classes,
teacher logits, the temperature softmax, and palette are imported from make_figures_11, so the
animation matches the static figure exactly.

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

from make_figures_11 import (
    AMBER,
    CLASSES,
    GRID,
    INK,
    NAVY,
    SLATE,
    TEACHER_LOGITS,
    softmax_with_temperature,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "09. LLMs" / "images"
DPI = 95
FPS = 18
ARGMAX = int(np.argmax(TEACHER_LOGITS))


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.tick_params(colors=INK)


def make_softening_gif() -> None:
    sweep = np.concatenate([
        np.linspace(1.0, 8.0, 44),   # soften: dark knowledge rises
        np.linspace(8.0, 1.0, 36),   # sharpen back
    ])
    holds = {0: 14, 43: 14}  # hold at T=1 (hard) and T=8 (soft)
    order: list[int] = []
    for i in range(len(sweep)):
        order.append(i)
        order += [i] * holds.get(i, 0)

    n = len(CLASSES)
    x = np.arange(n)
    colors = [NAVY if i == ARGMAX else AMBER for i in range(n)]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(x, np.zeros(n), color=colors, edgecolor="white", width=0.72, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("teacher probability")
    _style_axis(ax)
    title = ax.set_title("", fontweight="bold", color=INK)
    readout = fig.text(0.5, 0.015, "", ha="center", color=INK, fontsize=11, fontweight="bold")

    def update(i: int):
        t = float(sweep[i])
        probs = softmax_with_temperature(TEACHER_LOGITS, t)
        for b, p in zip(bars, probs):
            b.set_height(p)
        dark = ", ".join(f"{CLASSES[j]}={probs[j]:.2f}" for j in (1, 2))  # dog, lynx = dark knowledge
        title.set_text(f"Distillation temperature reveals dark knowledge  (T = {t:.1f})")
        readout.set_text(f"T = {t:.1f}    {CLASSES[ARGMAX]} = {probs[ARGMAX]:.2f}    "
                         f"dark knowledge: {dark}")
        return bars

    anim = FuncAnimation(fig, update, frames=order, interval=1000 / FPS, blit=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "kd_temperature_softening.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_softening_gif()
