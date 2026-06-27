"""Animated (GIF) figure generator for 20-Hallucination-and-Alignment-Basics.

Companion to make_figures_20.py. Animates the intuition figure: the toy knowledge distribution
with the temperature dial *swept*, so the reader watches mass bleed off the supported answer
(green) onto the wrong distractors (red) as T rises -- the visual cause of the rising
unsupported-claim rate.

    python make_animations_20.py

Writes ../../images/hall_temperature_dists.gif (matplotlib FuncAnimation + PillowWriter). Logits,
the temperature helper, the supported-token index, and palette are imported from the chapter's own
modules, so the animation matches the static T=0.5/1.0/2.0 figure exactly.

Verified on Python 3.12 / matplotlib 3.x / torch 2.x / Pillow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from hallucination_alignment import (
    ANSWER_VOCAB,
    SUPPORTED_IDX,
    base_logits_tensor,
    softmax_with_temperature,
)
from make_figures_20 import GREEN, GRID, INK, RED, SLATE

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 95
FPS = 18
LOGITS = base_logits_tensor(device="cpu")
BAR_COLORS = [GREEN if i == SUPPORTED_IDX else RED for i in range(len(ANSWER_VOCAB))]


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.tick_params(colors=INK)


def make_temperature_dists_gif() -> None:
    sweep = np.concatenate([
        np.linspace(0.5, 1.0, 26),
        np.linspace(1.0, 2.0, 34),
        np.linspace(2.0, 0.5, 40),
    ])
    holds = {0: 12, 25: 12, 59: 12}  # hold on T = 0.5, 1.0, 2.0
    order: list[int] = []
    for i in range(len(sweep)):
        order.append(i)
        order += [i] * holds.get(i, 0)

    n = len(ANSWER_VOCAB)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(x, np.zeros(n), color=BAR_COLORS, edgecolor="white", width=0.72, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}\n{'(supported)' if i == SUPPORTED_IDX else '(wrong)'}"
                        for i, w in enumerate(ANSWER_VOCAB)], fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("probability")
    _style_axis(ax)
    title = ax.set_title("", fontweight="bold", color=INK)
    readout = fig.text(0.5, 0.015, "", ha="center", color=INK, fontsize=11, fontweight="bold")

    def update(i: int):
        t = float(sweep[i])
        probs = softmax_with_temperature(LOGITS, t).numpy()
        for b, p in zip(bars, probs):
            b.set_height(p)
        supported = float(probs[SUPPORTED_IDX])
        title.set_text(f"Temperature bleeds mass off the supported answer  (T = {t:.2f})")
        readout.set_text(f"T = {t:.2f}    supported (green) mass = {supported:.2f}    "
                         f"→ the rest leaks onto wrong answers (red)")
        return bars

    anim = FuncAnimation(fig, update, frames=order, interval=1000 / FPS, blit=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "hall_temperature_dists.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_temperature_dists_gif()
