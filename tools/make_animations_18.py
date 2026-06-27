"""Animated (GIF) figure generator for 18-Decoding-and-Sampling.

Companion to make_figures_18.py. Animates the chapter's intuition figure: the same peaked
next-token distribution, but with the temperature dial *swept continuously* so the reader watches
the softmax sharpen (T<1) and flatten (T>1) in real time, with the entropy read-out moving with it.

    python make_animations_18.py

Writes ../../images/dec_temperature_softmax.gif (matplotlib FuncAnimation + PillowWriter). The
logits, temperature helper, entropy, and palette are imported from the chapter's own modules, so
the animation matches the static figure (T=0.5/1.0/2.0) exactly.

Verified on Python 3.12 / matplotlib 3.x / torch 2.x / Pillow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

import _pathsetup  # noqa: F401  (puts 09. LLMs/*/code + tools/ on sys.path for the moved generators)
from decoding_sampling import PEAKED_LOGITS, VOCAB, entropy_bits, softmax_with_temperature
from make_figures_18 import AMBER, GRID, INK, NAVY, SLATE

OUT_DIR = Path(__file__).resolve().parent.parent / "09. LLMs" / "images"
DPI = 95
FPS = 18
PEAKED = torch.tensor(PEAKED_LOGITS)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.tick_params(colors=INK)


def make_temperature_gif() -> None:
    # Sweep T over a smooth path, holding on the three anchor temperatures the static figure marks.
    sweep = np.concatenate([
        np.linspace(0.5, 1.0, 26),
        np.linspace(1.0, 2.0, 34),
        np.linspace(2.0, 0.5, 40),
    ])
    holds = {0: 12, 25: 12, 59: 12}  # frame index -> extra hold (T=0.5, 1.0, 2.0 anchors)
    order: list[int] = []
    for i in range(len(sweep)):
        order.append(i)
        order += [i] * holds.get(i, 0)

    n = len(VOCAB)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(x, np.zeros(n), color=NAVY, width=0.7, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(VOCAB, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("next-token candidates")
    ax.set_ylabel("probability")
    _style_axis(ax)
    title = ax.set_title("", fontweight="bold", color=INK)
    readout = fig.text(0.5, 0.015, "", ha="center", color=INK, fontsize=11, fontweight="bold")

    def update(i: int):
        t = float(sweep[i])
        probs = softmax_with_temperature(PEAKED, t).numpy()
        h = float(entropy_bits(softmax_with_temperature(PEAKED, t)))
        for b, p in zip(bars, probs):
            b.set_height(p)
            b.set_color(AMBER if t > 1.0 else NAVY)
        sharp = "sharpening → trust the top" if t < 1.0 else ("flattening → spread it around" if t > 1.0 else "the model's own distribution")
        title.set_text(f"Temperature reshapes the next-token softmax  (T = {t:.2f})")
        readout.set_text(f"T = {t:.2f}    entropy = {h:.2f} bits    {sharp}")
        return bars

    anim = FuncAnimation(fig, update, frames=order, interval=1000 / FPS, blit=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "dec_temperature_softmax.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_temperature_gif()
