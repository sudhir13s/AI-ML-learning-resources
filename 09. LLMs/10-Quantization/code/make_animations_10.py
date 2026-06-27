"""Animated (GIF) figure generator for 10-Quantization.

Companion to make_figures.py. Animates the intuition figure (affine_number_line): the same real
values, the same int8 grid and scale, but now each fp value *drops and snaps* onto its nearest grid
point one at a time -- so the reader watches the round-to-grid mechanism happen and sees the
quantization error (the gap) appear with each snap. The final frame is the static figure.

    python make_animations_10.py

Writes ../../images/affine_number_line.gif via Pillow (per-frame durations). The values, symmetric
scale s = max|x|/127, grid, and palette match make_figures.py exactly.

Verified on Python 3.12 / matplotlib 3.x / Pillow.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from make_figures import AMBER, BLUE, GRID, RED, SLATE

# 10-Quantization keeps its figures in a CHAPTER-LOCAL images/ dir (unlike the shared 09. LLMs/images/).
OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 95
INT8_QMAX = 127

X = np.array([-1.50, -0.30, 0.0, 0.42, 0.95, 2.10])
SCALE = np.abs(X).max() / INT8_QMAX
Q = np.round(X / SCALE).astype(int)
X_HAT = Q * SCALE


def _render(snapped: int) -> Image.Image:
    """Show all real values (blue); the first `snapped` of them dropped onto the grid (red)."""
    fig, ax = plt.subplots(figsize=(9.0, 2.9))
    ax.axhline(0, color=SLATE, linewidth=1.2, zorder=1)
    grid_vals = np.arange(-INT8_QMAX, INT8_QMAX + 1) * SCALE
    grid_vals = grid_vals[(grid_vals >= -1.7) & (grid_vals <= 2.3)]
    for g in grid_vals:
        ax.axvline(g, color=GRID, linewidth=0.6, zorder=0)

    ax.scatter(X, np.zeros_like(X) + 0.18, color=BLUE, s=70, zorder=3, label="real value  x")
    # Snapped values: red square on the grid + amber arrow + q code.
    if snapped > 0:
        ax.scatter(X_HAT[:snapped], np.zeros(snapped) - 0.18, color=RED, s=70, marker="s",
                   zorder=3, label="dequantized  x̂ = s·q")
        for xi, xh, qi in zip(X[:snapped], X_HAT[:snapped], Q[:snapped]):
            ax.annotate("", xy=(xh, -0.16), xytext=(xi, 0.16),
                        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.1))
            ax.text(xh, -0.42, f"q={qi}", ha="center", va="top", color=SLATE, fontsize=8)
    else:
        # keep the legend stable from the first frame
        ax.scatter([], [], color=RED, s=70, marker="s", label="dequantized  x̂ = s·q")

    ax.set_xlim(-1.8, 2.4)
    ax.set_ylim(-0.7, 0.6)
    ax.set_yticks([])
    ax.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SLATE)
    ax.tick_params(colors=SLATE)
    ax.set_title(
        f"Affine int8 (symmetric): s = max|x|/127 = {SCALE:.5f}, z = 0\n"
        f"each x rounds to the nearest grid point q·s; the gap is the quantization error",
        color=SLATE, fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.78, bottom=0.06, left=0.04, right=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_number_line_gif() -> None:
    schedule = [(k, 850) for k in range(len(X) + 1)]  # 0..6 values snapped
    schedule[0] = (0, 1100)       # linger on the bare real values first
    schedule[-1] = (len(X), 2600)  # hold the full static figure
    frames = [_render(k) for (k, _d) in schedule]
    durations = [d for (_k, d) in schedule]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "affine_number_line.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=2, optimize=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_number_line_gif()
