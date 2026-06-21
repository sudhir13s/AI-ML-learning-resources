"""CNN concept-page diagrams (muted palette, parallel matplotlib scale).

Two figures for 05. Deep_Learning/concepts/13-CNNs-and-Convolution.md:
  1. conv_op.png  -- the convolution mechanic: a 3x3 kernel over one patch of a
     5x5 input, element-wise multiply-and-sum -> one cell of the feature map.
  2. cnn_pool.png -- 2x2 max-pooling downsampling a 4x4 map to 2x2 (each colored
     region -> its max).
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "05. Deep_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _grid(ax, M, x0, y0, cell=1.0, highlight=None, hcolor=AMBER, base="#eef1f4",
          fontcolor="#222", title=None):
    rows, cols = M.shape
    for r in range(rows):
        for c in range(cols):
            hot = highlight is not None and (r, c) in highlight
            ax.add_patch(Rectangle((x0 + c * cell, y0 - r * cell), cell * 0.96, cell * 0.96,
                                   facecolor=hcolor if hot else base,
                                   edgecolor="white", lw=1.5, alpha=0.9 if hot else 1))
            ax.text(x0 + c * cell + cell / 2, y0 - r * cell + cell / 2, f"{M[r,c]:g}",
                    ha="center", va="center", fontsize=11,
                    color="#fff" if hot else fontcolor, fontweight="bold" if hot else "normal")
    if title:
        ax.text(x0 + cols * cell / 2, y0 + cell * 1.0, title, ha="center", fontsize=11.5, fontweight="bold")


def conv_op():
    rng = np.random.default_rng(2)
    inp = rng.integers(0, 3, (5, 5))
    ker = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])         # a vertical-edge filter
    patch = inp[0:3, 0:3]
    val = int((patch * ker).sum())
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    _grid(ax, inp, 0, 5, highlight={(r, c) for r in range(3) for c in range(3)},
          hcolor=BLUE, title="input (5×5)")
    _grid(ax, ker, 6.4, 4, base="#efeaf6", fontcolor=PURPLE, title="kernel (3×3)\nvertical-edge filter")
    out = np.full((3, 3), np.nan)
    # draw output grid with only the top-left cell filled
    rows, cols = 3, 3
    x0, y0, cell = 10.6, 5, 1.0
    for r in range(rows):
        for c in range(cols):
            filled = (r == 0 and c == 0)
            ax.add_patch(Rectangle((x0 + c * cell, y0 - r * cell), cell * 0.96, cell * 0.96,
                                   facecolor=GREEN if filled else "#eef1f4", edgecolor="white", lw=1.5,
                                   alpha=0.9 if filled else 1))
            if filled:
                ax.text(x0 + c * cell + cell / 2, y0 - r * cell + cell / 2, f"{val}",
                        ha="center", va="center", fontsize=12, color="#fff", fontweight="bold")
    ax.text(x0 + cols * cell / 2, y0 + cell * 1.0, "feature map (3×3)", ha="center", fontsize=11.5, fontweight="bold")
    ax.annotate("", xy=(10.4, 4.5), xytext=(9.5, 4.5), arrowprops=dict(arrowstyle="->", color=SLATE, lw=2))
    ax.text(6.7, 0.15, f"each output cell = Σ (overlapping input patch × kernel)  =  {val}  here  —  then slide by the stride and repeat",
            ha="center", fontsize=10.5, color=SLATE)
    ax.set_xlim(-0.4, 14); ax.set_ylim(-0.4, 6.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Convolution: slide a small kernel, multiply-and-sum each patch",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/conv_op.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote conv_op.png")


def cnn_pool():
    M = np.array([[1, 3, 2, 4], [5, 6, 1, 2], [7, 2, 9, 1], [3, 4, 0, 8]])
    cols_map = {(0, 0): BLUE, (0, 1): GREEN, (1, 0): AMBER, (1, 1): PURPLE}
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    cell = 1.0; x0, y0 = 0, 4
    for r in range(4):
        for c in range(4):
            reg = (r // 2, c // 2)
            ax.add_patch(Rectangle((x0 + c * cell, y0 - r * cell), cell * 0.96, cell * 0.96,
                                   facecolor=cols_map[reg], edgecolor="white", lw=1.6, alpha=0.55))
            ax.text(x0 + c * cell + cell / 2, y0 - r * cell + cell / 2, f"{M[r,c]}",
                    ha="center", va="center", fontsize=12, fontweight="bold", color="#222")
    ax.text(x0 + 2, y0 + 1.05, "input map (4×4)", ha="center", fontsize=11.5, fontweight="bold")
    # output 2x2 = max of each colored region
    out = np.array([[M[0:2, 0:2].max(), M[0:2, 2:4].max()], [M[2:4, 0:2].max(), M[2:4, 2:4].max()]])
    xo = 6.4
    for r in range(2):
        for c in range(2):
            reg = (r, c)
            ax.add_patch(Rectangle((xo + c * cell, y0 - r * cell - 0.5), cell * 0.96, cell * 0.96,
                                   facecolor=cols_map[reg], edgecolor="white", lw=1.6, alpha=0.9))
            ax.text(xo + c * cell + cell / 2, y0 - r * cell - 0.5 + cell / 2, f"{out[r,c]}",
                    ha="center", va="center", fontsize=13, fontweight="bold", color="#fff")
    ax.text(xo + 1, y0 + 1.05, "2×2 max-pool → (2×2)", ha="center", fontsize=11.5, fontweight="bold")
    ax.annotate("", xy=(6.2, 3.0), xytext=(4.3, 3.0), arrowprops=dict(arrowstyle="->", color=SLATE, lw=2))
    ax.text(5.25, 3.35, "max", ha="center", fontsize=10, color=SLATE, fontweight="bold")
    ax.set_xlim(-0.4, 9); ax.set_ylim(0.6, 5.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Max-pooling: keep the strongest activation per region (downsample + invariance)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/cnn_pool.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cnn_pool.png")


if __name__ == "__main__":
    conv_op()
    cnn_pool()
    print("OUT:", OUT)
