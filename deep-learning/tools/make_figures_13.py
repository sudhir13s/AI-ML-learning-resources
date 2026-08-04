"""Figure generator for 13-CNNs-and-Convolution — every quantitative figure from the REAL runs in ``cnn.py``.

The measured figures (a real Sobel filter on a real photo; the trained CNN's learned first-layer filters and
feature maps; the CNN-vs-MLP parameter/accuracy/shift-robustness bars; receptive-field growth) all come from
the same executed pipeline the chapter and notebook use — nothing quantitative is hand-typed. The remaining
figures are labelled schematics (one convolution value, the 4-D tensor-shape flow, pooling) drawn with the
worked-example's real numbers where numbers appear.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``dl13_``:

  dl13_conv_op.png         -- one convolution: the 3x3 vertical-edge kernel over the top-left patch of the 5x5
                              input, multiply-and-sum -> the single value 4 (worked_hand_conv, real numbers).
  dl13_conv_in_action.png  -- a real 3x3 Sobel edge filter applied with our own conv2d to a real photograph:
                              input, |Gx|, |Gy|, and the edge magnitude; our-conv == scipy to ~1e-13 (measured).
  dl13_tensor_flow.png     -- the 4-D shape flow through a small CNN (channels rise, spatial size falls);
                              illustrative schematic.
  dl13_pool.png            -- 2x2 max- vs average-pooling on a 4x4 map, halving resolution (real numbers).
  dl13_receptive_field.png -- RF = 1 + L(K-1) growth with depth, and the multiplicative speed-up a stride-2
                              pool adds (receptive_field_growth, exact integers).
  dl13_learned_filters.png -- the trained CNN's 8 learned first-layer 3x3 kernels + their feature maps on a
                              real test digit (train_cnn_vs_mlp, measured).
  dl13_cnn_vs_mlp.png      -- parameters, clean accuracy, and 1-pixel-shift accuracy for the trained CNN vs a
                              larger MLP: same clean accuracy at ~5x fewer params, and far more shift-robust.

    python make_figures_13.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / scipy 1.17 / torch 2.12 / scikit-learn 1.9.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in ``05. Deep_Learning/tools/``; the chapter module it demonstrates stays in that
# chapter's ``code/`` folder. Put that folder on sys.path so the ``cnn`` import resolves regardless of cwd.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "13-CNNs-and-Convolution" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from cnn import (  # noqa: E402  (resolved via the sys.path insert above)
    load_sample_gray,
    receptive_field_growth,
    sobel_edges,
    train_cnn_vs_mlp,
    worked_hand_conv,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # data / values forward
PURPLE = "#5D4A8A"  # process / conv
GREEN = "#2E7A5A"  # output / good
RED = "#8B3B4A"  # error / emphasis
AMBER = "#7A6528"  # highlight / kernel
SLATE = "#4A5B6E"  # neutral / pooling
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "dl13_"


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def _box(ax: plt.Axes, xy: tuple[float, float], text: str, colour: str, w: float = 1.5, h: float = 0.7, fs: float = 9.5) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=colour, edgecolor="none", zorder=2,
        )
    )
    ax.text(x, y, text, ha="center", va="center", color="white", fontsize=fs, zorder=3)


def _arrow(ax: plt.Axes, p0: tuple[float, float], p1: tuple[float, float], colour: str, dashed: bool = False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=13, linewidth=1.6, color=colour,
            linestyle="--" if dashed else "-", shrinkA=4, shrinkB=4, zorder=1,
        )
    )


def _grid_numbers(ax: plt.Axes, m: np.ndarray, x0: float, y0: float, cell: float, colour: str, fs: float = 11, highlight: tuple[int, int, int, int] | None = None) -> None:
    """Draw a matrix of numbers as a labelled grid; optionally highlight a (r0,c0,r1,c1) sub-block."""
    rows, cols = m.shape
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x0 + c * cell, y0 - (r + 1) * cell), cell, cell, facecolor="white", edgecolor=GRID, lw=1))
            ax.text(x0 + (c + 0.5) * cell, y0 - (r + 0.5) * cell, f"{m[r, c]:g}", ha="center", va="center", color=colour, fontsize=fs)
    if highlight is not None:
        r0, c0, r1, c1 = highlight
        ax.add_patch(Rectangle((x0 + c0 * cell, y0 - (r1 + 1) * cell), (c1 - c0 + 1) * cell, (r1 - r0 + 1) * cell,
                               fill=False, edgecolor=AMBER, lw=2.6, zorder=4))


# ============================ 1. one convolution value (schematic, real numbers) =================
def fig_conv_op() -> None:
    hc = worked_hand_conv()
    x = np.array([[2, 0, 0, 1, 3], [2, 1, 0, 2, 2], [2, 2, 2, 1, 0], [1, 2, 3, 2, 1], [0, 1, 2, 3, 2]], dtype=float)
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    _grid_numbers(ax, x, 0.3, 6.5, 0.9, BLUE, highlight=(0, 0, 2, 2))
    ax.text(0.3 + 2.5 * 0.9, 6.85, "5×5 input  (amber = current 3×3 patch)", color=INK, fontsize=10, ha="center")
    _grid_numbers(ax, hc.kernel, 5.3, 5.6, 0.9, AMBER)
    ax.text(5.3 + 1.5 * 0.9, 6.0, "3×3 kernel\n(vertical edge)", color=INK, fontsize=9.5, ha="center")
    _arrow(ax, (8.1, 4.7), (9.2, 4.7), PURPLE)
    ax.text(8.65, 5.1, "⊙ then Σ", color=PURPLE, fontsize=10, ha="center")
    _grid_numbers(ax, hc.feature_map, 9.4, 6.5, 0.9, GREEN, highlight=(0, 0, 0, 0))
    ax.text(9.4 + 1.5 * 0.9, 6.85, "3×3 feature map", color=INK, fontsize=10, ha="center")
    ax.text(6.5, 1.35, f"top-left cell = (2·1+0·0+0·−1) + (2·1+1·0+0·−1) + (2·1+2·0+2·−1) = 2 + 2 + 0 = {hc.top_left:.0f}",
            color=INK, fontsize=11, ha="center")
    ax.text(6.5, 0.55, "one output value = one dot product between the flattened patch and the flattened kernel",
            color=SLATE, fontsize=9.5, ha="center", style="italic")
    fig.suptitle("One convolution: slide a small kernel, multiply-and-sum each patch into one number", color=INK, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, f"{IMG_PREFIX}conv_op.png")


# ============================ 2. a real Sobel filter on a real photo (measured) ==================
def fig_conv_in_action() -> None:
    sob = sobel_edges(load_sample_gray())
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.4))
    panels = [
        (sob.gray, "input (real grayscale photo)", "gray"),
        (sob.gx, "|Gx|  vertical-edge response", "magma"),
        (sob.gy, "|Gy|  horizontal-edge response", "magma"),
        (sob.magnitude, "edge magnitude √(Gx²+Gy²)", "magma"),
    ]
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color=INK, fontsize=10.5)
        ax.axis("off")
    fig.suptitle(
        f"A real 3×3 Sobel edge filter, applied with our from-scratch conv2d — matches scipy to "
        f"{sob.ours_vs_scipy:.0e}. Flat regions vanish; intensity changes (edges) light up — exactly what a "
        f"CNN's first layer learns.",
        color=INK, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, f"{IMG_PREFIX}conv_in_action.png")


# ============================ 3. 4-D tensor-shape flow (schematic) ===============================
def fig_tensor_flow() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")
    stages = [
        ("input\n[3, 32, 32]", BLUE), ("conv 3→16\n[16, 32, 32]", PURPLE), ("pool ↓2\n[16, 16, 16]", SLATE),
        ("conv 16→32\n[32, 16, 16]", PURPLE), ("pool ↓2\n[32, 8, 8]", SLATE), ("flatten\n[2048]", AMBER),
        ("FC → 10\n[10]", GREEN),
    ]
    xs = np.linspace(1.4, 13.6, len(stages))
    for i, (label, colour) in enumerate(stages):
        _box(ax, (xs[i], 3.2), label, colour, w=1.75, h=1.15, fs=8.8)
        if i > 0:
            _arrow(ax, (xs[i - 1] + 0.9, 3.2), (xs[i] - 0.9, 3.2), INK)
    ax.text(7.5, 5.2, "Spatial size falls (32→16→8), channel depth rises (3→16→32): the net trades 'where' for 'what'",
            color=INK, fontsize=11, ha="center")
    ax.text(7.5, 1.2, "conv layers: tiny to store (448 + 4,640 params) · the single FC head holds 20,490 (80%) — why modern nets use global average pooling",
            color=SLATE, fontsize=9.3, ha="center", style="italic")
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}tensor_flow.png")


# ============================ 4. pooling (schematic, real numbers) ===============================
def fig_pool() -> None:
    x = np.array([[1, 3, 2, 4], [5, 6, 1, 2], [3, 1, 7, 0], [2, 0, 4, 9]], dtype=float)
    mx = np.array([[max(x[0:2, 0:2].max(), 0), x[0:2, 2:4].max()], [x[2:4, 0:2].max(), x[2:4, 2:4].max()]])
    av = np.array([[x[0:2, 0:2].mean(), x[0:2, 2:4].mean()], [x[2:4, 0:2].mean(), x[2:4, 2:4].mean()]])
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    _grid_numbers(ax, x, 0.4, 4.9, 0.85, BLUE)
    # colour the four 2x2 windows
    for (r0, c0), col in zip([(0, 0), (0, 2), (2, 0), (2, 2)], [AMBER, GREEN, RED, PURPLE]):
        ax.add_patch(Rectangle((0.4 + c0 * 0.85, 4.9 - (r0 + 2) * 0.85), 2 * 0.85, 2 * 0.85, fill=False, edgecolor=col, lw=2.4))
    ax.text(0.4 + 2 * 0.85, 5.25, "4×4 feature map (2×2 windows)", color=INK, fontsize=10, ha="center")
    _grid_numbers(ax, mx, 6.6, 4.5, 0.95, GREEN)
    ax.text(6.6 + 0.95, 4.85, "max-pool", color=INK, fontsize=10, ha="center")
    _grid_numbers(ax, av, 9.9, 4.5, 0.95, SLATE)
    ax.text(9.9 + 0.95, 4.85, "avg-pool", color=INK, fontsize=10, ha="center")
    _arrow(ax, (4.2, 3.3), (6.4, 3.9), GREEN)
    _arrow(ax, (4.2, 3.3), (9.7, 3.9), SLATE)
    ax.text(6.5, 1.0, "Pooling halves the resolution (4×4 → 2×2), has no parameters, and adds a little "
            "translation invariance:\nif a feature jitters within a window, the max is unchanged.",
            color=INK, fontsize=10, ha="center")
    fig.suptitle("Pooling: summarize each region into one number (max keeps the strongest, avg the mean)", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, f"{IMG_PREFIX}pool.png")


# ============================ 5. receptive-field growth (measured integers) ======================
def fig_receptive_field() -> None:
    rf = receptive_field_growth(k=3, depth=6)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    _style_axis(ax)
    ax.plot(rf.layers, rf.rf_stride1, "o-", color=PURPLE, lw=2, label="stacked 3×3, stride 1:  RF = 1 + L·(K−1) = 1 + 2L")
    for ell, r in zip(rf.layers, rf.rf_stride1):
        ax.annotate(f"{r}×{r}", (ell, r), textcoords="offset points", xytext=(0, 8), color=INK, fontsize=9, ha="center")
    # mark the two-3x3 = one-5x5 insight
    ax.annotate("two 3×3 convs = one 5×5 RF\n(18 vs 25 weights + an extra ReLU) — the VGG insight",
                xy=(2, 5), xytext=(0.55, 9.7), color=RED, fontsize=9, ha="left",
                arrowprops={"arrowstyle": "->", "color": RED})
    # the stride-2 trace as a second series
    names = [t[0] for t in rf.rf_with_pool]
    rfs = [t[2] for t in rf.rf_with_pool]
    ax.text(0.6, 12.2, "with a stride-2 pool inserted: " + " → ".join(f"{n.split(' ')[0]} RF {r}" for n, r in zip(names, rfs))
            + "  (a conv after the pool now adds 4, not 2)", color=SLATE, fontsize=8.6)
    ax.set_xlabel("number of stacked 3×3 conv layers (L)")
    ax.set_ylabel("receptive field (pixels of the input a neuron sees)")
    ax.set_ylim(0, 13.5)
    ax.set_title("Receptive field grows linearly with depth — and multiplicatively with stride/pooling", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}receptive_field.png")


# ============================ 6 & 7. learned filters + CNN-vs-MLP (measured) =====================
def fig_learned_and_comparison() -> None:
    tr = train_cnn_vs_mlp()

    # ---- learned first-layer filters + feature maps on a real digit ----
    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(2, 9, width_ratios=[1.6] + [1] * 8, wspace=0.25, hspace=0.35)
    ax_in = fig.add_subplot(gs[:, 0])
    ax_in.imshow(tr.sample_digit, cmap="gray")
    ax_in.set_title("input digit", color=INK, fontsize=10)
    ax_in.axis("off")
    for i in range(8):
        axf = fig.add_subplot(gs[0, i + 1])
        axf.imshow(tr.first_filters[i], cmap="RdBu_r")
        axf.set_title(f"k{i}", color=INK, fontsize=8.5)
        axf.axis("off")
        axm = fig.add_subplot(gs[1, i + 1])
        axm.imshow(tr.feature_maps[i], cmap="magma")
        axm.axis("off")
    fig.text(0.5, 0.95, "Learned first-layer 3×3 filters (top row) and their feature maps on a real test digit (bottom row)",
             color=INK, fontsize=11.5, ha="center")
    fig.text(0.5, 0.04, f"each learned kernel fires on a different local pattern (edges / strokes) — trained CNN, "
             f"{tr.cnn_acc * 100:.1f}% test accuracy", color=SLATE, fontsize=9.5, style="italic", ha="center")
    _save(fig, f"{IMG_PREFIX}learned_filters.png")

    # ---- CNN vs MLP: params, clean accuracy, shift accuracy ----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    labels = ["CNN", "MLP"]
    colours = [PURPLE, SLATE]

    _style_axis(axes[0])
    axes[0].bar(labels, [tr.cnn_params, tr.mlp_params], color=colours)
    axes[0].set_ylabel("trainable parameters")
    axes[0].set_title(f"Parameters — CNN is {tr.mlp_params / tr.cnn_params:.1f}× smaller", fontsize=10.5)
    for i, v in enumerate([tr.cnn_params, tr.mlp_params]):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", color=INK, fontsize=9.5)

    _style_axis(axes[1])
    axes[1].bar(labels, [tr.cnn_acc * 100, tr.mlp_acc * 100], color=colours)
    axes[1].set_ylabel("test accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Clean digits — matched accuracy", fontsize=10.5)
    for i, v in enumerate([tr.cnn_acc * 100, tr.mlp_acc * 100]):
        axes[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom", color=INK, fontsize=9.5)

    _style_axis(axes[2])
    axes[2].bar(labels, [tr.cnn_acc_shift * 100, tr.mlp_acc_shift * 100], color=[GREEN, RED])
    axes[2].set_ylabel("accuracy on 1-px-shifted digits (%)")
    axes[2].set_ylim(0, 105)
    axes[2].set_title("Shifted digits — CNN holds up (equivariance)", fontsize=10.5)
    for i, v in enumerate([tr.cnn_acc_shift * 100, tr.mlp_acc_shift * 100]):
        axes[2].text(i, v, f"{v:.1f}%", ha="center", va="bottom", color=INK, fontsize=9.5)

    fig.suptitle(
        f"CNN vs a larger MLP on scikit-learn digits: same clean accuracy at {tr.mlp_params / tr.cnn_params:.1f}× "
        f"fewer weights, and far more robust to translation ({tr.cnn_acc_shift * 100:.0f}% vs "
        f"{tr.mlp_acc_shift * 100:.0f}% shifted) — the weight-sharing win, measured",
        color=INK, fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, f"{IMG_PREFIX}cnn_vs_mlp.png")


def main() -> None:
    fig_conv_op()
    fig_conv_in_action()
    fig_tensor_flow()
    fig_pool()
    fig_receptive_field()
    fig_learned_and_comparison()
    print(f"\nall figures written to {OUT_DIR} with prefix '{IMG_PREFIX}'")


if __name__ == "__main__":
    main()
