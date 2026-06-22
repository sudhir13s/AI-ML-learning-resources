"""CNN concept-page diagrams (muted palette, parallel matplotlib scale).

Five figures for 05. Deep_Learning/concepts/13-CNNs-and-Convolution.md:
  1. conv_op.png        -- the convolution mechanic: a 3x3 kernel over one patch
     of a 5x5 input, element-wise multiply-and-sum -> one cell of the feature map.
  2. cnn_sobel.png      -- a real 3x3 Sobel edge kernel applied to a sample image
     (matplotlib's grace_hopper.jpg), showing what one learned-style filter "sees".
     MEASURED: the convolution is computed with scipy.signal.
  3. cnn_receptive_field.png -- how the receptive field grows across stacked 3x3
     conv layers (1 -> 3 -> 5 -> 7 -> ...), derived schematic with the formula.
  4. cnn_tensor_flow.png -- the 4D tensor-shape flow [N,C,H,W] through a small CNN
     (conv -> pool -> conv -> pool -> flatten -> FC), schematic.
  5. cnn_pool.png       -- 2x2 max-vs-avg pooling downsampling a 4x4 map to 2x2.

Run with the project's Python 3.12 env (torch + scipy + matplotlib available).
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.patches import Rectangle
import numpy as np
from scipy.signal import convolve2d

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


def cnn_sobel():
    """MEASURED: apply real 3x3 Sobel kernels to a sample image."""
    img = plt.imread(cbook.get_sample_data("grace_hopper.jpg"))
    gray = img[..., :3] @ np.array([0.299, 0.587, 0.114])      # luminance
    gray = gray / 255.0
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float)  # vertical edges
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float)  # horizontal edges
    gx = convolve2d(gray, Kx, mode="same", boundary="symm")
    gy = convolve2d(gray, Ky, mode="same", boundary="symm")
    mag = np.hypot(gx, gy)
    mag = mag / mag.max()

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.4))
    panels = [
        (gray, "input image (grayscale)", "gray"),
        (np.abs(gx), "vertical-edge response |Gx|\n(Sobel Kx)", "gray"),
        (np.abs(gy), "horizontal-edge response |Gy|\n(Sobel Ky)", "gray"),
        (mag, "edge magnitude √(Gx²+Gy²)\n— what the filter 'sees'", "magma"),
    ]
    border = [SLATE, BLUE, GREEN, AMBER]
    for ax, (im, t, cm), bc in zip(axes, panels, border):
        ax.imshow(im, cmap=cm)
        ax.set_title(t, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(bc); s.set_linewidth(3)
    fig.suptitle("A 3×3 Sobel edge filter, applied for real: the kernel responds where intensity changes",
                 fontsize=13.5, fontweight="bold", y=1.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/cnn_sobel.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cnn_sobel.png  (max edge mag =", f"{mag.max():.3f})")


def cnn_receptive_field():
    """Schematic: receptive field grows linearly with stacked 3x3 (stride 1) convs."""
    # RF after L layers of k=3, s=1:  RF = 1 + L*(k-1) = 1 + 2L
    layers = [0, 1, 2, 3, 4]
    rf = [1 + 2 * L for L in layers]                  # 1, 3, 5, 7, 9
    colors = [SLATE, BLUE, NAVY, PURPLE, GREEN]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 4.8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # LEFT: nested receptive-field squares on the input grid
    N = 9
    axL.add_patch(Rectangle((0, 0), N, N, facecolor="#eef1f4", edgecolor="#cfd6dd", lw=1))
    for i in range(N + 1):
        axL.plot([i, i], [0, N], color="#dfe4ea", lw=0.6)
        axL.plot([0, N], [i, i], color="#dfe4ea", lw=0.6)
    cx = N / 2
    for L, r, col in zip(layers[::-1], rf[::-1], colors[::-1]):
        half = r / 2
        axL.add_patch(Rectangle((cx - half, cx - half), r, r, facecolor="none",
                                edgecolor=col, lw=3.0))
    axL.plot(cx, cx, "o", color=RED, ms=9)
    axL.text(cx, -0.7, "one deep neuron sees a 9×9 patch of the input after 4 conv layers",
             ha="center", fontsize=10.5, color=SLATE)
    axL.set_xlim(-0.5, N + 0.5); axL.set_ylim(-1.4, N + 0.6)
    axL.set_aspect("equal"); axL.axis("off")
    axL.set_title("Receptive field grows with depth (nested on the input)", fontsize=12, fontweight="bold")

    # RIGHT: RF vs depth line plot
    axR.plot(layers, rf, "-o", color=PURPLE, lw=2.6, ms=8)
    for L, r in zip(layers, rf):
        axR.annotate(f"{r}×{r}", (L, r), textcoords="offset points", xytext=(6, 8),
                     fontsize=10.5, fontweight="bold", color=PURPLE)
    axR.set_xlabel("number of stacked 3×3 conv layers (L)", fontsize=11)
    axR.set_ylabel("receptive-field width", fontsize=11)
    axR.set_title("RF = 1 + L·(k−1) = 1 + 2L   (k=3, stride 1)", fontsize=11.5, fontweight="bold")
    axR.set_xticks(layers); axR.grid(alpha=0.25)
    axR.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Stack small filters and the field a neuron 'sees' expands — edges → parts → objects",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/cnn_receptive_field.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cnn_receptive_field.png")


def cnn_tensor_flow():
    """Schematic: 4D tensor shapes flowing through a small CNN on a 32x32x3 input."""
    # (label, C, H, W, color, kind)
    stages = [
        ("input\n3×32×32", 3, 32, 32, BLUE, "vol"),
        ("conv 3→16\n3×3, pad 1", 16, 32, 32, PURPLE, "vol"),
        ("max-pool\n2×2", 16, 16, 16, NAVY, "vol"),
        ("conv 16→32\n3×3, pad 1", 32, 16, 16, PURPLE, "vol"),
        ("max-pool\n2×2", 32, 8, 8, NAVY, "vol"),
        ("flatten\n2048", 0, 0, 0, AMBER, "flat"),
        ("FC → 10\nclass scores", 0, 0, 0, GREEN, "flat"),
    ]
    fig, ax = plt.subplots(figsize=(14.5, 4.6))
    x = 0.0
    centers = []
    for (label, C, H, W, col, kind) in stages:
        if kind == "vol":
            face = 0.9 + 1.7 * (H / 32)            # spatial -> face size
            depth = 0.35 + 0.9 * (np.log2(max(C, 1)) / 5)
            y0 = 2.4 - face / 2
            # back face (channel depth)
            ax.add_patch(Rectangle((x + depth, y0 + depth), face, face,
                                   facecolor=col, edgecolor="white", lw=1.4, alpha=0.45))
            # connecting top/side
            ax.fill([x, x + depth, x + depth + face, x + face],
                    [y0 + face, y0 + face + depth, y0 + face + depth, y0 + face],
                    color=col, alpha=0.32, edgecolor="white", lw=1)
            ax.fill([x + face, x + depth + face, x + depth + face, x + face],
                    [y0, y0 + depth, y0 + face + depth, y0 + face],
                    color=col, alpha=0.32, edgecolor="white", lw=1)
            # front face
            ax.add_patch(Rectangle((x, y0), face, face, facecolor=col,
                                   edgecolor="white", lw=1.6, alpha=0.9))
            ax.text(x + face / 2, y0 - 0.55, label, ha="center", va="top",
                    fontsize=10, fontweight="bold", color="#1d2733")
            centers.append((x + face / 2 + depth / 2, 2.4, face + depth))
            x += face + depth + 1.15
        else:
            w, h = 0.55, 2.1
            ax.add_patch(Rectangle((x, 2.4 - h / 2), w, h, facecolor=col,
                                   edgecolor="white", lw=1.6, alpha=0.9))
            ax.text(x + w / 2, 2.4 - h / 2 - 0.55, label, ha="center", va="top",
                    fontsize=10, fontweight="bold", color="#1d2733")
            centers.append((x + w / 2, 2.4, w))
            x += w + 1.15
    # arrows between stages
    for i in range(len(centers) - 1):
        x0 = centers[i][0] + centers[i][2] / 2 + 0.05
        x1 = centers[i + 1][0] - centers[i + 1][2] / 2 - 0.05
        ax.annotate("", xy=(x1, 2.4), xytext=(x0, 2.4),
                    arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.8))
    ax.text(x / 2, 4.62, "Tensor shape [N, C, H, W]: conv adds channels, pool shrinks H×W, flatten → FC",
            ha="center", fontsize=10.5, color=SLATE, style="italic")
    ax.set_xlim(-0.5, x); ax.set_ylim(0.2, 5.0); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("A small CNN's tensor-shape flow: spatial dims fall, channel depth rises",
                 fontsize=13.5, fontweight="bold", y=1.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/cnn_tensor_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cnn_tensor_flow.png")


def cnn_pool():
    M = np.array([[1, 3, 2, 4], [5, 6, 1, 2], [7, 2, 9, 1], [3, 4, 0, 8]])
    cols_map = {(0, 0): BLUE, (0, 1): GREEN, (1, 0): AMBER, (1, 1): PURPLE}
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    cell = 1.0; x0, y0 = 0, 4
    for r in range(4):
        for c in range(4):
            reg = (r // 2, c // 2)
            ax.add_patch(Rectangle((x0 + c * cell, y0 - r * cell), cell * 0.96, cell * 0.96,
                                   facecolor=cols_map[reg], edgecolor="white", lw=1.6, alpha=0.55))
            ax.text(x0 + c * cell + cell / 2, y0 - r * cell + cell / 2, f"{M[r,c]}",
                    ha="center", va="center", fontsize=12, fontweight="bold", color="#222")
    ax.text(x0 + 2, y0 + 1.05, "input map (4×4)", ha="center", fontsize=11.5, fontweight="bold")
    mx = np.array([[M[0:2, 0:2].max(), M[0:2, 2:4].max()], [M[2:4, 0:2].max(), M[2:4, 2:4].max()]])
    av = np.array([[M[0:2, 0:2].mean(), M[0:2, 2:4].mean()], [M[2:4, 0:2].mean(), M[2:4, 2:4].mean()]])
    # max-pool output
    xo = 6.2
    for r in range(2):
        for c in range(2):
            ax.add_patch(Rectangle((xo + c * cell, y0 - r * cell - 0.5), cell * 0.96, cell * 0.96,
                                   facecolor=cols_map[(r, c)], edgecolor="white", lw=1.6, alpha=0.92))
            ax.text(xo + c * cell + cell / 2, y0 - r * cell - 0.5 + cell / 2, f"{mx[r,c]:g}",
                    ha="center", va="center", fontsize=13, fontweight="bold", color="#fff")
    ax.text(xo + 1, y0 + 1.05, "2×2 MAX-pool", ha="center", fontsize=11.5, fontweight="bold")
    # avg-pool output
    xa = 9.4
    for r in range(2):
        for c in range(2):
            ax.add_patch(Rectangle((xa + c * cell, y0 - r * cell - 0.5), cell * 0.96, cell * 0.96,
                                   facecolor=cols_map[(r, c)], edgecolor="white", lw=1.6, alpha=0.92))
            ax.text(xa + c * cell + cell / 2, y0 - r * cell - 0.5 + cell / 2, f"{av[r,c]:.2g}",
                    ha="center", va="center", fontsize=12, fontweight="bold", color="#fff")
    ax.text(xa + 1, y0 + 1.05, "2×2 AVG-pool", ha="center", fontsize=11.5, fontweight="bold")
    ax.annotate("", xy=(6.0, 3.0), xytext=(4.3, 3.0), arrowprops=dict(arrowstyle="->", color=SLATE, lw=2))
    ax.annotate("", xy=(9.2, 3.0), xytext=(8.7, 3.0), arrowprops=dict(arrowstyle="->", color=SLATE, lw=2))
    ax.set_xlim(-0.4, 12); ax.set_ylim(0.6, 5.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Pooling downsamples 2× — max keeps the strongest activation, avg keeps the mean",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/cnn_pool.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cnn_pool.png")


if __name__ == "__main__":
    conv_op()
    cnn_sobel()
    cnn_receptive_field()
    cnn_tensor_flow()
    cnn_pool()
    print("OUT:", OUT)
