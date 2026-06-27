"""Reproducible figure generator for the Quantization chapter.

Renders every original PNG used on the page into ../images/, with a muted palette that matches
the chapter's Mermaid colors. The quantitative figures (outlier-error bars, error-vs-bits) call
the SAME seeded-torch routines as quantization.py — imported below — so the plotted numbers are
bit-identical in provenance to the code the page cites. Re-run after any numeric change:

    python make_figures.py

Verified on Python 3.12 / matplotlib 3.x, torch 2.x. CPU only, deterministic (seeded).
"""

from __future__ import annotations

import _pathsetup  # noqa: F401  (sys.path bootstrap for the moved generator)

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch

# Share ONE source of truth with the demo: import the seeded-torch quantizers and constants
# from quantization.py so the figures plot exactly what the code/notebook/page report.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantization as q  # noqa: E402  (path set above)

# Muted palette mirroring the chapter's Mermaid classDefs (all readable on white).
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
GRID = "#D8DCE0"

IMAGES_DIR = Path(__file__).resolve().parent.parent / "09. LLMs" / "10-Quantization" / "images"
DPI = 130
SEED = 0

# Quantization constants (kept identical to quantization.py).
INT8_QMAX = 127
INT4_QMAX = 7
GROUP_SIZE = 64
FP16_BYTES = 2.0


def _style_ax(ax: plt.Axes) -> None:
    """Apply the shared muted styling to an axis."""
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=SLATE)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(SLATE)


def fig_affine_number_line() -> str:
    """fp value -> int8 code mapping on a number line: scale s and the rounding step."""
    x = np.array([-1.50, -0.30, 0.0, 0.42, 0.95, 2.10])
    scale = np.abs(x).max() / INT8_QMAX          # symmetric scale, matches quantization.py
    q = np.round(x / scale).astype(int)

    fig, ax = plt.subplots(figsize=(9.0, 2.7))
    # the continuous real line
    ax.axhline(0, color=SLATE, linewidth=1.2, zorder=1)
    # quantization grid: faint ticks every `scale`
    grid_vals = np.arange(-INT8_QMAX, INT8_QMAX + 1) * scale
    grid_vals = grid_vals[(grid_vals >= -1.7) & (grid_vals <= 2.3)]
    for g in grid_vals:
        ax.axvline(g, color=GRID, linewidth=0.6, zorder=0)
    # plot the real values and their quantized reconstructions
    x_hat = q * scale
    ax.scatter(x, np.zeros_like(x) + 0.18, color=BLUE, s=70, zorder=3, label="real value  x")
    ax.scatter(x_hat, np.zeros_like(x_hat) - 0.18, color=RED, s=70, marker="s", zorder=3,
               label="dequantized  x̂ = s·q")
    for xi, xh, qi in zip(x, x_hat, q):
        ax.annotate("", xy=(xh, -0.16), xytext=(xi, 0.16),
                    arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.1))
        ax.text(xh, -0.42, f"q={qi}", ha="center", va="top", color=SLATE, fontsize=8)
    ax.set_xlim(-1.8, 2.4)
    ax.set_ylim(-0.7, 0.6)
    ax.set_yticks([])
    _style_ax(ax)
    ax.spines["left"].set_visible(False)
    ax.set_title(
        f"Affine int8 (symmetric): s = max|x|/127 = {scale:.5f}, z = 0\n"
        f"each x rounds to the nearest grid point q·s; the gap is the quantization error",
        color=SLATE, fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    out = IMAGES_DIR / "affine_number_line.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_weight_histogram_bins() -> str:
    """Histogram of a weight tensor with the int8 quantization grid overlaid."""
    rng = np.random.default_rng(SEED)
    w = rng.normal(0, 0.1, size=20000)
    scale = np.abs(w).max() / INT8_QMAX
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.hist(w, bins=120, color=BLUE, alpha=0.85, edgecolor="none")
    # overlay a sparse subset of the int8 grid lines so they're visible (full 255 would be a smear)
    grid = np.arange(-INT8_QMAX, INT8_QMAX + 1, 8) * scale
    for g in grid:
        ax.axvline(g, color=AMBER, linewidth=0.7, alpha=0.7)
    _style_ax(ax)
    ax.set_xlabel("weight value", color=SLATE)
    ax.set_ylabel("count", color=SLATE)
    ax.set_title(
        f"Weights ~ N(0, 0.1) snap onto the int8 grid (step s = {scale:.5f}; every 8th line shown)\n"
        "the dense centre is well resolved; few weights live in the tails",
        color=SLATE, fontsize=10)
    out = IMAGES_DIR / "weight_histogram_bins.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out.name


def _outlier_matrix() -> torch.Tensor:
    """Rebuild quantization.py §2's matrix EXACTLY: same seed, shape, and outlier column.

    Reproduces the §2 draw order bit-for-bit (seed 0 -> randn(8,128)*0.1, then column 17 set to
    randn(8)*10.0) so the figure plots the same numbers the code/notebook/page report.
    """
    torch.manual_seed(SEED)
    w = torch.randn(8, 128) * 0.1            # matches quantization.py §2: tame weights ~ N(0, 0.1)
    w[:, 17] = torch.randn(8) * 10.0         # the one ~100x-louder column, in every row
    return w


def fig_outlier_error_bars() -> str:
    """Reconstruction error: per-tensor vs per-channel vs per-group, under a column outlier.

    Uses the SAME seeded-torch quantizers as quantization.py §2, so the bars read the cited
    numbers (per-tensor 0.031840 / per-channel 0.016886 / per-group 0.002333).
    """
    w = _outlier_matrix()
    err = {
        "per-tensor\n(1 scale)": q.reconstruction_error(w, q.quantize_per_tensor_int8(w)),
        "per-channel\n(1 / row)": q.reconstruction_error(w, q.quantize_per_channel_int8(w)),
        "per-group\n(1 / 16 cols)": q.reconstruction_error(w, q.quantize_group_wise_int8(w, 16)),
    }
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    labels = list(err.keys())
    vals = list(err.values())
    bars = ax.bar(labels, vals, color=[RED, AMBER, GREEN], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom",
                color=SLATE, fontsize=9)
    _style_ax(ax)
    ax.set_ylabel("mean |x − x̂|  (lower = better)", color=SLATE)
    ax.set_title(
        "One 100× outlier COLUMN: finer granularity is the fix\n"
        "per-row can't help (every row holds the loud column); per-group isolates it",
        color=SLATE, fontsize=10)
    out = IMAGES_DIR / "outlier_error_bars.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_outlier_heatmap() -> str:
    """Heatmap of |W| showing a single high-magnitude column — the LLM outlier signature."""
    rng = np.random.default_rng(SEED)
    rows, cols = 16, 48
    w = np.abs(rng.normal(0, 0.1, size=(rows, cols)))
    w[:, 17] = np.abs(rng.normal(0, 10.0, size=rows))  # loud column
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    im = ax.imshow(w, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel("input channel (column)", color=SLATE)
    ax.set_ylabel("output channel (row)", color=SLATE)
    ax.tick_params(colors=SLATE)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("|weight|", color=SLATE)
    cbar.ax.tick_params(colors=SLATE)
    ax.annotate("outlier column 17", xy=(17, 4), xytext=(33, 7),
                color=RED, fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_title(
        "LLM outliers are STRUCTURED: a few fixed channels are ~100× louder than the rest\n"
        "one global (or even per-row) scale must stretch to cover them, crushing everything else",
        color=SLATE, fontsize=10, pad=12)
    out = IMAGES_DIR / "outlier_channel_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_int4_memory_bars() -> str:
    """Llama-2-70B memory across fp16 / int8 / int4(group) — the headline reduction."""
    b_int4 = 4 / 8 + FP16_BYTES / GROUP_SIZE  # 0.5 byte code + fp16 scale per group = 0.5312
    params = 70e9
    precisions = ["fp16", "int8", "int4\n(group-64)"]
    bpp = [FP16_BYTES, 1.0, b_int4]
    gb = [params * b / 1e9 for b in bpp]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    bars = ax.bar(precisions, gb, color=[SLATE, BLUE, GREEN], width=0.6)
    for b, g in zip(bars, gb):
        ax.text(b.get_x() + b.get_width() / 2, g, f"{g:.0f} GB", ha="center", va="bottom",
                color=SLATE, fontsize=10)
    ax.axhline(80, color=RED, linewidth=1.2, linestyle="--")
    ax.text(0.62, 84, "80 GB GPU", color=RED, fontsize=8, ha="center")
    ax.axhline(40, color=AMBER, linewidth=1.2, linestyle="--")
    ax.text(1.5, 44, "40 GB GPU", color=AMBER, fontsize=8, ha="center")
    ax.set_ylim(0, 152)
    _style_ax(ax)
    ax.set_ylabel("weight memory (GB)", color=SLATE)
    ax.set_title(
        "Llama-2-70B weights: int4 turns two 80GB GPUs into one 40GB GPU\n"
        "140 GB (fp16) → 70 GB (int8) → 37 GB (int4, +scale overhead)",
        color=SLATE, fontsize=10)
    out = IMAGES_DIR / "int4_memory_bars.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_error_vs_bits() -> str:
    """Reconstruction error vs bit-width for group-wise weight quantization.

    Uses the SAME seeded-torch weight block as quantization.py §3 (seed 0, randn(256,1024)*0.1,
    group=64), so the 4-bit point reads exactly the cited 0.009132.
    """
    torch.manual_seed(SEED)
    w = torch.randn(256, 1024) * 0.1          # matches quantization.py §3 exactly
    group = GROUP_SIZE
    rows, cols = w.shape
    bits_list = [8, 6, 5, 4, 3, 2]
    errs = []
    for b in bits_list:
        qmax = (1 << (b - 1)) - 1             # signed symmetric max for b bits
        wg = w.view(rows, cols // group, group)
        s = wg.abs().amax(dim=-1, keepdim=True) / qmax            # one scale per group
        s = torch.where(s == 0, torch.ones_like(s), s)
        w_hat = (torch.clamp(torch.round(wg / s), -qmax, qmax) * s).view(rows, cols)
        errs.append(q.reconstruction_error(w, w_hat))
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(bits_list, errs, "-o", color=PURPLE, linewidth=2, markersize=7)
    for b, e in zip(bits_list, errs):
        ax.text(b, e, f"  {e:.4f}", color=SLATE, fontsize=8, va="center")
    ax.invert_xaxis()  # fewer bits to the right = harder
    _style_ax(ax)
    ax.set_xlabel("bits per weight (group-wise, group=64)", color=SLATE)
    ax.set_ylabel("mean |x − x̂|", color=SLATE)
    ax.set_xticks(bits_list)
    ax.set_title(
        "Reconstruction error roughly halves per extra bit (error ≈ step ∝ 2⁻ᵇ)\n"
        "8→4 bits is nearly free on weights; below ~3 bits the error climbs fast",
        color=SLATE, fontsize=10)
    out = IMAGES_DIR / "error_vs_bits.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out.name


# The 16 NF4 levels from the QLoRA paper (Dettmers et al. 2023): quantiles of a standard normal,
# normalized to [-1, 1]. Denser near 0 (where Gaussian weights live), sparser in the tails.
NF4_LEVELS = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)


def fig_nf4_vs_int4_levels() -> str:
    """NF4 (quantile-spaced) vs evenly-spaced int4 levels over a Gaussian weight histogram."""
    rng = np.random.default_rng(SEED)
    w = rng.normal(0, 0.3, size=40000)        # a wider Gaussian so the level spread is visible
    absmax = np.abs(w).max()
    nf4 = np.array(NF4_LEVELS) * absmax        # NF4 levels scaled to the tensor's range
    int4 = np.linspace(-absmax, absmax, 16)    # 16 evenly-spaced int4 levels over the same range
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.hist(w, bins=120, color=BLUE, alpha=0.35, edgecolor="none", label="weights ~ N(0, 0.3)")
    ax.vlines(nf4, 0, ax.get_ylim()[1] * 0.62, color=GREEN, linewidth=1.6,
              label="NF4 levels (quantile-spaced)")
    ax.vlines(int4, ax.get_ylim()[1] * 0.66, ax.get_ylim()[1] * 0.98, color=AMBER, linewidth=1.6,
              label="int4 levels (evenly spaced)")
    _style_ax(ax)
    ax.set_xlabel("weight value", color=SLATE)
    ax.set_ylabel("count", color=SLATE)
    ax.set_title(
        "NF4 puts its 16 levels where the weights are: dense near 0, sparse in the tails\n"
        "evenly-spaced int4 wastes levels on the empty tails — NF4 is optimal for Gaussian data",
        color=SLATE, fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    out = IMAGES_DIR / "nf4_vs_int4_levels.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out.name


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    made = [
        fig_affine_number_line(),
        fig_weight_histogram_bins(),
        fig_outlier_heatmap(),
        fig_outlier_error_bars(),
        fig_int4_memory_bars(),
        fig_error_vs_bits(),
        fig_nf4_vs_int4_levels(),
    ]
    print(f"wrote {len(made)} figures to {IMAGES_DIR}:")
    for name in made:
        print("  -", name)


if __name__ == "__main__":
    main()
