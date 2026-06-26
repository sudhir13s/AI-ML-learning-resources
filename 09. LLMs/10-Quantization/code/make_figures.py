"""Reproducible figure generator for the Quantization chapter.

Renders every original PNG used on the page into ../images/, with a muted palette that matches
the chapter's Mermaid colors. Numbers here are derived from the same constants/algorithms as
quantization.py, so the figures and the prose stay in lockstep. Re-run after any numeric change:

    python make_figures.py

Verified on Python 3.12 / matplotlib 3.x. CPU only, deterministic (seeded).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np

# Muted palette mirroring the chapter's Mermaid classDefs (all readable on white).
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
GRID = "#D8DCE0"

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
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


def _quant_per_tensor(w: np.ndarray) -> np.ndarray:
    s = np.abs(w).max() / INT8_QMAX
    return np.clip(np.round(w / s), -INT8_QMAX, INT8_QMAX) * s


def _quant_per_row(w: np.ndarray) -> np.ndarray:
    s = np.abs(w).max(axis=1, keepdims=True) / INT8_QMAX
    s = np.where(s == 0, 1.0, s)
    return np.clip(np.round(w / s), -INT8_QMAX, INT8_QMAX) * s


def _quant_per_group_cols(w: np.ndarray, group: int) -> np.ndarray:
    rows, cols = w.shape
    wg = w.reshape(rows, cols // group, group)
    s = np.abs(wg).max(axis=-1, keepdims=True) / INT8_QMAX
    s = np.where(s == 0, 1.0, s)
    return (np.clip(np.round(wg / s), -INT8_QMAX, INT8_QMAX) * s).reshape(rows, cols)


def fig_outlier_error_bars() -> str:
    """Reconstruction error: per-tensor vs per-channel vs per-group, under a column outlier."""
    rng = np.random.default_rng(SEED)
    rows, cols = 8, 128
    w = rng.normal(0, 0.1, size=(rows, cols))
    w[:, 17] = rng.normal(0, 10.0, size=rows)  # one loud column across all rows
    err = {
        "per-tensor\n(1 scale)": np.abs(w - _quant_per_tensor(w)).mean(),
        "per-channel\n(1 / row)": np.abs(w - _quant_per_row(w)).mean(),
        "per-group\n(1 / 16 cols)": np.abs(w - _quant_per_group_cols(w, 16)).mean(),
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
    b_int4 = INT4_QMAX and (4 / 8 + FP16_BYTES / GROUP_SIZE)  # 0.5 code + scale overhead = 0.5312
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
    """Reconstruction error vs bit-width for group-wise weight quantization (measured here)."""
    rng = np.random.default_rng(SEED)
    w = rng.normal(0, 0.1, size=(256, 1024))
    group = GROUP_SIZE
    bits_list = [8, 6, 5, 4, 3, 2]
    errs = []
    for b in bits_list:
        qmax = (1 << (b - 1)) - 1
        wg = w.reshape(256, 1024 // group, group)
        s = np.abs(wg).max(axis=-1, keepdims=True) / qmax
        s = np.where(s == 0, 1.0, s)
        w_hat = (np.clip(np.round(wg / s), -qmax, qmax) * s).reshape(256, 1024)
        errs.append(np.abs(w - w_hat).mean())
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


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    made = [
        fig_affine_number_line(),
        fig_weight_histogram_bins(),
        fig_outlier_heatmap(),
        fig_outlier_error_bars(),
        fig_int4_memory_bars(),
        fig_error_vs_bits(),
    ]
    print(f"wrote {len(made)} figures to {IMAGES_DIR}:")
    for name in made:
        print("  -", name)


if __name__ == "__main__":
    main()
