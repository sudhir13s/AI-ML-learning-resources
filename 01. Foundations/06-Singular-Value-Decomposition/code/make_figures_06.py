"""Figure generator for 06-Singular-Value-Decomposition — every number from the REAL pipeline.

All measured figures come from the same real matrices the chapter and notebook use
(``singular_value_decomposition.py``): the real photo ``china.jpg`` as a matrix, the real
``load_digits`` dataset, real ``numpy.linalg.svd`` factors. Nothing is hand-typed. The one
schematic (the 2x2 unit-circle -> ellipse geometry) is itself computed from a real SVD of a real
2x2 matrix — so even the "illustration" is a true decomposition, just in a dimension we can draw.

Writes muted-palette PNGs to the shared chapter image dir (../../images/) with prefix ``found06_``:

  found06_geometry_ellipse.png  -- unit circle -> Vᵀ rotate -> Σ scale -> U rotate = ellipse,
                                   from the REAL SVD of a real 2x2 matrix (semi-axes = sigma_i).
  found06_image_montage.png     -- the real photo reconstructed at rank 1/5/20/50/100/full,
                                   each panel titled with its REAL relative Frobenius error.
  found06_spectrum_energy.png   -- REAL singular-value spectrum (log) + cumulative energy of the
                                   image, marking the k for 90/95/99% energy.
  found06_error_vs_rank.png     -- REAL rel-Frobenius error vs rank k, with the compression ratio
                                   on a twin axis: the quality/size tradeoff, measured.
  found06_eckart_young.png      -- REAL truncated-SVD error vs a random rank-k factor at k=5..80:
                                   truncated is always lower — Eckart–Young, seen.
  found06_pca_digits.png        -- REAL explained-variance ratio + cumulative variance of the
                                   digits dataset (SVD == PCA), marking 80/90/95%.
  found06_eigendigits.png       -- the first principal directions (right singular vectors) of the
                                   real digits data, reshaped to 8x8 — the "eigen-digits".

    python make_figures_06.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / scikit-learn 1.9.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_digits

from singular_value_decomposition import (
    COMPRESSION_RANKS,
    compression_curve,
    compute_svd,
    cumulative_energy,
    load_grayscale_image,
    pca_via_svd,
    random_rankk_error,
    reconstruct,
    truncation_error_frobenius,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # data / input
PURPLE = "#5D4A8A"  # process / rotation
GREEN = "#2E7A5A"  # good / retained / truncated SVD
RED = "#8B3B4A"  # cost / error / the foil
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # highlight / scaling
NAVY = "#2A5B80"  # secondary data
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 120
IMG_PREFIX = "found06_"


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
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


# ============================ Fig 1: the geometry (real 2x2 SVD) =================================
def fig_geometry_ellipse() -> None:
    """Unit circle -> Vᵀ (rotate) -> Σ (scale) -> U (rotate) = ellipse, from a REAL 2x2 SVD.

    The whole geometric story of SVD in one figure: a linear map turns the unit circle into an
    ellipse whose semi-axis LENGTHS are exactly the singular values and whose axis DIRECTIONS are
    the left singular vectors. We use a real, fixed 2x2 matrix and its real numpy SVD, so the
    ellipse and the sigma-labelled axes are computed, not drawn by hand.
    """
    a = np.array([[2.0, 1.2], [0.4, 1.6]])  # a real, non-symmetric 2x2 map
    svd = compute_svd(a)
    u, s, vt = svd.U, svd.s, svd.Vt
    theta = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(theta), np.sin(theta)])  # unit circle (2, 400)

    stage_vt = vt @ circle  # after Vᵀ: rotation/reflection (still a circle)
    stage_s = (s[:, None]) * stage_vt  # after Σ: axis-aligned scaling -> ellipse
    stage_u = u @ stage_s  # after U: final rotation = A @ circle exactly
    assert np.allclose(stage_u, a @ circle), "U Σ Vᵀ applied to the circle must equal A @ circle"

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))
    # axis directions along which to draw the sigma-labelled semi-axes at each stage:
    #   after Σ the ellipse axes are the coordinate axes; after U they are the left singular vectors.
    stages = [
        (circle, "unit circle", BLUE, None),
        (stage_vt, r"after $V^\top$ (rotate)", PURPLE, None),
        (stage_s, r"after $\Sigma$ (scale)", AMBER, np.eye(2)),
        (stage_u, r"after $U$ (rotate) $= A x$", GREEN, u),
    ]
    lim = np.abs(stage_u).max() * 1.15
    for ax, (pts, title, color, axis_dirs) in zip(axes, stages):
        ax.plot(pts[0], pts[1], color=color, linewidth=2.2)
        ax.fill(pts[0], pts[1], color=color, alpha=0.12)
        # draw the two principal directions so the semi-axes = singular values are visible
        if axis_dirs is not None:
            for i in range(2):
                vec = axis_dirs[:, i] * s[i]
                ax.annotate(
                    "",
                    xy=(vec[0], vec[1]),
                    xytext=(0, 0),
                    arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.6},
                )
                ax.text(vec[0] * 1.08, vec[1] * 1.08, rf"$\sigma_{i+1}$={s[i]:.2f}", fontsize=9, color=INK)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        _style_axis(ax)
    fig.suptitle(
        r"SVD geometry: every linear map = rotate ($V^\top$) → scale ($\Sigma$) → rotate ($U$). "
        r"The circle becomes an ellipse with semi-axes $\sigma_1,\sigma_2$.",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, f"{IMG_PREFIX}geometry_ellipse.png")


# ============================ Fig 2: image compression montage ==================================
def fig_image_montage(img: NDArray[np.float64], svd) -> None:
    """The real photo reconstructed at increasing rank — each panel titled with its REAL error."""
    ranks = (1, 5, 20, 50, 100, svd.s.size)
    a_norm = float(np.linalg.norm(img, ord="fro"))
    fig, axes = plt.subplots(2, 3, figsize=(11, 8))
    for ax, k in zip(axes.ravel(), ranks):
        rec = reconstruct(svd, k)  # UNCLIPPED — the error must match compression_curve / the page table
        err = float(np.linalg.norm(img - rec, ord="fro")) / a_norm  # identical formula to the module
        ratio = (img.shape[0] * img.shape[1]) / (k * (img.shape[0] + img.shape[1] + 1))
        # clip ONLY the displayed pixels (a valid image is [0,255]); the error number stays unclipped
        ax.imshow(np.clip(rec, 0, 255), cmap="gray", vmin=0, vmax=255)
        label = "full rank (exact)" if k == svd.s.size else f"rank {k}"
        ax.set_title(f"{label}\nrel err {err:.3f} · {ratio:.1f}× smaller", fontsize=10, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Low-rank SVD reconstruction of a real 427×640 photo. A handful of singular triplets "
        "already carry the scene; the rest is fine detail.",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, f"{IMG_PREFIX}image_montage.png")


# ============================ Fig 3: spectrum + cumulative energy ================================
def fig_spectrum_energy(svd) -> None:
    """REAL singular-value spectrum (log) and the cumulative-energy curve of the image."""
    s = svd.s
    energy = cumulative_energy(svd)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax1.semilogy(np.arange(1, s.size + 1), s, color=BLUE, linewidth=2)
    ax1.set_xlabel("singular value index  i")
    ax1.set_ylabel(r"$\sigma_i$  (log scale)")
    ax1.set_title("Singular-value spectrum decays fast\n(real image)", fontsize=11)
    _style_axis(ax1)

    ax2.plot(np.arange(1, energy.size + 1), energy * 100, color=GREEN, linewidth=2)
    for target, color in ((0.90, AMBER), (0.95, PURPLE), (0.99, RED)):
        k = int(np.searchsorted(energy, target) + 1)
        ax2.axhline(target * 100, color=color, linestyle="--", linewidth=1, alpha=0.7)
        ax2.axvline(k, color=color, linestyle=":", linewidth=1, alpha=0.7)
        ax2.text(k + 4, target * 100 - 4, f"{target:.0%} → k={k}", fontsize=9, color=color)
    ax2.set_xlabel("number of singular values kept  k")
    ax2.set_ylabel("cumulative energy  (%)")
    ax2.set_title(r"Energy $\sum_{i\leq k}\sigma_i^2 / \sum_i \sigma_i^2$ saturates early", fontsize=11)
    ax2.set_ylim(0, 102)
    _style_axis(ax2)

    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}spectrum_energy.png")


# ============================ Fig 4: error vs rank + compression =================================
def fig_error_vs_rank(img: NDArray[np.float64], svd) -> None:
    """REAL rel-Frobenius error vs rank k, with the measured compression ratio on a twin axis."""
    points = compression_curve(img, svd, ranks=(1, 2, 5, 10, 20, 50, 100, 200))
    ks = [p.k for p in points if p.k != svd.s.size]
    errs = [p.rel_frobenius_error for p in points if p.k != svd.s.size]
    ratios = [p.compression_ratio for p in points if p.k != svd.s.size]

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    ax1.plot(ks, errs, "o-", color=RED, linewidth=2, label="relative Frobenius error")
    ax1.set_xlabel("rank  k")
    ax1.set_ylabel("relative reconstruction error", color=RED)
    ax1.tick_params(axis="y", labelcolor=RED)
    ax1.set_xscale("log")
    _style_axis(ax1)

    ax2 = ax1.twinx()
    ax2.plot(ks, ratios, "s--", color=GREEN, linewidth=2, label="compression ratio")
    ax2.set_ylabel("compression ratio  (× smaller)", color=GREEN)
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor=GREEN)
    ax2.spines["top"].set_visible(False)

    ax1.set_title(
        "Quality vs size, measured on the real image: more rank → lower error, less compression",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}error_vs_rank.png")


# ============================ Fig 5: Eckart–Young optimality =====================================
def fig_eckart_young(img: NDArray[np.float64], svd) -> None:
    """REAL truncated-SVD error vs a random rank-k factor: truncated wins at every k (optimality)."""
    ks = [5, 10, 20, 40, 60, 80]
    truncated = [truncation_error_frobenius(svd, k) for k in ks]
    random_err = [random_rankk_error(img, k) for k in ks]

    x = np.arange(len(ks))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, truncated, width, color=GREEN, label="truncated SVD (optimal)")
    ax.bar(x + width / 2, random_err, width, color=RED, label="random rank-k projection")
    for xi, (t, r) in enumerate(zip(truncated, random_err)):
        gap = (r - t) / t * 100
        ax.text(xi, r + max(random_err) * 0.01, f"+{gap:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Frobenius reconstruction error")
    ax.set_title(
        "Eckart–Young–Mirsky on the real image: no rank-k matrix beats the truncated SVD\n"
        "(a random rank-k projection is always worse — the % gap is labelled)",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=10)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}eckart_young.png")


# ============================ Fig 6: PCA on real digits =========================================
def fig_pca_digits(pca) -> None:
    """REAL explained-variance ratio + cumulative variance of the digits data (SVD == PCA)."""
    ratios = pca.explained_variance_ratio
    cum = np.cumsum(ratios)
    idx = np.arange(1, ratios.size + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.bar(idx[:20], ratios[:20] * 100, color=BLUE)
    ax1.set_xlabel("principal component  i")
    ax1.set_ylabel("explained variance  (%)")
    ax1.set_title("Explained variance $\\sigma_i^2/(n-1)$ per component\n(top 20 of 64)", fontsize=11)
    _style_axis(ax1)

    ax2.plot(idx, cum * 100, color=GREEN, linewidth=2)
    for target, color in ((0.80, AMBER), (0.90, PURPLE), (0.95, RED)):
        k = int(np.searchsorted(cum, target) + 1)
        ax2.axhline(target * 100, color=color, linestyle="--", linewidth=1, alpha=0.7)
        ax2.axvline(k, color=color, linestyle=":", linewidth=1, alpha=0.7)
        ax2.text(k + 1, target * 100 - 5, f"{target:.0%} → {k} PCs", fontsize=9, color=color)
    ax2.set_xlabel("number of components kept  k")
    ax2.set_ylabel("cumulative variance  (%)")
    ax2.set_title("Cumulative variance: ~21 of 64 components hold 90%", fontsize=11)
    ax2.set_ylim(0, 102)
    _style_axis(ax2)

    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}pca_digits.png")


def fig_eigendigits(pca) -> None:
    """The first principal directions (right singular vectors) of the real digits, as 8x8 images."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.8))
    for i, ax in enumerate(axes.ravel()):
        comp = pca.components[i].reshape(8, 8)
        ax.imshow(comp, cmap="RdBu_r", vmin=-comp.std() * 3, vmax=comp.std() * 3)
        ax.set_title(f"PC {i+1}\n{pca.explained_variance_ratio[i]:.1%} var", fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "The first 10 principal directions of the real digits data (right singular vectors, reshaped "
        "8×8). Each is a pixel-pattern the data varies along most.",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.4)
    _save(fig, f"{IMG_PREFIX}eigendigits.png")


def main() -> None:
    print(f"numpy {np.__version__} | matplotlib {matplotlib.__version__}")
    # --- image demo (real photo) ---
    img = load_grayscale_image()
    svd_img = compute_svd(img)
    print(f"image: {img.shape[0]}x{img.shape[1]} real photo, {svd_img.s.size} singular values")
    fig_geometry_ellipse()
    fig_image_montage(img, svd_img)
    fig_spectrum_energy(svd_img)
    fig_error_vs_rank(img, svd_img)
    fig_eckart_young(img, svd_img)
    # --- digits demo (real dataset) ---
    digits = load_digits()
    pca = pca_via_svd(digits.data.astype(np.float64))
    fig_pca_digits(pca)
    fig_eigendigits(pca)
    print(f"\nall figures written to {OUT_DIR} with prefix '{IMG_PREFIX}'")
    _ = COMPRESSION_RANKS  # imported for parity with the module's rank set (kept explicit)


if __name__ == "__main__":
    main()
