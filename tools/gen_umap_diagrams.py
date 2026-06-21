"""UMAP concept-page diagrams (muted palette). REAL measured.

Four figures for 04. Unsupervised_Learning/concepts/08-UMAP.md:
  1. umap_digits.png     -- UMAP 2-D embedding of the digits dataset (64-D -> 2-D),
                            coloured by class: ten clean, well-separated clusters.
  2. umap_vs_tsne_pca.png-- UMAP vs t-SNE vs PCA on the SAME digits data, side by
                            side, each panel annotated with its wall-clock fit time:
                            the global-structure + speed story in one figure.
  3. umap_neighbors_mindist.png -- the two knobs, measured: top row sweeps
                            n_neighbors (local -> global), bottom row sweeps
                            min_dist (tight clumps -> even spread).
  4. umap_membership_curve.png -- the low-D membership weight Phi(d)=(1+a*d^(2b))^-1
                            fitted from several min_dist values (umap's own fit),
                            contrasted with the t-SNE Student-t tail.

All numbers/embeddings are measured at runtime; nothing is hand-drawn. Run with:
  /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_umap_diagrams.py
"""
import os
import time
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
import umap  # noqa: E402  (umap-learn)
from umap.umap_ import find_ab_params  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "04. Unsupervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
SEED = 42


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _scatter(ax, emb, y, title, s=5):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=s, alpha=0.85, linewidths=0)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    return sc


# ---------------------------------------------------------------- figure 1
def umap_digits(X, y):
    t0 = time.perf_counter()
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(X)
    dt = time.perf_counter() - t0
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    sc = _scatter(ax, emb, y, f"UMAP of the digits dataset (64-D → 2-D), coloured by digit", s=7)
    cb = fig.colorbar(sc, ax=ax, ticks=range(10), fraction=0.046, pad=0.04)
    cb.set_label("true digit class", fontsize=10)
    ax.text(0.015, 0.985, f"n_neighbors=15, min_dist=0.1\nfit {dt:.1f}s on {X.shape[0]} points",
            transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=SLATE, alpha=0.9))
    fig.tight_layout(); fig.savefig(f"{OUT}/umap_digits.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote umap_digits.png  (fit {dt:.2f}s)")
    return dt


# ---------------------------------------------------------------- figure 2
def umap_vs_tsne_pca(X, y):
    times = {}
    t0 = time.perf_counter(); pca = PCA(n_components=2, random_state=SEED).fit_transform(X); times["PCA"] = time.perf_counter() - t0
    t0 = time.perf_counter(); um = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(X); times["UMAP"] = time.perf_counter() - t0
    t0 = time.perf_counter(); ts = TSNE(n_components=2, init="pca", perplexity=30, random_state=SEED).fit_transform(X); times["t-SNE"] = time.perf_counter() - t0

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    _scatter(axes[0], pca, y, f"PCA (linear)\nfit {times['PCA']*1000:.0f} ms  —  instant, but classes overlap")
    _scatter(axes[1], ts, y, f"t-SNE (non-linear)\nfit {times['t-SNE']:.1f} s  —  clean clusters, global layout weak")
    _scatter(axes[2], um, y, f"UMAP (non-linear)\nfit {times['UMAP']:.1f} s  —  clean clusters + tighter global layout")
    fig.suptitle("Same digits data, three methods: PCA vs t-SNE vs UMAP", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/umap_vs_tsne_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote umap_vs_tsne_pca.png  (PCA {times['PCA']*1000:.0f}ms, UMAP {times['UMAP']:.2f}s, t-SNE {times['t-SNE']:.2f}s)")
    return times


# ---------------------------------------------------------------- figure 3
def umap_neighbors_mindist(X, y):
    nn_vals = [5, 15, 50, 200]
    md_vals = [0.0, 0.1, 0.5, 0.99]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, nn in zip(axes[0], nn_vals):
        emb = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=SEED).fit_transform(X)
        _scatter(ax, emb, y, f"n_neighbors = {nn}", s=4)
    for ax, md in zip(axes[1], md_vals):
        emb = umap.UMAP(n_neighbors=15, min_dist=md, random_state=SEED).fit_transform(X)
        _scatter(ax, emb, y, f"min_dist = {md}", s=4)
    axes[0, 0].set_ylabel("sweep n_neighbors\n(local → global)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("sweep min_dist\n(tight → spread)", fontsize=11, fontweight="bold")
    fig.suptitle("UMAP's two knobs, measured: n_neighbors (top) and min_dist (bottom)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}/umap_neighbors_mindist.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote umap_neighbors_mindist.png")


# ---------------------------------------------------------------- figure 4
def membership_curve():
    d = np.linspace(0, 4, 400)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    md_colors = [(0.0, BLUE), (0.1, GREEN), (0.5, AMBER), (0.99, RED)]
    for md, c in md_colors:
        a, b = find_ab_params(spread=1.0, min_dist=md)
        phi = 1.0 / (1.0 + a * d ** (2 * b))
        ax.plot(d, phi, color=c, lw=2.6, label=f"min_dist={md}  (a={a:.2f}, b={b:.2f})")
    # t-SNE Student-t (1 d.o.f.) tail for contrast
    student = 1.0 / (1.0 + d ** 2)
    ax.plot(d, student, color=SLATE, lw=2.0, ls="--", label="t-SNE Student-t  1/(1+d²)")
    ax.set_xlabel("low-D distance between two points  d")
    ax.set_ylabel("membership / similarity weight  Φ(d)")
    ax.set_title("Low-D membership curve Φ(d)=(1+a·d$^{2b}$)$^{-1}$ — min_dist sets the plateau",
                 fontsize=13.5, fontweight="bold")
    ax.annotate("small min_dist → flat-topped near 0:\npoints in the SAME cluster pile up tightly",
                (0.1, 0.99), xytext=(0.7, 0.78), fontsize=9.5, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.annotate("large min_dist → curve decays\nimmediately: points spread out evenly",
                (0.5, 0.55), xytext=(1.4, 0.5), fontsize=9.5, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.set_xlim(0, 4); ax.set_ylim(0, 1.03); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/umap_membership_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote umap_membership_curve.png")


if __name__ == "__main__":
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"digits: X={X.shape}, classes={len(np.unique(y))}")
    umap_digits(X, y)
    umap_vs_tsne_pca(X, y)
    umap_neighbors_mindist(X, y)
    membership_curve()
    print("OUT:", OUT)
