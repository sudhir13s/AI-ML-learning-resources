"""Dimensionality-Reduction / PCA concept-page diagrams (muted palette). REAL measured.

Four figures for 04. Unsupervised_Learning/concepts/06-Dimensionality-Reduction-Overview.md:
  1. pca_axes.png       -- 2-D correlated cloud with the two principal-axis arrows,
                           each scaled by sqrt(eigenvalue): PC1 is the long axis.
  2. pca_scree.png      -- scree + cumulative explained-variance on real digits (64-D),
                           with the 90% / 95% threshold lines and the chosen k.
  3. pca_reconstruct.png-- a digit reconstructed from k = 1..64 components, plus the
                           measured reconstruction-error-vs-k curve underneath.
  4. pca_vs_tsne.png     -- PCA (linear) vs t-SNE (non-linear) 2-D embeddings of the
                           SAME digits data: shows PCA's linear limitation.

All numbers are measured at runtime; nothing is hand-drawn. Run with:
  /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_dimreduction_diagrams.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

OUT = os.path.join(os.path.dirname(__file__), "..", "04. Unsupervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- figure 1
def pca_axes():
    """Correlated 2-D cloud + principal axes scaled by sqrt(eigenvalue)."""
    rng = np.random.default_rng(7)
    # a genuinely correlated cloud: stretch + rotate an isotropic Gaussian
    n = 300
    base = rng.normal(size=(n, 2)) * np.array([2.4, 0.7])
    theta = np.deg2rad(35.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X = base @ R.T + np.array([1.0, 0.5])

    pca = PCA(n_components=2).fit(X)
    mean = pca.mean_
    comps = pca.components_          # rows are unit principal directions
    eigvals = pca.explained_variance_   # variance along each PC

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(X[:, 0], X[:, 1], s=14, color=BLUE, alpha=0.45, edgecolors="none",
               label="data points")
    colors = [RED, GREEN]
    labels = [f"PC1  (var {eigvals[0]:.2f}, {100*eigvals[0]/eigvals.sum():.0f}%)",
              f"PC2  (var {eigvals[1]:.2f}, {100*eigvals[1]/eigvals.sum():.0f}%)"]
    for i in range(2):
        # arrow length proportional to standard deviation = sqrt(eigenvalue)
        v = comps[i] * np.sqrt(eigvals[i]) * 2.6
        ax.annotate("", xy=mean + v, xytext=mean,
                    arrowprops=dict(arrowstyle="-|>", lw=3.0, color=colors[i],
                                    mutation_scale=22))
        ax.text(*(mean + v * 1.08), labels[i], color=colors[i], fontsize=10.5,
                fontweight="bold", ha="center", va="center")
    ax.scatter(*mean, color="black", s=40, zorder=5)
    ax.text(mean[0], mean[1] - 0.4, "mean", fontsize=9, ha="center", color="black")
    ax.set_aspect("equal")
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.set_title("PCA finds the orthogonal axes of greatest variance\n"
                 "(arrow length = one standard deviation = $\\sqrt{\\lambda_i}$)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/pca_axes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote pca_axes.png  | eigvals={eigvals.round(3)} "
          f"ratio={ (eigvals/eigvals.sum()).round(3) }")


# ---------------------------------------------------------------- figure 2
def pca_scree():
    """Scree + cumulative explained variance on the digits dataset (8x8 = 64-D)."""
    X = load_digits().data
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    k95 = int(np.searchsorted(cum, 0.95) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    # scree (per-component)
    kk = np.arange(1, len(evr) + 1)
    ax1.bar(kk, evr * 100, color=BLUE, width=0.9)
    ax1.set_xlabel("principal component  i"); ax1.set_ylabel("explained variance  %")
    ax1.set_title("Scree plot: variance per component (the 'elbow')",
                  fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 40)
    _despine(ax1)
    # cumulative
    ax2.plot(kk, cum * 100, color=PURPLE, lw=2.6)
    for frac, k, col in [(90, k90, GREEN), (95, k95, AMBER)]:
        ax2.axhline(frac, color=col, ls="--", lw=1.4)
        ax2.axvline(k, color=col, ls=":", lw=1.4)
        ax2.scatter([k], [cum[k - 1] * 100], color=col, s=45, zorder=5)
        ax2.text(k + 1, frac - 6, f"{frac}% at k={k}", color=col, fontsize=10,
                 fontweight="bold")
    ax2.set_xlabel("number of components  k"); ax2.set_ylabel("cumulative explained variance  %")
    ax2.set_title("Cumulative variance: pick k for a target (e.g. 90 / 95%)",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 102)
    _despine(ax2)
    fig.tight_layout(); fig.savefig(f"{OUT}/pca_scree.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote pca_scree.png | top5 evr%={ (evr[:5]*100).round(1) } "
          f"k90={k90} k95={k95} total_dims={len(evr)}")


# ---------------------------------------------------------------- figure 3
def pca_reconstruct():
    """Reconstruct one digit from k components + error-vs-k curve (measured)."""
    digits = load_digits()
    X = digits.data
    mean = X.mean(axis=0)
    pca_full = PCA().fit(X)

    img_idx = 0                       # the canonical first '0' digit
    x = X[img_idx]
    ks = [1, 2, 5, 10, 20, 40, 64]
    recons = []
    for k in ks:
        comps = pca_full.components_[:k]                 # (k, 64)
        scores = (x - mean) @ comps.T                    # project
        rec = mean + scores @ comps                      # reconstruct
        recons.append(rec.reshape(8, 8))

    # measured mean reconstruction error vs k across the WHOLE dataset
    Xc = X - mean
    all_scores = Xc @ pca_full.components_.T              # (n, 64)
    err_k = []
    krange = np.arange(1, 65)
    for k in krange:
        rec = all_scores[:, :k] @ pca_full.components_[:k]
        err_k.append(np.mean(np.sum((Xc - rec) ** 2, axis=1)))
    err_k = np.array(err_k)

    fig = plt.figure(figsize=(12.8, 5.6))
    gs = fig.add_gridspec(2, len(ks) + 1, height_ratios=[1.0, 1.15], hspace=0.45, wspace=0.25)
    # top row: original + reconstructions
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(x.reshape(8, 8), cmap="gray_r"); ax0.set_title("original\n(64-D)", fontsize=10)
    ax0.axis("off")
    for j, (k, rec) in enumerate(zip(ks, recons)):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.imshow(rec, cmap="gray_r"); ax.set_title(f"k={k}", fontsize=10); ax.axis("off")
    # bottom row: error curve spanning full width
    axc = fig.add_subplot(gs[1, :])
    axc.plot(krange, err_k, color=RED, lw=2.6)
    for k in ks:
        axc.scatter([k], [err_k[k - 1]], color=BLUE, s=40, zorder=5)
    axc.set_xlabel("number of components  k"); axc.set_ylabel("mean reconstruction error  (SSE)")
    axc.set_title("Reconstruction error falls monotonically as k grows "
                  "(0 at k=64 = full rank)", fontsize=12, fontweight="bold")
    _despine(axc)
    fig.suptitle("PCA as lossy compression: a digit rebuilt from its top-k principal components",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.savefig(f"{OUT}/pca_reconstruct.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote pca_reconstruct.png | err(k=1)={err_k[0]:.1f} err(k=10)={err_k[9]:.1f} "
          f"err(k=64)={err_k[63]:.2e}")


# ---------------------------------------------------------------- figure 4
def pca_vs_tsne():
    """PCA (linear) vs t-SNE (non-linear) 2-D embedding of the SAME digits."""
    digits = load_digits()
    X, y = digits.data, digits.target
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2).fit(Xs)
    Z_pca = pca.transform(Xs)
    evr2 = pca.explained_variance_ratio_[:2].sum()
    Z_tsne = TSNE(n_components=2, init="pca", perplexity=30, random_state=0,
                  learning_rate="auto").fit_transform(Xs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.6))
    for ax, Z, ttl in [
        (ax1, Z_pca, f"PCA (linear): top 2 PCs capture only {100*evr2:.0f}%\n"
                     "of variance — clusters overlap"),
        (ax2, Z_tsne, "t-SNE (non-linear): unfolds the manifold\n"
                      "into clean, separated clusters")]:
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=14, alpha=0.7,
                        edgecolors="none")
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        _despine(ax)
    cbar = fig.colorbar(sc, ax=[ax1, ax2], fraction=0.025, pad=0.02, ticks=range(10))
    cbar.set_label("digit class", fontsize=10)
    fig.suptitle("Same data, two methods: PCA preserves global variance; "
                 "t-SNE preserves local neighbourhoods",
                 fontsize=13, fontweight="bold")
    fig.savefig(f"{OUT}/pca_vs_tsne.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote pca_vs_tsne.png | PCA top-2 evr={100*evr2:.1f}%")


if __name__ == "__main__":
    pca_axes()
    pca_scree()
    pca_reconstruct()
    pca_vs_tsne()
    print("OUT:", OUT)
