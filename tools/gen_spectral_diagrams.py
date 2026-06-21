"""Spectral Clustering concept-page diagrams (muted palette, parallel scale). REAL measured.

Four figures for 04. Unsupervised_Learning/concepts/05-Spectral-Clustering.md:
  1. spectral_vs_kmeans.png -- two-moons + concentric circles: spectral clustering
     separates the connectivity-defined groups, k-means cuts straight through them.
     Accuracy is measured and printed per panel.
  2. spectral_embedding.png -- the pipeline made visible: the similarity graph on
     two-moons, then the 2-D spectral embedding (the bottom non-trivial eigenvectors
     of L_sym) where the two moons become two tight, linearly separable blobs.
  3. spectral_eigengap.png -- the eigenvalue spectrum of L_sym for a 3-blob dataset;
     the large gap after the 3rd eigenvalue (the eigengap) is what picks k = 3.
  4. spectral_sigma_knn.png -- how the affinity scale changes everything: RBF sigma
     too small / right / too large, and k-NN with small vs large k, with measured
     accuracy showing the sweet spot and the failure modes.

All numbers are computed here so the page can quote them exactly. Run with:
  ~/.uv/envs/ml-py312/bin/python3 tools/gen_spectral_diagrams.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.neighbors import kneighbors_graph

OUT = os.path.join(os.path.dirname(__file__), "..", "04. Unsupervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
PAL = [BLUE, RED, GREEN, AMBER, PURPLE]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _acc(true, pred):
    """Clustering accuracy up to label permutation (2 clusters)."""
    a = np.mean(true == pred)
    return max(a, 1 - a)


def fig_spectral_vs_kmeans():
    """Spectral separates connectivity groups; k-means cuts through them."""
    Xm, ym = make_moons(n_samples=400, noise=0.06, random_state=0)
    Xc, yc = make_circles(n_samples=400, noise=0.04, factor=0.45, random_state=0)
    datasets = [("Two moons", Xm, ym), ("Concentric circles", Xc, yc)]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0))
    for row, (name, X, y) in enumerate(datasets):
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
        sc = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                                n_neighbors=10, assign_labels="kmeans",
                                random_state=0).fit_predict(X)
        panels = [("True clusters", y, None),
                  ("k-means (fails)", km, _acc(y, km)),
                  ("Spectral (k-NN graph)", sc, _acc(y, sc))]
        for col, (title, lab, acc) in enumerate(panels):
            ax = axes[row, col]
            for c in np.unique(lab):
                ax.scatter(X[lab == c, 0], X[lab == c, 1], s=10,
                           color=PAL[c % len(PAL)], alpha=0.85)
            t = f"{name}\n{title}" if col == 0 else title
            if acc is not None:
                t += f"  (acc {acc:.0%})"
            ax.set_title(t, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Spectral clustering finds connectivity; k-means assumes convex blobs",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, "spectral_vs_kmeans.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[1] {os.path.basename(p)}")
    print(f"    moons  : kmeans acc={_acc(ym, KMeans(2, n_init=10, random_state=0).fit_predict(Xm)):.0%}  "
          f"spectral acc={_acc(ym, SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=10, random_state=0).fit_predict(Xm)):.0%}")
    print(f"    circles: kmeans acc={_acc(yc, KMeans(2, n_init=10, random_state=0).fit_predict(Xc)):.0%}  "
          f"spectral acc={_acc(yc, SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=10, random_state=0).fit_predict(Xc)):.0%}")


def _sym_laplacian_embedding(X, k_eig, n_neighbors=10):
    """Bottom-k eigenvectors of L_sym = I - D^-1/2 W D^-1/2 (row-normalized, Ng-Jordan-Weiss)."""
    A = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity")
    W = 0.5 * (A + A.T).toarray()           # symmetrize the k-NN graph
    d = W.sum(axis=1)
    Dinv2 = np.diag(1.0 / np.sqrt(d))
    Lsym = np.eye(len(X)) - Dinv2 @ W @ Dinv2
    vals, vecs = np.linalg.eigh(Lsym)        # ascending
    U = vecs[:, :k_eig]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)   # row-normalize
    return W, vals, U


def fig_spectral_embedding():
    """Graph -> spectral embedding: tangled moons become two tight blobs."""
    X, y = make_moons(n_samples=300, noise=0.06, random_state=0)
    W, vals, U = _sym_laplacian_embedding(X, k_eig=2, n_neighbors=12)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    # (a) the similarity graph
    ax = axes[0]
    ii, jj = np.nonzero(np.triu(W))
    for a, b in zip(ii, jj):
        ax.plot([X[a, 0], X[b, 0]], [X[a, 1], X[b, 1]],
                color=SLATE, lw=0.3, alpha=0.4, zorder=1)
    for c in np.unique(y):
        ax.scatter(X[y == c, 0], X[y == c, 1], s=14, color=PAL[c], zorder=2)
    ax.set_title("1. Similarity graph\n(12-NN, edges = neighbors)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # (b) the spectral embedding (eigenvectors 1 and 2 of L_sym)
    ax = axes[1]
    for c in np.unique(y):
        ax.scatter(U[y == c, 0], U[y == c, 1], s=16, color=PAL[c])
    ax.set_title("2. Spectral embedding\n(rows of bottom-2 eigenvectors)", fontsize=11)
    ax.set_xlabel("eigenvector 1 (Fiedler)"); ax.set_ylabel("eigenvector 2")
    _despine(ax)

    # (c) k-means in the embedding recovers the moons
    lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(U)
    ax = axes[2]
    for c in np.unique(lab):
        ax.scatter(X[lab == c, 0], X[lab == c, 1], s=14, color=PAL[c])
    ax.set_title(f"3. k-means on embedding\n-> back in data space (acc {_acc(y, lab):.0%})", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("The spectral embedding makes non-convex clusters linearly separable",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "spectral_embedding.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[2] {os.path.basename(p)}")
    print(f"    L_sym smallest 5 eigenvalues: {np.round(vals[:5], 4)}")
    print(f"    k-means-on-embedding accuracy on two-moons: {_acc(y, lab):.0%}")


def fig_eigengap():
    """Eigenvalue spectrum + the eigengap that reveals k."""
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.70,
                      center_box=(-7.0, 7.0), random_state=7)
    W, vals, U = _sym_laplacian_embedding(X, k_eig=3, n_neighbors=8)
    k = 10
    ev = vals[:k]
    gaps = np.diff(ev)
    # Eigengap heuristic: the FIRST large jump in the (sorted) spectrum marks k.
    # Search the small-eigenvalue region (the only place a meaningful gap lives).
    kstar = int(np.argmax(gaps[:5])) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4),
                             gridspec_kw={"width_ratios": [1.0, 1.2]})
    ax = axes[0]
    for c in np.unique(y):
        ax.scatter(X[y == c, 0], X[y == c, 1], s=16, color=PAL[c])
    ax.set_title("Data: 3 blobs", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1]
    idx = np.arange(1, k + 1)
    ax.plot(idx, ev, color=BLUE, lw=1.8, marker="o", ms=6, zorder=2)
    ax.axvline(kstar + 0.5, color=RED, lw=2.0, ls="--",
               label=f"largest eigengap after k={kstar}")
    ax.annotate(f"eigengap  Δ={gaps[kstar-1]:.3f}",
                xy=(kstar + 0.5, (ev[kstar-1] + ev[kstar]) / 2),
                xytext=(kstar + 1.6, ev[kstar-1] + 0.012),
                color=RED, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("eigenvalue index  i"); ax.set_ylabel(r"eigenvalue $\lambda_i$ of $L_{sym}$")
    ax.set_title("Eigengap heuristic picks k", fontsize=11)
    ax.legend(fontsize=9, loc="upper left"); _despine(ax)

    fig.suptitle("The eigengap: a jump in the spectrum reveals the number of clusters",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "spectral_eigengap.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[3] {os.path.basename(p)}")
    print(f"    first {k} eigenvalues: {np.round(ev, 4)}")
    print(f"    largest gap after index {kstar} (Δ={gaps[kstar-1]:.4f}) -> pick k={kstar}")


def fig_sigma_knn():
    """How the affinity scale (RBF sigma) and k-NN k change the result."""
    X, y = make_moons(n_samples=400, noise=0.06, random_state=0)
    D2 = cdist(X, X, "sqeuclidean")

    def rbf_spectral(sigma):
        W = np.exp(-D2 / (2 * sigma ** 2))
        np.fill_diagonal(W, 0.0)
        sc = SpectralClustering(n_clusters=2, affinity="precomputed",
                                assign_labels="kmeans", random_state=0).fit_predict(W)
        return sc

    def knn_spectral(kk):
        sc = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                                n_neighbors=kk, random_state=0).fit_predict(X)
        return sc

    configs = [
        ("RBF σ=0.01 (too small)", rbf_spectral(0.01)),
        ("RBF σ=0.10 (right)", rbf_spectral(0.10)),
        ("RBF σ=2.0 (too large)", rbf_spectral(2.0)),
        ("k-NN k=3 (too sparse)", knn_spectral(3)),
        ("k-NN k=10 (right)", knn_spectral(10)),
        ("k-NN k=120 (too dense)", knn_spectral(120)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0))
    for ax, (title, lab) in zip(axes.ravel(), configs):
        for c in np.unique(lab):
            ax.scatter(X[lab == c, 0], X[lab == c, 1], s=9, color=PAL[c % len(PAL)], alpha=0.85)
        ax.set_title(f"{title}\nacc {_acc(y, lab):.0%}", fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The graph IS the model: σ and k decide success or failure",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "spectral_sigma_knn.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[4] {os.path.basename(p)}")
    for title, lab in configs:
        print(f"    {title:28s} acc={_acc(y, lab):.0%}")


if __name__ == "__main__":
    fig_spectral_vs_kmeans()
    fig_spectral_embedding()
    fig_eigengap()
    fig_sigma_knn()
    print("done.")
