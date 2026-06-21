"""Hierarchical-clustering concept-page diagrams (muted palette). REAL measured.

Four figures for 04. Unsupervised_Learning/concepts/02-Hierarchical-Clustering.md:
  1. hier_dendrogram_cut.png -- a real scipy dendrogram with a horizontal cut line
     that yields k flat clusters; merge heights and the cut are measured.
  2. hier_linkage_compare.png -- single vs complete vs Ward linkage on the SAME data,
     each producing a DIFFERENT flat clustering (measured labels).
  3. hier_merge_sequence.png -- the agglomerative merge sequence on a small 2-D point
     set, shown as panels (which pair merges next, by Ward).
  4. hier_chaining.png -- single-linkage CHAINING vs Ward on two-moons / elongated
     data: single follows the non-convex shape, Ward cuts it straight (measured).

Run: /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_hierarchical_diagrams.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs, make_moons

OUT = os.path.join(os.path.dirname(__file__), "..", "04. Unsupervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
PALETTE = [BLUE, GREEN, AMBER, RED, PURPLE, NAVY]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
def dendrogram_cut():
    """A measured dendrogram + a horizontal cut producing k clusters."""
    X, _ = make_blobs(n_samples=30, centers=4, cluster_std=0.65,
                      random_state=7)
    Z = linkage(X, method="ward")
    # cut so that exactly 4 clusters fall out; place the line between the
    # 4-cluster and 3-cluster merge heights.
    heights = Z[:, 2]
    cut = (heights[-4] + heights[-3]) / 2.0        # between merge n-4 and n-3
    labels = fcluster(Z, t=4, criterion="maxclust")
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    dendrogram(Z, ax=ax, color_threshold=cut,
               above_threshold_color=SLATE)
    ax.axhline(cut, color=RED, lw=2.2, ls="--")
    ax.text(0.5, cut, f"  cut at h={cut:.2f}  ->  {len(set(labels))} clusters",
            color=RED, fontsize=10.5, fontweight="bold", va="bottom")
    ax.set_title("Dendrogram: merge heights encode order; cutting yields a flat clustering",
                 fontsize=12.5, fontweight="bold")
    ax.set_xlabel("data points (leaves)", fontsize=10.5)
    ax.set_ylabel("merge height = Ward linkage distance", fontsize=10.5)
    _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/hier_dendrogram_cut.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote hier_dendrogram_cut.png  (cut=%.3f, k=%d)" % (cut, len(set(labels))))


# ---------------------------------------------------------------------------
def linkage_compare():
    """Single vs complete vs Ward on the SAME data -> different clusterings."""
    Xm, _ = make_moons(n_samples=160, noise=0.06, random_state=0)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    for ax, method in zip(axes, ("single", "complete", "ward")):
        Z = linkage(Xm, method=method)
        labels = fcluster(Z, t=2, criterion="maxclust")
        for c in np.unique(labels):
            m = labels == c
            ax.scatter(Xm[m, 0], Xm[m, 1], s=18, color=PALETTE[(c - 1) % len(PALETTE)])
        ax.set_title(f"{method.capitalize()} linkage  (k=2 cut)",
                     fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Same data, same k -- the LINKAGE decides the clustering",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/hier_linkage_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote hier_linkage_compare.png")


# ---------------------------------------------------------------------------
def merge_sequence():
    """Agglomerative merges step by step on a tiny hand-placed 2-D set (Ward).

    A compact 8-point set in two loose groups so each panel visibly shows the
    next-closest pair merging into a growing cluster.
    """
    X = np.array([
        [0.0, 0.0], [0.5, 0.4], [0.3, 1.0],          # lower-left trio
        [1.2, 0.6],                                   # bridge point
        [3.0, 2.6], [3.5, 3.0], [3.2, 2.0], [4.0, 2.7],  # upper-right group
    ])
    Z = linkage(X, method="ward")
    n = X.shape[0]
    targets = [n, 6, 4, 2]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.8))
    for ax, k in zip(axes, targets):
        labels = fcluster(Z, t=k, criterion="maxclust")
        for c in np.unique(labels):
            m = labels == c
            col = PALETTE[(c - 1) % len(PALETTE)]
            ax.scatter(X[m, 0], X[m, 1], s=150, color=col,
                       edgecolors="white", linewidths=1.1, zorder=3)
            if m.sum() > 1:                       # link members to their centroid
                cx, cy = X[m, 0].mean(), X[m, 1].mean()
                for px, py in X[m]:
                    ax.plot([cx, px], [cy, py], color=col, lw=1.4, alpha=0.55, zorder=1)
        ntitle = "8 singletons (start)" if k == n else f"{k} clusters"
        ax.set_title(ntitle, fontsize=11.5, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, 4.6); ax.set_ylim(-0.6, 3.6)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_alpha(0.3)
    fig.suptitle("Agglomerative merge sequence: 8 singletons -> repeatedly merge the closest pair (Ward)",
                 fontsize=12.5, fontweight="bold", y=1.05)
    fig.tight_layout(); fig.savefig(f"{OUT}/hier_merge_sequence.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote hier_merge_sequence.png")


# ---------------------------------------------------------------------------
def chaining():
    """Single-linkage chaining vs Ward on two-moons (non-convex)."""
    Xm, _ = make_moons(n_samples=220, noise=0.05, random_state=1)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, method, title in (
        (axes[0], "single", "Single linkage: follows the moons (chaining wins here)"),
        (axes[1], "ward", "Ward: compact blobs -> slices the moons in half")):
        Z = linkage(Xm, method=method)
        labels = fcluster(Z, t=2, criterion="maxclust")
        for c in np.unique(labels):
            m = labels == c
            ax.scatter(Xm[m, 0], Xm[m, 1], s=16, color=PALETTE[(c - 1) % len(PALETTE)])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Chaining is a feature OR a bug: single-linkage on non-convex data",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/hier_chaining.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote hier_chaining.png")


if __name__ == "__main__":
    dendrogram_cut()
    linkage_compare()
    merge_sequence()
    chaining()
    print("all hierarchical diagrams written to", OUT)
