"""DBSCAN concept-page diagrams (muted palette, parallel scale). REAL measured.

Four figures for 04. Unsupervised_Learning/concepts/03-DBSCAN.md:
  1. dbscan_taxonomy.png   -- core / border / noise point taxonomy with an eps
     neighborhood drawn around a core and a border point (annotated schematic on a
     real measured 2-D point set; minPts=4).
  2. dbscan_vs_kmeans.png  -- DBSCAN nails arbitrary shapes (two moons + concentric
     circles) and flags noise; k-means cuts them straight through. Measured labels.
  3. dbscan_kdistance.png  -- the sorted k-distance graph with the knee/elbow that
     picks eps (measured on a blobs dataset; knee detected by max curvature).
  4. dbscan_varying.png    -- the failure mode: one global eps cannot separate a
     DENSE and a SPARSE cluster -- small eps shatters the sparse one into noise,
     large eps merges them. Two measured panels.

Run with: /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_dbscan_diagrams.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.neighbors import NearestNeighbors

OUT = os.path.join(os.path.dirname(__file__), "..", "04. Unsupervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
PALETTE = [BLUE, GREEN, PURPLE, AMBER, NAVY]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def taxonomy():
    """Hand-laid measured 2-D point set; classify core/border/noise at eps, minPts=4."""
    eps, min_pts = 1.0, 4
    # A dense blob of CORES + a fringe BORDER reachable from a core + one NOISE point.
    pts = np.array([
        [2.0, 2.0],   # 0 core   (5 in eps)
        [2.6, 2.2],   # 1 core
        [2.3, 2.7],   # 2 core
        [1.6, 2.6],   # 3 core
        [1.7, 1.6],   # 4 core
        [3.3, 2.5],   # 5 border (in core 1's eps, but sparse itself)
        [6.0, 5.0],   # 6 noise  (far from everything)
    ])
    nbrs = NearestNeighbors(radius=eps).fit(pts)
    neigh = nbrs.radius_neighbors(pts, return_distance=False)
    counts = np.array([len(n) for n in neigh])          # includes self
    is_core = counts >= min_pts
    is_border = np.array([
        (not is_core[i]) and any(is_core[j] for j in neigh[i] if j != i)
        for i in range(len(pts))])
    is_noise = ~is_core & ~is_border

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    # eps circle around a representative core (filled) and around the border (outline)
    core0 = np.where(is_core)[0][0]
    ax.add_patch(Circle(pts[core0], eps, fill=True, fc=BLUE, ec=BLUE, alpha=0.10, lw=1.4, ls="--"))
    border_idx = np.where(is_border)[0]
    if len(border_idx):
        ax.add_patch(Circle(pts[border_idx[0]], eps, fill=False, ec=AMBER, alpha=0.8, lw=1.4, ls="--"))

    def plot(mask, color, label, marker="o", s=210, edge="white"):
        if mask.any():
            ax.scatter(pts[mask, 0], pts[mask, 1], c=color, s=s, marker=marker,
                       edgecolors=edge, linewidths=1.4, label=label, zorder=3)
    plot(is_core, BLUE, f"core  (≥ minPts={min_pts} in ε)")
    plot(is_border, AMBER, "border  (in a core's ε, not core)")
    plot(is_noise, RED, "noise  (neither)", marker="X", s=230)

    for i, (x, y) in enumerate(pts):
        ax.annotate(f"p{i}\n({counts[i]})", (x, y), textcoords="offset points",
                    xytext=(0, 17), ha="center", fontsize=8.5, color=SLATE)
    ax.annotate("ε", (pts[core0, 0] - eps * 0.74, pts[core0, 1] + eps * 0.30),
                fontsize=14, color=BLUE, fontweight="bold")
    ax.set_title(f"Core / border / noise at ε={eps}, minPts={min_pts}  "
                 "(number in () = points within ε, incl. self)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 7.5); ax.set_ylim(0, 6.5); ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/dbscan_taxonomy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote dbscan_taxonomy.png | core:", is_core.sum(),
          "border:", is_border.sum(), "noise:", is_noise.sum(), "| counts:", counts.tolist())


def vs_kmeans():
    """DBSCAN vs k-means on two non-convex datasets. Measured labels."""
    Xm, _ = make_moons(n_samples=300, noise=0.09, random_state=0)
    Xc, _ = make_circles(n_samples=300, noise=0.06, factor=0.5, random_state=0)
    datasets = [("two moons", Xm, 0.18, 5, 2), ("concentric circles", Xc, 0.16, 5, 2)]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.4))
    for row, (name, X, eps, mp, k) in enumerate(datasets):
        db = DBSCAN(eps=eps, min_samples=mp).fit(X)
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        # DBSCAN panel
        axd = axes[row, 0]
        lbl = db.labels_
        for c in sorted(set(lbl)):
            m = lbl == c
            if c == -1:
                axd.scatter(X[m, 0], X[m, 1], c=RED, marker="X", s=28, label="noise", edgecolors="none")
            else:
                axd.scatter(X[m, 0], X[m, 1], c=PALETTE[c % len(PALETTE)], s=26, edgecolors="none")
        n_noise = int((lbl == -1).sum()); n_clust = len(set(lbl)) - (1 if -1 in lbl else 0)
        axd.set_title(f"DBSCAN — {name}\n{n_clust} clusters, {n_noise} noise (shapes recovered)",
                      fontsize=10.5, fontweight="bold", color=GREEN)
        # k-means panel
        axk = axes[row, 1]
        for c in range(k):
            m = km.labels_ == c
            axk.scatter(X[m, 0], X[m, 1], c=PALETTE[c % len(PALETTE)], s=26, edgecolors="none")
        axk.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c="black",
                    marker="*", s=160, edgecolors="white", linewidths=1.0)
        axk.set_title(f"k-means (k={k}) — {name}\nstraight cut through the shapes (wrong)",
                      fontsize=10.5, fontweight="bold", color=RED)
        for ax in (axd, axk):
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); _despine(ax)
        if row == 0 and (db.labels_ == -1).any():
            axd.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("DBSCAN finds arbitrary shapes and flags noise; k-means assumes spherical blobs",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/dbscan_vs_kmeans.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dbscan_vs_kmeans.png")


def kdistance():
    """Sorted k-distance graph; knee = eps. Measured (k = minPts)."""
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    min_pts = 4                       # k = minPts; distance to the minPts-th neighbor
    nbrs = NearestNeighbors(n_neighbors=min_pts).fit(X)
    dist, _ = nbrs.kneighbors(X)
    kdist = np.sort(dist[:, -1])      # distance to the (minPts)-th nearest neighbor, ascending
    # knee via max distance from the chord between the two endpoints
    n = len(kdist); idx = np.arange(n)
    p1 = np.array([0, kdist[0]]); p2 = np.array([n - 1, kdist[-1]])
    line = p2 - p1; line = line / np.linalg.norm(line)
    pts = np.stack([idx, kdist], axis=1) - p1
    proj = pts - np.outer(pts @ line, line)
    knee = int(np.argmax(np.linalg.norm(proj, axis=1)))
    eps_pick = kdist[knee]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(idx, kdist, color=BLUE, lw=2.4)
    ax.axhline(eps_pick, color=RED, ls="--", lw=1.6)
    ax.scatter([knee], [eps_pick], color=RED, s=90, zorder=4)
    ax.annotate(f"knee → ε ≈ {eps_pick:.2f}",
                (knee, eps_pick), textcoords="offset points", xytext=(-10, 28),
                ha="right", fontsize=11, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel(f"points sorted by distance to their {min_pts}-th nearest neighbor")
    ax.set_ylabel(f"{min_pts}-NN distance")
    ax.set_title("k-distance graph: the elbow marks where density drops — read ε off the y-axis",
                 fontsize=11.5, fontweight="bold")
    _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/dbscan_kdistance.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote dbscan_kdistance.png | eps_pick={eps_pick:.3f}")
    return eps_pick


def varying():
    """Varying-density failure: one global eps can't fit a dense + a sparse cluster."""
    rng = np.random.default_rng(1)
    dense = rng.normal([0, 0], 0.30, size=(150, 2))     # tight cluster
    sparse = rng.normal([4.5, 0], 1.05, size=(80, 2))   # loose cluster, far away
    X = np.vstack([dense, sparse])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    for ax, eps, tag in [(axes[0], 0.35, "small ε (fits the dense one)"),
                         (axes[1], 1.10, "large ε (fits the sparse one)")]:
        lbl = DBSCAN(eps=eps, min_samples=5).fit(X).labels_
        for c in sorted(set(lbl)):
            m = lbl == c
            if c == -1:
                ax.scatter(X[m, 0], X[m, 1], c=RED, marker="X", s=24, edgecolors="none")
            else:
                ax.scatter(X[m, 0], X[m, 1], c=PALETTE[c % len(PALETTE)], s=24, edgecolors="none")
        n_noise = int((lbl == -1).sum()); n_clust = len(set(lbl)) - (1 if -1 in lbl else 0)
        ax.set_title(f"ε={eps}: {n_clust} cluster(s), {n_noise} noise\n{tag}",
                     fontsize=10.5, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); _despine(ax)
    fig.suptitle("Varying density breaks DBSCAN: no single ε fits a dense AND a sparse cluster "
                 "(→ HDBSCAN)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/dbscan_varying.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dbscan_varying.png")


if __name__ == "__main__":
    taxonomy()
    vs_kmeans()
    kdistance()
    varying()
    print("OUT:", OUT)
