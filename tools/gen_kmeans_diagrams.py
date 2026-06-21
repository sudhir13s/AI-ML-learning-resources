"""K-Means concept-page diagrams (muted palette, parallel scale). REAL measured.

Four figures for 04. Unsupervised_Learning/concepts/01-K-Means-Clustering.md:
  1. kmeans_lloyd_iters.png  -- Lloyd's algorithm: assignments + centroids at
     iter 0 (init), iter 1, and convergence, with the inertia J printed per panel
     so you SEE J decrease monotonically.
  2. kmeans_elbow_silhouette.png -- the elbow (inertia vs k) and silhouette (score
     vs k) curves on a 4-blob dataset; both agree k=4 is the right choice.
  3. kmeans_failure.png  -- k-means fails on non-spherical structure: two moons and
     anisotropic blobs, true labels vs k-means partition side by side.
  4. kmeans_init.png  -- k-means++ vs random init: distribution of final inertia
     over many seeds on a seed-sensitive layout (k-means++ is tighter + lower).

All numbers are computed here so the page can quote them exactly. Run with:
  ~/.uv/envs/ml-py312/bin/python3 tools/gen_kmeans_diagrams.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score

OUT = os.path.join(os.path.dirname(__file__), "..", "04. Unsupervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
CLUSTER_COLORS = [BLUE, GREEN, AMBER, RED, PURPLE, NAVY]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _inertia(X, labels, centers):
    return float(sum(((X[labels == c] - centers[c]) ** 2).sum() for c in range(len(centers))))


def lloyd_iters():
    """Show Lloyd's loop: init -> 1 step -> converged, with J printed each panel."""
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.05, random_state=7)
    k = 4
    rng = np.random.default_rng(3)
    # A deliberately mediocre random init so the iterations visibly move.
    centers = X[rng.choice(len(X), k, replace=False)].copy()

    snaps = []  # (labels, centers, J, title)
    # iter 0: assignment under the initial centroids
    for it in range(40):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        J = _inertia(X, labels, centers)
        if it in (0, 1):
            snaps.append((labels.copy(), centers.copy(), J, f"Iteration {it}: assign + J"))
        new = np.array([X[labels == c].mean(0) if (labels == c).any() else centers[c]
                        for c in range(k)])
        if np.allclose(new, centers):
            snaps.append((labels.copy(), new.copy(), _inertia(X, labels, new),
                          f"Converged (iter {it}): J"))
            centers = new
            break
        centers = new

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (labels, cen, J, title) in zip(axes, snaps):
        for c in range(k):
            pts = X[labels == c]
            ax.scatter(pts[:, 0], pts[:, 1], s=14, color=CLUSTER_COLORS[c], alpha=0.55)
        ax.scatter(cen[:, 0], cen[:, 1], s=240, marker="X",
                   edgecolor="white", linewidth=1.8,
                   color=[CLUSTER_COLORS[c] for c in range(k)], zorder=5)
        ax.set_title(f"{title} = {J:,.0f}", fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Lloyd's algorithm: each assign+update step strictly lowers inertia J until it stops moving",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/kmeans_lloyd_iters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote kmeans_lloyd_iters.png  | J per panel:", [round(s[2]) for s in snaps])
    return [round(s[2]) for s in snaps]


def elbow_silhouette():
    """Elbow (inertia) + silhouette vs k on a clean 4-blob dataset."""
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)
    ks = list(range(1, 9))
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_) if k >= 2 else np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax1.plot(ks, inertias, color=BLUE, lw=2.4, marker="o", ms=6)
    ax1.scatter([4], [inertias[3]], s=220, facecolor="none", edgecolor=RED, linewidth=2.4, zorder=5)
    ax1.annotate("elbow at k=4\n(diminishing returns)", xy=(4, inertias[3]),
                 xytext=(5.1, inertias[3] + (max(inertias) - min(inertias)) * 0.28),
                 fontsize=10.5, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax1.set_xlabel("number of clusters  k"); ax1.set_ylabel("inertia  J (within-cluster SS)")
    ax1.set_title("Elbow method: J keeps falling, but the bend marks k", fontsize=12, fontweight="bold")
    _despine(ax1)

    best_k = ks[1:][int(np.nanargmax(sils[1:]))]
    ax2.plot(ks[1:], sils[1:], color=GREEN, lw=2.4, marker="o", ms=6)
    ax2.scatter([best_k], [max(sils[1:])], s=220, facecolor="none", edgecolor=RED, linewidth=2.4, zorder=5)
    ax2.annotate(f"peak at k={best_k}\n(s={max(sils[1:]):.3f})", xy=(best_k, max(sils[1:])),
                 xytext=(best_k + 0.6, max(sils[1:]) - 0.12),
                 fontsize=10.5, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax2.set_xlabel("number of clusters  k"); ax2.set_ylabel("mean silhouette score")
    ax2.set_title("Silhouette: a peak (not a bend) — here it agrees, k=4", fontsize=12, fontweight="bold")
    _despine(ax2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/kmeans_elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote kmeans_elbow_silhouette.png | inertias:", [round(i) for i in inertias],
          "| sils:", [round(s, 3) for s in sils[1:]], "| best_k:", best_k)
    return [round(i) for i in inertias], [round(s, 3) for s in sils[1:]], best_k


def failure_modes():
    """k-means fails on non-spherical / anisotropic structure."""
    # two moons
    Xm, ym = make_moons(n_samples=400, noise=0.06, random_state=0)
    km_m = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xm)
    # anisotropic blobs (stretched by a shear transform)
    Xb, yb = make_blobs(n_samples=400, centers=3, cluster_std=0.7, random_state=4)
    Xb = Xb @ np.array([[0.6, -0.65], [-0.4, 0.85]])
    km_b = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xb)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    panels = [
        (axes[0, 0], Xm, ym, "Two moons — TRUE clusters", 2),
        (axes[0, 1], Xm, km_m.labels_, "Two moons — k-means (WRONG: cuts across)", 2),
        (axes[1, 0], Xb, yb, "Anisotropic blobs — TRUE clusters", 3),
        (axes[1, 1], Xb, km_b.labels_, "Anisotropic blobs — k-means (WRONG: splits a stripe)", 3),
    ]
    for ax, X, lab, title, k in panels:
        for c in range(k):
            pts = X[lab == c]
            ax.scatter(pts[:, 0], pts[:, 1], s=12, color=CLUSTER_COLORS[c], alpha=0.6)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("k-means assumes round, equal-size clusters — it breaks on curved or stretched structure",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/kmeans_failure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote kmeans_failure.png")


def init_comparison():
    """k-means++ vs random init: final inertia distribution over many single-start runs."""
    # A seed-sensitive layout: well-separated blobs where a bad random seed can
    # merge two true clusters and split a third (a classic local-optimum trap).
    X, _ = make_blobs(n_samples=500, centers=6, cluster_std=0.85, random_state=11)
    k = 6
    n_trials = 60
    rand_in, pp_in = [], []
    for s in range(n_trials):
        rand_in.append(KMeans(n_clusters=k, init="random", n_init=1, random_state=s).fit(X).inertia_)
        pp_in.append(KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=s).fit(X).inertia_)
    rand_in, pp_in = np.array(rand_in), np.array(pp_in)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bins = np.linspace(min(rand_in.min(), pp_in.min()) * 0.98,
                       max(rand_in.max(), pp_in.max()) * 1.02, 28)
    ax.hist(rand_in, bins=bins, color=RED, alpha=0.6, label=f"random init (mean {rand_in.mean():.0f})")
    ax.hist(pp_in, bins=bins, color=GREEN, alpha=0.7, label=f"k-means++ (mean {pp_in.mean():.0f})")
    ax.axvline(pp_in.min(), color=NAVY, lw=1.8, ls="--", label=f"best found = {pp_in.min():.0f}")
    ax.set_xlabel("final inertia J (single start — lower is better)")
    ax.set_ylabel("number of seeds (of 60)")
    ax.set_title("k-means++ vs random init: smarter seeding lands near the optimum far more often",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/kmeans_init.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote kmeans_init.png | random mean={rand_in.mean():.0f} std={rand_in.std():.0f} "
          f"max={rand_in.max():.0f} | kmeans++ mean={pp_in.mean():.0f} std={pp_in.std():.0f} "
          f"max={pp_in.max():.0f} | best={pp_in.min():.0f}")
    return rand_in, pp_in


if __name__ == "__main__":
    lloyd_iters()
    elbow_silhouette()
    failure_modes()
    init_comparison()
    print("OUT:", OUT)
