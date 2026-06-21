"""Anomaly / outlier-detection concept-page diagrams (muted palette, parallel scale).

Four MEASURED figures for 04. Unsupervised_Learning/concepts/09-Anomaly-Outlier-Detection.md:
  1. anom_decision_surfaces.png -- IsolationForest vs LOF vs OneClassSVM decision
     surfaces on the SAME 2-D data with planted outliers. PR-AUC printed per panel.
  2. anom_lof_local.png        -- a dense + a sparse cluster with a "local" outlier
     next to the dense one: a single global distance threshold misses it; LOF (a
     density RATIO) flags it. The point's LOF is printed.
  3. anom_mahalanobis.png      -- Mahalanobis vs Euclidean contours on correlated
     data: the same Euclidean distance maps to very different anomaly scores once
     the covariance is accounted for. The chi-square threshold ellipse is drawn.
  4. anom_isolation_paths.png  -- the isolation intuition: an outlier is isolated in
     a few random axis-parallel splits; a normal interior point needs many. Measured
     average path lengths over random trees are printed.

All numbers are computed at run time and printed to stdout so the page text can
quote them and they can be reproduced. Run with the project Python 3.12 env.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import average_precision_score

OUT = os.path.join(os.path.dirname(__file__), "..", "04. Unsupervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _planted_data(seed=7):
    """Two normal Gaussian blobs (inliers) + a sprinkle of uniform outliers."""
    rng = np.random.default_rng(seed)
    a = rng.normal([-2.0, -2.0], 0.6, size=(110, 2))
    b = rng.normal([2.2, 2.0], 0.7, size=(110, 2))
    inliers = np.vstack([a, b])
    outliers = rng.uniform(-6, 6, size=(20, 2))
    X = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(len(inliers)), np.ones(len(outliers))].astype(int)  # 1 = outlier
    return X, y, rng


def decision_surfaces():
    X, y, _ = _planted_data()
    xx, yy = np.meshgrid(np.linspace(-7, 7, 400), np.linspace(-7, 7, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]

    iforest = IsolationForest(n_estimators=300, contamination=0.13, random_state=0).fit(X)
    ocsvm = OneClassSVM(nu=0.13, gamma=0.10).fit(X)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.13, novelty=True).fit(X)

    models = [("Isolation Forest", iforest, BLUE),
              ("Local Outlier Factor", lof, PURPLE),
              ("One-Class SVM", ocsvm, GREEN)]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    aps = {}
    for ax, (name, mdl, col) in zip(axes, models):
        # higher score = MORE anomalous (sklearn's score_samples: high = inlier, so negate)
        anomaly_score = -mdl.score_samples(X)
        ap = average_precision_score(y, anomaly_score)
        aps[name] = ap
        Z = mdl.decision_function(grid).reshape(xx.shape)  # >0 inlier, <0 outlier
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 8), colors=[RED], alpha=0.08)
        ax.contour(xx, yy, Z, levels=[0], colors=[col], linewidths=2.2)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=14, color=SLATE, alpha=0.7, label="inlier")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=46, color=RED, edgecolor="white",
                   linewidth=0.6, label="planted outlier", marker="D")
        ax.set_title(f"{name}\nPR-AUC = {ap:.3f}", fontsize=12, fontweight="bold", color=col)
        ax.set_xlim(-7, 7); ax.set_ylim(-7, 7); _despine(ax)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    fig.suptitle("Same data, three detectors: the learned boundary around 'normal' (solid line) "
                 "and PR-AUC at catching the planted outliers",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/anom_decision_surfaces.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote anom_decision_surfaces.png  PR-AUC:", {k: round(v, 3) for k, v in aps.items()})
    return aps


def lof_local():
    """Dense cluster + sparse cluster + a point that is a LOCAL outlier near the dense one."""
    rng = np.random.default_rng(3)
    dense = rng.normal([0, 0], 0.30, size=(60, 2))     # tight
    sparse = rng.normal([5.0, 0], 1.05, size=(35, 2))  # loose
    local_out = np.array([[1.2, 0.0]])                 # near dense, but in its sparse fringe
    X = np.vstack([dense, sparse, local_out])
    idx = len(X) - 1

    lof = LocalOutlierFactor(n_neighbors=20)
    lof.fit_predict(X)
    scores = -lof.negative_outlier_factor_   # LOF >~1.5 typically anomalous
    lof_local = scores[idx]

    # the naive global rule: distance to nearest neighbour
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    d, _ = tree.query(X, k=2)
    nn_dist = d[:, 1]
    # a sparse-cluster inlier with LARGER nn distance than the local outlier
    sparse_nn_max = nn_dist[60:60 + 35].max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.6))
    # left: the data + the local outlier
    ax1.scatter(dense[:, 0], dense[:, 1], s=18, color=BLUE, alpha=0.8, label="dense cluster")
    ax1.scatter(sparse[:, 0], sparse[:, 1], s=18, color=GREEN, alpha=0.8, label="sparse cluster")
    ax1.scatter(*local_out[0], s=150, color=RED, edgecolor="white", marker="*",
                linewidth=0.8, zorder=5, label="local outlier")
    ax1.annotate(f"LOF = {lof_local:.2f}\n(flagged)", local_out[0], (1.8, 1.4),
                 fontsize=10, fontweight="bold", color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED))
    ax1.set_title("A LOCAL outlier: loose vs the dense cluster,\nbut closer than a sparse-cluster inlier",
                  fontsize=11.5, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5); _despine(ax1)
    ax1.set_xlabel("x1"); ax1.set_ylabel("x2")

    # right: why a global nn-distance threshold fails
    order = np.argsort(nn_dist)
    ax2.bar(range(len(X)), nn_dist[order], color=SLATE, alpha=0.6, width=1.0)
    rank_local = np.where(order == idx)[0][0]
    ax2.bar([rank_local], [nn_dist[idx]], color=RED, width=2.5)
    ax2.axhline(sparse_nn_max, color=GREEN, ls="--", lw=1.6,
                label=f"a sparse INLIER's nn-dist = {sparse_nn_max:.2f}")
    ax2.annotate(f"local outlier's nn-dist = {nn_dist[idx]:.2f}\n(SMALLER → a global threshold misses it)",
                 (rank_local, nn_dist[idx]), (len(X) * 0.18, sparse_nn_max * 0.78),
                 fontsize=9.5, fontweight="bold", color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED))
    ax2.set_title("Global distance threshold fails:\nthe outlier's nn-distance is small",
                  fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("points sorted by nearest-neighbour distance"); ax2.set_ylabel("nn distance")
    ax2.legend(loc="center left", fontsize=8.5); _despine(ax2)
    fig.suptitle("Why LOF's local density RATIO beats a single global distance threshold",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/anom_lof_local.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote anom_lof_local.png  LOF(local outlier)={lof_local:.2f}  "
          f"its nn-dist={nn_dist[idx]:.2f}  sparse-inlier nn-dist={sparse_nn_max:.2f}")
    return lof_local, nn_dist[idx], sparse_nn_max


def mahalanobis():
    rng = np.random.default_rng(1)
    mean = np.array([0.0, 0.0])
    cov = np.array([[3.0, 2.4], [2.4, 3.0]])           # strong positive correlation
    X = rng.multivariate_normal(mean, cov, size=600)
    Sinv = np.linalg.inv(cov)

    # two test points at the SAME Euclidean distance from the mean
    r = 3.2
    p_along = np.array([r / np.sqrt(2), r / np.sqrt(2)])    # along the correlation axis -> normal
    p_across = np.array([r / np.sqrt(2), -r / np.sqrt(2)])  # across it -> anomalous

    def md(p):
        return float(np.sqrt(p @ Sinv @ p))
    eucl = np.hypot(*p_along)
    md_along, md_across = md(p_along), md(p_across)
    thresh = np.sqrt(5.991)  # chi-square_2, 95%

    # contour grid of Mahalanobis distance
    xx, yy = np.meshgrid(np.linspace(-7, 7, 300), np.linspace(-7, 7, 300))
    pts = np.c_[xx.ravel(), yy.ravel()]
    mdist = np.sqrt(np.einsum("ij,jk,ik->i", pts, Sinv, pts)).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9.4, 8.4))
    ax.scatter(X[:, 0], X[:, 1], s=8, color=SLATE, alpha=0.30)
    cs = ax.contour(xx, yy, mdist, levels=[1, 2, 3], colors=[PURPLE],
                    linewidths=1.1, linestyles=":")
    ax.clabel(cs, fmt="MD=%.0f", fontsize=8, inline_spacing=2)
    # the 95% chi-square ellipse
    ax.contour(xx, yy, mdist, levels=[thresh], colors=[RED], linewidths=2.6)
    # equal-Euclidean circle through the two test points
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(th), r * np.sin(th), color=BLUE, lw=1.8, ls="--",
            label=f"equal Euclidean dist = {eucl:.2f}")
    ax.scatter(*p_along, s=150, color=GREEN, edgecolor="white", zorder=6)
    ax.annotate(f"ALONG axis\nEuclid = {eucl:.2f}\nMahal = {md_along:.2f}  (inlier)",
                p_along, (4.0, 4.6), fontsize=10.5, fontweight="bold", color=GREEN,
                ha="left", arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
    ax.scatter(*p_across, s=150, color=RED, marker="D", edgecolor="white", zorder=6)
    ax.annotate(f"ACROSS axis\nEuclid = {eucl:.2f}\nMahal = {md_across:.2f}  (outlier)",
                p_across, (-6.6, -5.2), fontsize=10.5, fontweight="bold", color=RED,
                ha="left", arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title("Same Euclidean distance, opposite verdicts\n"
                 f"(red ellipse = 95% chi-square boundary, MD = {thresh:.2f})",
                 fontsize=12.5, fontweight="bold")
    ax.set_xlim(-7.5, 7.5); ax.set_ylim(-7.5, 7.5); ax.set_aspect("equal"); _despine(ax)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(f"{OUT}/anom_mahalanobis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote anom_mahalanobis.png  Euclid(both)={eucl:.2f}  "
          f"MD_along={md_along:.2f}  MD_across={md_across:.2f}  chi2_thresh={thresh:.2f}")
    return eucl, md_along, md_across, thresh


def isolation_paths():
    """Measure expected isolation depth for an outlier vs an interior point."""
    rng = np.random.default_rng(5)
    normal = rng.normal(0, 1.0, size=(200, 2))
    X = np.vstack([normal, [[6.0, 6.0]]])
    out_idx = len(X) - 1
    centre_idx = int(np.argmin(np.hypot(X[:, 0], X[:, 1])))

    def isolate_depth(point_idx, n_trees=400):
        depths = []
        for _ in range(n_trees):
            idx = rng.choice(len(X), size=min(128, len(X)), replace=False)
            sub = X[idx]
            mask = np.ones(len(sub), bool)
            here = np.where(idx == point_idx)[0]
            if len(here) == 0:
                continue
            pos = here[0]
            depth = 0
            cur = mask.copy()
            while cur.sum() > 1:
                rows = np.where(cur)[0]
                lo = sub[rows].min(0); hi = sub[rows].max(0)
                feat = rng.integers(0, 2)
                if hi[feat] - lo[feat] < 1e-12:
                    break
                split = rng.uniform(lo[feat], hi[feat])
                go_left = sub[:, feat] < split
                side = go_left[pos]
                cur = cur & (go_left if side else ~go_left)
                depth += 1
                if depth > 60:
                    break
            depths.append(depth)
        return float(np.mean(depths))

    d_out = isolate_depth(out_idx)
    d_centre = isolate_depth(centre_idx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    ax1.scatter(normal[:, 0], normal[:, 1], s=14, color=SLATE, alpha=0.6, label="normal")
    ax1.scatter(*X[centre_idx], s=130, color=BLUE, edgecolor="white", zorder=5,
                label=f"interior pt (avg depth {d_centre:.1f})")
    ax1.scatter(*X[out_idx], s=150, color=RED, marker="D", edgecolor="white", zorder=5,
                label=f"outlier (avg depth {d_out:.1f})")
    ax1.set_title("Random axis splits isolate the outlier fast", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.5); _despine(ax1)
    ax1.set_xlabel("x1"); ax1.set_ylabel("x2")

    ax2.bar(["outlier", "interior point"], [d_out, d_centre], color=[RED, BLUE], width=0.55)
    for i, v in enumerate([d_out, d_centre]):
        ax2.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylabel("average isolation depth (over 400 random trees)")
    ax2.set_title(f"Shorter path = more anomalous\n(outlier isolated in ~{d_out:.1f} splits, "
                  f"interior needs ~{d_centre:.1f})", fontsize=11.5, fontweight="bold")
    _despine(ax2)
    fig.suptitle("The isolation intuition: anomalies need FEWER random splits to isolate",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/anom_isolation_paths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote anom_isolation_paths.png  outlier_depth={d_out:.2f}  interior_depth={d_centre:.2f}")
    return d_out, d_centre


if __name__ == "__main__":
    decision_surfaces()
    lof_local()
    mahalanobis()
    isolation_paths()
    print("OUT:", OUT)
