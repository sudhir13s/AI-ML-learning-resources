"""k-Nearest-Neighbors concept-page diagrams (muted palette, parallel scale). REAL measured.

Four figures for 03. Supervised_Learning/concepts/04-k-Nearest-Neighbors.md:
  1. knn_boundary.png       -- k=1 vs k=15 decision boundary: jagged/high-variance vs smooth/high-bias.
  2. knn_bias_variance.png  -- train vs test accuracy across k: the bias-variance U-curve (best k marked).
  3. knn_concentration.png  -- distance concentration: ratio nearest/farthest -> 1 as dimensionality grows.
  4. knn_scaling.png        -- effect of feature scaling: an unscaled large-range feature dominates the
                               distance, so the neighbor set (and prediction) flips after standardizing.

Run:  /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_knn_diagrams.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def knn_boundary():
    """k=1 (jagged, high variance) vs k=15 (smooth, high bias) on make_moons."""
    X, y = make_moons(n_samples=300, noise=0.30, random_state=3)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=0)
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    cm_bg = ListedColormap([BLUE, RED])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for ax, k in zip(axes, (1, 15)):
        clf = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.22, cmap=cm_bg)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[SLATE], linewidths=1.4)
        ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], c=BLUE, s=16, edgecolors="white", linewidths=0.4)
        ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], c=RED, s=16, edgecolors="white", linewidths=0.4)
        acc = clf.score(Xte, yte)
        label = "high variance (jagged)" if k == 1 else "high bias (smooth)"
        ax.set_title(f"k = {k}  —  {label}\ntest accuracy = {acc:.2f}", fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Small k overfits to noise; large k smooths the boundary — k is a bias–variance knob",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/knn_boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote knn_boundary.png")


def knn_bias_variance():
    """Train vs test accuracy across k -> the bias-variance U (error view)."""
    X, y = make_moons(n_samples=600, noise=0.30, random_state=3)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.40, random_state=0)
    ks = list(range(1, 80, 2))
    tr_err, te_err = [], []
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
        tr_err.append(1 - clf.score(Xtr, ytr))
        te_err.append(1 - clf.score(Xte, yte))
    best = ks[int(np.argmin(te_err))]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(ks, tr_err, color=BLUE, lw=2.2, marker="o", ms=3, label="training error")
    ax.plot(ks, te_err, color=RED, lw=2.4, marker="s", ms=3, label="test error")
    ax.axvline(best, color=GREEN, lw=1.6, ls="--")
    ax.text(best + 1.5, max(te_err) * 0.92, f"best k = {best}\n(lowest test error)",
            color=GREEN, fontsize=10, fontweight="bold")
    ax.annotate("small k: low bias,\nHIGH variance (overfit)", xy=(2, tr_err[0]),
                xytext=(8, max(te_err) * 0.55), fontsize=9.5, color=SLATE,
                arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.annotate("large k: HIGH bias,\nlow variance (underfit)", xy=(ks[-1], te_err[-1]),
                xytext=(46, max(te_err) * 0.30), fontsize=9.5, color=SLATE,
                arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.set_xlabel("number of neighbors  k   (model complexity ~ n/k  →  decreases to the right)")
    ax.set_ylabel("error rate (1 − accuracy)")
    ax.set_title("The bias–variance U: test error is lowest at an intermediate k",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=10); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/knn_bias_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote knn_bias_variance.png")


def knn_concentration():
    """Distance concentration: nearest/farthest -> 1 (left panel), and relative
    contrast (dmax-dmin)/dmin -> 0 (right panel). Averaged over many queries."""
    rng = np.random.default_rng(0)
    dims = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
    n, n_queries = 1000, 40
    ratio, contrast = [], []
    for d in dims:
        rr, cc = [], []
        for _ in range(n_queries):
            X = rng.uniform(0, 1, size=(n, d))
            q = rng.uniform(0, 1, size=(1, d))
            dist = np.sqrt(((X - q) ** 2).sum(axis=1))
            dmin, dmax = dist.min(), dist.max()
            rr.append(dmin / dmax)            # -> 1: near and far become equally far
            cc.append((dmax - dmin) / dmin)   # -> 0: relative contrast vanishes
        ratio.append(np.mean(rr)); contrast.append(np.mean(cc))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.7))
    ax1.plot(dims, ratio, color=BLUE, lw=2.4, marker="o", ms=5)
    ax1.axhline(1.0, color=SLATE, lw=1.2, ls=":")
    ax1.text(1.3, 1.01, "1.0 — all points equidistant", color=SLATE, fontsize=9.5, va="bottom")
    ax1.set_xscale("log"); ax1.set_ylim(0, 1.08)
    ax1.set_xlabel("dimensionality  d  (log scale)")
    ax1.set_ylabel("nearest distance / farthest distance")
    ax1.set_title("nearest / farthest  →  1\n(the nearest neighbor stops being special)",
                  fontsize=11, fontweight="bold")
    _despine(ax1)
    ax2.plot(dims, contrast, color=AMBER, lw=2.4, marker="^", ms=5)
    ax2.axhline(0.0, color=SLATE, lw=1.2, ls=":")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("dimensionality  d  (log scale)")
    ax2.set_ylabel("relative contrast  (dmax − dmin) / dmin")
    ax2.set_title("relative contrast  →  0\n(distance gap collapses vs the distance itself)",
                  fontsize=11, fontweight="bold")
    _despine(ax2)
    fig.suptitle("Curse of dimensionality: distances concentrate as d grows (measured, avg of 40 queries)",
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout(); fig.savefig(f"{OUT}/knn_concentration.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote knn_concentration.png  (d=2 ratio={ratio[1]:.3f}, "
                          f"d=1000 ratio={ratio[-1]:.3f})")


def knn_scaling():
    """Two features on wildly different scales: before scaling, the big-range feature owns the
    distance; standardizing rebalances and changes which points are the nearest neighbors."""
    rng = np.random.default_rng(7)
    # feature 1: age in years (20..60), feature 2: salary in dollars (20k..120k).
    # TRUE label depends on AGE only (young<40 -> 0, old>=40 -> 1); salary is a noise
    # dimension with a HUGE numeric range that will dominate raw Euclidean distance.
    age = rng.uniform(20, 60, 80)
    salary = rng.uniform(20000, 120000, 80)
    y = (age >= 40).astype(int)
    # Query: a 34-year-old (truly 'young' = class 0) who happens to earn ~70k.
    q = np.array([[34.0, 70000.0]])
    # To make the lesson bite, seed a band of OLD (class 1) workers at the query's
    # salary (~70k) but far away in age (50-58). In RAW space salary dominates, so
    # these salary-twins look 'nearest' and drag the vote to class 1 (WRONG). After
    # standardizing, the genuinely age-near young workers win and the vote flips to 0.
    band_age = rng.uniform(50, 58, 9)
    band_sal = rng.uniform(67000, 73000, 9)
    age = np.r_[age, band_age]; salary = np.r_[salary, band_sal]
    y = np.r_[y, np.ones(9, dtype=int)]
    X = np.c_[age, salary]

    def neighbors(Xs, qs, k=7):
        d = np.sqrt(((Xs - qs) ** 2).sum(axis=1))
        idx = np.argsort(d)[:k]
        return idx

    raw_idx = neighbors(X, q)
    sc = StandardScaler().fit(X)
    Xs, qs = sc.transform(X), sc.transform(q)
    sc_idx = neighbors(Xs, qs)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    # left: raw space (salary axis dominates)
    ax = axes[0]
    ax.scatter(age[y == 0], salary[y == 0], c=BLUE, s=22, label="young (0)")
    ax.scatter(age[y == 1], salary[y == 1], c=RED, s=22, label="old (1)")
    ax.scatter(age[raw_idx], salary[raw_idx], facecolors="none", edgecolors=GREEN,
               s=120, linewidths=1.8, label="7 nearest (raw)")
    ax.scatter(q[0, 0], q[0, 1], c=AMBER, marker="*", s=240, edgecolors="black",
               linewidths=0.6, label="query", zorder=5)
    vote_raw = int(round(y[raw_idx].mean()))
    ax.set_title(f"RAW features: salary (range ~100k) dominates\n→ neighbors pick on salary, vote = {vote_raw}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("age (years, 0–60)"); ax.set_ylabel("salary ($, 0–120k)")
    ax.legend(frameon=False, fontsize=8, loc="lower right"); _despine(ax)
    # right: standardized space
    ax = axes[1]
    ax.scatter(Xs[y == 0, 0], Xs[y == 0, 1], c=BLUE, s=22, label="young (0)")
    ax.scatter(Xs[y == 1, 0], Xs[y == 1, 1], c=RED, s=22, label="old (1)")
    ax.scatter(Xs[sc_idx, 0], Xs[sc_idx, 1], facecolors="none", edgecolors=GREEN,
               s=120, linewidths=1.8, label="7 nearest (scaled)")
    ax.scatter(qs[0, 0], qs[0, 1], c=AMBER, marker="*", s=240, edgecolors="black",
               linewidths=0.6, label="query", zorder=5)
    vote_sc = int(round(y[sc_idx].mean()))
    ax.set_title(f"STANDARDIZED: both features count equally\n→ neighbors pick on age, vote = {vote_sc}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("age (z-score)"); ax.set_ylabel("salary (z-score)")
    ax.legend(frameon=False, fontsize=8, loc="lower right"); _despine(ax)
    fig.suptitle("Feature scaling is mandatory: an unscaled large-range feature hijacks the distance",
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout(); fig.savefig(f"{OUT}/knn_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote knn_scaling.png")
    print(f"   raw neighbors vote={vote_raw}  scaled neighbors vote={vote_sc}  "
          f"(neighbor sets differ: {set(raw_idx.tolist()) != set(sc_idx.tolist())})")


if __name__ == "__main__":
    knn_boundary()
    knn_bias_variance()
    knn_concentration()
    knn_scaling()
    print("OUT:", OUT)
