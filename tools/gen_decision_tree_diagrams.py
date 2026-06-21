"""Decision-tree concept-page diagrams (muted palette, parallel scale).

Figures for 03. Supervised_Learning/concepts/07-Decision-Trees.md (REAL sklearn fits):
  1. dt_partition.png   -- a fitted tree carves the 2D feature space into axis-aligned
     rectangles (the tell-tale 'staircase' boundary).
  2. dt_overfit.png     -- train vs test accuracy as max_depth grows: train -> 100%
     while test peaks then declines (why a single deep tree is high-variance).
  3. dt_impurity.png    -- Gini vs entropy vs misclassification impurity for a binary
     node as the class proportion p sweeps 0..1 (why "best split" = lowest impurity).
  4. dt_regression.png  -- a regression tree as a piecewise-constant step function:
     each leaf predicts the mean of its interval (variance reduction). REAL sklearn fit.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def dt_partition():
    X, y = make_moons(n_samples=300, noise=0.28, random_state=0)
    clf = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.4, X[:, 0].max()+0.4, 400),
                         np.linspace(X[:, 1].min()-0.4, X[:, 1].max()+0.4, 400))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.contourf(xx, yy, Z, alpha=0.25, cmap=ListedColormap([BLUE, RED]))
    ax.contour(xx, yy, Z, levels=[0.5], colors=[SLATE], linewidths=1.6)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color=BLUE, s=22, edgecolor="white", lw=0.4, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color=RED, s=22, edgecolor="white", lw=0.4, label="class 1")
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.set_title("A decision tree carves the space into axis-aligned rectangles (the 'staircase')",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/dt_partition.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dt_partition.png")


def dt_overfit():
    X, y = make_moons(n_samples=400, noise=0.35, random_state=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=1)
    depths = range(1, 16)
    tr, te = [], []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        tr.append(clf.score(Xtr, ytr)); te.append(clf.score(Xte, yte))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(list(depths), tr, color=BLUE, lw=2.4, marker="o", ms=4, label="training accuracy → 100%")
    ax.plot(list(depths), te, color=RED, lw=2.4, marker="s", ms=4, label="test accuracy (peaks, then drops)")
    best = list(depths)[int(np.argmax(te))]
    ax.axvline(best, color=GREEN, ls="--", lw=2)
    ax.text(best+0.2, 0.6, f"sweet spot\n(depth {best})", color=GREEN, fontsize=9.5, fontweight="bold")
    ax.axvspan(best+0.5, 15, color=RED, alpha=0.05)
    ax.text((best+15)/2+1, 0.55, "overfitting:\ntree memorizes noise", ha="center", fontsize=9, color=RED)
    ax.set_xlabel("tree depth (max_depth)"); ax.set_ylabel("accuracy")
    ax.set_title("Why a single deep tree overfits: train → perfect, test gets worse",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="center right"); _despine(ax); ax.set_ylim(0.5, 1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/dt_overfit.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dt_overfit.png")


def dt_impurity():
    """The three classification impurity measures vs the proportion p of class 1 in a
    binary node. All are 0 at the pure ends (p=0, p=1) and peak at p=0.5; entropy peaks
    at 1 bit, Gini at 0.5, misclassification at 0.5. Entropy is scaled by 1/2 too so the
    shapes line up -- the point is the SHAPE, not the units."""
    p = np.linspace(1e-6, 1 - 1e-6, 500)
    gini = 1 - (p**2 + (1 - p)**2)                     # = 2p(1-p)
    ent = -(p*np.log2(p) + (1-p)*np.log2(1-p))         # bits, peak 1.0
    ent_half = ent / 2                                  # scaled to peak 0.5 for shape comparison
    misc = 1 - np.maximum(p, 1-p)                        # = min(p, 1-p)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(p, ent, color=PURPLE, lw=2.6, label="Entropy  (bits, peak 1.0)")
    ax.plot(p, ent_half, color=PURPLE, lw=1.6, ls=":", label="Entropy / 2  (shape vs Gini)")
    ax.plot(p, gini, color=BLUE, lw=2.6, label="Gini  =  2p(1−p)  (peak 0.5)")
    ax.plot(p, misc, color=RED, lw=2.6, label="Misclassification  (peak 0.5)")
    ax.axvline(0.5, color=SLATE, ls="--", lw=1.2)
    ax.text(0.5, 1.18, "p = 0.5  (max impurity)", ha="center", color=SLATE, fontsize=9)
    ax.text(0.015, 0.05, "pure\n(p=0)", color=GREEN, fontsize=9, fontweight="bold")
    ax.text(0.9, 0.05, "pure\n(p=1)", color=GREEN, fontsize=9, fontweight="bold")
    ax.annotate("Gini & entropy are smooth & strictly concave\n→ reward any move toward purity;\n"
                "misclassification is piecewise-linear\n(insensitive to within-leaf gains)",
                xy=(0.30, 0.42), xytext=(0.30, 0.66), ha="center", fontsize=8.5, color=NAVY,
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.0))
    ax.set_xlabel("proportion of class 1 in the node,  p")
    ax.set_ylabel("impurity")
    ax.set_title("Impurity measures for a binary node: Gini vs entropy vs misclassification",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right"); _despine(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.28)
    fig.tight_layout(); fig.savefig(f"{OUT}/dt_impurity.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dt_impurity.png")


def dt_regression():
    """A regression tree is a piecewise-constant step function: each leaf predicts the
    mean of the y-values whose x falls in that leaf's interval. Splits are placed to
    minimize within-leaf variance (= MSE). Shows a shallow (smooth) vs deep (overfit) fit."""
    rng = np.random.default_rng(3)
    X = np.sort(rng.uniform(0, 1, 80))[:, None]
    y = np.sin(2 * np.pi * X.ravel()) + rng.normal(0, 0.18, X.shape[0])
    grid = np.linspace(0, 1, 600)[:, None]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.scatter(X.ravel(), y, color=SLATE, s=20, alpha=0.6, label="training data (noisy sine)", zorder=2)
    for depth, col, lw, lab in [(2, GREEN, 2.8, "depth 2  (4 leaves — smooth)"),
                                (8, RED, 1.7, "depth 8  (many leaves — overfit)")]:
        reg = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X, y)
        ax.plot(grid.ravel(), reg.predict(grid), color=col, lw=lw, label=lab, zorder=3)
    ax.text(0.02, -1.15, "each flat segment = one leaf,\npredicting the MEAN of its interval\n"
            "(splits chosen to cut variance)", fontsize=8.5, color=NAVY)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Regression tree = piecewise-constant steps (each leaf predicts its interval mean)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper right"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/dt_regression.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dt_regression.png")


if __name__ == "__main__":
    dt_partition()
    dt_overfit()
    dt_impurity()
    dt_regression()
    print("OUT:", OUT)
