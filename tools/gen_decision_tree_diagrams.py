"""Decision-tree concept-page diagrams (muted palette, parallel scale).

Two figures for 03. Supervised_Learning/concepts/07-Decision-Trees.md (REAL sklearn fits):
  1. dt_partition.png -- a fitted tree carves the 2D feature space into axis-aligned
     rectangles (the tell-tale 'staircase' boundary).
  2. dt_overfit.png   -- train vs test accuracy as max_depth grows: train -> 100%
     while test peaks then declines (why a single deep tree is high-variance).
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
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


if __name__ == "__main__":
    dt_partition()
    dt_overfit()
    print("OUT:", OUT)
