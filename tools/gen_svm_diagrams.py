"""SVM concept-page diagrams (muted palette, parallel scale). REAL sklearn fits.

Two figures for 03. Supervised_Learning/concepts/06-Support-Vector-Machines.md:
  1. svm_margin.png -- a linear SVM's maximum-margin 'street': the decision
     boundary, the two margin lines, and the circled support vectors that define it.
  2. svm_kernel.png -- an RBF-kernel SVM carving a non-linear boundary around
     concentric circles that no straight line could separate.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_circles
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


def svm_margin():
    X, y = make_blobs(n_samples=80, centers=2, cluster_std=1.05, random_state=6)
    clf = SVC(kernel="linear", C=10).fit(X, y)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color=BLUE, s=34, edgecolor="white", lw=0.5, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color=RED, s=34, edgecolor="white", lw=0.5, label="class 1")
    xx, yy = np.meshgrid(np.linspace(*ax.get_xlim(), 300), np.linspace(*ax.get_ylim(), 300))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[SLATE, NAVY, SLATE],
               linestyles=["--", "-", "--"], linewidths=[1.5, 2.4, 1.5])
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=160, facecolors="none", edgecolors=GREEN, linewidths=2.2,
               label="support vectors", zorder=5)
    margin = 2 / np.linalg.norm(clf.coef_)
    ax.text(0.5, 0.02, f"margin = 2/‖w‖ = {margin:.2f}  (maximized);  decision boundary (solid), margins (dashed)",
            transform=ax.transAxes, ha="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.set_title("SVM finds the widest 'street': maximum margin, defined by support vectors",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=8.5); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/svm_margin.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote svm_margin.png")


def svm_kernel():
    X, y = make_circles(n_samples=300, factor=0.4, noise=0.12, random_state=0)
    clf = SVC(kernel="rbf", C=10, gamma=1.0).fit(X, y)
    xx, yy = np.meshgrid(np.linspace(-1.6, 1.6, 400), np.linspace(-1.6, 1.6, 400))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.contourf(xx, yy, Z, alpha=0.22, cmap=ListedColormap([BLUE, RED]))
    ax.contour(xx, yy, Z, levels=[0.5], colors=[NAVY], linewidths=2.2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color=BLUE, s=26, edgecolor="white", lw=0.4, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color=RED, s=26, edgecolor="white", lw=0.4, label="class 1")
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.set_title("The kernel trick: an RBF SVM draws a curved boundary\nno straight line could",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9); _despine(ax); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(f"{OUT}/svm_kernel.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote svm_kernel.png")


if __name__ == "__main__":
    svm_margin()
    svm_kernel()
    print("OUT:", OUT)
