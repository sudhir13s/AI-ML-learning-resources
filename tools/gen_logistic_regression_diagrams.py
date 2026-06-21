"""Logistic-regression concept-page diagrams (muted palette, parallel scale).

Two figures for 03. Supervised_Learning/concepts/02-Logistic-Regression.md:
  1. logreg_sigmoid.png -- the sigmoid maps the linear score z = w.x+b to a
     probability in (0,1); threshold at 0.5 (z=0) splits the classes.
  2. logreg_boundary.png -- a model fit from scratch on two 2D blobs: the LINEAR
     decision boundary with the sigmoid probability shaded behind it.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
sig = lambda z: 1 / (1 + np.exp(-z))


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def logreg_sigmoid():
    z = np.linspace(-6, 6, 400)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(z, sig(z), color=PURPLE, lw=3)
    ax.axhline(0.5, color=SLATE, ls=":", lw=1.3); ax.axvline(0, color=SLATE, ls=":", lw=1.3)
    ax.fill_between(z, 0, 1, where=(z > 0), color=GREEN, alpha=0.07)
    ax.fill_between(z, 0, 1, where=(z < 0), color=BLUE, alpha=0.07)
    ax.text(3.2, 0.18, "z > 0 → p > 0.5\npredict class 1", color=GREEN, fontsize=10, fontweight="bold", ha="center")
    ax.text(-3.2, 0.82, "z < 0 → p < 0.5\npredict class 0", color=BLUE, fontsize=10, fontweight="bold", ha="center")
    ax.scatter([0], [0.5], color=RED, s=70, zorder=6)
    ax.annotate("decision threshold\np = 0.5 at z = 0", (0, 0.5), textcoords="offset points",
                xytext=(20, -38), fontsize=9.5, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("linear score  z = w·x + b"); ax.set_ylabel("probability  p = σ(z)")
    ax.set_title("Logistic regression: squash a linear score into a probability",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.03, 1.03); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/logreg_sigmoid.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote logreg_sigmoid.png")


def logreg_boundary():
    rng = np.random.default_rng(1)
    X0 = rng.normal([-1.3, -0.6], 0.85, (120, 2))
    X1 = rng.normal([1.3, 0.8], 0.85, (120, 2))
    X = np.vstack([X0, X1]); y = np.concatenate([np.zeros(120), np.ones(120)])
    Xb = np.c_[np.ones(len(X)), X]                      # bias column
    w = np.zeros(3)
    for _ in range(3000):                               # gradient descent on log-loss
        p = sig(Xb @ w); w -= 0.05 * Xb.T @ (p - y) / len(y)
    # probability field
    xx, yy = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-3.5, 3.5, 300))
    grid = np.c_[np.ones(xx.size), xx.ravel(), yy.ravel()]
    P = sig(grid @ w).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    cf = ax.contourf(xx, yy, P, levels=20, cmap="RdBu_r", alpha=0.45, vmin=0, vmax=1)
    ax.contour(xx, yy, P, levels=[0.5], colors=[SLATE], linewidths=2.4)        # the boundary
    ax.scatter(X0[:, 0], X0[:, 1], color=BLUE, s=22, edgecolor="white", lw=0.4, label="class 0")
    ax.scatter(X1[:, 0], X1[:, 1], color=RED, s=22, edgecolor="white", lw=0.4, label="class 1")
    ax.text(0.05, 3.0, "decision boundary (p = 0.5) is a straight line:  w·x + b = 0",
            fontsize=9.5, color=SLATE, fontweight="bold", ha="center")
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046); cbar.set_label("P(class 1)")
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.set_title("The decision boundary is linear; the sigmoid shades the confidence",
                 fontsize=12.5, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/logreg_boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote logreg_boundary.png")


if __name__ == "__main__":
    logreg_sigmoid()
    logreg_boundary()
    print("OUT:", OUT)
