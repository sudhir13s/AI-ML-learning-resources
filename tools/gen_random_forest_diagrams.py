"""Random-forest concept-page diagrams (muted palette, parallel scale).

Two figures for 03. Supervised_Learning/concepts/09-Random-Forests.md:
  1. rf_boundary.png -- single deep tree (jagged, high-variance boundary) vs a
     random forest (smooth boundary): averaging many trees cuts variance. REAL sklearn.
  2. rf_decorrelation.png -- the variance-of-an-average formula: Var = rho*sigma^2 +
     (1-rho)sigma^2/n. Independent trees (rho=0) -> variance -> 0; correlated trees
     hit a floor of rho*sigma^2. This is WHY feature-randomness (lower rho) matters.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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


def rf_boundary():
    X, y = make_moons(n_samples=300, noise=0.32, random_state=0)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.4, X[:, 0].max()+0.4, 400),
                         np.linspace(X[:, 1].min()-0.4, X[:, 1].max()+0.4, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    for ax, (model, title) in zip(axes, [
            (DecisionTreeClassifier(random_state=0), "single deep tree → jagged, high-variance"),
            (RandomForestClassifier(n_estimators=200, random_state=0), "random forest (200 trees) → smooth")]):
        model.fit(X, y)
        Z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.22, cmap=ListedColormap([BLUE, RED]))
        ax.contour(xx, yy, Z, levels=[0.5], colors=[SLATE], linewidths=1.4)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], color=BLUE, s=16, edgecolor="white", lw=0.3)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], color=RED, s=16, edgecolor="white", lw=0.3)
        ax.set_title(title, fontsize=11.5, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Averaging many trees cuts variance: the forest boundary is far smoother",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}/rf_boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote rf_boundary.png")


def rf_decorrelation():
    n = np.arange(1, 51)
    sigma2 = 1.0
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for rho, col, lab in [(0.0, GREEN, "ρ = 0  (independent → plain bagging ideal)"),
                          (0.3, BLUE, "ρ = 0.3  (decorrelated — random forest)"),
                          (0.7, RED, "ρ = 0.7  (correlated — bagging without feature randomness)")]:
        var = rho * sigma2 + (1 - rho) * sigma2 / n
        ax.plot(n, var, color=col, lw=2.4, label=lab)
        ax.axhline(rho * sigma2, color=col, ls=":", lw=1.2, alpha=0.6)
    ax.text(34, 0.72, "floor = ρσ²\n(correlated trees can't\naverage variance away)", color=RED, fontsize=9, fontweight="bold")
    ax.set_xlabel("number of trees in the ensemble  (n)"); ax.set_ylabel("variance of the averaged prediction")
    ax.set_title("Why feature-randomness matters: lower correlation ρ → lower variance floor",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5); _despine(ax); ax.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(f"{OUT}/rf_decorrelation.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote rf_decorrelation.png")


if __name__ == "__main__":
    rf_boundary()
    rf_decorrelation()
    print("OUT:", OUT)
