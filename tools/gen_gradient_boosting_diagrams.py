"""Gradient-boosting concept-page diagrams (muted palette, parallel scale). REAL fits.

Two figures for 03. Supervised_Learning/concepts/10-Gradient-Boosting-XGBoost.md:
  1. gb_stages.png -- boosting fits a 1D curve progressively: the ensemble after
     1, 5, and 50 shallow trees converges to the true function (each tree fits the
     residual of the last).
  2. gb_lr.png -- test error vs number of trees for two learning rates: a large LR
     overfits fast, a small LR (shrinkage) generalizes better but needs more trees.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _boost(x, y, n_trees, lr=0.3, depth=3):
    """Gradient boosting from scratch (MSE loss): yield the ensemble prediction."""
    F = np.full_like(y, y.mean()); trees = []
    snaps = {}
    for m in range(1, n_trees + 1):
        r = y - F                                   # pseudo-residual = -gradient of MSE
        t = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(x[:, None], r)
        F = F + lr * t.predict(x[:, None]); trees.append(t)
        snaps[m] = F.copy()
    return snaps


def gb_stages():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-3, 3, 120)); y = np.sin(1.5 * x) + rng.normal(0, 0.18, len(x))
    snaps = _boost(x, y, 50, lr=0.3)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for ax, m in zip(axes, [1, 5, 50]):
        ax.scatter(x, y, color=SLATE, s=14, alpha=0.5)
        ax.plot(x, np.sin(1.5 * x), color=GREEN, lw=2, ls="--", label="true function")
        ax.plot(x, snaps[m], color=RED, lw=2.4, label=f"ensemble ({m} tree{'s' if m>1 else ''})")
        ax.set_title(f"after {m} tree{'s' if m>1 else ''}", fontsize=12, fontweight="bold")
        ax.set_xlabel("x"); ax.legend(frameon=False, fontsize=8.5, loc="upper center"); _despine(ax)
    axes[0].set_ylabel("y")
    fig.suptitle("Gradient boosting: each tree fits the residual; the ensemble converges to the signal",
                 fontsize=13.5, fontweight="bold", y=1.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/gb_stages.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote gb_stages.png")


def gb_lr():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(-3, 3, 150)); y = np.sin(1.5 * x) + rng.normal(0, 0.25, len(x))
    xte = np.sort(rng.uniform(-3, 3, 400)); yte = np.sin(1.5 * xte)        # clean test signal
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    N = 120
    for lr, col, lab in [(1.0, RED, "learning rate = 1.0 (fast → overfits)"),
                         (0.1, GREEN, "learning rate = 0.1 (shrinkage → generalizes)")]:
        F = np.full_like(y, y.mean()); Fte = np.full_like(yte, y.mean()); errs = []
        for m in range(N):
            r = y - F
            t = DecisionTreeRegressor(max_depth=3, random_state=0).fit(x[:, None], r)
            F = F + lr * t.predict(x[:, None]); Fte = Fte + lr * t.predict(xte[:, None])
            errs.append(np.mean((yte - Fte) ** 2))
        ax.plot(range(1, N + 1), errs, color=col, lw=2.3, label=lab)
        best = int(np.argmin(errs)) + 1
        ax.scatter([best], [min(errs)], color=col, s=45, zorder=5)
    ax.set_xlabel("number of trees (boosting rounds)"); ax.set_ylabel("test error (MSE)")
    ax.set_title("Shrinkage: a small learning rate needs more trees but generalizes better",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5); _despine(ax); ax.set_ylim(0, 0.35)
    fig.tight_layout(); fig.savefig(f"{OUT}/gb_lr.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote gb_lr.png")


if __name__ == "__main__":
    gb_stages()
    gb_lr()
    print("OUT:", OUT)
