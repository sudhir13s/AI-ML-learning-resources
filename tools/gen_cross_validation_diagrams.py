"""Cross-validation concept-page diagrams (muted palette, parallel scale).

Two figures for 03. Supervised_Learning/concepts/13-Cross-Validation.md:
  1. cv_kfold.png  -- the 5-fold layout: each row is one iteration; the validation
     fold rotates so every point is used for both training and validation.
  2. cv_stability.png -- REAL measurement: 300 single random train/test splits give
     a noisy spread of scores; 5-fold CV gives one stable, lower-variance estimate.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def cv_kfold():
    k = 5
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for it in range(k):                                  # one row per iteration
        y = k - 1 - it
        for fold in range(k):
            is_val = (fold == it)
            ax.add_patch(Rectangle((fold, y), 0.94, 0.84,
                                   facecolor=RED if is_val else BLUE,
                                   alpha=0.85 if is_val else 0.55, edgecolor="white", lw=2))
            ax.text(fold + 0.47, y + 0.42, "validate" if is_val else "train",
                    ha="center", va="center", color="#fff", fontsize=9.5, fontweight="bold")
        ax.text(-0.25, y + 0.42, f"iter {it+1}", ha="right", va="center", fontsize=10, fontweight="bold", color=SLATE)
    ax.text(k/2, k + 0.25, "the dataset, split into 5 folds  →  rotate the validation fold",
            ha="center", fontsize=10.5, color=SLATE, fontweight="bold")
    ax.text(k/2, -0.7, "final score = average of the 5 validation scores (every point validated exactly once)",
            ha="center", fontsize=9.5, color=NAVY, fontweight="bold")
    ax.set_xlim(-1.4, k + 0.2); ax.set_ylim(-1.1, k + 0.6); ax.axis("off")
    ax.set_title("5-fold cross-validation: every point is used for both training and validation",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/cv_kfold.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cv_kfold.png")


def cv_stability():
    X, y = make_classification(n_samples=300, n_features=20, n_informative=6,
                               class_sep=0.8, random_state=0)
    single = []
    for s in range(300):                                 # many single random splits
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=s)
        single.append(LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte, yte))
    cv = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.hist(single, bins=24, color=SLATE, alpha=0.55, label=f"300 single splits (noisy: {np.std(single):.3f} std)")
    ax.axvline(np.mean(single), color=SLATE, ls=":", lw=1.5)
    ax.axvline(cv.mean(), color=GREEN, lw=2.6, label=f"5-fold CV estimate = {cv.mean():.3f} ± {cv.std():.3f}")
    for c in cv:
        ax.plot([c], [2], marker="v", color=RED, ms=9)
    ax.text(cv.mean(), ax.get_ylim()[1]*0.55, "the 5 fold scores (red ▾)\naverage to one stable number",
            color=RED, fontsize=9, fontweight="bold", ha="center")
    ax.set_xlabel("accuracy"); ax.set_ylabel("count (of single splits)")
    ax.set_title("Why k-fold: one split is a noisy estimate; averaging folds is far steadier",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/cv_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cv_stability.png")


if __name__ == "__main__":
    cv_kfold()
    cv_stability()
    print("OUT:", OUT)
