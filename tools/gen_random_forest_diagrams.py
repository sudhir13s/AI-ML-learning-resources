"""Random-forest concept-page diagrams (muted palette, parallel scale).

Three figures for 03. Supervised_Learning/concepts/09-Random-Forests.md:
  1. rf_boundary.png -- single deep tree (jagged, high-variance boundary) vs a
     random forest (smooth boundary): averaging many trees cuts variance. REAL sklearn.
  2. rf_decorrelation.png -- the variance-of-an-average formula: Var = rho*sigma^2 +
     (1-rho)sigma^2/n. Independent trees (rho=0) -> variance -> 0; correlated trees
     hit a floor of rho*sigma^2. This is WHY feature-randomness (lower rho) matters.
  3. rf_importance.png -- two MEASURED panels: (left) impurity vs permutation feature
     importance, exposing the high-cardinality bias (a pure-noise continuous feature
     looks important to impurity but ~0 to permutation); (right) test accuracy vs
     n_estimators rising then plateauing while train acc pins at 1.0 -- more trees
     never overfit, they only reduce variance toward the floor. REAL sklearn.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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


def rf_importance():
    """Two MEASURED panels: the high-cardinality importance bias, and n_estimators
    never overfitting. Both are real sklearn runs."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # --- LEFT: impurity vs permutation importance -> high-cardinality bias ---
    rng = np.random.default_rng(0)
    n = 2000
    x_pred = rng.integers(0, 2, n)                       # binary, genuinely predictive
    y = (x_pred ^ (rng.random(n) < 0.1)).astype(int)     # ~90% determined by x_pred
    x_noise = rng.random(n)                              # continuous noise, many split points
    X = np.c_[x_pred, x_noise]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
    imp = rf.feature_importances_
    perm = permutation_importance(rf, Xte, yte, n_repeats=30, random_state=0).importances_mean
    labels = ["predictive\n(binary)", "PURE NOISE\n(continuous)"]
    xpos = np.arange(2); w = 0.36
    axL.bar(xpos - w/2, imp, w, color=RED, label="impurity (sklearn default)")
    axL.bar(xpos + w/2, perm, w, color=GREEN, label="permutation (held-out)")
    axL.axhline(0, color=SLATE, lw=0.8)
    axL.set_xticks(xpos); axL.set_xticklabels(labels, fontsize=10)
    axL.set_ylabel("feature importance")
    axL.set_title("Impurity importance is fooled by a high-cardinality\nnoise feature; permutation is not",
                  fontsize=11.5, fontweight="bold")
    axL.legend(frameon=False, fontsize=9.5, loc="upper right")
    axL.annotate("impurity scores pure\nnoise as 'important'", xy=(1 - w/2, imp[1]),
                 xytext=(0.05, 0.52), fontsize=9, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    _despine(axL)

    # --- RIGHT: test acc vs n_estimators (rises, plateaus; train pinned ~1.0) ---
    X2, y2 = make_classification(n_samples=1500, n_features=8, n_informative=4,
                                 n_redundant=0, random_state=0)
    Xtr2, Xte2, ytr2, yte2 = train_test_split(X2, y2, test_size=0.3, random_state=0)
    ns = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 400]
    tr_acc, te_acc = [], []
    for k in ns:
        m = RandomForestClassifier(n_estimators=k, random_state=0).fit(Xtr2, ytr2)
        tr_acc.append(m.score(Xtr2, ytr2)); te_acc.append(m.score(Xte2, yte2))
    axR.plot(ns, tr_acc, "o-", color=SLATE, lw=2.0, ms=4, label="train accuracy")
    axR.plot(ns, te_acc, "o-", color=BLUE, lw=2.4, ms=4, label="test accuracy")
    axR.set_xscale("log")
    axR.set_xlabel("number of trees  (n_estimators, log scale)")
    axR.set_ylabel("accuracy")
    axR.set_title("More trees never overfit: test accuracy rises\nthen plateaus (no down-turn)",
                  fontsize=11.5, fontweight="bold")
    axR.legend(frameon=False, fontsize=9.5, loc="lower right")
    axR.set_ylim(0.82, 1.02); _despine(axR)
    axR.grid(axis="y", ls=":", alpha=0.4)

    fig.tight_layout(); fig.savefig(f"{OUT}/rf_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote rf_importance.png")


if __name__ == "__main__":
    rf_boundary()
    rf_decorrelation()
    rf_importance()
    print("OUT:", OUT)
