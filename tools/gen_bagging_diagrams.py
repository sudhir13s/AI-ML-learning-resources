"""Bagging concept-page diagrams (muted palette, parallel scale). REAL measured.

Two figures for 03. Supervised_Learning/concepts/08-Bagging.md:
  1. bag_variance.png -- prediction variance vs ensemble size B for a bagged deep
     tree: averaging B models cuts variance roughly like 1/B toward a floor.
  2. bag_unstable.png -- bagging helps UNSTABLE learners (deep tree) a lot, but
     barely helps STABLE ones (linear model) -- the key requirement for bagging.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _data(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, 120); y = np.sin(1.5 * x) + rng.normal(0, 0.3, 120)
    return x[:, None], y


def _pred_variance(make_model, x0, runs=60):
    preds = [make_model().fit(*_data(r)).predict(x0) for r in range(runs)]
    return np.var(preds, axis=0).mean()


def bag_variance():
    x0 = np.linspace(-2.5, 2.5, 40)[:, None]
    Bs = [1, 2, 3, 5, 10, 20, 40, 80]
    var = [_pred_variance(lambda B=B: BaggingRegressor(
        DecisionTreeRegressor(max_depth=None), n_estimators=B, random_state=0), x0) for B in Bs]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(Bs, var, color=BLUE, lw=2.4, marker="o", ms=5, label="bagged deep trees (measured)")
    ax.plot(Bs, [var[0] / B for B in Bs], color=SLATE, lw=1.6, ls="--", label="ideal 1/B (independent) reference")
    ax.set_xlabel("number of bagged models  B"); ax.set_ylabel("prediction variance")
    ax.set_title("Bagging cuts variance: averaging B models shrinks it toward a floor",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5); _despine(ax); ax.set_ylim(0, max(var) * 1.05)
    fig.tight_layout(); fig.savefig(f"{OUT}/bag_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote bag_variance.png")


def bag_unstable():
    x0 = np.linspace(-2.5, 2.5, 40)[:, None]
    tree_single = _pred_variance(lambda: DecisionTreeRegressor(max_depth=None), x0)
    tree_bag = _pred_variance(lambda: BaggingRegressor(DecisionTreeRegressor(max_depth=None), n_estimators=50, random_state=0), x0)
    lin_single = _pred_variance(lambda: LinearRegression(), x0)
    lin_bag = _pred_variance(lambda: BaggingRegressor(LinearRegression(), n_estimators=50, random_state=0), x0)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    groups = ["deep tree\n(UNSTABLE)", "linear model\n(STABLE)"]
    single = [tree_single, lin_single]; bagged = [tree_bag, lin_bag]
    xpos = np.arange(2); w = 0.35
    ax.bar(xpos - w/2, single, w, color=RED, alpha=0.85, label="single model")
    ax.bar(xpos + w/2, bagged, w, color=GREEN, alpha=0.85, label="bagged (50 models)")
    for i in range(2):
        change = (bagged[i]/single[i] - 1) * 100 if single[i] > 0 else 0   # neg = variance dropped
        note = "  (barely changes)" if abs(change) < 15 else ""
        ax.text(xpos[i] + w/2, bagged[i] + max(single)*0.02, f"{change:+.0f}%{note}", ha="center",
                fontsize=10, color=GREEN if change < -15 else SLATE, fontweight="bold")
    ax.set_xticks(xpos); ax.set_xticklabels(groups); ax.set_ylabel("prediction variance")
    ax.set_title("Bagging only helps UNSTABLE learners — it barely touches a stable model's variance",
                 fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/bag_unstable.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote bag_unstable.png")


if __name__ == "__main__":
    bag_variance()
    bag_unstable()
    print("OUT:", OUT)
