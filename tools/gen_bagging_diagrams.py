"""Bagging concept-page diagrams (muted palette, parallel scale). REAL measured.

Four figures for 03. Supervised_Learning/concepts/08-Bagging.md:
  1. bag_variance.png  -- prediction variance vs ensemble size B for a bagged deep
     tree: averaging B models cuts variance roughly like 1/B toward a floor.
  2. bag_unstable.png  -- bagging helps UNSTABLE learners (deep tree) a lot, but
     barely helps STABLE ones (linear model) -- the key requirement for bagging.
  3. bag_bootstrap.png -- bootstrap-sampling schematic: from one n-row dataset, each
     bootstrap draws n rows WITH replacement; ~63.2% distinct rows are in-bag and
     ~36.8% are out-of-bag (free validation). Measured + the (1-1/n)^n -> 1/e curve.
  4. bag_oob.png       -- measured OOB error vs held-out TEST error as B grows: OOB
     tracks test error once B is large enough, so OOB is free cross-validation.
"""
import os, warnings, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

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
    floor = var[-1]
    ax.axhline(floor, color=RED, lw=1.3, ls=":", label=f"correlated floor ≈ ρσ² ≈ {floor:.3f}")
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


def bag_bootstrap():
    """Schematic of one bootstrap draw + the in-bag/out-of-bag fraction vs n."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.8),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # LEFT: a tiny n=10 dataset; one bootstrap sample drawn WITH replacement.
    rng = np.random.default_rng(7)
    n = 10
    draw = rng.integers(0, n, n)                 # bootstrap indices (with replacement)
    counts = np.bincount(draw, minlength=n)      # times each original row was picked
    in_bag = counts > 0
    # original rows column
    for i in range(n):
        c = BLUE if in_bag[i] else RED
        axL.add_patch(plt.Rectangle((0, n - 1 - i), 0.8, 0.8, color=c, alpha=0.9))
        axL.text(0.4, n - 1 - i + 0.4, f"x{i}", ha="center", va="center", color="#fff", fontsize=9)
        if counts[i] > 1:
            axL.text(0.95, n - 1 - i + 0.4, f"×{counts[i]}", ha="left", va="center",
                     color=SLATE, fontsize=8.5, fontweight="bold")
    axL.text(0.4, n + 0.2, "original\n(n=10 rows)", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    # bootstrap sample column (the actual multiset drawn)
    for j, idx in enumerate(sorted(draw)):
        axL.add_patch(plt.Rectangle((2.4, n - 1 - j), 0.8, 0.8, color=PURPLE, alpha=0.9))
        axL.text(2.8, n - 1 - j + 0.4, f"x{idx}", ha="center", va="center", color="#fff", fontsize=9)
    axL.text(2.8, n + 0.2, "bootstrap sample\n(n draws, replace)", ha="center", va="bottom",
             fontsize=9.5, fontweight="bold")
    axL.annotate("", xy=(2.3, n/2), xytext=(0.9, n/2),
                 arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.6))
    nib, noob = int(in_bag.sum()), int((~in_bag).sum())
    axL.text(1.6, -0.9, f"in-bag (blue): {nib}/{n}    out-of-bag (red): {noob}/{n}",
             ha="center", fontsize=9.5, color=SLATE, fontweight="bold")
    axL.set_xlim(-0.3, 3.6); axL.set_ylim(-1.4, n + 1.3); axL.axis("off")
    axL.set_title("One bootstrap draw: sample n rows WITH replacement",
                  fontsize=11, fontweight="bold")

    # RIGHT: out-of-bag fraction vs n, measured, converging to 1/e.
    ns = [2, 3, 5, 10, 20, 50, 100, 500, 2000]
    measured = []
    for nn in ns:
        fr = np.mean([1 - len(set(np.random.default_rng(s).integers(0, nn, nn))) / nn
                      for s in range(60)])
        measured.append(fr)
    theory = [(1 - 1/nn) ** nn for nn in ns]
    axR.plot(ns, measured, color=RED, lw=2.2, marker="o", ms=5, label="measured OOB fraction")
    axR.plot(ns, theory, color=SLATE, lw=1.6, ls="--", label="(1 − 1/n)ⁿ")
    axR.axhline(1/np.e, color=GREEN, lw=1.4, ls=":", label="limit 1/e ≈ 0.368")
    axR.set_xscale("log")
    axR.set_xlabel("dataset size  n  (log scale)"); axR.set_ylabel("out-of-bag fraction")
    axR.set_ylim(0.30, 0.55)
    axR.set_title("≈ 36.8% of rows are out-of-bag:  (1 − 1/n)ⁿ → 1/e",
                  fontsize=11, fontweight="bold")
    axR.legend(frameon=False, fontsize=9); _despine(axR)

    fig.tight_layout(); fig.savefig(f"{OUT}/bag_bootstrap.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote bag_bootstrap.png")


def bag_oob():
    """Measured OOB error vs held-out test error as B grows: OOB ≈ test = free CV."""
    X, y = make_moons(n_samples=800, noise=0.3, random_state=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    Bs = [8, 12, 20, 35, 60, 100, 160, 250]
    oob_err, test_err = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for B in Bs:
            bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=B,
                                    oob_score=True, random_state=0).fit(Xtr, ytr)
            oob_err.append(1 - bag.oob_score_)
            test_err.append(1 - bag.score(Xte, yte))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(Bs, oob_err, color=AMBER, lw=2.4, marker="s", ms=5, label="out-of-bag error (free)")
    ax.plot(Bs, test_err, color=GREEN, lw=2.4, marker="o", ms=5, label="held-out test error")
    ax.set_xlabel("number of bagged models  B"); ax.set_ylabel("misclassification error")
    ax.set_title("OOB error tracks test error as B grows — cross-validation for free",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5); _despine(ax)
    ax.set_ylim(0, max(max(oob_err), max(test_err)) * 1.15)
    fig.tight_layout(); fig.savefig(f"{OUT}/bag_oob.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote bag_oob.png")


if __name__ == "__main__":
    bag_variance()
    bag_unstable()
    bag_bootstrap()
    bag_oob()
    print("OUT:", OUT)
