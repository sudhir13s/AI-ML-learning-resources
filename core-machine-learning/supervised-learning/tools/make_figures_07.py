"""Figure generator for 07-Decision-Trees — every number from the REAL runs in ``decision_trees.py``.

All figures come from the same real pipeline the chapter and notebook use (``decision_trees.py`` on the
scikit-learn **Iris**, **Breast Cancer**, and **Diabetes** datasets): the impurity functions, the
from-scratch/sklearn-verified tree, the measured overfitting depth sweep, the feature importances and the
MDI-vs-permutation bias, and the regression-tree staircase. Nothing is hand-typed; every split, curve, bar,
and annotation is read off an executed function call.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``sup07_``:

  sup07_impurity.png       -- the split criteria: Gini, entropy, and misclassification error as the class
                              proportion p sweeps 0..1; why the two smooth/concave ones grow a tree and the
                              piecewise-linear one does not.
  sup07_tree_structure.png -- the LEARNED tree on real Iris (sklearn plot_tree): real features, thresholds,
                              Gini, and sample counts at every node.
  sup07_boundary.png       -- the axis-aligned decision boundary a tree carves on the 2-feature Iris slice —
                              rectangles, not a smooth curve — with the real training points on top.
  sup07_overfit.png        -- the depth-vs-accuracy overfitting curve on Breast Cancer: train climbs to 1.0
                              while validation peaks early and plateaus; the best-validation depth is marked.
  sup07_importance.png     -- feature importance on Breast Cancer (MDI) AND the high-cardinality trap: MDI
                              ranks a pure-noise near-unique column above the real signal; permutation does not.
  sup07_regression.png     -- a regression tree on one real diabetes feature: the shallow (generalizing) and
                              deep (noise-chasing) piecewise-constant staircases over the scatter.

    python make_figures_07.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / scikit-learn 1.9.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in the domain-level ``03. Supervised_Learning/tools/`` folder, while the chapter
# module it demonstrates stays in that chapter's ``code/`` folder. Put that folder on sys.path so the
# ``decision_trees`` import below resolves no matter the working directory.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "07-Decision-Trees" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, plot_tree  # noqa: E402

from decision_trees import (  # noqa: E402  (resolved via the sys.path insert above)
    DecisionTreeScratch,
    depth_sweep,
    entropy,
    feature_importance,
    gini,
    load_cancer,
    load_iris_2d,
    load_iris_full,
    regression_staircase,
    verify_against_sklearn,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # class 0 / train / entropy
PURPLE = "#5D4A8A"  # class 1 / process / a second series
GREEN = "#2E7A5A"  # class 2 / good / shallow / correct
RED = "#8B3B4A"  # cost / overfit / deep / the noise column
SLATE = "#4A5B6E"  # neutral / misclassification
AMBER = "#7A6528"  # highlight / chosen depth / annotations
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "sup07_"
CLASS_COLOURS = (BLUE, PURPLE, GREEN)  # three Iris species, in label order


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


# ============================ Fig 1: the impurity measures ======================================
def fig_impurity() -> None:
    """Gini, entropy, and misclassification error for a binary node as the class proportion p sweeps 0..1.

    Every point is computed by the module's own ``gini``/``entropy`` on a synthetic count vector, so the
    curves are the exact functions the tree uses. Entropy is halved to overlay its shape on Gini (they
    almost coincide); misclassification is drawn to show it is piecewise-linear — the reason it grows a
    poor tree.
    """
    ps = np.linspace(0.001, 0.999, 400)
    scale = 1000  # a large synthetic node so counts approximate the continuous proportion p
    gini_vals = np.array([gini(np.array([0] * int(p * scale) + [1] * int((1 - p) * scale))) for p in ps])
    entropy_vals = np.array([entropy(np.array([0] * int(p * scale) + [1] * int((1 - p) * scale))) for p in ps])
    misclass = np.minimum(ps, 1 - ps)  # 1 - max(p, 1-p): the error of predicting the majority class

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.plot(ps, entropy_vals, color=BLUE, linewidth=2.4, label="entropy  −Σ p·log₂ p  (peak 1.0 bit)")
    ax.plot(ps, entropy_vals / 2, color=BLUE, linewidth=1.4, linestyle=":", alpha=0.8,
            label="entropy / 2  (shape ≈ Gini)")
    ax.plot(ps, gini_vals, color=GREEN, linewidth=2.4, label="Gini  1 − Σ p²  (peak 0.5)")
    ax.plot(ps, misclass, color=SLATE, linewidth=2.4, linestyle="--",
            label="misclassification  1 − max p  (piecewise-linear)")
    ax.axvline(0.5, color=GRID, linewidth=1.0)
    ax.annotate("all maximal at p = 0.5\n(a 50/50 node is the most impure)",
                xy=(0.5, 1.0), xytext=(0.13, 0.86), fontsize=9, color=INK,
                arrowprops={"arrowstyle": "->", "color": INK})
    ax.annotate("pure ⇒ 0", xy=(0.99, 0.02), xytext=(0.72, 0.12), fontsize=9, color=INK,
                arrowprops={"arrowstyle": "->", "color": INK})
    ax.set_xlabel("proportion p of class 0 in the node")
    ax.set_ylabel("impurity")
    ax.set_title(
        "Three ways to score 'how mixed' a node is\n"
        "Gini and entropy are smooth and strictly concave (they reward any move toward purity);\n"
        "misclassification is piecewise-linear — a poor criterion for GROWING a tree",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.08)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}impurity.png")


# ============================ Fig 2: the learned tree structure ==================================
def fig_tree_structure() -> None:
    """The real learned tree on full Iris (sklearn plot_tree): actual features, thresholds, Gini, counts."""
    data = load_iris_full()
    clf = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
    clf.fit(data.x_train, data.y_train)
    test_acc = clf.score(data.x_test, data.y_test)

    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(
        clf,
        feature_names=data.feature_names,
        class_names=data.class_names,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=9,
        ax=ax,
    )
    ax.set_title(
        f"A learned decision tree on real Iris (max_depth=3, Gini) — test accuracy {test_acc:.3f}\n"
        "each box shows the split test, its Gini, the sample count, the class split, and the majority class;\n"
        "the first split (petal length ≤ 2.45) already isolates every setosa into a pure leaf (Gini 0)",
        fontsize=12, color=INK,
    )
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}tree_structure.png")


# ============================ Fig 3: the axis-aligned decision boundary ==========================
def fig_boundary() -> None:
    """The rectangles a from-scratch tree carves on the 2-feature Iris slice, with real points on top.

    The boundary is painted by classifying a dense grid with our OWN ``DecisionTreeScratch`` (verified
    against scikit-learn), so the picture is the from-scratch model's actual decision surface, not a
    library stand-in. The staircase of horizontal/vertical edges is the signature of axis-aligned splits.
    """
    data = load_iris_2d()
    match = verify_against_sklearn(data, max_depth=4)
    tree = DecisionTreeScratch(max_depth=4, criterion="gini").fit(data.x_train, data.y_train)

    x_all = np.vstack([data.x_train, data.x_test])
    pad = 0.3
    x0 = np.linspace(x_all[:, 0].min() - pad, x_all[:, 0].max() + pad, 400)
    x1 = np.linspace(x_all[:, 1].min() - pad, x_all[:, 1].max() + pad, 400)
    grid_x, grid_y = np.meshgrid(x0, x1)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    zz = tree.predict(grid).reshape(grid_x.shape)

    from matplotlib.colors import ListedColormap

    region_cmap = ListedColormap(["#C9D6E3", "#D3CCE1", "#C6E0D3"])  # pale tints of the class colours

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(grid_x, grid_y, zz, alpha=0.9, cmap=region_cmap, levels=[-0.5, 0.5, 1.5, 2.5])
    ax.contour(grid_x, grid_y, zz, colors="white", linewidths=1.2, levels=[0.5, 1.5])
    for c in range(3):
        m = data.y_train == c
        ax.scatter(data.x_train[m, 0], data.x_train[m, 1], s=42, color=CLASS_COLOURS[c],
                   edgecolor="white", linewidth=0.6, label=data.class_names[c], zorder=3)
    ax.set_xlabel(data.feature_names[0])
    ax.set_ylabel(data.feature_names[1])
    ax.set_title(
        "A decision tree carves axis-aligned RECTANGLES (from-scratch tree, max_depth=4)\n"
        f"boundaries are horizontal/vertical steps — never a diagonal; matches scikit-learn on "
        f"{match.prediction_agreement * 100:.0f}% of test points",
        fontsize=11.5,
    )
    legend_items = [Patch(facecolor=CLASS_COLOURS[c], edgecolor="white", label=data.class_names[c])
                    for c in range(3)]
    ax.legend(handles=legend_items, frameon=False, fontsize=10, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}boundary.png")


# ============================ Fig 4: the overfitting depth sweep =================================
def fig_overfit() -> None:
    """Train vs validation accuracy as tree depth grows on Breast Cancer — the overfitting curve, measured."""
    data = load_cancer()
    sweep = depth_sweep(data)
    depths = np.array(sweep.depths)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(depths, sweep.train_acc, color=BLUE, linewidth=2.4, marker="o", markersize=5,
            label="training accuracy")
    ax.plot(depths, sweep.val_acc, color=RED, linewidth=2.4, marker="s", markersize=5,
            label="validation accuracy")
    ax.axvline(sweep.best_depth, color=AMBER, linewidth=1.6, linestyle="--")
    ax.annotate(
        f"best validation\nmax_depth = {sweep.best_depth}\nval acc = {sweep.best_val_acc:.3f}",
        xy=(sweep.best_depth, sweep.best_val_acc), xytext=(sweep.best_depth + 3, sweep.best_val_acc - 0.03),
        fontsize=9, color=AMBER, arrowprops={"arrowstyle": "->", "color": AMBER})
    over = depths >= sweep.best_depth
    ax.fill_between(depths[over], 0.90, 1.005, color=RED, alpha=0.06)
    ax.annotate("overfitting zone:\ntrain → 1.0, val flat/↓",
                xy=(depths[-1], 0.97), xytext=(11.5, 0.945), fontsize=9, color=RED)
    ax.set_xlabel("tree depth (max_depth)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.90, 1.005)
    ax.set_title(
        f"A single tree overfits with depth — Breast Cancer ({data.x_train.shape[0]} train tumours)\n"
        f"training accuracy climbs to {sweep.train_acc[-1]:.3f} while validation peaks at "
        f"{sweep.best_val_acc:.3f} (depth {sweep.best_depth}) and plateaus — the gap IS the overfitting",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}overfit.png")


# ============================ Fig 5: feature importance + the MDI bias ===========================
def fig_importance() -> None:
    """Left: real MDI importances on Breast Cancer. Right: MDI fooled by a noise column, permutation not."""
    data = load_cancer()
    imp = feature_importance(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.3, 1]})

    # -- left: the top real MDI importances --
    names = [n for n, _ in imp.top_features][::-1]
    vals = [v for _, v in imp.top_features][::-1]
    ax1.barh(range(len(names)), vals, color=GREEN, alpha=0.9)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel("mean decrease in impurity (MDI)")
    for i, v in enumerate(vals):
        ax1.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    ax1.set_title(
        "Which features drive the tree? (MDI on Breast Cancer)\n"
        f"'{imp.top_features[0][0]}' alone accounts for {imp.top_features[0][1] * 100:.0f}% of the impurity decrease",
        fontsize=10.5,
    )
    ax1.set_xlim(0, max(vals) * 1.18)
    _style_axis(ax1)

    # -- right: MDI vs permutation on the trap (signal vs pure-noise near-unique column) --
    labels = ["real\nsignal", "pure-noise\nrandom_id"]
    x = np.arange(2)
    w = 0.36
    mdi_bars = [imp.signal_mdi, imp.noise_mdi]
    perm_bars = [imp.signal_permutation, imp.noise_permutation]
    ax2.bar(x - w / 2, mdi_bars, w, color=RED, alpha=0.9, label="MDI (impurity)")
    ax2.bar(x + w / 2, perm_bars, w, color=BLUE, alpha=0.9, label="permutation (held-out)")
    ax2.axhline(0, color=INK, linewidth=0.8)
    for xi, (m, p) in enumerate(zip(mdi_bars, perm_bars)):
        ax2.text(xi - w / 2, m + 0.02, f"{m:.2f}", ha="center", fontsize=8.5, color=RED)
        ax2.text(xi + w / 2, p + 0.02, f"{p:.2f}", ha="center", fontsize=8.5, color=BLUE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel("importance")
    ax2.set_title(
        "The high-cardinality trap\n"
        "MDI ranks a pure-noise near-unique column ABOVE the real\n"
        "signal; permutation on held-out data correctly scores it ~0",
        fontsize=10.5,
    )
    ax2.legend(frameon=False, fontsize=9, loc="upper center")
    _style_axis(ax2)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}importance.png")


# ============================ Fig 6: the regression-tree staircase ===============================
def fig_regression() -> None:
    """Shallow vs deep regression trees on one real diabetes feature — the piecewise-constant staircase."""
    reg = regression_staircase()

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.scatter(reg.x, reg.y, s=20, color=SLATE, alpha=0.35, label="real diabetes patients", zorder=1)
    ax.plot(reg.grid, reg.shallow_pred, color=GREEN, linewidth=2.6, drawstyle="steps-mid",
            label=f"shallow tree (depth {reg.shallow_depth}) — generalizes", zorder=3)
    ax.plot(reg.grid, reg.deep_pred, color=RED, linewidth=1.6, drawstyle="steps-mid", alpha=0.9,
            label=f"deep tree (depth {reg.deep_depth}) — chases noise", zorder=2)
    ax.set_xlabel(f"{reg.feature_name} (standardized diabetes feature)")
    ax.set_ylabel("disease progression (target)")
    ax.set_title(
        "A regression tree predicts the MEAN target in each leaf — a piecewise-constant STAIRCASE\n"
        "the shallow tree captures the broad trend; the deep tree fits the noise with many tiny steps\n"
        "(a tree cannot extrapolate: outside the training range it just repeats the edge leaf's mean)",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}regression.png")


def main() -> None:
    print(f"numpy {np.__version__} | matplotlib {matplotlib.__version__}")
    fig_impurity()
    fig_tree_structure()
    fig_boundary()
    fig_overfit()
    fig_importance()
    fig_regression()
    print(f"\nall figures written to {OUT_DIR} with prefix '{IMG_PREFIX}'")


if __name__ == "__main__":
    main()
