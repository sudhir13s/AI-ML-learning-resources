"""Generate the step-by-step teaching notebook (07-Decision-Trees.ipynb).

The notebook mirrors ``decision_trees.py`` one measurement at a time, so a learner can open it, run every
cell live, and *see* how a decision tree is grown on real data — the impurity functions, the information
gain of a worked split, the greedy best-split search, a tree grown from scratch, the verification against
scikit-learn, the learned-tree picture, the axis-aligned decision boundary, the overfitting depth sweep,
feature importance and its high-cardinality trap, and the regression-tree staircase. Each numbered step has
a short markdown lead-in (the intuition) followed by ONE focused code cell with real output. This generator
writes the .ipynb; a separate nbconvert pass executes it headless so the outputs are embedded.

    python build_notebook_07.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../07-Decision-Trees/code/07-Decision-Trees.ipynb"

This generator lives in the domain-level ``03. Supervised_Learning/tools/`` folder; the notebook it writes
(and the module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited
.ipynb) so the notebook and the module stay in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "07-Decision-Trees" / "code"
NB_PATH = _CHAPTER_CODE / "07-Decision-Trees.ipynb"

_CELL_ID = 0


def _next_id() -> str:
    """Stable, sequential cell id (silences nbformat's MissingIDFieldWarning)."""
    global _CELL_ID
    _CELL_ID += 1
    return f"cell-{_CELL_ID:02d}"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS: list[dict] = []


def add_md(source: str) -> None:
    CELLS.append(md(source))


def add_code(source: str) -> None:
    CELLS.append(code(source))


# ============================ Title ============================================================
add_md(
    "# Decision Trees — a step-by-step, runnable notebook\n"
    "\n"
    "A decision tree is the machine-learning version of *20 Questions*: to classify a point it asks a "
    "series of simple `feature <= threshold?` questions and walks down the branches to a leaf. It **learns** "
    "which questions to ask, in what order, by greedily picking at each step the split that most reduces "
    "**impurity** (how mixed the labels are). The result is a model you can read as a flowchart — and the "
    "atom that random forests and gradient boosting are built from.\n"
    "\n"
    "This notebook builds that, one measurement at a time, on **real scikit-learn datasets** (no download): "
    "**Iris** (150 flowers) for the from-scratch tree and the boundary, **Breast Cancer** (569 tumours) for "
    "overfitting and importance, and **Diabetes** (442 patients) for the regression staircase. It is the "
    "executable companion to the chapter and to `decision_trees.py`; every function used here lives in that "
    "module, imported so the notebook and the module can never drift apart.\n"
    "\n"
    "By the end you will have **measured**, not just been told:\n"
    "\n"
    "1. what **Gini** and **entropy** compute (0 when pure, maximal at 50/50);\n"
    "2. a real **information-gain** split, by hand and in code (parent − weighted children);\n"
    "3. a decision tree **grown from scratch** and **verified** to match scikit-learn;\n"
    "4. the **overfitting curve** — train → 1.0 while validation plateaus — and the sweet-spot depth;\n"
    "5. why **MDI feature importance is fooled** by a high-cardinality noise column, and permutation is not;\n"
    "6. a **regression tree** as a piecewise-constant staircase.\n"
    "\n"
    "Everything runs on CPU in a couple of seconds, seeded for reproducibility."
)

# ---- Step 0: setup ----
add_md(
    "## Step 0 — Setup and version banner\n"
    "\n"
    "We import the real functions from the chapter module (so this notebook uses the *exact same code* the "
    "figures and the page use) and print the library versions the results were produced on."
)
add_code(
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import sklearn\n"
    "from sklearn.tree import DecisionTreeClassifier, plot_tree\n"
    "\n"
    "from decision_trees import (\n"
    "    gini, entropy, information_gain, best_split,\n"
    "    DecisionTreeScratch, verify_against_sklearn,\n"
    "    load_iris_2d, load_iris_full, load_cancer,\n"
    "    depth_sweep, feature_importance, regression_staircase,\n"
    ")\n"
    "\n"
    "print(f'numpy {np.__version__} | scikit-learn {sklearn.__version__}')\n"
    "iris2d = load_iris_2d()\n"
    "print(f'Iris slice: {iris2d.x_train.shape[0]} train + {iris2d.x_test.shape[0]} test flowers, '\n"
    "      f'features {iris2d.feature_names}, classes {iris2d.class_names}')"
)

# ---- Step 1: impurity ----
add_md(
    "## Step 1 — Impurity: how mixed is a node?\n"
    "\n"
    "A split is 'good' when it makes the children **purer**. We need a number for 'mixed'. Two standard "
    "choices, writing $p_c$ for the fraction of class $c$ in a node:\n"
    "\n"
    "$$\\text{Gini} = 1 - \\sum_c p_c^2 \\qquad \\text{Entropy} = -\\sum_c p_c \\log_2 p_c$$\n"
    "\n"
    "Both are **0 when the node is pure** and **maximal at 50/50** (Gini 0.5, entropy 1 bit for two classes). "
    "Let's confirm on the most-mixed node (5 vs 5) and a pure node (10 vs 0)."
)
add_code(
    "fifty_fifty = np.array([0]*5 + [1]*5)\n"
    "pure = np.array([0]*10)\n"
    "print(f'50/50 node : Gini={gini(fifty_fifty):.3f}   entropy={entropy(fifty_fifty):.3f} bits   (MAX impurity)')\n"
    "print(f'pure  node : Gini={gini(pure):.3f}   entropy={entropy(pure):.3f} bits   (ZERO impurity)')\n"
    "print('\\nSplits try to move nodes from ~0.5 toward 0.')"
)

# ---- Step 2: information gain ----
add_md(
    "## Step 2 — Information gain: scoring a split\n"
    "\n"
    "A split's quality is its **information gain** — the parent's impurity minus the **weighted average** of "
    "the children's impurity:\n"
    "\n"
    "$$\\text{Gain} = I(\\text{parent}) - \\left(\\tfrac{n_L}{n} I(\\text{left}) + \\tfrac{n_R}{n} I(\\text{right})\\right)$$\n"
    "\n"
    "The weights stop the tree from 'improving' a node by carving off a tiny pure sliver. Let's reproduce the "
    "classic worked example: a parent with **9 'yes' / 5 'no'** split into children **(6 yes, 2 no)** and "
    "**(3 yes, 3 no)**."
)
add_code(
    "parent = np.array([1]*6 + [0]*2 + [1]*3 + [0]*3)   # left = first 8, right = last 6\n"
    "mask   = np.array([True]*8 + [False]*6)\n"
    "print(f'parent Gini            = {gini(parent):.3f}   (9 yes / 5 no)')\n"
    "print(f'left child (6/2) Gini  = {gini(parent[mask]):.3f}')\n"
    "print(f'right child (3/3) Gini = {gini(parent[~mask]):.3f}   (a 50/50 node)')\n"
    "print(f'information (Gini) gain = {information_gain(parent, mask):.3f}')\n"
    "print('\\nA small positive gain: it purified the left child but left a 50/50 mess on the right.')"
)

# ---- Step 3: best-split search ----
add_md(
    "## Step 3 — The greedy best-split search\n"
    "\n"
    "Growing a tree means, at each node, trying **every feature and every candidate threshold** (the midpoints "
    "between adjacent sorted values) and keeping the split with the highest gain. On the Iris petal features, "
    "the very first best split is the one that isolates *setosa*."
)
add_code(
    "split = best_split(iris2d.x_train, iris2d.y_train)\n"
    "print(f'best root split: {iris2d.feature_names[split.feature]} <= {split.threshold:.3f}  '\n"
    "      f'(Gini gain = {split.gain:.3f})')\n"
    "left = iris2d.x_train[:, split.feature] <= split.threshold\n"
    "print(f'left child  ({left.sum():>3} samples): class counts {np.bincount(iris2d.y_train[left])}')\n"
    "print(f'right child ({(~left).sum():>3} samples): class counts {np.bincount(iris2d.y_train[~left])}')\n"
    "print('\\nThe left child is a PURE setosa leaf already — one question separated a whole class.')"
)

# ---- Step 4: grow from scratch ----
add_md(
    "## Step 4 — Grow the whole tree from scratch\n"
    "\n"
    "Apply that split recursively — left, right, left-of-left, ... — until a node is pure, too small, or hits "
    "`max_depth`. `DecisionTreeScratch` does exactly this (Gini, greedy binary splits). Here is the learned "
    "tree printed as an indented flowchart of `feature <= threshold?` questions."
)
add_code(
    "tree = DecisionTreeScratch(max_depth=3, criterion='gini').fit(iris2d.x_train, iris2d.y_train)\n"
    "print(tree.describe(iris2d.feature_names, iris2d.class_names))\n"
    "print(f'\\nleaves={tree.n_leaves()}  '\n"
    "      f'train_acc={tree.score(iris2d.x_train, iris2d.y_train):.3f}  '\n"
    "      f'test_acc={tree.score(iris2d.x_test, iris2d.y_test):.3f}')"
)

# ---- Step 5: verify against sklearn ----
add_md(
    "## Step 5 — Is it the real thing? Verify against scikit-learn\n"
    "\n"
    "Grown with the **same criterion (Gini) and the same `max_depth`**, our from-scratch tree should reach the "
    "**same test accuracy** as scikit-learn's `DecisionTreeClassifier` and agree with it on the vast majority of "
    "predictions. (We don't demand an *exact* match: when two splits tie on gain, CART implementations break the "
    "tie by their own rule, so a different-but-equally-good tree is legitimate.)"
)
add_code(
    "match = verify_against_sklearn(iris2d, max_depth=3)\n"
    "print(f'from-scratch test accuracy : {match.scratch_test_acc:.3f}')\n"
    "print(f'scikit-learn test accuracy : {match.sklearn_test_acc:.3f}')\n"
    "print(f'they agree on {match.prediction_agreement*100:.1f}% of test predictions')\n"
    "assert abs(match.scratch_test_acc - match.sklearn_test_acc) < 1e-9\n"
    "print('\\n=> same accuracy, near-total agreement: the growing algorithm is genuine CART.')"
)

# ---- Step 6: plot the learned tree ----
add_md(
    "## Step 6 — See the learned tree (real thresholds and Gini)\n"
    "\n"
    "scikit-learn's `plot_tree` draws the fitted tree with the real split test, Gini, sample count, and class "
    "split at every node. We use the full 4-feature Iris here so every node's chosen feature is visible."
)
add_code(
    "iris_full = load_iris_full()\n"
    "clf = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42).fit(iris_full.x_train, iris_full.y_train)\n"
    "fig, ax = plt.subplots(figsize=(13, 7))\n"
    "plot_tree(clf, feature_names=iris_full.feature_names, class_names=list(iris_full.class_names),\n"
    "          filled=True, rounded=True, fontsize=9, ax=ax)\n"
    "ax.set_title(f'Learned Iris tree (max_depth=3, Gini) — test accuracy {clf.score(iris_full.x_test, iris_full.y_test):.3f}')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 7: decision boundary ----
add_md(
    "## Step 7 — What a tree computes: axis-aligned rectangles\n"
    "\n"
    "Because every split is `feature <= threshold`, a tree carves the feature space into **axis-aligned "
    "rectangles** — one predicted class per box. Paint the prediction of our **from-scratch** tree over a dense "
    "grid on the 2-feature Iris slice and the tell-tale **staircase** boundary appears: horizontal and vertical "
    "edges only, never a diagonal."
)
add_code(
    "tree2d = DecisionTreeScratch(max_depth=4, criterion='gini').fit(iris2d.x_train, iris2d.y_train)\n"
    "x_all = np.vstack([iris2d.x_train, iris2d.x_test])\n"
    "x0 = np.linspace(x_all[:,0].min()-0.3, x_all[:,0].max()+0.3, 300)\n"
    "x1 = np.linspace(x_all[:,1].min()-0.3, x_all[:,1].max()+0.3, 300)\n"
    "gx, gy = np.meshgrid(x0, x1)\n"
    "zz = tree2d.predict(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)\n"
    "plt.figure(figsize=(8, 6))\n"
    "plt.contourf(gx, gy, zz, alpha=0.25, levels=[-0.5,0.5,1.5,2.5], colors=['#3A6B96','#5D4A8A','#2E7A5A'])\n"
    "for c, colour in enumerate(['#3A6B96','#5D4A8A','#2E7A5A']):\n"
    "    m = iris2d.y_train == c\n"
    "    plt.scatter(iris2d.x_train[m,0], iris2d.x_train[m,1], s=32, color=colour,\n"
    "                edgecolor='white', lw=0.5, label=iris2d.class_names[c])\n"
    "plt.xlabel(iris2d.feature_names[0])\n"
    "plt.ylabel(iris2d.feature_names[1])\n"
    "plt.title('Axis-aligned decision boundary (from-scratch tree, depth 4)')\n"
    "plt.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 8: overfitting table ----
add_md(
    "## Step 8 — Why trees overfit: the depth sweep\n"
    "\n"
    "Keep splitting and a tree eventually isolates every training point in its own pure leaf — 100% training "
    "accuracy, but it has **memorized the noise**. On the noisier Breast Cancer data, sweep `max_depth` and "
    "watch training accuracy march to 1.0 while **validation** accuracy peaks early and plateaus."
)
add_code(
    "cancer = load_cancer()\n"
    "sweep = depth_sweep(cancer)\n"
    "print(f'{\"depth\":>6}{\"train acc\":>12}{\"val acc\":>10}')\n"
    "for d, tr, va in zip(sweep.depths, sweep.train_acc, sweep.val_acc):\n"
    "    mark = '   <- best val' if d == sweep.best_depth else ''\n"
    "    print(f'{d:>6}{tr:>12.3f}{va:>10.3f}{mark}')\n"
    "print(f'\\nbest validation acc {sweep.best_val_acc:.3f} at max_depth={sweep.best_depth}; '\n"
    "      f'deepest tree memorizes train to {sweep.train_acc[-1]:.3f}.')"
)

# ---- Step 9: overfitting plot ----
add_md(
    "## Step 9 — The overfitting curve, plotted\n"
    "\n"
    "The growing **gap** between the train and validation curves is overfitting made visible. The dashed line "
    "marks the depth with the best validation accuracy — the complexity you should actually keep."
)
add_code(
    "d = np.array(sweep.depths)\n"
    "plt.figure(figsize=(8.5, 5))\n"
    "plt.plot(d, sweep.train_acc, 'o-', color='#3A6B96', lw=2.2, label='training accuracy')\n"
    "plt.plot(d, sweep.val_acc, 's-', color='#8B3B4A', lw=2.2, label='validation accuracy')\n"
    "plt.axvline(sweep.best_depth, color='#7A6528', ls='--', lw=1.5,\n"
    "            label=f'best val depth = {sweep.best_depth}')\n"
    "plt.xlabel('tree depth (max_depth)')\n"
    "plt.ylabel('accuracy')\n"
    "plt.title('A single tree overfits with depth (Breast Cancer)')\n"
    "plt.legend()\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 10: MDI importance ----
add_md(
    "## Step 10 — Feature importance (mean decrease in impurity)\n"
    "\n"
    "A tree can tell you **which features mattered**: sum the impurity decrease each feature contributes over "
    "every split that used it. This is scikit-learn's `feature_importances_` (MDI). On Breast Cancer, one "
    "feature dominates."
)
add_code(
    "imp = feature_importance(cancer)\n"
    "print('top features by MDI (impurity decrease):')\n"
    "for name, val in imp.top_features:\n"
    "    print(f'  {name:<26}{val:.3f}')\n"
    "print(f'\\n{imp.top_features[0][0]!r} alone is {imp.top_features[0][1]*100:.0f}% of the impurity decrease.')"
)

# ---- Step 11: MDI trap ----
add_md(
    "## Step 11 — The trap: MDI is fooled by high-cardinality features\n"
    "\n"
    "MDI has a notorious flaw: it is **biased toward features with many distinct values** (they offer more "
    "candidate thresholds, so one often lucks into a spurious impurity decrease). Build a controlled case — one "
    "weak-but-**real** signal and one **pure-noise, near-unique** column — and watch MDI rank the noise *above* "
    "the signal, while **permutation importance** (measured on held-out data) correctly scores it ~0."
)
add_code(
    "print('              MDI (impurity)   permutation (held-out)')\n"
    "print(f'real signal   {imp.signal_mdi:>13.3f}   {imp.signal_permutation:>18.3f}')\n"
    "print(f'random_id     {imp.noise_mdi:>13.3f}   {imp.noise_permutation:>18.3f}   <- pure noise!')\n"
    "print('\\nMDI ranks the meaningless near-unique column HIGHER. Never feature-select from raw MDI')\n"
    "print('when you have high-cardinality features (IDs, timestamps) — use permutation importance or SHAP.')"
)

# ---- Step 12: regression staircase ----
add_md(
    "## Step 12 — Regression trees: variance reduction and the staircase\n"
    "\n"
    "The same algorithm does regression: swap Gini for **variance**, and majority-vote for the **mean**. Each "
    "leaf predicts the mean target of its interval, so the fit is **piecewise-constant** — a staircase. Fit a "
    "shallow and a deep regression tree to one real diabetes feature (`bmi`): the shallow tree captures the "
    "trend; the deep tree chases the noise."
)
add_code(
    "reg = regression_staircase()\n"
    "plt.figure(figsize=(9, 5.2))\n"
    "plt.scatter(reg.x, reg.y, s=18, color='#4A5B6E', alpha=0.35, label='real diabetes patients')\n"
    "plt.plot(reg.grid, reg.shallow_pred, drawstyle='steps-mid', color='#2E7A5A', lw=2.6,\n"
    "         label=f'shallow (depth {reg.shallow_depth}) — generalizes')\n"
    "plt.plot(reg.grid, reg.deep_pred, drawstyle='steps-mid', color='#8B3B4A', lw=1.5,\n"
    "         label=f'deep (depth {reg.deep_depth}) — chases noise')\n"
    "plt.xlabel(f'{reg.feature_name} (standardized)')\n"
    "plt.ylabel('disease progression')\n"
    "plt.title('Regression tree = piecewise-constant staircase (each step = one leaf mean)')\n"
    "plt.legend()\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "print(f'shallow: {len(np.unique(reg.shallow_pred))} distinct leaf values | '\n"
    "      f'deep: {len(np.unique(reg.deep_pred))} distinct leaf values')"
)

# ---- Recap ----
add_md(
    "## Recap\n"
    "\n"
    "On real scikit-learn datasets, one runnable notebook built the whole of decision trees:\n"
    "\n"
    "| Step | What we measured | The takeaway |\n"
    "|---|---|---|\n"
    "| 1–2 | Gini/entropy; information gain | 0 when pure, max at 50/50; gain = parent − weighted children |\n"
    "| 3–4 | best-split search; grow from scratch | greedy `feature <= t?` splits, recursed to a flowchart |\n"
    "| 5 | verify vs scikit-learn | **same test accuracy**, ~total prediction agreement — genuine CART |\n"
    "| 6–7 | plot the tree; the boundary | axis-aligned **rectangles** — a staircase, never a diagonal |\n"
    "| 8–9 | the depth sweep | train → **1.0** while validation plateaus; keep the best-val depth |\n"
    "| 10–11 | MDI importance; the trap | MDI is **fooled** by a high-cardinality noise column; permutation is not |\n"
    "| 12 | regression staircase | leaf = **mean**; piecewise-constant; deep overfits, and trees can't extrapolate |\n"
    "\n"
    "**A decision tree recursively splits the feature space with axis-aligned `feature <= t?` questions, "
    "greedily maximizing information gain.** It is low-bias but high-variance — which is exactly why "
    "**random forests** (↓ variance) and **gradient boosting** (↓ bias) ensemble trees. The single tree is "
    "the on-ramp; the ensembles are the destination."
)


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {NB_PATH} with {len(CELLS)} cells "
          f"({sum(c['cell_type'] == 'code' for c in CELLS)} code, "
          f"{sum(c['cell_type'] == 'markdown' for c in CELLS)} markdown)")


if __name__ == "__main__":
    main()
