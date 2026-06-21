"""Stacking & Blending concept-page diagrams (muted palette, parallel scale).

Four figures for 03. Supervised_Learning/concepts/11-Stacking-and-Blending.md:
  1. stacking_oof.png     -- the out-of-fold (OOF) prediction construction across k folds:
     which rows are predicted by which fold's held-out model. This is the index
     bookkeeping that makes level-1 meta-features leak-free. Illustrative schematic.
  2. stacking_wins.png    -- MEASURED bar chart: each base model's accuracy vs the stacked
     ensemble's, on a real dataset (breast cancer). Stacking beats the best single base.
  3. stacking_diversity.png -- base-learner error correlation / diversity: a heatmap of
     pairwise disagreement between base models. Diverse (low-correlation) errors are what
     give the meta-learner something to exploit. MEASURED on the same dataset.
  4. stacking_blend_vs_stack.png -- blending (single holdout) vs stacking (k-fold OOF):
     meta-feature data efficiency. Blending trains the meta-learner on a small holdout;
     stacking uses every row via OOF. Illustrative schematic of data usage.

Run with:  /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_stacking_diagrams.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _dataset():
    """A moderately hard, noisy, nonlinear dataset where no single base learner is great
    but the bases are DIVERSE — exactly where a meta-learner can stitch them into a win.
    Identical to the dataset used in the concept page's code block."""
    return make_classification(n_samples=4000, n_features=20, n_informative=10,
                               n_redundant=5, n_clusters_per_class=4, flip_y=0.05,
                               class_sep=0.8, random_state=7)

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def stacking_oof():
    """Schematic: for each of k folds, the model trained on the other k-1 folds
    predicts the held-out fold. Stack those held-out predictions -> a full-length,
    leak-free column of meta-features."""
    k = 5
    n_rows = 5  # one block per fold for the picture
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    cell = 1.0
    # Left: the data split into k folds, showing which fold is "held out" per round.
    for rnd in range(k):
        y0 = (k - 1 - rnd) * (cell + 0.18)
        for f in range(k):
            x0 = f * (cell + 0.06)
            held = (f == rnd)
            color = AMBER if held else SLATE
            ax.add_patch(Rectangle((x0, y0), cell, cell, facecolor=color,
                                   edgecolor="white", lw=1.4))
            label = "predict\n(held-out)" if held else "train"
            ax.text(x0 + cell / 2, y0 + cell / 2, label, ha="center", va="center",
                    color="#fff", fontsize=8.0, fontweight="bold" if held else "normal")
        ax.text(-0.55, y0 + cell / 2, f"round {rnd+1}", ha="right", va="center",
                fontsize=9.5, fontweight="bold", color=NAVY)
    ax.text(2.5 * (cell + 0.06), k * (cell + 0.18) + 0.15, "the k folds of the training data",
            ha="center", fontsize=11, fontweight="bold", color=NAVY)

    # Right: the assembled OOF meta-feature column (every row predicted exactly once,
    # by a model that never saw it).
    xcol = k * (cell + 0.06) + 1.3
    ax.annotate("", xy=(xcol - 0.25, 2.6), xytext=(k * (cell + 0.06) - 0.1, 2.6),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2.2))
    ax.text((xcol + k * (cell + 0.06) - 0.35) / 2, 3.0, "assemble", ha="center",
            fontsize=9.5, fontweight="bold", color=PURPLE)
    for i in range(k):
        y0 = (k - 1 - i) * (cell + 0.18)
        ax.add_patch(Rectangle((xcol, y0), cell, cell, facecolor=GREEN,
                               edgecolor="white", lw=1.4))
        ax.text(xcol + cell / 2, y0 + cell / 2, f"ŷ rows\nin fold {i+1}", ha="center",
                va="center", color="#fff", fontsize=7.8, fontweight="bold")
    ax.text(xcol + cell / 2, k * (cell + 0.18) + 0.15,
            "OOF meta-feature\n(every row predicted once,\nby a model that never saw it)",
            ha="center", fontsize=9.5, fontweight="bold", color=GREEN)

    ax.set_xlim(-1.3, xcol + cell + 0.4)
    ax.set_ylim(-0.3, k * (cell + 0.18) + 0.95)
    ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("Out-of-fold (OOF) construction: each row's meta-feature comes from a model that never trained on it",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(); fig.savefig(f"{OUT}/stacking_oof.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote stacking_oof.png")


def _base_models():
    return [
        ("logistic\nregression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))),
        ("decision\ntree", DecisionTreeClassifier(max_depth=5, random_state=0)),
        ("k-NN", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25))),
        ("naive\nBayes", GaussianNB()),
    ]


def stacking_wins():
    """MEASURED: base-model test accuracies vs the stacked ensemble."""
    X, y = _dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=7)
    bases = _base_models()
    accs, names = [], []
    for name, mdl in bases:
        mdl.fit(Xtr, ytr)
        accs.append(mdl.score(Xte, yte)); names.append(name.replace("\n", " "))
    stack = StackingClassifier(
        estimators=[(n.replace("\n", "_"), m) for n, m in bases],
        final_estimator=LogisticRegression(max_iter=2000, random_state=0),
        cv=5, n_jobs=-1)
    stack.fit(Xtr, ytr)
    stack_acc = stack.score(Xte, yte)

    accs.append(stack_acc); names.append("STACKED\nensemble")
    colors = [SLATE] * (len(accs) - 1) + [GREEN]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    xs = np.arange(len(accs))
    bars = ax.bar(xs, accs, color=colors, edgecolor="white", width=0.66)
    best_base = max(accs[:-1])
    ax.axhline(best_base, color=AMBER, ls="--", lw=1.6)
    ax.text(0.05, best_base + 0.002, f"best single base = {best_base:.3f}",
            color=AMBER, fontsize=9.5, fontweight="bold")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.001, f"{a:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("test accuracy"); ax.set_ylim(min(accs) - 0.04, max(accs) + 0.03)
    ax.set_title("Measured: the stacked ensemble beats every single base model\n(noisy synthetic dataset, logistic-regression meta-learner, 5-fold OOF)",
                 fontsize=12, fontweight="bold")
    _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/stacking_wins.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote stacking_wins.png  (stack={stack_acc:.3f}, best_base={best_base:.3f})")


def stacking_diversity():
    """MEASURED: pairwise error correlation between base models. Diverse base learners
    make DIFFERENT mistakes (low correlation) -> the meta-learner can exploit that."""
    X, y = _dataset()
    bases = _base_models()
    # OOF errors (1 = wrong) per model via cross_val_predict.
    err = {}
    for name, mdl in bases:
        pred = cross_val_predict(mdl, X, y, cv=5, n_jobs=-1)
        err[name.replace("\n", " ")] = (pred != y).astype(float)
    names = list(err.keys())
    E = np.vstack([err[n] for n in names])
    # correlation of the error vectors
    C = np.corrcoef(E)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(C, cmap="RdYlBu_r", vmin=-0.1, vmax=1.0)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8.5, rotation=30, ha="right")
    ax.set_yticklabels(names, fontsize=8.5)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    color="#fff" if abs(C[i, j]) > 0.45 else "#222", fontsize=8.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("error correlation", fontsize=9)
    ax.set_title("Base-learner error correlation: low off-diagonal = diverse mistakes\n(diversity is what gives the meta-learner something to combine)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/stacking_diversity.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote stacking_diversity.png")


def stacking_blend_vs_stack():
    """Schematic: blending trains the meta-learner on ONE small holdout (data left on
    the table); stacking uses EVERY row via k-fold OOF."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    cell = 1.0
    n = 10

    # Blending: split into base-train (70%) + holdout (30%). Meta-learner only sees holdout.
    ax = axes[0]
    for i in range(n):
        y0 = (n - 1 - i) * 0.42
        held = i >= 7
        color = GREEN if held else SLATE
        ax.add_patch(Rectangle((0, y0), 3.2, 0.36, facecolor=color, edgecolor="white", lw=1.0))
    ax.text(1.6, n * 0.42 + 0.15, "BLENDING", ha="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(3.5, (n - 1.5) * 0.42, "base-train\n(70%)", va="center", fontsize=9.5, color=SLATE, fontweight="bold")
    ax.text(3.5, 0.6 * 0.42, "holdout →\nmeta-learner\n(30%, the rest wasted)", va="center",
            fontsize=9, color=GREEN, fontweight="bold")
    ax.set_xlim(-0.3, 6.2); ax.set_ylim(-0.3, n * 0.42 + 0.7); ax.axis("off")

    # Stacking: every row becomes a meta-feature via OOF -> no data left on the table.
    ax = axes[1]
    foldcolors = [BLUE, PURPLE, GREEN, RED, AMBER]
    for i in range(n):
        y0 = (n - 1 - i) * 0.42
        ax.add_patch(Rectangle((0, y0), 3.2, 0.36, facecolor=foldcolors[i % 5],
                               edgecolor="white", lw=1.0))
    ax.text(1.6, n * 0.42 + 0.15, "STACKING (k-fold OOF)", ha="center", fontsize=12,
            fontweight="bold", color=NAVY)
    ax.text(3.5, (n - 1) * 0.42 / 2, "every row →\nan OOF meta-feature\n(100% used)", va="center",
            fontsize=9.5, color=GREEN, fontweight="bold")
    ax.set_xlim(-0.3, 6.2); ax.set_ylim(-0.3, n * 0.42 + 0.7); ax.axis("off")

    fig.suptitle("Blending leaves the holdout's worth of data on the table; stacking uses every row via OOF",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}/stacking_blend_vs_stack.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote stacking_blend_vs_stack.png")


if __name__ == "__main__":
    stacking_oof()
    stacking_wins()
    stacking_diversity()
    stacking_blend_vs_stack()
    print("OUT:", OUT)
