"""Classification-metrics concept-page diagrams (muted palette, parallel scale).

Two figures for 03. Supervised_Learning/concepts/14-Classification-Metrics.md:
  1. cm_matrix.png  -- the 2x2 confusion matrix (TP/FP/FN/TN) with precision read
     down the predicted-positive column and recall across the actual-positive row.
  2. cm_roc_pr.png  -- on an IMBALANCED set (5% positive): the ROC curve looks
     great (high AUC) while the PR curve reveals the model's real weakness.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def cm_matrix():
    TP, FP, FN, TN = 80, 30, 20, 870
    cells = {(0, 0): (f"TP\n{TP}", GREEN), (0, 1): (f"FN\n{FN}", RED),
             (1, 0): (f"FP\n{FP}", RED), (1, 1): (f"TN\n{TN}", GREEN)}
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    for (r, c), (label, col) in cells.items():
        ax.add_patch(Rectangle((c, -r), 1, 1, facecolor=col, alpha=0.8, edgecolor="white", lw=2))
        ax.text(c + 0.5, -r + 0.5, label, ha="center", va="center", color="#fff", fontsize=15, fontweight="bold")
    ax.text(0.5, 1.18, "Predicted +", ha="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(1.5, 1.18, "Predicted −", ha="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(-0.32, 0.5, "Actual +", ha="center", va="center", rotation=90, fontsize=12, fontweight="bold", color=NAVY)
    ax.text(-0.32, -0.5, "Actual −", ha="center", va="center", rotation=90, fontsize=12, fontweight="bold", color=NAVY)
    # precision down the predicted-+ column, recall across the actual-+ row
    ax.annotate("Precision = TP/(TP+FP)\n= 80/110 = 0.73\n(of flagged, how many right)",
                (0.5, -1.55), ha="center", fontsize=10, color=BLUE, fontweight="bold")
    ax.annotate("Recall = TP/(TP+FN)\n= 80/100 = 0.80\n(of actual +, how many caught)",
                (2.55, 0.5), ha="left", va="center", fontsize=10, color=PURPLE, fontweight="bold")
    ax.add_patch(Rectangle((0, -1), 1, 2, fill=False, edgecolor=BLUE, lw=2.5, ls="--"))
    ax.add_patch(Rectangle((0, 0), 2, 1, fill=False, edgecolor=PURPLE, lw=2.5, ls="--"))
    ax.set_xlim(-0.7, 4.4); ax.set_ylim(-2.1, 1.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("The confusion matrix: everything is built from these four cells",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/cm_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cm_matrix.png")


def _curves(scores, labels):
    order = np.argsort(-scores); y = labels[order]
    P = y.sum(); N = len(y) - P
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / P; fpr = fp / N
    prec = tp / (tp + fp); rec = tpr
    roc_auc = np.trapezoid(np.concatenate([[0], tpr]), np.concatenate([[0], fpr]))
    pr_auc = np.trapezoid(prec[::-1], rec[::-1])
    return fpr, tpr, rec, prec, roc_auc, abs(pr_auc)


def cm_roc_pr():
    rng = np.random.default_rng(0)
    Npos, Nneg = 50, 950                                # 5% positive — imbalanced
    scores = np.concatenate([rng.normal(1.0, 1.0, Npos), rng.normal(0.0, 1.0, Nneg)])
    labels = np.concatenate([np.ones(Npos), np.zeros(Nneg)]).astype(int)
    fpr, tpr, rec, prec, roc_auc, pr_auc = _curves(scores, labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.0))
    ax1.plot(fpr, tpr, color=BLUE, lw=2.6); ax1.plot([0, 1], [0, 1], color=SLATE, ls=":", lw=1.3)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate (recall)")
    ax1.set_title(f"ROC curve — looks great (AUC = {roc_auc:.2f})", fontsize=11.5, fontweight="bold", color=BLUE)
    _despine(ax1); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
    ax2.plot(rec, prec, color=RED, lw=2.6); ax2.axhline(Npos/(Npos+Nneg), color=SLATE, ls=":", lw=1.3)
    ax2.text(0.4, Npos/(Npos+Nneg)+0.02, "baseline = positive rate (0.05)", fontsize=8.5, color=SLATE)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title(f"PR curve — reveals the weakness (AUC = {pr_auc:.2f})", fontsize=11.5, fontweight="bold", color=RED)
    _despine(ax2); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    fig.suptitle("Same model, 5%-positive data: ROC flatters it, PR tells the truth",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/cm_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cm_roc_pr.png")


if __name__ == "__main__":
    cm_matrix()
    cm_roc_pr()
    print("OUT:", OUT)
