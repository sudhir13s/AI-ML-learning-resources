"""Reproducible figure generator for 10-Text-Classification-and-Sentiment-Analysis.

Every embedded PNG on the chapter page is produced here from the SAME seeded backend the page
and the notebook use -- the NB-by-hand log-posteriors, the NB/LogReg/SVM ladder, the confusion
matrix, the precision-recall and ROC curves, the threshold sweep, and the imbalance demo are all
IMPORTED from `text_classification.py`, so the figures cannot silently drift from the prose or
the notebook. Run:

    python make_figures_10.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at
150 dpi, prefixed `tc_`. The palette matches the chapter's Mermaid diagrams (muted, white text
on fills).

Figures produced (measured = from the live backend; illustrative = a labelled schematic):
  tc_ladder.png            -- illustrative: the representation ladder (BoW -> ... -> BERT)
  tc_cnn.png               -- illustrative: CNN-for-text, filters as n-gram detectors + max-pool
  tc_nb_byhand.png         -- MEASURED: the NB log-posterior for 'great fun' (pos vs neg), by hand
  tc_model_compare.png     -- MEASURED: accuracy / macro-F1 bars for NB vs LogReg vs SVM
  tc_confusion.png         -- MEASURED: confusion matrix for TF-IDF+LogReg + hard-case panel
  tc_pr_curve.png          -- MEASURED: precision-recall curve (+ average precision) for LogReg
  tc_threshold_sweep.png   -- MEASURED: precision/recall/F1 vs decision threshold
  tc_imbalance.png         -- MEASURED: accuracy lies under imbalance; macro-F1 / PR-AUC don't
  tc_pr_vs_roc.png         -- MEASURED: PR vs ROC for the same scores under heavy imbalance
  tc_leakage.png           -- MEASURED: honest (chance) vs leaky (fabricated) on pure noise

Verified on Python 3.12 / numpy 2.x / matplotlib 3.x / scikit-learn 1.9, CPU, deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from text_classification import (
    nb_by_hand,
    run_imbalance_demo,
    run_ladder_comparison,
    run_leakage_demo,
    threshold_sweep,
)

# ---- Palette (matches the chapter Mermaid classDefs) ----------------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"  # near-black for axis text
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 150

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax: plt.Axes) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------------------------
# 1. The representation / model ladder (illustrative schematic).
# --------------------------------------------------------------------------------------------
def fig_ladder() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    rungs = [
        ("BoW / TF-IDF  +  Naive Bayes / LogReg / linear SVM",
         "sparse counts; word identity only\nstrong, instant baseline", BLUE),
        ("Word embeddings + pooling (mean/max) -> MLP  (DAN)",
         "dense, captures word similarity\nstill order-blind", NAVY),
        ("CNN-for-text  (Kim 2014):  1-D filters = n-gram detectors",
         "local order: 'not good' as a unit\nmax-over-time pooling", GREEN),
        ("RNN / biLSTM:  sequence-aware encoder",
         "full left+right context\nlong-range, but slow", AMBER),
        ("Fine-tuned BERT:  [CLS] head, pretrain -> finetune",
         "deep bidirectional context\nSOTA for most tasks", PURPLE),
    ]
    n = len(rungs)
    x0 = 0.6
    for i, (title, sub, color) in enumerate(rungs):
        y = i * 1.15
        box = FancyBboxPatch(
            (x0 + i * 0.30, y), 5.6, 0.92,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color, edgecolor="white", lw=1.5, zorder=3,
        )
        ax.add_patch(box)
        ax.text(x0 + i * 0.30 + 2.8, y + 0.62, title, color="white", ha="center",
                va="center", fontsize=9.6, fontweight="bold", zorder=4)
        ax.text(x0 + i * 0.30 + 2.8, y + 0.25, sub, color="#e9e9ef", ha="center",
                va="center", fontsize=8.0, zorder=4)
    ax.add_patch(FancyArrowPatch((0.15, 0.2), (0.15, n * 1.15 - 0.1),
                                 arrowstyle="-|>", color=SLATE, lw=2.4,
                                 mutation_scale=22, zorder=2))
    ax.text(-0.08, n * 1.15 / 2 - 0.1,
            "more context modeled  ·  more data / compute  -->",
            rotation=90, va="center", ha="center", fontsize=9.5,
            color=SLATE, fontweight="bold")
    ax.text(x0 + 3.1, n * 1.15 + 0.18,
            "Accuracy generally climbs up the ladder — but so do cost and latency.\n"
            "A linear model on TF-IDF is still the baseline everything must beat.",
            ha="center", fontsize=9.0, color=SLATE, style="italic")
    ax.set_xlim(-0.6, 7.6)
    ax.set_ylim(-0.2, n * 1.15 + 0.8)
    ax.set_title("The text-classification representation ladder",
                 fontsize=13.5, fontweight="bold", pad=12, color=INK)
    _despine(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    _save(fig, "tc_ladder.png")


# --------------------------------------------------------------------------------------------
# 2. CNN-for-text schematic (illustrative).
# --------------------------------------------------------------------------------------------
def fig_cnn() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    tokens = ["the", "movie", "was", "not", "good", "at", "all"]
    n_tokens, d = len(tokens), 5
    x0, y0, cw, ch = 0.5, 2.6, 0.8, 0.42
    rng = np.random.default_rng(3)
    for i in range(n_tokens):
        for j in range(d):
            val = rng.standard_normal()
            c = BLUE if val >= 0 else SLATE
            ax.add_patch(Rectangle((x0 + i * cw, y0 + j * ch), cw * 0.92, ch * 0.9,
                                   facecolor=c, alpha=0.45 + 0.4 * min(abs(val), 1),
                                   edgecolor="white", lw=0.6))
        ax.text(x0 + i * cw + cw * 0.46, y0 - 0.28, tokens[i], ha="center",
                fontsize=9.5, fontweight="bold")
    ax.text(x0 - 0.35, y0 + d * ch / 2, "embedding\ndim (d)", rotation=90,
            va="center", ha="center", fontsize=9, color=SLATE)
    ax.text(x0 + n_tokens * cw / 2, y0 + d * ch + 0.18,
            "Embedding matrix:  sentence -> (T tokens x d dims)",
            ha="center", fontsize=10, fontweight="bold", color=NAVY)

    fx = x0 + 2 * cw
    ax.add_patch(Rectangle((fx - 0.02, y0 - 0.05), 3 * cw - 0.06, d * ch + 0.1,
                           facecolor="none", edgecolor=RED, lw=2.4, zorder=5))
    ax.text(fx + 1.5 * cw - 0.4, y0 + d * ch + 0.62,
            "width-3 filter\n(a 3-gram detector)", ha="center", color=RED,
            fontsize=9, fontweight="bold")
    ax.add_patch(FancyArrowPatch((fx + 1.5 * cw - 0.4, y0 + d * ch + 0.2),
                                 (fx + 1.5 * cw - 0.4, y0 + d * ch + 0.05),
                                 arrowstyle="-|>", color=RED, lw=1.6, mutation_scale=14))

    fy = 1.3
    feature_map = [0.2, 0.5, 1.0, 0.7, 0.3]  # window over 'was not good' fires hardest
    fmx0 = x0 + 0.9
    for k, v in enumerate(feature_map):
        h = 0.06 + 0.5 * v
        ax.add_patch(Rectangle((fmx0 + k * cw, fy), cw * 0.7, h,
                               facecolor=GREEN, edgecolor="white"))
    ax.text(fmx0 + len(feature_map) * cw / 2, fy - 0.25,
            "feature map: filter response at each window position", ha="center",
            fontsize=8.6, color=GREEN)
    peak = fmx0 + 2 * cw + cw * 0.35
    ax.add_patch(FancyArrowPatch((peak, fy + 0.56), (peak + 2.0, fy + 0.56),
                                 arrowstyle="-|>", color=AMBER, lw=2.0, mutation_scale=16))
    ax.text(peak + 1.0, fy + 0.78, "max-over-time pooling", ha="center",
            fontsize=9, color=AMBER, fontweight="bold")
    ax.add_patch(FancyBboxPatch((peak + 2.0, fy + 0.30), 1.5, 0.5,
                                boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor=PURPLE, edgecolor="white"))
    ax.text(peak + 2.75, fy + 0.55, "one feature\n(this n-gram\nis present)",
            ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    ax.text(x0 + n_tokens * cw / 2 + 0.5, 0.35,
            "Many filters of widths 2/3/4/5 -> concatenated max-pooled features -> softmax classifier",
            ha="center", fontsize=9.2, color=SLATE, style="italic")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 6.0)
    ax.set_title("CNN-for-text: 1-D filters as n-gram detectors + max-over-time pooling (Kim 2014)",
                 fontsize=12.5, fontweight="bold", pad=10, color=INK)
    _despine(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    _save(fig, "tc_cnn.png")


# --------------------------------------------------------------------------------------------
# 3. NB log-posterior, by hand (measured).
# --------------------------------------------------------------------------------------------
def fig_nb_byhand() -> None:
    bh = nb_by_hand()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    classes = ["positive", "negative"]
    scores = [bh.logp_pos, bh.logp_neg]
    colors = [GREEN, RED]
    bars = ax.barh(classes, scores, color=colors, edgecolor="white", height=0.55, zorder=3)
    for bar, s in zip(bars, scores):
        # value just past the bar's tip (bars are negative, so to the LEFT of the tip),
        # printed in the bar's own colour on the white background for guaranteed legibility
        ax.text(s - 0.08, bar.get_y() + bar.get_height() / 2, f"{s:.3f}",
                ha="right", va="center", color=bar.get_facecolor(), fontsize=12.5,
                fontweight="bold")
    winner = "positive" if bh.predicted == 1 else "negative"
    ax.set_xlabel("log-posterior score   log P(c) + Σ count·log P(w | c)   (less negative = more likely)")
    ax.set_title(f"Multinomial NB, by hand: doc = 'great fun'  ->  predicts {winner}\n"
                 f"gap = {abs(bh.logp_pos - bh.logp_neg):.2f} in log-space "
                 f"(≈ {np.exp(abs(bh.logp_pos - bh.logp_neg)):.0f}× more probable)",
                 fontsize=11.5, fontweight="bold")
    ax.axvline(0, color=SLATE, lw=0.8)
    _style_axis(ax)
    ax.set_xlim(min(scores) - 0.8, 0.2)
    fig.tight_layout()
    _save(fig, "tc_nb_byhand.png")


# --------------------------------------------------------------------------------------------
# 4. Measured model comparison: NB vs LogReg vs SVM.
# --------------------------------------------------------------------------------------------
def fig_model_compare(results) -> None:
    names = [r.name.replace(" + ", "+\n").replace(" ", "\n", 1) for r in results]
    acc = [r.accuracy for r in results]
    f1 = [r.macro_f1 for r in results]
    x = np.arange(len(results))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    b1 = ax.bar(x - w / 2, acc, w, label="Accuracy", color=BLUE, edgecolor="white", zorder=3)
    b2 = ax.bar(x + w / 2, f1, w, label="Macro-F1", color=GREEN, edgecolor="white", zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008,
                    f"{r.get_height():.3f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("score")
    ax.set_title("Measured: the classical baselines on a balanced synthetic sentiment set\n"
                 "(720 train / 480 test) — LogReg edges out NB; both beat a linear SVM here",
                 fontsize=11.8, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.95)
    _style_axis(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "tc_model_compare.png")


# --------------------------------------------------------------------------------------------
# 5. Confusion matrix + hard-case panel (measured).
# --------------------------------------------------------------------------------------------
def fig_confusion(cm: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})
    cm = np.asarray(cm)
    ax1.imshow(cm, cmap="Blues")
    labels = ["negative", "positive"]
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("predicted")
    ax1.set_ylabel("actual")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / total
            ax1.text(j, i, f"{cm[i, j]}\n({frac:.1%})", ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax1.set_title("Confusion matrix\nTF-IDF + LogReg (synthetic test)",
                  fontsize=11.5, fontweight="bold", color=INK)

    ax2.axis("off")
    rows = [
        ("\"not good\"", "negation", "wrong: BoW votes positive", RED),
        ("\"not bad at all\"", "double negation", "actually positive", AMBER),
        ("\"great cast, dull plot\"", "two aspects", "needs aspect-level", AMBER),
        ("\"yeah, brilliant. /s\"", "sarcasm", "surface says positive", RED),
        ("\"the movie was great\"", "clear polarity", "easy: positive", GREEN),
    ]
    ax2.text(0.5, 1.02, "Why sentiment is hard: where bag-of-words breaks",
             ha="center", fontsize=11.5, fontweight="bold", transform=ax2.transAxes, color=INK)
    y = 0.86
    for txt, why, verdict, color in rows:
        ax2.add_patch(FancyBboxPatch((0.02, y - 0.06), 0.96, 0.13,
                                     boxstyle="round,pad=0.01,rounding_size=0.02",
                                     facecolor=color, alpha=0.18, edgecolor=color,
                                     lw=1.2, transform=ax2.transAxes))
        ax2.text(0.05, y, txt, fontsize=9.3, fontweight="bold", va="center",
                 transform=ax2.transAxes, color=INK)
        ax2.text(0.50, y, why, fontsize=8.4, va="center", ha="center", color=SLATE,
                 transform=ax2.transAxes)
        ax2.text(0.97, y, verdict, fontsize=8.4, va="center", ha="right",
                 color=color, fontweight="bold", transform=ax2.transAxes)
        y -= 0.175
    fig.tight_layout()
    _save(fig, "tc_confusion.png")


# --------------------------------------------------------------------------------------------
# 6. Precision-recall curve (measured).
# --------------------------------------------------------------------------------------------
def fig_pr_curve(model, x_test, y_test, pr_auc: float) -> None:
    scores = model.scores(x_test)
    precision, recall, _ = precision_recall_curve(y_test, scores)
    pos_rate = float(np.mean(y_test))
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.plot(recall, precision, color=BLUE, lw=2.6, zorder=3,
            label=f"TF-IDF + LogReg  (AP = {pr_auc:.3f})")
    ax.axhline(pos_rate, color=SLATE, lw=1.4, ls="--", zorder=2,
               label=f"no-skill baseline = prevalence ({pos_rate:.2f})")
    ax.set_xlabel("recall  =  TP / (TP + FN)")
    ax.set_ylabel("precision  =  TP / (TP + FP)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.04)
    ax.set_title("Precision–Recall curve (positive class)\n"
                 "area under it = Average Precision (PR-AUC)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.95)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "tc_pr_curve.png")


# --------------------------------------------------------------------------------------------
# 7. Threshold sweep (measured).
# --------------------------------------------------------------------------------------------
def fig_threshold_sweep(model, x_test, y_test) -> None:
    scores = model.scores(x_test)
    sweep = threshold_sweep(scores, y_test)
    thr = sweep["thresholds"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(thr, sweep["precision"], color=BLUE, lw=2.4, label="precision", zorder=3)
    ax.plot(thr, sweep["recall"], color=RED, lw=2.4, label="recall", zorder=3)
    ax.plot(thr, sweep["f1"], color=GREEN, lw=2.4, label="F1", zorder=3)
    best_idx = int(np.argmax(sweep["f1"]))
    ax.axvline(thr[best_idx], color=AMBER, lw=1.6, ls="--", zorder=2,
               label=f"best-F1 threshold = {thr[best_idx]:.2f}")
    ax.axvline(0.5, color=SLATE, lw=1.2, ls=":", zorder=2, label="default 0.5")
    ax.set_xlabel("decision threshold on P(positive)")
    ax.set_ylabel("score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.04)
    ax.set_title("The threshold trade-off: raise it for precision, lower it for recall\n"
                 "the default 0.5 is just one column of this sweep",
                 fontsize=11.8, fontweight="bold")
    ax.legend(loc="lower center", framealpha=0.95, ncol=2, fontsize=9)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "tc_threshold_sweep.png")


# --------------------------------------------------------------------------------------------
# 8. Imbalance: accuracy lies, F1 / PR-AUC don't (measured).
# --------------------------------------------------------------------------------------------
def fig_imbalance(imb: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    metrics = ["Accuracy", "Macro-F1", "PR-AUC"]
    majority = [imb["majority_acc"], imb["majority_f1"], imb["majority_pr_auc"]]
    model = [imb["model_acc"], imb["model_f1"], imb["model_pr_auc"]]
    x = np.arange(len(metrics))
    w = 0.38
    b1 = ax.bar(x - w / 2, majority, w, color=RED, edgecolor="white", zorder=3,
                label='"always negative" (useless)')
    b2 = ax.bar(x + w / 2, model, w, color=GREEN, edgecolor="white", zorder=3,
                label="LogReg (class-weighted)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                    f"{r.get_height():.3f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title(f"Class imbalance ({imb['pos_rate']:.0%} positive): accuracy is a liar\n"
                 "the do-nothing model 'wins' on accuracy, loses on macro-F1 and PR-AUC",
                 fontsize=11.8, fontweight="bold")
    ax.legend(loc="upper center", framealpha=0.95)
    ax.annotate("accuracy looks great\nfor a useless model", xy=(-w / 2, majority[0]),
                xytext=(0.55, 0.66), fontsize=9, color=RED, fontweight="bold",
                ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    _style_axis(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "tc_imbalance.png")


# --------------------------------------------------------------------------------------------
# 9. PR vs ROC under heavy imbalance (measured) -- the Saito & Rehmsmeier point.
# --------------------------------------------------------------------------------------------
def fig_pr_vs_roc(imb: dict) -> None:
    model = imb["model"]
    x_test = imb["x_test"]
    y_test = imb["y_test"]
    scores = model.scores(x_test)
    precision, recall, _ = precision_recall_curve(y_test, scores)
    fpr, tpr, _ = roc_curve(y_test, scores)
    pos_rate = float(np.mean(y_test))
    roc_auc = roc_auc_score(y_test, scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.9))
    # ROC -- looks flattering: the no-skill line is the diagonal regardless of skew.
    ax1.plot(fpr, tpr, color=PURPLE, lw=2.6, zorder=3, label=f"ROC (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color=SLATE, lw=1.4, ls="--", label="no-skill diagonal")
    ax1.set_xlabel("false-positive rate")
    ax1.set_ylabel("true-positive rate (recall)")
    ax1.set_title("ROC: flattering under imbalance\n(no-skill = diagonal, unaffected by skew)",
                  fontsize=10.8, fontweight="bold", color=INK)
    ax1.legend(loc="lower right", framealpha=0.95, fontsize=9)
    _style_axis(ax1)
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.04)

    # PR -- honest: the no-skill line sits at the (tiny) prevalence.
    ax2.plot(recall, precision, color=BLUE, lw=2.6, zorder=3,
             label=f"PR (AP = {imb['model_pr_auc']:.3f})")
    ax2.axhline(pos_rate, color=SLATE, lw=1.4, ls="--",
                label=f"no-skill = prevalence ({pos_rate:.2f})")
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_title("PR: honest under imbalance\n(no-skill collapses to the prevalence)",
                  fontsize=10.8, fontweight="bold", color=INK)
    ax2.legend(loc="lower left", framealpha=0.95, fontsize=9)
    _style_axis(ax2)
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 1.04)

    fig.suptitle(f"Same scores, {pos_rate:.0%}-positive data: ROC stays high, PR exposes the difficulty",
                 fontsize=12.2, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "tc_pr_vs_roc.png")


# --------------------------------------------------------------------------------------------
# 10. Train/test leakage: honest (chance) vs leaky (fabricated) on the pure-noise corpus.
# --------------------------------------------------------------------------------------------
def fig_leakage(leak: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    labels = ["honest\n(select on train only)", "leaky\n(select on train+test)"]
    accs = [leak["honest_acc"], leak["leaky_acc"]]
    colors = [GREEN, RED]
    bars = ax.bar(labels, accs, width=0.55, color=colors, edgecolor="white", zorder=3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.012, f"{acc:.3f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
    ax.axhline(0.5, color=SLATE, lw=1.4, ls="--", zorder=2, label="chance (no signal exists)")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("reported test accuracy")
    ax.set_title("Leakage on a PURE-NOISE corpus (label independent of the text)\n"
                 f"honest stays at chance; the leak fabricates +{leak['gap']:.3f} of accuracy",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.95)
    _style_axis(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "tc_leakage.png")


def main() -> None:
    print("regenerating chapter-10 figures into", OUT_DIR)
    # Illustrative schematics (no data dependence).
    fig_ladder()
    fig_cnn()

    # Measured figures -- one shared balanced split for the ladder/confusion/PR/threshold set.
    results, lr, x_te, y_te = run_ladder_comparison()
    fig_nb_byhand()
    fig_model_compare(results)
    fig_confusion(results[1].cm)  # results[1] is TF-IDF + LogReg
    fig_pr_curve(lr, x_te, y_te, results[1].pr_auc)
    fig_threshold_sweep(lr, x_te, y_te)

    # Measured imbalance figures -- a separate skewed corpus.
    imb = run_imbalance_demo()
    fig_imbalance(imb)
    fig_pr_vs_roc(imb)

    # Measured leakage figure -- pure-noise corpus.
    fig_leakage(run_leakage_demo())
    print("done.")


if __name__ == "__main__":
    main()
