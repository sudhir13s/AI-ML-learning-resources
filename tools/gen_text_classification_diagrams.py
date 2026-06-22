"""Text-Classification & Sentiment-Analysis concept-page diagrams (muted palette).

Four figures for 06. NLP/concepts/10-Text-Classification-and-Sentiment-Analysis.md:
  1. textcls_ladder.png      -- the representation/model ladder (BoW -> embeddings ->
     CNN/RNN -> BERT) as a capability/accuracy progression (schematic, hand-placed).
  2. textcls_model_compare.png -- MEASURED accuracy / macro-F1 bars for NB vs
     TF-IDF+LogReg vs a transformer on an IMDb-style sentiment subset. Numbers are
     read from results computed by the companion verification script; if a transformer
     run is unavailable, the bar uses the published/measured value and is annotated.
  3. textcls_cnn.png         -- CNN-for-text schematic: 1-D filters (n-gram detectors)
     sliding over the embedding matrix + max-over-time pooling.
  4. textcls_confusion.png   -- MEASURED confusion matrix for the TF-IDF+LogReg
     sentiment model, plus a negation/aspect callout panel.

Run with ~/.uv/envs/ml-py312/bin/python3.  Outputs into the NLP concepts/images dir.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "06. NLP", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------------------
# 1. The representation / model ladder
# ----------------------------------------------------------------------------
def ladder():
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    rungs = [
        ("BoW / TF-IDF\n+ Naive Bayes / LogReg / linear SVM",
         "sparse counts; word identity only\nstrong, instant baseline", BLUE, 0),
        ("Word embeddings + pooling\n(mean/max) -> MLP  (DAN)",
         "dense, captures word similarity\nstill order-blind", NAVY, 1),
        ("CNN-for-text  (Kim 2014)\n1-D filters = n-gram detectors",
         "local order: 'not good' as a unit\nmax-over-time pooling", GREEN, 2),
        ("RNN / biLSTM\nsequence-aware encoder",
         "full left+right context\nlong-range, but slow", AMBER, 3),
        ("Fine-tuned BERT\n[CLS] head, pretrain -> finetune",
         "deep bidirectional context\nSOTA for most tasks", PURPLE, 4),
    ]
    n = len(rungs)
    x0, dx = 0.6, 1.0
    for title, sub, color, i in rungs:
        y = i * 1.15
        box = FancyBboxPatch((x0 + i * 0.30, y), 5.2, 0.92,
                             boxstyle="round,pad=0.04,rounding_size=0.08",
                             facecolor=color, edgecolor="white", lw=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x0 + i * 0.30 + 2.6, y + 0.60, title, color="white", ha="center",
                va="center", fontsize=10.5, fontweight="bold", zorder=4)
        ax.text(x0 + i * 0.30 + 2.6, y + 0.24, sub, color="#e9e9ef", ha="center",
                va="center", fontsize=8.2, zorder=4)
    # the rising arrow of capability
    ax.add_patch(FancyArrowPatch((0.15, 0.2), (0.15, n * 1.15 - 0.1),
                                 arrowstyle="-|>", color=SLATE, lw=2.4,
                                 mutation_scale=22, zorder=2))
    ax.text(-0.05, n * 1.15 / 2 - 0.1,
            "more context modeled  ·  more data / compute  -->",
            rotation=90, va="center", ha="center", fontsize=9.5,
            color=SLATE, fontweight="bold")
    ax.text(x0 + 2.9, n * 1.15 + 0.15,
            "Accuracy generally climbs up the ladder — but so do cost and latency.\n"
            "A linear model on TF-IDF is still the baseline that everything must beat.",
            ha="center", fontsize=9.0, color=SLATE, style="italic")
    ax.set_xlim(-0.6, 7.2); ax.set_ylim(-0.2, n * 1.15 + 0.7)
    ax.set_title("The text-classification representation ladder",
                 fontsize=13.5, fontweight="bold", pad=12)
    _despine(ax); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{OUT}/textcls_ladder.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote textcls_ladder.png")


# ----------------------------------------------------------------------------
# 2. Measured model comparison  (numbers filled from results dict)
# ----------------------------------------------------------------------------
def model_compare(results):
    models = ["Multinomial\nNaive Bayes", "TF-IDF +\nLogReg", "Linear SVM\n(TF-IDF)",
              "DistilBERT\n(fine-tuned)"]
    acc = [results[k]["acc"] for k in ["nb", "logreg", "svm", "bert"]]
    f1 = [results[k]["f1"] for k in ["nb", "logreg", "svm", "bert"]]
    note = [results[k].get("note", "") for k in ["nb", "logreg", "svm", "bert"]]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    b1 = ax.bar(x - w / 2, acc, w, label="Accuracy", color=BLUE, edgecolor="white")
    b2 = ax.bar(x + w / 2, f1, w, label="Macro-F1", color=GREEN, edgecolor="white")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008,
                    f"{r.get_height():.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    for i, nt in enumerate(note):
        if nt:
            ax.text(i, 0.04, nt, ha="center", fontsize=7.6, color="white",
                    style="italic", rotation=0)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9.5)
    ax.set_ylim(0, 1.02); ax.set_ylabel("score")
    ax.set_title("Measured: NB vs TF-IDF linear models vs transformer\n"
                 "(IMDb sentiment, 4k train / 2k test subset)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/textcls_model_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote textcls_model_compare.png")


# ----------------------------------------------------------------------------
# 3. CNN-for-text schematic
# ----------------------------------------------------------------------------
def cnn_text():
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    tokens = ["the", "movie", "was", "not", "good", "at", "all"]
    T, d = len(tokens), 5
    # embedding matrix grid
    x0, y0, cw, ch = 0.5, 2.6, 0.8, 0.42
    rng = np.random.default_rng(3)
    for i in range(T):
        for j in range(d):
            val = rng.standard_normal()
            c = BLUE if val >= 0 else SLATE
            ax.add_patch(Rectangle((x0 + i * cw, y0 + j * ch), cw * 0.92, ch * 0.9,
                                   facecolor=c, alpha=0.45 + 0.4 * min(abs(val), 1),
                                   edgecolor="white", lw=0.6))
        ax.text(x0 + i * cw + cw * 0.46, y0 - 0.28, tokens[i], ha="center",
                fontsize=9.5, fontweight="bold", rotation=0)
    ax.text(x0 - 0.35, y0 + d * ch / 2, "embedding\ndim (d)", rotation=90,
            va="center", ha="center", fontsize=9, color=SLATE)
    ax.text(x0 + T * cw / 2, y0 + d * ch + 0.18,
            "Embedding matrix:  sentence -> (T tokens x d dims)",
            ha="center", fontsize=10, fontweight="bold", color=NAVY)

    # a width-3 filter window highlighted over 'was not good'
    fx = x0 + 2 * cw
    ax.add_patch(Rectangle((fx - 0.02, y0 - 0.05), 3 * cw - 0.06, d * ch + 0.1,
                           facecolor="none", edgecolor=RED, lw=2.4, zorder=5))
    ax.text(fx + 1.5 * cw - 0.4, y0 + d * ch + 0.62,
            "width-3 filter\n(a 3-gram detector)", ha="center", color=RED,
            fontsize=9, fontweight="bold")
    ax.add_patch(FancyArrowPatch((fx + 1.5 * cw - 0.4, y0 + d * ch + 0.2),
                                 (fx + 1.5 * cw - 0.4, y0 + d * ch + 0.05),
                                 arrowstyle="-|>", color=RED, lw=1.6, mutation_scale=14))

    # feature map (one value per window position) + max-over-time
    fy = 1.3
    fm = [0.2, 0.5, 1.0, 0.7, 0.3]   # illustrative; window over 'not good' fires hardest
    fmx0 = x0 + 0.9
    for k, v in enumerate(fm):
        h = 0.06 + 0.5 * v
        ax.add_patch(Rectangle((fmx0 + k * cw, fy), cw * 0.7, h,
                               facecolor=GREEN, edgecolor="white"))
    ax.text(fmx0 + len(fm) * cw / 2, fy - 0.25,
            "feature map: filter response at each window position", ha="center",
            fontsize=8.6, color=GREEN)
    # max-over-time pooling -> single number
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

    ax.text(x0 + T * cw / 2 + 0.5, 0.35,
            "Many filters of widths 2/3/4/5 -> concatenated max-pooled features -> softmax classifier",
            ha="center", fontsize=9.2, color=SLATE, style="italic")
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 6.0)
    ax.set_title("CNN-for-text: 1-D filters as n-gram detectors + max-over-time pooling (Kim 2014)",
                 fontsize=12.5, fontweight="bold", pad=10)
    _despine(ax); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{OUT}/textcls_cnn.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote textcls_cnn.png")


# ----------------------------------------------------------------------------
# 4. Measured confusion matrix + negation/aspect callout
# ----------------------------------------------------------------------------
def confusion(cm):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})
    cm = np.array(cm)
    im = ax1.imshow(cm, cmap="Blues")
    labels = ["negative", "positive"]
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(labels); ax1.set_yticklabels(labels)
    ax1.set_xlabel("predicted"); ax1.set_ylabel("actual")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / total
            ax1.text(j, i, f"{cm[i, j]}\n({frac:.1%})", ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax1.set_title("Confusion matrix\nTF-IDF + LogReg (IMDb test)",
                  fontsize=11.5, fontweight="bold")

    # negation / aspect challenge panel
    ax2.axis("off")
    rows = [
        ("\"not good\"", "negation", "wrong: BoW votes positive", RED),
        ("\"not bad at all\"", "double negation", "actually positive", AMBER),
        ("\"great cast, dull plot\"", "two aspects", "needs aspect-level", AMBER),
        ("\"yeah, brilliant. /s\"", "sarcasm", "surface says positive", RED),
        ("\"the movie was great\"", "clear polarity", "easy: positive", GREEN),
    ]
    ax2.text(0.5, 1.02, "Why sentiment is hard: where bag-of-words breaks",
             ha="center", fontsize=11.5, fontweight="bold", transform=ax2.transAxes)
    y = 0.86
    for txt, why, verdict, color in rows:
        ax2.add_patch(FancyBboxPatch((0.02, y - 0.06), 0.96, 0.13,
                                     boxstyle="round,pad=0.01,rounding_size=0.02",
                                     facecolor=color, alpha=0.18, edgecolor=color,
                                     lw=1.2, transform=ax2.transAxes))
        ax2.text(0.05, y, txt, fontsize=9.3, fontweight="bold", va="center",
                 transform=ax2.transAxes)
        ax2.text(0.50, y, why, fontsize=8.4, va="center", ha="center", color=SLATE,
                 transform=ax2.transAxes)
        ax2.text(0.97, y, verdict, fontsize=8.4, va="center", ha="right",
                 color=color, fontweight="bold", transform=ax2.transAxes)
        y -= 0.175
    fig.suptitle("", fontsize=1)
    fig.tight_layout(); fig.savefig(f"{OUT}/textcls_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote textcls_confusion.png")


if __name__ == "__main__":
    import json, sys
    # results may be passed via a JSON file produced by the verification script;
    # fall back to measured defaults (computed 2026-06-22, see verify script output).
    res_path = os.path.join(os.path.dirname(__file__), "_textcls_results.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            blob = json.load(f)
        results, cm = blob["results"], blob["cm"]
    else:
        # measured 2026-06-22 in ml-py312 on IMDb (4k train / 2k test;
        # DistilBERT = 1 epoch on a 2k subset). Reproduce with tools/_verify_textcls.py.
        results = {
            "nb":     {"acc": 0.832, "f1": 0.831},
            "logreg": {"acc": 0.858, "f1": 0.858},
            "svm":    {"acc": 0.860, "f1": 0.860},
            "bert":   {"acc": 0.877, "f1": 0.877, "note": "measured"},
        }
        cm = [[867, 148], [136, 849]]
    ladder()
    model_compare(results)
    cnn_text()
    confusion(cm)
    print("OUT:", OUT)
