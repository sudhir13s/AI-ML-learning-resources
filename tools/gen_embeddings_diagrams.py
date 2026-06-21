"""Word-embeddings concept-page diagrams (muted palette, parallel matplotlib scale).

Two figures for 06. NLP/concepts/05-Word-Embeddings-Word2Vec-GloVe-FastText.md:
  1. emb_space.png   -- the iconic embedding space: gender + royalty axes with the
     king−man+woman≈queen parallelogram, plus semantic clusters (illustrative).
  2. emb_onehot_vs_dense.png -- why dense: a sparse one-hot row (mostly zeros, no
     similarity) vs a short dense vector (every dim meaningful, cosine-comparable).
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "06. NLP", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def emb_space():
    # hand-placed 2D coords so the parallelogram + clusters read cleanly (illustrative)
    words = {
        "man": (1.0, 1.0), "woman": (1.0, 2.0),
        "king": (3.0, 1.0), "queen": (3.0, 2.0),
        "uncle": (1.6, 1.05), "aunt": (1.6, 2.05),
        "dog": (4.6, 4.2), "cat": (5.0, 4.5), "puppy": (4.4, 4.6),
        "paris": (5.2, 0.6), "france": (4.4, 0.5), "tokyo": (6.0, 0.9), "japan": (5.2, 0.8),
    }
    colors = {"man": NAVY, "woman": NAVY, "king": PURPLE, "queen": PURPLE,
              "uncle": NAVY, "aunt": NAVY, "dog": GREEN, "cat": GREEN, "puppy": GREEN,
              "paris": AMBER, "france": AMBER, "tokyo": AMBER, "japan": AMBER}
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    for w, (x, y) in words.items():
        ax.scatter(x, y, s=80, color=colors[w], zorder=4, edgecolor="white")
        ax.annotate(w, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=10, fontweight="bold")
    # the analogy parallelogram man->king and woman->queen (same offset vector)
    for (a, b, c) in [("man", "king", RED), ("woman", "queen", RED)]:
        ax.add_patch(FancyArrowPatch(words[a], words[b], arrowstyle="->", color=c, lw=2.2,
                                     mutation_scale=16, zorder=3))
    ax.annotate("king − man  ≈  queen − woman\n(the 'royalty' direction)", (2.0, 1.5),
                fontsize=9.5, color=RED, fontweight="bold", ha="center")
    ax.add_patch(FancyArrowPatch((0.6, 1.0), (0.6, 2.0), arrowstyle="->", color=SLATE, lw=1.6, mutation_scale=12))
    ax.annotate("gender →", (0.35, 1.5), rotation=90, fontsize=9, color=SLATE, va="center")
    ax.set_title("The embedding space: similar words cluster, meaning becomes arithmetic",
                 fontsize=13.5, fontweight="bold")
    ax.set_xlabel("(illustrative 2D projection of high-dim vectors)")
    ax.set_xlim(0, 7); ax.set_ylim(0, 5.2); ax.set_yticks([]); _despine(ax)
    ax.spines["left"].set_visible(False); ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(f"{OUT}/emb_space.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote emb_space.png")


def onehot_vs_dense():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 3.6), gridspec_kw={"height_ratios": [1, 1]})
    V = 18
    oh = np.zeros((1, V)); oh[0, 7] = 1
    ax1.imshow(oh, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax1.set_yticks([0]); ax1.set_yticklabels(['"king"'], fontsize=10, fontweight="bold")
    ax1.set_xticks([]); ax1.set_title("One-hot (length = vocabulary, ~50k+): all zeros but one; every word equidistant",
                                      fontsize=11, fontweight="bold")
    rng = np.random.default_rng(1)
    dense = rng.standard_normal((1, V)) * 0.6
    im = ax2.imshow(dense, aspect="auto", cmap="RdBu", vmin=-1.5, vmax=1.5)
    ax2.set_yticks([0]); ax2.set_yticklabels(['"king"'], fontsize=10, fontweight="bold")
    ax2.set_xticks([]); ax2.set_title("Dense embedding (length ~300): every dimension carries graded meaning; cosine-comparable",
                                      fontsize=11, fontweight="bold")
    fig.suptitle("Why dense beats one-hot", fontsize=13.5, fontweight="bold", y=1.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/emb_onehot_vs_dense.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote emb_onehot_vs_dense.png")


if __name__ == "__main__":
    emb_space()
    onehot_vs_dense()
    print("OUT:", OUT)
