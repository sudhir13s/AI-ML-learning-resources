"""Word-Embeddings concept-page diagrams (muted palette, parallel matplotlib scale).

Four visuals for 06. NLP/concepts/05-Word-Embeddings-Word2Vec-GloVe-FastText.md:
  1. we_skipgram_vs_cbow.png  -- the two word2vec objectives side by side:
     skip-gram (center -> context) vs CBOW (context -> center).
  2. we_pca_real.png          -- MEASURED 2-D PCA of REAL pretrained GloVe-50
     vectors: royalty / animals / capitals cluster; parallel gender direction.
  3. we_analogy_parallelogram.png -- the king-man+woman~=queen parallelogram
     drawn from REAL GloVe vectors projected to 2-D.
  4. we_negsampling.png        -- negative sampling as pull/push: true context
     pulled together, k sampled negatives pushed apart; + unigram^0.75 bars.

All vector-based panels use REAL glove-wiki-gigaword-50 vectors via gensim
(downloaded once and cached by gensim.downloader). Run:
  /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_word_embeddings_diagrams.py
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


def _load_glove():
    import gensim.downloader as api
    return api.load("glove-wiki-gigaword-50")


# ---- 1. Skip-gram vs CBOW objectives ----------------------------------------
def skipgram_vs_cbow():
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))

    def box(ax, x, y, w, h, text, fc, fs=9.5):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="none",
                                   zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color="#fff", fontsize=fs, fontweight="bold", zorder=3)

    def arrow(ax, x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=14, color=SLATE, lw=1.8, zorder=1))

    # ---- Skip-gram: center -> context ----
    ax = axes[0]
    box(ax, 0.36, 0.62, 0.28, 0.16, "center\n'king'", AMBER)
    box(ax, 0.32, 0.40, 0.36, 0.14, "look up v_c", PURPLE)
    box(ax, 0.04, 0.10, 0.24, 0.14, "the", GREEN, 9)
    box(ax, 0.38, 0.10, 0.24, 0.14, "ruled", GREEN, 9)
    box(ax, 0.72, 0.10, 0.24, 0.14, "crown", GREEN, 9)
    arrow(ax, 0.50, 0.62, 0.50, 0.545)
    for cx in (0.16, 0.50, 0.84):
        arrow(ax, 0.50, 0.40, cx, 0.245)
    ax.set_title("Skip-gram: predict CONTEXT from center",
                 fontsize=13, fontweight="bold")
    ax.text(0.5, -0.04, "one center -> many context predictions",
            ha="center", fontsize=9, color=SLATE, transform=ax.transAxes)

    # ---- CBOW: context -> center ----
    ax = axes[1]
    box(ax, 0.04, 0.62, 0.24, 0.14, "the", BLUE, 9)
    box(ax, 0.38, 0.62, 0.24, 0.14, "ruled", BLUE, 9)
    box(ax, 0.72, 0.62, 0.24, 0.14, "crown", BLUE, 9)
    box(ax, 0.30, 0.38, 0.40, 0.14, "average\ncontext vecs", PURPLE)
    box(ax, 0.36, 0.10, 0.28, 0.16, "center\n'king'", AMBER)
    for cx in (0.16, 0.50, 0.84):
        arrow(ax, cx, 0.62, 0.50, 0.525)
    arrow(ax, 0.50, 0.38, 0.50, 0.265)
    ax.set_title("CBOW: predict CENTER from context",
                 fontsize=13, fontweight="bold")
    ax.text(0.5, -0.04, "many context words -> one center prediction",
            ha="center", fontsize=9, color=SLATE, transform=ax.transAxes)

    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.85); ax.axis("off")
    fig.suptitle("Two word2vec objectives (mirror images)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/we_skipgram_vs_cbow.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote we_skipgram_vs_cbow.png")


# ---- 2. Real PCA of GloVe-50 vectors ----------------------------------------
def pca_real(g):
    groups = {
        "royalty": (["king", "queen", "prince", "princess"], AMBER),
        "animals": (["cat", "dog", "rabbit", "horse"], GREEN),
        "capitals": (["paris", "london", "rome", "berlin"], BLUE),
        "countries": (["france", "england", "italy", "germany"], RED),
    }
    words, colors, labels = [], [], []
    for name, (ws, c) in groups.items():
        for w in ws:
            words.append(w); colors.append(c); labels.append(w)
    X = np.stack([g[w] for w in words]).astype(float)
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T                      # 2-D PCA projection of REAL vectors

    # per-word label offsets (dx, dy in points) to avoid overlaps in clusters
    off = {
        "king": (8, -14), "queen": (8, 8), "prince": (8, 6),
        "princess": (-58, 6), "cat": (-30, 6), "dog": (-30, -14),
        "rabbit": (10, 4), "horse": (8, 4), "paris": (8, 4),
        "london": (-58, 6), "rome": (8, 6), "berlin": (8, 4),
        "france": (8, 2), "england": (-72, 2), "italy": (8, -12),
        "germany": (8, 2),
    }
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for (px, py), c, lab in zip(P, colors, labels):
        ax.scatter(px, py, s=130, color=c, edgecolor="#fff", linewidth=1.2,
                   zorder=3)
        ax.annotate(lab, (px, py), textcoords="offset points",
                    xytext=off.get(lab, (8, 5)),
                    fontsize=10.5, fontweight="bold", color=c)
    # gender direction example: king->queen vs prince->princess parallels
    def proj(w):
        v = (g[w].astype(float) - X.mean(0)); return v @ Vt[:2].T
    for a, b in [("king", "queen"), ("prince", "princess")]:
        pa, pb = proj(a), proj(b)
        ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-|>", mutation_scale=13,
                     color=PURPLE, lw=1.8, ls="--", zorder=2, alpha=0.9))
    ax.text(0.50, 0.04, "dashed purple arrows = the shared male->female "
            "direction  (king->queen ~ prince->princess)",
            transform=ax.transAxes, ha="center",
            fontsize=9, color=PURPLE, va="bottom", fontweight="bold")
    ax.set_title("Real GloVe-50 vectors, PCA to 2-D: meaning clusters",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); _despine(ax)
    ax.grid(alpha=0.12)
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/we_pca_real.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote we_pca_real.png")


# ---- 3. Analogy parallelogram from REAL vectors -----------------------------
def analogy_parallelogram(g):
    # Project man, king, woman, queen + the computed analogy point to 2-D.
    anchor = ["man", "king", "woman", "queen"]
    X = np.stack([g[w].astype(float) for w in anchor])
    mean = X.mean(0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    def proj(vec):
        return (vec - mean) @ Vt[:2].T

    pm, pk, pw, pq = (proj(g[w].astype(float)) for w in anchor)
    analogy_vec = g["king"].astype(float) - g["man"].astype(float) + g["woman"].astype(float)
    pa = proj(analogy_vec)                  # king - man + woman, projected

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    pts = {"man": (pm, SLATE), "king": (pk, AMBER),
           "woman": (pw, RED), "queen": (pq, GREEN)}
    for lab, (p, c) in pts.items():
        ax.scatter(*p, s=150, color=c, edgecolor="#fff", linewidth=1.3, zorder=4)
        ax.annotate(lab, p, textcoords="offset points", xytext=(8, 6),
                    fontsize=11.5, fontweight="bold", color=c)
    # the two parallel "royalty" directions: man->king and woman->queen
    for a, b, c in [(pm, pk, AMBER), (pw, pq, GREEN)]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15,
                     color=c, lw=2.2, zorder=3))
    # the analogy computation point king-man+woman
    ax.scatter(*pa, s=170, facecolor="none", edgecolor=PURPLE, linewidth=2.6,
               zorder=5)
    ax.annotate("king - man + woman\n(lands ~ on queen)", pa,
                textcoords="offset points", xytext=(10, -28),
                fontsize=9.5, fontweight="bold", color=PURPLE)
    ax.add_patch(FancyArrowPatch(pw, pa, arrowstyle="-|>", mutation_scale=13,
                 color=PURPLE, lw=1.6, ls="--", zorder=2, alpha=0.8))
    ax.set_title("Analogy as a parallelogram (REAL GloVe vectors)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); _despine(ax)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/we_analogy_parallelogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote we_analogy_parallelogram.png")


# ---- 4. Negative sampling: pull / push + unigram^0.75 -----------------------
def negsampling():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- left: pull/push geometry ----
    vc = np.array([0.0, 0.0])
    pos = np.array([0.85, 0.45])
    negs = np.array([[-0.7, 0.6], [-0.5, -0.7], [0.3, -0.85]])
    axL.scatter(*vc, s=260, color=PURPLE, edgecolor="#fff", linewidth=1.4,
                zorder=4)
    axL.annotate("center v_c\n('king')", vc, textcoords="offset points",
                 xytext=(10, 8), fontsize=10, fontweight="bold", color=PURPLE)
    axL.scatter(*pos, s=220, color=GREEN, edgecolor="#fff", linewidth=1.4,
                zorder=4)
    axL.annotate("true context u_o\n('crown')  PULL", pos,
                 textcoords="offset points", xytext=(8, 6),
                 fontsize=9.5, fontweight="bold", color=GREEN)
    axL.add_patch(FancyArrowPatch(vc, pos * 0.92, arrowstyle="-|>",
                  mutation_scale=15, color=GREEN, lw=2.4, zorder=3))
    for i, n in enumerate(negs):
        axL.scatter(*n, s=190, color=RED, edgecolor="#fff", linewidth=1.3,
                    zorder=4)
        axL.add_patch(FancyArrowPatch(n * 0.5, n * 0.95, arrowstyle="-|>",
                      mutation_scale=13, color=RED, lw=2.0, zorder=3))
    axL.annotate("k sampled negatives\nPUSH away", negs[0],
                 textcoords="offset points", xytext=(-30, 14),
                 fontsize=9.5, fontweight="bold", color=RED)
    axL.set_xlim(-1.1, 1.3); axL.set_ylim(-1.15, 1.0); axL.axis("off")
    axL.set_title("Negative sampling: pull the real pair, push k negatives",
                  fontsize=12.5, fontweight="bold")

    # ---- right: unigram vs unigram^0.75 ----
    words = ["the", "of", "king", "queen", "kingdom"]
    counts = np.array([1000, 600, 40, 18, 6], dtype=float)
    p = counts / counts.sum()
    p75 = counts ** 0.75; p75 /= p75.sum()
    x = np.arange(len(words)); w = 0.38
    axR.bar(x - w / 2, p, w, color=SLATE, label="raw unigram  p(w)")
    axR.bar(x + w / 2, p75, w, color=AMBER, label="smoothed  p(w)^0.75")
    axR.set_xticks(x); axR.set_xticklabels(words, fontsize=9)
    axR.set_ylabel("sampling probability")
    axR.set_title("Why freq^0.75: lift rare words", fontsize=12.5,
                  fontweight="bold")
    axR.legend(frameon=False, fontsize=9); _despine(axR)
    axR.annotate("rare words sampled\nmore often", (4 + w / 2, p75[4]),
                 textcoords="offset points", xytext=(-58, 30), fontsize=8.5,
                 color=AMBER, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=AMBER))
    fig.tight_layout()
    fig.savefig(f"{OUT}/we_negsampling.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote we_negsampling.png")


if __name__ == "__main__":
    skipgram_vs_cbow()
    negsampling()
    print("loading real GloVe vectors (downloads once, then cached)...")
    g = _load_glove()
    pca_real(g)
    analogy_parallelogram(g)
    print("done.")
