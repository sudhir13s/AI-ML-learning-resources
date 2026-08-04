"""Generate the four TF-IDF / BM25 / cosine diagrams for the
Bag-of-Words & TF-IDF concept page, plus print the measured numbers
the markdown quotes (so the hand-worked examples stay in lock-step
with sklearn). Run with the ml-py312 env:

    /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 \
        "06. NLP/tools/gen_tfidf_diagrams.py"

Verified on Python 3.12, scikit-learn 1.9, numpy 2.4, matplotlib.
Outputs PNGs into ../concepts/images/ (relative to this file).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- palette
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#2A2A2A"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "axes.edgecolor": SLATE,
        "axes.linewidth": 0.9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.normpath(os.path.join(HERE, "..", "concepts", "images"))
os.makedirs(IMG, exist_ok=True)


def save(fig, name):
    path = os.path.join(IMG, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------- corpus
# Tiny 3-document corpus used verbatim in the hand-worked examples.
DOCS = [
    "the cat sat on the mat",          # D1
    "the dog sat on the log",          # D2
    "the happy cat chased the dog",    # D3
]
DOC_LABELS = ["D1", "D2", "D3"]


def tokenize(d):
    return d.lower().split()


def build_vocab(docs):
    vocab = sorted({w for d in docs for w in tokenize(d)})
    return vocab


# ======================================================================
# Figure 1 — document-term TF-IDF heatmap (measured via sklearn)
# ======================================================================
def fig1_heatmap():
    from sklearn.feature_extraction.text import TfidfVectorizer

    # sklearn defaults: smoothed idf = ln((1+N)/(1+df)) + 1, L2-normalized rows
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", norm="l2")
    X = vec.fit_transform(DOCS).toarray()
    terms = vec.get_feature_names_out()

    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    cmap = LinearSegmentedColormap.from_list("bg", ["#EEF2F5", BLUE, NAVY])
    im = ax.imshow(X, aspect="auto", cmap=cmap, vmin=0, vmax=X.max())

    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(terms, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(DOC_LABELS)))
    ax.set_yticklabels(DOC_LABELS, fontsize=11, fontweight="bold")

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            v = X[i, j]
            if v > 0:
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if v > X.max() * 0.45 else INK,
                )
    ax.set_title("TF-IDF document-term matrix (L2-normalized rows)")
    ax.set_xlabel("vocabulary term")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("tf-idf weight", fontsize=9)
    save(fig, "tfidf_matrix_heatmap.png")
    return vec, X, terms


# ======================================================================
# Figure 2 — idf vs document frequency curve (measured)
# ======================================================================
def fig2_idf_curve():
    N = 1000  # illustrative collection size
    df = np.arange(1, N + 1)
    idf_plain = np.log(N / df)                 # Sparck-Jones / textbook idf
    idf_smooth = np.log((1 + N) / (1 + df)) + 1  # sklearn smoothed idf

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(df, idf_plain, color=BLUE, lw=2.4, label=r"$\log(N/df_t)$  (textbook)")
    ax.plot(df, idf_smooth, color=PURPLE, lw=2.4, ls="--",
            label=r"$\log\frac{1+N}{1+df_t}+1$  (smoothed, sklearn)")

    # annotate a rare and a common term
    ax.annotate(
        "rare term (df=1):\nhigh idf", xy=(1, np.log(N / 1)),
        xytext=(120, 5.4), fontsize=9, color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc=GREEN, ec="none"),
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4),
    )
    ax.annotate(
        "near-universal (df=900):\nidf → 0", xy=(900, np.log(N / 900)),
        xytext=(540, 2.0), fontsize=9, color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc=RED, ec="none"),
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
    )
    ax.axhline(0, color=SLATE, lw=0.8, ls=":")
    ax.set_xlabel("document frequency  $df_t$  (docs containing term, N=1000)")
    ax.set_ylabel("inverse document frequency  idf$_t$")
    ax.set_title("IDF down-weights common terms, rewards rare ones")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.set_xlim(0, N)
    ax.grid(True, alpha=0.25)
    save(fig, "tfidf_idf_curve.png")


# ======================================================================
# Figure 3 — BM25 tf-saturation vs raw tf (k1 effect), measured
# ======================================================================
def fig3_bm25_saturation():
    tf = np.linspace(0, 20, 400)
    # BM25 tf component with b=0 (no length norm), varying k1
    def bm25_tf(tf, k1):
        return (tf * (k1 + 1)) / (tf + k1)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    # raw tf (TF-IDF style), normalized so we can compare shapes
    ax.plot(tf, tf, color=RED, lw=2.2, ls="--", label="raw tf (TF-IDF): linear, unbounded")
    for k1, col in [(0.5, GREEN), (1.2, BLUE), (3.0, AMBER)]:
        ax.plot(tf, bm25_tf(tf, k1), color=col, lw=2.4,
                label=fr"BM25 tf, $k_1={k1}$  (saturates → ${k1+1:.1f}$)")
        ax.axhline(k1 + 1, color=col, lw=0.8, ls=":")
    ax.set_xlabel("term frequency in document  (count of query term)")
    ax.set_ylabel("contribution to score (tf component)")
    ax.set_title("BM25 saturates term frequency; raw TF-IDF does not")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.25)
    save(fig, "tfidf_bm25_saturation.png")


# ======================================================================
# Figure 4 — measured retrieval ranking bar chart (cosine TF-IDF vs BM25)
# ======================================================================
def fig4_retrieval_ranking():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the happy cat chased the dog",
        "a quick brown fox jumps over the lazy dog",
        "cats and dogs are common household pets",
    ]
    labels = ["D1", "D2", "D3", "D4", "D5"]
    query = "happy cat"

    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", norm="l2")
    X = vec.fit_transform(corpus)
    qv = vec.transform([query])
    cos = cosine_similarity(qv, X).ravel()

    # BM25 scores (Okapi, k1=1.5, b=0.75)
    bm25 = bm25_scores(corpus, query, k1=1.5, b=0.75)

    order = np.argsort(-cos)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y = np.arange(len(corpus))[::-1]
    cos_o = cos[order]
    bm_o = bm25[order]
    lab_o = [labels[i] for i in order]

    h = 0.38
    ax.barh(y + h / 2, cos_o, height=h, color=BLUE, label="cosine TF-IDF")
    # normalize BM25 to [0,1] for visual comparison on same axis
    bm_norm = bm_o / (bm_o.max() if bm_o.max() > 0 else 1)
    ax.barh(y - h / 2, bm_norm, height=h, color=GREEN, label="BM25 (normalized)")

    for yi, (c, b, lab) in enumerate(zip(cos_o, bm_norm, lab_o)):
        ax.text(c + 0.01, y[yi] + h / 2, f"{c:.2f}", va="center", fontsize=8.5, color=BLUE)
        ax.text(b + 0.01, y[yi] - h / 2, f"{bm_o[yi]:.2f}", va="center", fontsize=8.5, color=GREEN)

    ax.set_yticks(y)
    ax.set_yticklabels(lab_o, fontweight="bold")
    ax.set_xlabel("relevance score")
    ax.set_title(f'Ranking documents for query: "{query}"')
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.set_xlim(0, 1.15)
    ax.grid(True, axis="x", alpha=0.25)
    save(fig, "tfidf_retrieval_ranking.png")
    return corpus, labels, query, cos, bm25


# ---------------------------------------------------------------- BM25 helper
def bm25_scores(corpus, query, k1=1.5, b=0.75):
    """Okapi BM25 with the Robertson/Sparck-Jones idf used by Lucene/Elasticsearch:
        idf = ln(1 + (N - df + 0.5)/(df + 0.5))
    """
    docs = [c.lower().split() for c in corpus]
    N = len(docs)
    avgdl = np.mean([len(d) for d in docs])
    qterms = query.lower().split()
    df = {t: sum(1 for d in docs if t in d) for t in qterms}
    scores = np.zeros(N)
    for i, d in enumerate(docs):
        dl = len(d)
        s = 0.0
        for t in qterms:
            f = d.count(t)
            if f == 0:
                continue
            idf = np.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores[i] = s
    return scores


# ======================================================================
# Print the measured numbers the markdown quotes (hand-check anchor)
# ======================================================================
def print_measured():
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    print("\n==== MEASURED NUMBERS (markdown must match) ====")
    print("Corpus:")
    for lab, d in zip(DOC_LABELS, DOCS):
        print(f"  {lab}: {d!r}")

    vocab = build_vocab(DOCS)
    print(f"\nVocabulary ({len(vocab)}): {vocab}")

    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    counts = cv.fit_transform(DOCS).toarray()
    print(f"\nCount (BoW) matrix, terms={list(cv.get_feature_names_out())}")
    for lab, row in zip(DOC_LABELS, counts):
        print(f"  {lab}: {row.tolist()}")

    # --- hand idf (textbook, no smoothing) for the worked example
    N = len(DOCS)
    print(f"\nN = {N}")
    for t in ["the", "cat", "dog", "happy", "mat"]:
        df = sum(1 for d in DOCS if t in tokenize(d))
        idf_plain = np.log(N / df)
        idf_smooth = np.log((1 + N) / (1 + df)) + 1
        print(f"  term {t!r}: df={df}  idf_plain=ln({N}/{df})={idf_plain:.4f}"
              f"   idf_smooth=ln({1+N}/{1+df})+1={idf_smooth:.4f}")

    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", norm=None, smooth_idf=True)
    Xraw = vec.fit_transform(DOCS).toarray()
    terms = vec.get_feature_names_out()
    print(f"\nsklearn raw tf-idf (norm=None, smooth_idf=True), terms={list(terms)}")
    for lab, row in zip(DOC_LABELS, Xraw):
        print(f"  {lab}: {np.round(row,4).tolist()}")

    vecn = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", norm="l2")
    Xn = vecn.fit_transform(DOCS).toarray()
    print(f"\nsklearn L2-normalized tf-idf, terms={list(terms)}")
    for lab, row in zip(DOC_LABELS, Xn):
        print(f"  {lab}: {np.round(row,4).tolist()}")

    cos = cosine_similarity(Xn)
    print("\nCosine similarity (L2 tf-idf):")
    print("       " + "   ".join(DOC_LABELS))
    for lab, row in zip(DOC_LABELS, cos):
        print(f"  {lab}: " + "  ".join(f"{v:.3f}" for v in row))

    # BM25 worked numbers for the figure-4 corpus
    print("\n---- BM25 retrieval corpus ----")
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the happy cat chased the dog",
        "a quick brown fox jumps over the lazy dog",
        "cats and dogs are common household pets",
    ]
    q = "happy cat"
    bm = bm25_scores(corpus, q, k1=1.5, b=0.75)
    for lab, d, s in zip(["D1", "D2", "D3", "D4", "D5"], corpus, bm):
        print(f"  {lab} BM25={s:.4f}  {d!r}")

    # saturation illustration numbers
    print("\n---- tf saturation (k1=1.5, b=0) ----")
    for f in [1, 2, 3, 5, 10, 50]:
        sat = (f * (1.5 + 1)) / (f + 1.5)
        print(f"  tf={f:>2}: raw={f:>3}   BM25_tf={sat:.4f}  (ceiling k1+1=2.5)")


if __name__ == "__main__":
    print("Generating TF-IDF diagrams ->", IMG)
    fig1_heatmap()
    fig2_idf_curve()
    fig3_bm25_saturation()
    fig4_retrieval_ranking()
    print_measured()
    print("\nDone.")
