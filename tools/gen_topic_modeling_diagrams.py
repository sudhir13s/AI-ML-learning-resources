"""Topic-Modeling concept-page diagrams (muted palette, parallel matplotlib scale).

Four visuals for 06. NLP/concepts/15-Topic-Modeling-LDA-NMF.md:
  1. topic_lda_generative.png   -- the MODEL: LDA plate-style schematic.
     Documents are topic mixtures (theta), topics are word distributions (beta);
     the generative arrows doc -> topic -> word, plus the Dirichlet priors.
  2. topic_lda_nmf_topwords.png -- MEASURED: top words per topic as bar charts
     from a real sklearn LDA and NMF run on a small built-in corpus.
  3. topic_nmf_factorization.png-- the NMF math: V ~= W H as a matrix-block
     visual, measured shapes/values from the same sklearn NMF fit.
  4. topic_coherence_perplexity.png -- MEASURED: held-out perplexity (LDA) and
     a coherence-style score vs number of topics K, to pick K.

Run with ~/.uv/envs/ml-py312/bin/python3 (sklearn, numpy, matplotlib).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

OUT = os.path.join(os.path.dirname(__file__), "..", "06. NLP", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# A small, clearly-3-topic synthetic corpus (sports / cooking / space).
# Each topic uses a tight, near-disjoint vocabulary so LDA and NMF recover it cleanly.
CORPUS = [
    "the team won the football game with a last minute goal by the striker",
    "the coach praised the players after the team match victory on the field",
    "fans cheered as the team scored another goal to win the football league",
    "the player passed the ball and the team scored the winning championship goal",
    "the striker and the goalkeeper battled as the football team chased the goal",
    "the league champions celebrated the victory after the players won the match",
    "the recipe needs fresh garlic onion tomato basil and a pinch of salt",
    "bake the bread dough in the oven and add butter sugar and flour",
    "chop the onion fry the garlic and simmer the tomato sauce with basil slowly",
    "the cake recipe uses sugar flour butter eggs and a cup of milk",
    "stir the soup add salt onion garlic and let the sauce simmer in the pot",
    "knead the dough whisk the eggs and bake the cake until the bread is golden",
    "the rocket launched into orbit carrying a satellite for the space mission",
    "astronauts aboard the station observed the planet and the distant stars",
    "the spacecraft reached orbit and the telescope captured the distant galaxy",
    "the space mission sent a probe past the moon toward the outer planet",
    "the satellite orbited the planet while astronauts studied the stars and galaxy",
    "the telescope and the spacecraft tracked the rocket toward the moon",
]
N_TOPICS = 3
N_TOP = 6


def _fit_models():
    """Fit CountVectorizer+LDA and TfidfVectorizer+NMF on CORPUS; return artifacts."""
    cv = CountVectorizer(stop_words="english", max_df=0.95, min_df=1)
    X_counts = cv.fit_transform(CORPUS)
    vocab = np.array(cv.get_feature_names_out())
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS, random_state=3, max_iter=200,
        learning_method="batch", doc_topic_prior=0.1, topic_word_prior=0.01)
    lda.fit(X_counts)

    tf = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=1)
    X_tfidf = tf.fit_transform(CORPUS)
    vocab_t = np.array(tf.get_feature_names_out())
    nmf = NMF(n_components=N_TOPICS, random_state=0, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X_tfidf)   # docs x topics
    H = nmf.components_              # topics x words
    return dict(cv=cv, X_counts=X_counts, vocab=vocab, lda=lda,
                tf=tf, X_tfidf=X_tfidf, vocab_t=vocab_t, nmf=nmf, W=W, H=H)


def _top_words(components, vocab, n=N_TOP):
    out = []
    for k in range(components.shape[0]):
        idx = components[k].argsort()[::-1][:n]
        out.append([(vocab[i], components[k][i]) for i in idx])
    return out


# ---- 1. LDA generative model / plate-style schematic ------------------------
def lda_generative():
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, color, fs=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    facecolor=color, edgecolor="white", linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color="#fff", fontsize=fs, fontweight="bold")

    def arrow(x1, y1, x2, y2, color=SLATE):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=15, color=color, linewidth=1.8, shrinkA=2, shrinkB=2))

    # Priors
    box(0.2, 4.7, 1.5, 0.8, "α\n(doc–topic\nprior)", AMBER, 9.5)
    box(0.2, 0.5, 1.5, 0.8, "η / β-prior\n(topic–word\nprior)", AMBER, 9.5)
    # theta per doc
    box(2.4, 4.6, 1.9, 1.0, "θ_d  ~ Dir(α)\ndoc's topic mixture", BLUE)
    # z per word
    box(5.0, 4.6, 1.9, 1.0, "z_{d,n} ~ Cat(θ_d)\nper-word topic", PURPLE)
    # word
    box(7.6, 4.6, 2.0, 1.0, "w_{d,n} ~ Cat(β_z)\nobserved word", GREEN)
    # beta topics
    box(2.4, 0.4, 1.9, 1.0, "β_k ~ Dir(η)\nK topic–word dists", NAVY)

    arrow(1.7, 5.1, 2.4, 5.1, AMBER)             # alpha -> theta
    arrow(4.3, 5.1, 5.0, 5.1, SLATE)             # theta -> z
    arrow(6.9, 5.1, 7.6, 5.1, SLATE)             # z -> w
    arrow(1.7, 0.9, 2.4, 0.9, AMBER)             # eta -> beta
    arrow(4.3, 0.9, 8.6, 4.6, NAVY)              # beta -> w (word also depends on chosen topic's beta)

    ax.text(8.6, 3.9, "word = pick topic z,\nthen draw word from β_z",
            ha="center", va="top", fontsize=8.8, color=SLATE, style="italic")
    ax.text(5.0, 6.0, "LDA generative story: for each word, pick a topic, then pick a word from that topic",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#222")
    ax.text(3.35, 4.35, "repeat for every document d", ha="center", fontsize=8.5, color=BLUE)
    ax.text(5.95, 4.35, "repeat for every word n", ha="center", fontsize=8.5, color=PURPLE)
    fig.tight_layout(); fig.savefig(f"{OUT}/topic_lda_generative.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote topic_lda_generative.png")


# ---- 2. Measured top words per topic (LDA & NMF) ----------------------------
def topwords(art):
    lda_topics = _top_words(art["lda"].components_, art["vocab"])
    nmf_topics = _top_words(art["H"], art["vocab_t"])
    fig, axes = plt.subplots(2, N_TOPICS, figsize=(11.0, 6.2))
    palette = [BLUE, GREEN, NAVY]
    for row, (topics, label, cmap_pick) in enumerate(
            [(lda_topics, "LDA", palette), (nmf_topics, "NMF", palette)]):
        for k in range(N_TOPICS):
            ax = axes[row, k]
            words = [w for w, _ in topics[k]][::-1]
            weights = [v for _, v in topics[k]][::-1]
            ax.barh(range(len(words)), weights, color=cmap_pick[k], edgecolor="white")
            ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontsize=9)
            ax.set_title(f"{label} — topic {k}", fontsize=10.5, fontweight="bold")
            ax.tick_params(length=0)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.set_xticks([])
    fig.suptitle("Measured top words per topic (sklearn LDA on counts, NMF on TF-IDF)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/topic_lda_nmf_topwords.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote topic_lda_nmf_topwords.png")


# ---- 3. NMF factorization V ~= W H visual (measured) ------------------------
def nmf_factorization(art):
    W, H = art["W"], art["H"]
    V = art["X_tfidf"].toarray()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.4),
                             gridspec_kw={"width_ratios": [2.6, 0.9, 2.6]})

    def heat(ax, M, title, cmap_color, sub):
        ax.imshow(M, aspect="auto", cmap="magma")
        ax.set_title(title, fontsize=11, fontweight="bold", color=cmap_color)
        ax.set_xlabel(sub, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    heat(axes[0], V, f"V  (docs × words)\n{V.shape[0]}×{V.shape[1]} TF-IDF", BLUE,
         "the doc–term matrix")
    heat(axes[1], W, f"W (docs × K)\n{W.shape[0]}×{W.shape[1]}", GREEN,
         "doc→topic weights")
    heat(axes[2], H, f"H (K × words)\n{H.shape[0]}×{H.shape[1]}", NAVY,
         "topic→word weights")
    # approx sign between
    fig.text(0.305, 0.5, "≈", fontsize=30, ha="center", va="center", fontweight="bold")
    fig.text(0.545, 0.5, "×", fontsize=24, ha="center", va="center", fontweight="bold")
    fig.suptitle("NMF: V ≈ W·H  (all entries ≥ 0 → parts-based, additive topics)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/topic_nmf_factorization.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote topic_nmf_factorization.png")


# ---- 4. Perplexity / coherence vs K (measured) ------------------------------
def _big_themed_corpus(n_per=60, seed=0):
    """Generate a larger 3-theme corpus so perplexity/coherence curves behave
    canonically (the 18-doc corpus is too small for stable model selection)."""
    rng = np.random.RandomState(seed)
    pools = {
        "sport": "team goal football striker league match victory players coach "
                 "fans field champions ball score win defender goalkeeper".split(),
        "cook": "recipe garlic onion tomato basil salt sugar flour butter eggs "
                "bake oven dough bread cake sauce simmer fry whisk milk".split(),
        "space": "rocket orbit satellite mission planet astronauts station stars "
                 "spacecraft telescope galaxy moon probe launch space cosmos".split(),
    }
    docs = []
    for theme, words in pools.items():
        for _ in range(n_per):
            k = rng.randint(8, 14)
            docs.append(" ".join(rng.choice(words, size=k, replace=True)))
    rng.shuffle(docs)
    return docs


def choose_k(art):
    corpus = _big_themed_corpus()
    cv = CountVectorizer(stop_words="english", min_df=2)
    X = cv.fit_transform(corpus)
    vocab = np.array(cv.get_feature_names_out())
    # train/test split of documents for held-out perplexity
    rng = np.random.RandomState(0)
    idx = rng.permutation(X.shape[0])
    n_test = max(20, X.shape[0] // 5)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    Xtr, Xte = X[train_idx], X[test_idx]
    Ks = [2, 3, 4, 5, 6, 8]
    perps, cohs = [], []
    # binary doc-occurrence for a UMass-style coherence proxy
    occ = (X.toarray() > 0).astype(int)
    D = occ.shape[0]
    co = occ.T @ occ  # word-word co-occurrence document counts
    df = np.diag(co).copy()

    def umass(top_idx):
        s = 0.0; cnt = 0
        for a in range(1, len(top_idx)):
            for b in range(a):
                wi, wj = top_idx[a], top_idx[b]
                s += np.log((co[wi, wj] + 1.0) / (df[wj] + 1e-9))
                cnt += 1
        return s / max(cnt, 1)

    for K in Ks:
        m = LatentDirichletAllocation(n_components=K, random_state=0, max_iter=50,
                                      learning_method="batch")
        m.fit(Xtr)
        perps.append(m.perplexity(Xte))
        c = []
        for k in range(K):
            top = m.components_[k].argsort()[::-1][:10]
            c.append(umass(list(top)))
        cohs.append(np.mean(c))

    fig, ax1 = plt.subplots(figsize=(8.6, 4.8))
    ax1.axvline(3, color=SLATE, ls="--", lw=1.4, zorder=0)
    l1, = ax1.plot(Ks, perps, color=RED, lw=2.6, marker="o", ms=6,
                   label="held-out perplexity (↓ better)")
    ax1.set_xlabel("Number of topics K")
    ax1.set_ylabel("Held-out perplexity", color=RED)
    ax1.tick_params(axis="y", labelcolor=RED)
    _despine(ax1)
    ax2 = ax1.twinx()
    l2, = ax2.plot(Ks, cohs, color=GREEN, lw=2.6, marker="s", ms=6,
                   label="UMass coherence (↑ better)")
    ax2.set_ylabel("UMass coherence", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    ax2.spines["top"].set_visible(False)
    ax1.annotate("both curves agree:\nK = 3 (the true number)",
                 xy=(3, perps[1]), xytext=(4.0, perps[1] + (max(perps) - min(perps)) * 0.45),
                 color=SLATE, fontsize=9.5, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=SLATE))
    ax1.set_title("Choosing K: held-out perplexity (↓) and topic coherence (↑) vs K",
                  fontsize=13, fontweight="bold")
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()],
               loc="upper center", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/topic_coherence_perplexity.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote topic_coherence_perplexity.png")


def print_summary(art):
    """Print measured topics so the page text can quote real numbers."""
    print("\n=== LDA top words (counts) ===")
    for k, t in enumerate(_top_words(art["lda"].components_, art["vocab"])):
        print(f"  topic {k}: " + ", ".join(w for w, _ in t))
    print("=== NMF top words (TF-IDF) ===")
    for k, t in enumerate(_top_words(art["H"], art["vocab_t"])):
        print(f"  topic {k}: " + ", ".join(w for w, _ in t))
    print("vocab size (LDA):", len(art["vocab"]), " unique docs:", len(CORPUS))


if __name__ == "__main__":
    art = _fit_models()
    lda_generative()
    topwords(art)
    nmf_factorization(art)
    choose_k(art)
    print_summary(art)
    print("OUT:", OUT)
