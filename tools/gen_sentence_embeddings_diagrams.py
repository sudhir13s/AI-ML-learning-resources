"""Sentence & Document Embeddings concept-page diagrams (muted palette).

Four visuals for 06. NLP/concepts/07-Sentence-and-Document-Embeddings.md:
  1. sentemb_bi_vs_cross.png   -- SCHEMATIC: bi-encoder (siamese, cache-and-reuse,
     cosine in vector space) vs cross-encoder (joint attention, O(n^2) re-score).
  2. sentemb_projection.png    -- MEASURED: 2-D PCA projection of real SBERT
     sentence embeddings -> paraphrase clusters tight, unrelated topics far apart.
  3. sentemb_search.png        -- MEASURED: semantic-search ranking, cosine(query, doc)
     bars for a small corpus, correct doc ranked first.
  4. sentemb_anisotropy.png    -- SCHEMATIC: raw-BERT anisotropic cone (poor cosine
     separation) vs SBERT-tuned isotropic spread (clean separation).

Run with:  /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_sentence_embeddings_diagrams.py
Needs sentence-transformers + scikit-learn for the two MEASURED figures; if the
model can't be downloaded it falls back to a clearly-labelled constructed example.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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


def _box(ax, xy, w, h, text, fc, tc="#fff", fs=10, weight="bold"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec="none", zorder=2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight=weight, zorder=3)


def _arrow(ax, p0, p1, color="#666", lw=2.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, zorder=1))


# ---- 1. Bi-encoder vs cross-encoder schematic -------------------------------
def bi_vs_cross():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 5.4))

    # ---- bi-encoder (left)
    ax = axL
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Bi-encoder (Sentence-BERT)\nencode once -> cosine, reusable & fast",
                 fontsize=12.5, fontweight="bold", color=GREEN)
    _box(ax, (0.04, 0.80), 0.40, 0.11, "Sentence A", BLUE)
    _box(ax, (0.56, 0.80), 0.40, 0.11, "Sentence B", BLUE)
    _box(ax, (0.04, 0.58), 0.40, 0.13, "BERT encoder\n(shared weights)", PURPLE)
    _box(ax, (0.56, 0.58), 0.40, 0.13, "BERT encoder\n(shared weights)", PURPLE)
    _box(ax, (0.10, 0.39), 0.28, 0.10, "mean-pool -> u", SLATE)
    _box(ax, (0.62, 0.39), 0.28, 0.10, "mean-pool -> v", SLATE)
    _box(ax, (0.28, 0.16), 0.44, 0.12, "cos(u, v)", GREEN)
    for x in (0.24, 0.76):
        _arrow(ax, (x, 0.80), (x, 0.715))
        _arrow(ax, (x, 0.58), (x, 0.495))
    _arrow(ax, (0.24, 0.39), (0.42, 0.285))
    _arrow(ax, (0.76, 0.39), (0.58, 0.285))
    ax.text(0.5, 0.05, "u, v cached & indexed -> any pair compared in O(1)",
            ha="center", fontsize=9.5, style="italic", color=GREEN)
    ax.text(0.5, 0.515, "siamese: weights tied", ha="center", fontsize=8.6,
            color="#fff", fontweight="bold")

    # ---- cross-encoder (right)
    ax = axR
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Cross-encoder\njoint attention -> 1 score, must re-run per pair",
                 fontsize=12.5, fontweight="bold", color=RED)
    _box(ax, (0.10, 0.80), 0.80, 0.11, "[CLS] Sentence A [SEP] Sentence B", BLUE, fs=9.5)
    _box(ax, (0.10, 0.52), 0.80, 0.18, "BERT — A & B attend to each other\n(full cross-attention)", PURPLE, fs=10.5)
    _box(ax, (0.28, 0.27), 0.44, 0.12, "relevance score", RED)
    _arrow(ax, (0.5, 0.80), (0.5, 0.705))
    _arrow(ax, (0.5, 0.52), (0.5, 0.395))
    ax.text(0.5, 0.13, "more accurate, but N queries x M docs = N·M\nforward passes — no caching",
            ha="center", fontsize=9.5, style="italic", color=RED)
    fig.suptitle("Two ways to compare two texts — and why retrieval uses the bi-encoder",
                 fontsize=14, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sentemb_bi_vs_cross.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote sentemb_bi_vs_cross.png")


# ---- model helper (shared by measured figures) ------------------------------
def _load_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:  # pragma: no cover
        print("  [warn] model unavailable, using constructed fallback:", e)
        return None


# ---- 2. Measured 2-D projection of sentence embeddings ----------------------
def projection():
    groups = {
        "Weather": [
            "It is raining heavily outside today.",
            "The downpour soaked the whole city this morning.",
            "Heavy rain is flooding the streets right now.",
        ],
        "Finance": [
            "The stock market fell sharply this afternoon.",
            "Equity prices plunged amid the sell-off.",
            "Shares tumbled as investors fled the market.",
        ],
        "Cooking": [
            "She simmered the tomato sauce for an hour.",
            "Let the pasta sauce reduce slowly on low heat.",
            "He stirred the soup gently until it thickened.",
        ],
    }
    sents, labels = [], []
    for g, ss in groups.items():
        sents += ss; labels += [g] * len(ss)
    model = _load_model()
    if model is not None:
        emb = model.encode(sents, normalize_embeddings=True)
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=0).fit_transform(emb)
        measured = True
    else:  # constructed fallback, clearly labelled
        rng = np.random.default_rng(0)
        centers = {"Weather": (-2.4, 1.4), "Finance": (2.6, 1.0), "Cooking": (0.1, -2.4)}
        xy = np.array([np.array(centers[l]) + rng.normal(0, 0.32, 2) for l in labels])
        measured = False
    colmap = {"Weather": BLUE, "Finance": RED, "Cooking": GREEN}
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    for g in groups:
        m = [i for i, l in enumerate(labels) if l == g]
        ax.scatter(xy[m, 0], xy[m, 1], s=150, color=colmap[g], edgecolor="white",
                   linewidth=1.4, label=g, zorder=3)
        cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
        ax.scatter([cx], [cy], s=520, color=colmap[g], alpha=0.12, zorder=1)
    ax.legend(title="Topic", frameon=False, loc="best", fontsize=10.5, title_fontsize=11)
    tag = "PCA of real all-MiniLM-L6-v2 embeddings" if measured else "constructed (model unavailable)"
    ax.set_title(f"Sentence embeddings cluster by meaning, not words\n({tag})",
                 fontsize=13.5, fontweight="bold")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); _despine(ax)
    ax.text(0.5, -0.13, "Paraphrases land together; unrelated topics separate — distance ≈ meaning.",
            transform=ax.transAxes, ha="center", fontsize=9.6, style="italic", color="#555")
    fig.tight_layout()
    fig.savefig(f"{OUT}/sentemb_projection.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote sentemb_projection.png  (measured=%s)" % measured)


# ---- 3. Measured semantic-search ranking ------------------------------------
def search():
    query = "How do I reset my account password?"
    corpus = [
        "To change your password, open Settings and click 'Reset password'.",
        "Our refund policy allows returns within 30 days of purchase.",
        "The mobile app supports dark mode and push notifications.",
        "If you forgot your login, use the 'Forgot password' link on the sign-in page.",
        "Premium plans include priority support and extra storage.",
    ]
    model = _load_model()
    if model is not None:
        q = model.encode([query], normalize_embeddings=True)[0]
        d = model.encode(corpus, normalize_embeddings=True)
        sims = d @ q
        measured = True
    else:
        sims = np.array([0.71, 0.09, 0.05, 0.66, 0.12]); measured = True and False
    order = np.argsort(-sims)
    sims_s = sims[order]
    docs_s = [corpus[i] for i in order]
    short = [(s[:54] + "…") if len(s) > 55 else s for s in docs_s]
    cols = [GREEN if i < 2 else SLATE for i in range(len(sims_s))]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    y = np.arange(len(sims_s))[::-1]
    ax.barh(y, sims_s, color=cols, edgecolor="white", height=0.62, zorder=3)
    for yi, s in zip(y, sims_s):
        ax.text(s + 0.01, yi, f"{s:.3f}", va="center", fontsize=10, fontweight="bold", color="#333")
    ax.set_yticks(y); ax.set_yticklabels(short, fontsize=9.4)
    ax.set_xlabel("cosine(query, document)")
    tag = "real all-MiniLM-L6-v2" if measured else "constructed"
    ax.set_title(f"Semantic search: rank a corpus by cosine to the query\n"
                 f"query: “{query}”   ({tag})", fontsize=12.5, fontweight="bold")
    ax.set_xlim(0, max(sims_s) * 1.18); _despine(ax)
    ax.text(0.99, 0.04, "top-2 (green) are the password docs — keyword 'reset/forgot' OR paraphrase both win",
            transform=ax.transAxes, ha="right", fontsize=9, style="italic", color=GREEN)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sentemb_search.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote sentemb_search.png  (measured=%s)" % measured)


# ---- 4. Anisotropy before/after schematic -----------------------------------
def anisotropy():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 5.4))
    rng = np.random.default_rng(1)

    # Left: raw BERT — narrow cone (anisotropic), everything looks similar
    ang = np.deg2rad(rng.normal(38, 6, 60))
    r = rng.uniform(0.5, 1.0, 60)
    xa, ya = r * np.cos(ang), r * np.sin(ang)
    xa, ya = xa * 1.05, ya * 1.05
    axL.scatter(xa, ya, s=46, color=SLATE, alpha=0.85, edgecolor="white", linewidth=0.5)
    axL.plot([0, 1.45], [0, np.tan(np.deg2rad(30)) * 1.45], color=RED, lw=1.4, ls="--")
    axL.plot([0, 1.45], [0, np.tan(np.deg2rad(46)) * 1.45], color=RED, lw=1.4, ls="--")
    axL.annotate("narrow cone:\nall cosines high\n(can't tell apart)", (-0.85, -0.95),
                 fontsize=10.5, color=RED, fontweight="bold", ha="center")
    axL.set_title("Raw BERT (mean-pooled)\nanisotropic — embeddings crammed in a cone",
                  fontsize=12, fontweight="bold", color=RED)

    # Right: SBERT — spread out (isotropic), three separable clusters
    cents = [(-1.0, 0.9, BLUE), (1.0, 0.8, GREEN), (0.0, -0.9, AMBER)]
    for cx, cy, c in cents:
        pts = rng.normal([cx, cy], 0.26, (22, 2))
        axR.scatter(pts[:, 0], pts[:, 1], s=46, color=c, alpha=0.9, edgecolor="white", linewidth=0.5)
    axR.annotate("spread out:\nsimilar near, different far\n(cosine separates)", (0.0, -0.05),
                 fontsize=10.5, color=GREEN, fontweight="bold", ha="center")
    axR.set_title("After SBERT fine-tune\nisotropic — clusters cleanly separable",
                  fontsize=12, fontweight="bold", color=GREEN)

    for ax in (axL, axR):
        ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_edgecolor("#ccc")
    fig.suptitle("Why raw BERT vectors compare poorly — and what SBERT fixes (schematic)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sentemb_anisotropy.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote sentemb_anisotropy.png")


if __name__ == "__main__":
    bi_vs_cross()
    projection()
    search()
    anisotropy()
    print("all sentence-embedding diagrams written to", OUT)
