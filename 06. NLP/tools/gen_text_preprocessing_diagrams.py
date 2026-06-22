"""Generate the four PNG figures for the Text Preprocessing & Normalization concept page.

Run with the project's Python 3.12 env:
    /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 \
        "06. NLP/tools/gen_text_preprocessing_diagrams.py"

Figures written to 06. NLP/concepts/images/:
  1. textprep_pipeline.png        - schematic of the classic pipeline (text shrinking per stage)
  2. textprep_classical_vs_neural.png - MEASURED stopword/stem effect on vocab size + accuracy
  3. textprep_stem_vs_lemma.png    - stemming vs lemmatization comparison table-figure
  4. textprep_unicode_norm.png     - MEASURED Unicode normalization collapsing variants

Palette (muted, color:#fff on fills):
  BLUE #3A6B96  PURPLE #5D4A8A  GREEN #2E7A5A  RED #8B3B4A
  SLATE #4A5B6E AMBER #7A6528  NAVY #2A5B80
"""
import os
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----------------------------------------------------------------------------
# Palette
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#222831"

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "concepts", "images")
os.makedirs(IMG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
    "savefig.dpi": 150,
})


def save(fig, name):
    path = os.path.join(IMG, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")


# ============================================================================
# Figure 1 — the classic preprocessing pipeline (schematic, text shrinking)
# ============================================================================
def fig_pipeline():
    W = 14.5
    fig, ax = plt.subplots(figsize=(W, 6.6))
    ax.set_xlim(0, W)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    ax.text(W / 2, 6.35, "The classic preprocessing pipeline",
            ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
    ax.text(W / 2, 5.98, "each stage cleans, normalizes, or reduces — text shrinks toward fewer, denser units",
            ha="center", va="center", fontsize=10.5, color="#555", style="italic")

    stages = [
        ("Raw text", '"HELLO!!! Visit\nhttps://x.io cafes\n& NLP :) #AI"', BLUE),
        ("Clean / strip\nHTML, URLs,\nemoji", '"HELLO!!! Visit\ncafes & NLP AI"', NAVY),
        ("Unicode NFKC\n+ case-fold", '"hello!!! visit\ncafes & nlp ai"', PURPLE),
        ("Tokenize\n(see #02)", '[hello, !, visit,\ncafes, &, nlp, ai]', AMBER),
        ("Remove stopwords\n+ punctuation", '[hello, visit,\ncafes, nlp, ai]', SLATE),
        ("Stem /\nlemmatize", '[hello, visit,\ncafe, nlp, ai]', GREEN),
    ]

    n = len(stages)
    box_w = 1.95
    gap = (W - n * box_w) / (n + 1)
    y_box = 3.7
    box_h = 1.55
    centers = []
    for i, (title, _txt, color) in enumerate(stages):
        x = gap + i * (box_w + gap)
        cx = x + box_w / 2
        centers.append(cx)
        box = FancyBboxPatch((x, y_box), box_w, box_h,
                             boxstyle="round,pad=0.02,rounding_size=0.10",
                             linewidth=0, facecolor=color)
        ax.add_patch(box)
        ax.text(cx, y_box + box_h - 0.46, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white")
        ax.text(cx, y_box + 0.3, f"stage {i}", ha="center", va="center",
                fontsize=8.2, color="white", alpha=0.85)

    # arrows between boxes
    for i in range(n - 1):
        a = FancyArrowPatch((centers[i] + box_w / 2, y_box + box_h / 2),
                            (centers[i + 1] - box_w / 2, y_box + box_h / 2),
                            arrowstyle="-|>", mutation_scale=13,
                            linewidth=1.6, color="#888")
        ax.add_patch(a)

    # sample text under each box
    for i, (_t, txt, color) in enumerate(stages):
        ax.text(centers[i], 3.05, txt, ha="center", va="top",
                fontsize=7.4, family="monospace", color=INK,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="#f3f4f6",
                          edgecolor=color, linewidth=1.1))

    # the "signal vs sparsity" banner
    ax.annotate("", xy=(centers[-1], 0.6), xytext=(centers[0], 0.6),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.2))
    ax.text(W / 2, 1.0, "vocabulary shrinks  •  sparsity drops  •  but each stage can also delete signal",
            ha="center", va="center", fontsize=10, color=GREEN, fontweight="bold")
    ax.text(centers[0], 0.22, "noisy, many surface forms", ha="center", fontsize=8.2, color="#777")
    ax.text(centers[-1], 0.22, "clean, few canonical forms", ha="center", fontsize=8.2, color="#777")

    save(fig, "textprep_pipeline.png")


# ============================================================================
# Figure 2 — MEASURED: classical (stopword+stem) vs neural (minimal)
# ============================================================================
def fig_classical_vs_neural():
    """Measure vocabulary size and a classifier's accuracy under increasingly
    aggressive cleaning on a real dataset (20 Newsgroups, 4 categories)."""
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    cats = ["rec.sport.baseball", "sci.med", "comp.graphics", "talk.politics.guns"]
    train = fetch_20newsgroups(subset="train", categories=cats,
                               remove=("headers", "footers", "quotes"))
    test = fetch_20newsgroups(subset="test", categories=cats,
                              remove=("headers", "footers", "quotes"))

    sw = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tok_re = re.compile(r"[a-z]+")

    def clean(text, drop_stop, do_stem):
        toks = tok_re.findall(text.lower())
        if drop_stop:
            toks = [t for t in toks if t not in sw]
        if do_stem:
            toks = [stemmer.stem(t) for t in toks]
        return " ".join(toks)

    configs = [
        ("raw\n(lowercase only)", False, False),
        ("+ stopword\nremoval", True, False),
        ("+ stemming\n(Porter)", True, True),
    ]
    results = []
    for label, ds, st in configs:
        Xtr = [clean(t, ds, st) for t in train.data]
        Xte = [clean(t, ds, st) for t in test.data]
        clf = make_pipeline(TfidfVectorizer(min_df=2),
                            LogisticRegression(max_iter=1000, C=10))
        clf.fit(Xtr, train.target)
        vocab = len(clf.named_steps["tfidfvectorizer"].vocabulary_)
        acc = clf.score(Xte, test.target)
        results.append((label, vocab, acc))
        print(f"  classical {label!r:30s} vocab={vocab:6d} acc={acc:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.suptitle("Measured: aggressive cleaning helps classical TF-IDF, but is the wrong move for transformers",
                 fontsize=14, fontweight="bold", y=1.02)

    labels = [r[0] for r in results]
    vocabs = [r[1] for r in results]
    accs = [r[2] for r in results]
    colors = [BLUE, AMBER, GREEN]

    bars = ax1.bar(labels, vocabs, color=colors, width=0.62)
    ax1.set_title("Vocabulary size (TF-IDF features)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("# distinct terms")
    ax1.bar_label(bars, fmt="%d", padding=3, fontsize=10, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_ylim(0, max(vocabs) * 1.18)
    ax1.tick_params(axis="x", labelsize=9)

    bars2 = ax2.bar(labels, [a * 100 for a in accs], color=colors, width=0.62)
    ax2.set_title("Classifier accuracy (LogReg, 4-class)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("test accuracy (%)")
    ax2.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=10, fontweight="bold")
    lo = min(accs) * 100 - 4
    ax2.set_ylim(max(0, lo), 100)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="x", labelsize=9)

    # annotation banner contrasting the neural path
    fig.text(0.5, -0.06,
             "Classical path: stopword removal + stemming cut the vocabulary ~"
             f"{vocabs[0] / vocabs[-1]:.1f}× and keep accuracy — fewer, denser features help a linear model.\n"
             "Neural path (BERT/LLM): the subword tokenizer + pretrained model want raw-ish text — "
             "lowercasing & stemming DELETE casing/morphology signal the model uses, so you do almost none of this.",
             ha="center", fontsize=9.6, color=INK,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2f6", edgecolor=SLATE))

    save(fig, "textprep_classical_vs_neural.png")
    return results


# ============================================================================
# Figure 3 — stemming vs lemmatization comparison table-figure
# ============================================================================
def fig_stem_vs_lemma():
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    stemmer = PorterStemmer()
    lemm = WordNetLemmatizer()

    # word, POS-for-lemmatizer
    words = [
        ("studies", "n"),
        ("studying", "v"),
        ("better", "a"),
        ("caring", "v"),
        ("mice", "n"),
        ("ponies", "n"),
        ("running", "v"),
        ("organization", "n"),
        ("happily", "r"),
        ("was", "v"),
    ]
    rows = []
    for w, pos in words:
        s = stemmer.stem(w)
        l = lemm.lemmatize(w, pos=pos)
        differ = (s != l)
        rows.append((w, s, l, differ))
        print(f"  {w:14s} stem={s:12s} lemma={l:10s} differ={differ}")

    fig, ax = plt.subplots(figsize=(9.6, 7.4))
    ax.axis("off")
    ax.set_title("Stemming vs Lemmatization — same words, different answers",
                 fontsize=15, fontweight="bold", pad=16)

    headers = ["word", "Porter stem", "lemma (POS-aware)", ""]
    col_x = [0.06, 0.34, 0.62, 0.90]
    y0 = 0.92
    dy = 0.068

    # header row
    for cx, h in zip(col_x, headers):
        ax.text(cx, y0, h, fontsize=12, fontweight="bold", color="white",
                ha="left" if cx < 0.85 else "center",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=NAVY, edgecolor="none"))
    ax.plot([0.04, 0.97], [y0 - 0.035, y0 - 0.035], color="#bbb", lw=1)

    for i, (w, s, l, differ) in enumerate(rows):
        y = y0 - (i + 1) * dy
        row_color = "#fbeeee" if differ else "#eef5ef"
        ax.add_patch(FancyBboxPatch((0.04, y - 0.028), 0.93, 0.058,
                                    boxstyle="round,pad=0.005",
                                    facecolor=row_color, edgecolor="none", zorder=0))
        ax.text(col_x[0], y, w, fontsize=11, family="monospace", va="center")
        ax.text(col_x[1], y, s, fontsize=11, family="monospace", va="center",
                color=RED if differ else INK)
        ax.text(col_x[2], y, l, fontsize=11, family="monospace", va="center",
                color=GREEN if differ else INK)
        tag = "differ" if differ else "same"
        ax.text(col_x[3], y, tag, fontsize=9.5, va="center", ha="center",
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=RED if differ else GREEN, edgecolor="none"))

    ax.text(0.5, 0.04,
            "Stemmer = blind suffix chopping (fast, can over-stem: 'studies'->'studi', not a word).\n"
            "Lemmatizer = dictionary + POS -> a real lemma ('better'->'good', 'was'->'be', 'mice'->'mouse').",
            ha="center", fontsize=10, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef2f6", edgecolor=SLATE))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "textprep_stem_vs_lemma.png")
    return rows


# ============================================================================
# Figure 4 — MEASURED: Unicode normalization collapses variants
# ============================================================================
def fig_unicode_norm():
    # Strings that LOOK alike or are visually equivalent but differ in code points
    samples = [
        ("café", "café", "e + combining acute  vs  precomposed é"),
        ("ﬁle", "file", "ligature ﬁ (fi)  vs  f + i"),
        ("Ⅸ", "IX", "Roman numeral Ⅸ  vs  I + X"),
        ("½", "1⁄2", "½  vs  1 ⁄ 2"),
        ("²", "2", "superscript 2  vs  2"),
        ("ℋ", "H", "script capital ℋ  vs  H"),
    ]
    rows = []
    for raw, _expect, desc in samples:
        nfc = unicodedata.normalize("NFC", raw)
        nfkc = unicodedata.normalize("NFKC", raw)
        rows.append((raw, desc, len(raw), len(nfc), nfkc, len(nfkc)))
        print(f"  raw={raw!r:14s} len={len(raw)} NFC_len={len(nfc)} NFKC={nfkc!r} ({desc})")

    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    ax.axis("off")
    ax.set_title("Unicode normalization (NFKC) collapses look-alike variants into one form",
                 fontsize=14.5, fontweight="bold", pad=14)

    col_x = [0.05, 0.36, 0.70]
    headers = ["raw code points", "what it is", "after NFKC"]
    y0 = 0.85
    dy = 0.115
    for cx, h in zip(col_x, headers):
        ax.text(cx, y0, h, fontsize=11.5, fontweight="bold", color="white", ha="left",
                bbox=dict(boxstyle="round,pad=0.32", facecolor=PURPLE, edgecolor="none"))
    ax.plot([0.03, 0.97], [y0 - 0.045, y0 - 0.045], color="#bbb", lw=1)

    for i, (raw, desc, lraw, lnfc, nfkc, lnfkc) in enumerate(rows):
        y = y0 - (i + 1) * dy
        ax.add_patch(FancyBboxPatch((0.03, y - 0.045), 0.94, 0.092,
                                    boxstyle="round,pad=0.004",
                                    facecolor="#f3f4f6" if i % 2 == 0 else "#e9edf1",
                                    edgecolor="none", zorder=0))
        ax.text(col_x[0], y, f"{raw}", fontsize=15, family="DejaVu Sans", va="center",
                color=RED)
        ax.text(col_x[0] + 0.10, y, f"({lraw} cp)", fontsize=8.5, va="center", color="#888")
        ax.text(col_x[1], y, desc, fontsize=9.3, va="center", color=INK)
        ax.text(col_x[2], y, f"{nfkc}", fontsize=15, va="center", color=GREEN, fontweight="bold")
        ax.text(col_x[2] + 0.17, y, f"({lnfkc} cp)", fontsize=8.5, va="center", color="#888")

    ax.text(0.5, 0.04,
            "Without normalization, 'café' typed two ways are DIFFERENT strings → two vocabulary entries, "
            "split counts, missed search hits.\nNFKC folds compatibility variants (ligatures, superscripts, fractions) "
            "to a canonical form — the right default for search/indexing.",
            ha="center", fontsize=9.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef2f6", edgecolor=SLATE))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "textprep_unicode_norm.png")
    return rows


if __name__ == "__main__":
    print("Figure 1: pipeline schematic")
    fig_pipeline()
    print("Figure 2: classical vs neural (measured)")
    fig_classical_vs_neural()
    print("Figure 3: stemming vs lemmatization")
    fig_stem_vs_lemma()
    print("Figure 4: unicode normalization (measured)")
    fig_unicode_norm()
    print("All figures written.")
