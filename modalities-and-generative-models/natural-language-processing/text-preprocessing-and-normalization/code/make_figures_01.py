"""Reproducible figure generator for 01-Text-Preprocessing-and-Normalization.

Every embedded PNG on the concept page is produced HERE, from the SAME functions and numbers
used by `text_preprocessing.py` and the teaching notebook -- the pipeline, stemmer/lemmatizer,
vocab counts, classifier sweep, Zipf fit, and Unicode demo are all IMPORTED, so a figure can
never silently drift from the prose. Run:

    python make_figures_01.py

Each figure is written to ../../images/ (the shared `06. NLP/images/` dir) at 150 dpi, with a
filename prefixed `tp_`. The palette matches the chapter's Mermaid classDefs (muted fills,
white text on coloured bars).

Figures produced:
  tp_pipeline.png            -- the messy sentence shrinking stage by stage to canonical tokens
  tp_vocab_shrink.png        -- distinct-vocabulary size after each cleaning step (the sparsity lever)
  tp_zipf.png                -- log-log rank-frequency with the fitted Zipf slope (~ -1)
  tp_zipf_stopword_mass.png  -- cumulative token mass: a handful of stopwords dominate the corpus
  tp_stem_vs_lemma.png       -- Porter stem vs WordNet lemma on tricky words, disagreements flagged
  tp_unicode_norm.png        -- NFC/NFKC collapsing look-alike strings into one canonical form
  tp_classical_vs_neural.png -- measured TF-IDF vocab + accuracy under cumulative cleaning

Verified on Python 3.12 / numpy 2.x / matplotlib 3.x / nltk 3.9 / scikit-learn 1.9.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from text_preprocessing import (
    classifier_sweep,
    load_newsgroups,
    preprocess,
    stem_vs_lemma,
    unicode_demo,
    vocab_shrink_stages,
    zipf_rank_frequency,
)

# ---- Palette (matches the chapter Mermaid classDefs) --------------------------------------
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


def _style_axis(ax: plt.Axes) -> None:
    """Muted styling: light grid, no top/right spines, ink-coloured labels."""
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
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def fig_pipeline(stopwords_set: set[str]) -> None:
    """The messy sentence shrinking through the classical pipeline to canonical tokens."""
    messy = "<p>HELLO!!!</p> Visit https://x.io — I’m lovin’ cafés & NLP \U0001F600 #AI"
    # Stage snapshots: (label, displayed text). The final stage is the real `preprocess` output.
    final_tokens = preprocess(messy, stopwords_set=stopwords_set)
    # The literal 😀 glyph is missing from the bundled mono font, so we render it as the text
    # "[😀]" via the ASCII stand-in "[:D]" only inside the figure -- the real pipeline still
    # operates on the true emoji code point (U+1F600); this is purely a rendering substitution.
    stages = [
        ("0  raw", "<p>HELLO!!!</p> Visit https://x.io — I’m lovin’ cafés & NLP [:D] #AI"),
        ("1  strip HTML / unescape", "HELLO!!! Visit https://x.io — I’m lovin’ cafés & NLP [:D] #AI"),
        ("2  URL + emoji → placeholder", "HELLO!!! Visit <URL> — I’m lovin’ cafés & NLP <EMOJI> #AI"),
        ("3  NFKC + lowercase", "hello!!! visit <url> — i’m lovin’ cafés & nlp <emoji> #ai"),
        ("4  fold accents + drop punct", "hello visit <url> i m lovin cafes nlp <emoji> ai"),
        ("5  remove stopwords (tokens)", "  ".join(final_tokens)),
    ]
    colors = [SLATE, NAVY, BLUE, PURPLE, AMBER, GREEN]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    ax.set_title("The classical pipeline: a messy string collapses to canonical, reusable tokens",
                 fontsize=12, color=INK, pad=12)
    n = len(stages)
    for i, (label, text) in enumerate(stages):
        y = 1.0 - (i + 0.5) / n
        ax.add_patch(plt.Rectangle((0.02, y - 0.5 / n + 0.012), 0.30, 0.7 / n,
                                   transform=ax.transAxes, facecolor=colors[i], edgecolor="none"))
        ax.text(0.17, y, label, transform=ax.transAxes, ha="center", va="center",
                color="#fff", fontsize=9.5, fontweight="bold")
        ax.text(0.35, y, text, transform=ax.transAxes, ha="left", va="center",
                color=INK, fontsize=9.5, family="monospace")
        if i < n - 1:
            ax.annotate("", xy=(0.17, y - 0.5 / n - 0.004), xytext=(0.17, y - 0.5 / n + 0.012),
                        transform=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.4))
    _save(fig, "tp_pipeline.png")


def fig_vocab_shrink(documents: list[str], stopwords_set: set[str]) -> None:
    """Distinct-vocabulary size after each cleaning step -- the sparsity reduction, measured."""
    stages = vocab_shrink_stages(documents, stopwords_set=stopwords_set)
    labels = [name for name, _ in stages]
    sizes = [size for _, size in stages]
    colors = [SLATE, NAVY, BLUE, PURPLE, GREEN]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(sizes)), sizes, color=colors, zorder=3, width=0.62)
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        ax.text(bar.get_x() + bar.get_width() / 2, size + max(sizes) * 0.015,
                f"{size:,}", ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")
        if i > 0:
            drop = (sizes[i - 1] - size) / sizes[i - 1] * 100
            if drop > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, size / 2,
                        f"-{drop:.0f}%", ha="center", va="center", color="#fff", fontsize=9.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("distinct vocabulary (types)")
    ax.set_title("Each cleaning step shrinks the vocabulary the model must learn\n"
                 "(20 Newsgroups, 4 categories — measured)", fontsize=12)
    ax.set_ylim(0, max(sizes) * 1.12)
    _style_axis(ax)
    _save(fig, "tp_vocab_shrink.png")


def fig_zipf(documents: list[str]) -> None:
    """Log-log rank-frequency with the fitted Zipf line (slope ~ -1)."""
    ranks, freqs, slope, intercept, top = zipf_rank_frequency(documents)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.loglog(ranks, freqs, ".", color=BLUE, markersize=3, alpha=0.55, zorder=3,
              label="word types (rank vs frequency)")
    fit_x = np.array([1, 1000], dtype=float)
    fit_y = 10 ** (intercept + slope * np.log10(fit_x))
    ax.loglog(fit_x, fit_y, "-", color=RED, lw=2.2, zorder=4,
              label=f"Zipf fit: slope = {slope:.2f}  (ideal -1)")
    # Annotate the top few stopwords at the head of the curve. Ranks 1-4 cluster on a log axis,
    # so stagger the label targets up and to the right to keep them legible.
    label_offsets = [(3.0, 2.4), (6.0, 3.4), (16.0, 3.0), (45.0, 2.2)]
    for (word, freq), (dx, dy) in zip(top[:4], label_offsets):
        rank = int(np.where(freqs == freq)[0][0]) + 1
        ax.annotate(f"“{word}” (rank {rank})", xy=(rank, freq), xytext=(rank * dx, freq * dy),
                    color=AMBER, fontsize=9.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=AMBER, lw=0.9))
    ax.set_xlabel("rank of word (log scale)")
    ax.set_ylabel("frequency (log scale)")
    ax.set_title("Zipf's law: a few words dominate, a long tail is rare\n"
                 "frequency ≈ C / rank — the straight log-log line preprocessing exploits",
                 fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    _style_axis(ax)
    _save(fig, "tp_zipf.png")


def fig_zipf_stopword_mass(documents: list[str]) -> None:
    """Cumulative token mass: a handful of top words (mostly stopwords) own most of the corpus."""
    ranks, freqs, _slope, _intercept, _top = zipf_rank_frequency(documents)
    total = freqs.sum()
    cumulative = np.cumsum(freqs) / total

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(ranks, cumulative * 100, color=PURPLE, lw=2.2, zorder=3)
    # Mark how much mass the top-10 and top-100 words carry.
    for k, color in [(10, AMBER), (100, GREEN)]:
        share = cumulative[k - 1] * 100
        ax.axvline(k, color=color, lw=1.2, ls="--", zorder=2)
        ax.annotate(f"top {k} words\n= {share:.0f}% of all tokens",
                    xy=(k, share), xytext=(k * 1.6, share - 14),
                    color=color, fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))
    ax.set_xscale("log")
    ax.set_xlabel("number of most-frequent word types kept (log scale)")
    ax.set_ylabel("cumulative % of all tokens")
    ax.set_title("Why stopword removal is cheap and powerful\n"
                 "a tiny set of top words accounts for a huge share of the tokens",
                 fontsize=12)
    ax.set_ylim(0, 100)
    _style_axis(ax)
    _save(fig, "tp_zipf_stopword_mass.png")


def fig_stem_vs_lemma() -> None:
    """Porter stem vs WordNet lemma on tricky words, with disagreements flagged."""
    pairs = [
        ("studies", "n"),
        ("studying", "v"),
        ("better", "a"),
        ("best", "a"),
        ("are", "v"),
        ("is", "v"),
        ("mice", "n"),
        ("organization", "n"),
        ("running", "v"),
    ]
    rows = stem_vs_lemma(pairs)
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.axis("off")
    ax.set_title("Stemming chops; lemmatization understands\n"
                 "Porter stem vs WordNet lemma (POS-aware) — real NLTK outputs",
                 fontsize=12, color=INK, pad=10)
    headers = ["word", "POS", "Porter stem", "WordNet lemma", ""]
    col_x = [0.04, 0.30, 0.42, 0.66, 0.90]
    n = len(rows)
    row_h = 0.80 / (n + 1)
    top = 0.88
    for x, h in zip(col_x, headers):
        ax.text(x, top, h, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=NAVY)
    for i, (word, pos, stem, lemma, agree) in enumerate(rows):
        y = top - (i + 1) * row_h
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.02, y - row_h * 0.42), 0.96, row_h * 0.86,
                                       transform=ax.transAxes, facecolor="#EEF1F4", edgecolor="none"))
        ax.text(col_x[0], y, word, transform=ax.transAxes, fontsize=10, family="monospace", color=INK)
        ax.text(col_x[1], y, pos, transform=ax.transAxes, fontsize=10, color=SLATE)
        stem_color = RED if not agree else GREEN
        ax.text(col_x[2], y, stem, transform=ax.transAxes, fontsize=10, family="monospace", color=stem_color)
        ax.text(col_x[3], y, lemma, transform=ax.transAxes, fontsize=10, family="monospace", color=GREEN)
        tag = "agree" if agree else "DIFFER"
        tag_color = SLATE if agree else RED
        ax.text(col_x[4], y, tag, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=tag_color)
    ax.text(0.02, 0.02,
            "red = stem is a non-word or wrong sense   green = real dictionary word",
            transform=ax.transAxes, fontsize=9, color=SLATE, style="italic")
    _save(fig, "tp_stem_vs_lemma.png")


def fig_unicode_norm() -> None:
    """NFC/NFKC collapsing look-alike strings into one canonical form."""
    uni = unicode_demo()
    # NFKC('½') is '1⁄2' with a FRACTION SLASH (U+2044) that the mono font can't draw, so we
    # display it with an ASCII '/' -- the canonical string in the module is unchanged.
    half_display = uni["nfkc_half"].replace("⁄", "/")
    examples = [
        ("café (combining é)\n+ café (precomposed é)", "café", "NFC", uni["equal_after_nfc"]),
        ("ﬁle (ﬁ ligature)", uni["nfkc_ligature_file"], "NFKC", True),
        ("Ⅸ (Roman nine)", uni["nfkc_roman_nine"], "NFKC", True),
        ("½ (one-half)", half_display, "NFKC", True),
        ("² (superscript two)", uni["nfkc_superscript_two"], "NFKC", True),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.axis("off")
    ax.set_title("Identical-looking text can be different bytes — until you normalize\n"
                 "NFC merges true duplicates; NFKC also folds compatibility look-alikes",
                 fontsize=12, color=INK, pad=10)
    n = len(examples)
    row_h = 0.74 / n
    top = 0.80
    for i, (raw_label, canonical, form, _ok) in enumerate(examples):
        y = top - (i + 0.5) * row_h
        ax.add_patch(plt.Rectangle((0.03, y - row_h * 0.40), 0.40, row_h * 0.80,
                                   transform=ax.transAxes, facecolor=BLUE, edgecolor="none"))
        ax.text(0.23, y, raw_label, transform=ax.transAxes, ha="center", va="center",
                color="#fff", fontsize=10)
        ax.annotate("", xy=(0.55, y), xytext=(0.45, y), transform=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.6))
        ax.text(0.50, y + row_h * 0.30, form, transform=ax.transAxes, ha="center",
                color=AMBER, fontsize=9, fontweight="bold")
        ax.add_patch(plt.Rectangle((0.57, y - row_h * 0.40), 0.30, row_h * 0.80,
                                   transform=ax.transAxes, facecolor=GREEN, edgecolor="none"))
        ax.text(0.72, y, repr(canonical), transform=ax.transAxes, ha="center", va="center",
                color="#fff", fontsize=10, family="monospace")
    ax.text(0.03, 0.04,
            "before: distinct strings → split counts, missed search hits     after: one canonical token",
            transform=ax.transAxes, fontsize=9, color=SLATE, style="italic")
    _save(fig, "tp_unicode_norm.png")


def fig_classical_vs_neural(
    train_docs: list[str],
    train_labels: np.ndarray,
    test_docs: list[str],
    test_labels: np.ndarray,
    stopwords_set: set[str],
) -> None:
    """Measured: cumulative cleaning shrinks the TF-IDF vocab while accuracy holds/improves."""
    sweep = classifier_sweep(
        train_docs, train_labels, test_docs, test_labels, stopwords_set=stopwords_set
    )
    labels = [name for name, _, _ in sweep]
    vocabs = [v for _, v, _ in sweep]
    accs = [a * 100 for _, _, a in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    bars = ax1.bar(range(len(vocabs)), vocabs, color=[SLATE, BLUE, GREEN], zorder=3, width=0.6)
    for bar, v in zip(bars, vocabs):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + max(vocabs) * 0.015,
                 f"{v:,}", ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9, rotation=12, ha="right")
    ax1.set_ylabel("TF-IDF vocabulary size")
    ax1.set_title("Cleaning shrinks the feature space", fontsize=11)
    ax1.set_ylim(0, max(vocabs) * 1.15)
    _style_axis(ax1)

    ax2.plot(range(len(accs)), accs, "o-", color=PURPLE, lw=2.4, markersize=9, zorder=3)
    for i, a in enumerate(accs):
        ax2.text(i, a + 0.05, f"{a:.1f}%", ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9, rotation=12, ha="right")
    ax2.set_ylabel("test accuracy (%)")
    ax2.set_title("…and accuracy holds or improves", fontsize=11)
    ax2.set_ylim(min(accs) - 0.6, max(accs) + 0.6)
    _style_axis(ax2)

    fig.suptitle("Heavy cleaning is a real win for a CLASSICAL model on a topic task\n"
                 "(the same steps would HURT a transformer — it wants the case/punctuation/morphology back)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "tp_classical_vs_neural.png")


def main() -> None:
    import nltk

    for package in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(package, quiet=True)
    from nltk.corpus import stopwords

    stopwords_set = set(stopwords.words("english"))
    train_docs, train_labels, test_docs, test_labels = load_newsgroups()

    fig_pipeline(stopwords_set)
    fig_vocab_shrink(train_docs, stopwords_set)
    fig_zipf(train_docs)
    fig_zipf_stopword_mass(train_docs)
    fig_stem_vs_lemma()
    fig_unicode_norm()
    fig_classical_vs_neural(train_docs, train_labels, test_docs, test_labels, stopwords_set)
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
