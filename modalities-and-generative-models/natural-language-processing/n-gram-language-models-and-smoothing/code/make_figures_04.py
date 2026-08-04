"""Reproducible figure generator for 04-N-gram-Language-Models-and-Smoothing.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
notebook -- the corpora, the smoothers, the counts, and every constant are IMPORTED from
`ngram_lm.py`, so the figures cannot silently drift from the prose or the demo. Run:

    python make_figures_04.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at 150 dpi,
prefixed `ng_`. The palette matches the chapter's Mermaid diagrams (muted, white text on fills).

Figures produced:
  ng_markov_window.png     -- the trigram Markov window sliding over a sentence (the n-gram idea)
  ng_zero_catastrophe.png  -- a full-sentence bigram probability collapsing to 0 from one unseen pair
  ng_zipf.png              -- the Zipfian long tail that GUARANTEES unseen n-grams (why smoothing exists)
  ng_freq_of_freqs.png     -- frequency-of-frequencies N_r on the animal bigrams (Good-Turing's raw data)
  ng_discount_curves.png   -- raw count vs absolute-discount (c-d) vs Good-Turing c* (the discounting idea)
  ng_smoothing_mass.png    -- next-word P after a context: MLE zero-cliff vs Laplace vs Kneser-Ney
  ng_perplexity_vs_n.png   -- held-out perplexity vs n under KN: the bias-variance turn
  ng_perplexity_vs_data.png-- bigram perplexity vs training fraction: Laplace vs Kneser-Ney

Verified on Python 3.12 / numpy 2.x / matplotlib 3.x. Pure-Python/NumPy, CPU, deterministic
(SEED from ngram_lm fixes the one shuffle).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from ngram_lm import (
    ANIMAL_TEXT,
    AddKBigram,
    KneserNeyOrderN,
    KneserNeyTrigram,
    animal_train_test,
    count_ngrams,
    frequency_of_frequencies,
    good_turing_reestimated_count,
    sentences_from_text,
    tokenize_toy,
    TOY_CORPUS,
)

# ---- Palette (matches the chapter Mermaid classDefs) ------------------------------------------
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
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
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
    print(f"wrote {path.relative_to(OUT_DIR.parent.parent)}")


def fig_markov_window() -> None:
    """The trigram Markov window sliding over a sentence: each prediction sees only the last two words."""
    tokens = ["<s>", "the", "cat", "sat", "on", "the", "mat", "</s>"]
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.set_xlim(-0.5, len(tokens) - 0.5)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    # Draw the token row.
    for i, tok in enumerate(tokens):
        ax.text(i, 0.5, tok, ha="center", va="center", fontsize=12, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="#EEF1F4", ec=SLATE, lw=1.0))
    # Highlight one trigram window predicting "sat" from ("the","cat").
    pred_i = 3  # predicting tokens[3] = "sat" from tokens[1:3]
    ctx = [pred_i - 2, pred_i - 1]
    for j in ctx:
        ax.add_patch(plt.Rectangle((j - 0.42, 0.1), 0.84, 0.8, fill=False, ec=BLUE, lw=2.4, zorder=5))
    ax.add_patch(plt.Rectangle((pred_i - 0.42, 0.1), 0.84, 0.8, fill=False, ec=GREEN, lw=2.6, zorder=5))
    ax.annotate("", xy=(pred_i, 1.05), xytext=((ctx[0] + ctx[1]) / 2, 1.05),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.0))
    ax.text((ctx[0] + ctx[1]) / 2, 2.0,
            r"context: previous 2 words" + "\n" + r"$w_{i-2},\,w_{i-1}$ = (the, cat)",
            ha="center", va="center", fontsize=10.5, color=BLUE)
    ax.text(pred_i + 0.1, 1.55, r"predict $w_i$ = sat" + "\n" + r"$P(\mathrm{sat}\mid\mathrm{the,\,cat})$",
            ha="center", va="center", fontsize=10.5, color=GREEN)
    ax.text(len(tokens) / 2 - 0.5, 3.05,
            "Trigram Markov window: each word is predicted from ONLY the previous two\n"
            "everything further left is ignored — that truncation is what makes the counts estimable",
            ha="center", va="center", fontsize=11, color=INK, fontweight="bold")
    _save(fig, "ng_markov_window.png")


def fig_zero_catastrophe() -> None:
    """One unseen bigram (Sam, do) multiplies the whole sentence probability to exactly zero."""
    toy = tokenize_toy(TOY_CORPUS)
    bi = count_ngrams(toy, 2)
    from ngram_lm import context_counts, mle_conditional

    ctx = context_counts(bi)
    # Two sentences: a seen one (nonzero) and one containing the unseen (Sam, do) bigram.
    seen = [("<s>", "I"), ("I", "am"), ("am", "Sam"), ("Sam", "</s>")]
    unseen = [("<s>", "Sam"), ("Sam", "do"), ("do", "not"), ("not", "</s>")]
    seen_p = [mle_conditional(b, (a,), bi, ctx) for a, b in seen]
    unseen_p = [mle_conditional(b, (a,), bi, ctx) for a, b in unseen]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for ax, pairs, probs, title, ok in (
        (ax1, seen, seen_p, "<s> I am Sam </s>   (all bigrams seen)", True),
        (ax2, unseen, unseen_p, "<s> Sam do not </s>   (one bigram unseen)", False),
    ):
        labels = [f"P({b}|{a})" for a, b in pairs]
        colors = [GREEN if p > 0 else RED for p in probs]
        ax.bar(range(len(probs)), probs, color=colors)
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("bigram probability")
        prod = np.prod(probs)
        sub = f"product = {prod:.3f}" if ok else f"product = {prod:.0f}  ← one zero kills the sentence"
        ax.set_title(f"{title}\n{sub}", fontsize=10, color=INK if ok else RED)
        if not ok:
            zero_idx = int(np.argmin(probs))
            ax.annotate("c(Sam, do) = 0", xy=(zero_idx, 0.02), xytext=(zero_idx, 0.55),
                        ha="center", color=RED, fontsize=9.5,
                        arrowprops=dict(arrowstyle="->", color=RED))
        _style_axis(ax)
    fig.suptitle("The zero-probability catastrophe: a product of probabilities is only as strong as its weakest factor",
                 fontsize=11.5, color=INK, y=1.02)
    _save(fig, "ng_zero_catastrophe.png")


def fig_zipf() -> None:
    """The Zipfian rank-frequency law on a corpus — why most n-grams are rare and many unseen."""
    # Use word unigrams of the animal corpus, plus an illustrative ideal Zipf line for contrast.
    sents = sentences_from_text(ANIMAL_TEXT)
    words = Counter(w for s in sents for w in s)
    freqs = np.array(sorted(words.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1)
    ideal = freqs[0] / ranks  # ideal Zipf: frequency ∝ 1/rank
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.loglog(ranks, freqs, color=BLUE, lw=0, marker="o", markersize=7,
              markeredgecolor="white", label="animal-corpus word frequencies")
    ax.loglog(ranks, ideal, color=RED, lw=2.0, ls="--", label=r"ideal Zipf  $f \propto 1/\mathrm{rank}$")
    ax.set_xlabel("word rank  (log scale)")
    ax.set_ylabel("frequency  (log scale)")
    ax.set_title("Word frequencies follow Zipf's law: a few words dominate, a long tail is rare\n"
                 "rare words → even rarer PAIRS → most plausible n-grams are simply never observed",
                 fontsize=10.5)
    ax.legend(frameon=False, loc="upper right")
    _style_axis(ax)
    _save(fig, "ng_zipf.png")


def fig_freq_of_freqs() -> None:
    """Frequency-of-frequencies N_r on the animal bigrams — the raw material of Good-Turing."""
    sents = sentences_from_text(ANIMAL_TEXT)
    bi = count_ngrams(sents, 2)
    nr = frequency_of_frequencies(bi)
    rs = sorted(nr)
    counts = [nr[r] for r in rs]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(rs, counts, color=PURPLE, edgecolor="white")
    bars[0].set_color(AMBER)  # highlight N_1, the singleton spike
    for r, c in zip(rs, counts):
        ax.text(r, c + 0.2, str(c), ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")
    n1, total = nr.get(1, 0), sum(bi.values())
    ax.set_xlabel("count  $r$  (times a bigram type appears)")
    ax.set_ylabel(r"$N_r$  (number of bigram TYPES with that count)")
    ax.set_title("Frequency of frequencies: singletons dominate ($N_1$ tallest)\n"
                 f"Good-Turing reserves $N_1/N = {n1}/{total} ≈ {n1 / total:.2f}$ of the mass for the unseen",
                 fontsize=10.5)
    ax.set_xticks(rs)
    _style_axis(ax)
    _save(fig, "ng_freq_of_freqs.png")


def fig_discount_curves() -> None:
    """Raw count vs absolute-discount (c-d) vs Good-Turing c* — three ways to shave mass off seen counts."""
    sents = sentences_from_text(ANIMAL_TEXT)
    bi = count_ngrams(sents, 2)
    nr = frequency_of_frequencies(bi)
    max_r = min(8, max(nr))
    rs = np.arange(1, max_r + 1)
    raw = rs.astype(float)
    d = 0.75
    abs_disc = np.maximum(rs - d, 0)
    gt = np.array([good_turing_reestimated_count(int(r), nr) for r in rs])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(rs, raw, color=SLATE, lw=2.0, ls="--", marker="o", markersize=6,
            markeredgecolor="white", label="raw count  $c$  (no discount)")
    ax.plot(rs, abs_disc, color=GREEN, lw=2.6, marker="s", markersize=6,
            markeredgecolor="white", label=r"absolute discount  $c - d$  ($d=0.75$)")
    ax.plot(rs, gt, color=AMBER, lw=2.2, marker="^", markersize=7,
            markeredgecolor="white", label=r"Good-Turing  $c^* = (c{+}1)\,N_{c+1}/N_c$")
    ax.set_xlabel("training count  $c$")
    ax.set_ylabel("reestimated count")
    ax.set_title("Discounting: every method shaves mass off seen counts to fund the unseen\n"
                 "absolute discount subtracts a clean constant; Good-Turing is jagged (noisy $N_r$ in the tail)",
                 fontsize=10.5)
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)
    _save(fig, "ng_discount_curves.png")


def fig_smoothing_mass() -> None:
    """Next-word distribution after a context: MLE zero-cliff vs Laplace flattening vs KN sharpness."""
    train, _ = animal_train_test()
    context_word = "the"
    # Candidate next words: a mix of seen continuations of "the" and a plausible-but-unseen one.
    candidates = ["cat", "dog", "mat", "log", "grass", "puppy", "rabbit"]
    bi = count_ngrams(train, 2)
    from ngram_lm import context_counts, add_k_conditional, mle_conditional, laplace_vocab_size

    ctx = context_counts(bi)
    vsize = laplace_vocab_size(train)
    kn = KneserNeyTrigram(train)
    mle = [mle_conditional(w, (context_word,), bi, ctx) for w in candidates]
    lap = [add_k_conditional(w, (context_word,), bi, ctx, vsize, k=1.0) for w in candidates]
    knp = [kn.p_bigram(w, context_word) for w in candidates]
    x = np.arange(len(candidates))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - width, mle, width, color=RED, label="MLE (raw counts)")
    ax.bar(x, lap, width, color=AMBER, label="Laplace (add-1)")
    ax.bar(x + width, knp, width, color=GREEN, label="Kneser-Ney")
    # Mark the zero cliff for the unseen continuation "rabbit".
    unseen_idx = candidates.index("rabbit")
    ax.annotate("MLE = 0 for the\nunseen 'the rabbit'", xy=(unseen_idx - width, 0.005),
                xytext=(unseen_idx - 0.2, max(mle) * 0.78), ha="center", va="center", color=RED, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=RED, connectionstyle="arc3,rad=-0.2"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"the {w}" for w in candidates], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(r"$P(\mathrm{word}\mid\mathrm{the})$")
    ax.set_title("Next-word probability after 'the': MLE zero-cliff vs Laplace vs Kneser-Ney\n"
                 "MLE zeros the unseen; Laplace lifts it but over-flattens; KN keeps the seen words sharp",
                 fontsize=10.5)
    ax.legend(frameon=False, loc="upper right")
    _style_axis(ax)
    _save(fig, "ng_smoothing_mass.png")


def fig_perplexity_vs_n() -> None:
    """Held-out perplexity vs n under interpolated Kneser-Ney: the bias-variance turn."""
    train, test = animal_train_test()
    orders = [1, 2, 3, 4]
    pps = [KneserNeyOrderN(train, n=n).perplexity(test) for n in orders]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(orders, pps, color=BLUE, lw=2.6, marker="o", markersize=9, markeredgecolor="white")
    ax.set_ylim(min(pps) - 1.2, max(pps) + 1.6)
    for n, pp in zip(orders, pps):
        dy = -18 if n == 1 else 13  # n=1 sits near the top; put its label BELOW the marker to clear the title
        ax.annotate(f"{pp:.2f}", xy=(n, pp), xytext=(0, dy), textcoords="offset points",
                    ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks(orders)
    ax.set_xticklabels(["unigram\n(n=1)", "bigram\n(n=2)", "trigram\n(n=3)", "4-gram\n(n=4)"])
    ax.set_xlabel("model order  $n$")
    ax.set_ylabel("held-out perplexity  (lower = better)")
    ax.set_title("Perplexity vs n (Kneser-Ney): a sharp drop, then diminishing returns\n"
                 "context helps (n=1→2); higher orders starve for counts — the bias-variance turn",
                 fontsize=10.0)
    _style_axis(ax)
    _save(fig, "ng_perplexity_vs_n.png")


def fig_perplexity_vs_data() -> None:
    """Bigram perplexity vs training fraction: Laplace vs Kneser-Ney as data grows."""
    train_full, test = animal_train_test()
    fractions = np.linspace(0.3, 1.0, 6)
    lap_pps, kn_pps = [], []
    for frac in fractions:
        n = max(2, int(len(train_full) * frac))
        subset = train_full[:n]
        lap_pps.append(AddKBigram(subset, k=1.0).perplexity(test))
        kn_pps.append(KneserNeyOrderN(subset, n=2).perplexity(test))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(fractions, lap_pps, color=AMBER, lw=2.6, marker="o", markersize=8,
            markeredgecolor="white", label="Laplace (add-1) bigram")
    ax.plot(fractions, kn_pps, color=GREEN, lw=2.6, marker="s", markersize=8,
            markeredgecolor="white", label="Kneser-Ney bigram")
    ax.set_xlabel("fraction of training data used")
    ax.set_ylabel("held-out bigram perplexity  (lower = better)")
    ax.set_title("Perplexity vs training size: Kneser-Ney ends below Laplace and falls faster\n"
                 "KN's continuation backoff makes better use of scarce counts as data grows",
                 fontsize=10.5)
    ax.legend(frameon=False, loc="upper right")
    _style_axis(ax)
    _save(fig, "ng_perplexity_vs_data.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    fig_markov_window()
    fig_zero_catastrophe()
    fig_zipf()
    fig_freq_of_freqs()
    fig_discount_curves()
    fig_smoothing_mass()
    fig_perplexity_vs_n()
    fig_perplexity_vs_data()
    print("done.")


if __name__ == "__main__":
    main()
