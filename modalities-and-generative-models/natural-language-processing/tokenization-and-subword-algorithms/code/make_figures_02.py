"""Reproducible figure generator for 02-Tokenization-and-Subword-Algorithms.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
notebook -- the corpus, the BPE trainer, the WordPiece scorer, the Viterbi segmenter, and the
vocab-size sweep are all IMPORTED from tokenization.py, so the figures cannot silently drift
from the prose or the demo. The two "measured" figures (GPT-4 vs BERT, multilingual tax) call
the real tiktoken / transformers tokenizers so their counts are literally the production numbers.

    python make_figures_02.py

Each figure is written to ../../images/ (the shared chapter image dir, ``06. NLP/images/``) at
150 dpi, prefixed ``tok_``. The palette matches the chapter's Mermaid diagrams (muted, white
text on coloured fills).

Figures produced:
  tok_granularity_tradeoff.png -- word vs char vs subword across vocab / seq-len / OOV
  tok_bpe_merges.png           -- vocabulary growth: one token per merge, labelled with the pair
  tok_bpe_segmentation.png     -- unseen words segmented into reused subwords (the coverage proof)
  tok_wordpiece_vs_bpe.png     -- same corpus, BPE's raw-frequency pick vs WordPiece's score pick
  tok_unigram_viterbi.png      -- the Viterbi lattice choosing the most-probable segmentation
  tok_vocab_sweep.png          -- vocab size vs tokens/word (compression) and OOV-floor rate
  tok_gpt4_vs_bert.png         -- MEASURED: the same sentence under GPT-4 BPE and BERT WordPiece
  tok_multilingual_cost.png    -- MEASURED: tokens for the same meaning across four languages

Verified on Python 3.12 / numpy 2.x / matplotlib 3.x (tiktoken + transformers for the two
measured figures).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from tokenization import (
    TOY_CORPUS,
    bert_tokens,
    bpe_base_alphabet,
    encode_bpe,
    first_merge_bpe_vs_wordpiece,
    gpt4_tokens,
    multilingual_gpt4_counts,
    train_bpe,
    train_corpus_words,
    viterbi_segment,
    vocab_size_sweep,
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
    print(f"wrote {path}")


def fig_granularity_tradeoff() -> None:
    """Word vs character vs subword across the three goals: vocab size, seq length, OOV risk.

    Illustrative bars (not from a single corpus) -- the point is the SHAPE of the trade-off:
    word-level is short but huge-vocab + OOV; char-level is tiny-vocab + no-OOV but long;
    subword sits in the middle on all three. Normalised 0-1 so the three axes are comparable.
    """
    metrics = ["vocabulary size", "sequence length", "OOV risk"]
    # normalised 0..1 (higher = worse), illustrative of the qualitative ordering
    word = [1.00, 0.20, 1.00]   # huge vocab, short seqs, high OOV
    char = [0.05, 1.00, 0.00]   # tiny vocab, long seqs, zero OOV
    sub = [0.35, 0.40, 0.05]    # middle vocab, middle seqs, ~zero OOV
    x = np.arange(len(metrics))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.bar(x - width, word, width, color=RED, edgecolor="white", linewidth=0.6,
           label="word-level", zorder=3)
    ax.bar(x, char, width, color=AMBER, edgecolor="white", linewidth=0.6,
           label="character-level", zorder=3)
    ax.bar(x + width, sub, width, color=GREEN, edgecolor="white", linewidth=0.6,
           label="subword (the settlement)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("relative cost  (0 = best, 1 = worst)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Three granularities, one trade-off: subword is the only row low on all three")
    ax.legend(frameon=False, fontsize=10, ncol=3, loc="upper center")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "tok_granularity_tradeoff.png")


def fig_bpe_merges() -> None:
    """Vocabulary growth on the toy corpus: starts at the alphabet, +1 token per merge."""
    merges, freqs, _ = train_bpe(TOY_CORPUS, 10)
    base = len(bpe_base_alphabet(TOY_CORPUS))
    steps = list(range(len(merges) + 1))
    sizes = [base + s for s in steps]
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    ax.plot(steps, sizes, "-o", color=BLUE, linewidth=2.6, markersize=8, zorder=4)
    ax.axhline(base, color=SLATE, linestyle=":", linewidth=1.6, zorder=2)
    ax.annotate(f"base alphabet = {base} symbols", (0, base), textcoords="offset points",
                xytext=(8, -16), fontsize=9, color=SLATE)
    # label each merge with the pair and its frequency
    for i, (pair, freq) in enumerate(zip(merges, freqs), 1):
        label = f"{pair[0]}+{pair[1]}\n(×{freq})"
        ax.annotate(label, (i, sizes[i]), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=INK)
    ax.set_xlabel("merge step")
    ax.set_ylabel("vocabulary size (symbols)")
    ax.set_xticks(steps)
    ax.set_ylim(base - 1, base + len(merges) + 2)
    ax.set_title("BPE grows the vocabulary by exactly one token per merge (toy corpus)")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "tok_bpe_merges.png")


def fig_bpe_segmentation() -> None:
    """Unseen words segmented into reused subwords -- the no-[UNK] coverage guarantee, visualised.

    Each row is a held-out word; coloured cells are the BPE pieces (the end marker trimmed for
    readability). Multi-char pieces (learned subwords) are blue/green; single-char fallbacks are
    slate -- so you can SEE how much reuse each word gets vs how much falls to the character floor.
    """
    merges, _, _ = train_bpe(train_corpus_words(), 40)
    words = ["lowest", "newness", "renewing", "boldest", "widening", "qux"]
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    row_h = 1.0
    for r, word in enumerate(words):
        pieces = [p for p in encode_bpe(word, merges) if p != "</w>"]
        x = 0.0
        for piece in pieces:
            w = max(0.6, len(piece) * 0.5)  # width scales with characters in the piece
            color = SLATE if len(piece) == 1 else (GREEN if len(piece) >= 3 else BLUE)
            y = len(words) - 1 - r
            ax.add_patch(plt.Rectangle((x, y), w, row_h * 0.8, facecolor=color,
                                       edgecolor="white", linewidth=1.4, zorder=3))
            ax.text(x + w / 2, y + row_h * 0.4, piece, ha="center", va="center",
                    color="white", fontsize=11, zorder=4)
            x += w + 0.08
        ax.text(-0.25, len(words) - 1 - r + row_h * 0.4, word, ha="right", va="center",
                fontsize=11, color=INK, family="monospace")
    ax.set_xlim(-2.6, 10)
    ax.set_ylim(-0.4, len(words))
    ax.axis("off")
    # legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=GREEN, edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor=SLATE, edgecolor="white"),
    ]
    ax.legend(handles, ["learned subword (3+ chars)", "learned subword (2 chars)",
                        "single-char fallback"],
              loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=9)
    ax.set_title("Unseen words always covered: each reuses learned subwords, "
                 "falling to characters only where it must (no [UNK])",
                 color=INK, fontsize=12)
    fig.tight_layout()
    _save(fig, "tok_bpe_segmentation.png")


def fig_wordpiece_vs_bpe() -> None:
    """Same corpus, two merge criteria: BPE ranks by raw frequency, WordPiece by normalised score.

    Left bars: raw pair frequency (BPE's ranking) -- (e,s) wins. Right bars: WordPiece score
    freq(a,b)/(freq(a)·freq(b)) -- (i,d) wins despite being far rarer. The two winners are
    highlighted; everything else is muted. This is the cleanest proof the algorithms differ.
    """
    bpe_pick, wp_pick, details = first_merge_bpe_vs_wordpiece(TOY_CORPUS)
    # show the few pairs that matter for the contrast, in a fixed order
    pairs = [("e", "s"), ("s", "t"), ("l", "o"), ("n", "e"), ("w", "i"), ("i", "d")]
    pairs = [p for p in pairs if p in details]
    labels = [f"{a}+{b}" for a, b in pairs]
    freqs = [details[p]["pair_freq"] for p in pairs]
    scores = [details[p]["score"] for p in pairs]
    x = np.arange(len(pairs))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    bpe_colors = [RED if p == bpe_pick else SLATE for p in pairs]
    ax1.bar(x, freqs, color=bpe_colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax1.set_title(f"BPE: rank by raw frequency  →  picks {bpe_pick[0]}+{bpe_pick[1]}", fontsize=11)
    ax1.set_ylabel("pair frequency")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    _style_axis(ax1)

    wp_colors = [GREEN if p == wp_pick else SLATE for p in pairs]
    ax2.bar(x, scores, color=wp_colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax2.set_title(
        f"WordPiece: rank by freq(a,b)/(freq(a)·freq(b))  →  picks {wp_pick[0]}+{wp_pick[1]}",
        fontsize=11,
    )
    ax2.set_ylabel("WordPiece score")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    _style_axis(ax2)

    fig.suptitle(
        "Same corpus, different first merge: BPE rewards frequency, WordPiece rewards association",
        fontsize=13, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "tok_wordpiece_vs_bpe.png")


def fig_unigram_viterbi() -> None:
    """The Viterbi lattice: every candidate piece is an arc; the best (most-probable) path wins.

    For the string 'lowest' under a tiny Unigram model, draw each in-vocab piece as an arc from
    its start to its end position, annotated with log p. The chosen segmentation (low | est) is
    the highest-summed-log-prob path -- a GLOBAL optimum, contrasted with greedy merge replay.
    """
    text = "lowest"
    pieces_vocab = {"low": 0.30, "est": 0.30, "lo": 0.05, "we": 0.05,
                    "l": 0.05, "o": 0.05, "w": 0.05, "e": 0.05, "s": 0.05, "t": 0.05}
    logp = {p: math.log(v) for p, v in pieces_vocab.items()}
    best_seg, best_score = viterbi_segment(text, logp)

    # node positions: one per character boundary 0..len(text)
    n = len(text)
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ys = 0.0
    for i in range(n + 1):
        ax.plot(i, ys, "o", color=NAVY, markersize=12, zorder=5)
        ax.text(i, ys - 0.5, str(i), ha="center", va="center", fontsize=10, color=INK)
    # character labels between nodes
    for i, ch in enumerate(text):
        ax.text(i + 0.5, ys + 1.9, ch, ha="center", va="center", fontsize=13,
                color=INK, family="monospace")

    # reconstruct which arcs are on the best path
    best_arcs = set()
    pos = 0
    for piece in best_seg:
        best_arcs.add((pos, pos + len(piece)))
        pos += len(piece)

    # draw every in-vocab arc; highlight the winning path
    arc_level = {}
    for start in range(n):
        for end in range(start + 1, n + 1):
            piece = text[start:end]
            if piece not in logp:
                continue
            on_best = (start, end) in best_arcs
            color = GREEN if on_best else SLATE
            lw = 3.2 if on_best else 1.2
            alpha = 1.0 if on_best else 0.45
            height = 0.5 + 0.45 * (end - start) + 0.18 * arc_level.get((start, end), 0)
            mid = (start + end) / 2
            ax.annotate("", xy=(end, ys), xytext=(start, ys),
                        arrowprops=dict(arrowstyle="-", color=color, lw=lw, alpha=alpha,
                                        connectionstyle=f"arc3,rad={-0.35 if on_best else -0.18}"),
                        zorder=4 if on_best else 2)
            if on_best:
                ax.text(mid, ys + height + 0.15, f"'{piece}'  logp={logp[piece]:.2f}",
                        ha="center", va="bottom", fontsize=10, color=GREEN, zorder=6,
                        fontweight="bold")
    ax.set_xlim(-0.6, n + 0.6)
    ax.set_ylim(-1.2, 4.6)
    ax.axis("off")
    ax.set_title(
        f"Unigram Viterbi over 'lowest': the green path  {' | '.join(best_seg)}  "
        f"maximises summed log-prob ({best_score:.2f}) — a GLOBAL optimum, not greedy replay",
        color=INK, fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "tok_unigram_viterbi.png")


def fig_vocab_sweep() -> None:
    """Vocab size vs tokens/word (compression) and char-fallback rate (coverage quality).

    Two falling curves on a shared x-axis (vocabulary size). More vocab buys shorter sequences
    AND fewer character fallbacks -- but costs embedding-table parameters and per-token data
    sparsity. The crossover region is where the vocab-size decision actually lives.
    """
    corpus = train_corpus_words()
    held_out = ["lowness", "newness", "broadest", "renewing", "widening", "boldness"]
    merge_counts = [0, 2, 5, 10, 20, 40]
    rows = vocab_size_sweep(corpus, held_out, merge_counts)
    vocab = [r["vocab_size"] for r in rows]
    tpw = [r["tokens_per_word"] for r in rows]
    oov = [r["oov_fallback_rate"] * 100 for r in rows]

    fig, ax1 = plt.subplots(figsize=(10.0, 5.2))
    ax1.plot(vocab, tpw, "-o", color=BLUE, linewidth=2.8, markersize=8,
             label="tokens / word (compression)", zorder=4)
    ax1.set_xlabel("vocabulary size  (base alphabet + merges)")
    ax1.set_ylabel("tokens per word", color=BLUE)
    ax1.tick_params(axis="y", colors=BLUE)
    _style_axis(ax1)
    ax1.set_ylim(0, max(tpw) * 1.15)

    ax2 = ax1.twinx()
    ax2.plot(vocab, oov, "-s", color=RED, linewidth=2.8, markersize=8,
             label="char-fallback rate (held-out)", zorder=4)
    ax2.set_ylabel("% held-out words that fall to single characters", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.set_ylim(0, 105)
    ax2.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=10, loc="upper right")
    ax1.set_title("Bigger vocabulary → shorter sequences AND fewer fallbacks "
                  "(paid for in embedding params + data sparsity)")
    fig.tight_layout()
    _save(fig, "tok_vocab_sweep.png")


def _token_strip(ax: plt.Axes, tokens: list[str], color: str, title: str, y: float) -> None:
    """Render a horizontal strip of token boxes (spaces shown as ␣ glued to the token)."""
    x = 0.0
    for tok in tokens:
        shown = tok.replace(" ", "␣")
        w = max(0.7, len(shown) * 0.42)
        ax.add_patch(plt.Rectangle((x, y), w, 0.8, facecolor=color, edgecolor="white",
                                   linewidth=1.4, zorder=3))
        ax.text(x + w / 2, y + 0.4, shown, ha="center", va="center", color="white",
                fontsize=9.5, zorder=4)
        x += w + 0.08
    ax.text(-0.2, y + 0.4, title, ha="right", va="center", fontsize=11, color=INK)
    return x


def fig_gpt4_vs_bert() -> None:
    """MEASURED: the same sentence under GPT-4 byte-level BPE vs BERT WordPiece, token by token."""
    gpt = gpt4_tokens()
    bert = bert_tokens()
    fig, ax = plt.subplots(figsize=(13.2, 4.4))
    end_gpt = _token_strip(ax, gpt, BLUE, f"GPT-4 BPE\n({len(gpt)} tokens)", 1.2)
    end_bert = _token_strip(ax, bert, PURPLE, f"BERT WordPiece\n({len(bert)} tokens)", 0.0)
    ax.set_xlim(-2.4, max(end_gpt, end_bert) + 0.3)
    ax.set_ylim(-0.4, 2.3)
    ax.axis("off")
    ax.set_title(
        "Same sentence, two real tokenizers: GPT-4 keeps leading spaces (␣) and casing; "
        "BERT lowercases and marks continuations with ##",
        color=INK, fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "tok_gpt4_vs_bert.png")


def fig_multilingual_cost() -> None:
    """MEASURED: GPT-4 token count for the SAME meaning across four languages -- the tokenizer tax."""
    counts = multilingual_gpt4_counts()
    langs = [c[0] for c in counts]
    n_tokens = [c[1] for c in counts]
    baseline = n_tokens[0]
    colors = [GREEN, BLUE, AMBER, RED]
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    bars = ax.bar(langs, n_tokens, color=colors[: len(langs)], edgecolor="white",
                  linewidth=0.6, zorder=3)
    for bar, n in zip(bars, n_tokens):
        mult = n / baseline
        ax.text(bar.get_x() + bar.get_width() / 2, n + 0.4,
                f"{n} tok\n{mult:.1f}×", ha="center", va="bottom", fontsize=10, color=INK)
    ax.axhline(baseline, color=SLATE, linestyle=":", linewidth=1.6, zorder=2)
    ax.set_ylabel("GPT-4 tokens for the SAME sentence")
    ax.set_ylim(0, max(n_tokens) * 1.2)
    ax.set_title("The multilingual 'tokenizer tax': identical meaning, 2–4× the tokens off English")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "tok_multilingual_cost.png")


def main() -> None:
    fig_granularity_tradeoff()
    fig_bpe_merges()
    fig_bpe_segmentation()
    fig_wordpiece_vs_bpe()
    fig_unigram_viterbi()
    fig_vocab_sweep()
    fig_gpt4_vs_bert()
    fig_multilingual_cost()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
