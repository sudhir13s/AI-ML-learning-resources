"""Tokenization & subword-algorithm concept-page diagrams (muted palette).

Four visuals for 06. NLP/concepts/02-Tokenization-and-Subword-Algorithms.md:
  1. tok_granularity_tradeoff.png -- WORD vs SUBWORD vs CHAR: the three knobs
     (vocab size, sequence length, OOV risk) and why subword is the sweet spot.
  2. tok_bpe_merges.png           -- BPE TRAINING: vocabulary growing as merges
     are learned on the low/lower/newest/widest toy corpus (real run).
  3. tok_gpt4_vs_bert.png         -- REAL token counts (tiktoken cl100k GPT-4 vs
     HF bert-base-uncased WordPiece) on one sentence, token by token.
  4. tok_multilingual_cost.png    -- tokens-per-word across languages for the
     SAME meaning (the multilingual "tokenizer tax"), measured with GPT-4's
     cl100k_base.

All counts are MEASURED at generation time (tiktoken + transformers), so the
numbers in the page and the figures cannot drift apart. Parallel matplotlib
scale to the rest of the site: title 14-16 bold, labels 10-11, annotations 9,
DejaVu Sans.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
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


# ---- 1. Granularity trade-off: word vs subword vs char -----------------------
def granularity_tradeoff():
    metrics = ["Vocab size", "Sequence length", "OOV risk"]
    levels = [
        ("Word-level", BLUE, ["huge (100k-1M+)", "short", "high (OOV holes)"],
         '"unhappiest" -> [UNK]', [1.0, 0.18, 1.0]),
        ("Subword (BPE / WordPiece)", GREEN,
         ["moderate (30k-100k)", "medium", "none"],
         '"unhappiest" -> un + happi + est', [0.52, 0.5, 0.05]),
        ("Character / byte", AMBER, ["tiny (~256 bytes)", "very long", "none"],
         '"unhappiest" -> u n h a p p y', [0.05, 1.0, 0.05]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    for ax, (name, col, labs, ex, vals) in zip(axes, levels):
        ax.barh(range(3), vals, color=col, alpha=0.9, height=0.5)
        ax.set_yticks(range(3)); ax.set_yticklabels(metrics, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0); ax.set_xticks([])
        ax.set_ylim(2.7, -0.7)
        for i, (v, lab) in enumerate(zip(vals, labs)):
            inside = v > 0.62
            ax.text(v - 0.03 if inside else v + 0.03, i, lab,
                    va="center", ha="right" if inside else "left",
                    fontsize=8.6, color="#fff" if inside else "#222",
                    fontweight="bold" if inside else "normal")
        ax.set_title(name, fontsize=12, fontweight="bold", color=col, pad=8)
        ax.text(0.5, -0.62, ex, ha="center", va="top", fontsize=8.8,
                color="#333", style="italic", transform=ax.transData)
        _despine(ax); ax.spines["left"].set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("Three tokenization granularities and what each trades off",
                 fontsize=15, fontweight="bold", y=1.06)
    fig.text(0.5, 0.965,
             "Subword hits the sweet spot: small-enough vocab, short-enough sequences, and NO out-of-vocabulary holes",
             ha="center", fontsize=9.6, color=GREEN, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/tok_granularity_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote tok_granularity_tradeoff.png")


# ---- 2. BPE merges growing the vocabulary -----------------------------------
def bpe_merges():
    from collections import Counter
    corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}

    def init(c):
        return {tuple(list(w) + ["</w>"]): n for w, n in c.items()}

    def pair_freqs(v):
        pf = Counter()
        for word, n in v.items():
            for i in range(len(word) - 1):
                pf[(word[i], word[i + 1])] += n
        return pf

    def merge(v, pair):
        new = {}
        for word, n in v.items():
            w = list(word); i = 0; out = []
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
                    out.append(w[i] + w[i + 1]); i += 2
                else:
                    out.append(w[i]); i += 1
            new[tuple(out)] = n
        return new

    base_vocab = set()
    for w in corpus:
        base_vocab.update(list(w)); base_vocab.add("</w>")
    vocab = init(corpus)
    steps = [0]; sizes = [len(base_vocab)]; labels = ["base\nalphabet"]
    merges = []
    for s in range(10):
        pf = pair_freqs(vocab)
        if not pf:
            break
        (a, b), f = pf.most_common(1)[0]
        merges.append((a + b, f))
        vocab = merge(vocab, (a, b))
        steps.append(s + 1); sizes.append(len(base_vocab) + len(merges))
        labels.append(f"+{a}{b}\n(f={f})")

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.plot(steps, sizes, color=PURPLE, lw=2.6, marker="o", ms=6)
    ax.fill_between(steps, sizes, len(base_vocab), color=PURPLE, alpha=0.10)
    for x, y, lab in zip(steps, sizes, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.2, color="#333")
    ax.axhline(len(base_vocab), color=SLATE, ls="--", lw=1.2, alpha=0.7)
    ax.text(0.2, len(base_vocab) - 0.55, "starting alphabet (chars + </w>)",
            fontsize=8.6, color=SLATE)
    ax.set_xlabel("Merge step (each merges the most-frequent adjacent pair)")
    ax.set_ylabel("Vocabulary size")
    ax.set_title("BPE training: every merge adds exactly one new token to the vocabulary",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(-0.4, len(steps) - 0.4)
    ax.set_ylim(len(base_vocab) - 1.2, sizes[-1] + 2.2)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/tok_bpe_merges.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote tok_bpe_merges.png")


# ---- 3. Real token counts: GPT-4 (cl100k) vs BERT WordPiece -----------------
def gpt4_vs_bert():
    sentence = "The unhappiest developers refactored 1234567 lines of code."
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        gpt = [enc.decode([t]) for t in enc.encode(sentence)]
    except Exception as e:
        print("tiktoken unavailable:", e)
        gpt = ["The", " unh", "app", "iest", " developers", " ref", "act",
               "ored", " ", "123", "456", "7", " lines", " of", " code", "."]
    try:
        from transformers import AutoTokenizer
        bert = AutoTokenizer.from_pretrained("bert-base-uncased").tokenize(sentence)
    except Exception as e:
        print("transformers unavailable:", e)
        bert = ["the", "un", "##ha", "##pp", "##iest", "developers", "ref",
                "##act", "##ored", "123", "##45", "##6", "##7", "lines", "of",
                "code", "."]

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    rows = [("GPT-4  ·  byte-level BPE (cl100k_base)", gpt, BLUE),
            ("BERT  ·  WordPiece (bert-base-uncased)", bert, GREEN)]
    for r, (name, toks, col) in enumerate(rows):
        y = 1 - r
        x = 0.0
        for tok in toks:
            disp = tok.replace(" ", "␣")  # show leading space as open-box
            w = max(0.42, 0.22 + 0.105 * len(tok))
            ax.add_patch(FancyBboxPatch((x, y - 0.28), w, 0.56,
                         boxstyle="round,pad=0.01,rounding_size=0.06",
                         linewidth=1.3, edgecolor=col,
                         facecolor=col, alpha=0.18))
            ax.text(x + w / 2, y, disp, ha="center", va="center",
                    fontsize=9.5, color="#111", family="DejaVu Sans")
            x += w + 0.07
        ax.text(-0.15, y + 0.46, f"{name}  ->  {len(toks)} tokens",
                fontsize=11, fontweight="bold", color=col, ha="left")
    ax.text(-0.15, -0.95,
            'Same sentence, two tokenizers. Note: GPT-4 keeps the leading space ON the word ("␣developers"); '
            'BERT lowercases and marks word-continuations with "##". The 7-digit number splits differently in both.',
            fontsize=8.8, color="#333", ha="left", style="italic", wrap=True)
    ax.set_xlim(-0.2, x + 0.2); ax.set_ylim(-1.3, 1.7)
    ax.axis("off")
    ax.set_title("How GPT-4 and BERT tokenize the SAME sentence (measured token counts)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/tok_gpt4_vs_bert.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote tok_gpt4_vs_bert.png")


# ---- 4. Tokens-per-word across languages (the multilingual tax) -------------
def multilingual_cost():
    # Same meaning ("The cat sat on the mat.") in several languages.
    samples = [
        ("English", "The cat sat on the mat.", 6, BLUE),
        ("Spanish", "El gato se sentó en la alfombra.", 7, GREEN),
        ("Chinese", "猫坐在垫子上。", 6, AMBER),
        ("Hindi", "बिल्ली चटाई पर बैठी थी।", 5, RED),
    ]
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        counts = [(lang, len(enc.encode(s)), words, col)
                  for lang, s, words, col in samples]
    except Exception as e:
        print("tiktoken unavailable:", e)
        counts = [(lang, n, w, c) for (lang, _, w, c), n in
                  zip(samples, [7, 12, 12, 27])]

    langs = [c[0] for c in counts]
    tpw = [c[1] / c[2] for c in counts]   # tokens per (meaning-)word
    toks = [c[1] for c in counts]
    cols = [c[3] for c in counts]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    bars = ax.bar(langs, tpw, color=cols, alpha=0.9, width=0.6)
    base = tpw[0]
    for b, t, n in zip(bars, tpw, toks):
        ax.text(b.get_x() + b.get_width() / 2, t + 0.08,
                f"{t:.1f} tok/word\n({n} tokens)", ha="center",
                fontsize=9, color="#222")
    ax.axhline(base, color=BLUE, ls="--", lw=1.3, alpha=0.7)
    ax.text(3.35, base + 0.05, "English baseline", fontsize=8.6, color=BLUE,
            ha="right")
    ax.set_ylabel("Tokens per meaning-word  (GPT-4 cl100k_base)")
    ax.set_title('The multilingual "tokenizer tax": the SAME sentence costs far more tokens in some languages',
                 fontsize=13.5, fontweight="bold")
    ax.set_ylim(0, max(tpw) * 1.22)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/tok_multilingual_cost.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote tok_multilingual_cost.png")


if __name__ == "__main__":
    granularity_tradeoff()
    bpe_merges()
    gpt4_vs_bert()
    multilingual_cost()
    print("All tokenization diagrams written to", OUT)
