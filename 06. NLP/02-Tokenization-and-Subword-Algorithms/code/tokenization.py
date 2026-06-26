"""From-scratch subword tokenization: BPE training, WordPiece scoring, Unigram Viterbi, and
the downstream trade-offs (vocab size vs sequence length vs OOV) -- one seeded source of truth.

Everything a model needs to turn text into integers, built from the standard library so it is
deterministic and inspectable: train BPE on a tiny corpus and watch the ordered merge list form,
encode brand-new words by replaying the merges, score the same pair the BPE way (raw frequency)
versus the WordPiece way (normalised likelihood), Viterbi-decode the best segmentation under a
Unigram model, and sweep vocabulary size to expose the compression/OOV trade-off.

The concept page (../02-Tokenization-and-Subword-Algorithms.md), the teaching notebook
(02-Tokenization-and-Subword-Algorithms.ipynb), and the figure generator (make_figures_02.py)
all import these SAME functions, so the prose, the diagrams, and the demo can never drift.

Device: cpu (pure-Python/numpy) -- the subword algorithms are integer/string bookkeeping with no
tensors, so there is nothing to accelerate; results are bit-for-bit identical on any machine. The
real-tokenizer helpers (tiktoken / transformers) are optional and only used to reproduce the
measured GPT-4-vs-BERT and multilingual-tax figures.

Run:
    python tokenization.py
"""

from __future__ import annotations

import math
import random
from collections import Counter

# ---- Determinism -------------------------------------------------------------------------
# The BPE/WordPiece/Unigram routines are fully deterministic (greedy argmax / DP), but we seed
# anyway so any future sampling-based extension (e.g. subword regularization) stays reproducible.
SEED = 0
random.seed(SEED)

# ---- The canonical toy corpus (Sennrich et al. 2016, the standard teaching example) -------
# word -> frequency. Small enough to redo every step by hand, rich enough to surface a shared
# suffix (-est) and a shared stem (low-).
TOY_CORPUS: dict[str, int] = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
END = "</w>"  # end-of-word marker: lets the model tell word-final "est" from word-internal "est"
N_MERGES = 10  # merges to learn on the toy corpus -- exactly the count worked through on the page


# =========================================================================================
# BPE -- byte-pair encoding (Sennrich, Haddow & Birch 2016)
# =========================================================================================
def to_symbols(corpus: dict[str, int]) -> dict[tuple[str, ...], int]:
    """Represent each word as a tuple of characters plus the end-of-word marker.

    The vocabulary starts as the set of characters; BPE grows it by merging.
    """
    return {tuple(list(word) + [END]): count for word, count in corpus.items()}


def pair_freqs(vocab: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    """Count every adjacent symbol pair across the corpus, weighted by word frequency."""
    freqs: Counter[tuple[str, str]] = Counter()
    for word, count in vocab.items():
        for i in range(len(word) - 1):
            freqs[(word[i], word[i + 1])] += count  # weight by how often the word occurs
    return freqs


def apply_merge(
    vocab: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int]:
    """Replace every adjacent occurrence of `pair` with its concatenation, in every word."""
    a, b = pair
    merged = a + b
    new_vocab: dict[tuple[str, ...], int] = {}
    for word, count in vocab.items():
        symbols = list(word)
        out: list[str] = []
        i = 0
        while i < len(symbols):
            # if the next two symbols are exactly the pair, glue them into one token
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        new_vocab[tuple(out)] = count
    return new_vocab


def train_bpe(
    corpus: dict[str, int], n_merges: int
) -> tuple[list[tuple[str, str]], list[int], dict[tuple[str, ...], int]]:
    """Greedily merge the most-frequent adjacent pair `n_merges` times.

    Returns (ordered merge list, per-step frequency of each merged pair, final segmented corpus).
    The merge LIST -- in order -- is the saved artifact: encoding replays it top to bottom.
    Ties on frequency are broken deterministically by Counter.most_common (insertion order),
    which is itself fixed by the corpus iteration order, so the run is fully reproducible.
    """
    vocab = to_symbols(corpus)
    merges: list[tuple[str, str]] = []
    freqs_at_merge: list[int] = []
    for _ in range(n_merges):
        freqs = pair_freqs(vocab)
        if not freqs:
            break
        best_pair, best_freq = freqs.most_common(1)[0]  # greedy: locally most-frequent pair
        merges.append(best_pair)
        freqs_at_merge.append(best_freq)
        vocab = apply_merge(vocab, best_pair)
    return merges, freqs_at_merge, vocab


def encode_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Encode a (possibly unseen) word by replaying the learned merges IN ORDER.

    This is the coverage guarantee in action: a never-seen word reuses learned pieces, and in
    the worst case falls back to single characters, which are always in the base vocabulary --
    so there is no `[UNK]`. Note this is a greedy replay of GLOBAL rules, not a search for the
    locally-best split of THIS word (the inflexibility Unigram LM was designed to fix).
    """
    symbols = list(word) + [END]
    for a, b in merges:  # replay merges top to bottom -- order is part of the model
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols[i : i + 2] = [a + b]  # glue the pair; do not advance, allow chained merges
            else:
                i += 1
    return symbols


def bpe_base_alphabet(corpus: dict[str, int]) -> set[str]:
    """The starting vocabulary: every character that appears, plus the end-of-word marker."""
    alphabet = {ch for word in corpus for ch in word}
    alphabet.add(END)
    return alphabet


# =========================================================================================
# WordPiece -- likelihood-scored merges (Schuster & Nakajima 2012; Wu et al. 2016)
# =========================================================================================
def symbol_freqs(vocab: dict[tuple[str, ...], int]) -> Counter[str]:
    """Count every individual symbol across the corpus, weighted by word frequency.

    This is the denominator WordPiece needs: how often each piece occurs ON ITS OWN.
    """
    freqs: Counter[str] = Counter()
    for word, count in vocab.items():
        for symbol in word:
            freqs[symbol] += count
    return freqs


def wordpiece_score(
    pair: tuple[str, str],
    pair_freq: int,
    sym_freq: Counter[str],
) -> float:
    """WordPiece merge score = freq(a,b) / (freq(a) * freq(b)).

    Rewards a pair that is frequent TOGETHER but rare APART (high pointwise mutual information),
    rather than just frequent -- the key contrast with BPE's raw-frequency criterion.
    """
    a, b = pair
    return pair_freq / (sym_freq[a] * sym_freq[b])


def first_merge_bpe_vs_wordpiece(
    corpus: dict[str, int],
) -> tuple[tuple[str, str], tuple[str, str], dict[tuple[str, str], dict[str, float]]]:
    """On the SAME corpus, return (BPE's first merge, WordPiece's first merge, per-pair details).

    The whole point: they disagree. BPE picks the highest raw-frequency pair; WordPiece picks the
    highest normalised-score pair, which can be a much rarer pair whose halves always co-occur.
    """
    vocab = to_symbols(corpus)
    pfreqs = pair_freqs(vocab)
    sfreqs = symbol_freqs(vocab)
    bpe_pick = pfreqs.most_common(1)[0][0]  # raw frequency
    details: dict[tuple[str, str], dict[str, float]] = {}
    for pair, pfreq in pfreqs.items():
        a, b = pair
        details[pair] = {
            "pair_freq": float(pfreq),
            "denom": float(sfreqs[a] * sfreqs[b]),
            "score": wordpiece_score(pair, pfreq, sfreqs),
        }
    wp_pick = max(details, key=lambda p: details[p]["score"])  # normalised likelihood score
    return bpe_pick, wp_pick, details


# =========================================================================================
# Unigram LM -- probabilistic segmentation via Viterbi (Kudo 2018)
# =========================================================================================
def viterbi_segment(
    text: str, vocab_logp: dict[str, float]
) -> tuple[list[str], float]:
    """Most-probable segmentation of `text` under a Unigram model, via dynamic programming.

    A Unigram model assigns each piece an independent log-probability; a segmentation's score is
    the SUM of its pieces' log-probs. Viterbi fills best[i] = the best score to reach position i,
    considering every vocabulary piece that ends there -- the GLOBAL optimum for this string, not
    a greedy merge replay. Returns (pieces, total log-prob). Pieces not in the vocab are skipped,
    so every single character must be present for full coverage (Unigram always keeps them).
    """
    n = len(text)
    best = [-math.inf] * (n + 1)  # best[i]: best log-prob to segment text[:i]
    back: list[int] = [-1] * (n + 1)  # back[i]: start index of the piece ending at i
    best[0] = 0.0
    for end in range(1, n + 1):
        for start in range(end):
            piece = text[start:end]
            if piece in vocab_logp and best[start] > -math.inf:
                score = best[start] + vocab_logp[piece]
                if score > best[end]:
                    best[end] = score
                    back[end] = start
    if best[n] == -math.inf:
        return [], -math.inf  # no full segmentation (a character was missing) -- coverage broke
    pieces: list[str] = []
    i = n
    while i > 0:
        start = back[i]
        pieces.append(text[start:i])
        i = start
    pieces.reverse()
    return pieces, best[n]


# =========================================================================================
# Downstream trade-off -- vocab size vs sequence length vs OOV (the central knob)
# =========================================================================================
def train_corpus_words() -> dict[str, int]:
    """A slightly larger, fixed word-frequency table for the vocab-size sweep.

    Chosen so that, at small vocab sizes, only the most common stems/suffixes earn a merge while
    rarer words stay fragmented -- exactly the regime where the compression/OOV curve is visible.
    Frequencies are illustrative (hand-set) but fixed, so the sweep is reproducible.
    """
    return {
        "low": 50, "lower": 12, "lowest": 8, "slow": 9, "slower": 4, "slowest": 3,
        "new": 40, "newer": 10, "newest": 22, "renew": 6, "renewed": 5,
        "wide": 18, "wider": 7, "widest": 14, "width": 6,
        "fast": 30, "faster": 11, "fastest": 9, "fasten": 4,
        "old": 25, "older": 9, "oldest": 7, "bold": 5, "boldest": 3,
        "the": 120, "of": 90, "and": 80, "to": 70, "in": 60,
    }


def tokens_per_word(corpus: dict[str, int], merges: list[tuple[str, str]]) -> float:
    """Average tokens per word produced by replaying `merges` on the corpus (the compression metric).

    Lower = better compression = shorter sequences. Weighted by word frequency so common words
    dominate, exactly as they dominate real sequence length.
    """
    total_tokens = 0
    total_words = 0
    for word, count in corpus.items():
        n_tokens = len(encode_bpe(word, merges))
        total_tokens += n_tokens * count
        total_words += count
    return total_tokens / total_words


def oov_char_fallback_rate(
    held_out: list[str], merges: list[tuple[str, str]]
) -> float:
    """Fraction of held-out words that fall ALL the way back to single characters (worst coverage).

    A subword tokenizer never emits a true OOV (`[UNK]`) -- it always reaches the character floor.
    But a word that ends up as ALL single characters got NO useful subword reuse, which is the
    practical analogue of "OOV": maximal sequence length, minimal shared structure. As the vocab
    grows, fewer held-out words hit this floor -- that is the curve we plot against compression.
    """
    if not held_out:
        return 0.0
    n_fallback = 0
    for word in held_out:
        pieces = encode_bpe(word, merges)
        # all pieces are single-char (ignoring the end marker) -> no multi-char subword reused
        core = [p for p in pieces if p != END]
        if all(len(p) == 1 for p in core):
            n_fallback += 1
    return n_fallback / len(held_out)


def vocab_size_sweep(
    corpus: dict[str, int], held_out: list[str], merge_counts: list[int]
) -> list[dict[str, float]]:
    """Sweep the number of merges (== vocab size) and record compression and OOV-floor rate.

    For each merge budget we train BPE, then measure (a) tokens/word on the training corpus
    (compression: falls as vocab grows) and (b) the char-fallback rate on held-out words
    (coverage quality: also falls as vocab grows). The two falling curves, traded against a
    growing vocabulary, ARE the vocab-size decision.
    """
    base_vocab = len(bpe_base_alphabet(corpus))
    rows: list[dict[str, float]] = []
    max_merges = max(merge_counts)
    all_merges, _, _ = train_bpe(corpus, max_merges)  # train once to the max, then slice
    for k in merge_counts:
        merges_k = all_merges[:k]
        rows.append(
            {
                "n_merges": float(k),
                "vocab_size": float(base_vocab + k),  # base alphabet + one token per merge
                "tokens_per_word": tokens_per_word(corpus, merges_k),
                "oov_fallback_rate": oov_char_fallback_rate(held_out, merges_k),
            }
        )
    return rows


# =========================================================================================
# Optional real-tokenizer helpers (only for reproducing the measured figures)
# =========================================================================================
SENTENCE = "The unhappiest developers refactored 1234567 lines of code."
MULTILINGUAL: list[tuple[str, str]] = [
    ("English", "The cat sat on the mat."),
    ("Spanish", "El gato se sentó en la alfombra."),
    ("Chinese", "猫坐在垫子上。"),
    ("Hindi", "बिल्ली चटाई पर बैठी थी।"),
]


def gpt4_tokens(sentence: str = SENTENCE) -> list[str]:
    """GPT-4 (cl100k_base, byte-level BPE) tokens as decoded strings. Requires tiktoken."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return [enc.decode([t]) for t in enc.encode(sentence)]


def bert_tokens(sentence: str = SENTENCE) -> list[str]:
    """BERT (bert-base-uncased, WordPiece) tokens. Requires transformers."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tok.tokenize(sentence)


def multilingual_gpt4_counts() -> list[tuple[str, int]]:
    """GPT-4 token count for the SAME sentence across languages -- the multilingual tax. tiktoken."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return [(lang, len(enc.encode(text))) for lang, text in MULTILINGUAL]


# =========================================================================================
# Self-check demo (asserts the qualitative point of each section BEFORE printing)
# =========================================================================================
def main() -> None:
    import sys

    print("device: cpu (pure-Python/numpy)")
    print("python:", sys.version.split()[0])
    try:
        import numpy

        print("numpy:", numpy.__version__)
    except ImportError:
        print("numpy: (not installed -- not required for the pure-Python algorithms)")
    print()

    # ---- BPE training: the ordered merge list -------------------------------------------
    merges, freqs, _ = train_bpe(TOY_CORPUS, N_MERGES)
    assert merges[0] == ("e", "s"), "BPE's first merge on the toy corpus must be (e, s)"
    assert len(merges) == N_MERGES
    print("BPE merge list (ordered) -- the saved artifact:")
    for i, (pair, freq) in enumerate(zip(merges, freqs), 1):
        print(f"  step {i:2d}: merge {pair!s:>20}  (freq {freq})")
    print()

    # ---- BPE encoding of an UNSEEN word -------------------------------------------------
    seg = encode_bpe("lowest", merges)
    assert seg == ["low", "est</w>"], f"expected ['low', 'est</w>'], got {seg}"
    print(f"encode unseen 'lowest' -> {seg}   (reused learned pieces; no [UNK])")
    # every unseen word still covered -- worst case, single characters (always in base vocab)
    for w in ("lowest", "newer", "xyz"):
        pieces = encode_bpe(w, merges)
        assert all(len(p) >= 1 for p in pieces)
        print(f"  encode {w!r:>10} -> {pieces}")
    print()

    # ---- BPE vs WordPiece first merge ---------------------------------------------------
    bpe_pick, wp_pick, details = first_merge_bpe_vs_wordpiece(TOY_CORPUS)
    assert bpe_pick == ("e", "s"), "BPE picks the highest raw-frequency pair"
    assert wp_pick == ("i", "d"), "WordPiece picks the highest normalised-score pair"
    assert bpe_pick != wp_pick, "the whole point: they disagree on identical data"
    print(f"BPE first merge      : {bpe_pick}  (raw freq {int(details[bpe_pick]['pair_freq'])})")
    print(
        f"WordPiece first merge: {wp_pick}  "
        f"(freq {int(details[wp_pick]['pair_freq'])}, score {details[wp_pick]['score']:.3f})"
    )
    print()

    # ---- Unigram Viterbi segmentation ---------------------------------------------------
    # A tiny hand-built Unigram model: 'est' is cheap (high prob), so the best split uses it.
    pieces_vocab = {"low": 0.30, "est": 0.30, "lo": 0.05, "we": 0.05,
                    "l": 0.05, "o": 0.05, "w": 0.05, "e": 0.05, "s": 0.05, "t": 0.05}
    logp = {p: math.log(v) for p, v in pieces_vocab.items()}
    seg_u, score_u = viterbi_segment("lowest", logp)
    assert seg_u == ["low", "est"], f"Viterbi should pick ['low', 'est'], got {seg_u}"
    print(f"Unigram Viterbi 'lowest' -> {seg_u}   (log-prob {score_u:.3f}, global optimum)")
    print()

    # ---- Vocab-size sweep: compression vs OOV-floor -------------------------------------
    corpus = train_corpus_words()
    held_out = ["lowness", "newness", "broadest", "renewing", "widening", "boldness"]
    merge_counts = [0, 2, 5, 10, 20, 40]
    rows = vocab_size_sweep(corpus, held_out, merge_counts)
    # larger vocab -> shorter sequences AND fewer char-fallbacks (both curves fall)
    assert rows[-1]["tokens_per_word"] <= rows[0]["tokens_per_word"], "more merges -> better compression"
    assert rows[-1]["oov_fallback_rate"] <= rows[0]["oov_fallback_rate"], "more merges -> less fallback"
    print("vocab-size sweep (more merges = bigger vocab):")
    print(f"  {'merges':>6} {'vocab':>6} {'tok/word':>9} {'char-fallback':>14}")
    for r in rows:
        print(
            f"  {int(r['n_merges']):>6} {int(r['vocab_size']):>6} "
            f"{r['tokens_per_word']:>9.2f} {r['oov_fallback_rate']:>13.0%}"
        )
    print()
    print("all self-checks passed.")


if __name__ == "__main__":
    main()
