"""N-gram language models and smoothing — from scratch, deterministic.

This is the single source of truth for the chapter: the concept page, the teaching notebook,
and the figure generator (`make_figures_04.py`) all import the corpora, the count machinery, and
every smoother defined here, so none of them can silently drift from the others. Every number on
the page is produced by this file.

The math is pure-Python / NumPy. The only stochastic step is text *generation* (sampling the next
word) and the train/test *shuffle*; both are seeded with `SEED`, so the same corpus always yields
bit-for-bit the same split, the same perplexities, and the same generated sentences on any machine.
Counts, MLE, Laplace/add-k, Good-Turing, and Kneser-Ney are exact functions of the counts and carry
no randomness at all.

Run:
    python ngram_lm.py
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

# --- Reproducibility -----------------------------------------------------------------------------
# One global seed drives the train/test shuffle and the generation sampling. Everything else is a
# deterministic function of the counts, so fixing this makes the whole chapter reproducible.
SEED = 0

# --- The toy corpus from SLP3 (Jurafsky & Martin, Ch. 3) -----------------------------------------
# Used for every by-hand worked example on the page: the bigram P(am|I)=2/3, the zero-probability of
# (Sam, do), the Laplace rescue, and the continuation probabilities. Padding is added per order.
TOY_CORPUS: tuple[str, ...] = (
    "I am Sam",
    "Sam I am",
    "I do not like green eggs and ham",
)

# --- A slightly larger toy corpus for the measured plots (train/score/generate) ------------------
# Small, closed-world animal sentences with heavy word reuse, so a trigram model has *some* repeated
# context to learn from while still being sparse enough to show smoothing differences clearly.
ANIMAL_TEXT = """
the cat sat on the mat . the cat saw the dog . the dog sat on the log .
a dog ran on the grass . the cat ran on the mat . the dog saw the cat .
the cat chased the dog . the dog chased the cat . a cat sat on a log .
the kitten saw the puppy . the puppy ran on the grass . a cat saw a dog .
"""

# Special tokens. <s> pads the left context (one per order below n); </s> terminates a sentence so
# the distribution normalizes across sequence lengths; <UNK> absorbs out-of-vocabulary words.
BOS = "<s>"
EOS = "</s>"
UNK = "<UNK>"

# Default absolute-discount for Kneser-Ney. Chen & Goodman (1999) find the optimal discount sits a
# little below 1; 0.75 is the textbook default and what SRILM/KenLM use as a starting point.
DEFAULT_DISCOUNT = 0.75


# =================================================================================================
# Tokenization and padding
# =================================================================================================
def sentences_from_text(text: str) -> list[list[str]]:
    """Split `.`-delimited text into lowercased word-token lists (one list per sentence)."""
    out: list[list[str]] = []
    for chunk in text.strip().split("."):
        words = re.findall(r"[a-z]+", chunk.lower())
        if words:
            out.append(words)
    return out


def tokenize_toy(corpus: Iterable[str]) -> list[list[str]]:
    """Tokenize the case-sensitive SLP3 toy corpus on whitespace (keeps 'I'/'Sam' capitalization)."""
    return [sentence.split() for sentence in corpus]


def pad(sentence: Sequence[str], n: int) -> list[str]:
    """Pad a sentence with (n-1) BOS markers on the left and one EOS on the right.

    The (n-1) left pads give the first real word a full (n-1)-word context; the single EOS lets the
    model place probability on *where a sentence ends*.
    """
    return [BOS] * (n - 1) + list(sentence) + [EOS]


# =================================================================================================
# Counting
# =================================================================================================
def count_ngrams(sentences: Iterable[Sequence[str]], n: int) -> Counter[tuple[str, ...]]:
    """Count all order-n n-grams across padded sentences. Returns Counter keyed by the n-gram tuple."""
    counts: Counter[tuple[str, ...]] = Counter()
    for sentence in sentences:
        padded = pad(sentence, n)
        for i in range(len(padded) - n + 1):
            counts[tuple(padded[i : i + n])] += 1
    return counts


def context_counts(ngram_counts: Counter[tuple[str, ...]]) -> Counter[tuple[str, ...]]:
    """Sum n-gram counts by their (n-1)-word prefix — the denominator c(context) for MLE."""
    ctx: Counter[tuple[str, ...]] = Counter()
    for gram, c in ngram_counts.items():
        ctx[gram[:-1]] += c
    return ctx


def laplace_vocab_size(sentences: Iterable[Sequence[str]]) -> int:
    """Vocabulary size for add-k smoothing: distinct tokens after bigram padding, i.e. content words
    plus the <s> and </s> boundary markers. This is the SLP3 convention (V=12 on the toy corpus),
    and the V that makes the +k*V denominator normalize the conditional over every possible next word.
    """
    vocab = {token for sentence in sentences for token in pad(sentence, 2)}
    return len(vocab)


def continuation_counts_bigram(
    sentences: Iterable[Sequence[str]],
) -> tuple[dict[str, int], int]:
    """Return (distinct-preceder count per word, number of distinct bigram types) at BIGRAM order.

    This backs the by-hand continuation-probability example: P_cont(w) = (# distinct preceders of w)
    / (# distinct bigram types). Bigram-order padding gives one <s>, hence 15 bigram types on the
    toy corpus, matching the worked example exactly.
    """
    preceders: defaultdict[str, set[str]] = defaultdict(set)
    bigram_types: set[tuple[str, str]] = set()
    for sentence in sentences:
        p2 = pad(sentence, 2)
        for a, b in zip(p2, p2[1:]):
            preceders[b].add(a)
            bigram_types.add((a, b))
    return {w: len(ps) for w, ps in preceders.items()}, len(bigram_types)


# =================================================================================================
# Maximum-likelihood estimate (raw counts) — exhibits the zero-probability catastrophe
# =================================================================================================
def mle_conditional(
    word: str, ctx: tuple[str, ...], ngram_counts: Counter[tuple[str, ...]],
    ctx_counts: Counter[tuple[str, ...]],
) -> float:
    """MLE P(word | ctx) = c(ctx, word) / c(ctx). Returns 0.0 for any unseen n-gram (the catastrophe)."""
    denom = ctx_counts.get(ctx, 0)
    if denom == 0:
        return 0.0
    return ngram_counts.get((*ctx, word), 0) / denom


# =================================================================================================
# Laplace (add-one) and add-k smoothing
# =================================================================================================
def add_k_conditional(
    word: str, ctx: tuple[str, ...], ngram_counts: Counter[tuple[str, ...]],
    ctx_counts: Counter[tuple[str, ...]], vocab_size: int, k: float = 1.0,
) -> float:
    """add-k P(word | ctx) = (c(ctx, word) + k) / (c(ctx) + k*V).  k=1 is Laplace/add-one.

    The k*V in the denominator is the bookkeeping that keeps the conditional summing to 1: we added k
    to each of V possible next words, so the denominator must grow by k*V to match.
    """
    num = ngram_counts.get((*ctx, word), 0) + k
    denom = ctx_counts.get(ctx, 0) + k * vocab_size
    return num / denom


# =================================================================================================
# Good-Turing — frequency of frequencies, reestimated counts, and the missing mass N1/N
# =================================================================================================
def frequency_of_frequencies(ngram_counts: Counter[tuple[str, ...]]) -> dict[int, int]:
    """N_r = number of distinct n-gram TYPES that occur exactly r times ('frequency of frequencies')."""
    nr: Counter[int] = Counter()
    for c in ngram_counts.values():
        nr[c] += 1
    return dict(nr)


def good_turing_reestimated_count(r: int, nr: dict[int, int]) -> float:
    """Good-Turing reestimated count c* = (r+1) * N_{r+1} / N_r.

    Falls back to the raw r when N_r or N_{r+1} is missing in the tail (where Good-Turing is unstable
    and practical implementations smooth the N_r first); this keeps the demo well-defined.
    """
    n_r = nr.get(r, 0)
    n_r1 = nr.get(r + 1, 0)
    if n_r == 0 or n_r1 == 0:
        return float(r)
    return (r + 1) * n_r1 / n_r


def good_turing_missing_mass(ngram_counts: Counter[tuple[str, ...]]) -> float:
    """P_GT(unseen) = N_1 / N: the fraction of singletons estimates the mass to hold back for the unseen."""
    nr = frequency_of_frequencies(ngram_counts)
    n1 = nr.get(1, 0)
    total = sum(ngram_counts.values())
    return n1 / total if total else 0.0


# =================================================================================================
# Interpolated Kneser-Ney trigram model — the gold-standard classical smoother
# =================================================================================================
class KneserNeyTrigram:
    """Interpolated Kneser-Ney trigram LM: absolute discounting + continuation-probability backoff.

    The continuation probability is the crux: when the model backs off, it scores a word by *how many
    distinct contexts it completes* (its versatility), not by raw frequency — so 'Francisco', frequent
    but appearing almost only after 'San', is correctly judged a poor general-purpose continuation.
    """

    def __init__(self, train: Iterable[Sequence[str]], d: float = DEFAULT_DISCOUNT) -> None:
        self.d = d
        self.uni: Counter[str] = Counter()
        self.bi: Counter[tuple[str, str]] = Counter()
        self.tri: Counter[tuple[str, str, str]] = Counter()
        self.ctx_bi: Counter[tuple[str, ...]] = Counter()      # c(a) for bigram contexts
        self.ctx_tri: Counter[tuple[str, str]] = Counter()     # c(a, b) for trigram contexts
        self.preceders: defaultdict[str, set[str]] = defaultdict(set)   # distinct words preceding w
        self.followers1: defaultdict[str, set[str]] = defaultdict(set)  # distinct words following a
        self.followers2: defaultdict[tuple[str, str], set[str]] = defaultdict(set)  # following (a,b)
        for sentence in train:
            # Bigram-level statistics use BIGRAM padding (one <s>): this is what makes the
            # continuation example come out to /15 on the toy corpus and keeps this trigram model's
            # lower-order numbers identical to KneserNeyOrderN's. Trigram counts use trigram padding.
            p2 = pad(sentence, 2)
            p3 = pad(sentence, 3)
            for w in p2:
                self.uni[w] += 1
            for a, b in zip(p2, p2[1:]):
                self.bi[(a, b)] += 1
                self.ctx_bi[(a,)] += 1
                self.preceders[b].add(a)
                self.followers1[a].add(b)
            for a, b, c in zip(p3, p3[1:], p3[2:]):
                self.tri[(a, b, c)] += 1
                self.ctx_tri[(a, b)] += 1
                self.followers2[(a, b)].add(c)
        self.vocab_size = len(self.uni)
        self.n_bigram_types = len(self.bi)

    def continuation_prob(self, w: str) -> float:
        """P_cont(w) = (# distinct preceders of w) / (# distinct bigram types), with a +1 OOV floor.

        The +1 / (types + V) floor keeps a never-before-seen word from collapsing to exactly zero.
        """
        return (len(self.preceders.get(w, ())) + 1) / (self.n_bigram_types + self.vocab_size)

    def p_bigram(self, w: str, a: str) -> float:
        """Interpolated KN bigram: max(c(a,w)-d, 0)/c(a) + lambda(a) * P_cont(w)."""
        c = self.bi.get((a, w), 0)
        cc = self.ctx_bi.get((a,), 0)
        if cc == 0:
            return self.continuation_prob(w)
        lam = self.d * len(self.followers1[a]) / cc
        return max(c - self.d, 0) / cc + lam * self.continuation_prob(w)

    def p_trigram(self, w: str, a: str, b: str) -> float:
        """Interpolated KN trigram: discount the trigram, interpolate with the KN bigram below it."""
        c = self.tri.get((a, b, w), 0)
        cc = self.ctx_tri.get((a, b), 0)
        if cc == 0:
            return self.p_bigram(w, b)  # unseen context → back off entirely to the KN bigram
        lam = self.d * len(self.followers2[(a, b)]) / cc
        return max(c - self.d, 0) / cc + lam * self.p_bigram(w, b)

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        """Perplexity = 2^(cross-entropy) of held-out trigrams under the model (lower is better)."""
        log2_sum, n_tokens = 0.0, 0
        for sentence in sentences:
            p3 = pad(sentence, 3)
            for a, b, c in zip(p3, p3[1:], p3[2:]):
                log2_sum += math.log2(max(self.p_trigram(c, a, b), 1e-12))
                n_tokens += 1
        if n_tokens == 0:
            return float("inf")
        return 2 ** (-log2_sum / n_tokens)

    def generate(self, max_len: int = 12, rng: random.Random | None = None) -> str:
        """Sample a sentence by drawing each next word from the model's own trigram distribution."""
        rng = rng or random.Random(SEED)
        a, b, out = BOS, BOS, []
        vocab = [w for w in self.uni if w != BOS]
        for _ in range(max_len):
            weighted = [(w, self.p_trigram(w, a, b)) for w in vocab]
            threshold = rng.random() * sum(p for _, p in weighted)
            cumulative = 0.0
            chosen = vocab[-1]
            for w, p in weighted:
                cumulative += p
                if cumulative >= threshold:
                    chosen = w
                    break
            if chosen == EOS:
                break
            out.append(chosen)
            a, b = b, chosen
        return " ".join(out)


# =================================================================================================
# A pure add-k bigram model, for the smoother-vs-data-size comparison figure
# =================================================================================================
class AddKBigram:
    """A plain add-k (default Laplace) bigram LM, used as the baseline KN is compared against."""

    def __init__(self, train: Iterable[Sequence[str]], k: float = 1.0) -> None:
        self.k = k
        self.bi: Counter[tuple[str, str]] = Counter()
        self.ctx: Counter[tuple[str, ...]] = Counter()
        self.uni: Counter[str] = Counter()
        for sentence in train:
            p2 = pad(sentence, 2)
            for w in p2:
                self.uni[w] += 1
            for a, b in zip(p2, p2[1:]):
                self.bi[(a, b)] += 1
                self.ctx[(a,)] += 1
        self.vocab_size = len(self.uni)

    def p_bigram(self, w: str, a: str) -> float:
        """add-k P(w | a) = (c(a,w)+k) / (c(a)+k*V)."""
        return add_k_conditional(w, (a,), self.bi, self.ctx, self.vocab_size, k=self.k)

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        """Perplexity of held-out bigrams under the add-k model."""
        log2_sum, n_tokens = 0.0, 0
        for sentence in sentences:
            p2 = pad(sentence, 2)
            for a, b in zip(p2, p2[1:]):
                log2_sum += math.log2(max(self.p_bigram(b, a), 1e-12))
                n_tokens += 1
        if n_tokens == 0:
            return float("inf")
        return 2 ** (-log2_sum / n_tokens)


# =================================================================================================
# An interpolated KN model of arbitrary order, for the perplexity-vs-n figure
# =================================================================================================
class KneserNeyOrderN:
    """Interpolated Kneser-Ney of arbitrary order n (recurses down to a unigram continuation floor).

    Generalizes KneserNeyTrigram to any order so the perplexity-vs-n curve can be measured with one
    consistent smoother. Continuation counts replace raw counts at every lower order, exactly as in
    the modified-KN recursion.
    """

    def __init__(self, train: Sequence[Sequence[str]], n: int, d: float = DEFAULT_DISCOUNT) -> None:
        self.n = n
        self.d = d
        self.train = [list(s) for s in train]
        # Per-order n-gram and context counts.
        self.grams: list[Counter[tuple[str, ...]]] = []
        self.ctxs: list[Counter[tuple[str, ...]]] = []
        for order in range(1, n + 1):
            g = count_ngrams(self.train, order)
            self.grams.append(g)
            self.ctxs.append(context_counts(g) if order > 1 else Counter())
        # Distinct followers per context (for the lambda mass) and distinct preceders (for P_cont).
        self.followers: list[defaultdict[tuple[str, ...], set[str]]] = []
        for order in range(1, n + 1):
            fol: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
            for gram in self.grams[order - 1]:
                fol[gram[:-1]].add(gram[-1])
            self.followers.append(fol)
        bigrams = count_ngrams(self.train, 2)
        self.preceders: defaultdict[str, set[str]] = defaultdict(set)
        for (a, b) in bigrams:
            self.preceders[b].add(a)
        self.n_bigram_types = len(bigrams)
        self.vocab_size = len(self.grams[0])

    def _p_unigram(self, w: str) -> float:
        """Unigram floor as a Kneser-Ney continuation probability (versatility, not frequency)."""
        return (len(self.preceders.get(w, ())) + 1) / (
            self.n_bigram_types + self.vocab_size
        )

    def prob(self, w: str, ctx: tuple[str, ...]) -> float:
        """Interpolated KN P(w | ctx) at order len(ctx)+1, recursing to the unigram continuation floor."""
        order = len(ctx) + 1
        if order == 1:
            return self._p_unigram(w)
        grams = self.grams[order - 1]
        ctxs = self.ctxs[order - 1]
        cc = ctxs.get(ctx, 0)
        if cc == 0:
            return self.prob(w, ctx[1:])  # unseen context → drop the oldest word and recurse down
        c = grams.get((*ctx, w), 0)
        lam = self.d * len(self.followers[order - 1][ctx]) / cc
        return max(c - self.d, 0) / cc + lam * self.prob(w, ctx[1:])

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        """Perplexity of held-out order-n n-grams under this KN model."""
        log2_sum, n_tokens = 0.0, 0
        for sentence in sentences:
            padded = pad(sentence, self.n)
            for i in range(self.n - 1, len(padded)):
                ctx = tuple(padded[i - self.n + 1 : i])
                w = padded[i]
                log2_sum += math.log2(max(self.prob(w, ctx), 1e-12))
                n_tokens += 1
        if n_tokens == 0:
            return float("inf")
        return 2 ** (-log2_sum / n_tokens)


# =================================================================================================
# Data prep used by the notebook and figures (seeded split of the animal corpus)
# =================================================================================================
def animal_train_test(test_fraction: float = 0.2, seed: int = SEED) -> tuple[list[list[str]], list[list[str]]]:
    """Tokenize the animal corpus and split into (train, test) with a seeded shuffle."""
    sentences = sentences_from_text(ANIMAL_TEXT)
    rng = random.Random(seed)
    rng.shuffle(sentences)
    split = int(len(sentences) * (1 - test_fraction))
    return sentences[:split], sentences[split:]


# =================================================================================================
# Self-check / demo
# =================================================================================================
def main() -> None:
    import numpy as np

    print("device: cpu (pure-Python/numpy)")
    print("numpy:", np.__version__)
    try:
        import torch

        print("torch:", torch.__version__, "(not used; n-grams are count-based)")
    except ImportError:
        print("torch: not installed (not needed)")
    print()

    # --- Worked examples on the SLP3 toy corpus -------------------------------------------------
    toy = tokenize_toy(TOY_CORPUS)
    bi = count_ngrams(toy, 2)
    ctx_bi = context_counts(bi)
    vocab_size = laplace_vocab_size(toy)  # SLP3 V=12 (10 content words + <s> + </s>)

    p_am_given_i = mle_conditional("am", ("I",), bi, ctx_bi)
    print(f"P(am | I)           MLE      = {p_am_given_i:.3f}   (expect 0.667)")
    # Full-sentence bigram probability of "<s> I am Sam </s>".
    sent_prob = (
        mle_conditional("I", (BOS,), bi, ctx_bi)
        * mle_conditional("am", ("I",), bi, ctx_bi)
        * mle_conditional("Sam", ("am",), bi, ctx_bi)
        * mle_conditional(EOS, ("Sam",), bi, ctx_bi)
    )
    print(f"P(<s> I am Sam </s>) MLE      = {sent_prob:.3f}   (expect 0.111)")

    # The zero-probability catastrophe: (Sam, do) was never seen.
    p_do_given_sam = mle_conditional("do", ("Sam",), bi, ctx_bi)
    print(f"P(do | Sam)         MLE      = {p_do_given_sam:.3f}   (the catastrophe: exactly 0)")

    # Laplace rescues it.
    p_do_laplace = add_k_conditional("do", ("Sam",), bi, ctx_bi, vocab_size, k=1.0)
    print(f"P(do | Sam)         Laplace  = {p_do_laplace:.4f}  (expect 0.0714 = 1/14)")
    p_am_laplace = add_k_conditional("am", ("I",), bi, ctx_bi, vocab_size, k=1.0)
    print(f"P(am | I)           Laplace  = {p_am_laplace:.3f}   (discounted from 0.667 → 0.2)")

    # Worked example 6 — Laplace perplexity of the held-out toy sentence "<s> I am Sam </s>"
    # (V=12, 4 bigrams): PP = 2^(-1/N * sum_i log2 P(w_i | w_{i-1})).
    toy_sentence_bigrams = [(BOS, "I"), ("I", "am"), ("am", "Sam"), ("Sam", EOS)]
    toy_log2 = sum(
        math.log2(add_k_conditional(w, (prev,), bi, ctx_bi, vocab_size, k=1.0))
        for prev, w in toy_sentence_bigrams
    )
    toy_laplace_pp = 2.0 ** (-toy_log2 / len(toy_sentence_bigrams))
    print(f"Laplace PP (toy sentence) = {toy_laplace_pp:.3f}  (expect 5.916 over 4 bigrams)")
    assert abs(toy_laplace_pp - 5.916) < 1e-2, "toy-sentence Laplace perplexity must be ~5.916"

    # Good-Turing missing mass — measured on the (larger) animal corpus, where N_r is meaningful.
    animal_all = sentences_from_text(ANIMAL_TEXT)
    animal_bi = count_ngrams(animal_all, 2)
    nr = frequency_of_frequencies(animal_bi)
    print(
        f"Good-Turing (animal bigrams): N1={nr.get(1)} N2={nr.get(2)} N3={nr.get(3)}  "
        f"N1/N={good_turing_missing_mass(animal_bi):.3f}  c*(1)={good_turing_reestimated_count(1, nr):.2f}"
    )

    # Kneser-Ney continuation probabilities (versatility), at bigram order → matches the by-hand 1/15.
    pre, n_bi_types = continuation_counts_bigram(toy)
    print(f"continuation (toy, {n_bi_types} bigram types):", end="  ")
    for word in ("am", "Sam", EOS):
        print(f"P_cont({word})={pre.get(word, 0)}/{n_bi_types}={pre.get(word, 0) / n_bi_types:.3f}", end="  ")
    print()

    print()
    # --- Measured model on the animal corpus ----------------------------------------------------
    train, test = animal_train_test()
    kn_tri = KneserNeyTrigram(train)
    print(f"animal corpus: {len(train)} train / {len(test)} test sentences, vocab={kn_tri.vocab_size}")
    print(f"KN trigram held-out perplexity = {kn_tri.perplexity(test):.2f}")
    rng = random.Random(SEED)
    print("generated:", kn_tri.generate(rng=rng))
    print("generated:", kn_tri.generate(rng=rng))

    # Perplexity vs n (KN), and Laplace vs KN bigram.
    print()
    for order in (1, 2, 3, 4):
        model = KneserNeyOrderN(train, n=order)
        print(f"  KN perplexity (n={order}) = {model.perplexity(test):.2f}")
    laplace = AddKBigram(train, k=1.0)
    kn_bi = KneserNeyOrderN(train, n=2)
    print(f"  Laplace bigram perplexity = {laplace.perplexity(test):.2f}")
    print(f"  KN bigram perplexity      = {kn_bi.perplexity(test):.2f}")


if __name__ == "__main__":
    main()
