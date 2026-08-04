"""Bag-of-Words, TF-IDF, cosine retrieval, and BM25 — from scratch, verified against scikit-learn.

This is the single source of truth for the chapter: the concept page, the teaching notebook, and
the figure generator (`make_figures_03.py`) all import the functions and constants defined here, so
none of them can silently drift from the others. Every number on the page is produced by this file.

The math is pure-Python / NumPy and fully deterministic — there is nothing stochastic to seed (BoW
counts, IDF, TF-IDF, cosine, and BM25 are exact functions of the corpus), so the same corpus always
yields bit-for-bit the same vectors on any machine. The optional cross-check against scikit-learn
reproduces the by-hand TF-IDF matrix to machine precision.

Run:
    python bow_tfidf.py
"""

from __future__ import annotations

import re

import numpy as np

# --- The toy corpus used everywhere on the page, in the notebook, and in the figures -----------
# Three short documents whose overlaps are easy to reason about by eye: D1 and D2 share the whole
# sentence frame "the ... sat on the ...", while D3 shares only "cat" and "the" with D1.
CORPUS: tuple[str, ...] = (
    "the cat sat on the mat",  # D1
    "the dog sat on the log",  # D2
    "the happy cat chased the dog",  # D3
)

# Two extra documents used ONLY for the BM25 retrieval demo, so the corpus has documents that match
# neither query term (to show they score exactly zero) and varying lengths (to exercise the b knob).
BM25_EXTRA_DOCS: tuple[str, ...] = (
    "a quick brown fox jumps over the lazy dog",  # D4
    "cats and dogs are common household pets",  # D5
)

# The query used for the retrieval demo and the ranking figure.
QUERY: str = "happy cat"

# scikit-learn's default word token: a run of word characters delimited by word boundaries.
TOKEN_PATTERN: str = r"(?u)\b\w+\b"

# BM25 Okapi defaults (Robertson & Zaragoza): k1 in [1.2, 2.0] sets TF saturation, b in [0,1] sets
# document-length normalization. 1.5 / 0.75 are the Lucene/Elasticsearch defaults.
BM25_K1: float = 1.5
BM25_B: float = 0.75

# Raw counts used by the TF-variant comparison table and figure: one term seen 1, 5, 50 times.
TF_DEMO_COUNTS: tuple[int, ...] = (1, 5, 50)

# Collection size used purely to draw the standalone IDF-vs-df curve (independent of CORPUS).
IDF_CURVE_N: int = 1000


def tokenize(text: str) -> list[str]:
    """Lowercase then split on scikit-learn's default word-token pattern.

    Matching scikit-learn's tokenizer exactly is what lets the by-hand numbers below line up with
    `CountVectorizer` / `TfidfVectorizer` to machine precision.
    """
    return re.findall(TOKEN_PATTERN, text.lower())


def build_vocabulary(corpus: tuple[str, ...]) -> list[str]:
    """Return the sorted list of distinct tokens across the corpus (the column order of the matrix)."""
    vocab: set[str] = set()
    for doc in corpus:
        vocab.update(tokenize(doc))
    return sorted(vocab)  # sorted so the column order is deterministic and matches scikit-learn


def count_matrix(corpus: tuple[str, ...], vocab: list[str]) -> np.ndarray:
    """Bag-of-Words document-term matrix: row d, column t = raw count of term t in document d."""
    index = {term: j for j, term in enumerate(vocab)}
    counts = np.zeros((len(corpus), len(vocab)), dtype=np.int64)
    for i, doc in enumerate(corpus):
        for tok in tokenize(doc):
            counts[i, index[tok]] += 1  # bag: order is discarded, only the tally survives
    return counts


def document_frequency(counts: np.ndarray) -> np.ndarray:
    """Per term, the number of documents that contain it at least once (df_t), capped at 1 per doc."""
    return (counts > 0).sum(axis=0)  # (counts > 0) collapses repeats: presence, not total count


def smoothed_idf(counts: np.ndarray) -> np.ndarray:
    """scikit-learn's smoothed IDF: log((1+N)/(1+df)) + 1.

    The +1 inside acts like one extra document containing every term (no division by zero); the
    trailing +1 keeps every IDF strictly positive so a term in all documents is damped, not deleted.
    """
    n_docs = counts.shape[0]
    df = document_frequency(counts)
    return np.log((1 + n_docs) / (1 + df)) + 1.0


def plain_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """Textbook (unsmoothed) IDF = log(N/df) — the self-information of 'this term appears'."""
    return np.log(n_docs / df)


def tfidf_matrix(corpus: tuple[str, ...], vocab: list[str]) -> np.ndarray:
    """L2-normalized TF-IDF matrix: raw-count TF times smoothed IDF, each row scaled to unit length.

    This reproduces scikit-learn's `TfidfVectorizer(norm="l2")` defaults (raw TF, smoothed IDF, L2).
    """
    counts = count_matrix(corpus, vocab)
    idf = smoothed_idf(counts)
    weighted = counts * idf  # broadcast IDF across rows: tfidf_{d,t} = count_{d,t} * idf_t
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard the all-zero row (a document with no in-vocab tokens)
    return weighted / norms


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows; for already-L2-normalized rows this is just X @ X.T."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms
    return unit @ unit.T  # row i · row j = cos(i, j) since each row is unit length


def tfidf_query_scores(corpus: tuple[str, ...], query: str) -> np.ndarray:
    """Cosine TF-IDF relevance of `query` against each document, with correct fit/transform discipline.

    The vocabulary AND the IDF weights are learned from the *corpus only*; the query is then projected
    into that fitted space (its terms reuse the corpus IDF, OOV query terms are dropped). This mirrors
    production retrieval and avoids the classic leak of fitting on the query/test set. Returns one
    cosine score per document, in corpus order.
    """
    vocab = build_vocabulary(corpus)  # vocabulary from the corpus only — the query cannot add columns
    index = {term: j for j, term in enumerate(vocab)}
    counts = count_matrix(corpus, vocab)
    idf = smoothed_idf(counts)  # IDF learned from the corpus only

    doc_weighted = counts * idf
    doc_norms = np.linalg.norm(doc_weighted, axis=1, keepdims=True)
    doc_norms[doc_norms == 0] = 1.0
    doc_unit = doc_weighted / doc_norms

    q_counts = np.zeros(len(vocab))
    for tok in tokenize(query):
        if tok in index:  # OOV query terms have no column and are silently dropped
            q_counts[index[tok]] += 1
    q_weighted = q_counts * idf  # reuse the corpus IDF — do NOT recompute from the query
    q_norm = np.linalg.norm(q_weighted)
    if q_norm == 0:
        return np.zeros(len(corpus))
    q_unit = q_weighted / q_norm
    return doc_unit @ q_unit  # unit·unit = cosine


def log_normalized_tf(count: float) -> float:
    """Sublinear TF: 1 + ln(count) for count > 0, else 0 — the 'diminishing returns' transform."""
    if count <= 0:
        return 0.0
    return 1.0 + np.log(count)


def bm25_scores(
    corpus: tuple[str, ...],
    query: str,
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> np.ndarray:
    """Okapi BM25 score of `query` against every document in `corpus`.

    score(Q, D) = sum_{t in Q} idf(t) * f(t,D) * (k1+1) / (f(t,D) + k1*(1 - b + b*|D|/avgdl))
    with the Robertson-Spärck-Jones IDF idf(t) = ln(1 + (N - df_t + 0.5)/(df_t + 0.5)).
    """
    tokenized = [tokenize(doc) for doc in corpus]
    n_docs = len(tokenized)
    avgdl = float(np.mean([len(doc) for doc in tokenized]))
    q_terms = tokenize(query)
    # df over the SAME corpus we are scoring (a term's rarity is a property of this collection).
    df_q = {term: sum(term in doc for doc in tokenized) for term in q_terms}
    scores = np.zeros(n_docs)
    for i, doc in enumerate(tokenized):
        for term in q_terms:
            freq = doc.count(term)
            if freq == 0:
                continue  # a query term absent from this doc contributes nothing
            idf = np.log(1 + (n_docs - df_q[term] + 0.5) / (df_q[term] + 0.5))
            denom = freq + k1 * (1 - b + b * len(doc) / avgdl)
            scores[i] += idf * freq * (k1 + 1) / denom
    return scores


def bm25_tf_component(freq: float, k1: float = BM25_K1) -> float:
    """BM25's length-off TF term f*(k1+1)/(f+k1): saturates toward the ceiling k1+1 as f grows."""
    return freq * (k1 + 1) / (freq + k1)


def main() -> None:
    """Print every headline number on the page and verify the by-hand TF-IDF against scikit-learn."""
    print("device: cpu (pure-Python/numpy)")
    print("numpy:", np.__version__)
    try:
        import torch

        print("torch:", torch.__version__)
    except ImportError:
        print("torch: not importable (not needed — this chapter is pure NumPy)")
    print()

    vocab = build_vocabulary(CORPUS)
    counts = count_matrix(CORPUS, vocab)
    print("vocabulary:", vocab)
    print("BoW counts (rows = docs):")
    print(counts)
    print()

    idf = smoothed_idf(counts)
    print("smoothed idf:", {term: round(float(v), 4) for term, v in zip(vocab, idf)})

    tfidf = tfidf_matrix(CORPUS, vocab)

    # Cross-check against scikit-learn if available; the math above is self-contained without it.
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(token_pattern=TOKEN_PATTERN, norm="l2")
        tfidf_sklearn = vectorizer.fit_transform(CORPUS).toarray()
        max_diff = float(np.abs(tfidf - tfidf_sklearn).max())
        assert np.allclose(tfidf, tfidf_sklearn, atol=1e-9), "by-hand TF-IDF diverged from sklearn"
        print(f"hand == sklearn: True | max abs diff: {max_diff:.2e}")
    except ImportError:
        print("hand == sklearn: (scikit-learn not installed — skipping cross-check)")
    print()

    cos = cosine_similarity_matrix(tfidf)
    print(f"cosine(D1,D2) = {cos[0, 1]:.3f}   cosine(D1,D3) = {cos[0, 2]:.3f}")
    print()

    full_corpus = CORPUS + BM25_EXTRA_DOCS
    scores = bm25_scores(full_corpus, QUERY)
    print(f"BM25 ranking for {QUERY!r}:")
    for i in np.argsort(-scores):
        print(f"  D{i + 1}: {scores[i]:.3f}  {full_corpus[i]!r}")


if __name__ == "__main__":
    main()
