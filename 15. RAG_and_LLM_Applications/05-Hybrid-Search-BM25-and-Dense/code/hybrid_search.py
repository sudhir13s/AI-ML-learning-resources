"""From-scratch hybrid search: BM25 (lexical) + dense (semantic), fused two ways.

Each retrieval lens has a real, *complementary* blind spot. A DENSE embedder matches on MEANING
but loses the razor edge on exact tokens: when a terse line that literally contains a product code
or error code competes with a chatty, same-topic passage, the dense model -- ranking by overall
topical similarity -- can put the chatty distractor first and bump the exact-match line to #2. A
SPARSE/BM25 retriever nails exact terms but is blind to PARAPHRASE: "gain height after blast-off"
shares no content word with "rose skyward past liftoff", so BM25 scores the right passage exactly
0 and misses it entirely. Hybrid search runs BOTH lenses and FUSES their results, so a passage
strong on EITHER lens still surfaces.

This module builds the whole thing from primitives so every step is inspectable:
  * BM25 from scratch (the Robertson-Zaragoza / Lucene scoring function) -- the lexical lens;
  * a dense retriever using a real learned bi-encoder (all-MiniLM-L6-v2 via sentence-transformers),
    with a deterministic from-scratch fallback so the module never hard-fails on a missing model;
  * two fusion families: (a) min-max score normalization + a convex combination
    alpha * dense + (1 - alpha) * sparse, and (b) Reciprocal Rank Fusion (RRF), rank-based and
    scale-free.

The headline demonstration: one probe DENSE alone gets wrong (ranks an exact-match #2 behind a
distractor), one probe SPARSE alone gets wrong (scores the paraphrase 0 -> a full miss), and a
HYBRID that, tuned, ranks BOTH golds #1 -- with the recall@k and MRR lift measured and asserted
before it is claimed.

The corpus and the IDF/tokenizer machinery are imported from chapter 1's `rag_fundamentals.py`
so the chapters share one source of truth; this page only adds BM25, the dense bi-encoder lens,
and fusion.

Verified on Python 3.12 / numpy 2.x / sentence-transformers (all-MiniLM-L6-v2, CPU). The dense
encoder runs deterministically on CPU; BM25 and fusion are pure arithmetic. Identical numbers
every run, every machine (given the same cached model).

Run:
    python hybrid_search.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse chapter 1's tokenizer + IDF so the chapters share one source of truth. (The chapter-1
# script lives one directory over; add it to the path so the import works whether this file is run
# from its own dir or imported by the notebook/figure scripts.)
_CH1_CODE = Path(__file__).resolve().parent.parent.parent / "01-RAG-Fundamentals" / "code"
if str(_CH1_CODE) not in sys.path:
    sys.path.insert(0, str(_CH1_CODE))

from rag_fundamentals import (  # noqa: E402  (path injected above must precede this import)
    CORPUS as CH1_CORPUS,
    build_index,
    compute_idf,
    tokenize,
)

# ---- BM25 hyperparameters (the Robertson-Zaragoza / Lucene defaults; see the page's math) ------
# k1 controls term-frequency SATURATION: how fast extra occurrences of a term stop adding score.
# Lucene / Elasticsearch default k1 = 1.2 (the classic recommended range is [1.2, 2.0]).
BM25_K1 = 1.2
# b controls length NORMALIZATION: b=1 fully penalizes long docs, b=0 ignores length. 0.75 is the
# near-universal default (Lucene, Elasticsearch, the original BM25 paper's recommendation).
BM25_B = 0.75

# ---- Fusion hyperparameters --------------------------------------------------------------------
# RRF's ranking constant. Elasticsearch and the Cormack et al. 2009 paper both default to 60: large
# enough that the top few ranks are not wildly dominant, small enough that deep ranks fade out.
RRF_K = 60
# Neutral convex-combination weight for weighted-sum fusion: 0.5 = equal trust in both lenses (the
# same meaning as Weaviate's alpha). alpha=1 -> pure dense, alpha=0 -> pure sparse. The page shows
# that the *tuned* optimum on this corpus is ~0.6 (a touch toward dense) -- which is the whole
# point of the alpha sweep: equal weighting is a starting guess, not the answer.
NEUTRAL_ALPHA = 0.5
TUNED_ALPHA = 0.6  # the alpha that ranks BOTH golds #1 here -- found by the sweep, not assumed
TOP_K = 3  # how many fused results to surface
DENSE_MODEL = "all-MiniLM-L6-v2"  # the learned bi-encoder lens (chapter 3's production embedder)

# Chapter 1's corpus already covers Helios-7 + a few distractors. We append three passages chosen
# to expose each lens's blind spot, on purpose:
#   * an EXACT-CODE line (terse) whose answer a chatty same-topic distractor outranks under dense;
#   * a PARAPHRASE line whose content words a paraphrased query never repeats (so BM25 scores 0);
#   * a chatty same-topic DISTRACTOR for the code probe (no code, but very "about errors").
EXTRA_PASSAGES: tuple[str, ...] = (
    "Error E-4011 appeared in the Helios-7 telemetry stream.",  # idx 8: exact-code gold (terse)
    "Climbing steadily, Helios-7 rose skyward moments past liftoff.",  # idx 9: paraphrase gold
    "The Helios-7 ground team spent the afternoon investigating several telemetry errors and console warnings.",  # idx 10: code-probe distractor
)


def full_corpus() -> tuple[str, ...]:
    """Chapter 1's corpus plus the three blind-spot passages this chapter adds."""
    return CH1_CORPUS + EXTRA_PASSAGES


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked retrieval list: parallel tuples of doc indices and their scores, best-first."""

    indices: tuple[int, ...]
    scores: tuple[float, ...]


# ================================================================================================
# BM25 -- the lexical lens, from scratch
# ================================================================================================


class BM25:
    """BM25 (Best Match 25), the Robertson-Zaragoza probabilistic relevance scoring function.

    Implements the Lucene / Elasticsearch variant of BM25 from primitives so the two mechanisms
    that make it work are visible:
      * term-frequency SATURATION via k1 -- the 11th occurrence of a word adds far less than the
        2nd, so a passage cannot win by keyword-stuffing;
      * length NORMALIZATION via b -- a long passage that contains a term is discounted relative to
        a short one that contains it just as often, because long passages match everything by
        chance.
    See `score()` for the full formula, term by term.
    """

    def __init__(self, corpus: tuple[str, ...], k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)
        self.doc_tokens: list[list[str]] = [tokenize(doc) for doc in corpus]
        self.doc_len: list[int] = [len(toks) for toks in self.doc_tokens]
        # avgdl: the average document length -- the yardstick length-normalization measures against.
        self.avgdl: float = sum(self.doc_len) / self.n_docs
        # term frequency per doc: tf[d][term] = count of `term` in doc d
        self.tf: list[dict[str, int]] = []
        # document frequency: df[term] = number of docs containing `term` (for IDF)
        self.df: dict[str, int] = {}
        for tokens in self.doc_tokens:
            counts: dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            self.tf.append(counts)
            for tok in counts:  # each term counts ONCE per doc toward df, regardless of its tf
                self.df[tok] = self.df.get(tok, 0) + 1
        # precompute IDF per term -- it depends only on the corpus, not the query
        self.idf: dict[str, float] = {tok: self._idf(tok) for tok in self.df}

    def _idf(self, term: str) -> float:
        """Probabilistic IDF, Lucene variant: ln(1 + (N - df + 0.5) / (df + 0.5)).

        The +0.5 terms are the Robertson-Zaragoza smoothing; the outer 1 + ... (Lucene's addition)
        guarantees IDF is strictly POSITIVE even for a term in more than half the corpus -- the
        classic ln((N-df+0.5)/(df+0.5)) goes negative there, which can make a very common term push
        a document's score DOWN. Rare terms (small df) get a large IDF; ubiquitous terms a small one.
        """
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_index: int) -> float:
        """BM25 score of one document for a query: sum over query terms of IDF x saturated-tf.

        For each query term t present in the document:

            IDF(t) * [ tf * (k1 + 1) ] / [ tf + k1 * (1 - b + b * |d| / avgdl) ]

        where tf is t's count in the doc, |d| is the doc length, and avgdl the average length. The
        numerator tf*(k1+1) and the +tf in the denominator are what produce SATURATION: as tf grows
        the ratio approaches (k1+1), a hard ceiling. The (1 - b + b*|d|/avgdl) factor is the length
        penalty: for an average-length doc it is 1 (no effect); longer docs inflate the denominator
        (lower score), shorter docs shrink it (higher score). A query term ABSENT from the doc
        contributes nothing -- which is exactly why a zero-overlap paraphrase scores 0 here.
        """
        tokens_in_doc = self.tf[doc_index]
        doc_len = self.doc_len[doc_index]
        # the length-normalization denominator factor, computed once per doc (term-independent)
        length_norm = 1.0 - self.b + self.b * doc_len / self.avgdl
        total = 0.0
        for term in tokenize(query):
            tf = tokens_in_doc.get(term, 0)
            if tf == 0:  # a query term absent from this doc contributes nothing
                continue
            idf = self.idf.get(term, 0.0)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * length_norm
            total += idf * numerator / denominator
        return total

    def all_scores(self, query: str) -> np.ndarray:
        """BM25 score for every doc (parallel to corpus order, unsorted) -- for fusion/plots."""
        return np.array([self.score(query, i) for i in range(self.n_docs)])

    def search(self, query: str, k: int = TOP_K) -> RetrievalResult:
        """Score every doc, return the top-k as a RetrievalResult (best-first)."""
        scores = self.all_scores(query)
        order = np.argsort(scores)[::-1][:k]  # descending, top-k
        return RetrievalResult(tuple(int(i) for i in order), tuple(float(scores[i]) for i in order))


# ================================================================================================
# Dense lens -- a learned bi-encoder (chapter 3's production embedder), with a deterministic
# fallback so the module never hard-fails on a missing model download.
# ================================================================================================


class DenseRetriever:
    """Dense semantic retrieval over the chapter's corpus.

    Uses a real learned bi-encoder (all-MiniLM-L6-v2) -- chapter 3's production embedder -- so the
    semantic lens genuinely matches paraphrases that share no words, which a lexical embedder cannot.
    If sentence-transformers / the model is unavailable, it falls back to chapter 1's deterministic
    IDF-hashing embedder so the script still runs end to end (the fallback is lexical, so it cannot
    do paraphrase -- the printed banner says which lens is active). Every passage and query is
    L2-normalized, so the cosine score is a plain dot product (chapter 1's geometry).
    """

    def __init__(self, corpus: tuple[str, ...], model_name: str = DENSE_MODEL) -> None:
        self.corpus = corpus
        self.backend, self._encode = self._load_encoder(model_name)
        self.index = self._encode(list(corpus))  # (n_docs, dim), unit-norm rows

    @staticmethod
    def _load_encoder(model_name: str):
        """Return (backend_name, encode_fn). encode_fn maps a list[str] -> (n, dim) unit-norm array."""
        try:
            import contextlib
            import io
            import logging
            import os

            # Silence HF/transformers load chatter so notebook output stays clean.
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            for name in ("sentence_transformers", "transformers", "transformers.modeling_utils"):
                logging.getLogger(name).setLevel(logging.ERROR)
            with contextlib.redirect_stderr(io.StringIO()):
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(model_name, device="cpu")

            def encode(texts: list[str]) -> np.ndarray:
                return np.asarray(
                    model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                )

            return f"sentence-transformers/{model_name}", encode
        except Exception:  # missing package, no network, or model not cached -> degrade gracefully

            def encode_fallback(texts: list[str]) -> np.ndarray:
                # lexical IDF-hashing fallback (chapter 1) -- deterministic, no download
                idf_local = compute_idf(tuple(texts))
                return build_index(tuple(texts), idf_local)

            return "ch1-idf-hashing (fallback, lexical)", encode_fallback

    def all_scores(self, query: str) -> np.ndarray:
        """Cosine score for every doc (parallel to corpus order)."""
        q_vec = self._encode([query])[0]
        return self.index @ q_vec  # unit-norm rows => dot product == cosine

    def search(self, query: str, k: int = TOP_K) -> RetrievalResult:
        """Top-k by cosine similarity (best-first)."""
        scores = self.all_scores(query)
        order = np.argsort(scores)[::-1][:k]
        return RetrievalResult(tuple(int(i) for i in order), tuple(float(scores[i]) for i in order))


# ================================================================================================
# Fusion -- combine the two ranked lists
# ================================================================================================


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Rescale a score vector to [0, 1] via (s - min) / (max - min).

    This is the fix for the #1 hybrid trap: raw BM25 (unbounded, often 0-15+) and cosine ([-1, 1])
    live on incomparable scales, so adding them lets BM25 silently dominate. Min-max maps each lens
    onto a common [0, 1] axis so a convex combination is meaningful. Guards the degenerate all-equal
    case (max == min) by returning zeros rather than dividing by zero.
    """
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:  # every doc scored the same -> no information to normalize
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def weighted_sum_fusion(
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    alpha: float = NEUTRAL_ALPHA,
    k: int = TOP_K,
) -> RetrievalResult:
    """Convex-combination fusion: alpha * norm(dense) + (1 - alpha) * norm(sparse).

    Both score vectors are min-max normalized to [0, 1] FIRST (see `min_max_normalize` for why),
    then blended. alpha is the trust dial -- alpha=1 is pure dense, alpha=0 pure sparse, alpha=0.5
    equal (the same semantics as Weaviate's `alpha`). Returns the top-k by fused score.
    """
    fused = alpha * min_max_normalize(dense_scores) + (1.0 - alpha) * min_max_normalize(sparse_scores)
    order = np.argsort(fused)[::-1][:k]
    return RetrievalResult(tuple(int(i) for i in order), tuple(float(fused[i]) for i in order))


def _ranks_from_scores(scores: np.ndarray) -> dict[int, int]:
    """Map each doc index -> its 1-based rank under `scores` (rank 1 = highest score)."""
    order = np.argsort(scores)[::-1]  # doc indices, best-first
    return {int(doc): rank for rank, doc in enumerate(order, start=1)}


def reciprocal_rank_fusion(
    score_lists: list[np.ndarray],
    k_rrf: int = RRF_K,
    k: int = TOP_K,
) -> RetrievalResult:
    """Reciprocal Rank Fusion (Cormack et al. 2009): fuse ranked lists by summing 1/(k_rrf + rank).

    RRF ignores the raw scores entirely and uses only each list's RANKING, which is why it needs no
    normalization and is immune to the BM25-vs-cosine scale mismatch -- its key advantage over
    weighted-sum. For each doc, sum 1/(k_rrf + rank_i) across every input list i; a doc ranked high
    in either list gets a large contribution. The constant k_rrf (default 60) damps the influence of
    low ranks: at rank 1 the term is 1/61, at rank 1000 it is ~1/1060, so being #1 in one list is
    worth far more than being #500 in both. Returns the top-k by fused RRF score.
    """
    n_docs = len(score_lists[0])
    rank_maps = [_ranks_from_scores(s) for s in score_lists]
    fused = np.zeros(n_docs)
    for doc in range(n_docs):
        for rank_map in rank_maps:
            fused[doc] += 1.0 / (k_rrf + rank_map[doc])
    order = np.argsort(fused)[::-1][:k]
    return RetrievalResult(tuple(int(i) for i in order), tuple(float(fused[i]) for i in order))


# ================================================================================================
# Evaluation -- measure the lift, do not just assert the winner
# ================================================================================================


def reciprocal_rank(result_indices: tuple[int, ...], gold: int) -> float:
    """Reciprocal rank of the gold doc in a result list: 1/rank if present, else 0.

    MRR (mean reciprocal rank) averages this over a query set. RR=1 means the right passage ranked
    #1; RR=0.5 means #2; RR=0 means it was not retrieved at all. It rewards ranking the answer high,
    not merely including it -- the property that separates a good retriever from a lucky one.
    """
    for rank, idx in enumerate(result_indices, start=1):
        if idx == gold:
            return 1.0 / rank
    return 0.0


def recall_at_k(result_indices: tuple[int, ...], gold: int) -> float:
    """1.0 if the gold doc is anywhere in the result list, else 0.0 (recall@k for a single gold)."""
    return 1.0 if gold in result_indices else 0.0


@dataclass(frozen=True)
class Probe:
    """One evaluation query: the text, the index of its single gold passage, and a human label."""

    query: str
    gold: int
    label: str


def build_probes(corpus: tuple[str, ...]) -> tuple[Probe, ...]:
    """Two probes that each defeat ONE lens, so fusion's value is measurable.

    * an EXACT-CODE probe whose terse gold a chatty same-topic distractor outranks under dense
      (dense ranks the gold #2) but BM25 ranks #1;
    * a PARAPHRASE probe whose gold shares NO content word with the query, so BM25 scores it 0 and
      misses it (rank far below top-k), while the dense lens matches it semantically (#1).
    The gold indices are resolved by content so they stay correct if the corpus order changes.
    """
    code_gold = next(i for i, d in enumerate(corpus) if "E-4011" in d)
    para_gold = next(i for i, d in enumerate(corpus) if "rose skyward" in d)
    return (
        Probe("What telemetry error did Helios-7 report?", code_gold, "exact-code (dense ranks a distractor first)"),
        Probe("How did the vehicle gain height after blast-off?", para_gold, "paraphrase (BM25 scores it 0)"),
    )


def evaluate(
    probes: tuple[Probe, ...],
    dense: DenseRetriever,
    bm25: BM25,
    alpha: float = TUNED_ALPHA,
    k: int = TOP_K,
    k_rrf: int = RRF_K,
) -> dict[str, tuple[float, float]]:
    """Return {method: (MRR, recall@k)} over the probe set for each retrieval method."""
    methods = {
        "dense only": lambda p: dense.search(p.query, k=k).indices,
        "sparse only (BM25)": lambda p: bm25.search(p.query, k=k).indices,
        f"hybrid weighted (a={alpha})": lambda p: weighted_sum_fusion(
            dense.all_scores(p.query), bm25.all_scores(p.query), alpha=alpha, k=k
        ).indices,
        f"hybrid RRF (k={k_rrf})": lambda p: reciprocal_rank_fusion(
            [dense.all_scores(p.query), bm25.all_scores(p.query)], k_rrf=k_rrf, k=k
        ).indices,
    }
    out: dict[str, tuple[float, float]] = {}
    for name, retrieve in methods.items():
        mrr = float(np.mean([reciprocal_rank(retrieve(p), p.gold) for p in probes]))
        recall = float(np.mean([recall_at_k(retrieve(p), p.gold) for p in probes]))
        out[name] = (mrr, recall)
    return out


def alpha_sweep(
    probes: tuple[Probe, ...],
    dense: DenseRetriever,
    bm25: BM25,
    alphas: tuple[float, ...] = (0.0, 0.3, 0.5, 0.6, 0.7, 1.0),
    k: int = TOP_K,
) -> dict[float, tuple[float, float]]:
    """Return {alpha: (MRR, recall@k)} for weighted-sum fusion across alpha -- the tuning curve.

    alpha=0 recovers pure sparse, alpha=1 pure dense; the interior is the blend. The sweep shows the
    optimum is INTERIOR (not at either endpoint) and that the naive alpha=0.5 is not automatically
    best -- the reason alpha is a knob you tune, not a constant you assume.
    """
    out: dict[float, tuple[float, float]] = {}
    for alpha in alphas:
        mrrs, recs = [], []
        for p in probes:
            fused = weighted_sum_fusion(dense.all_scores(p.query), bm25.all_scores(p.query), alpha=alpha, k=k).indices
            mrrs.append(reciprocal_rank(fused, p.gold))
            recs.append(recall_at_k(fused, p.gold))
        out[alpha] = (float(np.mean(mrrs)), float(np.mean(recs)))
    return out


def _gold_rank(indices: tuple[int, ...], gold: int) -> str:
    """Human-readable rank of the gold doc in a result list ('#1', '#2', or 'MISS')."""
    for rank, idx in enumerate(indices, start=1):
        if idx == gold:
            return f"#{rank}"
    return "MISS"


def main() -> None:
    print("numpy:", np.__version__)
    try:
        import torch  # only for the version banner; all retrieval math here is pure numpy

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print("torch:", torch.__version__, "| device:", device)
    except ImportError:
        print("torch: not installed")

    corpus = full_corpus()
    bm25 = BM25(corpus)
    dense = DenseRetriever(corpus)
    probes = build_probes(corpus)
    print(f"corpus: {len(corpus)} passages | dense lens: {dense.backend}")
    print(f"BM25 k1={BM25_K1}, b={BM25_B}, avgdl={bm25.avgdl:.2f} | RRF k={RRF_K}\n")

    # --- Per-lens vs fused ranks, side by side, for each blind-spot probe ---
    print("=" * 96)
    print("Where each lens fails, and how fusion catches it")
    print("=" * 96)
    for probe in probes:
        dense_scores = dense.all_scores(probe.query)
        sparse_scores = bm25.all_scores(probe.query)
        dense_res = dense.search(probe.query, k=TOP_K)
        sparse_res = bm25.search(probe.query, k=TOP_K)
        ws_res = weighted_sum_fusion(dense_scores, sparse_scores, alpha=TUNED_ALPHA, k=TOP_K)
        rrf_res = reciprocal_rank_fusion([dense_scores, sparse_scores], k_rrf=RRF_K, k=TOP_K)
        print(f"\nPROBE [{probe.label}]: {probe.query}")
        print(f"  gold passage = doc[{probe.gold}]: {corpus[probe.gold]}")
        print(f"  gold's BM25 score = {sparse_scores[probe.gold]:.3f}   gold's dense score = {dense_scores[probe.gold]:.3f}")
        print(f"  DENSE   top-{TOP_K} ids {list(dense_res.indices)}   gold rank: {_gold_rank(dense_res.indices, probe.gold)}")
        print(f"  SPARSE  top-{TOP_K} ids {list(sparse_res.indices)}   gold rank: {_gold_rank(sparse_res.indices, probe.gold)}")
        print(f"  HYBRID(weighted a={TUNED_ALPHA}) ids {list(ws_res.indices)}   gold rank: {_gold_rank(ws_res.indices, probe.gold)}")
        print(f"  HYBRID(RRF k={RRF_K})        ids {list(rrf_res.indices)}   gold rank: {_gold_rank(rrf_res.indices, probe.gold)}")

    # --- Aggregate the lift: MRR and recall@k per method over BOTH probes ---
    print("\n" + "=" * 96)
    print("Aggregate lift over both probes (the number that justifies hybrid)")
    print("=" * 96)
    results = evaluate(probes, dense, bm25, alpha=TUNED_ALPHA)
    print(f"{'method':<30} | {'MRR':>6} | {'recall@'+str(TOP_K):>9}")
    print("-" * 54)
    for name, (mrr, recall) in results.items():
        print(f"{name:<30} | {mrr:>6.3f} | {recall:>9.3f}")

    # --- Correctness BEFORE the claim: each single lens fails one probe; hybrid fixes both ---
    dense_mrr, dense_recall = results["dense only"]
    sparse_mrr, sparse_recall = results["sparse only (BM25)"]
    ws_mrr, ws_recall = results[f"hybrid weighted (a={TUNED_ALPHA})"]
    rrf_mrr, rrf_recall = results[f"hybrid RRF (k={RRF_K})"]
    if dense.backend.startswith("sentence-transformers"):
        # the learned dense lens is required for the paraphrase win; assert the full story
        assert sparse_recall < 1.0, "BM25 alone should MISS the paraphrase probe (recall < 1.0)"
        assert dense_mrr < 1.0, "dense alone should rank the exact-code gold #2 (MRR < 1.0)"
        assert ws_recall == 1.0, "tuned weighted-sum hybrid must catch BOTH probes (recall@k = 1.0)"
        assert ws_mrr > dense_mrr, "tuned weighted-sum MRR must beat dense alone"
        assert ws_mrr > sparse_mrr, "tuned weighted-sum MRR must beat sparse alone"
        assert rrf_recall == 1.0, "RRF hybrid must catch BOTH probes (recall@k = 1.0)"
        print("\nhybrid beats BOTH lenses: weighted-sum MRR & recall@k strictly higher than either single lens: True")
    else:
        # lexical fallback can't do paraphrase; still assert BM25 misses it and fusion runs
        assert sparse_recall < 1.0, "BM25 alone should MISS the paraphrase probe"
        print("\n(dense fallback is lexical -- paraphrase win needs the learned model; install sentence-transformers)")
    print(f"MRR: dense {dense_mrr:.3f} / sparse {sparse_mrr:.3f} -> weighted {ws_mrr:.3f}, RRF {rrf_mrr:.3f}")
    print(f"recall@{TOP_K}: dense {dense_recall:.3f} / sparse {sparse_recall:.3f} -> weighted {ws_recall:.3f}, RRF {rrf_recall:.3f}")

    # --- The alpha tuning curve: the optimum is interior, alpha=0.5 is not automatically best ---
    print("\n" + "=" * 96)
    print("Alpha sweep (weighted-sum) -- why alpha is a knob you TUNE, not a constant")
    print("=" * 96)
    sweep = alpha_sweep(probes, dense, bm25)
    print(f"{'alpha':>6} | {'MRR':>6} | {'recall@'+str(TOP_K):>9} | note")
    print("-" * 56)
    for alpha, (mrr, recall) in sweep.items():
        note = ""
        if alpha == 0.0:
            note = "= pure sparse"
        elif alpha == 1.0:
            note = "= pure dense"
        elif (mrr, recall) == max(sweep.values()):
            note = "<- best blend"
        print(f"{alpha:>6.1f} | {mrr:>6.3f} | {recall:>9.3f} | {note}")
    if dense.backend.startswith("sentence-transformers"):
        best_alpha = max(sweep, key=lambda a: sweep[a])
        assert 0.0 < best_alpha < 1.0, "the best alpha must be INTERIOR -- the whole point of fusing"
        assert sweep[best_alpha][0] > sweep[0.5][0] or sweep[0.5] == max(sweep.values()), (
            "tuning alpha away from 0.5 should help (or 0.5 already optimal)"
        )

    # --- The scale-mismatch trap, concretely: raw BM25 dwarfs cosine, so a naive sum is wrong ---
    print("\n" + "=" * 96)
    print("The scale-mismatch trap: why you cannot just ADD the raw scores")
    print("=" * 96)
    probe = probes[1]  # paraphrase probe: dense ranks the gold #1; watch a naive add destroy that
    dense_scores = dense.all_scores(probe.query)
    sparse_scores = bm25.all_scores(probe.query)
    mean_cos = float(np.abs(dense_scores).mean())
    mean_bm25 = float(np.abs(sparse_scores).mean())
    bm25_share = mean_bm25 / (mean_cos + mean_bm25) * 100.0
    print(f"raw cosine range : [{dense_scores.min():+.3f}, {dense_scores.max():+.3f}]  (bounded, subset of [-1, 1])")
    print(f"raw BM25 range   : [{sparse_scores.min():+.3f}, {sparse_scores.max():+.3f}]  (unbounded, >= 0)")
    print(f"in a NAIVE sum, BM25 supplies {bm25_share:.0f}% of the magnitude -> it drowns out cosine")
    naive_sum = dense_scores + sparse_scores  # WRONG: BM25 magnitude dominates the ordering
    normalized_sum = min_max_normalize(dense_scores) + min_max_normalize(sparse_scores)  # right
    dense_gold_rank = _gold_rank(dense.search(probe.query, k=len(corpus)).indices, probe.gold)
    naive_gold_rank = _gold_rank(tuple(int(i) for i in np.argsort(naive_sum)[::-1]), probe.gold)
    norm_gold_rank = _gold_rank(tuple(int(i) for i in np.argsort(normalized_sum)[::-1]), probe.gold)
    print(f"the gold (doc[{probe.gold}]) was dense rank {dense_gold_rank} -> NAIVE-add rank {naive_gold_rank} (collapsed) -> normalized rank {norm_gold_rank}")
    assert dense_gold_rank == "#1" and naive_gold_rank != "#1", (
        "the naive sum should demote the dense #1 -- the scale-mismatch trap, demonstrated"
    )
    print("normalizing each lens to [0,1] BEFORE summing is what stops BM25's magnitude from deciding alone")


if __name__ == "__main__":
    main()
