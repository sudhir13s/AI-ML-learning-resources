"""From-scratch re-ranking: a bi-encoder retrieves, a cross-encoder re-ranks for precision.

First-stage retrieval (a bi-encoder, BM25, or hybrid) ranks passages by INDEPENDENT encodings: it
embeds the query once and every passage once, then compares vectors. That is fast and precomputable
-- but coarse, because the query and the passage never "see" each other. The truly-best passage
often lands in the top-k yet not at the top, because a bi-encoder cannot model fine-grained
query<->document token interaction (it scores "deputy project lead" and "leadership rotates
annually" as almost equally relevant).

A CROSS-ENCODER fixes this: it feeds the query and ONE passage through the transformer TOGETHER
([CLS] query [SEP] passage), so every query token can attend to every passage token, and emits a
single scalar relevance score. Far more accurate -- but it runs the model once per (query, passage)
pair at query time, so it cannot precompute passage vectors and is far too slow for first-stage
retrieval over a big corpus. The standard architecture uses BOTH: retrieve top-k with the cheap
bi-encoder, then re-rank those few with the expensive cross-encoder. Cheap RECALL, then expensive
PRECISION.

This module builds the two-stage pipeline so every step is inspectable:
  * a bi-encoder retriever (all-MiniLM-L6-v2, chapter 3's embedder) -- the fast first stage;
  * a cross-encoder re-ranker (cross-encoder/ms-marco-MiniLM-L-6-v2) -- the precise second stage,
    with a deterministic from-scratch fallback (a transparent term-interaction scorer) so the
    module never hard-fails on a missing model download;
  * ranking-quality metrics from scratch: nDCG@k and MRR.

The headline demonstration: on a corpus of look-alike "Helios-7 leadership" passages, the
bi-encoder ranks the true answer #4 (outside the top-3), and the cross-encoder re-ranks it to #1 --
nDCG@3 0.000 -> 1.000, MRR 0.250 -> 1.000, asserted before it is claimed.

Verified on Python 3.12 / numpy 2.x / sentence-transformers (both MiniLM models, CPU). The models
load deterministically on CPU; the metrics are pure arithmetic.

Run:
    python reranking.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- Hyperparameters (hoisted; no magic numbers inline) ----------------------------------------
BI_ENCODER_MODEL = "all-MiniLM-L6-v2"  # chapter 3's bi-encoder -- the fast first stage
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # the precise second-stage re-ranker
RETRIEVE_K = 10  # how many candidates the bi-encoder retrieves into the re-rank pool
FINAL_K = 3  # how many we keep after re-ranking (the funnel's narrow end)
NDCG_KS = (3, 5, 10)  # the cutoffs we report nDCG at

# A query whose answer ("deputy project lead") is one specific role among many look-alike
# leadership passages. A bi-encoder scores them all similarly (shared leadership vocabulary); only
# joint query<->passage attention ties "deputy project lead" to the right person.
QUERY = "Who is the deputy project lead for Helios-7?"

# The corpus: doc[0] is the GOLD (it literally answers the query). The rest are topically-near
# distractors -- other Helios-7 roles / leadership statements -- chosen so the bi-encoder ranks
# several of them above the gold, the exact failure a cross-encoder repairs.
CORPUS: tuple[str, ...] = (
    "The deputy project lead for Helios-7 is Dr. Amara Okoye, who runs the Nairobi office.",  # 0 GOLD
    "Dr. Liang Wei serves as the principal investigator of the Helios-7 science mission.",  # 1
    "Helios-7's mission operations are directed by flight director Captain Elena Reyes.",  # 2
    "The Helios-7 program is overseen by a steering committee of the partner agencies.",  # 3
    "The launch operations lead for Helios-7 coordinated the Kourou countdown sequence.",  # 4
    "Project leadership responsibilities for Helios-7 rotate annually among the agencies.",  # 5
    "The systems engineering lead for Helios-7 owns the spacecraft integration schedule.",  # 6
    "The Helios-7 ground segment lead manages the network of receiving stations.",  # 7
    "The chief scientist for the Helios-7 payload reports to the principal investigator.",  # 8
    "The deputy flight director supports Captain Reyes during Helios-7 operations.",  # 9
    "The Helios-7 mission director sets overall priorities for the program.",  # 10
    "The lead engineer for the Helios-7 imager designed its calibration procedure.",  # 11
)
GOLD_INDEX = 0  # the single passage that actually answers QUERY


@dataclass(frozen=True)
class RankedList:
    """A ranked retrieval/re-rank result: doc indices best-first, with their scores."""

    indices: tuple[int, ...]
    scores: tuple[float, ...]


# ================================================================================================
# Stage 1 -- the bi-encoder retriever (independent encodings; fast, precomputable)
# ================================================================================================


class BiEncoderRetriever:
    """Dense first-stage retrieval: encode the query and every passage INDEPENDENTLY, rank by cosine.

    This is chapter 3's bi-encoder. The passage vectors are precomputed once (here at construction);
    at query time we embed only the query and take dot products -- O(N) cheap comparisons, no
    transformer forward pass per passage. That independence is exactly what makes it fast and exactly
    why it is coarse: the query never attends to the passage, so it cannot weigh which passage truly
    *answers* the query versus merely shares its topic.
    """

    def __init__(self, corpus: tuple[str, ...], model_name: str = BI_ENCODER_MODEL) -> None:
        self.corpus = corpus
        self.backend, self._encode = _load_bi_encoder(model_name)
        self.doc_vectors = self._encode(list(corpus))  # (N, d) unit-norm rows -- precomputed once

    def all_scores(self, query: str) -> np.ndarray:
        """Cosine score for every passage (unit-norm rows => dot product == cosine)."""
        q_vec = self._encode([query])[0]
        return self.doc_vectors @ q_vec

    def retrieve(self, query: str, k: int = RETRIEVE_K) -> RankedList:
        """Top-k passages by cosine similarity (best-first) -- the candidate pool for re-ranking."""
        scores = self.all_scores(query)
        order = np.argsort(scores)[::-1][:k]
        return RankedList(tuple(int(i) for i in order), tuple(float(scores[i]) for i in order))


# ================================================================================================
# Stage 2 -- the cross-encoder re-ranker (joint encoding; accurate, query-time only)
# ================================================================================================


class CrossEncoderReranker:
    """Precision second stage: score each (query, passage) pair by JOINT encoding.

    A cross-encoder concatenates the query and one passage ([CLS] query [SEP] passage) and runs the
    transformer over the pair, so query tokens attend to passage tokens and back. The output head
    emits a single scalar relevance logit. Because the score depends on the query<->passage
    interaction, it CANNOT be precomputed (a passage's score is different for every query) and costs
    one transformer forward pass per pair -- which is why you only ever run it on the top-k the
    bi-encoder already retrieved, never the whole corpus.

    Uses the real `cross-encoder/ms-marco-MiniLM-L-6-v2`. If sentence-transformers / the model is
    unavailable, falls back to a transparent term-interaction scorer (see `_TermInteractionScorer`)
    so the module still runs end to end -- the printed banner says which scorer is active.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL) -> None:
        self.backend, self._score_pairs = _load_cross_encoder(model_name)

    def scores_for(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance logit for each (query, passage) pair -- one joint forward pass per passage."""
        return self._score_pairs([(query, p) for p in passages])

    def rerank(self, query: str, candidate_indices: tuple[int, ...], corpus: tuple[str, ...]) -> RankedList:
        """Re-score the candidate pool with joint encoding and return it re-ordered, best-first.

        Note this only REORDERS the candidates it is given -- it cannot add a passage the first stage
        failed to retrieve (the recall-ceiling limit the page discusses).
        """
        passages = [corpus[i] for i in candidate_indices]
        scores = self.scores_for(query, passages)
        order = np.argsort(scores)[::-1]
        reranked = tuple(int(candidate_indices[j]) for j in order)
        reranked_scores = tuple(float(scores[j]) for j in order)
        return RankedList(reranked, reranked_scores)


# ================================================================================================
# Model loaders (real models, with deterministic fallbacks so the module never hard-fails)
# ================================================================================================


def _silence_hf() -> None:
    """Quiet the HF/transformers load chatter so notebook output stays clean."""
    import logging
    import os

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for name in ("sentence_transformers", "transformers", "transformers.modeling_utils"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _load_bi_encoder(model_name: str):
    """Return (backend_name, encode_fn). encode_fn: list[str] -> (n, d) unit-norm array."""
    try:
        import contextlib
        import io

        _silence_hf()
        with contextlib.redirect_stderr(io.StringIO()):
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device="cpu")

        def encode(texts: list[str]) -> np.ndarray:
            return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))

        return f"sentence-transformers/{model_name}", encode
    except Exception:  # no package / no network / model not cached -> deterministic lexical fallback

        def encode_fallback(texts: list[str]) -> np.ndarray:
            return _bow_unit_vectors(texts)

        return "bow-fallback (lexical, deterministic)", encode_fallback


def _load_cross_encoder(model_name: str):
    """Return (backend_name, score_fn). score_fn: list[(q, p)] -> np.ndarray of relevance logits."""
    try:
        import contextlib
        import io

        _silence_hf()
        with contextlib.redirect_stderr(io.StringIO()):
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, device="cpu")

        def score(pairs: list[tuple[str, str]]) -> np.ndarray:
            return np.asarray(model.predict(pairs))

        return f"sentence-transformers/{model_name}", score
    except Exception:  # fallback: a transparent joint term-interaction scorer (see class below)
        scorer = _TermInteractionScorer()
        return "term-interaction-fallback (joint, deterministic)", scorer.score_pairs


class _TermInteractionScorer:
    """A transparent stand-in for a cross-encoder when the real model is unavailable.

    A real cross-encoder learns query<->passage interaction. This deterministic fallback approximates
    the SIGNAL a bi-encoder structurally cannot capture: it rewards a passage for containing the
    query's *content* terms together (an interaction feature), not just for being topically near.
    It is NOT a trained model -- it exists only so the module runs end to end offline; the printed
    banner makes the fallback explicit so no one mistakes it for the real thing.
    """

    _STOPWORDS = frozenset(
        {"the", "a", "an", "is", "are", "was", "of", "for", "who", "what", "which", "to", "in", "on"}
    )

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        import re

        token_re = re.compile(r"[a-z0-9-]+")

        def content_terms(text: str) -> set[str]:
            return {t for t in token_re.findall(text.lower()) if t not in self._STOPWORDS}

        scores = []
        for query, passage in pairs:
            q_terms = content_terms(query)
            p_terms = content_terms(passage)
            overlap = len(q_terms & p_terms)
            coverage = overlap / max(len(q_terms), 1)  # fraction of query content terms present
            scores.append(float(overlap) + 2.0 * coverage)  # interaction signal, higher = better
        return np.asarray(scores)


# ================================================================================================
# Fallback bi-encoder embedding (only used if sentence-transformers is unavailable)
# ================================================================================================


def _bow_unit_vectors(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-words unit vectors over a shared vocabulary -- lexical fallback only."""
    import re

    token_re = re.compile(r"[a-z0-9-]+")
    tokenized = [token_re.findall(t.lower()) for t in texts]
    vocab = sorted({tok for toks in tokenized for tok in toks})
    index = {tok: i for i, tok in enumerate(vocab)}
    vectors = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for row, toks in enumerate(tokenized):
        for tok in toks:
            vectors[row, index[tok]] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# ================================================================================================
# Ranking-quality metrics, from scratch
# ================================================================================================


def dcg_at_k(order: tuple[int, ...], gold: int, k: int) -> float:
    """Discounted Cumulative Gain at k for a single relevant doc (binary relevance).

    DCG = sum over the top-k of rel_i / log2(i + 1), with i the 1-based rank. With one relevant doc
    (rel=1 for the gold, 0 otherwise) only the gold contributes, and its contribution shrinks with
    rank -- the log discount is what makes ranking the answer *higher* score better.
    """
    dcg = 0.0
    for rank, doc in enumerate(order[:k], start=1):
        relevance = 1.0 if doc == gold else 0.0
        dcg += relevance / np.log2(rank + 1)
    return dcg


def ndcg_at_k(order: tuple[int, ...], gold: int, k: int) -> float:
    """Normalized DCG at k: DCG / IDCG. IDCG is the best possible (gold at rank 1).

    Dividing by the ideal DCG puts the score in [0, 1]: 1.0 means the gold is ranked #1, 0.0 means it
    is absent from the top-k. With one gold, IDCG = 1 / log2(2) = 1, so nDCG@k here equals the gold's
    own discounted gain -- 1.0 at rank 1, ~0.63 at rank 2, 0.0 once it falls past rank k.
    """
    idcg = 1.0 / np.log2(1 + 1)  # ideal: the single gold at rank 1
    return dcg_at_k(order, gold, k) / idcg


def reciprocal_rank(order: tuple[int, ...], gold: int) -> float:
    """1/rank of the gold (0 if absent). MRR averages this over a query set."""
    for rank, doc in enumerate(order, start=1):
        if doc == gold:
            return 1.0 / rank
    return 0.0


# ================================================================================================
# The two-stage pipeline
# ================================================================================================


def retrieve_then_rerank(
    query: str,
    corpus: tuple[str, ...],
    bi: BiEncoderRetriever,
    cross: CrossEncoderReranker,
    retrieve_k: int = RETRIEVE_K,
) -> tuple[RankedList, RankedList]:
    """Run stage 1 (bi-encoder retrieve top-k) then stage 2 (cross-encoder re-rank the pool).

    Returns (bi_ranked_full, reranked): the bi-encoder's full ranking and the re-ranked pool, so a
    caller can compare the two orderings and measure the lift.
    """
    bi_full = bi.retrieve(query, k=len(corpus))  # full bi-encoder ranking, for the before/after
    pool = bi.retrieve(query, k=retrieve_k).indices  # the candidates we actually re-rank
    reranked = cross.rerank(query, pool, corpus)
    return bi_full, reranked


def main() -> None:
    print("numpy:", np.__version__)
    try:
        import torch  # version banner only; the metrics are pure numpy

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

    bi = BiEncoderRetriever(CORPUS)
    cross = CrossEncoderReranker()
    print(f"corpus: {len(CORPUS)} passages | retrieve_k={RETRIEVE_K} -> final_k={FINAL_K}")
    print(f"bi-encoder : {bi.backend}")
    print(f"cross-enc  : {cross.backend}\n")

    bi_full, reranked = retrieve_then_rerank(QUERY, CORPUS, bi, cross)

    # --- Before/after ranking of the gold ---
    bi_gold_rank = bi_full.indices.index(GOLD_INDEX) + 1
    rr_gold_rank = reranked.indices.index(GOLD_INDEX) + 1 if GOLD_INDEX in reranked.indices else None
    print("=" * 88)
    print("Bi-encoder retrieval vs cross-encoder re-ranking")
    print("=" * 88)
    print(f"QUERY: {QUERY}")
    print(f"GOLD : doc[{GOLD_INDEX}]: {CORPUS[GOLD_INDEX]}\n")
    print(f"bi-encoder full ranking (doc ids): {list(bi_full.indices)}")
    print(f"  gold doc[{GOLD_INDEX}] bi-encoder rank: #{bi_gold_rank}  (outside the top-{FINAL_K} cutoff)")
    print(f"re-ranked pool (doc ids):          {list(reranked.indices)}")
    print(f"  gold doc[{GOLD_INDEX}] re-ranked rank: #{rr_gold_rank}\n")

    # --- The measured lift: nDCG@k and MRR, before vs after ---
    print(f"{'metric':<10} | {'bi-encoder':>11} | {'re-ranked':>10}")
    print("-" * 38)
    for k in NDCG_KS:
        before = ndcg_at_k(bi_full.indices, GOLD_INDEX, k)
        after = ndcg_at_k(reranked.indices, GOLD_INDEX, k)
        print(f"{'nDCG@'+str(k):<10} | {before:>11.3f} | {after:>10.3f}")
    mrr_before = reciprocal_rank(bi_full.indices, GOLD_INDEX)
    mrr_after = reciprocal_rank(reranked.indices, GOLD_INDEX)
    print(f"{'MRR':<10} | {mrr_before:>11.3f} | {mrr_after:>10.3f}")

    # --- Correctness BEFORE the claim ---
    assert bi_gold_rank > FINAL_K, "the demo needs the bi-encoder to bury the gold past the top-k"
    assert rr_gold_rank == 1, "the cross-encoder must re-rank the gold to #1"
    assert ndcg_at_k(bi_full.indices, GOLD_INDEX, FINAL_K) < ndcg_at_k(reranked.indices, GOLD_INDEX, FINAL_K)
    assert mrr_after > mrr_before, "re-ranking must raise MRR"
    print("\nre-ranking lifts the gold from outside the top-k to #1; nDCG@k and MRR strictly improve: True")

    # --- The recall ceiling: rerank can only REORDER what stage 1 retrieved ---
    print("\n" + "=" * 88)
    print("The recall ceiling: re-ranking cannot rescue a passage retrieval missed")
    print("=" * 88)
    narrow_pool = bi.retrieve(QUERY, k=FINAL_K).indices  # retrieve ONLY top-3, then rerank
    narrow_rerank = cross.rerank(QUERY, narrow_pool, CORPUS)
    print(f"retrieve only top-{FINAL_K} -> pool {list(narrow_pool)} | gold in pool: {GOLD_INDEX in narrow_pool}")
    if GOLD_INDEX in narrow_rerank.indices:
        print(f"  re-ranked gold rank: #{narrow_rerank.indices.index(GOLD_INDEX) + 1}")
    else:
        print("  re-ranked gold rank: MISS -- unrecoverable (rerank reorders, it cannot retrieve)")
    assert GOLD_INDEX not in narrow_pool, "this demo needs the narrow pool to exclude the gold"
    print(f"  => retrieve a WIDE pool (k={RETRIEVE_K}) so the gold is present for re-ranking to find.")


if __name__ == "__main__":
    main()
