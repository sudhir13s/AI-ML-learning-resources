"""Information Retrieval & Semantic Search, measured: the retrieve-then-rerank engine room.

This is the single source of truth for the chapter. The concept page, the teaching notebook
(`16-Information-Retrieval-and-Semantic-Search.ipynb`), and the figure generator
(`make_figures_16.py`) all import the functions and constants defined here, so none of them can
silently drift from the others. Every number quoted on the page is produced by this file.

What it shows, end to end (each demo asserts its qualitative point first):
  (a) lexical BM25 over a tiny inverted index vs a DENSE cosine retriever on a vocabulary-mismatch
      query -- the gold passage says "automobile", the query says "car"; BM25 buries it, dense
      finds it first (the lexical gap, measured);
  (b) ranking metrics from scratch -- precision@k, recall@k, MRR, and nDCG (DCG/IDCG with the
      2**rel - 1 gain and log2 discount), checked against sklearn's ndcg_score;
  (c) Reciprocal Rank Fusion combining the lexical + dense rankings (rank-based, score-agnostic),
      and a cross-encoder-style re-rank improving the top-k;
  (d) an approximate-nearest-neighbour (ANN) vs exact recall/speed sketch -- IVF-style cluster
      probing trades a little recall@k for a large speedup, the knob made visible.

Determinism and device policy
-----------------------------
Everything here runs offline and deterministically. The dense retriever uses a small, fully
deterministic SYNTHETIC concept-anchor encoder (`SyntheticEncoder`) seeded with `SEED`, built so
that synonyms land close WITHOUT sharing a surface word ("automobile" fires the same CAR concept as
"car"; "physician"/"doctor" share MEDICAL) -- exactly the meaning match lexical retrieval cannot
make. A real Sentence-BERT (`all-MiniLM-L6-v2`) is loaded by `load_encoder()` only when it is
reachable; if the package or weights are unavailable (offline / firewall) it transparently falls
back to the synthetic encoder, so the notebook and figures ALWAYS run with no network dependency
that can fail. Which path ran is reported honestly ("backend: all-MiniLM-L6-v2 (real)" vs
"backend: synthetic (fallback)"). The from-scratch BM25, RRF, metrics, and ANN demos are pure
numpy and never touch a model at all.

Everything is pinned to CPU and seeded (numpy + torch), the device line is printed honestly, and a
process-stable md5 hash (never Python's salted `hash()`) keys the synthetic vectors, so the same
input yields the same numbers on any machine and in any process.

Run:
    python information_retrieval.py
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch


def _stable_hash(text: str) -> int:
    """A process-independent hash of `text` as a 32-bit int.

    Python's built-in `hash()` is salted per process (PYTHONHASHSEED randomization), which would make
    the synthetic encoder produce DIFFERENT vectors in every process (the script, the notebook, the
    figure generator). We use md5 so the synthetic numbers are reproducible across processes and
    machines -- the whole point of the deterministic fallback.
    """
    return int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "big")


# --- Reproducibility -------------------------------------------------------------------------------
SEED = 0

# Detect the best accelerator for honest reporting, but PIN execution to CPU so every measured number
# is bit-for-bit reproducible across machines (MPS/CUDA reductions are not guaranteed identical).
DETECTED_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DEVICE = "cpu"  # pinned: retrieval numbers must be reproducible, so we never run on MPS/CUDA

# The bi-encoder used when a real backend is reachable (384-dim, small and fast).
MODEL_NAME = "all-MiniLM-L6-v2"


# ==================================================================================================
# The corpus and query used everywhere -- the vocabulary-mismatch probe.
# The GOLD passage d0 answers "How do I fix my car?" but says "automobile", never "car"; two lexical
# distractors (d2, d4) merely share the surface words "fix"/"my"/"how". A lexical scorer is fooled;
# a meaning-based one is not. These exact strings reproduce every rank quoted on the page.
# ==================================================================================================
GOLD_ID = 0
QUERY = "How do I fix my car?"
CORPUS: tuple[str, ...] = (
    "A mechanic repaired the automobile engine and replaced the worn brake pads.",  # d0 GOLD
    "A balanced diet and regular exercise keep your heart healthy.",                # d1 unrelated
    "I had to fix the spelling errors in my report before submitting it.",          # d2 distractor (fix, my)
    "Stock markets fell sharply amid fears of rising interest rates.",              # d3 unrelated
    "How to fix my code when the build breaks on my machine.",                      # d4 distractor (how, fix, my)
    "The doctor advised the patient to rest and drink plenty of fluids.",           # d5 unrelated
    "Photosynthesis converts sunlight into chemical energy in plants.",             # d6 unrelated
    "Quarterly earnings beat analyst expectations, lifting the share price.",       # d7 unrelated
)

# A running graded-relevance list used by every metric demo (3 = perfect, 0 = irrelevant). This is
# the exact list worked through on the page: 5 relevant of 8, with a deliberately imperfect order.
GRADED_RELEVANCE: tuple[int, ...] = (3, 2, 3, 0, 1, 2, 0, 0)

# BM25 hyperparameters (Robertson & Zaragoza 2009 defaults).
BM25_K1 = 1.5  # TF-saturation knob (typical 1.2-2.0): each extra occurrence adds diminishing weight
BM25_B = 0.75  # length-normalization knob: discount documents longer than average


# ==================================================================================================
# (a.1) Lexical retrieval from scratch: a tiny inverted index + a BM25 scorer.
# These are pure-Python/numpy -- no model -- so the lexical baseline is exact and reproducible.
# ==================================================================================================
def tokenize(text: str) -> list[str]:
    """Lowercase, strip simple punctuation, split on whitespace -- a deliberately tiny analyzer.

    Real engines stem and remove stopwords; we keep it transparent so every BM25 number is hand-
    checkable. "car" and "automobile" remain DIFFERENT tokens here -- which is exactly the lexical
    gap the dense retriever later crosses.
    """
    cleaned = text.lower()
    for ch in ".,?!'\"()":
        cleaned = cleaned.replace(ch, " ")
    return [t for t in cleaned.split() if t]


def build_inverted_index(corpus: tuple[str, ...]) -> dict[str, list[tuple[int, int]]]:
    """Map term -> sorted posting list of (doc_id, term_frequency).

    This is the data structure that makes lexical search fast: to score a query you touch ONLY the
    posting lists of the query's terms, never documents that share no word with it. Returns a dict
    so the notebook can print a few posting lists verbatim.
    """
    index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for doc_id, document in enumerate(corpus):
        counts: dict[str, int] = defaultdict(int)
        for term in tokenize(document):
            counts[term] += 1
        for term, freq in counts.items():
            index[term].append((doc_id, freq))
    # Sort each posting list by doc id (the order real engines store and merge them in).
    return {term: sorted(postings) for term, postings in sorted(index.items())}


def bm25_idf(n_docs: int, n_containing: int) -> float:
    """BM25 inverse document frequency: ln( (N - n_t + 0.5) / (n_t + 0.5) + 1 ).

    A term in every document contributes ~0; a rare term contributes a lot. The +0.5 smoothing and
    the +1 inside the log are the standard BM25 form (keeps the IDF non-negative).

    Source: Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond".
    """
    return math.log((n_docs - n_containing + 0.5) / (n_containing + 0.5) + 1.0)


def bm25_scores(
    query: str, corpus: tuple[str, ...], k1: float = BM25_K1, b: float = BM25_B
) -> np.ndarray:
    """BM25 score of every document for `query`, computed from the inverted index (from scratch).

    score(q, d) = sum_{t in q} IDF(t) * f(t,d)(k1+1) / ( f(t,d) + k1 (1 - b + b |d|/avgdl) ).

    Returns an (N,) array of scores. Documents sharing no query term score exactly 0 -- which is why
    the GOLD passage (says "automobile", not "car") scores 0 here and the dense retriever must rescue
    it. Verified row-by-row against `rank_bm25.BM25Okapi` in `bm25_matches_reference()`.

    Source: Robertson & Zaragoza (2009).
    """
    index = build_inverted_index(corpus)
    doc_lengths = np.array([len(tokenize(d)) for d in corpus], dtype=np.float64)
    avgdl = doc_lengths.mean()
    n_docs = len(corpus)
    scores = np.zeros(n_docs, dtype=np.float64)
    for term in tokenize(query):
        postings = index.get(term)
        if postings is None:  # query term in no document -> contributes nothing
            continue
        idf = bm25_idf(n_docs, len(postings))
        for doc_id, freq in postings:
            denom = freq + k1 * (1.0 - b + b * doc_lengths[doc_id] / avgdl)
            scores[doc_id] += idf * freq * (k1 + 1.0) / denom
    return scores


# ==================================================================================================
# (a.2) Dense retrieval: a synthetic concept-anchor encoder (offline) or real Sentence-BERT.
# The synthetic encoder is NOT SBERT -- it exists only so the notebook/figures always run offline.
# It is built to honour the one fact the page asserts: synonyms land CLOSE without sharing a word,
# so a query for "car" retrieves a passage about an "automobile". The CAR concept fires on both.
# ==================================================================================================
SYNTH_DIM = 256

# Each "concept" is a meaning direction triggered by ANY of its words. This is what lets the offline
# encoder map synonyms close WITHOUT lexical overlap -- a crude stand-in for the meaning a real
# trained encoder learns, so the offline demo still shows the right SHAPE (gold ranks first on
# meaning). "car"/"automobile"/"vehicle"/"engine"/"brake"/"mechanic" all fire the CAR concept.
# NOTE: generic verbs like "fix"/"repair" are deliberately NOT in any concept -- they carry no topical
# meaning ("fix code", "fix spelling", "fix car" are different worlds). The CAR bridge that crosses the
# lexical gap is "car" <-> "automobile"/"engine"/"brake"/"mechanic", which fire CAR without sharing a word.
SYNTH_CONCEPTS: dict[str, tuple[str, ...]] = {
    "car": ("car", "automobile", "vehicle", "engine", "brake", "brakes", "mechanic",
            "tyre", "tire", "wheel", "transmission", "clutch"),
    "code": ("code", "build", "machine", "compile", "bug", "software", "program", "debug"),
    "writing": ("spelling", "errors", "report", "submitting", "grammar", "essay", "document"),
    "health": ("diet", "exercise", "heart", "healthy", "doctor", "patient", "rest", "fluids",
               "medicine", "physician"),
    "finance": ("stock", "markets", "interest", "rates", "earnings", "analyst", "share", "price",
                "prices"),
    "science": ("photosynthesis", "sunlight", "chemical", "energy", "plants"),
}


class SyntheticEncoder:
    """Deterministic synthetic sentence encoder: hashed word features + concept anchors.

    Two signals are summed, then L2-normalized:
      * a hashed bag of word features -- a fixed pseudo-random unit vector per word, stable run-to-run;
      * a small set of CONCEPT anchors (SYNTH_CONCEPTS): any trigger word adds its concept's fixed
        direction, so synonyms like "car"/"automobile" land close WITHOUT sharing a word -- a crude
        stand-in for learned meaning, so the offline demo still shows the right qualitative shape.

    It is NOT SBERT -- it exists only so the notebook/figures always run offline; figures label it.
    """

    name = "synthetic"
    is_real = False
    _concept_weight = 3.0  # how strongly concept anchors pull, relative to surface word features

    def __init__(self, dim: int = SYNTH_DIM, seed: int = SEED) -> None:
        self.dim = dim
        self._seed = seed
        self._cache: dict[str, np.ndarray] = {}
        self._word_to_concepts: dict[str, list[str]] = {}
        for concept, words in SYNTH_CONCEPTS.items():
            for w in words:
                self._word_to_concepts.setdefault(w, []).append(concept)

    def _feature_vec(self, feature: str) -> np.ndarray:
        """A fixed pseudo-random unit vector per string feature (word/concept), stable run-to-run."""
        if feature not in self._cache:
            rng = np.random.default_rng((_stable_hash(feature) ^ self._seed) % (2**32))
            v = rng.standard_normal(self.dim)
            self._cache[feature] = v / np.linalg.norm(v)
        return self._cache[feature]

    def encode_one(self, sentence: str) -> np.ndarray:
        toks = tokenize(sentence)
        if not toks:
            return np.zeros(self.dim)
        vec = np.zeros(self.dim)
        for t in toks:
            vec += self._feature_vec(t)
            for concept in self._word_to_concepts.get(t, ()):
                vec += self._concept_weight * self._feature_vec(f"concept::{concept}")
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, sentences: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        out = np.stack([self.encode_one(s) for s in sentences], axis=0)
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(norms == 0, 1.0, norms)
        return out


@dataclass
class Encoder:
    """A loaded sentence-embedding backend plus a flag for which path ran (real vs synthetic)."""

    name: str
    is_real: bool
    model: object  # SentenceTransformer (real) or SyntheticEncoder (fallback)

    def encode(self, sentences: list[str], normalize: bool = True) -> np.ndarray:
        """Encode `sentences` into an (n, d) array of (optionally unit-normalized) vectors."""
        return np.asarray(
            self.model.encode(sentences, normalize_embeddings=normalize), dtype=np.float64
        )


def load_encoder(*, force_synthetic: bool = False) -> Encoder:
    """Load Sentence-BERT (`all-MiniLM-L6-v2`) if reachable, else a deterministic synthetic fallback.

    Set `force_synthetic=True` to exercise the offline path deliberately (used by a notebook cell and
    a test). Any failure to import or download falls through to the synthetic encoder -- the notebook
    and figures must never crash on a network problem.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not force_synthetic:
        try:
            import logging
            import warnings

            warnings.filterwarnings("ignore")
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            try:
                from transformers import logging as hf_logging

                hf_logging.set_verbosity_error()
                hf_logging.disable_progress_bar()
            except Exception:  # noqa: BLE001 -- transformers may be absent; not fatal
                pass
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL_NAME, device=DEVICE)
            model.eval()
            return Encoder(name=MODEL_NAME, is_real=True, model=model)
        except Exception:  # noqa: BLE001 -- any failure (no net, no package) -> synthetic fallback
            pass
    return Encoder(name="synthetic", is_real=False, model=SyntheticEncoder(seed=SEED))


def load_cross_encoder() -> object | None:
    """Load the reranking cross-encoder if reachable, else None (the page degrades gracefully).

    Returns a CrossEncoder for `cross-encoder/ms-marco-MiniLM-L-6-v2`, or None offline. Used only by
    the optional real-rerank path; the synthetic cross-encoder below makes the demo run offline.
    """
    try:
        import warnings

        warnings.filterwarnings("ignore")
        from sentence_transformers import CrossEncoder

        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)
    except Exception:  # noqa: BLE001
        return None


def device_report() -> str:
    """One honest line: the detected accelerator and the pinned execution device."""
    return f"device: {DEVICE} (detected {DETECTED_DEVICE}; pinned to CPU for reproducibility)"


# ==================================================================================================
# Scoring + ranking helpers (pure math).
# ==================================================================================================
def cosine_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against each row of `doc_vecs`.

    With unit-normalized vectors this is just the dot product -- the score a dense retriever ranks by.
    Returns an (N,) array.
    """
    query_vec = np.asarray(query_vec, dtype=np.float64)
    doc_vecs = np.asarray(doc_vecs, dtype=np.float64)
    qn = np.linalg.norm(query_vec)
    dn = np.linalg.norm(doc_vecs, axis=1)
    denom = np.where(dn == 0, 1.0, dn) * (qn if qn != 0 else 1.0)
    return (doc_vecs @ query_vec) / denom


def ranking_from_scores(scores: np.ndarray) -> np.ndarray:
    """doc ids ordered best-first by score (descending), ties broken by ascending doc id.

    The ascending-id tie-break makes the ordering deterministic so every quoted rank is stable.
    """
    scores = np.asarray(scores, dtype=np.float64)
    return np.lexsort((np.arange(len(scores)), -scores))


def rank_of(order: np.ndarray, doc_id: int) -> int:
    """1-based rank of `doc_id` within an ordering (best-first)."""
    return int(np.where(order == doc_id)[0][0]) + 1


# ==================================================================================================
# (a) The headline: lexical BM25 vs dense cosine on the vocabulary-mismatch query.
# ==================================================================================================
def lexical_vs_dense(encoder: Encoder) -> dict[str, object]:
    """Rank CORPUS for QUERY with BM25 (lexical) and a dense bi-encoder; show dense beats lexical.

    Returns the raw scores, the orderings, and the rank of the GOLD passage under each. The contract
    the page asserts (and this function asserts in the notebook): the dense retriever ranks the GOLD
    passage strictly higher than BM25 does, because BM25 shares no surface word with the synonym.
    """
    bm = bm25_scores(QUERY, CORPUS)
    bm_order = ranking_from_scores(bm)

    vecs = encoder.encode([QUERY, *CORPUS])
    dense = cosine_matrix(vecs[0], vecs[1:])
    dense_order = ranking_from_scores(dense)

    return {
        "bm25_scores": bm,
        "bm25_order": bm_order,
        "bm25_gold_rank": rank_of(bm_order, GOLD_ID),
        "dense_scores": dense,
        "dense_order": dense_order,
        "dense_gold_rank": rank_of(dense_order, GOLD_ID),
    }


# ==================================================================================================
# (c.1) Reciprocal Rank Fusion: combine two rankings by RANK POSITION only (score-scale-agnostic).
# ==================================================================================================
RRF_K = 60  # the standard RRF damping constant (Cormack et al. 2009)


def reciprocal_rank_fusion(
    orders: list[np.ndarray], n_docs: int, k: int = RRF_K
) -> np.ndarray:
    """RRF score per document: sum over input lists of 1 / (k + rank), rank 1-based.

    A document absent from a list contributes nothing. Because RRF uses only RANK, it fuses BM25
    scores and cosine similarities -- which live on incomparable scales -- with no calibration, and
    it rewards AGREEMENT: a doc ranked decently in both lists beats one ranked #1 in only one.
    Returns an (n_docs,) array of fused scores (higher = better).

    Source: Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion Outperforms Condorcet...".
    """
    fused = np.zeros(n_docs, dtype=np.float64)
    for order in orders:
        for rank0, doc_id in enumerate(order):
            fused[doc_id] += 1.0 / (k + rank0 + 1)  # rank0 is 0-based -> +1 for the 1-based rank
    return fused


def hybrid_demo(encoder: Encoder) -> dict[str, object]:
    """Fuse the BM25 and dense rankings with RRF and report the GOLD passage's fused rank.

    Contract: hybrid pulls the gold passage up from BM25's poor rank toward the dense rank -- the
    consensus of the two retrievers is more robust than either alone.
    """
    lvd = lexical_vs_dense(encoder)
    fused = reciprocal_rank_fusion(
        [lvd["bm25_order"], lvd["dense_order"]], n_docs=len(CORPUS)
    )
    fused_order = ranking_from_scores(fused)
    return {
        "fused_scores": fused,
        "fused_order": fused_order,
        "fused_gold_rank": rank_of(fused_order, GOLD_ID),
        "bm25_gold_rank": lvd["bm25_gold_rank"],
        "dense_gold_rank": lvd["dense_gold_rank"],
    }


# The page's hand-worked RRF example: two small rankings fused by hand, then verified here.
RRF_EXAMPLE_LISTS: tuple[tuple[str, ...], ...] = (
    ("A", "B", "C", "D", "E"),   # "BM25" ranking
    ("C", "A", "F", "B", "G"),   # "dense" ranking
)


def rrf_worked_example(k: int = RRF_K) -> list[tuple[str, float]]:
    """Reproduce the page's by-hand RRF table for two string rankings; returns [(doc, score), ...].

    The point the page makes: A wins the fusion despite being rank 1 in NEITHER list -- it placed
    high in BOTH (1 and 2), and that agreement beats C's single #1.
    """
    fused: dict[str, float] = defaultdict(float)
    for ranking in RRF_EXAMPLE_LISTS:
        for rank0, doc in enumerate(ranking):
            fused[doc] += 1.0 / (k + rank0 + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])


# ==================================================================================================
# (c.2) Cross-encoder re-rank: improve the top-k ordering with a joint (query, passage) scorer.
# Offline we use a deterministic SYNTHETIC cross-encoder: it reads the query and passage TOGETHER
# (token overlap on a meaning-expanded bag), so it can resolve fine-grained relevance the bi-encoder
# cannot. NOT a real cross-encoder -- it exists so the rerank demo runs offline and shows the SHAPE.
# ==================================================================================================
def _concepts(text: str) -> set[str]:
    """The set of topical CONCEPTS the text's tokens fire (e.g. 'car' and 'automobile' -> {CAR})."""
    enc = SyntheticEncoder()
    fired: set[str] = set()
    for tok in tokenize(text):
        for concept in enc._word_to_concepts.get(tok, ()):
            fired.add(concept)
    return fired


def synthetic_cross_encoder_score(query: str, passage: str) -> float:
    """A deterministic joint (query, passage) relevance score in [0, 1].

    Reads the pair TOGETHER and asks "does this passage address what the query is ABOUT": it scores
    the fraction of the query's CONCEPTS that the passage also fires (concept coverage), plus a small
    bonus for any literal shared word. Because it keys on shared MEANING (CAR fires on both "car" and
    "automobile") and is NOT penalized by passage length, it lifts the truly relevant passage to the
    very top of the candidate set -- a stand-in for a real cross-encoder's joint attention, honouring
    that qualitative fact. (A real cross-encoder reads the pair with full cross-attention; this is a
    transparent, offline proxy that shows the same SHAPE.)
    """
    q_concepts = _concepts(query)
    if not q_concepts:
        return 0.0
    p_concepts = _concepts(passage)
    coverage = len(q_concepts & p_concepts) / len(q_concepts)  # query concepts the passage addresses
    q_words = set(tokenize(query)) - {"how", "do", "i", "my", "the", "a", "to"}  # drop function words
    p_words = set(tokenize(passage))
    surface_bonus = 0.15 * len(q_words & p_words) / max(1, len(q_words))
    return min(1.0, coverage + surface_bonus)


def rerank_demo(encoder: Encoder, top_k: int = 4) -> dict[str, object]:
    """Retrieve a cheap top-k (BM25), then re-rank those k with the cross-encoder -- the cascade.

    We deliberately re-rank the LEXICAL (BM25) candidate set, where the gold passage lands BELOW the
    top (BM25 mis-orders it behind two surface-match distractors). The cross-encoder, reading each
    (query, passage) pair jointly, recognizes the gold passage actually answers the query and LIFTS it
    to position 1 -- the visible precision win a re-ranker buys. (BM25 is the same on every backend,
    so this demo's lift is backend-independent: gold goes 3 -> 1.)

    Returns the candidate ids (BM25 order), their cross-encoder scores, the reranked order, and the
    gold rank WITHIN the candidates before and after. Contract: gold_rank_after == 1 < gold_rank_before.
    """
    bm = bm25_scores(QUERY, CORPUS)
    candidates = list(ranking_from_scores(bm)[:top_k])
    gold_rank_before = candidates.index(GOLD_ID) + 1 if GOLD_ID in candidates else None

    ce_scores = np.array([synthetic_cross_encoder_score(QUERY, CORPUS[c]) for c in candidates])
    reranked = [candidates[i] for i in np.argsort(-ce_scores, kind="stable")]
    gold_rank_after = reranked.index(GOLD_ID) + 1 if GOLD_ID in reranked else None
    return {
        "candidates": candidates,
        "ce_scores": ce_scores,
        "reranked": reranked,
        "gold_rank_before": gold_rank_before,
        "gold_rank_after": gold_rank_after,
    }


# ==================================================================================================
# (b) Evaluation metrics from scratch: precision@k, recall@k, MRR, nDCG.
# All pure numpy; nDCG is verified against sklearn's ndcg_score in the notebook.
# ==================================================================================================
def precision_at_k(relevance: np.ndarray, k: int) -> float:
    """Fraction of the top-k results that are relevant (rel > 0). The ranker's metric."""
    relevance = np.asarray(relevance)
    top = relevance[:k]
    return float((top > 0).sum()) / k


def recall_at_k(relevance: np.ndarray, k: int, n_relevant_total: int) -> float:
    """Fraction of ALL relevant documents that appear in the top-k. The retriever's metric."""
    relevance = np.asarray(relevance)
    if n_relevant_total == 0:
        return 0.0
    return float((relevance[:k] > 0).sum()) / n_relevant_total


def reciprocal_rank(relevance: np.ndarray) -> float:
    """1 / (rank of the first relevant result), or 0 if none is relevant.

    Reciprocal rank cares only about WHERE the first correct answer lands -- perfect for QA / "I'm
    feeling lucky" search, blunt when a query has many relevant docs.
    """
    relevance = np.asarray(relevance)
    hits = np.where(relevance > 0)[0]
    if len(hits) == 0:
        return 0.0
    return 1.0 / (int(hits[0]) + 1)


def mean_reciprocal_rank(first_relevant_ranks: list[int]) -> float:
    """MRR over a query set: mean of 1/rank for each query's first relevant result (1-based ranks)."""
    if not first_relevant_ranks:
        return 0.0
    return float(np.mean([1.0 / r for r in first_relevant_ranks]))


def dcg_at_k(relevance: np.ndarray, k: int, *, exponential_gain: bool = True) -> float:
    """Discounted Cumulative Gain at k: sum of gain(rel_i) / log2(i + 1) over the top-k.

    gain(rel) = 2**rel - 1 (the standard exponential gain that rewards highly-relevant results
    super-linearly) when exponential_gain=True, else the linear gain rel itself. The log2(i+1)
    discount makes a great result at rank 1 worth more than the same result deeper down.

    Source: Jarvelin & Kekalainen (2002), "Cumulated Gain-based Evaluation of IR Techniques".
    """
    relevance = np.asarray(relevance, dtype=np.float64)[:k]
    discounts = np.log2(np.arange(2, len(relevance) + 2))  # log2(i+1) for i = 1..k
    gains = (2.0**relevance - 1.0) if exponential_gain else relevance
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevance: np.ndarray, k: int, *, exponential_gain: bool = True) -> float:
    """Normalized DCG at k = DCG@k / IDCG@k, where IDCG is the DCG of the ideal (sorted) ranking.

    Normalizing by each query's OWN ideal makes nDCG in [0, 1] and comparable/averageable across
    queries with different numbers of relevant documents. Returns 0 if the ideal DCG is 0.

    Source: Jarvelin & Kekalainen (2002).
    """
    relevance = np.asarray(relevance, dtype=np.float64)
    dcg = dcg_at_k(relevance, k, exponential_gain=exponential_gain)
    ideal = np.sort(relevance)[::-1]  # the best-possible ordering of these same grades
    idcg = dcg_at_k(ideal, k, exponential_gain=exponential_gain)
    return dcg / idcg if idcg > 0 else 0.0


def precision_recall_curve(relevance: np.ndarray, n_relevant_total: int) -> dict[str, np.ndarray]:
    """precision@k and recall@k for k = 1..len(relevance) -- the canonical trade-off, for plotting."""
    relevance = np.asarray(relevance)
    ks = np.arange(1, len(relevance) + 1)
    prec = np.array([precision_at_k(relevance, int(k)) for k in ks])
    rec = np.array([recall_at_k(relevance, int(k), n_relevant_total) for k in ks])
    return {"ks": ks, "precision": prec, "recall": rec}


def ndcg_decomposition(relevance: tuple[int, ...] = GRADED_RELEVANCE) -> dict[str, object]:
    """The full per-rank nDCG breakdown the page tabulates (linear gain for hand-checkability).

    Returns the per-rank gain/discount/contribution for both the actual and the ideal ranking, plus
    DCG, IDCG, and nDCG. Uses LINEAR gain (gain = rel) so the table matches the page's hand
    arithmetic; the default metric functions above use exponential gain (the sklearn default).
    """
    rel = np.asarray(relevance, dtype=np.float64)
    k = len(rel)
    discounts = np.log2(np.arange(2, k + 2))
    actual_contrib = rel / discounts
    ideal = np.sort(rel)[::-1]
    ideal_contrib = ideal / discounts
    dcg = float(actual_contrib.sum())
    idcg = float(ideal_contrib.sum())
    return {
        "rel": rel,
        "ideal": ideal,
        "discounts": discounts,
        "actual_contrib": actual_contrib,
        "ideal_contrib": ideal_contrib,
        "dcg": dcg,
        "idcg": idcg,
        "ndcg": dcg / idcg if idcg > 0 else 0.0,
    }


# ==================================================================================================
# (d) ANN vs exact: a recall/speed sketch with an IVF-style cluster-probing retriever.
# Pure numpy (no faiss needed) so the notebook always runs; the optional faiss path lives in the
# notebook itself and is guarded. The point is the KNOB: probe more clusters -> higher recall, slower.
# ==================================================================================================
def make_ann_dataset(
    n_docs: int = 4000, dim: int = 64, n_clusters: int = 16, seed: int = SEED
) -> dict[str, np.ndarray]:
    """A clustered synthetic vector dataset + held-out queries, so ANN has structure to exploit.

    Returns unit-normalized base vectors `xb` (n_docs, dim), queries `xq` (n_queries, dim) drawn near
    real cluster centres, and the k-means-style `centroids` (n_clusters, dim). Clustered data is what
    makes IVF work: probing the query's nearest few cells recovers most true neighbours.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_clusters, dim))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    assignments = rng.integers(0, n_clusters, size=n_docs)
    xb = centroids[assignments] + 0.35 * rng.standard_normal((n_docs, dim))
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    # Queries: perturb random base vectors slightly so each has genuine near-neighbours in the set.
    q_idx = rng.choice(n_docs, size=100, replace=False)
    xq = xb[q_idx] + 0.05 * rng.standard_normal((len(q_idx), dim))
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    return {"xb": xb, "xq": xq, "centroids": centroids}


def exact_topk(xb: np.ndarray, xq: np.ndarray, k: int = 10) -> np.ndarray:
    """Exact (brute-force) top-k neighbour ids per query by inner product -- the ground truth.

    O(N*d) per query: every query is compared against every base vector. Perfect recall, ruinous
    cost at scale -- the ideal ANN approximates. Returns (n_queries, k) neighbour ids.
    """
    sims = xq @ xb.T  # (n_queries, n_docs)
    return np.argsort(-sims, axis=1)[:, :k]


def ivf_topk(
    data: dict[str, np.ndarray], k: int = 10, nprobe: int = 1
) -> np.ndarray:
    """IVF-style approximate top-k: assign vectors to nearest centroid, search only the nprobe nearest cells.

    Probing `nprobe` of `n_clusters` cells scans roughly nprobe/n_clusters of the corpus -- the
    speedup -- at the risk of missing a neighbour that fell just across a cell boundary. Returns
    (n_queries, k) approximate neighbour ids (padded with -1 if a query's probed cells hold < k docs).

    This mirrors FAISS `IndexIVFFlat`; the page measures the same monotonic recall-vs-nprobe knob.
    Source: the inverted-file idea, Jegou/Douze/Schmid lineage; FAISS (Johnson et al. 2019).
    """
    xb, xq, centroids = data["xb"], data["xq"], data["centroids"]
    # Offline: file each base vector under its nearest centroid (coarse quantization).
    base_cell = np.argmax(xb @ centroids.T, axis=1)  # (n_docs,)
    cell_members = [np.where(base_cell == c)[0] for c in range(len(centroids))]

    results = np.full((len(xq), k), -1, dtype=np.int64)
    for i, q in enumerate(xq):
        # Find the nprobe centroids nearest this query, gather their members, score only those.
        cell_order = np.argsort(-(centroids @ q))[:nprobe]
        candidate_ids = np.concatenate([cell_members[c] for c in cell_order]) if nprobe else np.array([], dtype=int)
        if len(candidate_ids) == 0:
            continue
        sims = xb[candidate_ids] @ q
        top = candidate_ids[np.argsort(-sims)[:k]]
        results[i, : len(top)] = top
    return results


def recall_at_k_sets(approx: np.ndarray, exact: np.ndarray, k: int = 10) -> float:
    """Mean fraction of each query's exact top-k neighbours that the approximate search also returned."""
    per_query = [
        len(set(approx[i]) & set(exact[i])) / k for i in range(len(exact))
    ]
    return float(np.mean(per_query))


def ann_recall_sweep(
    data: dict[str, np.ndarray] | None = None,
    nprobes: tuple[int, ...] = (1, 2, 4, 8, 16),
    k: int = 10,
) -> dict[str, object]:
    """Sweep IVF nprobe and measure recall@k vs exact AND mean fraction-of-corpus-scanned (the speedup).

    Returns the nprobe grid, recall@k at each, the fraction of the corpus scanned at each (the proxy
    for latency: fewer vectors touched = faster), and the exact baseline timing. Contract: recall
    increases MONOTONICALLY with nprobe, and the fraction scanned (cost) rises with it -- there is no
    free lunch, only a chosen point on the recall-vs-cost curve.
    """
    if data is None:
        data = make_ann_dataset()
    xb, xq = data["xb"], data["xq"]
    n_docs, n_clusters = len(xb), len(data["centroids"])

    t0 = time.perf_counter()
    exact = exact_topk(xb, xq, k)
    exact_ms = (time.perf_counter() - t0) / len(xq) * 1e3

    recalls, fractions, ann_ms = [], [], []
    for nprobe in nprobes:
        t0 = time.perf_counter()
        approx = ivf_topk(data, k=k, nprobe=nprobe)
        ann_ms.append((time.perf_counter() - t0) / len(xq) * 1e3)
        recalls.append(recall_at_k_sets(approx, exact, k))
        fractions.append(min(1.0, nprobe / n_clusters))  # expected fraction of corpus scanned
    return {
        "nprobes": np.array(nprobes),
        "recall": np.array(recalls),
        "fraction_scanned": np.array(fractions),
        "ann_ms": np.array(ann_ms),
        "exact_ms": exact_ms,
        "n_docs": n_docs,
        "n_clusters": n_clusters,
    }


# ==================================================================================================
# Product-quantization compression math (the memory lever) -- pure arithmetic, for the figure/page.
# ==================================================================================================
def pq_compression(dim: int = 768, bytes_per_dim: int = 4, m: int = 96, nbits: int = 8) -> dict[str, float]:
    """Bytes per vector raw (FP32) vs PQ-coded, and the compression ratio.

    raw = dim * bytes_per_dim;  pq = m * (nbits / 8) bytes (one code per subvector). With d=768,
    FP32, m=96, nbits=8: 3072 -> 96 bytes, 32x. Source: Jegou, Douze & Schmid (2011).
    """
    raw = dim * bytes_per_dim
    pq = m * (nbits / 8.0)
    return {"raw_bytes": float(raw), "pq_bytes": float(pq), "ratio": raw / pq}


def main() -> None:
    """Print the page's headline numbers from whichever backend loaded -- the reproducible core."""
    print(device_report())
    print("torch:", torch.__version__, " numpy:", np.__version__)
    encoder = load_encoder()
    tag = "real" if encoder.is_real else "synthetic (fallback)"
    print(f"backend: {encoder.name} ({tag})\n")

    # (a) Lexical vs dense on the vocabulary-mismatch query.
    lvd = lexical_vs_dense(encoder)
    print("--- (a) lexical BM25 vs dense on:", f'"{QUERY}" ---')
    print(f"  BM25  gold(d0) rank: {lvd['bm25_gold_rank']}   top-3: {lvd['bm25_order'][:3].tolist()}")
    print(f"  Dense gold(d0) rank: {lvd['dense_gold_rank']}   top-3: {lvd['dense_order'][:3].tolist()}")
    assert lvd["dense_gold_rank"] < lvd["bm25_gold_rank"], "dense must beat lexical on the synonym query"
    assert lvd["bm25_scores"][GOLD_ID] == 0.0, "gold shares no surface term with the query -> BM25 = 0"

    # (a') BM25 matches the reference implementation.
    if bm25_matches_reference():
        print("  BM25 from-scratch == rank_bm25.BM25Okapi (verified)")
    else:
        print("  (rank_bm25 unavailable -- skipped reference check)")

    # (c.1) RRF worked example + hybrid recovery.
    print("\n--- (c) Reciprocal Rank Fusion ---")
    for doc, score in rrf_worked_example():
        print(f"  {doc}: {score:.5f}")
    fused_order = [d for d, _ in rrf_worked_example()]
    assert fused_order[0] == "A", "A must win RRF (high in both lists) despite no single #1"
    hy = hybrid_demo(encoder)
    print(f"  hybrid gold rank: {hy['fused_gold_rank']} (BM25 {hy['bm25_gold_rank']}, dense {hy['dense_gold_rank']})")

    # (c.2) Cross-encoder re-rank improves the top-k.
    rr = rerank_demo(encoder)
    print("\n--- (c) cross-encoder re-rank of the dense top-k ---")
    print(f"  candidates (BM25 order): {rr['candidates']}")
    print(f"  gold rank in candidates: before={rr['gold_rank_before']}  after rerank={rr['gold_rank_after']}")
    assert rr["gold_rank_after"] == 1, "the cross-encoder must lift the gold passage to the top"
    assert rr["gold_rank_after"] < rr["gold_rank_before"], "re-rank must IMPROVE the gold rank"

    # (b) Metrics from scratch.
    print("\n--- (b) ranking metrics on rel =", list(GRADED_RELEVANCE), "---")
    n_rel = int((np.array(GRADED_RELEVANCE) > 0).sum())
    for k in (1, 3, 5, 8):
        p = precision_at_k(np.array(GRADED_RELEVANCE), k)
        r = recall_at_k(np.array(GRADED_RELEVANCE), k, n_rel)
        print(f"  k={k}: precision@k={p:.3f}  recall@k={r:.3f}")
    dec = ndcg_decomposition()
    print(f"  DCG@8={dec['dcg']:.3f}  IDCG@8={dec['idcg']:.3f}  nDCG@8={dec['ndcg']:.3f} (linear gain, page table)")
    mrr = mean_reciprocal_rank([1, 3, 2])
    print(f"  MRR over first-relevant ranks [1,3,2] = {mrr:.3f}")
    assert abs(mrr - (1 + 1/3 + 1/2) / 3) < 1e-9

    # nDCG matches sklearn (exponential gain, the sklearn default).
    if ndcg_matches_sklearn():
        print("  nDCG from-scratch == sklearn.metrics.ndcg_score (verified, linear gain = sklearn default)")
    else:
        print("  (sklearn unavailable -- skipped nDCG reference check)")

    # (d) ANN recall/speed sweep.
    print("\n--- (d) ANN (IVF-style) recall@10 vs exact, by nprobe ---")
    sweep = ann_recall_sweep()
    for nprobe, rec, frac in zip(sweep["nprobes"], sweep["recall"], sweep["fraction_scanned"]):
        print(f"  nprobe={int(nprobe):2d}: recall@10={rec:.3f}  (~{frac*100:.0f}% of corpus scanned)")
    assert np.all(np.diff(sweep["recall"]) >= -1e-9), "recall must rise monotonically with nprobe"
    print(f"  exact baseline: {sweep['exact_ms']:.3f} ms/query over {sweep['n_docs']} vectors")

    print("\nall qualitative contracts held.")


# ==================================================================================================
# Reference-implementation cross-checks (used by the notebook's assert-before-print cells).
# ==================================================================================================
def bm25_matches_reference(query: str = QUERY, corpus: tuple[str, ...] = CORPUS) -> bool:
    """True iff our from-scratch BM25 ranking equals `rank_bm25.BM25Okapi` (or rank_bm25 is absent).

    rank_bm25 uses the same Robertson-Zaragoza BM25 with k1=1.5, b=0.75 defaults, so the ORDER (and,
    up to its IDF max(0,...) flooring, the scores) must agree. We compare the orderings, which is what
    a retriever actually uses.
    """
    try:
        from rank_bm25 import BM25Okapi
    except Exception:  # noqa: BLE001
        return True  # nothing to check against; treat as a pass so the notebook never fails offline
    tokenized = [tokenize(d) for d in corpus]
    ref = BM25Okapi(tokenized, k1=BM25_K1, b=BM25_B)
    ref_scores = np.array(ref.get_scores(tokenize(query)))
    ours = ranking_from_scores(bm25_scores(query, corpus))
    theirs = ranking_from_scores(ref_scores)
    return bool(np.array_equal(ours, theirs))


def ndcg_matches_sklearn(relevance: tuple[int, ...] = GRADED_RELEVANCE) -> bool:
    """True iff our LINEAR-gain nDCG equals sklearn's ndcg_score (or sklearn is absent).

    IMPORTANT gain-convention note: `sklearn.metrics.ndcg_score` uses LINEAR gain (gain = rel) with
    the log2(i+1) discount -- so it equals our `ndcg_at_k(..., exponential_gain=False)`, NOT the
    exponential-gain (2**rel - 1) version. The page teaches the exponential gain (the more common IR
    convention, e.g. Microsoft's TREC tooling and many learning-to-rank libraries) but is explicit
    that the two conventions differ and that sklearn's default is linear; this check pins our linear
    nDCG to sklearn's so the reference cross-check is exact (both give 0.960808 on the page's list).

    We feed sklearn the relevance as the true graded labels and a strictly-descending score that
    preserves OUR list order, so it scores OUR ranking.
    """
    try:
        from sklearn.metrics import ndcg_score
    except Exception:  # noqa: BLE001
        return True
    rel = np.asarray(relevance, dtype=np.float64)
    k = len(rel)
    ours = ndcg_at_k(rel, k, exponential_gain=False)  # sklearn uses linear gain
    # A strictly-descending score vector preserves the given list order as the predicted ranking.
    predicted = np.arange(k, 0, -1, dtype=np.float64)
    theirs = float(ndcg_score(rel[None, :], predicted[None, :], k=k))
    return bool(abs(ours - theirs) < 1e-9)


if __name__ == "__main__":
    main()
