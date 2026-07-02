"""Real two-stage retrieve-then-rerank over a real labelled IR benchmark (BeIR/scifact).

First-stage retrieval (a bi-encoder, BM25, or hybrid) ranks passages by INDEPENDENT encodings: it
embeds the query once and every passage once, then compares vectors. That is fast and precomputable
-- but coarse, because the query and the passage never "see" each other. The truly-best passage
often lands in the top-k yet not at the top, because a bi-encoder cannot model fine-grained
query<->document token interaction (it scores a paraphrase and the actual answer as almost equally
relevant).

A CROSS-ENCODER fixes this: it feeds the query and ONE passage through the transformer TOGETHER
([CLS] query [SEP] passage), so every query token can attend to every passage token, and emits a
single scalar relevance score. Far more accurate -- but it runs the model once per (query, passage)
pair at query time, so it cannot precompute passage vectors and is far too slow for first-stage
retrieval over a big corpus. The standard architecture uses BOTH: retrieve top-K with the cheap
bi-encoder, then re-rank those few with the expensive cross-encoder. Cheap RECALL, then expensive
PRECISION.

This module is NOT a toy. It builds the real two-stage pipeline over **BeIR/scifact** -- a real
scientific-claim retrieval benchmark: 5,183 real abstracts, 300 real test queries, and real human
relevance judgments (qrels) so every metric (nDCG@10, MRR@10, Recall@k, Precision@k) is grounded in
truth, not invented. Stage 1 is a real bi-encoder (`all-MiniLM-L6-v2`); stage 2 is a real
cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). We measure the REAL quality lift of adding
the reranker, the REAL quality-vs-depth curve as the rerank pool K grows, and the REAL per-query
latency of each stage.

No faiss: first-stage top-K is exact cosine via numpy `argpartition` (fine at this corpus size and
it dodges the torch/faiss OpenMP conflict entirely). Corpus embeddings are cached to `data/` so the
notebook and figures reuse one embedding pass.

Device-agnostic: models run on cuda -> mps -> cpu (detected + printed in the banner). Inference is
deterministic (dropout off, no sampling), so ranks and metrics are reproducible run to run;
wall-clock latency varies and is reported as a median.

Run:
    python reranking.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---- Configuration (hoisted; no magic numbers inline) ------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
BEIR_DATASET = "scifact"  # a real BeIR benchmark: scientific claim -> evidence abstract
BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # chapter 3's bi-encoder (first stage)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # the cross-encoder re-ranker
RETRIEVE_K = 100  # first-stage pool size the cross-encoder re-ranks (the funnel's wide end)
METRIC_K = 10  # the cutoff for the headline metrics (nDCG@10, MRR@10)
K_SWEEP = (1, 5, 10, 20, 50, 100)  # rerank-depth sweep: quality/latency vs how deep we re-rank
EMBED_BATCH = 128  # bi-encoder encode batch size
CE_BATCH = 64  # cross-encoder predict batch size
LATENCY_QUERIES = 30  # how many queries to time one-at-a-time for the latency medians


# ================================================================================================
# The real corpus + queries + relevance labels (BeIR/scifact)
# ================================================================================================
@dataclass
class ScifactData:
    """A real labelled IR benchmark: passages, queries, and human relevance judgments (gold sets)."""

    doc_texts: list[str]  # N passage strings ("title. text")
    doc_ids: list[str]  # the corpus ids, aligned with doc_texts
    query_texts: list[str]  # Q query strings (test split, those that have gold)
    query_ids: list[str]  # aligned query ids
    gold: list[set[int]]  # per query: the set of RELEVANT doc indices (from qrels); usually 1-2

    @property
    def n_docs(self) -> int:
        return len(self.doc_texts)

    @property
    def n_queries(self) -> int:
        return len(self.query_texts)


def load_scifact(dataset: str = BEIR_DATASET) -> ScifactData:
    """Load BeIR corpus + queries + test qrels and align them into index-based gold sets.

    qrels give (query-id, corpus-id, score); we keep score>0 as relevant and map corpus-ids to
    positions in `doc_texts`, so `gold[i]` is the set of relevant doc INDICES for query i -- exactly
    what the metrics below consume. Only test queries that have at least one relevant doc are kept.
    """
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{dataset}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{dataset}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{dataset}-qrels", split="test")

    doc_ids = [str(r["_id"]) for r in corpus]
    doc_texts = [f"{r['title']}. {r['text']}".strip() for r in corpus]
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    q_id_to_text = {str(r["_id"]): r["text"] for r in queries}

    gold_by_qid: dict[str, set[int]] = {}
    for r in qrels:
        qid, cid = str(r["query-id"]), str(r["corpus-id"])
        if r["score"] > 0 and cid in doc_id_to_idx:
            gold_by_qid.setdefault(qid, set()).add(doc_id_to_idx[cid])

    kept = [qid for qid in gold_by_qid if qid in q_id_to_text]
    kept.sort(key=int)  # deterministic query order
    return ScifactData(
        doc_texts=doc_texts,
        doc_ids=doc_ids,
        query_texts=[q_id_to_text[q] for q in kept],
        query_ids=kept,
        gold=[gold_by_qid[q] for q in kept],
    )


# ================================================================================================
# Device + model loading (device-agnostic, real models only)
# ================================================================================================
def detect_device() -> str:
    """Best available torch device: cuda -> mps (Apple) -> cpu. The models run wherever this points."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ================================================================================================
# Stage 1 -- the bi-encoder retriever (independent encodings; fast, precomputable, cacheable)
# ================================================================================================
class BiEncoderRetriever:
    """Dense first-stage retrieval: encode query and passages INDEPENDENTLY, rank by cosine.

    The passage vectors are precomputed once (and cached to disk); at query time we embed only the
    query and take dot products -- O(N) cheap comparisons, no transformer forward pass per passage.
    That independence is exactly what makes it fast and exactly why it is coarse: the query never
    attends to the passage, so it cannot weigh which passage truly *answers* the query versus merely
    shares its topic. Top-K is exact cosine via numpy argpartition (no faiss needed at this scale).
    """

    def __init__(self, data: ScifactData, model_name: str = BI_ENCODER_MODEL, device: str | None = None) -> None:
        self.data = data
        self.model_name = model_name
        self.device = device or detect_device()
        self.doc_vectors = self._embed_corpus()  # (N, d) unit-norm rows -- precomputed once, cached
        self.query_vectors = self._embed_queries()  # (Q, d) unit-norm

    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name, device=self.device)

    def _embed_corpus(self) -> np.ndarray:
        cache = DATA_DIR / f"{BEIR_DATASET}_doc_emb.npy"
        if cache.exists():
            return np.load(cache).astype(np.float32)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        emb = self._model().encode(
            self.data.doc_texts, batch_size=EMBED_BATCH, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)
        np.save(cache, emb)
        return emb

    def _embed_queries(self) -> np.ndarray:
        cache = DATA_DIR / f"{BEIR_DATASET}_query_emb.npy"
        if cache.exists():
            return np.load(cache).astype(np.float32)
        emb = self._model().encode(
            self.data.query_texts, batch_size=EMBED_BATCH, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)
        np.save(cache, emb)
        return emb

    def retrieve(self, query_index: int, k: int = RETRIEVE_K) -> np.ndarray:
        """Top-k doc indices for a query, best-first (unit-norm rows => dot product == cosine).

        `argpartition` finds the top-k in O(N) without fully sorting all N, then we sort just those
        k -- the cheap exact first stage whose output the cross-encoder re-ranks.
        """
        sims = self.doc_vectors @ self.query_vectors[query_index]  # (N,) cosine scores
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]  # k best (unordered)
        return top[np.argsort(-sims[top])]  # sorted best-first


# ================================================================================================
# Stage 2 -- the cross-encoder re-ranker (joint encoding; accurate, query-time only)
# ================================================================================================
class CrossEncoderReranker:
    """Precision second stage: score each (query, passage) pair by JOINT encoding, then re-sort.

    A cross-encoder concatenates the query and one passage ([CLS] query [SEP] passage) and runs the
    transformer over the pair, so query tokens attend to passage tokens and back. The output head
    emits a single scalar relevance logit. Because the score depends on the query<->passage
    interaction, it CANNOT be precomputed (a passage's score is different for every query) and costs
    one transformer forward pass per pair -- which is why you only ever run it on the top-K the
    bi-encoder already retrieved, never the whole corpus. Uses the real ms-marco-MiniLM cross-encoder.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = device or detect_device()
        self.model = CrossEncoder(model_name, device=self.device)

    def scores(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance logit for each (query, passage) pair -- one joint forward pass per passage."""
        pairs = [(query, p) for p in passages]
        return np.asarray(self.model.predict(pairs, batch_size=CE_BATCH, show_progress_bar=False))

    def rerank(self, query: str, candidate_indices: np.ndarray, doc_texts: list[str]) -> np.ndarray:
        """Re-score the candidate pool with joint encoding; return it re-ordered best-first.

        Only REORDERS the candidates it is given -- it cannot add a passage the first stage failed to
        retrieve (the recall-ceiling limit the page discusses).
        """
        passages = [doc_texts[i] for i in candidate_indices]
        order = np.argsort(-self.scores(query, passages))
        return np.asarray(candidate_indices)[order]


# ================================================================================================
# Ranking-quality metrics, from scratch (support MULTIPLE relevant docs per query)
# ================================================================================================
def dcg_at_k(ranked: np.ndarray, gold: set[int], k: int) -> float:
    """Discounted Cumulative Gain at k: sum rel_i / log2(i+1) over the top-k (i is the 1-based rank).

    Binary relevance (rel=1 if the doc is in the gold set, else 0). The log discount is what makes
    ranking a relevant doc *higher* score more -- a relevant doc at rank 1 contributes 1/log2(2)=1,
    at rank 4 only 1/log2(5)=0.43.
    """
    dcg = 0.0
    for i, doc in enumerate(ranked[:k], start=1):
        if int(doc) in gold:
            dcg += 1.0 / np.log2(i + 1)
    return dcg


def ndcg_at_k(ranked: np.ndarray, gold: set[int], k: int) -> float:
    """Normalized DCG at k = DCG / IDCG, in [0,1]. IDCG = the best possible ordering (all gold on top).

    With |gold| relevant docs, IDCG = sum_{i=1..min(|gold|,k)} 1/log2(i+1). Dividing by it makes the
    metric comparable across queries with different numbers of relevant docs.
    """
    dcg = dcg_at_k(ranked, gold, k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked: np.ndarray, gold: set[int], k: int) -> float:
    """Reciprocal rank of the FIRST relevant doc within the top-k (0 if none). MRR averages this."""
    for i, doc in enumerate(ranked[:k], start=1):
        if int(doc) in gold:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked: np.ndarray, gold: set[int], k: int) -> float:
    """Fraction of the gold set present in the top-k -- the ceiling re-ranking cannot exceed."""
    if not gold:
        return 0.0
    return len(set(int(d) for d in ranked[:k]) & gold) / len(gold)


def precision_at_k(ranked: np.ndarray, gold: set[int], k: int) -> float:
    """Fraction of the top-k that is relevant -- what re-ranking directly improves."""
    if k == 0:
        return 0.0
    return len(set(int(d) for d in ranked[:k]) & gold) / k


# ================================================================================================
# Evaluation: aggregate metrics before vs after rerank, plus the K-sweep and latency
# ================================================================================================
@dataclass
class MetricRow:
    """Mean of one metric over all queries, bi-encoder vs reranked."""

    name: str
    bi: float
    reranked: float

    @property
    def delta(self) -> float:
        return self.reranked - self.bi


@dataclass
class Evaluation:
    """Everything measured over the real query set."""

    n_queries: int
    rows: list[MetricRow] = field(default_factory=list)


def evaluate(
    data: ScifactData, bi: BiEncoderRetriever, cross: CrossEncoderReranker,
    retrieve_k: int = RETRIEVE_K, metric_k: int = METRIC_K, n_queries: int | None = None,
) -> tuple[Evaluation, list[np.ndarray], list[np.ndarray]]:
    """Retrieve top-K then rerank for every query; return aggregate before/after metrics + orderings.

    Returns (Evaluation, bi_orderings, reranked_orderings) so callers (figures/notebook) can reuse the
    exact rankings without recomputing the expensive cross-encoder passes.
    """
    q_indices = range(data.n_queries if n_queries is None else min(n_queries, data.n_queries))
    bi_orders, rr_orders = [], []
    pool_metric = f"Recall@{retrieve_k}"  # the POOL ceiling: recall over the whole retrieved pool
    acc = {m: {"bi": [], "rr": []} for m in ("nDCG@10", "MRR@10", "Recall@10", "Precision@10", pool_metric)}
    for qi in q_indices:
        gold = data.gold[qi]
        pool = bi.retrieve(qi, k=retrieve_k)
        reranked = cross.rerank(data.query_texts[qi], pool, data.doc_texts)
        bi_orders.append(pool)
        rr_orders.append(reranked)
        acc["nDCG@10"]["bi"].append(ndcg_at_k(pool, gold, metric_k))
        acc["nDCG@10"]["rr"].append(ndcg_at_k(reranked, gold, metric_k))
        acc["MRR@10"]["bi"].append(mrr_at_k(pool, gold, metric_k))
        acc["MRR@10"]["rr"].append(mrr_at_k(reranked, gold, metric_k))
        # Recall@10 IMPROVES with reranking (gold from ranks 11-K promoted into the top-10);
        # Recall@K (the full pool) is the true ceiling and is INVARIANT to reranking.
        acc["Recall@10"]["bi"].append(recall_at_k(pool, gold, metric_k))
        acc["Recall@10"]["rr"].append(recall_at_k(reranked, gold, metric_k))
        acc[pool_metric]["bi"].append(recall_at_k(pool, gold, retrieve_k))
        acc[pool_metric]["rr"].append(recall_at_k(reranked, gold, retrieve_k))
        acc["Precision@10"]["bi"].append(precision_at_k(pool, gold, metric_k))
        acc["Precision@10"]["rr"].append(precision_at_k(reranked, gold, metric_k))
    ev = Evaluation(n_queries=len(bi_orders))
    for m, d in acc.items():
        ev.rows.append(MetricRow(m, float(np.mean(d["bi"])), float(np.mean(d["rr"]))))
    return ev, bi_orders, rr_orders


def cache_orderings(bi_orders: list[np.ndarray], rr_orders: list[np.ndarray]) -> None:
    """Persist the per-query bi-encoder and reranked orderings so figures reuse the CE passes once.

    Reranking all queries costs tens of thousands of cross-encoder forward passes; caching the
    resulting rankings means the figure generator (and any re-analysis) never has to repeat them.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA_DIR / f"{BEIR_DATASET}_orderings.npz",
        **{f"bi_{i}": o for i, o in enumerate(bi_orders)},
        **{f"rr_{i}": o for i, o in enumerate(rr_orders)},
        n=np.array([len(bi_orders)]),
    )


def load_orderings() -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    """Load cached orderings if present, else None (so the caller can recompute)."""
    path = DATA_DIR / f"{BEIR_DATASET}_orderings.npz"
    if not path.exists():
        return None
    z = np.load(path)
    n = int(z["n"][0])
    return [z[f"bi_{i}"] for i in range(n)], [z[f"rr_{i}"] for i in range(n)]


def hero_query(
    data: ScifactData, bi_orders: list[np.ndarray], rr_orders: list[np.ndarray], metric_k: int = METRIC_K
) -> int:
    """Pick the query whose gold is most dramatically lifted by reranking (biggest MRR gain).

    A real, honest 'hero' example for the rank-shuffle figure: not hand-picked prose, but the query
    the reranker helps most on this real benchmark -- a buried gold pulled toward the top.
    """
    best_qi, best_gain = 0, -1.0
    for qi, (bi_ord, rr_ord) in enumerate(zip(bi_orders, rr_orders)):
        gold = data.gold[qi]
        gain = mrr_at_k(rr_ord, gold, metric_k) - mrr_at_k(bi_ord, gold, metric_k)
        if gain > best_gain:
            best_qi, best_gain = qi, gain
    return best_qi


def sweep_rerank_depth(
    data: ScifactData, bi_orders: list[np.ndarray], rr_full_orders: list[np.ndarray],
    ks=K_SWEEP, metric_k: int = METRIC_K,
) -> dict[int, tuple[float, float]]:
    """nDCG@10 as a function of how deep we re-rank (top-K reranked, rest kept in bi-encoder order).

    Reranking only the top-K of the pool and leaving the tail in first-stage order is exactly what a
    real system does when it rerank-truncates for latency. Returns {K: (bi_ndcg, reranked_ndcg)}.
    """
    out: dict[int, tuple[float, float]] = {}
    for k_depth in ks:
        bi_scores, rr_scores = [], []
        for qi, (bi_ord, rr_full) in enumerate(zip(bi_orders, rr_full_orders)):
            gold = data.gold[qi]
            # rerank only the top-k_depth: take rr_full's ordering restricted to the top-k_depth pool
            head_pool = set(int(x) for x in bi_ord[:k_depth])
            reranked_head = [int(x) for x in rr_full if int(x) in head_pool]
            tail = [int(x) for x in bi_ord[k_depth:]]
            hybrid = np.array(reranked_head + tail)
            bi_scores.append(ndcg_at_k(bi_ord, gold, metric_k))
            rr_scores.append(ndcg_at_k(hybrid, gold, metric_k))
        out[k_depth] = (float(np.mean(bi_scores)), float(np.mean(rr_scores)))
    return out


def measure_latency(
    data: ScifactData, bi: BiEncoderRetriever, cross: CrossEncoderReranker,
    retrieve_k: int = RETRIEVE_K, n: int = LATENCY_QUERIES,
) -> tuple[float, float]:
    """Median per-query latency (ms): stage-1 retrieve vs stage-2 rerank of the top-K pool.

    Times each stage one query at a time -- the honest per-request serving cost. Wall-clock timing is
    noisy; we take the median over n queries.
    """
    ret_ms, rr_ms = [], []
    for qi in range(min(n, data.n_queries)):
        t0 = time.perf_counter()
        pool = bi.retrieve(qi, k=retrieve_k)
        ret_ms.append((time.perf_counter() - t0) * 1000)
        passages = [data.doc_texts[i] for i in pool]
        t0 = time.perf_counter()
        cross.scores(data.query_texts[qi], passages)
        rr_ms.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ret_ms)), float(np.median(rr_ms))


def main() -> None:
    device = detect_device()
    try:
        import sentence_transformers
        import torch

        print(f"torch {torch.__version__} | sentence-transformers {sentence_transformers.__version__} "
              f"| numpy {np.__version__} | device: {device}")
    except ImportError:
        print(f"numpy {np.__version__} | device: {device}")

    data = load_scifact()
    print(f"\nBeIR/{BEIR_DATASET}: {data.n_docs:,} passages | {data.n_queries} test queries with gold "
          f"| mean {np.mean([len(g) for g in data.gold]):.2f} relevant docs/query")
    print(f"bi-encoder: {BI_ENCODER_MODEL} | cross-encoder: {CROSS_ENCODER_MODEL}")
    print(f"retrieve_k={RETRIEVE_K} -> rerank -> metric@{METRIC_K}\n")

    bi = BiEncoderRetriever(data, device=device)
    cross = CrossEncoderReranker(device=device)

    ev, bi_orders, rr_orders = evaluate(data, bi, cross)
    cache_orderings(bi_orders, rr_orders)  # persist so figures reuse the CE passes once
    print(f"REAL quality lift over {ev.n_queries} queries (retrieve top-{RETRIEVE_K}, rerank):")
    print(f"  {'metric':<13} | {'bi-encoder':>10} | {'reranked':>9} | {'delta':>7}")
    print("  " + "-" * 48)
    for r in ev.rows:
        print(f"  {r.name:<13} | {r.bi:>10.3f} | {r.reranked:>9.3f} | {r.delta:>+7.3f}")

    # correctness before the claim: reranking should raise nDCG/MRR, and the POOL recall (the true
    # ceiling, Recall@retrieve_k) must be invariant -- reranking reorders the same K docs, so it can
    # promote gold from ranks 11-K into the top-10 (Recall@10 CAN rise) but cannot change the pool set.
    by = {r.name: r for r in ev.rows}
    assert by["nDCG@10"].reranked >= by["nDCG@10"].bi, "reranking should not lower nDCG@10"
    assert by["MRR@10"].reranked >= by["MRR@10"].bi, "reranking should not lower MRR@10"
    pool_metric = f"Recall@{RETRIEVE_K}"
    assert abs(by[pool_metric].reranked - by[pool_metric].bi) < 1e-9, "pool recall (the ceiling) must be invariant"

    sweep = sweep_rerank_depth(data, bi_orders, rr_orders)
    print(f"\nQuality vs rerank depth K (nDCG@{METRIC_K}):")
    print(f"  {'K':>4} | {'bi-encoder':>10} | {'reranked':>9}")
    print("  " + "-" * 30)
    for k_depth, (b, r) in sweep.items():
        print(f"  {k_depth:>4} | {b:>10.3f} | {r:>9.3f}")

    ret_ms, rr_ms = measure_latency(data, bi, cross)
    print(f"\nREAL per-query latency (median, {device}): retrieve {ret_ms:.2f} ms | "
          f"rerank top-{RETRIEVE_K} {rr_ms:.1f} ms  (rerank is ~{rr_ms / ret_ms:.0f}x the retrieve cost)")

    # persist the shipped numbers so figures/notebook/page share one source of truth
    summary = {
        "dataset": f"BeIR/{BEIR_DATASET}", "n_docs": data.n_docs, "n_queries": ev.n_queries,
        "bi_encoder": BI_ENCODER_MODEL, "cross_encoder": CROSS_ENCODER_MODEL,
        "retrieve_k": RETRIEVE_K, "metric_k": METRIC_K, "device": device,
        "metrics": {r.name: {"bi": r.bi, "reranked": r.reranked, "delta": r.delta} for r in ev.rows},
        "retrieve_ms": ret_ms, "rerank_ms": rr_ms,
    }
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {DATA_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
