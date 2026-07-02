"""Real ANN indexes over a real corpus with FAISS — the production way to search vectors.

This is the load-bearing module for the chapter. It is **not** a toy: it loads real
sentence-transformer embeddings of ~40k real Wikipedia passages (produced offline by
`embed_corpus.py`) and builds **real FAISS indexes** — `IndexFlatIP` (exact baseline),
`IndexIVFFlat` (real inverted-file / Voronoi index), and `IndexHNSWFlat` (real Hierarchical
Navigable Small World graph) — then MEASURES their approximation quality (recall@k vs exact)
and their real query latency as you turn the recall/speed knob (`nprobe` for IVF, `efSearch`
for HNSW). Optional `IndexIVFPQ` shows real product-quantization memory compression.

Design notes that matter in production and in this environment:

* **torch and FAISS must not co-load here.** On macOS both link `libomp`; importing them in
  one process double-initialises OpenMP and crashes (SIGABRT/SIGSEGV) even with
  `KMP_DUPLICATE_LIB_OK=TRUE`. So this module imports **only numpy + faiss** — never torch.
  Embeddings arrive as plain `.npy` files from the separate `embed_corpus.py` (torch) process.
  That is also correct architecture: embedding is a batch job; indexing/serving is its own job.
* **Cosine == inner product on normalised vectors.** `embed_corpus.py` L2-normalises every
  vector, so we use inner-product indexes (`IndexFlatIP`) and cosine ranking is a dot product.
* **Recall is measured, never assumed.** The exact `IndexFlatIP` gives ground-truth top-k; every
  ANN index is scored against it. Real recall, real latency — reported as medians (wall-clock
  timing is inherently variable; we fix what is seedable and report what is not).

Run standalone (after `python embed_corpus.py` has populated `data/`):

    python vector_indexes.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

# Measure the HONEST serving cost: one query at a time, on one core. FAISS's default multi-thread
# search parallelises a whole query *batch* across cores, which hides per-request latency and makes
# even exact search look free. A real RAG request embeds ONE query and searches it; that is what we
# time. (Pin to 1 thread so Flat vs IVF vs HNSW are compared on equal, single-core footing.)
faiss.omp_set_num_threads(1)

# ---- Paths + hyperparameters (hoisted; no magic numbers inline) ---------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
TOP_K = 10  # retrieve the 10 nearest neighbours; recall@10 is the headline metric
IVF_NLIST = 256  # IVF Voronoi cells (~ a few × sqrt(N) for N~40k; FAISS trains k-means to this)
IVF_NPROBE_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128, 256)  # cells probed per query — the IVF knob
HNSW_M = 32  # HNSW links per node (graph degree) — higher = better recall, more memory
HNSW_EF_CONSTRUCTION = 200  # build-time candidate list — higher = better graph, slower build
HNSW_EFSEARCH_SWEEP = (8, 16, 32, 64, 128, 256, 512)  # query-time candidate list — the HNSW knob
PQ_M = 48  # PQ subquantizers (must divide the embedding dim; 384 / 48 = 8 dims per subvector)
PQ_NBITS = 8  # bits per PQ code (2**8 = 256 centroids per subquantizer)
LATENCY_REPEATS = 3  # repeat each timed query batch and take the median (timing is noisy)
FLOAT_BYTES = 4  # a float32 component is 4 bytes


# ============================ loading the real corpus ============================================
@dataclass
class Corpus:
    """A real embedded corpus: unit-norm passage vectors, query vectors, text, and metadata."""

    embeddings: np.ndarray  # (N, dim) float32, L2-normalised
    queries: np.ndarray  # (Q, dim) float32, L2-normalised
    query_idx: np.ndarray  # (Q,) index of each query's source passage in `embeddings`
    passages: list[dict[str, str]]  # N passage dicts: {id, title, text}
    meta: dict[str, object]  # dataset / model / sizes recorded by embed_corpus.py

    @property
    def n(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


def load_corpus(data_dir: Path = DATA_DIR) -> Corpus:
    """Load the real embeddings + passages produced by `embed_corpus.py`.

    Raises a clear error (not a fake fallback) if the cache is missing — the honest failure
    mode is "run embed_corpus.py first", never "silently substitute toy vectors".
    """
    emb_path = data_dir / "corpus_emb.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} not found. Run `python embed_corpus.py` first to build the real "
            "corpus (it embeds ~40k Wikipedia passages and writes this cache)."
        )
    embeddings = np.load(emb_path).astype(np.float32)
    queries = np.load(data_dir / "query_emb.npy").astype(np.float32)
    query_idx = np.load(data_dir / "query_idx.npy").astype(np.int64)
    passages = [json.loads(line) for line in (data_dir / "passages.jsonl").read_text("utf-8").splitlines()]
    meta = json.loads((data_dir / "meta.json").read_text("utf-8"))
    return Corpus(embeddings, queries, query_idx, passages, meta)


# ============================ exact baseline (FlatIP) ============================================
def build_flat(embeddings: np.ndarray) -> faiss.Index:
    """Exact inner-product index — the ground-truth baseline (cosine on unit vectors).

    `IndexFlatIP` scans every vector: O(N*d) per query. It is the *right* choice below ~10k
    vectors and, at any scale, the reference whose top-k defines recall for every ANN index.
    """
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def exact_topk(flat: faiss.Index, queries: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Ground-truth top-k neighbour ids for each query, from the exact Flat index."""
    _scores, ids = flat.search(queries, k)
    return ids


# ============================ IVF (inverted file / Voronoi cells) ================================
def build_ivf(embeddings: np.ndarray, nlist: int = IVF_NLIST) -> faiss.IndexIVFFlat:
    """Real IVF index: FAISS trains k-means into `nlist` Voronoi cells, then adds all vectors.

    Query time probes only the `nprobe` nearest cells (set on the returned index) instead of
    scanning all N — the core ANN speedup. Inner-product metric to match our unit-norm cosine.
    """
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])  # the coarse quantizer (finds nearest cells)
    index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)  # k-means over the corpus -> the cell centroids
    index.add(embeddings)  # assign every vector to its nearest cell (build the inverted lists)
    return index


# ============================ HNSW (navigable small-world graph) =================================
def build_hnsw(
    embeddings: np.ndarray, m: int = HNSW_M, ef_construction: int = HNSW_EF_CONSTRUCTION
) -> faiss.IndexHNSWFlat:
    """Real HNSW index: FAISS wires a multi-layer navigable graph (M links/node, efConstruction).

    Query time does greedy descent through the layers, touching ~O(log N) nodes; `efSearch`
    (set per query) is the recall/speed knob. Inner-product metric for cosine on unit vectors.
    """
    index = faiss.IndexHNSWFlat(embeddings.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)  # incremental graph construction (no separate train step)
    return index


# ============================ recall + latency measurement ======================================
def recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray, k: int = TOP_K) -> float:
    """Mean recall@k over a batch: fraction of each query's true top-k that the ANN index found.

    `retrieved` and `ground_truth` are (Q, k) id matrices. This is the *real* approximation
    quality of the index — 1.0 means it matched exact search; lower means it missed neighbours.
    """
    hits, total = 0, 0
    for got, truth in zip(retrieved, ground_truth):
        truth_set = set(truth[:k].tolist())
        hits += len(truth_set & set(got[:k].tolist()))
        total += len(truth_set)
    return hits / total if total else 0.0


LATENCY_QUERY_SAMPLE = 100  # how many distinct queries to time one-at-a-time (kept modest for speed)


def _median_query_latency_ms(index: faiss.Index, queries: np.ndarray, k: int, repeats: int) -> float:
    """Median SINGLE-query latency (ms) — the honest per-request serving cost, one query, one core.

    We search queries **one at a time** (not as a batch) so the number is what a real RAG request
    pays: embed one query, search it, get top-k. Timing is noisy (OS scheduling, caches, thermal),
    so we time many individual queries across `repeats` passes and take the median — robust to
    outliers. (Batch search would parallelise across queries and hide this cost; see the module
    header note on why we pin to one thread.)
    """
    sample = queries[: min(LATENCY_QUERY_SAMPLE, len(queries))]
    times = []
    for _ in range(repeats):
        for i in range(len(sample)):
            t0 = time.perf_counter()
            index.search(sample[i : i + 1], k)  # ONE query — the serving reality
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


@dataclass
class SweepPoint:
    """One (knob-value → recall, latency) measurement on a real index."""

    knob: int  # nprobe (IVF) or efSearch (HNSW)
    recall: float  # mean recall@k vs exact
    latency_ms: float  # median per-query latency


@dataclass
class IndexReport:
    """Everything measured for one index family: build cost, and the recall/latency sweep."""

    name: str
    build_seconds: float
    sweep: list[SweepPoint] = field(default_factory=list)


def sweep_ivf(
    index: faiss.IndexIVFFlat,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    nprobe_values=IVF_NPROBE_SWEEP,
    k: int = TOP_K,
    repeats: int = LATENCY_REPEATS,
) -> list[SweepPoint]:
    """Sweep IVF `nprobe`: for each, measure real recall@k vs exact and real median latency."""
    points: list[SweepPoint] = []
    for nprobe in nprobe_values:
        index.nprobe = min(nprobe, index.nlist)  # can't probe more cells than exist
        _scores, ids = index.search(queries, k)
        recall = recall_at_k(ids, ground_truth, k)
        latency = _median_query_latency_ms(index, queries, k, repeats)
        points.append(SweepPoint(knob=int(index.nprobe), recall=recall, latency_ms=latency))
    return points


def sweep_hnsw(
    index: faiss.IndexHNSWFlat,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    efsearch_values=HNSW_EFSEARCH_SWEEP,
    k: int = TOP_K,
    repeats: int = LATENCY_REPEATS,
) -> list[SweepPoint]:
    """Sweep HNSW `efSearch`: for each, measure real recall@k vs exact and real median latency."""
    points: list[SweepPoint] = []
    for ef in efsearch_values:
        index.hnsw.efSearch = ef
        _scores, ids = index.search(queries, k)
        recall = recall_at_k(ids, ground_truth, k)
        latency = _median_query_latency_ms(index, queries, k, repeats)
        points.append(SweepPoint(knob=int(ef), recall=recall, latency_ms=latency))
    return points


# ============================ product quantization (memory) =====================================
def build_ivfpq(
    embeddings: np.ndarray, nlist: int = IVF_NLIST, m: int = PQ_M, nbits: int = PQ_NBITS
) -> faiss.IndexIVFPQ:
    """Real IVF+PQ index: IVF routing plus product-quantized vectors (the billion-scale workhorse)."""
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIVFPQ(quantizer, embeddings.shape[1], nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    return index


def pq_memory_bytes(dim: int, m: int = PQ_M, nbits: int = PQ_NBITS) -> tuple[int, int, float]:
    """Bytes per vector: raw float32 vs PQ code, and the compression ratio.

    Raw: dim * 4 bytes. PQ: m codes of `nbits` bits each -> m*nbits/8 bytes. Returns
    (raw_bytes, pq_bytes, ratio).
    """
    raw = dim * FLOAT_BYTES
    pq = (m * nbits) // 8
    return raw, pq, raw / pq


# ============================ latency of exact search (why ANN is needed) ========================
def exact_latency_ms(flat: faiss.Index, queries: np.ndarray, k: int = TOP_K, repeats: int = LATENCY_REPEATS) -> float:
    """Median per-query latency of exact Flat search — the cost ANN exists to beat."""
    return _median_query_latency_ms(flat, queries, k, repeats)


def main() -> None:
    print("faiss:", faiss.__version__, "| numpy:", np.__version__)
    corpus = load_corpus()
    print(
        f"corpus: {corpus.n:,} real passages x {corpus.dim}-dim "
        f"({corpus.meta['embed_model']} on {corpus.meta['dataset']}) | "
        f"{len(corpus.queries):,} queries\n"
    )

    # ---- exact baseline: ground truth + its latency (the wall) ----
    t0 = time.perf_counter()
    flat = build_flat(corpus.embeddings)
    flat_build = time.perf_counter() - t0
    ground_truth = exact_topk(flat, corpus.queries, TOP_K)
    flat_ms = exact_latency_ms(flat, corpus.queries)
    print(f"FLAT (exact): build {flat_build:.2f}s | {flat_ms:.3f} ms/query over {corpus.n:,} vectors")

    # ---- IVF sweep ----
    t0 = time.perf_counter()
    ivf = build_ivf(corpus.embeddings)
    ivf_build = time.perf_counter() - t0
    ivf_points = sweep_ivf(ivf, corpus.queries, ground_truth)
    print(f"\nIVF (nlist={IVF_NLIST}): build {ivf_build:.2f}s")
    print(f"  {'nprobe':>6} | {'recall@10':>9} | {'ms/query':>9} | {'speedup vs flat':>15}")
    print("  " + "-" * 52)
    for p in ivf_points:
        print(f"  {p.knob:>6} | {p.recall:>9.3f} | {p.latency_ms:>9.3f} | {flat_ms / p.latency_ms:>14.1f}x")

    # ---- HNSW sweep ----
    t0 = time.perf_counter()
    hnsw = build_hnsw(corpus.embeddings)
    hnsw_build = time.perf_counter() - t0
    hnsw_points = sweep_hnsw(hnsw, corpus.queries, ground_truth)
    print(f"\nHNSW (M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}): build {hnsw_build:.2f}s")
    print(f"  {'efSearch':>8} | {'recall@10':>9} | {'ms/query':>9} | {'speedup vs flat':>15}")
    print("  " + "-" * 54)
    for p in hnsw_points:
        print(f"  {p.knob:>8} | {p.recall:>9.3f} | {p.latency_ms:>9.3f} | {flat_ms / p.latency_ms:>14.1f}x")

    # ---- correctness: recall must be monotone-ish in the knob and reach exact at the top ----
    ivf_recalls = [p.recall for p in ivf_points]
    assert ivf_recalls[0] < ivf_recalls[-1], "IVF recall must rise as nprobe grows"
    assert ivf_recalls[-1] >= 0.99, "probing all cells must recover ~exact recall"
    hnsw_recalls = [p.recall for p in hnsw_points]
    assert hnsw_recalls[-1] > hnsw_recalls[0], "HNSW recall must rise as efSearch grows"

    # ---- PQ memory ----
    raw, pq, ratio = pq_memory_bytes(corpus.dim)
    print(
        f"\nProduct Quantization memory: raw {raw} bytes/vector "
        f"({corpus.dim} dims x {FLOAT_BYTES}B) -> PQ {pq} bytes "
        f"(m={PQ_M}, {PQ_NBITS} bits) = {ratio:.0f}x smaller"
    )
    print(
        f"  full corpus: raw {corpus.n * raw / 1e6:.1f} MB -> PQ {corpus.n * pq / 1e6:.1f} MB "
        f"(the RAM saving that makes billion-scale search fit)"
    )


if __name__ == "__main__":
    main()
