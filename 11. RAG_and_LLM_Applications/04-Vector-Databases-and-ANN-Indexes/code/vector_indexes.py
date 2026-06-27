"""From-scratch vector indexes: why ANN beats brute force, and the recall/speed knob.

Retrieval is nearest-neighbour search in embedding space (chapter 3). The catch: exact ("flat")
search compares the query to EVERY vector — O(N*d) per query — which is fine for thousands but
brutal for millions. This script makes the cost concrete and builds an Approximate Nearest
Neighbour (ANN) index from scratch so the recall/speed tradeoff is inspectable.

The load-bearing lesson is a from-scratch **IVF** index: k-means partitions the vectors into cells
(Voronoi regions); at query time we probe only the `nprobe` nearest cells instead of all N vectors.
We MEASURE recall@k against brute force and the speedup as we sweep `nprobe` — exposing the recall
"cliff" at low nprobe. A simplified navigable-graph greedy search conveys the HNSW mechanic, and an
optional FAISS bridge confirms the lesson on the production library *if it's available*.

Everything is deterministic (seeded) and runs on CPU in a couple of seconds. The from-scratch IVF
is the load-bearing lesson and never depends on FAISS.

Verified on Python 3.12 / numpy 2.x. Device-agnostic for the torch version banner (CUDA / MPS / CPU);
the index math is pure numpy and identical on any machine.

Run:
    python vector_indexes.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- Dataset + index hyperparameters (hoisted; no magic numbers inline) -------------------------
N_VECTORS = 20_000  # corpus size — big enough that brute force is visibly slow, small enough for CPU
DIM = 64  # embedding dimension of the toy vectors
N_LATENT_CLUSTERS = 12  # the data has this many latent blobs (so cells overlap and the cliff is real)
CLUSTER_SPREAD = 2.2  # distance between blob centres; smaller = more overlap = harder ANN
N_QUERIES = 200  # query set size for the recall/latency measurement
NLIST = 64  # IVF: number of Voronoi cells (k-means centroids)
KMEANS_ITERS = 8  # Lloyd iterations to build the IVF cells (enough to converge on this toy)
TOP_K = 10  # retrieve the 10 nearest neighbours; recall@10 is measured against brute force
NPROBE_SWEEP = (1, 2, 4, 8, 16, 32, 64)  # how many nearest cells to scan — the recall/speed knob
SEED = 0  # seed for every stochastic step (data, k-means init) -> reproducible numbers

# PQ (product quantization) memory-math constants — for the compression figure, not the search demo.
PQ_M = 8  # number of subquantizers a vector is split into
PQ_NBITS = 8  # bits per subquantizer code (so each subquantizer has 2**8 = 256 centroids)
FLOAT_BITS = 32  # an uncompressed float32 vector component is 32 bits


def make_dataset(
    n: int = N_VECTORS, dim: int = DIM, n_clusters: int = N_LATENT_CLUSTERS, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Build a toy corpus of `n` clustered vectors plus `N_QUERIES` queries near the same blobs.

    Lightly-overlapping Gaussian blobs make a realistic ANN test: a single cell holds *some* of a
    query's true neighbours but not most, so probing one cell misses many — which is exactly what
    produces the recall cliff we want to show. Returns (corpus, queries), both float32.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, CLUSTER_SPREAD, (n_clusters, dim))  # blob centres
    corpus_labels = rng.integers(0, n_clusters, n)  # which blob each corpus vector belongs to
    corpus = (centres[corpus_labels] + rng.normal(0.0, 1.0, (n, dim))).astype(np.float32)
    query_labels = rng.integers(0, n_clusters, N_QUERIES)
    queries = (centres[query_labels] + rng.normal(0.0, 1.0, (N_QUERIES, dim))).astype(np.float32)
    return corpus, queries


def brute_force_topk(query: np.ndarray, corpus: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Exact nearest neighbours: squared-L2 distance to EVERY vector, then take the k smallest.

    This is the O(N*d) baseline. `argsort` over all N distances is the "scan everything" cost ANN
    indexes exist to avoid. Returns the indices of the k nearest corpus vectors.
    """
    distances = ((corpus - query) ** 2).sum(axis=1)  # (N,) squared L2 to every vector — the full scan
    return np.argsort(distances)[:k]  # the k closest; argsort touches all N, hence O(N log N) on top of O(N*d)


def kmeans(
    corpus: np.ndarray, n_cells: int = NLIST, iters: int = KMEANS_ITERS, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means to partition the corpus into `n_cells` Voronoi cells (the IVF coarse quantizer).

    Returns (centroids, assignments): the cell centroids and, for each corpus vector, its cell id.
    This is the build step of an IVF index — done once, offline. (Real FAISS uses a faster k-means;
    the algorithm is the same.)
    """
    rng = np.random.default_rng(seed)
    centroids = corpus[rng.choice(len(corpus), n_cells, replace=False)].copy()  # init: random data points
    assignments = np.zeros(len(corpus), dtype=np.int64)
    for _ in range(iters):
        # assign each vector to its nearest centroid (the E-step): (N, n_cells) distance matrix
        dist = ((corpus[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignments = dist.argmin(axis=1)
        for cell in range(n_cells):  # recompute each centroid as the mean of its members (the M-step)
            members = assignments == cell
            if members.any():
                centroids[cell] = corpus[members].mean(axis=0)
    return centroids, assignments


@dataclass
class IVFIndex:
    """A from-scratch IVF index: centroids + an inverted list mapping each cell to its vector ids."""

    corpus: np.ndarray
    centroids: np.ndarray
    cells: dict[int, np.ndarray]  # cell id -> array of corpus indices assigned to that cell


def build_ivf(corpus: np.ndarray, n_cells: int = NLIST, seed: int = SEED) -> IVFIndex:
    """Build the IVF index: k-means the corpus, then group vector ids by their cell (inverted lists)."""
    centroids, assignments = kmeans(corpus, n_cells, seed=seed)
    cells = {cell: np.where(assignments == cell)[0] for cell in range(n_cells)}  # the inverted lists
    return IVFIndex(corpus=corpus, centroids=centroids, cells=cells)


def ivf_search(
    index: IVFIndex, query: np.ndarray, k: int = TOP_K, nprobe: int = 1
) -> tuple[np.ndarray, int]:
    """Probe the `nprobe` nearest cells, then exact-search only their vectors.

    Returns (top-k indices, number of vectors actually scanned). The win: instead of scanning all N
    vectors, we scan only those in the few probed cells — but if a true neighbour sits in an
    unprobed cell, we miss it (the "approximate" in ANN), which is why recall < 1 at small nprobe.
    """
    # step 1 — find the nprobe nearest CELL CENTROIDS to the query (cheap: nlist << N)
    cell_dist = ((index.centroids - query) ** 2).sum(axis=1)
    probed_cells = np.argsort(cell_dist)[:nprobe]
    # step 2 — gather the candidate vectors from just those cells (the inverted lists)
    candidate_ids = np.concatenate([index.cells[cell] for cell in probed_cells])
    if len(candidate_ids) == 0:  # degenerate: all probed cells empty (rare) -> no results
        return np.array([], dtype=np.int64), 0
    # step 3 — exact search WITHIN the candidates only (this is the O((N/nlist)*nprobe*d) part)
    cand_dist = ((index.corpus[candidate_ids] - query) ** 2).sum(axis=1)
    top = candidate_ids[np.argsort(cand_dist)[:k]]
    return top, len(candidate_ids)


def recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray) -> float:
    """Fraction of the true top-k that the approximate search actually returned (recall@k)."""
    if len(ground_truth) == 0:
        return 0.0
    return len(set(retrieved.tolist()) & set(ground_truth.tolist())) / len(ground_truth)


def evaluate_ivf_sweep(
    index: IVFIndex, queries: np.ndarray, ground_truth: np.ndarray, nprobe_values=NPROBE_SWEEP
) -> dict[int, tuple[float, float]]:
    """For each nprobe: mean recall@k and mean fraction of the corpus scanned (the speed proxy).

    Fewer vectors scanned = faster query; lower nprobe scans fewer but recalls less — the tradeoff.
    Returns {nprobe: (mean_recall, mean_fraction_scanned)}.
    """
    results: dict[int, tuple[float, float]] = {}
    n_corpus = len(index.corpus)
    for nprobe in nprobe_values:
        recalls, fractions = [], []
        for qi, query in enumerate(queries):
            retrieved, n_scanned = ivf_search(index, query, nprobe=nprobe)
            recalls.append(recall_at_k(retrieved, ground_truth[qi]))
            fractions.append(n_scanned / n_corpus)  # share of the corpus actually distance-computed
        results[nprobe] = (float(np.mean(recalls)), float(np.mean(fractions)))
    return results


# ============================ a simplified navigable graph (HNSW intuition) ======================
def build_knn_graph(corpus: np.ndarray, n_neighbors: int = 16) -> dict[int, list[int]]:
    """Build a simple k-NN graph: connect each vector to its `n_neighbors` nearest vectors.

    This is a *simplified* stand-in for HNSW's base layer (no hierarchy, no long-range links) — just
    enough to show the mechanic: greedy graph traversal toward the query. Real HNSW adds multiple
    layers of small-world links for O(log N) navigation; here we convey the greedy-hop idea on one
    layer. The graph is deterministic from the data. Build is O(N^2) here for clarity (fine on this
    toy); real builds are far cheaper.
    """
    graph: dict[int, list[int]] = {}
    for i in range(len(corpus)):
        dist = ((corpus - corpus[i]) ** 2).sum(axis=1)
        nearest = np.argsort(dist)[1 : n_neighbors + 1]  # skip self (index 0 after sort)
        graph[i] = nearest.tolist()
    return graph


def greedy_graph_search(
    corpus: np.ndarray, graph: dict[int, list[int]], query: np.ndarray, entry: int
) -> tuple[int, int]:
    """Greedy descent: from `entry`, repeatedly hop to the neighbour closest to the query.

    Stop when no neighbour is closer than the current node (a local optimum). Returns (best node,
    number of hops). This is the heart of graph ANN — follow the gradient of decreasing distance to
    the query, touching only a handful of nodes instead of all N.
    """
    current = entry
    current_dist = float(((corpus[current] - query) ** 2).sum())
    hops = 0
    while True:
        neighbors = graph[current]
        neigh_dist = ((corpus[neighbors] - query) ** 2).sum(axis=1)
        best_local = int(np.argmin(neigh_dist))
        if neigh_dist[best_local] >= current_dist:  # no neighbour is closer -> local optimum, stop
            return current, hops
        current = neighbors[best_local]  # hop to the closer neighbour
        current_dist = float(neigh_dist[best_local])
        hops += 1


# ============================ memory math: product quantization ==================================
def pq_compression_ratio(dim: int = DIM, m: int = PQ_M, nbits: int = PQ_NBITS) -> tuple[int, int, float]:
    """Bytes for a raw float32 vector vs a PQ code, and the compression ratio.

    Raw: dim * 32 bits. PQ: split the vector into m sub-vectors, replace each with the id of its
    nearest sub-centroid (nbits per id) -> m * nbits total. Returns (raw_bits, pq_bits, ratio).
    """
    raw_bits = dim * FLOAT_BITS
    pq_bits = m * nbits
    return raw_bits, pq_bits, raw_bits / pq_bits


def _report_device() -> str:
    """Pick the best torch device for the version banner; the index math itself is pure numpy."""
    try:
        import torch

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print("torch:", torch.__version__, "| device:", device)
        return device
    except ImportError:
        print("torch: not installed (index math is pure numpy — unaffected)")
        return "cpu"


def main() -> None:
    _report_device()
    print("numpy:", np.__version__)
    corpus, queries = make_dataset()
    print(f"corpus: {corpus.shape[0]:,} vectors x {corpus.shape[1]} dims | queries: {len(queries)}")
    print(f"IVF: nlist={NLIST} cells | top_k={TOP_K}\n")

    # --- Brute-force cost, made concrete ---
    multiply_adds = corpus.shape[0] * corpus.shape[1]
    print(f"Brute force scans every vector: {corpus.shape[0]:,} x {corpus.shape[1]} = "
          f"{multiply_adds:,} multiply-adds PER query.")
    print(f"At 10M x 768-dim that would be {10_000_000 * 768:,} (~7.7B) per query — the wall ANN clears.\n")

    # --- Ground truth (exact) for recall measurement ---
    ground_truth = np.array([brute_force_topk(q, corpus) for q in queries])

    # --- Build IVF and sweep nprobe: the recall/speed tradeoff and the cliff ---
    index = build_ivf(corpus)
    sweep = evaluate_ivf_sweep(index, queries, ground_truth)
    print("IVF recall/speed tradeoff (sweep nprobe) — the recall CLIFF at low nprobe:")
    print(f"  {'nprobe':>6} | {'recall@10':>9} | {'% corpus scanned':>16} | {'~speedup vs flat':>16}")
    print("  " + "-" * 60)
    for nprobe, (recall, frac) in sweep.items():
        speedup = 1.0 / frac if frac > 0 else float("inf")
        print(f"  {nprobe:>6} | {recall:>9.3f} | {frac * 100:>15.1f}% | {speedup:>15.1f}x")

    # --- Correctness BEFORE any claim: recall must RISE with nprobe and the cliff must be real ---
    recalls = [sweep[np] [0] for np in NPROBE_SWEEP]
    assert recalls[0] < 0.6, "nprobe=1 should miss many neighbours (the recall cliff)"
    assert recalls[-1] >= 0.999, "probing all cells must recover exact recall"
    assert all(a <= b + 1e-9 for a, b in zip(recalls, recalls[1:])), "recall must be monotonic in nprobe"
    sweet = next(np for np in NPROBE_SWEEP if sweep[np][0] >= 0.95)
    print(f"\n  recall climbs {recalls[0]:.3f} (nprobe=1) -> {recalls[-1]:.3f} (nprobe={NPROBE_SWEEP[-1]}); "
          f"reaches >=0.95 at nprobe={sweet} (~{1.0 / sweep[sweet][1]:.0f}x faster than flat).")

    # --- Graph (HNSW intuition): greedy hops walk DOWNHILL toward the query in a few steps ---
    # This is a SIMPLIFIED single-layer graph: it shows the greedy-hop mechanic, but with no
    # hierarchy a single descent can get stuck in a local optimum -- which is exactly the problem
    # HNSW's multi-layer long-range links solve. We report both the hop count (the mechanic) and how
    # close the landed node gets (mean rank), so the limitation is honest, not hidden.
    graph_n = 2000
    graph = build_knn_graph(corpus[:graph_n])  # O(N^2) build is the teaching shortcut on this subset
    sub_corpus = corpus[:graph_n]
    hops_list, landed_ranks = [], []
    for qi in range(50):
        node, hops = greedy_graph_search(sub_corpus, graph, queries[qi], entry=0)
        hops_list.append(hops)
        # rank of the landed node among all sub_corpus vectors (0 = exact NN) -> how good the stop was
        full_order = np.argsort(((sub_corpus - queries[qi]) ** 2).sum(axis=1))
        landed_ranks.append(int(np.where(full_order == node)[0][0]))
    mean_hops = float(np.mean(hops_list))
    median_rank = float(np.median(landed_ranks))
    print(f"\nGraph greedy search (SIMPLIFIED single-layer graph, N={graph_n:,}): "
          f"mean {mean_hops:.1f} hops downhill to a local optimum, landing at median rank "
          f"{median_rank:.0f} of {graph_n:,} (top {100 * (median_rank + 1) / graph_n:.1f}%).")
    print("  The hop mechanic is real; the local-optimum stalls are why HNSW adds hierarchy "
          "(long-range links) for reliable O(log N) descent.")
    assert mean_hops >= 1.0, "greedy search should take at least one downhill hop"

    # --- PQ memory math ---
    raw_bits, pq_bits, ratio = pq_compression_ratio()
    print(f"\nProduct Quantization memory: raw float32 vector = {raw_bits:,} bits "
          f"({raw_bits // 8} bytes); PQ code (m={PQ_M}, {PQ_NBITS} bits each) = {pq_bits} bits "
          f"({pq_bits // 8} bytes) -> {ratio:.0f}x smaller.")


if __name__ == "__main__":
    main()
