"""Generate the teaching notebook 04-Vector-Databases-and-ANN-Indexes.ipynb.

Authoring the notebook as code keeps every cell under version control and lets us regenerate it
deterministically. The notebook itself imports ONLY numpy + faiss + the local `vector_indexes`
module — never torch (faiss + torch co-loading crashes on this box; embeddings are precomputed by
`embed_corpus.py`). Execute headless with:

    python -m nbconvert --to notebook --execute --inplace 04-Vector-Databases-and-ANN-Indexes.ipynb

Run this generator with:  python build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "04-Vector-Databases-and-ANN-Indexes.ipynb"


_CELL_COUNTER = [0]


def _next_id(kind: str) -> str:
    _CELL_COUNTER[0] += 1
    return f"{kind}-{_CELL_COUNTER[0]:02d}"


def md(*lines: str) -> dict:
    src = "\n".join(lines)
    return {
        "cell_type": "markdown",
        "id": _next_id("md"),
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(*lines: str) -> dict:
    src = "\n".join(lines)
    return {
        "cell_type": "code",
        "id": _next_id("code"),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELLS = [
    md(
        "# Vector Databases & ANN Indexes — a stepwise walkthrough over a *real* corpus",
        "",
        "This notebook is **not a toy**, and it doesn't skip steps. It loads real sentence-transformer "
        "embeddings of **30,000 real Wikipedia passages**, then walks — one operation at a time — through "
        "exactly how a real-world vector search engine is built and tuned: the exact baseline, the "
        "k-means partition that IVF is built on, how inverted lists are scanned, how the HNSW graph's "
        "layer pyramid gives $O(\\log N)$ navigation, and how Product Quantization encodes a vector into "
        "a few bytes. At every step we *measure* what an ANN engineer measures — **recall@10 against "
        "exact search** and **real single-query latency** — and read the result off the real index.",
        "",
        "**Why the embeddings are precomputed.** On this machine `faiss` and `torch` both link `libomp`; "
        "importing them in one process double-initialises OpenMP and *crashes* the kernel (even with "
        "`KMP_DUPLICATE_LIB_OK=TRUE`). So the embedding step (torch / sentence-transformers) runs once in "
        "a separate process — `python embed_corpus.py` — and writes plain `.npy` vectors to `data/`. This "
        "notebook is the **indexing/serving** half: numpy + faiss only. That split is also exactly how "
        "real-world systems work — embedding is a batch job, the index is a serving system.",
        "",
        "> Run `python embed_corpus.py` once first if `data/` is empty. Latency here is real wall-clock and "
        "varies run to run (we report medians, measured one query at a time on one core — the honest "
        "per-request cost); recall is deterministic given the cached vectors.",
    ),
    # ---------------------------------------------------------------- Step 1
    md(
        "## Step 1 — load the real corpus and feel the exact-search wall",
        "",
        "We load the cached real embeddings and print what we actually have: how many passages, the "
        "embedding dimension, the model, and the source dataset. Then we count the brute-force cost. "
        "Exact ('flat') search compares the query to **every** vector — $O(N \\cdot d)$ per query. At our "
        "real scale that is already millions of multiply-adds per query; at a real-world 10M×768 corpus it "
        "is **~7.7 billion** per query. That linear wall is the whole reason ANN indexes exist.",
    ),
    code(
        "import time",
        "",
        "import faiss",
        "import numpy as np",
        "",
        "from vector_indexes import (",
        "    TOP_K, HNSW_M, HNSW_EF_CONSTRUCTION, PQ_M, PQ_NBITS,",
        "    load_corpus, build_flat, exact_topk, exact_latency_ms,",
        "    build_ivf, sweep_ivf, ivf_cell_sizes, ivf_centroids,",
        "    build_hnsw, sweep_hnsw, hnsw_level_counts,",
        "    build_ivfpq, pq_encode_decode, pq_memory_bytes, recall_at_k,",
        ")",
        "",
        "print('faiss:', faiss.__version__, '| numpy:', np.__version__)",
        "corpus = load_corpus()",
        "print(f'corpus : {corpus.n:,} real passages x {corpus.dim}-dim')",
        "print(f'model  : {corpus.meta[\"embed_model\"]}')",
        "print(f'dataset: {corpus.meta[\"dataset\"]}')",
        "print(f'queries: {len(corpus.queries):,}  (held-out real passages, re-embedded)')",
        "",
        "mults = corpus.n * corpus.dim",
        "print(f'\\nbrute force scans every vector: {corpus.n:,} x {corpus.dim} = {mults:,} multiply-adds / query')",
        "print(f'at a 10M x 768 corpus that would be {10_000_000 * 768:,} (~7.7B) / query — the wall ANN clears.')",
    ),
    # ---------------------------------------------------------------- Step 2
    md(
        "## Step 2 — look at the real data (and confirm the vectors are unit-norm)",
        "",
        "These are real Wikipedia passages — the exact kind of text a RAG system indexes. A query is a "
        "held-out passage's own text, so we know a strong true neighbour exists. Crucially, we also check "
        "that every vector is **L2-normalised** (unit length). That's not a detail — it's *why* we can use "
        "inner-product indexes below: for unit vectors, the dot product $\\mathbf{q}\\cdot\\mathbf{v}$ **is** "
        "the cosine similarity. So 'maximise inner product' and 'maximise cosine' are the same ranking, and "
        "FAISS's `METRIC_INNER_PRODUCT` gives us cosine search for free.",
    ),
    code(
        "for p in corpus.passages[:3]:",
        "    print(f'[{p[\"id\"]}] {p[\"title\"]}: {p[\"text\"][:130]}...')",
        "",
        "norms = np.linalg.norm(corpus.embeddings[:1000], axis=1)",
        "print(f'\\nvector L2 norms: min {norms.min():.4f}, max {norms.max():.4f} '",
        "      f'(≈1.0 → inner product == cosine similarity)')",
        "q0 = int(corpus.query_idx[0])",
        "print(f'\\nexample query (passage {q0}, title \"{corpus.passages[q0][\"title\"]}\"):')",
        "print(' ', corpus.passages[q0]['text'][:180], '...')",
    ),
    # ---------------------------------------------------------------- Step 3
    md(
        "## Step 3 — build the exact baseline (`IndexFlatIP`) and read a real result",
        "",
        "`IndexFlatIP` is FAISS's exact inner-product index: it stores every vector and, per query, computes "
        "the dot product against all N and returns the top-k. It does no approximation, so its top-k is the "
        "**ground truth** we score every ANN index against. `build_flat` just constructs it and calls "
        "`.add()` (which copies the vectors in — Flat needs no training). We retrieve for one real query and "
        "read the passages back: exact search returns genuinely on-topic neighbours, which is the point of "
        "using real data — you can *read* that retrieval works before we start approximating it.",
    ),
    code(
        "flat = build_flat(corpus.embeddings)          # IndexFlatIP — exact, stores all N vectors",
        "print(f'flat index holds {flat.ntotal:,} vectors, dim {flat.d}')",
        "",
        "qi = 0",
        "_scores, ids = flat.search(corpus.queries[qi:qi+1], TOP_K)   # exact top-10 for one query",
        "print(f'\\nquery: {corpus.passages[int(corpus.query_idx[qi])][\"title\"]}')",
        "for rank, doc_id in enumerate(ids[0][:5]):",
        "    p = corpus.passages[int(doc_id)]",
        "    print(f'  #{rank+1}  sim={_scores[0][rank]:.3f}  {p[\"title\"]}: {p[\"text\"][:80]}...')",
    ),
    # ---------------------------------------------------------------- Step 4
    md(
        "## Step 4 — the ground truth and the exact-search latency (the wall, measured)",
        "",
        "To score approximate indexes we need the exact top-10 for *all* 500 queries — that's our ground "
        "truth. We also measure Flat's real latency the honest way: **one query at a time, on one core**. "
        "(FAISS's batch search parallelises across queries and cores, which hides the per-request cost; a "
        "real RAG request embeds and searches a single query.) On our 30k×384 corpus this lands around "
        "**~1 ms/query** — already fast, because FAISS's SIMD dot products are excellent. That's an honest, "
        "important nuance: at *this* scale ANN's win is real but modest; the dramatic $O(N)$ blow-up bites "
        "at 10M–1B vectors. Even here, though, we'll measure a genuine 15–30× speedup.",
    ),
    code(
        "t0 = time.perf_counter()",
        "ground_truth = exact_topk(flat, corpus.queries, TOP_K)   # (500, 10) exact neighbour ids",
        "print(f'exact top-{TOP_K} for all {len(corpus.queries):,} queries in {time.perf_counter()-t0:.2f}s')",
        "flat_ms = exact_latency_ms(flat, corpus.queries)         # single-query, single-thread median",
        "print(f'exact search latency: {flat_ms:.3f} ms/query over {corpus.n:,} vectors (one core)')",
    ),
    # ---------------------------------------------------------------- Step 5
    md(
        "## Step 5 — build IVF, and *look inside* the k-means partition",
        "",
        "IVF's build has two distinct phases, and it's worth seeing both. **`index.train(x)`** runs "
        "**k-means** over the corpus to learn `nlist` centroids — these become the centres of `nlist` "
        "**Voronoi cells** (every point belongs to the cell of its nearest centroid). **`index.add(x)`** "
        "then assigns each vector to its cell and appends its id to that cell's **inverted list**. So the "
        "index is really a coarse quantizer (the centroids) plus `nlist` little id-lists. Below we build it "
        "and *inspect the real partition*: the centroid matrix and the cell sizes. Real cells are not "
        "perfectly balanced — Wikipedia clusters topically — which is itself a tuning signal.",
    ),
    code(
        "ivf = build_ivf(corpus.embeddings)   # train() = k-means → centroids; add() = fill inverted lists",
        "cents = ivf_centroids(ivf)           # (nlist, dim) — the k-means codebook = cell centres",
        "sizes = ivf_cell_sizes(ivf)          # vectors per cell (inverted-list lengths)",
        "print(f'nlist = {ivf.nlist} cells | centroid matrix {cents.shape}')",
        "print(f'cell sizes: sum {sizes.sum():,} (=N), mean {sizes.mean():.0f}, '",
        "      f'min {sizes.min()}, max {sizes.max()}  (real data → imbalanced cells)')",
        "print(f'a query at nprobe=1 scans ~{sizes.mean():.0f} vectors instead of {corpus.n:,} '",
        "      f'(~{corpus.n/sizes.mean():.0f}x fewer distance computations)')",
    ),
    # ---------------------------------------------------------------- Step 6
    md(
        "## Step 6 — one IVF query: route to the nearest cells, scan only those",
        "",
        "Now watch a single query flow through IVF. Set `nprobe` (how many cells to probe). FAISS first "
        "computes the query→centroid similarities and picks the `nprobe` nearest cells (cheap: only `nlist` "
        "dot products), then exact-searches **only the vectors in those cells' inverted lists** — a tiny "
        "fraction of the corpus. If a true neighbour lives in a cell we didn't probe, we miss it: that's the "
        "'approximate' in ANN, and it's exactly why recall < 1 at small `nprobe`. We show the recall for "
        "this one query at `nprobe=8` and confirm it scanned far fewer than N vectors.",
    ),
    code(
        "ivf.nprobe = 8",
        "_s, ids8 = ivf.search(corpus.queries[qi:qi+1], TOP_K)",
        "r = recall_at_k(ids8, ground_truth[qi:qi+1], TOP_K)",
        "# how many vectors did probing 8 cells actually scan? sum the 8 nearest cells' sizes",
        "cell_sims = corpus.queries[qi] @ cents.T",
        "probed = np.argsort(-cell_sims)[:ivf.nprobe]",
        "scanned = int(sizes[probed].sum())",
        "print(f'nprobe={ivf.nprobe}: recall@{TOP_K} for query {qi} = {r:.2f}')",
        "print(f'scanned ~{scanned:,} of {corpus.n:,} vectors '",
        "      f'({100*scanned/corpus.n:.1f}% of the corpus) — the IVF speedup')",
    ),
    # ---------------------------------------------------------------- Step 7
    md(
        "## Step 7 — sweep `nprobe`: the real recall cliff",
        "",
        "This is the headline. `sweep_ivf` sets `nprobe` to each value, searches all 500 queries, and "
        "records the **real mean recall@10 vs exact** and the **real median single-query latency**. Read the "
        "table top to bottom: at `nprobe=1` the index is fastest but recall is low (it misses neighbours in "
        "unprobed cells); as `nprobe` grows recall climbs steeply then plateaus at ~1.0 (where you've "
        "effectively scanned everything — and are now *slower* than exact because of the centroid-scan "
        "overhead). The **sweet spot** is the smallest `nprobe` that clears your recall target — most of the "
        "recall for a fraction of the exact-search cost.",
    ),
    code(
        "ivf_points = sweep_ivf(ivf, corpus.queries, ground_truth)",
        "print(f'{\"nprobe\":>6} | {\"recall@10\":>9} | {\"ms/query\":>9} | {\"speedup vs exact\":>16}')",
        "print('-' * 50)",
        "for p in ivf_points:",
        "    print(f'{p.knob:>6} | {p.recall:>9.3f} | {p.latency_ms:>9.3f} | {flat_ms / p.latency_ms:>15.1f}x')",
        "",
        "sweet = next((p for p in ivf_points if p.recall >= 0.95), ivf_points[-1])",
        "print(f'\\nsweet spot: nprobe={sweet.knob} -> recall {sweet.recall:.3f} at {sweet.latency_ms:.3f} ms '",
        "      f'({flat_ms / sweet.latency_ms:.0f}x faster than exact).')",
    ),
    # ---------------------------------------------------------------- Step 8
    md(
        "## Step 8 — build HNSW, and *see the layer pyramid* that gives $O(\\log N)$",
        "",
        "HNSW takes the other route: instead of partitioning space it wires the vectors into a **multi-layer "
        "navigable graph**. `index.add(x)` builds it incrementally — there's no separate `train()`. When each "
        "node is inserted, it is assigned a **maximum layer** drawn from an exponentially-decaying "
        "distribution with multiplier $m_L = 1/\\ln M$, so the layers form a **pyramid**: layer 0 (the base) "
        "holds *all* N nodes densely connected; each layer up keeps only $\\approx 1/M$ as many (here "
        "$1/32 \\approx 0.031$, since $e^{-1/m_L} = e^{-\\ln M} = 1/M$), with long-range links. A query enters "
        "at the sparse top and greedily descends — that shrinking pyramid is precisely what turns 'walk the "
        "graph' into $\\approx O(\\log N)$ hops. Below we read the **real per-layer node counts** off the built "
        "index and watch the geometric decay.",
    ),
    code(
        "hnsw = build_hnsw(corpus.embeddings)     # M=32 links/node, efConstruction=200 — the real graph",
        "counts = hnsw_level_counts(hnsw)         # nodes at each layer, base (level 0) first",
        "print(f'HNSW built: M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}')",
        "print(f'layer node counts (base → top): {counts.tolist()}')",
        "for lvl in range(len(counts) - 1):",
        "    print(f'  level {lvl} → {lvl+1}: {counts[lvl]:,} → {counts[lvl+1]:,} '",
        "          f'(×{counts[lvl+1]/counts[lvl]:.3f}; ~1/M = 1/{HNSW_M} ≈ {1/HNSW_M:.3f} is the target decay)')",
        "print(f'~log_M(N) ≈ {np.log(corpus.n)/np.log(HNSW_M):.1f} layers to descend — the pyramid gives O(log N).')",
    ),
    # ---------------------------------------------------------------- Step 9
    md(
        "## Step 9 — sweep `efSearch`: HNSW's recall/speed knob",
        "",
        "`efSearch` is the size of the candidate list HNSW keeps during the greedy walk on the base layer — "
        "the graph analogue of IVF's `nprobe`. A bigger `efSearch` explores more of the neighbourhood before "
        "committing (higher recall, slower); a smaller one commits faster (lower recall). We sweep it and "
        "measure the same real recall/latency. Notice how much *higher and flatter* HNSW's curve starts than "
        "IVF's — even at `efSearch=8` it's already near-perfect recall while tens of times faster than exact. "
        "That head start is HNSW's reputation, made concrete.",
    ),
    code(
        "hnsw_points = sweep_hnsw(hnsw, corpus.queries, ground_truth)",
        "print(f'{\"efSearch\":>8} | {\"recall@10\":>9} | {\"ms/query\":>9} | {\"speedup vs exact\":>16}')",
        "print('-' * 52)",
        "for p in hnsw_points:",
        "    print(f'{p.knob:>8} | {p.recall:>9.3f} | {p.latency_ms:>9.3f} | {flat_ms / p.latency_ms:>15.1f}x')",
    ),
    # ---------------------------------------------------------------- Step 10
    md(
        "## Step 10 — IVF vs HNSW on the recall/latency frontier",
        "",
        "The honest way to compare two ANN indexes is not a single number but the **recall/latency "
        "frontier**: at a target recall, which index is faster? (This is what [ANN-Benchmarks]"
        "(https://ann-benchmarks.com/) plots.) We pick a recall target and report the fastest setting of "
        "each index that clears it. On this corpus HNSW typically reaches high recall at lower latency, "
        "which is why it's the default in most modern vector DBs — at the cost of more memory and a graph "
        "that's slower to build and harder to update (both of which we saw and measured above).",
    ),
    code(
        "target = 0.95",
        "def fastest_at(points, target):",
        "    ok = [p for p in points if p.recall >= target]",
        "    return min(ok, key=lambda p: p.latency_ms) if ok else max(points, key=lambda p: p.recall)",
        "",
        "iv = fastest_at(ivf_points, target)",
        "hn = fastest_at(hnsw_points, target)",
        "print(f'at recall >= {target}:')",
        "print(f'  IVF  nprobe={iv.knob:<4} recall {iv.recall:.3f}  {iv.latency_ms:.3f} ms  '",
        "      f'({flat_ms/iv.latency_ms:.0f}x vs exact)')",
        "print(f'  HNSW efSearch={hn.knob:<4} recall {hn.recall:.3f}  {hn.latency_ms:.3f} ms  '",
        "      f'({flat_ms/hn.latency_ms:.0f}x vs exact)')",
        "faster = 'HNSW' if hn.latency_ms < iv.latency_ms else 'IVF'",
        "print(f'  -> {faster} is faster at this recall on this corpus (result is corpus/hardware dependent).')",
    ),
    # ---------------------------------------------------------------- Step 11
    md(
        "## Step 11 — Product Quantization, step 1: *encode* a real vector into a few bytes",
        "",
        "Now the memory story, shown mechanically. `pq_encode_decode` trains a real Product Quantizer: it "
        "**splits each 384-dim vector into `m=48` subvectors of 8 dims**, and for each subspace runs k-means "
        "to learn a **codebook of $2^8 = 256$ centroids**. To *encode* a vector, each subvector is replaced "
        "by the **id of its nearest sub-centroid** — an 8-bit number. The concatenation of those 48 ids "
        "**is** the code: 48 bytes, versus 1,536 bytes for the raw float32 vector. Below we encode the real "
        "corpus and print one actual code — 48 small integers standing in for a whole embedding.",
    ),
    code(
        "codes, recon, errors = pq_encode_decode(corpus.embeddings)   # train PQ, encode ALL vectors",
        "print(f'codes shape {codes.shape}, dtype {codes.dtype}   (N x m = {corpus.n:,} x {PQ_M})')",
        "print(f'one vector, before: {corpus.dim} float32 = {corpus.dim*4:,} bytes')",
        "print(f'one vector, after : {PQ_M} centroid-ids (uint8) = {PQ_M} bytes  '",
        "      f'({corpus.dim*4//PQ_M}x smaller)')",
        "print(f'\\nthe actual code for passage 0 (its {PQ_M} sub-centroid ids):')",
        "print(' ', codes[0].tolist())",
    ),
    # ---------------------------------------------------------------- Step 12
    md(
        "## Step 12 — Product Quantization, step 2: *decode* and see the quantization error",
        "",
        "To *decode*, PQ looks up each of the 48 stored centroid-ids in its codebook and concatenates the "
        "centroids back into a 384-dim vector — a lossy reconstruction. The gap between original and "
        "reconstruction is the **quantization error**, and it's the price of the 32× compression: search on "
        "PQ codes uses **asymmetric distance** (the *query* stays full-precision; only the *database* "
        "vectors are the coarse decodes), so a little error creeps into the ranking and costs some recall. "
        "We measure the real mean reconstruction error and the cosine between original and decode.",
    ),
    code(
        "cos = (corpus.embeddings * recon).sum(axis=1) / (",
        "    np.linalg.norm(corpus.embeddings, axis=1) * np.linalg.norm(recon, axis=1) + 1e-9)",
        "print(f'mean L2 reconstruction error: {errors.mean():.4f}  (0 = lossless)')",
        "print(f'mean cosine(original, decoded): {cos.mean():.4f}  '",
        "      f'(≈1 would be lossless; the gap is what costs recall)')",
        "raw_b, pq_b, ratio = pq_memory_bytes(corpus.dim, PQ_M, PQ_NBITS)",
        "print(f'\\nmemory: raw {raw_b} B/vector -> PQ {pq_b} B ({ratio:.0f}x); '",
        "      f'full corpus {corpus.n*raw_b/1e6:.1f} MB -> {corpus.n*pq_b/1e6:.2f} MB')",
    ),
    # ---------------------------------------------------------------- Step 13
    md(
        "## Step 13 — put it together: a real `IndexIVFPQ` (IVF routing + PQ codes)",
        "",
        "The billion-scale workhorse combines both ideas: **IVF routing** (probe a few cells) over "
        "**PQ-compressed vectors** (48 bytes each). `build_ivfpq` trains the coarse k-means *and* the PQ "
        "codebooks, then adds the compressed codes. We measure its real recall — lower than plain IVF at the "
        "same `nprobe`, because PQ's approximate distances add a second source of error on top of the cell "
        "misses. That's the deal: you accept a recall hit you can partly buy back by over-fetching and "
        "re-ranking on exact vectors, in exchange for fitting billions of vectors in RAM.",
    ),
    code(
        "ivfpq = build_ivfpq(corpus.embeddings)   # trains coarse k-means AND the PQ codebooks",
        "ivfpq.nprobe = 16",
        "_s, ids_pq = ivfpq.search(corpus.queries, TOP_K)",
        "r_pq = recall_at_k(ids_pq, ground_truth, TOP_K)",
        "r_ivf16 = next(p.recall for p in ivf_points if p.knob == 16)",
        "print(f'at nprobe=16:  plain IVF recall@{TOP_K} = {r_ivf16:.3f}   '",
        "      f'IVFPQ recall@{TOP_K} = {r_pq:.3f}')",
        "print(f'PQ costs {r_ivf16 - r_pq:.3f} recall here, in exchange for {ratio:.0f}x less memory '",
        "      f'({corpus.n*raw_b/1e6:.0f} MB -> {corpus.n*pq_b/1e6:.1f} MB).')",
    ),
    # ---------------------------------------------------------------- Try it yourself
    md(
        "## Try it yourself",
        "",
        "Before you run anything, **predict**:",
        "",
        "1. The IVF cliff came from `nlist=256` cells. Rebuild with **more** cells "
        "(`build_ivf(corpus.embeddings, nlist=1024)`) so each cell is *smaller*. What happens to recall at a "
        "fixed `nprobe=8` — up or down? (Hint: smaller cells → fewer of a query's neighbours per cell → you "
        "must probe more to recover them.)",
        "2. Raise HNSW `M` (rebuild with `build_hnsw(corpus.embeddings, m=64)`). What happens to recall at a "
        "fixed `efSearch`, to the layer pyramid (`hnsw_level_counts`), and to memory?",
        "3. Increase the PQ compression: `pq_encode_decode(corpus.embeddings, m=24)` → coarser codes, more "
        "compression. Predict the reconstruction error and recall change, then measure it.",
        "",
        "Then try each and check your prediction against the real measured numbers. That predict→measure loop "
        "is exactly how you tune a real vector store.",
    ),
    # ---------------------------------------------------------------- What we saw
    md(
        "## What we saw",
        "",
        "- **Exact search is $O(N \\cdot d)$** — real, measured latency over 30k×384 vectors (~1 ms/query); "
        "hopeless at 10M×768 (~7.7B ops/query). That linear wall is why ANN exists.",
        "- **IVF is a k-means partition** — we looked inside it: real centroids and imbalanced cell sizes; a "
        "query routes to the `nprobe` nearest cells and scans only those. We *measured* the recall cliff and "
        "the sweet spot.",
        "- **HNSW is a layered graph** — we read the real layer pyramid (each level ~1/M of the one below, "
        "with $m_L=1/\\ln M$), which is what makes greedy descent ~$O(\\log N)$; `efSearch` is its recall/speed "
        "knob, and it sits higher-and-left on the recall/latency frontier.",
        "- **PQ is a codebook compression** — we encoded real vectors into 48-byte codes and decoded them, "
        "measuring the reconstruction error that trades a little recall for ~32× memory; `IndexIVFPQ` fuses "
        "it with IVF routing for billion-scale search.",
        "- **The frontier is the tool** — you tune the smallest knob that clears your recall SLO, not the "
        "fastest setting.",
        "",
        "Everything here is real: a real corpus, real FAISS indexes inspected from the inside, real recall "
        "measured against exact search, real latency. That's the difference between reading about ANN and "
        "being able to *operate* it.",
    ),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_md = sum(1 for c in CELLS if c["cell_type"] == "markdown")
    n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"wrote {NB_PATH} with {len(CELLS)} cells ({n_md} md, {n_code} code)")


if __name__ == "__main__":
    main()
