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
        "# Vector Databases & ANN Indexes — searching a *real* corpus in milliseconds",
        "",
        "This notebook is **not a toy**. It loads real sentence-transformer embeddings of "
        "**30,000 real Wikipedia passages**, builds **real FAISS indexes** (exact Flat, IVF, HNSW, "
        "IVF+PQ), runs real semantic queries, and *measures* what every ANN engineer measures: "
        "**recall@10 against exact search** and **real query latency** as you turn the recall/speed "
        "knob. Every number below is produced live by the cell above it.",
        "",
        "**Why the embeddings are precomputed.** On this machine `faiss` and `torch` both link "
        "`libomp`; importing them in one process double-initialises OpenMP and *crashes* the kernel "
        "(even with `KMP_DUPLICATE_LIB_OK=TRUE`). So the embedding step (torch / sentence-transformers) "
        "runs once in a separate process — `python embed_corpus.py` — and writes plain `.npy` vectors "
        "to `data/`. This notebook is the **indexing/serving** half: numpy + faiss only. That split is "
        "also exactly how production works — embedding is a batch job, the index is a serving system.",
        "",
        "> Run `python embed_corpus.py` once first if `data/` is empty. Timing here is real wall-clock "
        "and varies run to run (we report medians); recall is deterministic given the cached vectors.",
    ),
    md(
        "## Step 1 — load the real corpus and feel the exact-search wall",
        "",
        "We load the cached real embeddings and print what we actually have: how many passages, the "
        "embedding dimension, the model, and the source dataset. Then we count the brute-force cost. "
        "Exact ('flat') search compares the query to **every** vector — $O(N \\cdot d)$ per query. At "
        "our real scale that is already millions of multiply-adds per query; at a production "
        "10M×768 corpus it is **~7.7 billion** per query. That linear wall is the whole reason ANN "
        "indexes exist.",
    ),
    code(
        "import time",
        "",
        "import faiss",
        "import numpy as np",
        "",
        "from vector_indexes import (",
        "    TOP_K, IVF_NLIST, HNSW_M, HNSW_EF_CONSTRUCTION, PQ_M, PQ_NBITS,",
        "    load_corpus, build_flat, exact_topk, exact_latency_ms,",
        "    build_ivf, sweep_ivf, build_hnsw, sweep_hnsw,",
        "    build_ivfpq, pq_memory_bytes, recall_at_k,",
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
    md(
        "## Step 2 — look at the real data",
        "",
        "These are real Wikipedia passages — the exact kind of text a RAG system indexes. A query is a "
        "held-out passage's own text, so we know a strong true neighbour exists (the passage itself and "
        "its topical siblings). Seeing the real text makes the retrieved results below interpretable — "
        "you can *read* whether a neighbour is genuinely on-topic.",
    ),
    code(
        "for p in corpus.passages[:3]:",
        "    print(f'[{p[\"id\"]}] {p[\"title\"]}: {p[\"text\"][:140]}...')",
        "",
        "q0 = int(corpus.query_idx[0])",
        "print('\\nexample query (passage', q0, '):')",
        "print(' ', corpus.passages[q0]['text'][:200], '...')",
    ),
    md(
        "## Step 3 — the exact baseline (`IndexFlatIP`) and its latency",
        "",
        "`IndexFlatIP` is FAISS's exact inner-product index. Because `embed_corpus.py` L2-normalised "
        "every vector, inner product **is** cosine similarity, and Flat's top-k is the *ground truth* "
        "we score every approximate index against. We also measure its real per-query latency — the "
        "cost we are trying to beat. Then we retrieve for one real query and read the passages back: "
        "exact search returns genuinely on-topic neighbours.",
    ),
    code(
        "flat = build_flat(corpus.embeddings)",
        "t0 = time.perf_counter(); ground_truth = exact_topk(flat, corpus.queries, TOP_K)",
        "print(f'exact top-{TOP_K} for all {len(corpus.queries):,} queries in {time.perf_counter()-t0:.2f}s')",
        "flat_ms = exact_latency_ms(flat, corpus.queries)",
        "print(f'exact search latency: {flat_ms:.3f} ms/query over {corpus.n:,} vectors')",
        "",
        "# read back the neighbours of one query — are they on-topic?",
        "qi = 0",
        "print(f'\\nquery: {corpus.passages[int(corpus.query_idx[qi])][\"title\"]}')",
        "for rank, doc_id in enumerate(ground_truth[qi][:5]):",
        "    p = corpus.passages[int(doc_id)]",
        "    print(f'  #{rank+1}  {p[\"title\"]}: {p[\"text\"][:90]}...')",
    ),
    md(
        "## Step 4 — build a real IVF index and probe a few cells",
        "",
        "IVF (Inverted File) trains **k-means** to partition the corpus into `nlist` Voronoi cells, then "
        "at query time probes only the `nprobe` nearest cells instead of scanning all N. FAISS does the "
        "real k-means training and inverted-list construction for us. With a small `nprobe` the index "
        "touches a fraction of the corpus — that's the speedup — but a true neighbour sitting in an "
        "unprobed cell is missed, which is the recall cost we'll measure next.",
    ),
    code(
        "ivf = build_ivf(corpus.embeddings)   # FAISS trains k-means into nlist cells + builds inverted lists",
        "print(f'IVF trained: nlist={ivf.nlist} cells over {corpus.n:,} vectors')",
        "",
        "ivf.nprobe = 8",
        "_scores, ids = ivf.search(corpus.queries[:1], TOP_K)",
        "r = recall_at_k(ids, ground_truth[:1], TOP_K)",
        "print(f'nprobe={ivf.nprobe}: recall@{TOP_K} for query 0 = {r:.2f} '",
        "      f'(probing {ivf.nprobe} of {ivf.nlist} cells)')",
    ),
    md(
        "## Step 5 — sweep `nprobe`: the real recall cliff",
        "",
        "This is the headline measurement. We sweep `nprobe` from 1 up to `nlist` and record, for each, "
        "the **real mean recall@10 vs exact** and the **real median latency**. Read the table top to "
        "bottom: at `nprobe=1` the index is fastest but recall is low (it misses neighbours in unprobed "
        "cells); as `nprobe` grows recall climbs steeply then plateaus at ~1.0 (where you've effectively "
        "scanned everything). The **sweet spot** is the smallest `nprobe` that clears your recall target "
        "— most of the recall for a fraction of the exact-search cost.",
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
    md(
        "## Step 6 — build a real HNSW index and sweep `efSearch`",
        "",
        "HNSW takes the other route: instead of partitioning space it wires the vectors into a **multi-"
        "layer navigable graph** and answers a query by greedily hopping neighbour→neighbour toward it, "
        "touching ~$O(\\log N)$ nodes. FAISS builds the real graph (`M` links per node — printed below). "
        "The query-time knob is `efSearch` (the candidate-list size): raise it for higher recall, lower it for speed — "
        "HNSW's analogue of `nprobe`. We sweep it and measure the same real recall/latency.",
    ),
    code(
        "hnsw = build_hnsw(corpus.embeddings)   # FAISS builds the multi-layer navigable graph",
        "print(f'HNSW built: M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}')",
        "hnsw_points = sweep_hnsw(hnsw, corpus.queries, ground_truth)",
        "print(f'{\"efSearch\":>8} | {\"recall@10\":>9} | {\"ms/query\":>9} | {\"speedup vs exact\":>16}')",
        "print('-' * 52)",
        "for p in hnsw_points:",
        "    print(f'{p.knob:>8} | {p.recall:>9.3f} | {p.latency_ms:>9.3f} | {flat_ms / p.latency_ms:>15.1f}x')",
    ),
    md(
        "## Step 7 — IVF vs HNSW on the recall/latency frontier",
        "",
        "The honest way to compare two ANN indexes is not a single number but the **recall/latency "
        "frontier**: at a target recall, which index is faster? (This is what [ANN-Benchmarks]"
        "(https://ann-benchmarks.com/) plots.) We pick a recall target and report the fastest setting of "
        "each index that clears it. On this corpus HNSW typically reaches high recall at lower latency, "
        "which is why it's the default in most modern vector DBs — at the cost of more memory and a "
        "graph that's harder to update.",
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
    md(
        "## Step 8 — Product Quantization: the memory story",
        "",
        "At billion scale the raw float32 vectors don't fit in RAM. **Product Quantization** splits each "
        "vector into `m` sub-vectors and replaces each with a small code id from a learned codebook — "
        "shrinking a vector from `dim×4` bytes to `m×nbits/8` bytes. We build a real `IndexIVFPQ` (IVF "
        "routing + PQ compression — the billion-scale workhorse), measure its real recall (PQ adds a "
        "second, small approximation on top of the cell misses), and print the real memory saving on "
        "our corpus.",
    ),
    code(
        "raw_b, pq_b, ratio = pq_memory_bytes(corpus.dim, PQ_M, PQ_NBITS)",
        "print(f'per vector: raw {raw_b} bytes ({corpus.dim} dims x 4B) -> PQ {pq_b} bytes '",
        "      f'(m={PQ_M}, {PQ_NBITS} bits) = {ratio:.0f}x smaller')",
        "print(f'full corpus: raw {corpus.n*raw_b/1e6:.1f} MB -> PQ {corpus.n*pq_b/1e6:.2f} MB')",
        "",
        "ivfpq = build_ivfpq(corpus.embeddings)",
        "ivfpq.nprobe = 16",
        "_s, ids = ivfpq.search(corpus.queries, TOP_K)",
        "print(f'\\nIVFPQ recall@{TOP_K} at nprobe=16: {recall_at_k(ids, ground_truth, TOP_K):.3f} '",
        "      f'(PQ trades a little recall for {ratio:.0f}x less memory)')",
    ),
    md(
        "## Try it yourself",
        "",
        "Before you run anything, **predict**:",
        "",
        "1. The IVF cliff came from `nlist=256` cells. If you rebuild with **more** cells (`build_ivf(corpus.embeddings, nlist=1024)`) so each cell is *smaller*, what happens to recall at a fixed `nprobe=8` — up or down? (Hint: smaller cells → fewer of a query's neighbours per cell → you must probe more to recover them.)",
        "2. Raise HNSW `M` (rebuild with `build_hnsw(corpus.embeddings, m=64)`). What happens to recall at a fixed `efSearch`, and to the index's memory?",
        "3. Increase the PQ compression (`PQ_M=24` → coarser codes, more compression). Predict the recall change, then measure it.",
        "",
        "Then try each and check your prediction against the real measured numbers. That prediction→measure loop is exactly how you tune a real vector store.",
    ),
    md(
        "## What we saw",
        "",
        "- **Exact search is $O(N \\cdot d)$** — real, measured latency over ~40k×384 vectors; hopeless "
        "at 10M×768 (~7.7B ops/query). That linear wall is why ANN exists.",
        "- **IVF skips most vectors** — real k-means cells let a query probe only `nprobe` nearby cells; "
        "we *measured* the recall cliff (low `nprobe` = fast but misses neighbours) and the sweet spot.",
        "- **HNSW walks a graph** — real multi-layer navigable graph, ~$O(\\log N)$ descent; `efSearch` is "
        "its recall/speed knob, and it usually sits higher-and-left on the recall/latency frontier.",
        "- **The frontier is the tool** — you tune the smallest knob that clears your recall SLO, not the "
        "fastest setting.",
        "- **PQ buys memory** — real ~Nx compression from float32 vectors to codes, for a small extra "
        "recall cost; combined with IVF for billion-scale search.",
        "",
        "Everything here is real: a real corpus, real FAISS indexes, real recall measured against exact "
        "search, real latency. That's the difference between reading about ANN and being able to *operate* it.",
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
    print("wrote", NB_PATH, "with", len(CELLS), "cells")


if __name__ == "__main__":
    main()
