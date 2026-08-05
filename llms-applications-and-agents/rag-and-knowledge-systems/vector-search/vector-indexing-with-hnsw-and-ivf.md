---
id: vector-indexing-with-hnsw-and-ivf
title: Vector Indexing with HNSW and IVF
minutes: 15
category: rag-and-knowledge-systems
---
# Vector index primer (HNSW / IVF)

> Curated concept note, migrated from the in-app practice library into the canonical repo so the content lives once at the source. Original wording.

## Problem statement

You need to serve approximate nearest-neighbor (ANN) search over 100M embeddings of 768 dimensions at p99 under 20 ms, with daily reindexing. Pick an index type (HNSW, IVF-PQ, ScaNN, DiskANN) and justify. The interviewer will push on recall-latency tradeoffs, memory cost, and behavior under incremental updates and deletes.

## Core concepts

- **Exact k-NN** on 100M × 768-dim vectors is O(N·d) per query — infeasible at serving latencies.
- **Approximate k-NN (ANN)** trades a few percent of recall for 100-1000× speedup.
- Two dominant index families: **graph-based** (HNSW) and **partition-based** (IVF, IVF-PQ).
- **Quantization** (PQ, SQ, BQ) shrinks vectors to fit larger corpora in RAM at a recall cost.
- Every index has knobs that trade recall vs latency vs memory — there is no free lunch. Benchmark on YOUR data.

## Deep dive

### HNSW (Hierarchical Navigable Small World)

A multi-layer graph where each node is a vector. Upper layers are sparse shortcut graphs; lower layers are dense. Search starts at the top entry point and greedily descends to the nearest neighbor at each layer.

- **Parameters**: `M` = neighbors per node (usually 16-64), `efConstruction` (index build effort), `efSearch` (query-time effort — higher ef → higher recall → more latency).
- **Strengths**: best recall/latency Pareto for in-memory workloads; simple to tune (just crank `efSearch` until recall is good).
- **Weaknesses**: RAM-heavy (full vectors + graph edges). 100M × 768 × 4 bytes = 300 GB vectors alone; graph adds ~30 GB at M=32. Does not fit a single node without quantization.
- **Updates**: insert is cheap; delete is NOT native — most libs mark as tombstone and rebuild periodically.

### IVF (Inverted File)

Cluster vectors into `nlist` Voronoi cells using k-means. At query time, probe the `nprobe` closest cells and brute-force within them.

- **Parameters**: `nlist` (typically `4 * sqrt(N)` → ~40k for 100M vectors), `nprobe` (higher → higher recall → more latency).
- **Strengths**: simple to distribute (each cell is independent), cheap incremental updates.
- **Weaknesses**: recall-latency curve worse than HNSW for in-memory corpora; cell imbalance hurts tail latency.
- Rarely deployed alone; usually combined with PQ.

### IVF-PQ (+ Product Quantization)

After IVF partitioning, each vector inside a cell is compressed via PQ: split 768 dims into `m` subvectors (e.g. 48 groups of 16 dims each), train a 256-centroid codebook per subgroup, store each vector as `m` bytes.

- **Compression**: 768-dim × 4 bytes = 3 kB → 48 bytes. ~60× reduction.
- **Tradeoff**: quantization error degrades recall. Often combined with **re-ranking**: retrieve top-100 with PQ, re-score top-100 exactly with full-precision vectors, return top-10.
- The default choice when your corpus cannot fit in RAM at full precision.

### ScaNN

Google's library. Uses **anisotropic quantization** — tunes quantization to emphasize direction over magnitude — plus a learned partitioner. Near-HNSW recall at lower RAM.

- Strong in the 1B+ scale regime. Less mature ops-wise than FAISS outside Google.

### DiskANN / Vamana

Graph-based like HNSW but designed for SSD. Stores a single-layer graph on disk and reads pages during search. Scales to 1B+ vectors on a single node at the cost of 5-10× the latency vs in-memory.

### Flat (brute-force)

No index. O(N·d) per query with SIMD, ~100 MQPS throughput per core on a single vector. Use for:

- < 100k vectors (brute force is faster than any index build).
- Ground-truth generation for recall evaluation.
- The re-rank stage on top of an approximate first pass.

### Sizing example (100M × 768 dims)

| Index | Memory | p99 @ recall=0.95 | Build time |
|---|---|---|---|
| Flat | 300 GB | ~30 sec (single node) | 0 |
| HNSW (M=32) | ~330 GB | 5-10 ms | 2-6 hours |
| IVF-PQ (nlist=40k, m=48) | ~6 GB | 10-30 ms | 1-3 hours |
| ScaNN | ~15 GB | 8-20 ms | 1-2 hours |
| DiskANN | 30 GB RAM + 300 GB SSD | 30-100 ms | 8-24 hours |

### Recall-latency curve + tuning

Always plot recall-vs-latency on your data with your target queries. Knob sweeps:

- HNSW: vary `efSearch` from 16 to 512.
- IVF-PQ: vary `nprobe` from 1 to 128.

Pick the knob value that gives you the required recall at the budgeted latency. "Recall" alone is meaningless without a latency pairing.

### Deletes + updates

- **HNSW**: native delete is hard; practical approach is tombstone-and-rebuild or "fresh index daily + merge".
- **IVF-PQ**: cells are independent; delete = mark tombstone in the cell and occasionally re-cluster.
- For high-churn corpora (messaging, feed data): consider a two-tier design — live index for recent updates, periodically merged into the main index (log-structured).

## Discussion prompts

1. Your dataset is 10M vectors × 768 dims. Pick an index and defend the pick with memory + latency numbers.
2. Same dataset but growing to 5B and must stay on one node. What changes?
3. HNSW gives 0.99 recall at 8 ms p99 but memory is 2× your budget. Walk through three mitigation paths.
4. How do you measure recall of an approximate index? Where do you get the ground truth?
5. Incremental updates every 1 second. Which index type suffers and how do you work around it?

## Things to defend

- [ ] Differentiate graph-based (HNSW) from partition-based (IVF) with a one-liner on each.
- [ ] Know that PQ compresses vectors ~60× with a recall penalty and is the standard answer for RAM-constrained corpora.
- [ ] Explain re-ranking — retrieve with quantized index, re-score with full-precision — and when you need it.
- [ ] Quote approximate memory + latency figures for 100M-scale HNSW vs IVF-PQ.
- [ ] Describe the HNSW `efSearch` knob and how it trades recall for latency.
- [ ] State the delete/update limitation of HNSW and a production workaround.
- [ ] Defend a choice between in-memory (HNSW) and SSD-backed (DiskANN) with a concrete scale threshold.
- [ ] Know that recall without a latency pairing is meaningless.
