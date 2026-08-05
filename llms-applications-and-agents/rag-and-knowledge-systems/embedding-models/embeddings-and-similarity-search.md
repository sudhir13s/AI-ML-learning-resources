---
id: embeddings-and-similarity-search
title: Embeddings and Similarity Search
minutes: 15
category: rag-and-knowledge-systems
---
# Embeddings and similarity search

> Curated concept note, migrated from the in-app practice library into the canonical repo so the content lives once at the source. Original wording.

## Problem statement

You are building semantic search over a large corpus — documents, products, or images — where users expect "find things that *mean* the same" rather than "find things that share keywords." The system must convert each item into a vector once at ingestion, convert the query into a vector at request time, and return the nearest items by some notion of distance, all within a single-digit-millisecond search budget at billions of items. The interviewer will push on what an embedding actually *is*, which similarity metric you pick and why, whether you normalize, and where approximate search trades recall for speed.

## Core concepts

- An **embedding** is a fixed-length **dense vector** (e.g. 384–1536 floats) that places an item at a point in a high-dimensional space where **geometric proximity encodes semantic similarity** — two photos of the same red sneaker land close together; a red sneaker and a red high heel land far apart despite sharing color.
- Embeddings are produced by an **encoder model** (a bi-encoder such as `text-embedding-3-small`, `bge-large`, or a vision-language model like **CLIP**), trained so that related inputs map to nearby vectors and unrelated inputs map far apart.
- **Similarity is a distance computation** in that space: **cosine**, **dot product (inner product)**, or **L2 (Euclidean)**. The right choice depends on whether your vectors are normalized and whether magnitude carries meaning.
- At scale you cannot scan every vector. **Approximate nearest neighbor (ANN)** indexes (HNSW, IVF-PQ, ScaNN) trade a few percent of **recall** for a ~1000× speedup over exact brute force.
- The same encoder MUST embed both documents and queries. Mixing encoders puts query and corpus in **different geometric spaces** and silently produces meaningless scores.

## Deep dive

### What the encoder learns

A bi-encoder maps each input to one vector independently. CLIP, for example, runs a Vision Transformer over images and a Transformer over text, trained with a contrastive objective (**InfoNCE**) to maximize cosine similarity between matching (image, text) pairs and minimize it for non-matching pairs. The result is a **shared embedding space**: a text query like "red running shoe" produces a vector directly comparable to image vectors, which is what makes cross-modal search possible. Zero-shot encoders are strong on common concepts but plateau on fine-grained distinctions (shoe brands, fabric textures) — production systems fine-tune on domain data with **hard negative mining** to push recall@10 from ~74% to 90%+.

### Dimensions and storage cost

Embedding dimensionality is a direct cost lever. A 768-dim float32 vector is `768 × 4 = 3,072 bytes`. At 1B vectors that is ~2.86 TB of raw vectors alone, before the index graph. Higher dimensions can capture more nuance but multiply both **storage** and **per-comparison compute** (each distance is O(dim)). This is why production systems quantize — scalar quantization (int8) roughly halves the footprint, product quantization (PQ) can cut 768 floats to ~96 bytes — and why "just use a bigger embedding" is never free.

### Distance and similarity metrics

- **Cosine similarity** measures the **angle** between two vectors, ignoring magnitude. It is the default for text and most semantic search because document length / token count should not change relevance. Range −1 to 1; higher is more similar.
- **Dot product (inner product)** is cosine *scaled by both magnitudes*. It is correct only when magnitude is meaningful (e.g. ads/recommendation models that bake popularity into vector length) or when vectors are already unit-normalized — in which case **dot product equals cosine** and is cheaper to compute.
- **L2 (Euclidean) distance** measures straight-line distance; smaller is more similar. On unit-normalized vectors L2 and cosine produce the same ranking, so the metric choice mostly matters when vectors are *not* normalized.

The interview-grade summary: **normalize and use cosine** for text/semantic retrieval; use **inner product** only when magnitude is a real signal and you understand why; reserve raw L2 for spaces where absolute position matters.

### Normalization

**Normalizing** a vector to unit length (dividing by its L2 norm) collapses the cosine/dot/L2 distinction and is assumed by many ANN libraries. Normalize at *both* ingest and query time, consistently. The classic bug is using a **dot-product** metric on **unnormalized** vectors: longer vectors win regardless of semantic relevance, so a generic high-magnitude chunk outranks the truly relevant one and the cause is invisible in the scores.

### Exact vs ANN at scale

Brute-force (flat) search computes the distance to every vector — exact and simple, but O(N) per query. It is the right baseline up to ~10K–1M vectors per shard, and useful as an **exact rescore** stage over a small candidate set. Beyond that, an **ANN index** is mandatory:

- **HNSW** — a navigable small-world graph; best recall/latency in RAM, but the whole graph must live in memory.
- **IVF-PQ** — coarse centroid partitioning plus product quantization; memory-efficient, good for billion-scale with an exact rerank on top-K.
- **ScaNN / DiskANN** — billion-scale options (inner-product-optimized, or SSD-resident).

ANN is *approximate*: it may miss some true neighbors, which is why recall is a first-class, monitored quantity rather than an assumption.

### Recall@k

**Recall@k** is the fraction of the true top-k nearest neighbors that the ANN search actually returns, measured against a brute-force ground truth on a benchmark query set. It is the SLO that tells you whether your index is healthy: tuning `ef_search` (HNSW) or `nprobe` (IVF) trades recall against latency, and recall can **silently decay** as deletes leave tombstones in the graph. Treat recall@10 as a monitored metric (alert below ~0.95), not a one-time benchmark.

## Common pitfalls

- **Mismatched query/document encoders** — embedding the corpus with one model and queries with another (or an upgraded version) puts them in different spaces; scores become meaningless with no error. The embedding model is part of the index contract; changing it is a migration (blue-green re-embed), not a config tweak.
- **Unnormalized dot product** — using inner product on non-unit vectors lets magnitude masquerade as relevance. Normalize, or use cosine.
- **Dimensionality and storage cost** — high-dim float32 vectors blow up RAM and per-query compute; budget for quantization rather than assuming the raw index fits.
- **Treating ANN as exact** — forgetting that approximate search misses neighbors, and not monitoring recall@k as it decays under deletes/compaction.
