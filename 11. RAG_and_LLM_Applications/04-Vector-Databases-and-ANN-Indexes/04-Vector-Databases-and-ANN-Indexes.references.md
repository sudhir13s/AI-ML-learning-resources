---
id: "11-rag-and-llm-apps/vector-databases-ann-indexes/references"
topic: "Vector Databases & ANN Indexes — References"
parent: "11-rag-and-llm-apps/vector-databases-ann-indexes"
type: references
updated: 2026-06-27
---

# Vector Databases & ANN Indexes — references and further reading

> Companion link library for **[Vector Databases & ANN Indexes](04-Vector-Databases-and-ANN-Indexes.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is a free, no-paywall link from a primary author or a recognized deep explainer — chosen for depth on *this* topic (how vector search scales: IVF, HNSW, PQ), not popularity.

**Start here — suggested path**:
1. **Get the why** — watch [What is a Vector Database?](https://www.youtube.com/watch?v=gl1r1XV0SLw) (**IBM Technology**). *Frames why specialized ANN stores exist vs scanning every vector.*
2. **Learn the index families** — read [Nearest Neighbor Indexes for Similarity Search](https://www.pinecone.io/learn/series/faiss/vector-indexes/) (**Pinecone**). *Flat → IVF → HNSW → PQ, with the trade-offs that matter.*
3. **Understand HNSW deeply** — watch [HNSW Explained](https://www.youtube.com/watch?v=77QH0Y2PYKg) (**DataMListic**), then read [Pinecone: HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/). *The graph-traversal intuition plus the `M`/`efSearch` knobs.*
4. **Understand IVF** — watch [Inverted File Index (IVF) Explained](https://www.youtube.com/watch?v=-vh6huY2rgE) (**TensorTeach**). *Voronoi cells + `nprobe`; the cluster-then-probe alternative to graphs.*
5. **Read the sources** — skim [HNSW (Malkov & Yashunin)](https://arxiv.org/abs/1603.09320) and the [FAISS GPU paper](https://arxiv.org/abs/1702.08734). *Where the algorithm and the de-facto library come from.*

**Videos**:
- [What is a Vector Database? Powering Semantic Search](https://www.youtube.com/watch?v=gl1r1XV0SLw) — **IBM Technology** — clean conceptual intro to vector DBs and ANN.
- [Vector Database Search — HNSW Explained](https://www.youtube.com/watch?v=77QH0Y2PYKg) — **DataMListic** — the layered small-world graph and greedy search, visually.
- [AI Search with HNSW](https://www.youtube.com/watch?v=7XLRCpUmiaQ) — **ObjectBox** — HNSW construction and query, end to end.
- [Inverted File Index (IVF) Explained](https://www.youtube.com/watch?v=-vh6huY2rgE) — **TensorTeach** — k-means partitioning, Voronoi cells, and `nprobe` for the IVF family.

**Interactive & visual**:
- [ANN-Benchmarks](https://ann-benchmarks.com/) — **Aumüller, Bernhardsson & Faithfull** — the standard recall-vs-queries-per-second leaderboard across ANN libraries; *see* the recall/latency Pareto frontier you tune against.
- [Faiss indexes (wiki)](https://github.com/facebookresearch/faiss/wiki) — **Meta FAISS** — the authoritative, browsable reference for choosing and configuring indexes (Flat, IVF, HNSW, PQ).

**Courses (free)**:
- [Faiss: The Missing Manual](https://www.pinecone.io/learn/series/faiss/) — **Pinecone (James Briggs)** — a full free course on vector indexes (Flat, IVF, HNSW, PQ) with runnable Python.
- [Vector Search lessons (LangChain: Chat with Your Data)](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) — **DeepLearning.AI** — the VectorStores lesson connects embeddings to a working retriever.

**Articles / blogs (free, no paywall)**:
- [Nearest Neighbor Indexes for Similarity Search](https://www.pinecone.io/learn/series/faiss/vector-indexes/) — **Pinecone** — the canonical Flat→IVF→HNSW→PQ rundown with trade-offs.
- [Hierarchical Navigable Small Worlds (HNSW)](https://www.pinecone.io/learn/series/faiss/hnsw/) — **Pinecone** — the deep dive on HNSW parameters (`M`, `efConstruction`, `efSearch`) and behavior.
- [pgvector — README (HNSW / IVFFlat indexes and parameters)](https://github.com/pgvector/pgvector) — **pgvector** — the source for the verified defaults on the page (HNSW `m = 16`, `ef_construction = 64`, `ef_search = 40`; IVFFlat `lists` guidance).
- [Vector Search Explained](https://weaviate.io/blog/vector-search-explained) — **Weaviate** — how a vector DB combines ANN with filtering and updates in production (the filtering pitfall).
- [Filtering: The Missing WHERE Clause in Vector Search](https://www.pinecone.io/learn/vector-search-filtering/) — **Pinecone** — why metadata filtering + ANN is hard (post- vs pre-filter) and how native filtered search fixes it.

**Key papers**:
- [Product Quantization for Nearest Neighbor Search (PAMI 2011)](https://inria.hal.science/inria-00514462v2/document) — **Jégou, Douze & Schmid (2011)** — defines the **inverted-file (IVF)** structure and **product quantization (PQ)**; the source for the IVF cost and PQ compression formulas on the page.
- [Efficient and Robust ANN Search using HNSW Graphs (arXiv:1603.09320)](https://arxiv.org/abs/1603.09320) — **Malkov & Yashunin (2016/2018)** — the **HNSW** algorithm; the source for the $O(\log N)$ navigation and the `M`/`efConstruction`/`efSearch` parameters.
- [Billion-scale similarity search with GPUs (arXiv:1702.08734)](https://arxiv.org/abs/1702.08734) — **Johnson, Douze & Jégou (2017/2019)** — the FAISS GPU paper; the source for the exact (flat) $O(N \cdot d)$ baseline ANN exists to beat.
- [The FAISS Library (arXiv:2401.08281)](https://arxiv.org/abs/2401.08281) — **Douze et al. (2024)** — the design of the most-used similarity-search library (IVF, PQ, HNSW); the reference for index choice.
- [DiskANN: Fast Accurate Billion-point NN Search on a Single Node](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) — **Subramanya et al. (2019, NeurIPS)** — graph ANN that spills to SSD for billion-scale search beyond RAM; the frontier past in-memory HNSW.

**Books (free, with chapters)**:
- [Introduction to Information Retrieval — Ch. 6–7 (scoring, the vector space model, efficient ranking)](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html) — **Manning, Raghavan & Schütze** — the IR foundations of similarity scoring and index efficiency, free online.

**In this platform**:
- Concept page (full explanation): [Vector Databases & ANN Indexes](04-Vector-Databases-and-ANN-Indexes.md)
- Prereq (what gets indexed): [03 Embedding Models for Retrieval](../03-Embedding-Models-for-Retrieval/03-Embedding-Models-for-Retrieval.md) · [01 RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md)
- The math under IVF (the partitioning): [04. Unsupervised Learning — K-Means Clustering](../../04.%20Unsupervised_Learning/01-K-Means-Clustering/01-K-Means-Clustering.md)
- Foundations (the geometry of "near"): [AI-ML-intuition 1.07–1.08 Euclidean vs Cosine](../../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md) · [1.06 Vector Similarities](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md)
- Next in this domain (sharpen what the index returns): [05 Hybrid Search (BM25 + Dense)](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md) · [06 Re-ranking with Cross-Encoders](../06-Re-ranking-Cross-Encoders/06-Re-ranking-Cross-Encoders.md)
