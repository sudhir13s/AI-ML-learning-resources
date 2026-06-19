---
id: "16-rag-and-llm-apps/vector-databases-ann-indexes"
topic: "Vector Databases & ANN Indexes (HNSW · IVF)"
parent: "16-rag-and-llm-apps"
level: intermediate
prereqs: ["embedding-models-for-retrieval", "cosine-similarity", "clustering"]
interview_frequency: high
updated: 2026-06-20
---

# Vector Databases & ANN Indexes — HNSW · IVF
> Where the embeddings live and how you search millions of them in milliseconds. Exact nearest-neighbor
> is O(N); production systems trade a little recall for huge speed using **approximate nearest neighbor
> (ANN)** indexes — graph-based **HNSW** and cluster-based **IVF** (often with product quantization) —
> served by vector databases (FAISS, Qdrant, Weaviate, pgvector, Pinecone, Milvus, Chroma).

**Why it matters:** the systems half of RAG — *how does vector search scale?* Interviewers ask how
HNSW's layered small-world graph routes a query, how IVF partitions space with k-means and probes
`nprobe` cells, the recall/latency/memory trade-offs (`efSearch`, `M`, PQ compression), and how to
pick/operate a vector store (filtering, updates, sharding).

**⭐ Start here — suggested path:**

1. **Get the why** — watch [What is a Vector Database?](https://www.youtube.com/watch?v=gl1r1XV0SLw). *Frames why specialized ANN stores exist vs scanning every vector.*
2. **Learn the two index families** — read [Pinecone: Nearest Neighbor Indexes](https://www.pinecone.io/learn/series/faiss/vector-indexes/). *Flat → IVF → HNSW → PQ, with the trade-offs that matter in interviews.*
3. **Understand HNSW deeply** — watch [HNSW Explained](https://www.youtube.com/watch?v=77QH0Y2PYKg), then read [Pinecone: HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/). *The graph-traversal intuition plus `M`/`efSearch` knobs.*
4. **Understand IVF** — watch [Inverted File Index (IVF) Explained](https://www.youtube.com/watch?v=-vh6huY2rgE). *Voronoi cells + `nprobe`; the cluster-then-probe alternative to graphs.*
5. **Read the sources** — skim [HNSW (Malkov & Yashunin)](https://arxiv.org/abs/1603.09320) and the [FAISS paper](https://arxiv.org/abs/2401.08281). *Where the algorithms and the de-facto library come from.*

## 🎓 Courses (free)
- [Faiss: The Missing Manual](https://www.pinecone.io/learn/series/faiss/) — **Pinecone (James Briggs)** — a full free course on vector indexes (Flat, IVF, HNSW, PQ) with runnable Python.
- [Vector Search lessons (LangChain: Chat with Your Data)](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) — **DeepLearning.AI** — the VectorStores lesson connects embeddings to a working retriever.

## 🎥 Videos
- [What is a Vector Database? Powering Semantic Search](https://www.youtube.com/watch?v=gl1r1XV0SLw) — **IBM Technology** — clean conceptual intro to vector DBs and ANN.
- [Vector Database Search — HNSW Explained](https://www.youtube.com/watch?v=77QH0Y2PYKg) — **DataMListic** — the layered small-world graph and greedy search, visually.
- [AI Search with HNSW](https://www.youtube.com/watch?v=7XLRCpUmiaQ) — **ObjectBox** — HNSW construction and query, end to end.
- [Inverted File Index (IVF) Explained](https://www.youtube.com/watch?v=-vh6huY2rgE) — **TensorTeach** — k-means partitioning, Voronoi cells, and `nprobe` for the IVF family.

## 📄 Key Papers
- [Efficient and Robust ANN Search using HNSW Graphs](https://arxiv.org/abs/1603.09320) — **Malkov & Yashunin (2016)** — the HNSW algorithm itself.
- [The FAISS Library](https://arxiv.org/abs/2401.08281) — **Douze et al. (2024)** — the design of the most-used similarity-search library (IVF, PQ, HNSW).
- [Product Quantization for Nearest Neighbor Search](https://inria.hal.science/inria-00514462v2/document) — **Jégou et al. (2011)** — the compression behind billion-scale vector search.

## 📰 Articles / Blogs (free, no paywall)
- [Nearest Neighbor Indexes for Similarity Search](https://www.pinecone.io/learn/series/faiss/vector-indexes/) — **Pinecone** — the canonical Flat→IVF→HNSW→PQ rundown with trade-offs.
- [Hierarchical Navigable Small Worlds (HNSW)](https://www.pinecone.io/learn/series/faiss/hnsw/) — **Pinecone** — the deep dive on HNSW parameters and behavior.
- [Faiss indexes (wiki)](https://github.com/facebookresearch/faiss/wiki) — **Meta FAISS** — authoritative reference for choosing and configuring indexes.
- [Vector Search Explained](https://weaviate.io/blog/vector-search-explained) — **Weaviate** — how a vector DB combines ANN with filtering and updates in production.

## 📚 Books (free, with chapters)
- [Introduction to Information Retrieval — **Ch. 6–7 (scoring, vector space model, efficient ranking)**](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html) — **Manning, Raghavan & Schütze** — the IR foundations of similarity scoring and index efficiency, free online.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.07–1.08 Euclidean vs Cosine](../../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md) · [1.06 Vector Similarities](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md)
- ANN/clustering math (the geometry): [04. Unsupervised Learning — K-Means](../../04.%20Unsupervised_Learning/concepts/01-K-Means-Clustering.md) (the partitioning behind IVF)
- Prereqs: [03 Embedding Models for Retrieval](03-Embedding-Models-for-Retrieval.md) · Next: [05 Hybrid Search](05-Hybrid-Search-BM25-and-Dense.md)
