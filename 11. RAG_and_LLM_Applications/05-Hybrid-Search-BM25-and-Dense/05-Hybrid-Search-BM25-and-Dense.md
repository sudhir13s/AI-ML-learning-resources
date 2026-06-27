---
id: "11-rag-and-llm-apps/hybrid-search"
topic: "Hybrid Search (BM25 + dense)"
parent: "11-rag-and-llm-apps"
level: intermediate
prereqs: ["vector-databases-ann-indexes", "tf-idf", "embedding-models-for-retrieval"]
interview_frequency: high
updated: 2026-06-20
---

# Hybrid Search — BM25 + Dense
> Dense vector search nails *meaning* but fumbles exact tokens (IDs, error codes, rare names); sparse
> lexical search (BM25 / SPLADE) nails *keywords* but misses paraphrase. **Hybrid search** runs both
> and fuses the rankings — usually with **Reciprocal Rank Fusion (RRF)** — to beat either alone. The
> single most reliable retrieval-quality upgrade in production RAG.

**Why it matters:** a frequent practical question — *why isn't dense retrieval enough?* You'll be
asked how BM25 scores (TF saturation + length normalization), what learned sparse (SPLADE) adds, why
RRF fuses *ranks* instead of incompatible scores, and how to weight/tune the two signals for a given
corpus.

**⭐ Start here — suggested path:**

1. **Master the sparse baseline** — watch [A No-Nonsense Intro to BM25](https://www.youtube.com/watch?v=TW9vHU1GpU4). *BM25 is the keyword half of hybrid; understand TF saturation and length normalization first.*
2. **See why you need both** — read [Weaviate: Hybrid Search Explained](https://weaviate.io/blog/hybrid-search-explained). *Where dense fails, where sparse fails, and how fusion fixes both.*
3. **Understand the fusion** — read [Pinecone: Getting Started with Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/). *Combining sparse + dense vectors and the alpha/weighting knob.*
4. **Add learned sparse** — watch [SPLADE + Sentence Transformers (medical search)](https://www.youtube.com/watch?v=a3-RM_u5YoU). *SPLADE = sparse vectors that beat BM25 by fixing vocabulary mismatch.*
5. **Build it** — follow [Qdrant: Hybrid Search with the Query API](https://qdrant.tech/articles/hybrid-search/). *A production-shaped hybrid pipeline with RRF fusion end to end.*

## 🎓 Courses (free)
- [Faiss: The Missing Manual — sparse & hybrid sections](https://www.pinecone.io/learn/series/faiss/) — **Pinecone (James Briggs)** — covers sparse vectors and combining them with dense retrieval.
- [RAG from Scratch — Routing & Retrieval](https://github.com/langchain-ai/rag-from-scratch) — **LangChain (Lance Martin)** — situates hybrid/keyword+vector retrieval within the broader RAG pipeline, with code.

## 🎥 Videos
- [A No-Nonsense Intro to BM25](https://www.youtube.com/watch?v=TW9vHU1GpU4) — **Abhishek Thakur** — the sparse ranking function behind hybrid search, explained cleanly.
- [What is BM25 and How Does it Work?](https://www.youtube.com/watch?v=hiJcEaiuw_E) — **FSG Book** — short, visual walkthrough of the BM25 scoring formula.
- [Medical Search Engine with SPLADE + Sentence Transformers](https://www.youtube.com/watch?v=a3-RM_u5YoU) — **James Briggs** — learned sparse (SPLADE) combined with dense embeddings in a real hybrid pipeline.
- [Exploring Pinecone's Sparse-Dense Index](https://www.youtube.com/watch?v=EZfONAne55M) — **Pinecone** — how a vector DB stores and queries sparse + dense vectors together.

## 📄 Key Papers
- [SPLADE: Sparse Lexical and Expansion Model for First-Stage Ranking](https://arxiv.org/abs/2107.05720) — **Formal et al. (2021)** — learned sparse retrieval that beats BM25 while staying interpretable.
- [Reciprocal Rank Fusion Outperforms Condorcet and Learning-to-Rank](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — **Cormack et al. (2009)** — the one-line fusion algorithm behind modern hybrid search.
- [BEIR: A Heterogeneous Benchmark for Zero-Shot IR](https://arxiv.org/abs/2104.08663) — **Thakur et al. (2021)** — shows BM25 is a tough baseline and motivates combining lexical + dense.

## 📰 Articles / Blogs (free, no paywall)
- [Hybrid Search Explained](https://weaviate.io/blog/hybrid-search-explained) — **Weaviate** — clear account of why hybrid wins and how RRF fuses results (RRF is Weaviate's default).
- [Getting Started with Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/) — **Pinecone** — combining sparse + dense vectors with a tunable weighting.
- [Hybrid Search with Qdrant's Query API](https://qdrant.tech/articles/hybrid-search/) — **Qdrant** — production patterns for dense+sparse fusion.
- [SPLADE for Sparse Vector Search Explained](https://www.pinecone.io/learn/splade/) — **Pinecone** — how learned sparse vectors improve over BM25.

## 📚 Books (free, with chapters)
- [Introduction to Information Retrieval — **Ch. 6 "Scoring, term weighting & the vector space model"**](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html) — **Manning, Raghavan & Schütze** — the TF-IDF/BM25 foundations behind the sparse half of hybrid, free online.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.06 Vector Similarities](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md) · [1.17 BoW & TF-IDF](../../../AI-ML-intuition/Module_1_Representation/1.17_BoW_and_TF-IDF.md)
- Prereqs: [04 Vector Databases & ANN Indexes](04-Vector-Databases-and-ANN-Indexes.md) · Next: [06 Re-ranking (cross-encoders)](06-Re-ranking-Cross-Encoders.md)
- Related domain: [06. NLP — Bag-of-Words & TF-IDF](../../06.%20NLP/03-Bag-of-Words-and-TF-IDF/03-Bag-of-Words-and-TF-IDF.md) · [06. NLP — Information Retrieval & Semantic Search](../../06.%20NLP/16-Information-Retrieval-and-Semantic-Search/16-Information-Retrieval-and-Semantic-Search.md)
