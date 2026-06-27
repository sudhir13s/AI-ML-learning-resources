---
id: "11-rag-and-llm-apps/hybrid-search-bm25-and-dense/references"
topic: "Hybrid Search (BM25 + Dense) — References"
parent: "11-rag-and-llm-apps/hybrid-search-bm25-and-dense"
type: references
updated: 2026-06-27
---

# Hybrid Search (BM25 + Dense) — references and further reading

> Companion link library for **[Hybrid Search (BM25 + Dense)](05-Hybrid-Search-BM25-and-Dense.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is a free, no-paywall link from a primary author or a recognized deep explainer — chosen for depth on *this* topic (combining lexical and dense retrieval, and the fusion that does it), not popularity.

**Start here — suggested path**:
1. **Master the sparse baseline** — watch [A No-Nonsense Intro to BM25](https://www.youtube.com/watch?v=TW9vHU1GpU4) (**Abhishek Thakur**). *BM25 is the keyword half of hybrid; understand tf-saturation and length normalization first.*
2. **See why you need both** — read [Hybrid Search Explained](https://weaviate.io/blog/hybrid-search-explained) (**Weaviate**). *Where dense fails, where sparse fails, and how fusion fixes both — with the `alpha` dial.*
3. **Understand the fusion** — read [Getting Started with Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/) (**Pinecone**). *Combining sparse + dense vectors and the weighting knob.*
4. **Read the fusion source** — skim [Reciprocal Rank Fusion (Cormack et al. 2009)](https://plg.uwaterloo.ca/~gvcormack/cormacksigir09-rrf.pdf). *The one-line, rank-based fusion behind every production hybrid stack — and the $k=60$ default.*
5. **Get the BM25 math right** — read [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) (**Robertson & Zaragoza**), §3. *The full BM25 derivation — IDF, saturation ($k_1$), length-norm ($b$).*

**Videos**:
- [A No-Nonsense Intro to BM25](https://www.youtube.com/watch?v=TW9vHU1GpU4) — **Abhishek Thakur** — the sparse ranking function behind hybrid, with tf-saturation and length normalization explained cleanly.
- [What is BM25 and How Does it Work?](https://www.youtube.com/watch?v=hiJcEaiuw_E) — **FSG** — short, visual walkthrough of the BM25 scoring formula, term by term.
- [Medical Search Engine with SPLADE + Sentence Transformers](https://www.youtube.com/watch?v=a3-RM_u5YoU) — **James Briggs** — learned sparse (SPLADE) combined with dense embeddings in a real hybrid pipeline; the natural next step past BM25.
- [Exploring Pinecone's Sparse-Dense Index](https://www.youtube.com/watch?v=EZfONAne55M) — **Pinecone** — how a vector DB stores and queries sparse + dense vectors together in one index.

**Interactive & visual**:
- [BM25 interactive demo / explainer](https://www.elastic.co/search-labs/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) — **Elastic (Search Labs)** — the BM25 variables ($k_1$, $b$) walked through with worked examples on a real index; the clearest hands-on for the saturation/length-norm knobs.
- [Reciprocal Rank Fusion in Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) — **Elastic** — the production RRF reference with the exact formula, the `rank_constant` (default 60) and `rank_window_size` parameters you'd actually set.

**Courses (free)**:
- [Faiss: The Missing Manual — sparse & hybrid sections](https://www.pinecone.io/learn/series/faiss/) — **Pinecone (James Briggs)** — covers sparse vectors and combining them with dense retrieval, with code.
- [RAG from Scratch — Routing & Retrieval](https://github.com/langchain-ai/rag-from-scratch) — **LangChain (Lance Martin)** — situates hybrid (keyword + vector) retrieval within the broader RAG pipeline, runnable.

**Articles / blogs (free, no paywall)**:
- [Hybrid Search Explained](https://weaviate.io/blog/hybrid-search-explained) — **Weaviate** — clear account of why hybrid wins, the `alpha` dial (0 = BM25, 1 = vector), and the two fusion methods (relativeScoreFusion vs rankedFusion).
- [Getting Started with Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/) — **Pinecone** — combining sparse + dense vectors with a tunable weighting; the sparse-dense index idea.
- [Hybrid Search with Qdrant's Query API](https://qdrant.tech/articles/hybrid-search/) — **Qdrant** — production patterns for dense+sparse fusion (RRF and DBSF) in one round trip.
- [SPLADE for Sparse Vector Search Explained](https://www.pinecone.io/learn/splade/) — **Pinecone** — how learned sparse vectors fix the vocabulary-mismatch limitation of BM25, the frontier of the lexical lens.
- [Practical BM25 — Part 2: The BM25 Algorithm and its Variables](https://www.elastic.co/search-labs/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) — **Elastic** — the most practical term-by-term tour of BM25, $k_1$, and $b$ with real numbers.

**Key papers / primary sources**:
- [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — **Robertson & Zaragoza (FnTIR 2009)** — the canonical BM25 derivation; §3 gives the saturation-and-length-normalization scoring function and the probabilistic IDF the page's formula is taken from.
- [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormack/cormacksigir09-rrf.pdf) — **Cormack, Clarke & Büttcher (SIGIR 2009)** — introduces RRF (Eq. 1) and the $k=60$ constant; the source for the rank-fusion formula on the page.
- [An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934) — **Bruch, Gai & Ingber (ACM TOIS 2023)** — analyzes convex-combination (weighted-sum) fusion of normalized lexical + dense scores; the source for the min-max-normalize-then-weight formula and the "tuned convex combination can beat RRF" claim.
- [SPLADE: Sparse Lexical and Expansion Model for First-Stage Ranking](https://arxiv.org/abs/2107.05720) — **Formal et al. (2021)** — learned sparse retrieval that beats BM25 while staying interpretable; the modern evolution of the lexical lens.
- [Dense Passage Retrieval for Open-Domain QA (DPR)](https://arxiv.org/abs/2004.04906) — **Karpukhin et al. (2020)** — the dense dual-encoder that hybrid pairs with BM25; established that learned dense retrieval beats BM25 on many tasks (but not all — motivating fusion).
- [BEIR: A Heterogeneous Benchmark for Zero-Shot IR](https://arxiv.org/abs/2104.08663) — **Thakur et al. (2021)** — shows BM25 is a tough zero-shot baseline that dense models don't uniformly beat, the empirical case for keeping the lexical lens.
- [A Replication Study of Dense Passage Retriever](https://arxiv.org/abs/2104.05740) — **Ma, Sun, Pradeep & Lin (2021)** — shows the original DPR work under-reported the BM25 baseline, and that **dense–sparse hybrid** retrieval is stronger than either alone — the empirical case for fusing the two lenses.

**Books (free, with chapters)**:
- [Introduction to Information Retrieval — Ch. 6 "Scoring, term weighting & the vector space model"](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html) — **Manning, Raghavan & Schütze** — the TF-IDF/BM25 foundations behind the sparse half of hybrid, free online.
- [Speech and Language Processing, 3rd ed. — Ch. 14 "Question Answering and Information Retrieval"](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — BM25, dense retrieval, and how they combine, in one reference chapter (free PDF).

**In this platform**:
- Concept page (full explanation): [Hybrid Search (BM25 + Dense)](05-Hybrid-Search-BM25-and-Dense.md)
- Prereqs (the two lenses this fuses): [01 RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · [03 Embedding Models for Retrieval](../03-Embedding-Models-for-Retrieval/03-Embedding-Models-for-Retrieval.md) · [04 Vector Databases & ANN Indexes](../04-Vector-Databases-and-ANN-Indexes/04-Vector-Databases-and-ANN-Indexes.md)
- Foundations (the *why* behind the math): [AI-ML-intuition 1.06 Vector Similarities — Scaled Dot-Product](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md) · [1.17 BoW & TF-IDF](../../../AI-ML-intuition/Module_1_Representation/1.17_BoW_and_TF-IDF.md) · [06. NLP — Bag-of-Words & TF-IDF](../../06.%20NLP/03-Bag-of-Words-and-TF-IDF/03-Bag-of-Words-and-TF-IDF.md) · [06. NLP — Information Retrieval & Semantic Search](../../06.%20NLP/16-Information-Retrieval-and-Semantic-Search/16-Information-Retrieval-and-Semantic-Search.md)
- Next in this domain (sharpen the fused candidates): [06 Re-ranking with Cross-Encoders](../06-Re-ranking-Cross-Encoders/06-Re-ranking-Cross-Encoders.md) · [07 Query Transformation (HyDE, Multi-Query)](../07-Query-Transformation-HyDE-Multi-Query/07-Query-Transformation-HyDE-Multi-Query.md)
