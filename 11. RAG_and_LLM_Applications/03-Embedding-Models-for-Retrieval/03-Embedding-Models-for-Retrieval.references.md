---
id: "11-rag-and-llm-apps/embedding-models-for-retrieval/references"
topic: "Embedding Models for Retrieval — References"
parent: "11-rag-and-llm-apps/embedding-models-for-retrieval"
type: references
updated: 2026-06-27
---

# Embedding Models for Retrieval — references and further reading

> Companion link library for **[Embedding Models for Retrieval](03-Embedding-Models-for-Retrieval.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is a free, no-paywall link from a primary author or a recognized deep explainer — chosen for depth on *this* topic (the embedding model that defines retrieval geometry), not popularity.

**Start here — suggested path**:
1. **See what an embedding *is* for search** — watch [Intro to Sentence Embeddings with Transformers](https://www.youtube.com/watch?v=WS1uVMGhlWQ) (**James Briggs**). *Grounds the "sentence → vector → cosine similarity" pipeline you retrieve with.*
2. **Connect it to retrieval** — watch [Introduction to Semantic Search](https://www.youtube.com/watch?v=OcJZ6XWrTEA) (**Luis Serrano, Cohere**). *Why dense retrieval beats keyword search, told visually.*
3. **Get the asymmetry right** — read [Sentence-Transformers: Semantic Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html). *Symmetric vs asymmetric search and how to encode queries vs documents correctly.*
4. **Choose a model deliberately** — read [Choosing an Embedding Model](https://www.pinecone.io/learn/series/rag/embedding-models-rundown/) (**Pinecone**) + the [MTEB blog](https://huggingface.co/blog/mteb) (**Hugging Face**). *How to read the Retrieval task — not the headline average.*
5. **Read the source** — skim [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) (**Karpukhin et al. 2020**). *The dual-encoder + in-batch negatives that proved dense retrievers beat BM25.*

**Videos**:
- [Intro to Sentence Embeddings with Transformers](https://www.youtube.com/watch?v=WS1uVMGhlWQ) — **James Briggs** — from BERT token vectors to pooled sentence embeddings; the foundation of dense retrieval.
- [Introduction to Semantic Search](https://www.youtube.com/watch?v=OcJZ6XWrTEA) — **Luis Serrano (Cohere)** — clear visual case for embedding-based retrieval over keyword search.
- [Text Embeddings, Classification, and Semantic Search (with Python)](https://www.youtube.com/watch?v=sNa_uiqSlJo) — **Shaw Talebi** — code-first walkthrough of generating embeddings and building semantic search.
- [What is a Vector Database? Powering Semantic Search](https://www.youtube.com/watch?v=gl1r1XV0SLw) — **IBM Technology** — how query/document embeddings are stored and matched downstream (bridges to chapter 4).

**Interactive & visual**:
- [Embedding Projector](https://projector.tensorflow.org/) — **TensorFlow** — explore a real embedding space in 2D/3D; *see* semantically similar text cluster, the geometry retrieval relies on.
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — **Hugging Face** — the live, interactive leaderboard; filter by the **Retrieval** task and your language to choose a model.

**Courses (free)**:
- [Sentence Transformers — official training & usage docs](https://www.sbert.net/) — **Nils Reimers / UKP** — the canonical free guide to using and fine-tuning bi-encoders (pooling, normalization, losses).
- [LangChain: Chat with Your Data — Embeddings & VectorStores](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) — **DeepLearning.AI** — hands-on lesson on embedding documents and querying by similarity.

**Articles / blogs (free, no paywall)**:
- [Semantic Search (asymmetric vs symmetric)](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) — **Sentence-Transformers** — the canonical how-to for query vs document encoding and when search is asymmetric.
- [Choosing an Embedding Model](https://www.pinecone.io/learn/series/rag/embedding-models-rundown/) — **Pinecone** — practical rundown of model families and selection criteria for RAG.
- [MTEB: Massive Text Embedding Benchmark](https://huggingface.co/blog/mteb) — **Hugging Face** — what the leaderboard measures and how to read the Retrieval task.
- [New embedding models and API updates (`text-embedding-3`)](https://openai.com/index/new-embedding-models-and-api-updates/) — **OpenAI** — the source for `text-embedding-3-small` (1536) / `-large` (3072) dims and Matryoshka `dimensions` shortening.
- [Introducing Embed v3](https://cohere.com/blog/introducing-embed-v3) — **Cohere** — the embed-v3 family, input types (`search_query`/`search_document`), and the asymmetric treatment baked in.

**Key papers**:
- [Representation Learning with Contrastive Predictive Coding (InfoNCE)](https://arxiv.org/abs/1807.03748) — **van den Oord, Li & Vinyals (2018)** — introduces the **InfoNCE** loss (Eq. 4) used to train retrieval embedders contrastively; the source for the loss on the concept page.
- [Dense Passage Retrieval for Open-Domain QA (DPR)](https://arxiv.org/abs/2004.04906) — **Karpukhin et al. (2020)** — the dual encoder trained with **in-batch negatives**; proved learned dense retrieval beats BM25.
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — **Reimers & Gurevych (2019)** — siamese/triplet training + mean-pooling that makes BERT usable for fast similarity search (the bi-encoder + pooling on the page).
- [Matryoshka Representation Learning (MRL)](https://arxiv.org/abs/2205.13147) — **Kusupati et al. (2022)** — nested embeddings you can truncate; the source for the dimension-shortening (`dimensions` parameter) on the page.
- [Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)](https://arxiv.org/abs/2212.03533) — **Wang et al. (2022)** — a leading open retrieval-embedding recipe and the source of the `"query:"`/`"passage:"` asymmetric prefixes.
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — **Muennighoff et al. (2022)** — the benchmark (incl. the Retrieval task) behind the leaderboard you cite when choosing a model.

**Books (free, with chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 6 "Vector Semantics and Embeddings"](https://web.stanford.edu/~jurafsky/slp3/6.pdf) — **Jurafsky & Martin** — the reference chapter on vector semantics, cosine, and the geometry underlying retrieval embeddings (free PDF).

**In this platform**:
- Concept page (full explanation): [Embedding Models for Retrieval](03-Embedding-Models-for-Retrieval.md)
- Prereqs (the pipeline + chunks this embeds): [01 RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · [02 Document Chunking Strategies](../02-Document-Chunking-Strategies/02-Document-Chunking-Strategies.md)
- Foundations (the *why* behind vectors & similarity): [AI-ML-intuition 1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md) · [1.06 Vector Similarities — Scaled Dot-Product](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md) · [06. NLP — Sentence & Document Embeddings](../../06.%20NLP/07-Sentence-and-Document-Embeddings/07-Sentence-and-Document-Embeddings.md)
- Next in this domain (search the geometry): [04 Vector Databases & ANN Indexes](../04-Vector-Databases-and-ANN-Indexes/04-Vector-Databases-and-ANN-Indexes.md) · [05 Hybrid Search (BM25 + Dense)](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md) · [06 Re-ranking with Cross-Encoders](../06-Re-ranking-Cross-Encoders/06-Re-ranking-Cross-Encoders.md)
