---
id: "11-rag-and-llm-apps/embedding-models-for-retrieval"
topic: "Embedding Models for Retrieval"
parent: "11-rag-and-llm-apps"
level: intermediate
prereqs: ["rag-fundamentals", "sentence-embeddings", "cosine-similarity"]
interview_frequency: high
updated: 2026-06-20
---

# Embedding Models for Retrieval
> The encoder that turns chunks and queries into vectors so "close in meaning" becomes "close in
> space." For RAG you want **retrieval-tuned** embeddings (bi-encoders / dense retrievers like
> Sentence-BERT, E5, BGE, OpenAI/Cohere) — and you need to pick dimension, context length, and
> domain fit, and read the MTEB leaderboard without being misled by it.

**Why it matters:** the practical "which embedding model, and why?" question — bi-encoder vs
cross-encoder, symmetric vs asymmetric search (short query → long passage), how contrastive training
(in-batch negatives) shapes a retriever, what NDCG@10 on MTEB Retrieval actually tells you, and the
trade-offs of dimension/latency/cost when you move to production.

**⭐ Start here — suggested path:**

1. **See what an embedding *is* for search** — watch [Intro to Sentence Embeddings with Transformers](https://www.youtube.com/watch?v=WS1uVMGhlWQ). *Grounds the "sentence → vector → cosine similarity" pipeline you'll retrieve with.*
2. **Connect it to retrieval** — watch [Introduction to Semantic Search](https://www.youtube.com/watch?v=OcJZ6XWrTEA). *Why dense retrieval beats keyword search, told visually by Cohere's Luis Serrano.*
3. **Get the asymmetry right** — read [Sentence-Transformers: Semantic Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html). *Symmetric vs asymmetric search and how to encode queries vs documents correctly.*
4. **Choose a model deliberately** — read [Pinecone: Choosing an Embedding Model](https://www.pinecone.io/learn/series/rag/embedding-models-rundown/) + [HF MTEB blog](https://huggingface.co/blog/mteb). *How to read the leaderboard for the Retrieval task — not the headline average.*
5. **Read the source** — skim [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906). *The dual-encoder that proved learned dense retrievers beat BM25 for open-domain QA.*

## 🎓 Courses (free)
- [LangChain: Chat with Your Data — Embeddings & VectorStores](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) — **DeepLearning.AI** — hands-on lesson on embedding documents and querying by similarity.
- [Sentence Transformers — official training/usage docs](https://www.sbert.net/) — **Nils Reimers / UKP** — the canonical free guide to using and fine-tuning retrieval embeddings (bi-encoders).

## 🎥 Videos
- [Intro to Sentence Embeddings with Transformers](https://www.youtube.com/watch?v=WS1uVMGhlWQ) — **James Briggs** — from BERT token vectors to pooled sentence embeddings; the foundation of dense retrieval.
- [Introduction to Semantic Search](https://www.youtube.com/watch?v=OcJZ6XWrTEA) — **Luis Serrano (Cohere)** — clear visual case for embedding-based retrieval over keyword search.
- [Text Embeddings, Classification, and Semantic Search (w/ Python)](https://www.youtube.com/watch?v=sNa_uiqSlJo) — **Shaw Talebi** — code-first walkthrough of generating embeddings and building semantic search.
- [What is a Vector Database? Powering Semantic Search](https://www.youtube.com/watch?v=gl1r1XV0SLw) — **IBM Technology** — shows how query/document embeddings are stored and matched downstream.

## 📄 Key Papers
- [Dense Passage Retrieval for Open-Domain QA (DPR)](https://arxiv.org/abs/2004.04906) — **Karpukhin et al. (2020)** — the dual-encoder retriever trained with in-batch negatives.
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — **Reimers & Gurevych (2019)** — makes BERT usable for fast similarity search via siamese training.
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — **Muennighoff et al. (2022)** — the benchmark behind the leaderboard you'll cite when choosing a model.
- [Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)](https://arxiv.org/abs/2212.03533) — **Wang et al. (2022)** — a leading open retrieval-embedding recipe.

## 📰 Articles / Blogs (free, no paywall)
- [Choosing an Embedding Model](https://www.pinecone.io/learn/series/rag/embedding-models-rundown/) — **Pinecone** — practical rundown of model families and selection criteria for RAG.
- [Semantic Search (asymmetric vs symmetric)](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) — **Sentence-Transformers** — the canonical how-to for query/document encoding.
- [MTEB: Massive Text Embedding Benchmark](https://huggingface.co/blog/mteb) — **Hugging Face** — what the leaderboard measures and how to read the Retrieval task.
- [Introduction to Embeddings at Cohere](https://docs.cohere.com/docs/embeddings) — **Cohere** — clear vendor docs on embedding dimensions, input types, and similarity.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 6 "Vector Semantics and Embeddings"**](https://web.stanford.edu/~jurafsky/slp3/6.pdf) — **Jurafsky & Martin** — the reference chapter on vector semantics underlying retrieval embeddings.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md) · [1.06 Vector Similarities — Scaled Dot-Product](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md) · [1.07–1.08 Euclidean vs Cosine](../../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md)
- Prereqs: [01 RAG Fundamentals](01-RAG-Fundamentals.md) · Next: [04 Vector Databases & ANN Indexes](04-Vector-Databases-and-ANN-Indexes.md)
- Related domain: [06. NLP — Sentence & Document Embeddings](../../06.%20NLP/concepts/07-Sentence-and-Document-Embeddings.md) · [06. NLP — Word Embeddings](../../06.%20NLP/concepts/05-Word-Embeddings-Word2Vec-GloVe-FastText.md)
