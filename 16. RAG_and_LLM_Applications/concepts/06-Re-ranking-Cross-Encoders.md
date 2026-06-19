---
id: "16-rag-and-llm-apps/re-ranking-cross-encoders"
topic: "Re-ranking (cross-encoders)"
parent: "16-rag-and-llm-apps"
level: intermediate
prereqs: ["embedding-models-for-retrieval", "hybrid-search"]
interview_frequency: high
updated: 2026-06-20
---

# Re-ranking — Cross-Encoders
> First-stage retrieval (bi-encoder / BM25) is fast but coarse: it scores query and document
> *separately*. A **cross-encoder reranker** reads the query and each candidate *together* with full
> attention, producing a far more accurate relevance score — so you retrieve top-100 cheaply, then
> rerank to top-5. The classic **two-stage retrieval** that lifts RAG precision the most for the least
> effort.

**Why it matters:** the canonical "how do you improve retrieval precision?" answer. Interviewers
probe bi-encoder vs cross-encoder (why you can't just cross-encode everything — it's O(N) forward
passes), where the reranker sits in the pipeline, latency/cost trade-offs, and options (Cohere/Voyage
APIs, open BGE/MiniLM cross-encoders, ColBERT-style late interaction).

**⭐ Start here — suggested path:**

1. **Get the two-stage picture** — read [Pinecone: Rerankers and Two-Stage Retrieval](https://www.pinecone.io/learn/series/rag/rerankers/). *Why a cheap recall stage + an accurate rerank stage beats either alone.*
2. **See the architecture difference** — read [Sentence-Transformers: Retrieve & Re-Rank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html). *Bi-encoder (separate) vs cross-encoder (joint) encoding, with code.*
3. **Watch it in a pipeline** — watch [Reranking with Sentence Transformers and BM25](https://www.youtube.com/watch?v=V58mPkLB95o). *Plugs a cross-encoder onto first-stage results end to end.*
4. **Use a hosted reranker** — watch [Semantic Search and Reranking with Cohere and Pinecone](https://www.youtube.com/watch?v=e7x1wJlmDjs). *How a production rerank API slots into retrieval.*
5. **Read the source** — skim [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085). *The paper that established BERT cross-encoders as rerankers.*

## 🎓 Courses (free)
- [Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI × LlamaIndex** — covers sentence-window + rerank as a core retrieval upgrade.
- [Sentence Transformers — Cross-Encoders docs](https://www.sbert.net/examples/cross_encoder/applications/README.html) — **UKP / Nils Reimers** — the free reference for training and applying cross-encoder rerankers.

## 🎥 Videos
- [Advanced RAG 03 — Reranking with Sentence Transformers and BM25](https://www.youtube.com/watch?v=V58mPkLB95o) — **Sunny Savita** — wiring a cross-encoder reranker onto first-stage retrieval.
- [Advanced RAG 04 — Reranking with Cross Encoders and Cohere API](https://www.youtube.com/watch?v=ZFbaA9eM0uo) — **Sunny Savita** — open cross-encoder vs hosted Cohere Rerank, in code.
- [Semantic Search and Reranking with Cohere and Pinecone](https://www.youtube.com/watch?v=e7x1wJlmDjs) — **Pinecone** — a full two-stage retrieve-then-rerank pipeline.
- [Supercharging Semantic Search with Pinecone and Cohere](https://www.youtube.com/watch?v=e2g5ya4ZFro) — **Pinecone** — how reranking refines vector-search results in practice.

## 📄 Key Papers
- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) — **Nogueira & Cho (2019)** — the foundational BERT cross-encoder reranker.
- [ColBERT: Efficient Passage Search via Late Interaction over BERT](https://arxiv.org/abs/2004.12832) — **Khattab & Zaharia (2020)** — token-level late interaction; the middle ground between bi- and cross-encoders.
- [BEIR: Zero-Shot Evaluation of IR Models](https://arxiv.org/abs/2104.08663) — **Thakur et al. (2021)** — benchmark showing rerankers' gains across diverse retrieval tasks.

## 📰 Articles / Blogs (free, no paywall)
- [Rerankers and Two-Stage Retrieval](https://www.pinecone.io/learn/series/rag/rerankers/) — **Pinecone** — the canonical explainer of why and how reranking works.
- [Retrieve & Re-Rank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) — **Sentence-Transformers** — the bi-encoder + cross-encoder pattern with runnable code.
- [Search Reranking with Cross-Encoders](https://developers.openai.com/cookbook/examples/search_reranking_with_cross-encoders) — **OpenAI Cookbook** — a worked example of reranking retrieved candidates.
- [Introducing Rerank](https://cohere.com/blog/rerank) — **Cohere** — what a production rerank endpoint does and when to use it.

## 📚 Books (free, with chapters)
- [Introduction to Information Retrieval — **Ch. 15 "Support vector machines & machine-learned ranking"**](https://nlp.stanford.edu/IR-book/html/htmledition/support-vector-machines-and-machine-learning-on-documents-1.html) — **Manning, Raghavan & Schütze** — the learning-to-rank foundations behind rerankers, free online.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.06 Vector Similarities — Scaled Dot-Product](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md)
- Prereqs: [03 Embedding Models for Retrieval](03-Embedding-Models-for-Retrieval.md) · [05 Hybrid Search](05-Hybrid-Search-BM25-and-Dense.md) · Next: [07 Query Transformation](07-Query-Transformation-HyDE-Multi-Query.md)
- Related domain: [06. NLP — Information Retrieval & Semantic Search](../../06.%20NLP/concepts/16-Information-Retrieval-and-Semantic-Search.md) · [05. Deep Learning — Attention](../../05.%20Deep_Learning/concepts/README.md)
