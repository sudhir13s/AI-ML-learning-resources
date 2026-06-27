---
id: "11-rag-and-llm-apps/re-ranking-cross-encoders/references"
topic: "Re-ranking (Cross-Encoders) — References"
parent: "11-rag-and-llm-apps/re-ranking-cross-encoders"
type: references
updated: 2026-06-27
---

# Re-ranking with Cross-Encoders — references and further reading

> Companion link library for **[Re-ranking with Cross-Encoders](06-Re-ranking-Cross-Encoders.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is a free, no-paywall link from a primary author or a recognized deep explainer — chosen for depth on *this* topic (two-stage retrieve-then-rerank and the cross-encoder that does the precision stage), not popularity.

**Start here — suggested path**:
1. **Get the two-stage picture** — read [Rerankers and Two-Stage Retrieval](https://www.pinecone.io/learn/series/rag/rerankers/) (**Pinecone**). *Why a cheap recall stage + an accurate rerank stage beats either alone.*
2. **See the architecture difference** — read [Retrieve & Re-Rank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) (**Sentence-Transformers**). *Bi-encoder (independent) vs cross-encoder (joint) encoding, with runnable code.*
3. **Watch it in a pipeline** — watch [Reranking with Sentence Transformers and BM25](https://www.youtube.com/watch?v=V58mPkLB95o) (**Sunny Savita**). *Plugs a cross-encoder onto first-stage results end to end.*
4. **Read the cross-encoder source** — skim [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) (**Nogueira & Cho 2019**). *The paper that established BERT cross-encoders as re-rankers — the score on the page.*
5. **Understand why two stages** — skim [Sentence-BERT](https://arxiv.org/abs/1908.10084) (**Reimers & Gurevych 2019**), §1–2. *The cost argument (a forward pass per pair) that forces bi-encoder retrieval + cross-encoder re-ranking.*

**Videos**:
- [Advanced RAG — Reranking with Sentence Transformers and BM25](https://www.youtube.com/watch?v=V58mPkLB95o) — **Sunny Savita** — wiring a cross-encoder re-ranker onto first-stage retrieval, in code.
- [Advanced RAG — Reranking with Cross-Encoders and the Cohere API](https://www.youtube.com/watch?v=ZFbaA9eM0uo) — **Sunny Savita** — open cross-encoder vs hosted Cohere Rerank, side by side.
- [Semantic Search and Reranking with Cohere and Pinecone](https://www.youtube.com/watch?v=e7x1wJlmDjs) — **Pinecone** — a full two-stage retrieve-then-rerank pipeline.
- [Supercharging Semantic Search with Pinecone and Cohere](https://www.youtube.com/watch?v=e2g5ya4ZFro) — **Pinecone** — how reranking refines vector-search results in practice.

**Interactive & visual**:
- [Retrieve & Re-Rank — Sentence-Transformers docs (with code)](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) — **Nils Reimers / UKP** — the canonical bi-encoder-retrieve + cross-encoder-rerank example you can run and modify.
- [Pretrained Cross-Encoders (model list + accuracy/speed table)](https://www.sbert.net/docs/cross_encoder/pretrained_models.html) — **Sentence-Transformers** — the actual `ms-marco-MiniLM` re-ranker zoo with their accuracy/latency tradeoffs, for picking a model.

**Courses (free)**:
- [Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI × LlamaIndex** — covers sentence-window retrieval + re-ranking as a core precision upgrade, with evaluation.
- [Sentence Transformers — Cross-Encoder docs (train & apply)](https://www.sbert.net/examples/cross_encoder/applications/README.html) — **UKP / Nils Reimers** — the free reference for training and applying cross-encoder re-rankers.

**Articles / blogs (free, no paywall)**:
- [Rerankers and Two-Stage Retrieval](https://www.pinecone.io/learn/series/rag/rerankers/) — **Pinecone** — the canonical explainer of why and how reranking works, with the two-stage funnel.
- [Retrieve & Re-Rank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) — **Sentence-Transformers** — the bi-encoder + cross-encoder pattern with code.
- [Search Reranking with Cross-Encoders](https://developers.openai.com/cookbook/examples/search_reranking_with_cross-encoders) — **OpenAI Cookbook** — a worked example of re-ranking retrieved candidates.
- [Introducing Rerank 3](https://cohere.com/blog/rerank-3) — **Cohere** — what a production rerank endpoint does, its inputs/outputs, and when to use it.
- [BGE-Reranker (FlagEmbedding) — model card & usage](https://huggingface.co/BAAI/bge-reranker-large) — **BAAI** — the leading open cross-encoder re-ranker family (`bge-reranker-base/large/v2-m3`), with usage and the relevance-score semantics.

**Key papers / primary sources**:
- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) — **Nogueira & Cho (2019)** — feeds `[CLS] query [SEP] passage` through BERT with a linear relevance head; the cross-encoder re-ranker score on the page is taken from §3.
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — **Reimers & Gurevych (2019)** — the bi-encoder (SBERT) and the cost argument (a forward pass per pair) that motivates retrieve-then-rerank; the source for the cost-asymmetry section.
- [Cumulated Gain-based Evaluation of IR Techniques](https://dl.acm.org/doi/10.1145/582415.582418) — **Järvelin & Kekäläinen (ACM TOIS 2002)** — introduces DCG / nDCG with the $\log_2(i+1)$ rank discount; the ranking metric defined and used on the page.
- [ColBERT: Efficient Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) — **Khattab & Zaharia (2020)** — per-token embeddings + late MaxSim interaction; the precomputable middle ground between bi- and cross-encoders.
- [MS MARCO: A Human-Generated Machine Reading Comprehension Dataset](https://arxiv.org/abs/1611.09268) — **Bajaj et al. (2016)** — the passage-ranking dataset the `ms-marco-MiniLM` re-rankers (and most open cross-encoders) are trained on.
- [BEIR: A Heterogeneous Benchmark for Zero-Shot Evaluation of IR Models](https://arxiv.org/abs/2104.08663) — **Thakur et al. (2021)** — shows cross-encoder re-ranking gains (and limits) across diverse retrieval tasks; the empirical case for re-ranking.

**Books (free, with chapters)**:
- [Introduction to Information Retrieval — Ch. 8 "Evaluation in information retrieval"](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html) — **Manning, Raghavan & Schütze** — the IR evaluation foundations (precision, MAP, the lineage of nDCG) behind measuring a re-ranker, free online.
- [Speech and Language Processing, 3rd ed. — Ch. 14 "Question Answering and Information Retrieval"](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — dense retrieval, re-ranking, and ranking metrics in one reference chapter (free PDF).

**In this platform**:
- Concept page (full explanation): [Re-ranking with Cross-Encoders](06-Re-ranking-Cross-Encoders.md)
- Prereqs (the first stage this re-ranks): [03 Embedding Models for Retrieval](../03-Embedding-Models-for-Retrieval/03-Embedding-Models-for-Retrieval.md) · [04 Vector Databases & ANN Indexes](../04-Vector-Databases-and-ANN-Indexes/04-Vector-Databases-and-ANN-Indexes.md) · [05 Hybrid Search (BM25 + Dense)](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md)
- Foundations (the *why* behind attention & similarity): [AI-ML-intuition 1.06 Vector Similarities — Scaled Dot-Product](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md) · [05. Deep Learning — Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md) · [06. NLP — Information Retrieval & Semantic Search](../../06.%20NLP/16-Information-Retrieval-and-Semantic-Search/16-Information-Retrieval-and-Semantic-Search.md)
- Next in this domain (raise the recall ceiling re-ranking is bounded by): [07 Query Transformation (HyDE, Multi-Query)](../07-Query-Transformation-HyDE-Multi-Query/07-Query-Transformation-HyDE-Multi-Query.md)
