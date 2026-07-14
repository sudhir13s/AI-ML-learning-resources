---
id: "15-rag-and-llm-apps/query-transformation-hyde-multi-query/references"
topic: "Query Transformation (HyDE & Multi-Query) — References"
parent: "15-rag-and-llm-apps/query-transformation-hyde-multi-query"
type: references
updated: 2026-07-01
---

# Query Transformation (HyDE & Multi-Query) — references and further reading

> Companion link library for **[Query Transformation (HyDE & Multi-Query)](07-Query-Transformation-HyDE-Multi-Query.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is free/open (no paywall) and chosen for depth on *this* topic — the question↔document asymmetry, HyDE, Multi-Query / RAG-Fusion, and the fusion that ties them together.

**Start here — suggested path**:
1. **Feel the problem** — read [the HyDE paper](https://arxiv.org/abs/2212.10496) (**Gao et al. 2022**), abstract + §1–3. *Why a hypothetical answer is a better retrieval probe than the raw question — the core idea, stated cleanly.*
2. **See it as query expansion** — read [query2doc](https://arxiv.org/abs/2303.07678) (**Wang et al. 2023**). *The append-the-pseudo-document variant, with concrete BM25 gains — sharpens the "replace vs append" distinction.*
3. **Build Multi-Query** — read [LangChain's MultiQueryRetriever how-to](https://python.langchain.com/docs/how_to/MultiQueryRetriever/) (**LangChain**). *Generate N reformulations, retrieve each, return the unique union — the pattern the page builds by hand.*
4. **Add the fusion** — read the [RAG-Fusion write-up](https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1/) (**Adrian Raudaschl / TDS**). *Multi-Query + RRF, the pattern most production stacks default to.*
5. **Wire HyDE in a framework** — read [LlamaIndex's HyDE query-transform docs](https://developers.llamaindex.ai/python/framework/optimizing/advanced_retrieval/query_transformations/). *`HyDEQueryTransform` + `TransformQueryEngine`, the library one-liner and its knobs.*

**Videos**:
- [Advanced RAG — Query Transformations (HyDE, Multi-Query, RAG-Fusion)](https://www.youtube.com/watch?v=sVcwVQRHIc8) — **LangChain (Lance Martin)** — the "RAG from Scratch" episode that walks HyDE, multi-query, and decomposition on real code; the single best overview of this chapter's topic.
- [HyDE — Hypothetical Document Embeddings, explained](https://www.youtube.com/watch?v=v_BnBEubv58) — **Connor Shorten (Weaviate)** — the paper's idea in plain terms, with the "wrong hypothetical still works" intuition.
- [Multi-Query & RAG-Fusion — better retrieval by rewriting the query](https://www.youtube.com/watch?v=77qELPbNgxA) — **LangChain (Lance Martin)** — the multi-query + reciprocal-rank-fusion pattern, built step by step.
- [RAG-Fusion — how it works and why it beats plain RAG](https://www.youtube.com/watch?v=GchC5WxeXGc) — **Prompt Engineering** — the fan-out-then-fuse pattern with a worked example.
- [Query Transformations for RAG (step-back, decomposition, HyDE)](https://www.youtube.com/watch?v=miDqLc4-nyc) — **LlamaIndex / community** — the wider family of query rewrites and when each helps.

**Interactive & visual**:
- [Nearest-neighbour / embedding explorer (TensorFlow Embedding Projector)](https://projector.tensorflow.org/) — **Google** — project real embeddings to 2D/3D and *see* the question↔answer gap the page measures, on your own text.
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — **Hugging Face** — the retrieval benchmark to sanity-check any encoder you'd pair with HyDE/Multi-Query.

**Courses (free)**:
- [LangChain — RAG from Scratch (query translation)](https://github.com/langchain-ai/rag-from-scratch) — **LangChain** — the notebooks behind the videos above: multi-query, RAG-fusion, decomposition, step-back, and HyDE, each runnable.
- [DeepLearning.AI — Building and Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI + LlamaIndex (free short course)** — query transformation inside a full advanced-RAG pipeline, with evaluation.

**Articles / blogs (free, no paywall)**:
- [RAG-Fusion: the next frontier of search](https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1/) — **Adrian Raudaschl (Towards Data Science)** — the article that popularized Multi-Query + RRF, with the reasoning and code.
- [How to use the MultiQueryRetriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/) — **LangChain docs** — the exact API the page cites (default 3 reformulations, unique union), with a runnable example.
- [Query Transformations](https://blog.langchain.dev/query-transformations/) — **LangChain blog (Lance Martin)** — a taxonomy of query rewrites (rewrite-retrieve-read, multi-query, HyDE, decomposition, step-back) and when each applies.
- [Advanced Retrieval — Query Transformations (HyDE)](https://developers.llamaindex.ai/python/framework/optimizing/advanced_retrieval/query_transformations/) — **LlamaIndex docs** — `HyDEQueryTransform` + `TransformQueryEngine`, including the "HyDE can produce nonsense" caveat the pitfalls section echoes.
- [Advanced RAG: Query Expansion](https://www.pinecone.io/learn/query-expansion/) — **Pinecone** — HyDE and generated-query expansion framed as retrieval-side upgrades, vendor-neutral.

**Key papers**:
- [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)](https://arxiv.org/abs/2212.10496) — **Gao, Ma, Lin & Callan (2022)** — the HyDE paper: generate a hypothetical document, encode it, retrieve; the encoder's "dense bottleneck filters out the incorrect details." The primary source for this page's HyDE section.
- [Query2doc: Query Expansion with Large Language Models](https://arxiv.org/abs/2303.07678) — **Wang, Yang & Wei (2023)** — the append-the-pseudo-document variant; +3–15% for BM25 on MS-MARCO / TREC DL, and gains for dense retrievers, without fine-tuning.
- [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://doi.org/10.1145/1571941.1572114) — **Cormack, Clarke & Büttcher (SIGIR 2009)** — the RRF paper: the $k=60$ default and the "high in any list wins" rank fusion the Multi-Query section reuses. ([free PDF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf))
- [Query Rewriting for Retrieval-Augmented Large Language Models (Rewrite-Retrieve-Read)](https://arxiv.org/abs/2305.14283) — **Ma, Gong, He, Zhao & Duan (2023)** — a trainable query-rewriter placed before retrieval; the "learn the transform" end of the design space.
- [Take a Step Back: Evoking Reasoning via Abstraction (Step-Back Prompting)](https://arxiv.org/abs/2310.06117) — **Zheng et al. (2023, Google DeepMind)** — the step-back transform (ask a more general question first), a sibling query rewrite for reasoning-heavy retrieval.
- [Dense Passage Retrieval for Open-Domain QA (DPR)](https://arxiv.org/abs/2004.04906) — **Karpukhin et al. (2020)** — the bi-encoder retrieval this page transforms the input to; grounds the question↔document asymmetry (separate query/passage encoders).

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 14 "Question Answering & Information Retrieval"](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — dense retrieval, the query/passage encoders, and the retrieval metrics (recall@k, MRR) this page measures against.

**In this platform**:
- Concept page (full explanation): [Query Transformation (HyDE & Multi-Query)](07-Query-Transformation-HyDE-Multi-Query.md)
- Foundations this builds on: [RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · [Embedding Models for Retrieval](../03-Embedding-Models-for-Retrieval/03-Embedding-Models-for-Retrieval.md) · [Hybrid Search (BM25 + Dense) — the RRF this page reuses](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md)
- Puts it to work: [Re-ranking with Cross-Encoders (the backstop after transformation)](../06-Re-ranking-Cross-Encoders/06-Re-ranking-Cross-Encoders.md) · [Advanced RAG (Parent-Doc, Fusion, Self-RAG)](../08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG/08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.md) · [Agentic RAG (query decomposition, multi-hop)](../10-Agentic-RAG/10-Agentic-RAG.md)
- Measure it: [RAG Evaluation (recall@k, MRR, and the metrics this page reports)](../11-RAG-Evaluation/11-RAG-Evaluation.md)
