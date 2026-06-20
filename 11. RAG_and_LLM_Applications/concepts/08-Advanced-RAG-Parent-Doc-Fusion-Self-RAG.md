---
id: "11-rag-and-llm-apps/advanced-rag"
topic: "Advanced RAG (parent-doc · fusion · self-RAG)"
parent: "11-rag-and-llm-apps"
level: advanced
prereqs: ["query-transformation", "re-ranking-cross-encoders", "rag-fundamentals"]
interview_frequency: medium
updated: 2026-06-20
---

# Advanced RAG — Parent-Doc · Fusion · Self-RAG
> Once naive "embed-chunk-retrieve-stuff" plateaus, you add structure and feedback: **small-to-big /
> parent-document** retrieval (embed precise chunks, return their richer parent), **RAPTOR** (recursive
> summary trees for multi-level retrieval), and **self-reflective** loops — **Self-RAG** (the model
> decides when to retrieve and grades its own output) and **CRAG** (a retrieval evaluator triggers
> correction or web search). The patterns that turn a demo into a reliable system.

**Why it matters:** the "your basic RAG is failing — what do you do?" interview. You'll be asked to
diagnose precision-vs-context tension (parent-document fixes it), how RAPTOR handles questions
spanning many chunks, and how self-reflection (Self-RAG/CRAG, usually orchestrated with LangGraph)
catches irrelevant retrievals and hallucinations.

**⭐ Start here — suggested path:**

1. **See the upgrade menu** — read [Pinecone: Advanced RAG Techniques](https://www.pinecone.io/learn/advanced-rag-techniques/). *A map of sentence-window, parent-document, and reranking upgrades over naive RAG.*
2. **Fix precision vs context** — watch [Parent-Child Retriever for better RAG](https://www.youtube.com/watch?v=wSi0fxkH6e0). *Embed small chunks, return their parent — the small-to-big pattern.*
3. **Add multi-level retrieval** — watch [RAG From Scratch — RAPTOR](https://www.youtube.com/watch?v=z_6EeA2LDSw). *Recursive summary trees so retrieval works from detail to big-picture.*
4. **Add self-reflection** — watch [Self-reflective RAG with LangGraph: Self-RAG and CRAG](https://www.youtube.com/watch?v=pbAd8O1Lvm4). *Grade retrieved docs and the answer; retry or correct on failure.*
5. **Read the sources** — skim [Self-RAG](https://arxiv.org/abs/2310.11511) and [CRAG](https://arxiv.org/abs/2401.15884). *The two canonical self-correcting RAG papers.*

## 🎓 Courses (free)
- [Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI × LlamaIndex** — sentence-window and auto-merging (parent-doc) retrieval, with evaluation.
- [RAG from Scratch — Indexing & RAPTOR/CRAG](https://github.com/langchain-ai/rag-from-scratch) — **LangChain (Lance Martin)** — multi-representation indexing, RAPTOR, and self-corrective flows in notebooks.

## 🎥 Videos
- [LangChain Parent-Child Retriever for better RAG](https://www.youtube.com/watch?v=wSi0fxkH6e0) — **Learn Data with Mark** — the small-to-big / parent-document pattern, in code.
- [RAG From Scratch — RAPTOR](https://www.youtube.com/watch?v=z_6EeA2LDSw) — **LangChain (Lance Martin)** — recursive clustering + summarization for multi-level retrieval.
- [Self-reflective RAG with LangGraph: Self-RAG and CRAG](https://www.youtube.com/watch?v=pbAd8O1Lvm4) — **LangChain (Lance Martin)** — orchestrating grading and correction loops.
- [Building Corrective RAG with open-source, local LLMs](https://www.youtube.com/watch?v=E2shqsYwxck) — **LangChain (Lance Martin)** — CRAG end to end with a retrieval evaluator + fallback.

## 📄 Key Papers
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) — **Asai et al. (2023)** — the model decides when to retrieve and grades its own outputs.
- [Corrective Retrieval-Augmented Generation (CRAG)](https://arxiv.org/abs/2401.15884) — **Yan et al. (2024)** — a lightweight evaluator triggers correction or web search.
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059) — **Sarthi et al. (2024)** — recursive summary trees for multi-scale retrieval.

## 📰 Articles / Blogs (free, no paywall)
- [Advanced RAG Techniques](https://www.pinecone.io/learn/advanced-rag-techniques/) — **Pinecone** — the practical upgrade catalog (sentence-window, parent-document, reranking).
- [Self-Reflective RAG with LangGraph](https://www.langchain.com/blog/agentic-rag-with-langgraph) — **LangChain** — how Self-RAG/CRAG are implemented as graphs with feedback.
- [Parent Document Retriever (how-to)](https://python.langchain.com/docs/how_to/parent_document_retriever/) — **LangChain** — the canonical small-to-big retrieval recipe with code.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 14 (IR & RAG)**](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — the retrieve-then-read foundation these patterns extend, free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md)
- Prereqs: [07 Query Transformation](07-Query-Transformation-HyDE-Multi-Query.md) · [06 Re-ranking](06-Re-ranking-Cross-Encoders.md) · Next: [09 GraphRAG](09-GraphRAG.md)
- Related concept: [10 Agentic RAG](10-Agentic-RAG.md) (self-corrective flows become full agent loops) · Related domain: [09. LLMs — Hallucination & Alignment](../../09.%20LLMs/concepts/20-Hallucination-and-Alignment-Basics.md)
