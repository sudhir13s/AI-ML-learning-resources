---
id: "16-rag-and-llm-apps/query-transformation"
topic: "Query Transformation (HyDE · multi-query)"
parent: "16-rag-and-llm-apps"
level: intermediate
prereqs: ["rag-fundamentals", "embedding-models-for-retrieval", "prompting"]
interview_frequency: medium
updated: 2026-06-20
---

# Query Transformation — HyDE · Multi-Query
> The user's raw question is often a bad retrieval query — too short, ambiguous, or phrased unlike the
> documents. **Query transformation** uses the LLM to rewrite it before retrieval: **multi-query**
> (several rephrasings), **RAG-Fusion** (rewrite + reciprocal-rank-fuse), **decomposition** (split into
> sub-questions), **step-back** (ask a more general question), and **HyDE** (embed a hypothetical
> answer instead of the question). Cheap rewrites that fix the recall bottleneck.

**Why it matters:** a favorite "how do you improve retrieval *without* touching the index?" question.
You'll explain why query-document mismatch hurts dense retrieval, when to fan out vs decompose vs
step-back, why HyDE embeds a *generated answer*, and the latency/cost of extra LLM calls.

**⭐ Start here — suggested path:**

1. **See the menu** — read [LangChain: Query Transformations](https://blog.langchain.dev/query-transformations/). *The map of multi-query, RAG-fusion, decomposition, step-back, HyDE in one place.*
2. **Start with multi-query** — watch [RAG From Scratch — Multi-Query](https://www.youtube.com/watch?v=JChPi0CRnDY). *The simplest, most-used rewrite: fan out perspectives, dedupe results.*
3. **Add fusion** — watch [RAG From Scratch — RAG Fusion](https://www.youtube.com/watch?v=77qELPbNgxA). *Combine multi-query with reciprocal rank fusion for a single robust ranking.*
4. **Understand HyDE** — watch [RAG From Scratch — HyDE](https://www.youtube.com/watch?v=SaDzIVkYqyY), then skim the [HyDE paper](https://arxiv.org/abs/2212.10496). *Why embedding a hypothetical document can beat embedding the bare query.*
5. **Decompose hard questions** — watch [RAG From Scratch — Decomposition](https://www.youtube.com/watch?v=h0OPWlEOank). *Break multi-hop questions into sub-questions, retrieve per sub-question.*

## 🎓 Courses (free)
- [RAG from Scratch — Query Translation](https://github.com/langchain-ai/rag-from-scratch) — **LangChain (Lance Martin)** — the dedicated query-translation notebooks (multi-query, fusion, decomposition, step-back, HyDE) with code.
- [Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI × LlamaIndex** — query rewriting/expansion as part of advanced retrieval.

## 🎥 Videos
- [RAG from Scratch — Multi-Query](https://www.youtube.com/watch?v=JChPi0CRnDY) — **LangChain (Lance Martin)** — rewrite the question from multiple angles to widen recall.
- [RAG from Scratch — RAG Fusion](https://www.youtube.com/watch?v=77qELPbNgxA) — **LangChain (Lance Martin)** — multi-query plus reciprocal rank fusion.
- [RAG from Scratch — HyDE](https://www.youtube.com/watch?v=SaDzIVkYqyY) — **LangChain (Lance Martin)** — embed a generated hypothetical answer for zero-shot dense retrieval.
- [RAG from Scratch — Decomposition](https://www.youtube.com/watch?v=h0OPWlEOank) — **LangChain (Lance Martin)** — split complex questions into retrievable sub-questions.

## 📄 Key Papers
- [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)](https://arxiv.org/abs/2212.10496) — **Gao et al. (2022)** — generate a hypothetical document, embed it, retrieve.
- [Query2doc: Query Expansion with LLMs](https://arxiv.org/abs/2303.07678) — **Wang et al. (2023)** — LLM-generated pseudo-documents to expand sparse and dense queries.
- [Take a Step Back: Evoking Reasoning via Abstraction (Step-Back Prompting)](https://arxiv.org/abs/2310.06117) — **Zheng et al. (2023)** — ask a more general question first to retrieve better evidence.

## 📰 Articles / Blogs (free, no paywall)
- [Query Transformations](https://blog.langchain.dev/query-transformations/) — **LangChain** — the canonical survey of rewrite strategies and when to use each.
- [RAG-Fusion (reference implementation)](https://github.com/Raudaschl/rag-fusion) — **Adrian Raudaschl** — the original RAG-Fusion recipe and code.
- [Advanced RAG: Zero-Shot Dense Retrieval with HyDE](https://www.lancedb.com/blog/advanced-rag-precise-zero-shot-dense-retrieval-with-hyde-0946c54dfdcb) — **LanceDB** — a clear, code-backed walkthrough of HyDE.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 14 (IR & RAG — query processing)**](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — query formulation/expansion foundations behind these techniques, free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.01 In-Context Learning & Prompting](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md) · [8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md)
- Prereqs: [01 RAG Fundamentals](01-RAG-Fundamentals.md) · [03 Embedding Models for Retrieval](03-Embedding-Models-for-Retrieval.md) · Next: [08 Advanced RAG](08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.md)
- Related domain: [08. LLMs — Prompting & In-Context Learning](../../08.%20LLMs/concepts/16-Prompting-and-In-Context-Learning.md) · [08. LLMs — Chain-of-Thought Reasoning](../../08.%20LLMs/concepts/17-Chain-of-Thought-Reasoning.md)
