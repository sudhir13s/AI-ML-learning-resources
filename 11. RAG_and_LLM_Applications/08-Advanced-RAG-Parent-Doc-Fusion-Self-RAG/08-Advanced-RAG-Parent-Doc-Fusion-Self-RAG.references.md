---
id: "11-rag-and-llm-apps/advanced-rag-parent-doc-fusion-self-rag/references"
topic: "Advanced RAG (Parent-Document · RAG-Fusion · Self-RAG) — References"
parent: "11-rag-and-llm-apps/advanced-rag-parent-doc-fusion-self-rag"
type: references
updated: 2026-07-01
---

# Advanced RAG (Parent-Document · RAG-Fusion · Self-RAG) — references and further reading

> Companion link library for **[Advanced RAG (Parent-Document · RAG-Fusion · Self-RAG)](08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.md)** (the concept page). External sources *and* internal cross-links, kept separate so it can be reused as a standalone list. Grouped by type, best-first. Every entry is free/open (no paywall) and chosen for depth on *this* topic — decoupling retrieval from generation (parent-document / sentence-window / auto-merging), RAG-Fusion, and self-reflective RAG (Self-RAG, CRAG).

**Start here — suggested path**:
1. **See the retrieve-small-read-large idea** — read [LangChain's ParentDocumentRetriever how-to](https://python.langchain.com/docs/how_to/parent_document_retriever/). *The exact tension it resolves — small chunks embed sharply, large chunks keep context — and the child→parent mechanism, with runnable code.*
2. **See the adaptive-parent variant** — read [LlamaIndex's Auto Merging Retriever guide](https://developers.llamaindex.ai/python/framework/integrations/retrievers/auto_merging_retriever/). *A graded hierarchy (2048/512/128) where the parent size adapts to how many leaves hit.*
3. **Understand self-reflective RAG** — read [the Self-RAG paper](https://arxiv.org/abs/2310.11511) (**Asai et al. 2023**), §2–3 + Table 1. *The four reflection tokens and the retrieve-on-demand + self-critique loop — the primary source.*
4. **See the corrective cousin** — read [the CRAG paper](https://arxiv.org/abs/2401.15884) (**Yan et al. 2024**). *A lightweight retrieval evaluator + web-search fallback, plug-and-play over any RAG.*
5. **Build the loop yourself** — read [LangGraph's Self-RAG tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/). *The reflect loop (grade docs → grade grounding → grade usefulness → regenerate) implemented as a graph.*

**Videos**:
- [Advanced RAG — Parent-Document & small-to-big retrieval](https://www.youtube.com/watch?v=lQhU3Rmc410) — **LlamaIndex (Jerry Liu)** — sentence-window and auto-merging retrievers, the retrieve-small-read-large pattern, from the framework's author.
- [Self-RAG explained — retrieve, generate, critique](https://www.youtube.com/watch?v=Eb7QF1nDWGU) — **Sam Witteveen** — the reflection-token idea and the reflect loop in plain terms, with code.
- [Self-Reflective RAG with LangGraph (Self-RAG + CRAG)](https://www.youtube.com/watch?v=pbAd8O1Lvm4) — **LangChain (Lance Martin)** — implementing Self-RAG and Corrective RAG as graphs; the clearest walkthrough of the grading loop.
- [RAG-Fusion — better retrieval by fanning out queries](https://www.youtube.com/watch?v=GchC5WxeXGc) — **Prompt Engineering** — the multi-query + RRF pattern this chapter reuses, with a worked example.
- [ParentDocumentRetriever in LangChain](https://www.youtube.com/watch?v=wSi0fxkH6e0) — **Coding Crash Courses** — building the child→parent retriever step by step.

**Interactive & visual**:
- [LangGraph Self-RAG notebook](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_self_rag.ipynb) — **LangChain** — a runnable graph of the grade-docs → grade-grounding → grade-usefulness loop.
- [Self-RAG project page + demo](https://selfrag.github.io/) — **Asai et al.** — the authors' models, data, and an interactive demo of reflection-token generation.

**Courses (free)**:
- [DeepLearning.AI — Building and Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI + LlamaIndex** — sentence-window and auto-merging retrieval inside a full advanced-RAG pipeline, with evaluation.
- [LangChain — RAG from Scratch (parts 10–14: routing, query construction, indexing, retrieval, generation)](https://github.com/langchain-ai/rag-from-scratch) — **LangChain** — the notebooks behind the advanced-RAG videos, including multi-query/RAG-Fusion and self-reflective retrieval.

**Articles / blogs (free, no paywall)**:
- [How to use the ParentDocumentRetriever](https://python.langchain.com/docs/how_to/parent_document_retriever/) — **LangChain docs** — the API the page cites (child/parent splitters, retrieve-small-return-large), with runnable examples.
- [Auto Merging Retriever](https://developers.llamaindex.ai/python/framework/integrations/retrievers/auto_merging_retriever/) — **LlamaIndex docs** — the hierarchical node parser (2048/512/128) and threshold-merging retriever.
- [Advanced RAG 01 — Small-to-Big Retrieval](https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4/) — **Sophia Yang (Towards Data Science)** — sentence-window and parent-document retrieval, why decoupling the units works.
- [Self-RAG: AI That Knows When to Look Things Up](https://blog.langchain.dev/agentic-rag-with-langgraph/) — **LangChain blog** — Self-RAG and corrective RAG framed as agentic retrieval, with the grading loop.
- [Corrective RAG (CRAG) explained](https://blog.lancedb.com/implementing-corrective-rag-in-the-easiest-way-2/) — **LanceDB** — the retrieval-evaluator + fallback mechanism, implemented simply.

**Key papers**:
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) — **Asai, Wu, Wang, Sil & Hajishirzi (2023)** — the primary source: reflection tokens (`Retrieve`/`ISREL`/`ISSUP`/`ISUSE`), retrieve-on-demand, and self-critique; outperforms ChatGPT and retrieval-augmented Llama2-chat on QA/reasoning/fact-verification.
- [Corrective Retrieval Augmented Generation (CRAG)](https://arxiv.org/abs/2401.15884) — **Yan, Gu, Zhu & Ling (2024)** — a lightweight retrieval evaluator grading docs (correct/ambiguous/incorrect) + web-search fallback + decompose-then-recompose; plug-and-play over any RAG.
- [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://doi.org/10.1145/1571941.1572114) — **Cormack, Clarke & Büttcher (SIGIR 2009)** — the RRF that RAG-Fusion fuses with ($k=60$); worked in full in [chapter 5](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md). ([free PDF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf))
- [Dense Passage Retrieval for Open-Domain QA (DPR)](https://arxiv.org/abs/2004.04906) — **Karpukhin et al. (2020)** — the dense bi-encoder retrieval every technique here sits on top of.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — **Liu et al. (2023)** — why bigger context isn't free (models under-use mid-context evidence): the reason a parent must be section-sized, not a whole chapter.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 14 "Question Answering & Information Retrieval"](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — retrieval, grounding, and the recall/precision metrics this page measures against.

**In this platform**:
- Concept page (full explanation): [Advanced RAG (Parent-Document · RAG-Fusion · Self-RAG)](08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.md)
- Foundations this builds on: [RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · [Document Chunking (the small-vs-big dilemma this resolves)](../02-Document-Chunking-Strategies/02-Document-Chunking-Strategies.md) · [Hybrid Search + RRF](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md) · [Query Transformation (RAG-Fusion = multi-query + RRF)](../07-Query-Transformation-HyDE-Multi-Query/07-Query-Transformation-HyDE-Multi-Query.md)
- Composes with: [Re-ranking with Cross-Encoders (order the retrieved pool)](../06-Re-ranking-Cross-Encoders/06-Re-ranking-Cross-Encoders.md) · [Agentic RAG (the loop taken further)](../10-Agentic-RAG/10-Agentic-RAG.md)
- Measure it: [RAG Evaluation (grounding, faithfulness, recall@k, MRR)](../11-RAG-Evaluation/11-RAG-Evaluation.md) · [Guardrails & Hallucination Mitigation (the support-check, productionized)](../14-Guardrails-and-Hallucination-Mitigation/14-Guardrails-and-Hallucination-Mitigation.md)
