---
id: "11-rag-and-llm-apps/agentic-rag/references"
topic: "Agentic RAG — References"
parent: "11-rag-and-llm-apps/agentic-rag"
type: references
updated: 2026-07-02
---

# Agentic RAG — references and further reading

> Companion link library for **[Agentic RAG](10-Agentic-RAG.md)** (the concept page). External
> sources *and* internal cross-links, kept separate so it can be reused as a standalone list. Grouped
> by type, best-first. Every entry is free/open (no paywall) and chosen for depth on *this* topic —
> putting an LLM in a decision loop over retrieval: the ReAct loop, tool use, routing, multi-step /
> self-correcting retrieval, and the cost/latency tradeoffs. Every formula/mechanism cited on the
> concept page (ReAct, Reflexion, Toolformer, routing, Adaptive-RAG) appears here as a primary source.

**Start here — suggested path**:
1. **Get the framing** — watch [What is Agentic RAG?](https://www.youtube.com/watch?v=0z9_MhcYvcY) (**IBM Technology**). *Static pipeline → an agent that decides, grades, and acts.*
2. **See the evolution** — watch [RAG's Evolution: From Simple Retrieval to Agentic AI](https://www.youtube.com/watch?v=JB2P5Gk23VI) (**IBM Technology**). *How naive → advanced → agentic RAG progresses, and why.*
3. **Read the loop's primary source** — read [ReAct](https://arxiv.org/abs/2210.03629) (**Yao et al. 2022**), §2. *Reason → Act → Observe: the control loop the whole chapter is built on.*
4. **Read the pattern write-up** — read [Weaviate: What is Agentic RAG?](https://weaviate.io/blog/what-is-agentic-rag). *Agents, tools, routing, and validation around retrieval, clearly framed.*
5. **Build a routing / self-correcting agent** — watch [Self-reflective RAG with LangGraph](https://www.youtube.com/watch?v=pbAd8O1Lvm4) (**LangChain**), then follow [LangGraph: build a RAG agent](https://docs.langchain.com/oss/python/langgraph/agentic-rag). *Route by complexity, grade docs, retry — in a real stateful graph.*

**Videos**:
- [What is Agentic RAG?](https://www.youtube.com/watch?v=0z9_MhcYvcY) — **IBM Technology** — a clear intro to agents-over-RAG and why adaptivity helps.
- [RAG's Evolution: From Simple Retrieval to Agentic AI](https://www.youtube.com/watch?v=JB2P5Gk23VI) — **IBM Technology** — the naive → advanced → agentic progression, and where each earns its cost.
- [Building Adaptive RAG from scratch with Command-R](https://www.youtube.com/watch?v=04ighIjMcAI) — **LangChain (Lance Martin)** — query-complexity routing to different retrieval strategies (the Adaptive-RAG idea, hands-on).
- [Self-reflective RAG with LangGraph: Self-RAG and CRAG](https://www.youtube.com/watch?v=pbAd8O1Lvm4) — **LangChain (Lance Martin)** — grading + correction loops implemented as a graph; the clearest walkthrough of the agentic control flow.

**Interactive & visual**:
- [ReAct project page (worked traces)](https://react-lm.github.io/) — **Yao et al.** — the reason→act→observe loop with real Thought/Action/Observation traces you can read line by line.
- [Self-RAG project page + demo](https://selfrag.github.io/) — **Asai et al.** — models, data, and an interactive demo of reflection-token generation (an agentic RAG policy).
- [LangGraph agentic-RAG tutorial (runnable)](https://docs.langchain.com/oss/python/langgraph/agentic-rag) — **LangChain** — the loop as a stateful graph with cycles; the canonical build-it-yourself notebook.

**Courses (free)**:
- [RAG from Scratch — Routing, CRAG & Adaptive RAG](https://github.com/langchain-ai/rag-from-scratch) — **LangChain (Lance Martin)** — routing, self-correction, and adaptive flows as runnable notebooks.
- [Build a custom RAG agent with LangGraph](https://docs.langchain.com/oss/python/langgraph/agentic-rag) — **LangChain** — the canonical free tutorial for an agent that decides when/what to retrieve and grades results.

**Articles / blogs (free, no paywall)**:
- [What is Agentic RAG?](https://weaviate.io/blog/what-is-agentic-rag) — **Weaviate** — the clearest conceptual write-up (single-agent vs multi-agent, tools, routing, validation).
- [What is agentic RAG?](https://www.ibm.com/think/topics/agentic-rag) — **IBM** — definitions, components, and trade-offs, concise.
- [Agentic RAG with LangGraph](https://qdrant.tech/articles/agentic-rag/) — **Qdrant** — a build-focused walkthrough with document grading and a web-search fallback.
- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — **OpenAI** — the tool-calling primitive the loop's `Action` is built on: schema-driven tool calls whose descriptions the router scores against.
- [LlamaIndex — ReActAgent](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/) — **LlamaIndex** — the ReAct loop as a production object over your tools (`max_iterations` is the step budget).
- [LlamaIndex — RouterQueryEngine](https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine/) — **LlamaIndex** — routing as pick-one-engine-by-description: the argmax-cosine router, productionized.
- [LlamaIndex — SubQuestionQueryEngine](https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/) — **LlamaIndex** — decompose a compound query into sub-questions, answer each, combine.

**Key papers**:
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — **Yao, Zhao, Yu, Du, Shafran, Narasimhan & Cao (2022)** — the primary source for the Thought → Action → Observation loop this chapter builds; the trace-conditioned policy $a_t = \pi(q, h_t)$ comes from §2.
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) — **Shinn, Cassano, Berman, Gopinath, Narasimhan & Yao (2023)** — the *critique-and-revise-the-trace* extension: the agent reflects on failures and improves on retry (the self-correction lever).
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — **Schick, Dwivedi-Yu, Dessì, Raileanu, Lomeli, Zettlemoyer, Cancedda & Scialom (2023)** — learning *when and how* to call tools (retriever, calculator, search) from the trace — the tool-use foundation the router/registry generalize.
- [Agentic Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2501.09136) — **Singh, Ehtesham, Kumar & Khoei (2025)** — the map of agentic RAG patterns (routing, tools, single- vs multi-agent) — the reference for the whole design space.
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) — **Asai, Wu, Wang, Sil & Hajishirzi (2023)** — on-demand retrieval + self-critique via reflection tokens; an agentic-RAG cornerstone (built in [chapter 8](../08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG/08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.md)) and a policy over the loop here.
- [Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity](https://arxiv.org/abs/2403.14403) — **Jeong, Baek, Cho, Hwang & Park (2024)** — route simple queries to a one-shot path and complex ones to the multi-step loop — the fix for the over-agentic / cost pitfall, and the routing math's learned counterpart.
- [Corrective Retrieval Augmented Generation (CRAG)](https://arxiv.org/abs/2401.15884) — **Yan, Gu, Zhu & Ling (2024)** — a lightweight retrieval evaluator + web-search fallback: the self-correcting policy, plug-and-play over any RAG.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — **Lewis et al. (2020)** — the original RAG the agent generalizes from a fixed pipeline into a loop.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 14 "Question Answering & Information Retrieval"](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — the retrieve-then-read foundation the agent loop wraps, free PDF.

**In this platform**:
- Concept page (full explanation): [Agentic RAG](10-Agentic-RAG.md)
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md) · [8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md)
- Foundations this builds on: [RAG Fundamentals (retrieve-then-generate)](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · [Hybrid Search + the DenseRetriever this reuses](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md) · [Query Transformation (multi-query, the decompose idea)](../07-Query-Transformation-HyDE-Multi-Query/07-Query-Transformation-HyDE-Multi-Query.md)
- Special cases of this loop: [08 Advanced RAG (Self-RAG grading + retrieve-on-demand)](../08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG/08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.md)
- Measure it: [11 RAG Evaluation](../11-RAG-Evaluation/11-RAG-Evaluation.md) · [14 Guardrails & Hallucination Mitigation](../14-Guardrails-and-Hallucination-Mitigation/14-Guardrails-and-Hallucination-Mitigation.md)
- Related domain: [12. Agentic AI](../../12.%20Agentic_AI/README.md) (the general agent loop this specializes for retrieval)
