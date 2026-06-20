---
id: "11-rag-and-llm-apps/agentic-rag"
topic: "Agentic RAG"
parent: "11-rag-and-llm-apps"
level: advanced
prereqs: ["advanced-rag", "query-transformation", "rag-fundamentals"]
interview_frequency: medium
updated: 2026-06-20
---

# Agentic RAG
> Turn the fixed retrieve-then-generate pipeline into a **decision-making loop**: an LLM agent decides
> *whether* to retrieve, *which* tool/source to use, *grades* what it got, *rewrites* the query or
> *searches the web* if it's weak, and *iterates* until the answer is good. Self-RAG/CRAG/Adaptive-RAG
> are special cases; the general form adds tool use, routing, and multi-step retrieval — usually
> orchestrated as a graph (LangGraph).

**Why it matters:** the frontier "make RAG reliable and dynamic" question and the bridge to agents.
You'll contrast static RAG vs an agent that plans/grades/retries, explain document-grading and
query-rewrite loops, when to route to vector store vs web vs SQL, and the cost/latency/looping risks
(and how to bound them).

**⭐ Start here — suggested path:**

1. **Get the framing** — watch [What is Agentic RAG?](https://www.youtube.com/watch?v=0z9_MhcYvcY). *Static pipeline → agent that decides, grades, and acts.*
2. **See the evolution** — watch [RAG's Evolution: From Simple Retrieval to Agentic AI](https://www.youtube.com/watch?v=JB2P5Gk23VI). *How naive → advanced → agentic RAG progresses, and why.*
3. **Read the pattern** — read [Weaviate: What is Agentic RAG?](https://weaviate.io/blog/what-is-agentic-rag). *Agents, tools, routing, and validation around retrieval.*
4. **Build a routing agent** — watch [Building Adaptive RAG with Command-R](https://www.youtube.com/watch?v=04ighIjMcAI), then follow [LangGraph: build a RAG agent](https://docs.langchain.com/oss/python/langgraph/agentic-rag). *Route by query complexity; grade docs; retry — in a real graph.*
5. **Add self-correction** — watch [Self-reflective RAG with LangGraph](https://www.youtube.com/watch?v=pbAd8O1Lvm4). *Self-RAG/CRAG as concrete agentic loops over retrieval.*

## 🎓 Courses (free)
- [RAG from Scratch — Routing, CRAG & Adaptive RAG](https://github.com/langchain-ai/rag-from-scratch) — **LangChain (Lance Martin)** — routing, self-correction, and adaptive flows as runnable notebooks.
- [Build a custom RAG agent with LangGraph](https://docs.langchain.com/oss/python/langgraph/agentic-rag) — **LangChain** — the canonical free tutorial for an agent that decides when/what to retrieve and grades results.

## 🎥 Videos
- [What is Agentic RAG?](https://www.youtube.com/watch?v=0z9_MhcYvcY) — **IBM Technology** — clear intro to agents-over-RAG and why adaptivity helps.
- [RAG's Evolution: From Simple Retrieval to Agentic AI](https://www.youtube.com/watch?v=JB2P5Gk23VI) — **IBM Technology** — the naive → advanced → agentic progression.
- [Building Adaptive RAG from scratch with Command-R](https://www.youtube.com/watch?v=04ighIjMcAI) — **LangChain (Lance Martin)** — query-complexity routing to different retrieval strategies.
- [Self-reflective RAG with LangGraph: Self-RAG and CRAG](https://www.youtube.com/watch?v=pbAd8O1Lvm4) — **LangChain (Lance Martin)** — grading + correction loops, the core agentic pattern.

## 📄 Key Papers
- [Agentic Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2501.09136) — **Singh et al. (2025)** — the map of agentic RAG patterns (routing, tools, multi-agent).
- [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511) — **Asai et al. (2023)** — on-demand retrieval + self-critique, an agentic RAG cornerstone.
- [Adaptive-RAG: Learning to Adapt Retrieval Strategies to Query Complexity](https://arxiv.org/abs/2403.14403) — **Jeong et al. (2024)** — route simple vs complex queries to different RAG strategies.

## 📰 Articles / Blogs (free, no paywall)
- [What is Agentic RAG?](https://weaviate.io/blog/what-is-agentic-rag) — **Weaviate** — the clearest conceptual write-up (single-agent vs multi-agent, tools, routing).
- [What is agentic RAG?](https://www.ibm.com/think/topics/agentic-rag) — **IBM** — definitions, components, and trade-offs.
- [Agentic RAG with LangGraph](https://qdrant.tech/articles/agentic-rag/) — **Qdrant** — a build-focused walkthrough with grading and web-search fallback.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 14 (IR & RAG)**](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — the retrieve-then-read foundation the agent loop wraps, free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md) · [8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md)
- Prereqs: [08 Advanced RAG](08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.md) · [07 Query Transformation](07-Query-Transformation-HyDE-Multi-Query.md) · Next: [11 RAG Evaluation](11-RAG-Evaluation.md)
- Related domain: [15. Agentic AI](../../12.%20Agentic_AI/concepts/README.md) (the general agent loop this specializes)
