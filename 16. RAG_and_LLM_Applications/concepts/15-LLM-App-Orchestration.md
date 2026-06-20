---
id: "16-rag-and-llm-apps/llm-app-orchestration"
topic: "LLM App Orchestration (chains · routing)"
parent: "16-rag-and-llm-apps"
level: intermediate
prereqs: ["rag-fundamentals", "prompting"]
interview_frequency: medium
updated: 2026-06-20
---

# LLM App Orchestration — Chains · Routing
> Real apps aren't one prompt — they're **pipelines**: load → retrieve → prompt → call model → parse →
> maybe call a tool → format. **Orchestration** frameworks (LangChain/LCEL, LlamaIndex, LangGraph)
> let you compose these steps as **chains** (linear `A | B | C`), add **routing** (send the query to
> the right datastore/prompt/model — logical or semantic), and graduate to stateful **graphs** with
> branches, loops, and memory.

**Why it matters:** the "how do you actually build and structure an LLM app?" question. You'll explain
chains vs agents vs graphs, the LCEL pipe model (`Runnable` composition, streaming, async), logical vs
semantic routing, and when to reach for a state machine (LangGraph) over a linear chain.

**⭐ Start here — suggested path:**

1. **Compose with chains** — watch [LangChain Expression Language (LCEL) Explained](https://www.youtube.com/watch?v=O0dUOtOIrfs). *The pipe `|` model: compose `Runnable`s into a chain with streaming/async for free.*
2. **Read the model** — read [LangChain: LCEL concepts](https://python.langchain.com/docs/concepts/lcel/). *Why everything is a `Runnable` and how composition works.*
3. **Add routing** — watch [RAG From Scratch — Routing](https://www.youtube.com/watch?v=pfpIndq7Fi8), then read [LangChain: How to route](https://python.langchain.com/docs/how_to/routing/). *Logical vs semantic routing to the right datastore/prompt.*
4. **Build the chain hands-on** — watch [LCEL for Chaining Components](https://www.youtube.com/watch?v=8aUYzb1aYDU). *RunnableParallel/Passthrough/Lambda and streaming in practice.*
5. **Graduate to graphs** — watch [LangGraph Tutorial](https://www.youtube.com/watch?v=1w5cCXlh7JQ) + [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview). *Stateful branches, loops, and memory when a linear chain isn't enough.*

## 🎓 Courses (free)
- [LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) — **DeepLearning.AI × LangChain** — the canonical free course on chains, memory, routing, and agents.
- [LangChain Expression Language Explained](https://www.pinecone.io/learn/series/langchain/langchain-expression-language/) — **Pinecone** — a free written course on LCEL composition with runnable examples.

## 🎥 Videos
- [LangChain Expression Language (LCEL) Explained!](https://www.youtube.com/watch?v=O0dUOtOIrfs) — **James Briggs** — the pipe model and Runnable composition, clearly.
- [LCEL for Chaining the Components — All Runnables, Async & Streaming](https://www.youtube.com/watch?v=8aUYzb1aYDU) — **Sunny Savita** — hands-on RunnableParallel/Passthrough/Lambda.
- [RAG from Scratch — Routing](https://www.youtube.com/watch?v=pfpIndq7Fi8) — **LangChain (Lance Martin)** — logical vs semantic routing to the right source/prompt.
- [LangGraph Tutorial — Build Advanced AI Agent Systems](https://www.youtube.com/watch?v=1w5cCXlh7JQ) — **Tech With Tim** — stateful graph orchestration with branches and loops.

## 📄 Key Papers
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — **Yao et al. (2022)** — the reason-act loop that underpins tool-using orchestration.
- [MRKL Systems: Modular Reasoning, Knowledge and Language](https://arxiv.org/abs/2205.00445) — **Karpas et al. (2022)** — routing queries to expert modules/tools, the conceptual basis of routers.
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — **Schick et al. (2023)** — models invoking tools, a building block of orchestrated apps.

## 📰 Articles / Blogs (free, no paywall)
- [LCEL: LangChain Expression Language (concepts)](https://python.langchain.com/docs/concepts/lcel/) — **LangChain** — the authoritative explanation of Runnable composition.
- [How to route between sub-chains](https://python.langchain.com/docs/how_to/routing/) — **LangChain** — logical and semantic routing recipes.
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — **LangChain** — when and how to use a stateful graph for orchestration.
- [LangChain Expression Language Explained](https://www.pinecone.io/learn/series/langchain/langchain-expression-language/) — **Pinecone** — clear walkthrough with examples.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 14 (QA, IR & RAG — the retrieve→read pipeline)**](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — the pipeline these frameworks orchestrate, free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md) · [8.01 In-Context Learning & Prompting](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md)
- Prereqs: [01 RAG Fundamentals](01-RAG-Fundamentals.md) · Next: [16 Caching & Cost Optimization](16-Caching-and-Cost-Optimization.md)
- Related domain: [15. Agentic AI](../../15.%20Agentic_AI/concepts/README.md) (orchestration becomes agent loops) · [08. LLMs — Prompting & In-Context Learning](../../08.%20LLMs/concepts/16-Prompting-and-In-Context-Learning.md)
