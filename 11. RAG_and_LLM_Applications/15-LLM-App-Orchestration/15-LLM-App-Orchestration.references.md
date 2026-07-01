---
id: "11-rag-and-llm-apps/llm-app-orchestration/references"
topic: "LLM App Orchestration (chains · routing · stateful graphs) — References"
parent: "11-rag-and-llm-apps/llm-app-orchestration"
type: references
updated: 2026-07-02
---

# LLM App Orchestration — references and further reading

> Companion link library for **[LLM App Orchestration](15-LLM-App-Orchestration.md)** (the concept
> page). External sources *and* internal cross-links, kept separate so it can be reused as a standalone
> list. Grouped by type, best-first. Every entry is free/open (no paywall) and chosen for depth on
> *this* topic — composing an LLM app as chains, routing to the right path, and stateful cyclic graphs.
> Every framework/mechanism cited on the concept page (LCEL/Runnables, semantic routing, LangGraph
> `StateGraph`, DSPy) appears here as a primary source.

**Start here — suggested path**:
1. **Compose with chains** — watch [LangChain Expression Language (LCEL) Explained](https://www.youtube.com/watch?v=O0dUOtOIrfs) (**James Briggs**). *The pipe `|` model: compose Runnables into a chain with streaming/async for free.*
2. **Read the model** — read [LangChain: LCEL concepts](https://python.langchain.com/docs/concepts/lcel/). *Why everything is a Runnable and how composition works.*
3. **Add routing** — watch [RAG From Scratch — Routing](https://www.youtube.com/watch?v=pfpIndq7Fi8) (**Lance Martin**), then read [LangChain: How to route](https://python.langchain.com/docs/how_to/routing/). *Logical vs semantic routing to the right path — the cosine-argmax decision.*
4. **Graduate to graphs** — read the [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/) + watch [LangGraph Tutorial](https://www.youtube.com/watch?v=1w5cCXlh7JQ) (**Tech With Tim**). *Stateful branches, loops, and a step budget when a linear chain isn't enough.*
5. **See the frontier** — skim the [DSPy paper](https://arxiv.org/abs/2310.03714) (**Khattab et al. 2023**). *Declarative modules a compiler optimizes — orchestration where the prompts tune themselves.*

**Videos**:
- [LangChain Expression Language (LCEL) Explained!](https://www.youtube.com/watch?v=O0dUOtOIrfs) — **James Briggs** — the pipe model and Runnable composition, clearly.
- [RAG from Scratch — Routing](https://www.youtube.com/watch?v=pfpIndq7Fi8) — **LangChain (Lance Martin)** — logical vs semantic routing to the right source/prompt.
- [LangGraph Tutorial — Build Advanced AI Agent Systems](https://www.youtube.com/watch?v=1w5cCXlh7JQ) — **Tech With Tim** — stateful graph orchestration with branches and loops.
- [LCEL for Chaining the Components — All Runnables, Async & Streaming](https://www.youtube.com/watch?v=8aUYzb1aYDU) — **Sunny Savita** — hands-on RunnableParallel/Passthrough/Lambda and streaming.

**Interactive & visual**:
- [LangGraph — Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart) — **LangChain** — build, compile, and invoke a `StateGraph` end to end (the stateful-graph API used on the page), runnable.
- [LangGraph — Workflows & Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — **LangChain** — the orchestration patterns (prompt chaining, routing, parallelization, orchestrator-worker) as runnable `StateGraph` code.
- [DSPy — GitHub](https://github.com/stanfordnlp/dspy) — **Stanford NLP** — the open-source framework to read and run: signatures, modules, and the compiler/optimizer.

**Courses (free)**:
- [LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) — **DeepLearning.AI × LangChain** — the canonical free course on chains, memory, routing, and agents.
- [LangChain Expression Language Explained](https://www.pinecone.io/learn/series/langchain/langchain-expression-language/) — **Pinecone** — a free written course on LCEL composition with runnable examples.

**Articles / blogs (free, no paywall)**:
- [LCEL: LangChain Expression Language (concepts)](https://python.langchain.com/docs/concepts/lcel/) — **LangChain** — the authoritative explanation of Runnable composition (`prompt | model | parser`).
- [How to route between sub-chains](https://python.langchain.com/docs/how_to/routing/) — **LangChain** — logical and semantic routing recipes (the routing decision the page's Router implements).
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/) — **LangChain** — when and how to use a stateful, cyclic graph for orchestration (`StateGraph` / `add_node` / `add_conditional_edges`).
- [Introducing Query Pipelines](https://www.llamaindex.ai/blog/introducing-query-pipelines-025dc2bb0537) — **LlamaIndex** — building advanced RAG DAGs by declaring modules and links. (LlamaIndex now recommends **Workflows** for new pipelines; `QueryPipeline` remains for existing DAGs.)
- [Haystack — Pipelines](https://docs.haystack.deepset.ai/docs/pipelines) — **deepset** — a node-and-edge pipeline framework for search/RAG, the same compose-typed-components model.

**Key papers**:
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714) — **Khattab et al. (2023)** — declarative modules (signatures) a compiler optimizes to a metric; orchestration where the prompts tune themselves rather than being hand-written.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — **Yao et al. (2022)** — the reason-act loop that underpins the cyclic-graph / agent-loop end of orchestration (the step-budget run loop this chapter builds).
- [MRKL Systems: Modular Reasoning, Knowledge and Language](https://arxiv.org/abs/2205.00445) — **Karpas et al. (2022)** — routing queries to expert modules/tools, the conceptual basis of the router.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — **Lewis et al. (2020)** — the retrieve→generate pipeline these frameworks orchestrate.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 14 "Question Answering & Information Retrieval"](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — the retrieve→read pipeline these frameworks orchestrate, free PDF.

**In this platform**:
- Concept page (full explanation): [LLM App Orchestration](15-LLM-App-Orchestration.md)
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md) · [8.01 In-Context Learning & Prompting](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md)
- Steps wired here: [05 Hybrid Search (the DenseRetriever)](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md) · [06 Re-ranking (the cross-encoder rerank)](../06-Re-ranking-Cross-Encoders/06-Re-ranking-Cross-Encoders.md) · [10 Agentic RAG (the cosine router + run loop)](../10-Agentic-RAG/10-Agentic-RAG.md) · [14 Guardrails (the grounding/abstention step)](../14-Guardrails-and-Hallucination-Mitigation/14-Guardrails-and-Hallucination-Mitigation.md)
- Foundations: [01 RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md)
- Next / related: [16 Caching & Cost Optimization](../16-Caching-and-Cost-Optimization/16-Caching-and-Cost-Optimization.md)
- Related domain: [12. Agentic AI](../../12.%20Agentic_AI/README.md) (orchestration becomes agent loops) · [09. LLMs — Prompting & In-Context Learning](../../09.%20LLMs/16-Prompting-and-In-Context-Learning/16-Prompting-and-In-Context-Learning.md)
