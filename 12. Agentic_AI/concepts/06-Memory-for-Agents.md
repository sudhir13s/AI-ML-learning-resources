---
id: "12-agentic-ai/memory"
topic: "Memory for Agents (short- & long-term)"
parent: "12-agentic-ai"
level: advanced
prereqs: ["llm-agents-overview"]
interview_frequency: high
updated: 2026-06-20
---

# Memory for Agents (short- & long-term)
> **Short-term memory** is the context window (recent turns, scratchpad). **Long-term memory** lives
> *outside* the model — usually a vector store you retrieve from — so the agent recalls facts,
> preferences, and past episodes across sessions. Memory is what turns a stateless chat into a
> persistent agent.

**Why it matters:** the practical answer to "how does an agent remember beyond the context window?"
Interviews want the short-vs-long-term split, how long-term memory is *retrieval* (embeddings + vector
DB — the bridge to RAG), the types (episodic / semantic / procedural), and the hard parts: what to
write, when to summarize/compact, and how to avoid stale or contradictory memories.

**⭐ Start here — suggested path:**

1. **Frame the components** — read the **memory** section of ⭐ [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/). *Maps agent memory onto human memory types — the standard framing.*
2. **Make it concrete** — read [Memory in Agents](https://www.philschmid.de/memory-in-agents). *Clear, code-level treatment of short- vs long-term memory.*
3. **See production patterns** — read [Memory for Agents](https://www.langchain.com/blog/memory-for-agents). *Semantic/episodic/procedural memory and when to use each.*
4. **Watch a build** — watch [LangGraph Crash Course](https://www.youtube.com/watch?v=PqS1kib7RTw). *Persistent state and memory wired into an agent.*
5. **Study a memory-driven agent** — read [Generative Agents](https://arxiv.org/abs/2304.03442). *A memory stream + retrieval + reflection that produces believable long-horizon behavior.*

## 🎓 Courses (free)
- [Long-Term Agentic Memory with LangGraph](https://learn.deeplearning.ai/courses/long-term-agentic-memory-with-langgraph) — **DeepLearning.AI × LangChain** — free short course on semantic/episodic/procedural memory.
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit1/introduction) — **Hugging Face** — adding memory to an agent, hands-on.

## 🎥 Videos
- [LangGraph Crash Course](https://www.youtube.com/watch?v=PqS1kib7RTw) — **Sam Witteveen** — persistent state and memory in an agent graph.
- [LangGraph: Intro](https://www.youtube.com/watch?v=5h-JBkySK34) — **LangChain** — state and checkpointing as the substrate for memory.
- [AI Agents Fundamentals in 21 Minutes](https://www.youtube.com/watch?v=qU3fmidNbJE) — **Tina Huang** — where memory sits in the agent loop.
- [How We Build Effective Agents](https://www.youtube.com/watch?v=D7_ipDqhtwk) — **Barry Zhang (Anthropic)** — context and state management in production agents.

## 📄 Key Papers
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — **Park et al. (2023)** — memory stream + retrieval + reflection for long-horizon behavior.
- [Voyager: An Open-Ended Embodied Agent with LLMs](https://arxiv.org/abs/2305.16291) — **Wang et al. (2023)** — a growing *skill library* as procedural long-term memory.
- [Reflexion: Language Agents with Verbal RL](https://arxiv.org/abs/2303.11366) — **Shinn et al. (2023)** — episodic memory of past attempts to improve next time.

## 📰 Articles / Blogs (free, no paywall)
- [LLM Powered Autonomous Agents — Memory](https://lilianweng.github.io/posts/2023-06-23-agent/) — **Lilian Weng** — short- vs long-term memory mapped to human memory.
- [Memory in Agents](https://www.philschmid.de/memory-in-agents) — **Philipp Schmid** — code-level short- and long-term memory, free.
- [Memory for Agents](https://www.langchain.com/blog/memory-for-agents) — **LangChain** — semantic, episodic, and procedural memory in practice.

## 📚 Books (free, with chapters)
- [Artificial Intelligence: A Modern Approach — **Ch. 2 "Intelligent Agents"** (state & internal models)](https://aima.cs.berkeley.edu/) — **Russell & Norvig** — model-based agents keep internal state — the classical root of agent memory.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
- Prev / next: [05 Reflection & Self-Critique](05-Reflection-and-Self-Critique.md) · [07 Multi-Agent Systems](07-Multi-Agent-Systems-and-Orchestration.md)
- Related (canonical home): long-term memory is retrieval — see [RAG & LLM Applications](../../11.%20RAG_and_LLM_Applications/concepts/README.md)
