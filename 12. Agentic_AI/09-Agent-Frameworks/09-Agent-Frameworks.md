---
id: "12-agentic-ai/frameworks"
topic: "Agent Frameworks (LangGraph, etc., conceptual)"
parent: "12-agentic-ai"
level: advanced
prereqs: ["llm-agents-overview", "tool-use-function-calling"]
interview_frequency: medium
updated: 2026-06-20
---

# Agent Frameworks (LangGraph, etc., conceptual)
> Libraries that give you the agent loop, tool wiring, state, and orchestration so you don't rebuild
> it each time. **LangGraph** models agents as a stateful graph (nodes = steps, edges = control flow);
> **AutoGen** and **CrewAI** focus on multi-agent conversation/roles. The concepts transfer across all.

**Why it matters:** interviews care less about API trivia and more about *what frameworks abstract*
and *when not to use one*. Be ready to explain why durable, cyclic, stateful graphs (LangGraph) beat
linear chains for real agents, the difference between a **framework** and a hand-rolled loop, and
Anthropic's caution that the best systems are often simple, composable patterns — frameworks add
value but also indirection.

**⭐ Start here — suggested path:**

1. **Know when to reach for one** — read ⭐ [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents). *Frameworks vs simple composable code — the judgment interviews probe.*
2. **Get the dominant model** — read [LangGraph overview](https://langchain-ai.github.io/langgraph/). *Stateful graph of nodes/edges — the abstraction most frameworks converge toward.*
3. **See the contrast** — watch [LangChain vs LangGraph](https://www.youtube.com/watch?v=qAF1NjEVHhY). *Chains vs graphs: why agents need cycles and state.*
4. **Watch a build** — watch [LangGraph Crash Course](https://www.youtube.com/watch?v=PqS1kib7RTw). *Nodes, edges, state, and tools assembled into an agent.*
5. **Compare the field** — skim [AutoGen](https://arxiv.org/abs/2308.08155) and [CrewAI docs](https://docs.crewai.com/). *Conversation-first vs role-first multi-agent designs.*

## 🎓 Courses (free)
- [AI Agents in LangGraph](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph) — **DeepLearning.AI × LangChain** — build agents from scratch, then with LangGraph.
- [Multi-AI Agent Systems with crewAI](https://learn.deeplearning.ai/courses/multi-ai-agent-systems-with-crewai) — **DeepLearning.AI × crewAI** — role-based agent framework, free.

## 🎥 Videos
- [LangGraph: Intro](https://www.youtube.com/watch?v=5h-JBkySK34) — **LangChain** — the stateful-graph model for agents.
- [LangChain vs LangGraph: A Tale of Two Frameworks](https://www.youtube.com/watch?v=qAF1NjEVHhY) — **IBM Technology** — chains vs graphs, clearly contrasted.
- [LangGraph Crash Course](https://www.youtube.com/watch?v=PqS1kib7RTw) — **Sam Witteveen** — hands-on nodes/edges/state/tools.
- [LangGraph Complete Course for Beginners](https://www.youtube.com/watch?v=jGg_1h0qzaM) — **freeCodeCamp** — long-form, end-to-end agent build.

## 📄 Key Papers / Specs
- [AutoGen: Multi-Agent Conversation Framework](https://arxiv.org/abs/2308.08155) — **Wu et al. (2023)** — the conversational multi-agent framework design.
- [MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — **Hong et al. (2023)** — role/SOP-based agent framework.
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) — **Yao et al. (2022)** — the loop every framework implements under the hood.

## 📰 Articles / Blogs (free, no paywall)
- [LangGraph overview](https://langchain-ai.github.io/langgraph/) — **LangChain** — the stateful-graph framework, free docs.
- [CrewAI documentation](https://docs.crewai.com/) — **CrewAI** — role-based multi-agent framework, free docs.
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — **Anthropic** — when frameworks help vs hurt.

## 📚 Books (free, with chapters)
- [Artificial Intelligence: A Modern Approach — **Ch. 2 "Intelligent Agents"** (agent program structure)](https://aima.cs.berkeley.edu/) — **Russell & Norvig** — the agent-program abstractions frameworks operationalize.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
- Prev / next: [08 Model Context Protocol (MCP)](../08-Model-Context-Protocol-MCP/08-Model-Context-Protocol-MCP.md) · [07 Multi-Agent Systems](../07-Multi-Agent-Systems-and-Orchestration/07-Multi-Agent-Systems-and-Orchestration.md) · [10 Code Agents](../10-Code-Agents/10-Code-Agents.md)
- Related (canonical home): [Prompting & In-Context Learning](../../09.%20LLMs/16-Prompting-and-In-Context-Learning/16-Prompting-and-In-Context-Learning.md)
