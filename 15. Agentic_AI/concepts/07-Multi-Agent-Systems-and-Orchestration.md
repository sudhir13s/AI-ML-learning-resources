---
id: "15-agentic-ai/multi-agent"
topic: "Multi-Agent Systems & Orchestration"
parent: "15-agentic-ai"
level: advanced
prereqs: ["llm-agents-overview", "planning"]
interview_frequency: high
updated: 2026-06-20
---

# Multi-Agent Systems & Orchestration
> Instead of one agent doing everything, use **multiple specialized agents** that communicate — an
> orchestrator/planner delegating to workers, debating peers, or a role-based "crew." Orchestration
> is the control layer that routes messages, manages shared state, and decides who acts next.

**Why it matters:** the senior-level "how do you scale beyond a single agent?" question. Be ready to
contrast topologies (**orchestrator–worker**, **hierarchical**, **debate/peer**, **network**),
explain *when multi-agent helps* (parallelism, separation of concerns, specialized tools) vs *when it
just adds cost and failure modes*, and name the trade-offs Anthropic flags in their multi-agent
research system (token cost, coordination overhead, evaluation difficulty).

**⭐ Start here — suggested path:**

1. **Decide if you even need it** — read ⭐ [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents). *Orchestrator–worker patterns and the "keep it simple" bar before going multi-agent.*
2. **See topologies** — watch [Conceptual Guide: Multi-Agent Architectures](https://www.youtube.com/watch?v=4nZl32FwU-o). *Network, supervisor, hierarchical — the standard orchestration shapes.*
3. **Read a real system** — read [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system). *Concrete wins and costs of orchestrator + sub-agents at scale.*
4. **Study a framework's take** — read [AutoGen](https://arxiv.org/abs/2308.08155) (conversational) and [MetaGPT](https://arxiv.org/abs/2308.00352) (role-based). *Two influential designs for agent collaboration.*
5. **Build one** — do [Multi-AI Agent Systems with crewAI](https://learn.deeplearning.ai/courses/multi-ai-agent-systems-with-crewai). *Role-based crews make orchestration tangible.*

## 🎓 Courses (free)
- [Multi-AI Agent Systems with crewAI](https://learn.deeplearning.ai/courses/multi-ai-agent-systems-with-crewai) — **DeepLearning.AI × crewAI** — role-based multi-agent teams, free.
- [AI Agentic Design Patterns with AutoGen](https://learn.deeplearning.ai/courses/ai-agentic-design-patterns-with-autogen) — **DeepLearning.AI × Microsoft/Penn State** — conversational multi-agent patterns.

## 🎥 Videos
- [Conceptual Guide: Multi-Agent Architectures](https://www.youtube.com/watch?v=4nZl32FwU-o) — **LangChain** — supervisor, hierarchical, and network topologies.
- [How We Build Effective Agents](https://www.youtube.com/watch?v=D7_ipDqhtwk) — **Barry Zhang (Anthropic)** — orchestrator–worker as a first-class pattern.
- [Andrew Ng on AI Agentic Workflows](https://www.youtube.com/watch?v=q1XFm21I-VQ) — **Andrew Ng** — multi-agent collaboration as a design pattern.
- [Tips for Building AI Agents](https://www.youtube.com/watch?v=LP5OCa20Zpg) — **Anthropic** — when multiple agents help vs add cost.

## 📄 Key Papers
- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155) — **Wu et al. (2023)** — conversational multi-agent framework.
- [MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — **Hong et al. (2023)** — role-based agents with SOP-style workflows.
- [The Rise and Potential of LLM-Based Agents: A Survey](https://arxiv.org/abs/2308.11432) — **Xi et al. (2023)** — single- vs multi-agent taxonomy and coordination.

## 📰 Articles / Blogs (free, no paywall)
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — **Anthropic** — orchestrator–worker and parallelization workflows.
- [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — **Anthropic** — real costs, wins, and evaluation of a multi-agent system.
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — **Lilian Weng** — agent components that multi-agent systems compose.

## 📚 Books (free, with chapters)
- [Artificial Intelligence: A Modern Approach — **Ch. 18 "Multiagent Decision Making"**](https://aima.cs.berkeley.edu/) — **Russell & Norvig** — the classical foundations of agents interacting and cooperating.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
- Prev / next: [04 Planning](04-Planning-Task-Decomposition.md) · [08 Model Context Protocol (MCP)](08-Model-Context-Protocol-MCP.md) · [09 Agent Frameworks](09-Agent-Frameworks.md)
- Related (canonical home): [Prompting & In-Context Learning](../../08.%20LLMs/concepts/16-Prompting-and-In-Context-Learning.md)
