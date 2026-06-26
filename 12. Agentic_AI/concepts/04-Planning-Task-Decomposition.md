---
id: "12-agentic-ai/planning"
topic: "Planning — Task Decomposition & Plan-and-Execute"
parent: "12-agentic-ai"
level: advanced
prereqs: ["llm-agents-overview", "react", "chain-of-thought-reasoning"]
interview_frequency: high
updated: 2026-06-20
---

# Planning — Task Decomposition & Plan-and-Execute
> Break a complex goal into ordered sub-tasks *before* (or alongside) acting. **Plan-and-execute**
> writes the whole plan up front then runs steps; **ReAct-style** planning decides the next step each
> turn. Richer variants — **Tree of Thoughts**, **plan-and-solve** — search over multiple plans.

**Why it matters:** the difference between an agent that flails and one that finishes long-horizon
tasks. Interviews probe the trade-off between **upfront planning** (fewer LLM calls, brittle if the
world changes) and **reactive/step-wise planning** (adapts, but slower and can loop), plus how
decomposition, re-planning on failure, and search-based methods (ToT) fit together.

**⭐ Start here — suggested path:**

1. **Anchor on the survey** — read the **planning** section of ⭐ [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/). *Defines task decomposition, sub-goals, and reflection in one place.*
2. **See the upfront-plan paper** — read [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091). *The simplest "plan, then execute the plan" idea and why it helps.*
3. **Learn search over plans** — read [Tree of Thoughts](https://arxiv.org/abs/2305.10601). *Deliberate search across reasoning paths — the planning-as-search view.*
4. **Watch the design pattern** — watch [Andrew Ng on AI Agentic Workflows](https://www.youtube.com/watch?v=q1XFm21I-VQ). *Planning as one of the four agentic patterns, with intuition for when it pays off.*
5. **Build a plan-and-execute agent** — do [AI Agents in LangGraph](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph). *Implement explicit plan → execute → re-plan control flow.*

## 🎓 Courses (free)
- [AI Agents in LangGraph](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph) — **DeepLearning.AI × LangChain** — builds plan-and-execute and re-planning loops.
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit1/introduction) — **Hugging Face** — decomposition and multi-step task handling, free.

## 🎥 Videos
- [Andrew Ng on AI Agentic Workflows](https://www.youtube.com/watch?v=q1XFm21I-VQ) — **Andrew Ng** — planning as a core agentic design pattern.
- [Learn to Build Effective Agentic AI Systems](https://www.youtube.com/watch?v=w7vqXL4PWEE) — **Andrew Ng (DeepLearning.AI)** — decomposition and evaluation-driven agent design.
- [How We Build Effective Agents](https://www.youtube.com/watch?v=D7_ipDqhtwk) — **Barry Zhang (Anthropic)** — orchestrator–worker and planner patterns.
- [Tips for Building AI Agents](https://www.youtube.com/watch?v=LP5OCa20Zpg) — **Anthropic** — when planning helps vs when it adds brittle complexity.

## 📄 Key Papers
- [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091) — **Wang et al. (2023)** — devise a plan, then carry it out, to reduce missing-step errors.
- [Tree of Thoughts: Deliberate Problem Solving](https://arxiv.org/abs/2305.10601) — **Yao et al. (2023)** — search over multiple reasoning/plan branches.
- [HuggingGPT](https://arxiv.org/abs/2303.17580) — **Shen et al. (2023)** — plan a task as a graph of tool/model calls, then execute.
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) — **Wei et al. (2022)** — the step-by-step reasoning that planning generalizes.

## 📰 Articles / Blogs (free, no paywall)
- [LLM Powered Autonomous Agents — Planning](https://lilianweng.github.io/posts/2023-06-23-agent/) — **Lilian Weng** — task decomposition, sub-goals, and self-reflection.
- [How Agents Can Improve LLM Performance](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/) — **Andrew Ng (The Batch)** — the planning design pattern in context.
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — **Anthropic** — orchestrator–worker and planning workflows, with caveats.

## 📚 Books (free, with chapters)
- [Artificial Intelligence: A Modern Approach — **Ch. 11 "Automated Planning"**](https://aima.cs.berkeley.edu/) — **Russell & Norvig** — the classical planning foundations LLM planners echo (free chapter PDFs).

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
- Prev / next: [02 ReAct](02-ReAct-Reason-and-Act.md) · [05 Reflection & Self-Critique](05-Reflection-and-Self-Critique.md) · [07 Multi-Agent Systems](07-Multi-Agent-Systems-and-Orchestration.md)
- Related (canonical home): [Chain-of-Thought Reasoning](../../09.%20LLMs/17-Chain-of-Thought-Reasoning/17-Chain-of-Thought-Reasoning.md)
