---
id: "12-agentic-ai/llm-agents-overview"
topic: "LLM Agents — Overview & the Agent Loop"
parent: "12-agentic-ai"
level: advanced
prereqs: ["llms", "prompting-and-in-context-learning"]
interview_frequency: very-high
updated: 2026-06-20
---

# LLM Agents — Overview & the Agent Loop
> An *agent* wraps an LLM in a loop: it observes a goal, **decides** what to do, **acts** via tools,
> **observes** the result, and repeats until done. The mental model is **`Agent = LLM + planning +
> memory + tool use`** — the LLM is the controller, not the whole system.

**Why it matters:** the framing question for the entire field — "what makes something an *agent* vs a
plain prompt or a fixed workflow?" Interviewers want the agent loop (think → act → observe), the
difference between **workflows** (predefined paths) and **agents** (the model drives control flow),
and *when not* to build an agent (cost, latency, reliability) — the single most common senior-level
trap in this area.

**⭐ Start here — suggested path:**

1. **Get the canonical mental model** — read ⭐ [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/). *The reference survey: it defines the planning + memory + tool-use decomposition everyone else builds on.*
2. **Learn the engineering judgment** — read [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents). *Workflows vs agents and when each is the right tool — the part interviews probe.*
3. **See it framed visually** — watch [How We Build Effective Agents (Barry Zhang)](https://www.youtube.com/watch?v=D7_ipDqhtwk). *The same ideas from the people who wrote the guide, with concrete patterns.*
4. **Ground it historically** — skim [AIMA Ch. 2 "Intelligent Agents"](https://aima.cs.berkeley.edu/). *Agents, environments, and rationality predate LLMs; this is where the vocabulary comes from.*
5. **Build one** — do the [Hugging Face Agents Course, Unit 1](https://huggingface.co/learn/agents-course/unit1/introduction). *Implementing a think-act-observe loop makes the abstraction concrete.*

## 🎓 Courses (free)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit1/introduction) — **Hugging Face** — free end-to-end course: agent loop, tools, memory, frameworks.
- [AI Agents in LangGraph](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph) — **DeepLearning.AI × LangChain** — free short course building agent loops from scratch.

## 🎥 Videos
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — the "LLM OS" framing where the model orchestrates tools and memory.
- [How We Build Effective Agents](https://www.youtube.com/watch?v=D7_ipDqhtwk) — **Barry Zhang (Anthropic)** — workflows vs agents and the patterns that actually work.
- [Tips for Building AI Agents](https://www.youtube.com/watch?v=LP5OCa20Zpg) — **Anthropic** — practical do's and don'ts from a team shipping agents.
- [AI Agents Fundamentals in 21 Minutes](https://www.youtube.com/watch?v=qU3fmidNbJE) — **Tina Huang** — fast, concrete walk through the loop, tools, and memory.

## 📄 Key Papers
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — **Lilian Weng (2023)** — the survey that defined the modern agent decomposition.
- [The Rise and Potential of LLM-Based Agents: A Survey](https://arxiv.org/abs/2308.11432) — **Xi et al. (2023)** — broad taxonomy of agent components, single- vs multi-agent.
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) — **Yao et al. (2022)** — the reason→act→observe loop that underlies most agents.

## 📰 Articles / Blogs (free, no paywall)
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — **Anthropic** — the clearest, least-hyped guide; workflows vs agents and when to use each.
- [LLM Agents](https://www.promptingguide.ai/research/llm-agents) — **Prompt Engineering Guide** — concise, open overview of the agent components and patterns.
- [How Agents Can Improve LLM Performance](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/) — **Andrew Ng (The Batch)** — the four agentic design patterns (reflection, tool use, planning, multi-agent).

## 📚 Books (free, with chapters)
- [Artificial Intelligence: A Modern Approach — **Ch. 2 "Intelligent Agents"**](https://aima.cs.berkeley.edu/) — **Russell & Norvig** — the classic definition of agents, environments, and rationality (free chapter PDFs on the book site).

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
- Next concepts: [02 ReAct](02-ReAct-Reason-and-Act.md) · [03 Tool Use & Function Calling](03-Tool-Use-and-Function-Calling.md) · [04 Planning](04-Planning-Task-Decomposition.md)
- Related (canonical home): [Prompting & In-Context Learning](../../09.%20LLMs/concepts/16-Prompting-and-In-Context-Learning.md) · [Chain-of-Thought](../../09.%20LLMs/concepts/17-Chain-of-Thought-Reasoning.md)
