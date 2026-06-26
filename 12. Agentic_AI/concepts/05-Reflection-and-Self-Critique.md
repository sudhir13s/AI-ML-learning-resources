---
id: "12-agentic-ai/reflection"
topic: "Reflection & Self-Critique"
parent: "12-agentic-ai"
level: advanced
prereqs: ["llm-agents-overview", "react"]
interview_frequency: high
updated: 2026-06-20
---

# Reflection & Self-Critique
> Have the agent **review its own output** — critique a draft, spot errors, then revise — often using
> feedback from tools, tests, or a separate "critic" pass. Turns a one-shot answer into an iterative
> *generate → critique → improve* loop, the cheapest reliable quality boost in agentic systems.

**Why it matters:** reflection is one of Andrew Ng's four agentic design patterns and a favorite
follow-up to ReAct. Be ready to explain *self-refine* (model critiques itself) vs *Reflexion*
(verbal feedback stored across attempts), why external signals (unit tests, tool errors) make
reflection far more reliable than pure self-grading, and the cost/latency trade-off of extra passes.

**⭐ Start here — suggested path:**

1. **Get the pattern** — read [How Agents Can Improve LLM Performance](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/). *Frames reflection as a design pattern with concrete gains.*
2. **Read self-critique source** — read ⭐ [Self-Refine](https://arxiv.org/abs/2303.17651). *Generate → self-feedback → refine, with no extra training.*
3. **See feedback that sticks** — read [Reflexion](https://arxiv.org/abs/2303.11366). *Stores verbal self-reflections across attempts — "verbal RL."*
4. **Watch the design pattern** — watch [Andrew Ng on AI Agentic Workflows](https://www.youtube.com/watch?v=q1XFm21I-VQ). *Why iterating beats one-shot, with the reflection pattern called out.*
5. **Connect to grounding** — skim [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents). *Evaluator–optimizer loops and using real environment feedback, not just self-grading.*

## 🎓 Courses (free)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit1/introduction) — **Hugging Face** — agents that observe results and revise, hands-on.
- [AI Agents in LangGraph](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph) — **DeepLearning.AI × LangChain** — builds evaluator/critic loops over an agent.

## 🎥 Videos
- [Andrew Ng on AI Agentic Workflows](https://www.youtube.com/watch?v=q1XFm21I-VQ) — **Andrew Ng** — reflection as a core agentic design pattern.
- [Learn to Build Effective Agentic AI Systems](https://www.youtube.com/watch?v=w7vqXL4PWEE) — **Andrew Ng (DeepLearning.AI)** — error analysis and iterative improvement of agents.
- [How We Build Effective Agents](https://www.youtube.com/watch?v=D7_ipDqhtwk) — **Barry Zhang (Anthropic)** — evaluator–optimizer and feedback loops in practice.
- [Tips for Building AI Agents](https://www.youtube.com/watch?v=LP5OCa20Zpg) — **Anthropic** — using ground-truth feedback to drive self-correction.

## 📄 Key Papers
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) — **Madaan et al. (2023)** — generate, self-critique, revise — no extra training.
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) — **Shinn et al. (2023)** — store verbal self-reflections across attempts.
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) — **Yao et al. (2022)** — the act/observe loop reflection builds on.

## 📰 Articles / Blogs (free, no paywall)
- [How Agents Can Improve LLM Performance](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/) — **Andrew Ng (The Batch)** — reflection as design pattern #1.
- [LLM Powered Autonomous Agents — Self-Reflection](https://lilianweng.github.io/posts/2023-06-23-agent/) — **Lilian Weng** — Reflexion/Chain-of-Hindsight in the broader agent picture.
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — **Anthropic** — the evaluator–optimizer workflow.

## 📚 Books (free, with chapters)
- [Artificial Intelligence: A Modern Approach — **Ch. 2 "Intelligent Agents"** (learning agents)](https://aima.cs.berkeley.edu/) — **Russell & Norvig** — the "learning element + critic" agent design that reflection instantiates.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
- Prev / next: [02 ReAct](02-ReAct-Reason-and-Act.md) · [06 Memory for Agents](06-Memory-for-Agents.md) · [12 Agent Evaluation & Benchmarks](12-Agent-Evaluation-and-Benchmarks.md)
- Related (canonical home): [Chain-of-Thought Reasoning](../../09.%20LLMs/17-Chain-of-Thought-Reasoning/17-Chain-of-Thought-Reasoning.md)
