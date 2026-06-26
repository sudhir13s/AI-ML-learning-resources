---
id: "12-agentic-ai/react"
topic: "ReAct — Reason + Act"
parent: "12-agentic-ai"
level: advanced
prereqs: ["llm-agents-overview", "chain-of-thought-reasoning"]
interview_frequency: very-high
updated: 2026-06-20
---

# ReAct — Reason + Act
> Interleave **reasoning traces** ("Thought:") with **actions** ("Action:" → tool call) and
> **observations** ("Observation:" ← tool result) in one prompt loop. The model thinks about what to
> do, does it, reads the result, and thinks again — grounding chain-of-thought in real tool feedback.

**Why it matters:** ReAct is *the* canonical agent pattern and a near-guaranteed interview topic. Be
ready to contrast it with plain chain-of-thought (which reasons but never acts) and with
**plan-then-execute** (which plans up front instead of reacting step by step), to explain why
interleaving reduces hallucinated facts, and to recognize failure modes (loops, ignored
observations, prompt-format drift).

**⭐ Start here — suggested path:**

1. **See the pattern** — read the ⭐ [ReAct project page](https://react-lm.github.io/). *The Thought/Action/Observation trace shown concretely on real tasks — the fastest way to "get it."*
2. **Read the source** — read [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629). *The actual claims and ablations: why interleaving beats reason-only or act-only.*
3. **Connect it to the loop** — read [ReAct Prompting](https://www.promptingguide.ai/techniques/react). *Shows the exact prompt template you'd write or be asked to write on a whiteboard.*
4. **Watch it run** — watch [Understanding ReAct with LangChain](https://www.youtube.com/watch?v=Eug2clsLtFs). *See the loop execute against tools, including where it goes wrong.*
5. **Place it among patterns** — skim [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents). *Understand when a simple ReAct loop is enough vs when you need planning or orchestration.*

## 🎓 Courses (free)
- [Hugging Face Agents Course — Unit 1 (the loop)](https://huggingface.co/learn/agents-course/unit1/introduction) — **Hugging Face** — builds the Thought/Action/Observation cycle hands-on.
- [AI Agents in LangGraph](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph) — **DeepLearning.AI × LangChain** — implements a ReAct agent from scratch, then with a framework.

## 🎥 Videos
- [Understanding ReAct with LangChain](https://www.youtube.com/watch?v=Eug2clsLtFs) — **Sam Witteveen** — clear walkthrough of the ReAct prompt and loop in code.
- [How We Build Effective Agents](https://www.youtube.com/watch?v=D7_ipDqhtwk) — **Barry Zhang (Anthropic)** — situates ReAct-style loops among production agent patterns.
- [Tips for Building AI Agents](https://www.youtube.com/watch?v=LP5OCa20Zpg) — **Anthropic** — practical guidance on tool loops, observations, and avoiding runaway iterations.
- [AI Agents Fundamentals in 21 Minutes](https://www.youtube.com/watch?v=qU3fmidNbJE) — **Tina Huang** — the reason-act-observe cycle explained from zero.

## 📄 Key Papers
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — **Yao et al. (2022)** — introduces the interleaved reasoning + acting paradigm.
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) — **Wei et al. (2022)** — the reasoning half of ReAct; read to see what ReAct adds on top.
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) — **Shinn et al. (2023)** — extends ReAct loops with self-reflection between attempts.

## 📰 Articles / Blogs (free, no paywall)
- [ReAct project page](https://react-lm.github.io/) — **Yao et al.** — interactive examples of the Thought/Action/Observation trace.
- [ReAct Prompting](https://www.promptingguide.ai/techniques/react) — **Prompt Engineering Guide** — the prompt template and a worked example, free.
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — **Anthropic** — when a ReAct-style loop is the right level of complexity.

## 📚 Books (free, with chapters)
- [Artificial Intelligence: A Modern Approach — **Ch. 2 "Intelligent Agents"**](https://aima.cs.berkeley.edu/) — **Russell & Norvig** — the percept→action loop that ReAct is a modern, language-based instance of.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
- Prev / next: [01 LLM Agents Overview](01-LLM-Agents-Overview.md) · [03 Tool Use & Function Calling](03-Tool-Use-and-Function-Calling.md) · [05 Reflection & Self-Critique](05-Reflection-and-Self-Critique.md)
- Related (canonical home): [Chain-of-Thought Reasoning](../../09.%20LLMs/17-Chain-of-Thought-Reasoning/17-Chain-of-Thought-Reasoning.md)
