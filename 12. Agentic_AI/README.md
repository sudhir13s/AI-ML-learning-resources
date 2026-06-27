---
id: "12-agentic-ai"
topic: "Agentic AI & Tool Use"
level: advanced
prereqs: ["llms"]
updated: 2026-06-27
---

# Agentic AI & Tool Use
> LLMs that *act* — reasoning loops, tool/function calling, memory, planning, and multi-agent
> systems. The fastest-moving area in applied AI.

**⭐ Start here:** [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — **Anthropic** — the clearest, least-hyped guide to when (and when not) to build agents.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page — a short guided
learning path plus the best **free, open** course, video, paper, article, or book for that topic.
> **✅ ready.** New here? Start with the field overview below, then work top to bottom.

### Foundations
1. ✅ [LLM Agents — Overview & the Agent Loop](01-LLM-Agents-Overview/01-LLM-Agents-Overview.md)
2. ✅ [ReAct — Reason + Act](02-ReAct-Reason-and-Act/02-ReAct-Reason-and-Act.md)
3. ✅ [Tool Use & Function Calling](03-Tool-Use-and-Function-Calling/03-Tool-Use-and-Function-Calling.md)

### Reasoning, planning & memory
4. ✅ [Planning — Task Decomposition & Plan-and-Execute](04-Planning-Task-Decomposition/04-Planning-Task-Decomposition.md)
5. ✅ [Reflection & Self-Critique](05-Reflection-and-Self-Critique/05-Reflection-and-Self-Critique.md)
6. ✅ [Memory for Agents (short- & long-term)](06-Memory-for-Agents/06-Memory-for-Agents.md)

### Systems, protocols & frameworks
7. ✅ [Multi-Agent Systems & Orchestration](07-Multi-Agent-Systems-and-Orchestration/07-Multi-Agent-Systems-and-Orchestration.md)
8. ✅ [Model Context Protocol (MCP)](08-Model-Context-Protocol-MCP/08-Model-Context-Protocol-MCP.md)
9. ✅ [Agent Frameworks (LangGraph, etc., conceptual)](09-Agent-Frameworks/09-Agent-Frameworks.md)

### Applied agents
10. ✅ [Code Agents](10-Code-Agents/10-Code-Agents.md)
11. ✅ [Computer-Use & GUI Agents](11-Computer-Use-and-GUI-Agents/11-Computer-Use-and-GUI-Agents.md)

### Evaluation & safety
12. ✅ [Agent Evaluation & Benchmarks (AgentBench · SWE-bench)](12-Agent-Evaluation-and-Benchmarks/12-Agent-Evaluation-and-Benchmarks.md)
13. ✅ [Safety, Guardrails & Human-in-the-Loop](13-Safety-Guardrails-and-Human-in-the-Loop/13-Safety-Guardrails-and-Human-in-the-Loop.md)

### Related concepts (canonical home is another section)
> These topics are foundations or applications of agents, but their canonical home is another section —
> linked here to avoid repetition.
- **Prompting & In-Context Learning · Chain-of-Thought · Fine-tuning / SFT · RLHF** → [LLMs](../09.%20LLMs/README.md) ([Prompting](../09.%20LLMs/16-Prompting-and-In-Context-Learning/16-Prompting-and-In-Context-Learning.md) · [Chain-of-Thought](../09.%20LLMs/17-Chain-of-Thought-Reasoning/17-Chain-of-Thought-Reasoning.md) · [SFT](../09.%20LLMs/13-Supervised-Fine-Tuning/13-Supervised-Fine-Tuning.md) · [RLHF & DPO](../09.%20LLMs/15-RLHF-and-DPO/15-RLHF-and-DPO.md))
- **Retrieval-Augmented Generation (RAG) & retrieval** → [RAG & LLM Applications](../11.%20RAG_and_LLM_Applications/README.md)
- **RL foundations (MDPs · policies · reward)** → [Reinforcement Learning](../08.%20Reinforcement_Learning/README.md)

## 🎓 Courses (free)
- [AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) — **DeepLearning.AI × LangChain** — free short course on agent loops.
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course) — **Hugging Face** — free, build agents with tools and memory.

## 🎥 Videos
- [Intro to LLMs + agents](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — the "LLM OS" framing.
- [How we build effective agents](https://www.youtube.com/watch?v=D7_ipDqhtwk) — **Anthropic** — workflows vs agents, in practice.

## 📄 Key Papers / Specs
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) — **Yao et al. (2022)** — the reason→act→observe loop.
- [Toolformer](https://arxiv.org/abs/2302.04761) — **Schick et al. (2023)** — models learning to call tools.
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) — **Anthropic** — the emerging tool-interface standard.

## 📰 Articles
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — **Lilian Weng** — the canonical survey (planning, memory, tools).

## 🔗 In this platform
- Math/mechanism: [AI-ML-intuition 8.03 Agents & Tool Use](../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
