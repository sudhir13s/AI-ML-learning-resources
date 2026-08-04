---
id: "16-agentic-ai"
topic: "Agentic AI & Tool Use"
level: advanced
built_from: ["llms"]
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
1. ✅ [LLM Agents — Overview & the Agent Loop](agent-foundations/agent-foundations.md)
2. ✅ [ReAct — Reason + Act](reason-and-act/reason-and-act.md)
3. ✅ [Tool Use & Function Calling](tool-use/tool-use.md)

### Reasoning, planning & memory
4. ✅ [Planning — Task Decomposition & Plan-and-Execute](planning/planning.md)
5. ✅ [Reflection & Self-Critique](reflection/reflection.md)
6. ✅ [Memory for Agents (short- & long-term)](memory/memory.md)

### Systems, protocols & frameworks
7. ✅ [Multi-Agent Systems & Orchestration](multi-agent-systems/multi-agent-systems.md)
8. ✅ [Model Context Protocol (MCP)](model-context-protocol/model-context-protocol.md)
9. ✅ [Agent Frameworks (LangGraph, etc., conceptual)](agent-frameworks/agent-frameworks.md)

### Applied agents
10. ✅ [Code Agents](coding-and-computer-use-agents/code-agents.md)
11. ✅ [Computer-Use & GUI Agents](coding-and-computer-use-agents/computer-use-and-gui-agents.md)

### Evaluation & safety
12. ✅ [Agent Evaluation & Benchmarks (AgentBench · SWE-bench)](agent-evaluation/agent-evaluation.md)
13. ✅ [Safety, Guardrails & Human-in-the-Loop](agent-safety/agent-safety.md)

### Related concepts (canonical home is another section)
> These topics are foundations or applications of agents, but their canonical home is another section —
> linked here to avoid repetition.
- **Prompting & In-Context Learning · Chain-of-Thought · Fine-tuning / SFT · RLHF** → [LLMs](../README.md) ([Prompting](../reasoning-evaluation-and-alignment/prompting-and-in-context-learning/prompting-and-in-context-learning.md) · [Chain-of-Thought](../reasoning-evaluation-and-alignment/chain-of-thought-and-reasoning/chain-of-thought-and-reasoning.md) · [SFT](../training-and-adaptation/supervised-fine-tuning/supervised-fine-tuning.md) · [RLHF & DPO](../training-and-adaptation/preference-and-alignment-training/preference-and-alignment-training.md))
- **Retrieval-Augmented Generation (RAG) & retrieval** → [RAG & LLM Applications](../rag-and-knowledge-systems/overview.md)
- **RL foundations (MDPs · policies · reward)** → [Reinforcement Learning](../../core-machine-learning/reinforcement-learning/README.md)

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
- Math/mechanism: [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/reasoning-and-agency/agents-and-tools/agent-loop-and-tool-use-intuition.md)
