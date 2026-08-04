---
id: "16-agentic-ai/react/references"
topic: "ReAct — References"
parent: "16-agentic-ai/react"
type: references
updated: 2026-07-02
---

# ReAct (Reason + Act) — references and further reading

> Companion link library for **[ReAct — Reason + Act](reason-and-act.md)** (the concept
> page). Kept separate so it can be reused as a standalone reference list. Grouped by type,
> best-first. Everything here is **free / open** — no paywall. Every source cited under a
> "Source / derivation" line on the concept page appears here, so each claim is traceable to a
> primary source. Chosen for depth on *this* topic, not popularity.

**⭐ Start here — suggested path**:
1. **See the pattern** — open the ⭐ [ReAct project page](https://react-lm.github.io/) (**Yao et al.**). *The Thought/Action/Observation trace shown on real tasks — the fastest way to "get it."*
2. **Read the source** — read [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (**Yao et al., 2022**). *The actual claims and ablations: why interleaving beats reason-only and act-only.*
3. **Get the prompt** — read [ReAct Prompting](https://www.promptingguide.ai/techniques/react) (**Prompt Engineering Guide**). *The exact template you'd write on a whiteboard, worked end to end.*
4. **Watch it run** — watch [Building a ReAct Agent from scratch](https://www.youtube.com/watch?v=hKVhRA9kfeM) (**Sam Witteveen**). *The loop executing against tools in code, including where it breaks.*
5. **Place it among patterns** — read [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) (**Anthropic**). *When a simple ReAct loop is enough vs when you need planning or orchestration.*

**🎥 Videos** (trusted creators only):
- [Building a ReAct Agent from Scratch](https://www.youtube.com/watch?v=hKVhRA9kfeM) — **Sam Witteveen** — implements the Thought/Action/Observation loop over tools in plain Python; closest to this chapter's build.
- [ReAct: Reasoning + Acting with LLMs (paper walkthrough)](https://www.youtube.com/watch?v=Eug2clsLtFs) — **Sam Witteveen** — a clear read of the ReAct prompt and loop with running code.
- [How We Build Effective Agents](https://www.youtube.com/watch?v=D7_ipDqhtwk) — **Barry Zhang (Anthropic)** — situates ReAct-style loops among production agent patterns; when to use which.
- [AI Agents Fundamentals (the reason–act–observe cycle)](https://www.youtube.com/watch?v=qU3fmidNbJE) — **Tina Huang** — the loop explained from zero, no framework.
- [Let's build a ReAct agent (LangGraph)](https://www.youtube.com/watch?v=uRya4zRrRx4) — **LangChain** — the same loop expressed as a graph with tool nodes and a termination condition.

**🎓 Courses (free)**:
- [Hugging Face Agents Course — Unit 1 (the loop)](https://huggingface.co/learn/agents-course/unit1/introduction) — **Hugging Face** — builds the Thought/Action/Observation cycle hands-on, tool by tool.
- [AI Agents in LangGraph](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph) — **DeepLearning.AI × LangChain** — implements a ReAct agent from scratch, then with a framework; the exact reason→act→observe graph.
- [Functions, Tools and Agents with LangChain](https://learn.deeplearning.ai/courses/functions-tools-agents-langchain) — **DeepLearning.AI × LangChain** — how the text-parsed actions of ReAct evolve into structured tool/function calling.

**📰 Articles / blogs (free, no paywall)**:
- [ReAct project page](https://react-lm.github.io/) — **Yao et al.** — interactive examples of the Thought/Action/Observation trace on HotpotQA and ALFWorld; the reference visualisation of the pattern.
- [ReAct Prompting](https://www.promptingguide.ai/techniques/react) — **Prompt Engineering Guide** — the prompt template and a worked example, free and copy-pasteable.
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — **Anthropic** — the "when is a ReAct loop the right level of complexity?" question, answered with a taxonomy of agent patterns.
- [ReActAgent — implementation docs](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/) — **LlamaIndex** — a production ReAct agent's code and prompt, a good cross-check against this chapter's module.

**📄 Key papers**:
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — **Yao et al. (2022), ICLR 2023**. — the source: introduces the interleaved reason+act paradigm, the Thought/Action/Observation prompt, and the ablations showing the *interleaving* (not either half) drives the gains, largely by reducing hallucinated facts through observation grounding. *(Every "Source / derivation" on the concept page traces here or below.)*
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — **Wei et al. (2022), NeurIPS 2022**. — the reasoning half of ReAct; read it to see exactly what ReAct *adds* (an acting channel + real observations) on top of pure step-by-step reasoning.
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — **Schick et al. (2023)**. — the forward-link where tool use moves from *prompt* (ReAct) into *weights*: the model self-supervises when/how to call APIs. Same "act on real results" idea, learned rather than prompted.
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) — **Shinn et al. (2023), NeurIPS 2023**. — wraps a ReAct inner loop with an outer self-reflection loop: on failure the agent writes a critique and retries, improving over trials with no weight updates.

**📚 Books (free chapters / full PDFs)**:
- [Artificial Intelligence: A Modern Approach — Ch. 2 "Intelligent Agents"](https://aima.cs.berkeley.edu/) — **Russell & Norvig** — the percept→action agent loop that ReAct is a modern, language-based instance of; the classical grounding for "an agent senses and acts."
- [Speech and Language Processing (3rd ed. draft) — chapters on prompting & agents](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — free draft; the LLM-prompting and tool-use context ReAct sits in, from an academic NLP viewpoint.

**🔗 In this platform**:
- Concept page (full explanation): [ReAct — Reason + Act](reason-and-act.md)
- Fast buzzword-level overview (the recall layer): [AI Buzzword Knowledge — Agents](../../../../AI-Buzzword-Knowledge/05-Agents.md) · [AI Buzzword Knowledge — Agentic Workflows](../../../../AI-Buzzword-Knowledge/06-Agentic-Workflows.md) · [AI Buzzword Knowledge — Tool Use & MCP](../../../../AI-Buzzword-Knowledge/08-Tool-Use-and-MCP.md)
- Prerequisite (the reasoning half): [Chain-of-Thought Reasoning](../../reasoning-evaluation-and-alignment/chain-of-thought-and-reasoning/chain-of-thought-and-reasoning.md) · prompting foundations in [09 LLMs — Prompting](../../README.md)
- Prev / next in this domain: [01 LLM Agents Overview](../agent-foundations/agent-foundations.md) · [03 Tool Use & Function Calling](../tool-use/tool-use.md) · [04 Planning & Task Decomposition](../planning/planning.md)
- Where it goes next: [05 Reflection & Self-Critique](../reflection/reflection.md) (the Reflexion outer loop) · [06 Memory for Agents](../memory/memory.md) (when the scratchpad outgrows the context)
- Tool + safety context: [08 Model Context Protocol (MCP)](../model-context-protocol/model-context-protocol.md) · [13 Safety, Guardrails & Human-in-the-Loop](../agent-safety/agent-safety.md)
- Frameworks that implement this loop: [09 Agent Frameworks](../agent-frameworks/agent-frameworks.md)
