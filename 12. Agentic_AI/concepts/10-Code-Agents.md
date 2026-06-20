---
id: "12-agentic-ai/code-agents"
topic: "Code Agents"
parent: "12-agentic-ai"
level: advanced
prereqs: ["llm-agents-overview", "tool-use-function-calling", "planning"]
interview_frequency: high
updated: 2026-06-20
---

# Code Agents
> Agents that **read, write, run, and debug code** in a real repo — locating the right files, editing
> them, running tests, and iterating on failures. They turn an LLM into something that can resolve a
> GitHub issue end-to-end. Measured by benchmarks like **SWE-bench**.

**Why it matters:** the highest-value, most-evaluated agent application (and the one powering coding
assistants). Interviews probe the loop (localize → edit → run tests → reflect → repeat), why
**execution feedback** (test results, stack traces) makes code agents far more reliable than
open-ended tasks, the design of the agent–computer interface (file/edit/shell tools), and why
SWE-bench scores are the field's headline metric.

**⭐ Start here — suggested path:**

1. **See the task & benchmark** — read ⭐ [SWE-bench](https://arxiv.org/abs/2310.06770). *Defines the "resolve a real GitHub issue" task that frames the whole area.*
2. **Study a concrete agent** — read [SWE-agent](https://arxiv.org/abs/2405.15793). *The agent–computer interface (file viewer, editor, shell) that drives strong SWE-bench results.*
3. **See production framing** — read [Raising the bar on SWE-bench Verified](https://www.anthropic.com/engineering/swe-bench-sonnet). *What a simple, well-designed scaffold + good tools achieves.*
4. **Watch one run for hours** — watch [Claude SDK: 24-Hour Coding Agent](https://www.youtube.com/watch?v=BGouphNN5hg). *Long-horizon code agents and where they break.*
5. **Build one** — do [Building Code Agents with smolagents](https://www.deeplearning.ai/courses/building-code-agents-with-hugging-face-smolagents). *Agents that write code as their action space.*

## 🎓 Courses (free)
- [Building Code Agents with Hugging Face smolagents](https://www.deeplearning.ai/courses/building-code-agents-with-hugging-face-smolagents) — **DeepLearning.AI × Hugging Face** — code-writing agents, free.
- [HF Agents Course — Code Agents](https://huggingface.co/learn/agents-course/unit2/smolagents/code_agents) — **Hugging Face** — agents whose actions are executable code.

## 🎥 Videos
- [Claude SDK: 24-Hour Coding Agent](https://www.youtube.com/watch?v=BGouphNN5hg) — **Cole Medin / Ray Fernando** — long-running code agents in practice.
- [How We Build Effective Agents](https://www.youtube.com/watch?v=D7_ipDqhtwk) — **Barry Zhang (Anthropic)** — tool/loop design that underlies coding agents.
- [Tips for Building AI Agents](https://www.youtube.com/watch?v=LP5OCa20Zpg) — **Anthropic** — designing agent–environment interfaces (incl. shell/file tools).
- [Learn to Build Effective Agentic AI Systems](https://www.youtube.com/watch?v=w7vqXL4PWEE) — **Andrew Ng (DeepLearning.AI)** — evaluation-driven agent development.

## 📄 Key Papers
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) — **Jimenez et al. (2023)** — the canonical code-agent benchmark.
- [SWE-agent: Agent–Computer Interfaces Enable Software Engineering](https://arxiv.org/abs/2405.15793) — **Yang et al. (2024)** — tool design that drives strong issue-resolution.
- [Voyager: An Open-Ended Embodied Agent with LLMs](https://arxiv.org/abs/2305.16291) — **Wang et al. (2023)** — writing & reusing code as a growing skill library.

## 📰 Articles / Blogs (free, no paywall)
- [SWE-bench leaderboard & docs](https://www.swebench.com/) — **SWE-bench team** — the benchmark, tasks, and current standings.
- [Raising the bar on SWE-bench Verified](https://www.anthropic.com/engineering/swe-bench-sonnet) — **Anthropic** — scaffold and tool design for code agents.
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — **Anthropic** — agentic patterns that coding agents instantiate.

## 📚 Books (free, with chapters)
- [Artificial Intelligence: A Modern Approach — **Ch. 2 "Intelligent Agents"**](https://aima.cs.berkeley.edu/) — **Russell & Norvig** — the percept→action loop; a code agent's environment is the repo + shell.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
- Prev / next: [09 Agent Frameworks](09-Agent-Frameworks.md) · [11 Computer-Use & GUI Agents](11-Computer-Use-and-GUI-Agents.md) · [12 Agent Evaluation & Benchmarks](12-Agent-Evaluation-and-Benchmarks.md)
- Related (canonical home): [Prompting & In-Context Learning](../../09.%20LLMs/concepts/16-Prompting-and-In-Context-Learning.md)
