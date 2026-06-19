---
id: "15-agentic-ai/computer-use"
topic: "Computer-Use & GUI Agents"
parent: "15-agentic-ai"
level: advanced
prereqs: ["llm-agents-overview", "tool-use-function-calling"]
interview_frequency: medium
updated: 2026-06-20
---

# Computer-Use & GUI Agents
> Agents that operate a **graphical interface** like a person — looking at a screenshot, then issuing
> mouse/keyboard actions (click, type, scroll) to use websites and desktop apps. The action space is
> pixels and UI elements, not a clean API, which makes perception and grounding the hard part.

**Why it matters:** the frontier of "agents that can do anything a human can on a computer." Be ready
to explain why GUI control is harder than API tool use (visual grounding, brittle layouts, long
horizons, no structured feedback), the screenshot→action loop, why benchmarks like **OSWorld** and
**WebArena** matter, and the outsized safety risk (an agent clicking around with real credentials).

**⭐ Start here — suggested path:**

1. **See it work** — read ⭐ [Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use). *The clearest demo of screenshot-in → action-out.*
2. **Understand the interface** — read [Computer use tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool). *The actual action space (screenshot, click, type) an agent is given.*
3. **Watch a real run** — watch [Claude — Computer use for automating operations](https://www.youtube.com/watch?v=ODaHJzOyVCQ). *See the loop and its failure modes.*
4. **Learn the benchmark** — read [OSWorld](https://arxiv.org/abs/2404.07972). *How GUI agents are measured on real desktop/web tasks.*
5. **Survey the field** — read [GUI Agents with Foundation Models: A Survey](https://arxiv.org/abs/2411.04890). *Perception, grounding, and action-space designs across systems.*

## 🎓 Courses (free)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit1/introduction) — **Hugging Face** — agent loops and tools that GUI agents extend to UI actions.
- [AI Agents in LangGraph](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph) — **DeepLearning.AI × LangChain** — the stateful loop GUI agents run on.

## 🎥 Videos
- [Claude — Computer use for automating operations](https://www.youtube.com/watch?v=ODaHJzOyVCQ) — **Anthropic** — the screenshot→action loop demonstrated.
- [Claude's Computer Use Is A Game Changer | YC Decoded](https://www.youtube.com/watch?v=VDmU0jjklBo) — **Y Combinator** — what GUI control unlocks and its limits.
- [Claude NEW Computer Use in 6 Minutes](https://www.youtube.com/watch?v=ZUBJqLGKoZI) — **Developers Digest** — quick concrete walkthrough.
- [Tips for Building AI Agents](https://www.youtube.com/watch?v=LP5OCa20Zpg) — **Anthropic** — designing reliable action loops (applies directly to GUI control).

## 📄 Key Papers
- [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks](https://arxiv.org/abs/2404.07972) — **Xie et al. (2024)** — real computer-environment benchmark for GUI agents.
- [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854) — **Zhou et al. (2023)** — agents acting on realistic websites.
- [GUI Agents with Foundation Models: A Survey](https://arxiv.org/abs/2411.04890) — **Wang et al. (2024)** — perception, grounding, and action spaces, surveyed.

## 📰 Articles / Blogs (free, no paywall)
- [Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — **Anthropic** — the launch post with demos and caveats.
- [Computer use tool documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool) — **Anthropic** — the action space and safety guidance.
- [OSWorld project page](https://os-world.github.io/) — **OSWorld team** — tasks, environments, and leaderboard.

## 📚 Books (free, with chapters)
- [Artificial Intelligence: A Modern Approach — **Ch. 2 "Intelligent Agents"** (sensors & actuators)](https://aima.cs.berkeley.edu/) — **Russell & Norvig** — GUI agents are the literal sensor (screenshot) + actuator (click) agent.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
- Prev / next: [10 Code Agents](10-Code-Agents.md) · [12 Agent Evaluation & Benchmarks](12-Agent-Evaluation-and-Benchmarks.md) · [13 Safety, Guardrails & HITL](13-Safety-Guardrails-and-Human-in-the-Loop.md)
- Related (canonical home): [Prompting & In-Context Learning](../../08.%20LLMs/concepts/16-Prompting-and-In-Context-Learning.md)
