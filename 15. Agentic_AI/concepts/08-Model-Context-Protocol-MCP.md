---
id: "15-agentic-ai/mcp"
topic: "Model Context Protocol (MCP)"
parent: "15-agentic-ai"
level: advanced
prereqs: ["tool-use-function-calling"]
interview_frequency: high
updated: 2026-06-20
---

# Model Context Protocol (MCP)
> An **open standard** for connecting LLM apps to tools and data. Instead of bespoke integrations per
> app, an MCP **client** (the agent host) talks to MCP **servers** that expose **tools**,
> **resources**, and **prompts** over a common protocol — the "USB-C port for AI."

**Why it matters:** MCP is the fast-rising answer to "how do agents integrate with the world without
N×M custom glue?" Be ready to explain the **client/server/host** architecture, the three primitives
(tools / resources / prompts), how it differs from raw function calling (a *standard transport &
discovery layer*, not a replacement for the model deciding what to call), and the security surface
(servers run code; prompt injection via resources).

**⭐ Start here — suggested path:**

1. **Get the why** — read ⭐ [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol). *The motivation and the USB-C-for-AI framing in two minutes.*
2. **See the architecture** — read [MCP — Getting Started / Intro](https://modelcontextprotocol.io/docs/getting-started/intro). *Client/server/host and the tools/resources/prompts primitives, from the spec.*
3. **Watch the clear explainer** — watch [Model Context Protocol (MCP), clearly explained](https://www.youtube.com/watch?v=7j_NE6Pjv-E). *Why it matters and how the pieces connect.*
4. **Hear it from the source** — watch [The Model Context Protocol (MCP)](https://www.youtube.com/watch?v=CQywdSdi5iA). *Anthropic walking through goals and design.*
5. **Place it vs function calling** — re-read [Tool Use overview](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview). *MCP standardizes *how tools are served*; the model still chooses calls via tool use.*

## 🎓 Courses (free)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit1/introduction) — **Hugging Face** — tools and integrations, the layer MCP standardizes.
- [MCP — Getting Started](https://modelcontextprotocol.io/docs/getting-started/intro) — **MCP / Anthropic** — official walkthrough: build a client and a server.

## 🎥 Videos
- [Model Context Protocol (MCP), clearly explained](https://www.youtube.com/watch?v=7j_NE6Pjv-E) — **Greg Isenberg** — why MCP matters and how it fits agents.
- [The Model Context Protocol (MCP)](https://www.youtube.com/watch?v=CQywdSdi5iA) — **Anthropic** — design and goals from the team that built it.
- [What is MCP? Integrate AI Agents with Databases & APIs](https://www.youtube.com/watch?v=eur8dUO9mvE) — **IBM Technology** — concise architecture overview.
- [Model Context Protocol (MCP) Explained in 20 Minutes](https://www.youtube.com/watch?v=N3vHJcHBS-w) — **Shaw Talebi** — end-to-end with a worked example.

## 📄 Key Papers / Specs
- [Model Context Protocol — Specification](https://modelcontextprotocol.io/) — **Anthropic / MCP** — the protocol itself (transport, primitives, lifecycle).
- [Tool Use overview](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — **Anthropic** — the function-calling layer MCP servers plug into.
- [Toolformer](https://arxiv.org/abs/2302.04761) — **Schick et al. (2023)** — background on models learning to call external tools.

## 📰 Articles / Blogs (free, no paywall)
- [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — **Anthropic** — the launch post and rationale.
- [MCP — Getting Started / Intro](https://modelcontextprotocol.io/docs/getting-started/intro) — **MCP** — architecture and primitives, free and open.
- [What is Model Context Protocol (MCP)?](https://www.ibm.com/think/topics/model-context-protocol) — **IBM** — neutral overview of the standard.

## 📚 Books (free, with chapters)
- [Artificial Intelligence: A Modern Approach — **Ch. 2 "Intelligent Agents"** (agent–environment interface)](https://aima.cs.berkeley.edu/) — **Russell & Norvig** — MCP is a modern standard for the sensor/actuator interface between agent and environment.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
- Prev / next: [03 Tool Use & Function Calling](03-Tool-Use-and-Function-Calling.md) · [09 Agent Frameworks](09-Agent-Frameworks.md)
- Related (canonical home): [Prompting & In-Context Learning](../../08.%20LLMs/concepts/16-Prompting-and-In-Context-Learning.md)
