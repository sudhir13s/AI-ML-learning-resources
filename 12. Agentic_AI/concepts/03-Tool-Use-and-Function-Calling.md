---
id: "12-agentic-ai/tool-use-function-calling"
topic: "Tool Use & Function Calling"
parent: "12-agentic-ai"
level: advanced
prereqs: ["llm-agents-overview"]
interview_frequency: very-high
updated: 2026-06-20
---

# Tool Use & Function Calling
> Give the model a set of typed **tool schemas** (name, description, JSON parameters); the model
> emits a structured **call** with arguments; your code runs it and feeds the **result** back. This
> is how an agent touches the outside world — search, databases, code execution, APIs.

**Why it matters:** the practical core of every agent. Expect questions on the **schema → call →
execute → observe** round-trip, how the model is trained/prompted to produce *valid* arguments,
**parallel** tool calls, error handling and retries, and the security surface (the model picks the
tool, so untrusted output can drive injection). Function calling is also what makes outputs
*structured* and reliable.

**⭐ Start here — suggested path:**

1. **See the contract** — read ⭐ [OpenAI: Function Calling guide](https://platform.openai.com/docs/guides/function-calling). *The clearest spec of tool schemas and the request/response round-trip.*
2. **Compare a second implementation** — read [Anthropic: Tool Use overview](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview). *Seeing two vendors' APIs reveals what's essential vs incidental.*
3. **Understand how models learn it** — read [Toolformer](https://arxiv.org/abs/2302.04761). *Where the ability to decide when/how to call a tool comes from.*
4. **Watch a build** — watch [Understanding ReAct with LangChain](https://www.youtube.com/watch?v=Eug2clsLtFs). *Tool definitions wired into an agent loop, end to end.*
5. **Run real examples** — work through the [OpenAI Cookbook: calling functions](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models). *Hands-on parallel calls, argument validation, and feeding results back.*

## 🎓 Courses (free)
- [Hugging Face Agents Course — tools](https://huggingface.co/learn/agents-course/unit1/introduction) — **Hugging Face** — defining tools and wiring them into an agent, free.
- [AI Agents in LangGraph](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph) — **DeepLearning.AI × LangChain** — tool nodes, parallel calls, and routing.

## 🎥 Videos
- [Understanding ReAct with LangChain](https://www.youtube.com/watch?v=Eug2clsLtFs) — **Sam Witteveen** — defining tools and letting the model choose and call them.
- [How We Build Effective Agents](https://www.youtube.com/watch?v=D7_ipDqhtwk) — **Barry Zhang (Anthropic)** — tool design and the agent–environment feedback loop.
- [Tips for Building AI Agents](https://www.youtube.com/watch?v=LP5OCa20Zpg) — **Anthropic** — designing good tool interfaces and handling failures.
- [AI Agents Fundamentals in 21 Minutes](https://www.youtube.com/watch?v=qU3fmidNbJE) — **Tina Huang** — function calling explained alongside the full agent loop.

## 📄 Key Papers
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — **Schick et al. (2023)** — models learning when and how to call APIs.
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) — **Yao et al. (2022)** — tool calls as actions inside a reasoning loop.
- [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends](https://arxiv.org/abs/2303.17580) — **Shen et al. (2023)** — an LLM orchestrating many specialized tools/models.

## 📰 Articles / Blogs (free, no paywall)
- [Function Calling guide](https://platform.openai.com/docs/guides/function-calling) — **OpenAI** — the canonical tool-schema + call round-trip spec.
- [Tool Use overview](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — **Anthropic** — tool definitions, tool_use/tool_result, and best practices.
- [How to call functions with chat models](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models) — **OpenAI Cookbook** — runnable examples incl. parallel calls.

## 📚 Books (free, with chapters)
- [Artificial Intelligence: A Modern Approach — **Ch. 2 "Intelligent Agents"**](https://aima.cs.berkeley.edu/) — **Russell & Norvig** — actuators/effectors: tools are the LLM agent's way of acting on the environment.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
- Prev / next: [02 ReAct](02-ReAct-Reason-and-Act.md) · [08 Model Context Protocol (MCP)](08-Model-Context-Protocol-MCP.md) · [09 Agent Frameworks](09-Agent-Frameworks.md)
- Related (canonical home): [Prompting & In-Context Learning](../../09.%20LLMs/concepts/16-Prompting-and-In-Context-Learning.md)
