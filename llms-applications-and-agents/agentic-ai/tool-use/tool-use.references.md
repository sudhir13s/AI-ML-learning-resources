---
id: "16-agentic-ai/tool-use-function-calling/references"
topic: "Tool Use & Function Calling — References"
parent: "16-agentic-ai/tool-use-function-calling"
type: references
updated: 2026-07-02
---

# Tool Use & Function Calling — references and further reading

> Companion link library for **[Tool Use & Function Calling](tool-use.md)** (the
> concept page). Kept separate so it can be reused as a standalone reference list. Grouped by type,
> best-first. Everything here is **free / open** — no paywall. Every source cited under a
> "Source / derivation" line on the concept page appears here, so each claim is traceable to a primary
> source. Chosen for depth on *this* topic, not popularity.

**⭐ Start here — suggested path**:
1. **See the contract** — read ⭐ [OpenAI: Function Calling guide](https://platform.openai.com/docs/guides/function-calling) (**OpenAI**). *The clearest spec of tool schemas and the request/response round-trip — the pattern this chapter mirrors.*
2. **Compare a second implementation** — read [Anthropic: Tool Use overview](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) (**Anthropic**). *Two vendors converging on `tool_use`/`tool_result` reveals what's essential vs incidental.*
3. **See it on an open model** — read [Qwen: Function Calling / Hermes-style tool template](https://qwen.readthedocs.io/en/latest/framework/function_call.html) (**Qwen team**). *Exactly the `<tool_call>` chat-template convention the runnable code on this page uses.*
4. **Understand where the ability comes from** — read [Toolformer](https://arxiv.org/abs/2302.04761) (**Schick et al., 2023**). *Tool use as a learned competence, not a prompt trick.*
5. **See how "correct call" is measured** — open the [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) (**Patil et al.**). *The large-scale version of this chapter's structured-vs-text reliability test.*

**🎥 Videos** (trusted creators only):
- [OpenAI Function Calling — full beginner tutorial](https://www.youtube.com/watch?v=aqdWSYWC_LI) — the schema → `tool_calls` → `role: tool` round-trip in plain code, closest to this chapter's build.
- [Tips for Building AI Agents](https://www.youtube.com/watch?v=LP5OCa20Zpg) — **Anthropic** — designing good tool *schemas* and interfaces, and handling failures — the highest-leverage part of tool use.
- [How We Build Effective Agents](https://www.youtube.com/watch?v=D7_ipDqhtwk) — **Barry Zhang (Anthropic)** — tool design and the agent–environment feedback loop; when tool calling is enough.
- [Gorilla LLM: Teach LLMs to Use Tools at Scale](https://www.youtube.com/watch?v=iz-ITyzteE8) — **Shishir Patil (Gorilla / UC Berkeley)** — the first author on teaching LLMs to emit correct, executable API calls at scale; the dedicated deep pointer for the *measured-reliability* axis (the Berkeley Function-Calling Leaderboard).
- [Understanding ReAct with LangChain](https://www.youtube.com/watch?v=Eug2clsLtFs) — **Sam Witteveen** — the text-parsed tool loop this chapter contrasts with; watch it to feel the brittleness structured calling removes.

**🎓 Courses (free)**:
- [Function-Calling and Data Extraction with LLMs](https://learn.deeplearning.ai/courses/function-calling-and-data-extraction-with-llms) — **DeepLearning.AI × Nexusflow** — schemas, structured calls, parallel calls, and extraction, hands-on.
- [Functions, Tools and Agents with LangChain](https://learn.deeplearning.ai/courses/functions-tools-agents-langchain) — **DeepLearning.AI × LangChain** — exactly how the text-parsed actions of ReAct evolve into structured tool/function calling.
- [Hugging Face Agents Course — Unit 1 (tools)](https://huggingface.co/learn/agents-course/unit1/introduction) — **Hugging Face** — defining tools and wiring them into an agent, tool by tool, free.

**📰 Articles / docs (free, no paywall)**:
- [Function Calling guide](https://platform.openai.com/docs/guides/function-calling) — **OpenAI** — the canonical tool-schema + call round-trip spec; parallel calls, forced calls, and feeding results back.
- [Tool Use overview](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — **Anthropic** — tool definitions, `tool_use`/`tool_result` content blocks, and best practices for tool design.
- [Chat templates (incl. tool use)](https://huggingface.co/docs/transformers/main/en/chat_templating) — **Hugging Face** — how `apply_chat_template(tools=...)` renders JSON schemas into the prompt; the exact mechanism the runnable code uses.
- [Qwen: Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html) — **Qwen team** — the `<tool_call>` template convention and the tool-calling message roles for the open model on this page.
- [OpenAI Cookbook: How to call functions with chat models](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models) — **OpenAI** — runnable examples incl. parallel calls, argument validation, and result-passing.

**📄 Key papers**:
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — **Schick et al. (2023)**. — the source for "the model natively knows how to call tools": a model fine-tuned to insert API calls into its own generations, self-supervised by whether a call reduced its loss. *(The concept page's first "Source / derivation" traces here.)*
- [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334) — **Patil et al. (2023)**. — trains + evaluates LLMs on producing *correct, executable* API calls and introduces the AST-matching evaluation behind the Berkeley Function-Calling Leaderboard — the formal version of this chapter's "did we get a dispatchable call?" axis. *(The concept page's second "Source / derivation" traces here.)*
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — **Yao et al. (2022), ICLR 2023**. — the text-protocol contrast: tool calls as free-text actions inside a reasoning loop. Read it to see exactly what structured function calling *replaces* (prose parsing) and *keeps* (the reason→act→observe loop).
- [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends](https://arxiv.org/abs/2303.17580) — **Shen et al. (2023)**. — an LLM orchestrating many specialised tools/models via structured calls; a vivid picture of where reliable tool calling leads.

**📚 Books (free chapters / full PDFs)**:
- [Artificial Intelligence: A Modern Approach — Ch. 2 "Intelligent Agents"](https://aima.cs.berkeley.edu/) — **Russell & Norvig** — actuators/effectors: tools are the LLM agent's way of acting on the environment; the classical grounding for "an agent acts through a defined interface."
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — free draft; the LLM-prompting and tool-use context function calling sits in, from an academic NLP viewpoint.

**🔗 In this platform**:
- Concept page (full explanation): [Tool Use & Function Calling](tool-use.md)
- Fast buzzword-level overview (the recall layer): [AI Buzzword Knowledge — Tool Use & MCP](../../../../AI-Buzzword-Knowledge/08-Tool-Use-and-MCP.md) · [AI Buzzword Knowledge — Agents](../../../../AI-Buzzword-Knowledge/05-Agents.md)
- The text-protocol contrast (read these together): [02 ReAct — Reason + Act](../reason-and-act/reason-and-act.md) — free-text actions parsed by regex; *this* page is the structured evolution.
- Where tools get standardised across the ecosystem: [08 Model Context Protocol (MCP)](../model-context-protocol/model-context-protocol.md) — the forward-link; function calling is how *one model* calls a tool, MCP is how the *ecosystem* shares tools.
- Prev / next in this domain: [02 ReAct](../reason-and-act/reason-and-act.md) · [04 Planning & Task Decomposition](../planning/planning.md)
- Safety context (the tool-input + tool-output trust boundary): [13 Safety, Guardrails & Human-in-the-Loop](../agent-safety/agent-safety.md)
- Frameworks that implement this loop: [09 Agent Frameworks](../agent-frameworks/agent-frameworks.md)
- Related (canonical home for prompting foundations): [Prompting & In-Context Learning](../../reasoning-evaluation-and-alignment/prompting-and-in-context-learning/prompting-and-in-context-learning.md)
