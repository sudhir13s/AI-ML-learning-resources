---
id: "09-llms/chain-of-thought-reasoning/references"
topic: "Chain-of-Thought Reasoning — References"
parent: "09-llms/chain-of-thought-reasoning"
type: references
updated: 2026-06-27
---

# Chain-of-Thought Reasoning — references and further reading

> Companion link library for **[Chain-of-Thought Reasoning](17-Chain-of-Thought-Reasoning.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity. All links are free / no-paywall.

**Start here — suggested path**:
1. **Build the intuition** — watch [Chain-of-Thought Prompting — Explained!](https://www.youtube.com/watch?v=AFE6x81AP4k) (**CodeEmporium**). *What CoT is and why writing steps before the answer helps.*
2. **Read the source** — [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903) (**Wei et al. 2022**). *The founding few-shot result and the emergent-at-scale finding.*
3. **See the minimal trigger** — [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) (**Kojima et al. 2022**). *"Let's think step by step" — zero-shot CoT, no examples.*
4. **Boost reliability** — [Self-Consistency Improves CoT Reasoning](https://arxiv.org/abs/2203.11171) (**Wang et al. 2022**). *Sample many chains, majority-vote — the key practical upgrade.*
5. **Connect to the frontier** — [Learning to Reason with LLMs (o1)](https://openai.com/index/learning-to-reason-with-llms/) (**OpenAI**). *CoT as a trained capability and a test-time-compute dial.*

**Videos**:
- [Chain-of-Thought Prompting — Explained!](https://www.youtube.com/watch?v=AFE6x81AP4k) — **CodeEmporium** — the cleanest short intro to what CoT is and why it boosts reasoning.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — why intermediate "thinking tokens" before the answer help, placed in the full LLM picture.
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — the "system 2" reasoning framing that motivates CoT and reasoning models.
- [Tree of Thoughts — Deliberate Problem Solving with LLMs](https://www.youtube.com/watch?v=ut5kp56wW_4) — **Yannic Kilcher** — a paper walkthrough of search-over-reasoning (ToT), the branching generalization of CoT.

**Interactive & visual**:
- [Prompt Engineering Guide — Chain-of-Thought](https://www.promptingguide.ai/techniques/cot) — **DAIR.AI** — side-by-side worked CoT / zero-shot-CoT / self-consistency examples you can copy and run.
- [LLM Visualizer (3D)](https://bbycroft.net/llm) — **Brendan Bycroft** — walk a token through a small GPT's forward pass; makes "each generated token is another full pass" concrete (the serial-compute argument for why CoT helps).

**Courses (free)**:
- [Stanford CS336 — Language Modeling from Scratch (reasoning & post-training)](https://stanford-cs336.github.io/spring2025/) — **Stanford** — how reasoning behavior is elicited and trained in the full LLM stack.
- [DeepLearning.AI — ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) — **Andrew Ng & OpenAI** — practical reasoning-prompt patterns including step-by-step prompting.

**Articles / blogs (free, no paywall)**:
- [Language Models Perform Reasoning via Chain of Thought](https://research.google/blog/language-models-perform-reasoning-via-chain-of-thought/) — **Google Research** — the accessible summary of the founding CoT result, with the GSM8K numbers.
- [Learning to Reason with LLMs (o1)](https://openai.com/index/learning-to-reason-with-llms/) — **OpenAI** — CoT trained into long internal chains; the test-time-compute paradigm.
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — **Lilian Weng (OpenAI)** — CoT / ReAct as the planning core of agents.
- [Prompt Engineering Guide — reasoning techniques](https://www.promptingguide.ai/) — **DAIR.AI** — CoT, zero-shot CoT, self-consistency, ToT compared with examples.

**Key papers**:
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — **Wei et al. (2022)** — the founding few-shot CoT result; emergent at scale; the source of the PaLM-540B GSM8K ~18%→~57% jump.
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) — **Kojima et al. (2022)** — zero-shot CoT via "Let's think step by step."
- [Self-Consistency Improves Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) — **Wang et al. (2022)** — sample diverse chains and majority-vote (GSM8K 56.6%→74.4% on PaLM-540B).
- [Least-to-Most Prompting Enables Complex Reasoning in LLMs](https://arxiv.org/abs/2205.10625) — **Zhou et al. (2022)** — decompose into ordered sub-questions, solve in sequence.
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) — **Yao et al. (2023)** — search over a tree of reasoning steps with backtracking.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — **Yao et al. (2022)** — interleave reasoning with tool actions; the planning core of agents.
- [PAL: Program-aided Language Models](https://arxiv.org/abs/2211.10435) — **Gao et al. (2022)** — emit code and run it; offload exact computation from the chain.
- [Language Models Don't Always Say What They Think (unfaithful CoT)](https://arxiv.org/abs/2305.04388) — **Turpin et al. (2023)** — CoT explanations can be systematically unfaithful to the true computation.
- [Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective](https://arxiv.org/abs/2305.15408) — **Feng et al. (2023)** — why intermediate tokens increase a transformer's expressive power.
- [The Expressive Power of Transformers with Chain of Thought](https://arxiv.org/abs/2310.07923) — **Merrill & Sabharwal (2023)** — formal serial-depth gains from chain-of-thought decoding.
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Parameters](https://arxiv.org/abs/2408.03314) — **Snell et al. (2024)** — when spending inference compute (samples / longer chains) beats a bigger model.

**Books (free, with chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 12 "Model Alignment, Prompting & In-Context Learning"](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — the prompting/reasoning chapter, free PDF.
- [A Survey of Large Language Models — §6.2 Chain-of-Thought Reasoning](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — CoT and reasoning techniques surveyed with citations.

**In this platform**:
- Concept page (full explanation): [Chain-of-Thought Reasoning](17-Chain-of-Thought-Reasoning.md)
- The general technique it specializes: [Prompting & In-Context Learning](../16-Prompting-and-In-Context-Learning/16-Prompting-and-In-Context-Learning.md) — CoT is *one* prompting technique; that page covers the broader inference-time conditioning toolkit.
- Compressing reasoning into a smaller model: [Knowledge Distillation](../11-Knowledge-Distillation/11-Knowledge-Distillation.md) — the mechanism behind *CoT distillation*.
- What controls the chains at decode time: [Decoding & Sampling](../18-Decoding-and-Sampling/18-Decoding-and-Sampling.md) — temperature/sampling is what makes self-consistency's chains diverge.
- How reasoning is measured: [LLM Evaluation & Benchmarks](../19-LLM-Evaluation-and-Benchmarks/19-LLM-Evaluation-and-Benchmarks.md) — GSM8K and friends, and the answer-extraction pitfall.
- Intuition track: [Module 8.01 In-Context Learning & Prompting](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md) · [Module 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
