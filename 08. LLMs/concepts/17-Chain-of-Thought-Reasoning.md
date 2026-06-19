---
id: "08-llms/chain-of-thought-reasoning"
topic: "Chain-of-Thought Reasoning"
parent: "08-llms"
level: advanced
prereqs: ["prompting-and-in-context-learning", "decoding-and-sampling"]
interview_frequency: high
updated: 2026-06-20
---

# Chain-of-Thought Reasoning
> Ask the model to "think step by step" and it generates intermediate reasoning **before** the answer —
> dramatically improving multi-step math, logic, and commonsense tasks. Variants: few-shot CoT,
> zero-shot CoT ("Let's think step by step"), **self-consistency** (sample many chains, majority-vote),
> and tree/graph-of-thought search. The seed of modern "reasoning models."

**Why it matters:** a hot interview topic. Be ready to explain *why* CoT helps (more compute/tokens
spent before answering; decomposition), distinguish few-shot vs zero-shot CoT, describe
self-consistency, and discuss faithfulness (the stated reasoning may not be the real cause of the answer).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Chain-of-thought prompting — Explained!](https://www.youtube.com/watch?v=AFE6x81AP4k). *What CoT is and why it boosts reasoning.*
2. **See the prompting context** — [Karpathy: Deep Dive into LLMs](https://www.youtube.com/watch?v=7xTGNNLPyMI). *Why "thinking tokens" before the answer help, in the bigger picture.*
3. **Read the source** — [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903). *The few-shot CoT result on math/logic benchmarks.*
4. **Read the simple variant** — [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916). *"Let's think step by step" — zero-shot CoT.*
5. **Boost reliability** — [Self-Consistency Improves CoT Reasoning](https://arxiv.org/abs/2203.11171). *Sample many chains, vote — a key practical upgrade.*

## 🎓 Courses (free)
- [Prompt Engineering Guide — CoT & reasoning](https://www.promptingguide.ai/) — **DAIR.AI** — CoT, zero-shot CoT, self-consistency, tree-of-thought.
- [Stanford CS336 — Reasoning & post-training](https://stanford-cs336.github.io/spring2025/) — **Stanford** — reasoning behavior and how it's elicited/trained.

## 🎥 Videos
- [Chain-of-thought prompting — Explained!](https://www.youtube.com/watch?v=AFE6x81AP4k) — **CodeEmporium** — the cleanest intro to CoT.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — why intermediate tokens before the answer help.
- [Prompt Engineering Tutorial](https://www.youtube.com/watch?v=_ZvnD73m40o) — **freeCodeCamp** — CoT among prompting techniques, with examples.
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — "system 2" reasoning framing for LLMs.

## 📄 Key Papers
- [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903) — **Wei et al. (2022)** — the founding few-shot CoT result.
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) — **Kojima et al. (2022)** — zero-shot CoT ("Let's think step by step").
- [Self-Consistency Improves Chain-of-Thought Reasoning](https://arxiv.org/abs/2203.11171) — **Wang et al. (2022)** — sample diverse chains and majority-vote.

## 📰 Articles / Blogs (free, no paywall)
- [Prompt Engineering (controllable generation)](https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/) — **Lilian Weng** — reasoning-eliciting prompt techniques.
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — **Lilian Weng** — CoT/ReAct as the planning core of agents.
- [Prompt Engineering Guide — reasoning techniques](https://www.promptingguide.ai/) — **DAIR.AI** — CoT variants compared.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 12 "Model Alignment, Prompting & In-Context Learning"**](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — prompting/reasoning chapter.
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — §6.2 chain-of-thought and reasoning, free reference.

## 🔗 In this platform
- Concept depth (the *why*): [Module 8.01 In-Context Learning & Prompting](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md) · [Module 8.03 Agents & Tool Use](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
- Related concepts: [Prompting & In-Context Learning](16-Prompting-and-In-Context-Learning.md) · [Decoding & Sampling](18-Decoding-and-Sampling.md) · [LLM Evaluation & Benchmarks](19-LLM-Evaluation-and-Benchmarks.md)
