---
id: "09-llms/hallucination-and-alignment-basics"
topic: "Hallucination & Alignment Basics"
parent: "09-llms"
level: advanced
prereqs: ["rlhf-and-dpo", "llm-evaluation-and-benchmarks", "decoding-and-sampling"]
interview_frequency: high
updated: 2026-06-20
---

# Hallucination & Alignment Basics
> **Hallucination**: fluent, confident output that is factually wrong or unsupported — a structural
> consequence of next-token prediction (and of training/eval that reward guessing over abstaining).
> **Alignment**: making models helpful, honest, and harmless (HHH) via RLHF/DPO, Constitutional AI,
> and guardrails. Mitigations include RAG (ground in retrieval), calibration/abstention, and red-teaming.

**Why it matters:** the safety/reliability interview that every applied LLM role asks. Be ready to
explain *why* LLMs hallucinate (no grounding; train/eval incentives), list mitigations (RAG, decoding,
self-consistency, abstention), and define alignment (HHH), RLHF vs Constitutional AI, and red-teaming.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Why Large Language Models Hallucinate](https://www.youtube.com/watch?v=cfqtFvWOfg0). *Types of hallucination, causes, and practical mitigations.*
2. **Read the deep dive** — [Lilian Weng: Extrinsic Hallucinations in LLMs](https://lilianweng.github.io/posts/2024-07-07-hallucination/). *A rigorous taxonomy + detection/mitigation survey.*
3. **Understand the incentive** — [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664). *Training/eval reward guessing over saying "I don't know."*
4. **Connect to alignment** — [Anthropic: Core Views on AI Safety](https://www.anthropic.com/research/core-views-on-ai-safety). *What "alignment" means and why it's hard.*
5. **See the mechanism of fixes** — [RLHF & DPO](../15-RLHF-and-DPO/15-RLHF-and-DPO.md) + ground with [RAG](../../11.%20RAG_and_LLM_Applications/concepts/README.md). *Preference tuning + retrieval are the main levers.*

## 🎓 Courses (free)
- [Stanford CS324 — Harms, safety & alignment](https://stanford-cs324.github.io/winter2022/) — **Stanford** — capabilities vs harms, alignment, and evaluation.
- [Stanford CS336 — Alignment & safety](https://stanford-cs336.github.io/spring2025/) — **Stanford** — alignment in the post-training pipeline.

## 🎥 Videos
- [Why Large Language Models Hallucinate](https://www.youtube.com/watch?v=cfqtFvWOfg0) — **IBM Technology** — the clearest intro to causes and mitigations.
- [Reinforcement Learning with Human Feedback (RLHF), Clearly Explained](https://www.youtube.com/watch?v=qPN_XZcJf_s) — **StatQuest** — the main alignment mechanism.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — where alignment and hallucination fit in the pipeline.
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — the "LLM OS" + safety/jailbreak discussion.

## 📄 Key Papers
- [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) — **Kalai et al. (2025)** — hallucination as a consequence of training/eval incentives.
- [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) — **Ji et al. (2022)** — the taxonomy + causes + metrics reference.
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — **Bai et al. (2022)** — RLAIF: align with a written constitution, less human labeling.

## 📰 Articles / Blogs (free, no paywall)
- [Extrinsic Hallucinations in LLMs](https://lilianweng.github.io/posts/2024-07-07-hallucination/) — **Lilian Weng** — the definitive free survey of hallucination + mitigations.
- [Core Views on AI Safety](https://www.anthropic.com/research/core-views-on-ai-safety) — **Anthropic** — what alignment is and why it matters.
- [Transformer Circuits — Mechanistic Interpretability](https://transformer-circuits.pub/2021/framework/index.html) — **Anthropic** — understanding model internals as a path to alignment.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 12 "Model Alignment, Prompting & In-Context Learning"**](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — alignment and its failure modes.
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — §5/§7 alignment and the hallucination/safety landscape (free reference).

## 🔗 In this platform
- Concept depth (the *why*): [Module 6.03 PPO and RLHF](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.03_PPO_and_RLHF.md)
- Applications (covered elsewhere): [Retrieval-Augmented Generation → RAG & LLM Applications](../../11.%20RAG_and_LLM_Applications/concepts/README.md)
- Related concepts: [RLHF & DPO](../15-RLHF-and-DPO/15-RLHF-and-DPO.md) · [LLM Evaluation & Benchmarks](../19-LLM-Evaluation-and-Benchmarks/19-LLM-Evaluation-and-Benchmarks.md) · [Chain-of-Thought Reasoning](../17-Chain-of-Thought-Reasoning/17-Chain-of-Thought-Reasoning.md)
