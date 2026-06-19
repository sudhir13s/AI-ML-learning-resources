---
id: "08-llms/llm-evaluation-and-benchmarks"
topic: "LLM Evaluation & Benchmarks"
parent: "08-llms"
level: advanced
prereqs: ["language-modeling-objectives", "perplexity"]
interview_frequency: high
updated: 2026-06-20
---

# LLM Evaluation & Benchmarks
> How do you know a model is good? **Intrinsic**: perplexity (how well it predicts held-out text).
> **Knowledge/reasoning benchmarks**: MMLU, GSM8K, BIG-bench, HellaSwag, ARC. **Holistic**: HELM.
> **Human/preference**: Chatbot Arena (Elo), MT-Bench, and **LLM-as-a-judge**. Each has blind spots —
> contamination, saturation, and gaming — so evaluation is itself an open research problem.

**Why it matters:** a practical interview staple. Be ready to define perplexity, explain why
multiple-choice benchmarks (MMLU) ≠ chat quality, describe pairwise/Elo and LLM-as-judge (and their
biases), and discuss train/test contamination and benchmark saturation.

**⭐ Start here — suggested path:**

1. **Get the landscape** — watch [What are LLM Benchmarks?](https://www.youtube.com/watch?v=kDY4TodQwbg). *MMLU, scoring, and what benchmarks do/don't measure.*
2. **See preference eval** — read [Chatbot Arena (LMSYS)](https://lmsys.org/blog/2023-05-03-arena/). *Elo ratings from human pairwise votes — the live leaderboard.*
3. **Understand holistic eval** — [HELM](https://crfm.stanford.edu/helm/). *Many scenarios × many metrics — the "transparency" framework.*
4. **Read on LLM-as-judge** — [LLM-as-a-Judge guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge). *Using strong LLMs to score outputs — and the biases to watch.*
5. **Run an eval** — [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). *The standard tool that powers the Open LLM Leaderboard.*

## 🎓 Courses (free)
- [Stanford CS324 — Evaluation & harms](https://stanford-cs324.github.io/winter2022/) — **Stanford** — how LMs are measured and where benchmarks mislead.
- [Hugging Face — Evaluate library & LLM eval](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — metrics and evaluation workflow.

## 🎥 Videos
- [What are Large Language Model (LLM) Benchmarks?](https://www.youtube.com/watch?v=kDY4TodQwbg) — **IBM Technology** — the clearest intro to benchmarks and scoring.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — how models are evaluated across the pipeline.
- [Reinforcement Learning with Human Feedback (RLHF), Clearly Explained](https://www.youtube.com/watch?v=qPN_XZcJf_s) — **StatQuest** — preference data, the basis for preference eval.
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — capabilities/limits framing for evaluation.

## 📄 Key Papers
- [Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300) — **Hendrycks et al. (2020)** — the 57-subject knowledge benchmark.
- [Beyond the Imitation Game (BIG-bench)](https://arxiv.org/abs/2206.04615) — **Srivastava et al. (2022)** — 200+ tasks probing capabilities.
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — **Zheng et al. (2023)** — LLM judges and Elo from human votes.

## 📰 Articles / Blogs (free, no paywall)
- [Chatbot Arena: Benchmarking LLMs in the Wild](https://lmsys.org/blog/2023-05-03-arena/) — **LMSYS** — crowdsourced Elo, the most-watched leaderboard.
- [LLM-as-a-judge: a complete guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge) — **Evidently AI** — methodology and the biases of LLM judges.
- [HELM: Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/) — **Stanford CRFM** — multi-metric, multi-scenario evaluation.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 10 "Large Language Models"** (perplexity & evaluation)](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — intrinsic LM evaluation.
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — §7 evaluation: benchmarks, metrics, and pitfalls (free reference).

## 🔗 In this platform
- Foundations (covered elsewhere): [NLP Evaluation Metrics (perplexity, BLEU, ROUGE, BERTScore)](../../06.%20NLP/concepts/18-NLP-Evaluation-Metrics.md)
- Concept depth (the *why*): [Module 5.01 Information Theory: Entropy & KL](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
- Related concepts: [Hallucination & Alignment Basics](20-Hallucination-and-Alignment-Basics.md) · [Chain-of-Thought Reasoning](17-Chain-of-Thought-Reasoning.md) · [RLHF & DPO](15-RLHF-and-DPO.md)
