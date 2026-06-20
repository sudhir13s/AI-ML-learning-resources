---
id: "09-llms/pretraining-at-scale"
topic: "Pretraining at Scale"
parent: "09-llms"
level: advanced
prereqs: ["language-modeling-objectives", "transformer-architecture", "optimization"]
interview_frequency: high
updated: 2026-06-20
---

# Pretraining at Scale
> Turning the next-token objective into a foundation model: trillions of tokens of curated web text,
> thousands of GPUs, weeks of training, and the data/optimization/parallelism tricks that keep a run
> stable. This is the expensive, capability-creating phase before any fine-tuning.

**Why it matters:** interviewers probe the *systems* of LLMs — how you shard a model across GPUs
(data/tensor/pipeline parallelism), why data quality and dedup dominate, what learning-rate schedule
and warmup you'd use, how loss spikes are handled, and the compute/cost intuition (FLOPs ≈ 6·N·D).

**⭐ Start here — suggested path:**

1. **See a real run** — watch [Karpathy: Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU). *An actual end-to-end pretraining run: data, init, LR schedule, throughput.*
2. **Get the cost math** — read [Transformer Math 101](https://blog.eleuther.ai/transformer-math/). *Memory, FLOPs, and the 6ND compute rule you'll be asked to estimate.*
3. **Understand the pipeline** — read [SLP3 Ch. 10](https://web.stanford.edu/~jurafsky/slp3/10.pdf). *Data sourcing, tokenization, and training the objective at scale.*
4. **Read the system blueprint** — skim the [LLaMA paper](https://arxiv.org/abs/2302.13971). *Concrete recipe: data mix, hyperparameters, tokens-per-parameter.*
5. **Survey the field** — [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) §4 (pre-training). *Maps data, architecture, and training-stability choices across models.*

## 🎓 Courses (free)
- [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) — **Stanford** — the definitive build-an-LLM course: data, training, parallelism, systems.
- [Stanford CS324 — Large Language Models](https://stanford-cs324.github.io/winter2022/) — **Stanford** — data, scaling, and the costs/harms of pretraining.

## 🎥 Videos
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — **Andrej Karpathy** — a full pretraining run, optimizations and all.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — where pretraining sits in the full pipeline (data → base model → SFT → RLHF).
- [Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **Andrej Karpathy** — the training loop, scaled-down but complete.
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — **Andrej Karpathy** — tokenization, the first step of any pretraining pipeline.

## 📄 Key Papers
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — **Touvron et al. (2023)** — a fully documented open pretraining recipe.
- [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) — **Hoffmann et al. (2022)** — how to allocate a fixed compute budget between model size and data.
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — §4 pre-training data, architecture, and training tricks.

## 📰 Articles / Blogs (free, no paywall)
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) — **EleutherAI** — memory/compute/parallelism arithmetic for real training runs.
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) — **Kipply** — the FLOPs/memory reasoning that also governs training cost.
- [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — **Sebastian Raschka** — a guided path through the foundational pretraining papers.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 10 "Large Language Models"**](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — pretraining corpora, tokenization, and the objective at scale.
- [Dive into Deep Learning — **Ch. 11 "Optimization"**](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) — **Zhang et al.** — the optimizers and schedules large runs rely on.

## 🔗 In this platform
- Concept depth (the *why*): [Module 7.01 Neural Scaling Laws / Chinchilla](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.01_Neural_Scaling_Laws_Chinchilla.md) · [Module 5.05 Autoregressive Generation & Sampling](../../../AI-ML-intuition/Module_5_Generation/5.05_Autoregressive_Generation_Sampling.md)
- Foundations (covered elsewhere): [Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md)
- Related concepts: [Scaling Laws](03-Scaling-Laws.md) · [Decoder-only Architecture](04-Decoder-only-Architecture.md) · [Language Modeling Objectives](01-Language-Modeling-Objectives.md)
