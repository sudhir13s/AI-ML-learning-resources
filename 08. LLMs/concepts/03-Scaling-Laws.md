---
id: "08-llms/scaling-laws"
topic: "Scaling Laws (Kaplan → Chinchilla)"
parent: "08-llms"
level: advanced
prereqs: ["pretraining-at-scale", "power-laws"]
interview_frequency: very-high
updated: 2026-06-20
---

# Scaling Laws — Kaplan → Chinchilla
> Loss falls as a smooth **power law** in model size (N), data (D), and compute (C). Kaplan (2020)
> said "make models bigger"; Chinchilla (2022) corrected it — for a fixed compute budget you should
> scale **N and D together** (≈20 tokens per parameter). This is *the* equation that decides how to
> spend a training budget.

**Why it matters:** a top interview favorite. Be ready to write `L(N,D) = E + A/Nᵃ + B/Dᵇ`, explain
why GPT-3 was "undertrained," derive the compute-optimal N*/D* split from the `C ≈ 6ND` constraint,
and discuss why inference cost pushes real deployments *past* Chinchilla-optimal (train longer, smaller).

**⭐ Start here — suggested path:**

1. **Get the intuition** — watch [Understanding Chinchilla Scaling Laws](https://www.youtube.com/watch?v=TYCw8QSNT9w). *Why "bigger isn't always better" and what compute-optimal means.*
2. **See the field context** — [Karpathy: Deep Dive into LLMs](https://www.youtube.com/watch?v=7xTGNNLPyMI). *Where scaling laws drive the pretraining decisions.*
3. **Read the correction** — the [Chinchilla paper](https://arxiv.org/abs/2203.15556). *The 3 estimation methods → the ~20 tokens/param rule.*
4. **Read the original** — [Kaplan et al. Scaling Laws for Neural LMs](https://arxiv.org/abs/2001.08361). *The first power-law fits — and the assumption Chinchilla overturned.*
5. **Internalize the trade-off** — [Module 7.01 intuition](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.01_Neural_Scaling_Laws_Chinchilla.md). *Solidifies the N-vs-D allocation and the `6ND` compute math.*

## 🎓 Courses (free)
- [Stanford CS324 — Scaling Laws lecture](https://stanford-cs324.github.io/winter2022/) — **Stanford** — derives and interprets the power-law fits.
- [Stanford CS336 — Scaling & systems](https://stanford-cs336.github.io/spring2025/) — **Stanford** — how scaling laws guide real budget allocation.

## 🎥 Videos
- [Understanding LLM Chinchilla Scaling Laws](https://www.youtube.com/watch?v=TYCw8QSNT9w) — **Cloudvala** — the compute-optimal idea, visually and concretely.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — scaling laws in the context of the full LLM pipeline.
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — the "scaling laws are remarkably smooth" insight in plain terms.
- [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — **3Blue1Brown** — the architecture whose loss these laws describe.

## 📄 Key Papers
- [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) — **Hoffmann et al. (2022)** — the compute-optimal correction; ~20 tokens/parameter.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — **Kaplan et al. (2020)** — the original power-law study (N, D, C).
- [LLaMA: Open and Efficient Foundation LMs](https://arxiv.org/abs/2302.13971) — **Touvron et al. (2023)** — trains smaller models on *more* tokens, citing inference cost.

## 📰 Articles / Blogs (free, no paywall)
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) — **EleutherAI** — the `C ≈ 6ND` compute arithmetic behind the laws.
- [Go smol or go home (context length & scaling economics)](https://www.harmdevries.com/post/context-length/) — **Harm de Vries** — why real deployments push past Chinchilla-optimal.
- [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — **Sebastian Raschka** — scaling laws within the broader research arc.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 10 "Large Language Models"**](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — model size, data, and the scaling intuition.
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — §3 surveys scaling-law findings across models (free arXiv "book-length" reference).

## 🔗 In this platform
- Concept depth (the *why*): [Module 7.01 Neural Scaling Laws / Chinchilla](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.01_Neural_Scaling_Laws_Chinchilla.md)
- Related concepts: [Pretraining at Scale](02-Pretraining-at-Scale.md) · [Mixture-of-Experts](07-Mixture-of-Experts.md) · [Quantization](10-Quantization.md)
