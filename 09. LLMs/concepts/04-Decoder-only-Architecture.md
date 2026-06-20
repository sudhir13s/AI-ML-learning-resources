---
id: "09-llms/decoder-only-architecture"
topic: "Decoder-only Architecture (the GPT family)"
parent: "09-llms"
level: advanced
prereqs: ["transformer-architecture", "language-modeling-objectives", "attention"]
interview_frequency: very-high
updated: 2026-06-20
---

# Decoder-only Architecture — the GPT Family
> The dominant LLM blueprint: a stack of transformer blocks with **causal (masked) self-attention**,
> trained left-to-right on next-token prediction. No encoder, no cross-attention — just pre-norm
> blocks, RoPE, SwiGLU, and a tied output head. This is the shape of GPT, LLaMA, Mistral, and almost
> every modern base model.

**Why it matters:** the bread-and-butter system-design question — "draw a GPT block." Expect to
explain the causal mask, pre-LN vs post-LN, why decoder-only beat encoder-decoder for generation,
and the modern swaps (RoPE over learned positions, RMSNorm, SwiGLU, GQA) that define LLaMA-class models.

**⭐ Start here — suggested path:**

1. **See the architecture** — watch [3Blue1Brown: Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M). *The clearest picture of the decoder stack and information flow.*
2. **Build one** — [Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY). *Every block coded by hand — the canonical way to truly get it.*
3. **Read it as code** — [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/). *A complete, dependency-free decoder forward pass.*
4. **Read the source** — the [GPT-3 paper](https://arxiv.org/abs/2005.14165) (architecture) then [LLaMA](https://arxiv.org/abs/2302.13971) (modern swaps). *Pre-norm, RoPE, SwiGLU, RMSNorm — the current standard.*
5. **Trace inference** — [How a Transformer works at inference vs training](https://www.youtube.com/watch?v=IGu7ivuy1Ag). *Why generation is one-token-at-a-time and why that motivates the KV cache.*

## 🎓 Courses (free)
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — **Andrej Karpathy** — builds a full decoder-only GPT step by step.
- [Hugging Face LLM Course — Ch. 1](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — encoder vs decoder vs encoder-decoder, with examples.

## 🎥 Videos
- [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — **3Blue1Brown** — the decoder stack visualized end to end.
- [Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **Andrej Karpathy** — implements a decoder-only transformer line by line.
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — **Andrej Karpathy** — the GPT-2 architecture at full scale.
- [How a Transformer works at inference vs training time](https://www.youtube.com/watch?v=IGu7ivuy1Ag) — **Niels Rogge** — the autoregressive decode loop, clearly.

## 📄 Key Papers
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — **Brown et al. (2020)** — the canonical decoder-only LM at scale.
- [LLaMA: Open and Efficient Foundation LMs](https://arxiv.org/abs/2302.13971) — **Touvron et al. (2023)** — RoPE + SwiGLU + RMSNorm: the modern decoder recipe.
- [RoFormer (Rotary Position Embedding / RoPE)](https://arxiv.org/abs/2104.09864) — **Su et al. (2021)** — the positional scheme used by most decoder-only LLMs.

## 📰 Articles / Blogs (free, no paywall)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — **Jay Alammar** — the visual canon for decoder-only models.
- [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/) — **Jay Mody** — a complete decoder forward pass, nothing hidden.
- [The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — **Lilian Weng** — surveys decoder variants and the modern component swaps.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 10 "Large Language Models"**](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — the decoder-only LM architecture in depth.
- [Dive into Deep Learning — **Ch. 11 "Attention & Transformers"**](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) — **Zhang et al.** — transformer blocks with runnable code.

## 🔗 In this platform
- Concept depth (the *why*): [Module 5.05 Autoregressive Generation & Sampling](../../../AI-ML-intuition/Module_5_Generation/5.05_Autoregressive_Generation_Sampling.md)
- Foundations (covered elsewhere): [Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md) · [Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md) · [Positional Encoding](../../05.%20Deep_Learning/concepts/17-Positional-Encoding.md)
- Related concepts: [Language Modeling Objectives](01-Language-Modeling-Objectives.md) · [KV Cache](05-KV-Cache.md) · [Long-Context Methods](08-Long-Context-Methods.md)
