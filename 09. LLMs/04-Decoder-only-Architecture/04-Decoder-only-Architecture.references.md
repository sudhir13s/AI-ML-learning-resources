---
id: "09-llms/decoder-only-architecture/references"
topic: "Decoder-only Architecture — References"
parent: "09-llms/decoder-only-architecture"
type: references
updated: 2026-06-22
---

# Decoder-only Architecture — references and further reading

> Companion link library for **[Decoder-only Architecture](04-Decoder-only-Architecture.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **See the architecture** — watch [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) (**3Blue1Brown**). *The clearest picture of the decoder stack and how information flows through it.*
2. **Build one by hand** — watch [Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) (**Andrej Karpathy**). *Every block, the causal mask, and the training loop coded line by line — the canonical way to truly get it.*
3. **Read it as code** — [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/) (**Jay Mody**). *A complete, dependency-free decoder forward pass — nothing hidden.*
4. **Read the source** — the [GPT-3 paper](https://arxiv.org/abs/2005.14165) (the decoder-only LM at scale) then [LLaMA](https://arxiv.org/abs/2302.13971) (the modern RMSNorm + RoPE + SwiGLU recipe).
5. **Trace inference** — [How a Transformer works at inference vs training](https://www.youtube.com/watch?v=IGu7ivuy1Ag) (**Niels Rogge**). *Why generation is one-token-at-a-time, which motivates prefill/decode and the KV cache.*

**Videos**:
- [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — **3Blue1Brown** — the decoder stack visualized end to end; the best single visual intro.
- [Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **Andrej Karpathy** — implements a decoder-only transformer (mask, blocks, head) line by line in PyTorch.
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — **Andrej Karpathy** — the exact GPT-2 architecture this page measures, at full training scale.
- [Visualizing Attention, a Transformer's Heart](https://www.youtube.com/watch?v=eMlx5fFNoYc) — **3Blue1Brown** — the most visual explanation of Q, K, V; watch if causal masking still feels abstract.
- [How a Transformer works at inference vs training time](https://www.youtube.com/watch?v=IGu7ivuy1Ag) — **Niels Rogge (Hugging Face)** — the autoregressive decode loop and why prefill/decode differ.

**Interactive & visual**:
- [LLM Visualizer (3D)](https://bbycroft.net/llm) — **Brendan Bycroft** — walk a single token through a small GPT's full decoder forward pass, per layer; the clearest way to *see* the stack.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — **Jay Alammar** — the visual canon for the decoder-only architecture and its causal masking.

**Courses (free)**:
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — **Andrej Karpathy** — builds a full decoder-only GPT step by step from first principles.
- [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) — **Stanford** — the decoder-only LM within the whole training-and-inference stack.
- [Hugging Face LLM Course — Ch. 1 (Transformer architectures)](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — encoder vs decoder vs encoder-decoder, with examples.

**Articles / blogs (free, no paywall)**:
- [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/) — **Jay Mody** — a complete decoder-only forward pass in plain NumPy.
- [The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — **Lilian Weng (OpenAI)** — surveys decoder variants and the modern component swaps (pre-norm, RoPE, etc.).
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — **Jay Alammar** — the visual foundation under the decoder block (attention, residuals, norms).
- [Introducing Meta Llama 3](https://huggingface.co/blog/llama3) — **Meta / Hugging Face** — the modern decoder recipe (RMSNorm, RoPE, SwiGLU, GQA) and configs in a real model release.

**Key papers**:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — the original encoder-decoder Transformer; the decoder block and causal masking come from here.
- [Improving Language Understanding by Generative Pre-Training (GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — **Radford et al. (2018)** — the first decoder-only generative-pretraining recipe.
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — **Radford et al. (2019)** — zero-shot from a scaled decoder-only LM.
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — **Brown et al. (2020)** — the canonical decoder-only LM at scale; in-context learning emerges.
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — **Touvron et al. (2023)** — RMSNorm + RoPE + SwiGLU: the modern decoder recipe, spelled out.
- [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683) — **Raffel et al. (2020)** — the encoder-decoder "text-to-text" framing that decoder-only was weighed against.
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — **Devlin et al. (2018)** — the encoder-only / bidirectional alternative and what it's best at.
- [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/abs/2204.05832) — **Wang et al. (2022)** — the controlled study showing causal decoder-only wins zero-shot.
- [Using the Output Embedding to Improve Language Models (weight tying)](https://arxiv.org/abs/1608.05859) — **Press & Wolf (2017)** — why the LM head is tied to the input embedding.
- [RoFormer: Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) — **Su et al. (2021)** — the relative-position scheme used by most modern decoder-only LLMs.
- [GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/abs/2002.05202) — **Shazeer (2020)** — the gated FFN that replaced the plain GELU MLP.
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — **Ainslie et al. (2023)** — grouped-query attention, the KV-cache-shrinking choice in the modern recipe.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 10 "Large Language Models"](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — the decoder-only LM architecture and autoregressive decoding, in depth.
- [Dive into Deep Learning — Ch. 11 "Attention & Transformers"](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) — **Zhang et al.** — transformer blocks with runnable code.

**In this platform**:
- Concept page (full explanation): [Decoder-only Architecture](04-Decoder-only-Architecture.md)
- Foundations (covered elsewhere): [Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md) · [Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md) · [Positional Encoding](../../05.%20Deep_Learning/concepts/17-Positional-Encoding.md) · [Normalization](../../05.%20Deep_Learning/concepts/11-Normalization.md) · [Activation Functions (SwiGLU)](../../05.%20Deep_Learning/concepts/03-Activation-Functions.md)
- The objective it's trained on: [Language Modeling Objectives](../01-Language-Modeling-Objectives/01-Language-Modeling-Objectives.md) · [Pretraining at Scale](../02-Pretraining-at-Scale/02-Pretraining-at-Scale.md) · [Scaling Laws](../03-Scaling-Laws/03-Scaling-Laws.md)
- The inference it enables: [KV Cache](../05-KV-Cache/05-KV-Cache.md) · [Long-Context Methods](../08-Long-Context-Methods/08-Long-Context-Methods.md) · [Mixture of Experts](../07-Mixture-of-Experts/07-Mixture-of-Experts.md) · [Inference Optimization & Serving](../09-Inference-Optimization-and-Serving/09-Inference-Optimization-and-Serving.md)
- What it unlocks: [Prompting & In-Context Learning](../16-Prompting-and-In-Context-Learning/16-Prompting-and-In-Context-Learning.md) · [Decoding & Sampling](../18-Decoding-and-Sampling/18-Decoding-and-Sampling.md) · [Supervised Fine-Tuning](../13-Supervised-Fine-Tuning/13-Supervised-Fine-Tuning.md) · [RLHF & DPO](../15-RLHF-and-DPO/15-RLHF-and-DPO.md)
