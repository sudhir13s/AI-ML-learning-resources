---
id: "05-deep-learning/positional-encoding"
topic: "Positional Encoding"
parent: "05-deep-learning"
level: intermediate
prereqs: ["attention", "transformer", "linear-algebra"]
interview_frequency: high
updated: 2026-06-19
---

# Positional Encoding
> Self-attention is permutation-invariant — it sees a *set* of tokens, not a sequence — so a
> transformer needs an explicit signal of *order*. **Positional encoding** injects that: the original
> sinusoidal scheme adds fixed sine/cosine vectors of varying frequency; learned embeddings train a
> position vector; modern LLMs use **RoPE** (rotary) or **ALiBi** to encode relative position and
> generalize to longer contexts.

**Why it matters:** a high-frequency transformer follow-up — explain *why* attention needs position
info at all, derive the sinusoidal formula and why multiple frequencies let the model represent both
fine and coarse position, compare absolute vs relative encodings, and explain why **RoPE/ALiBi**
extrapolate to longer sequences better than learned absolute embeddings.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Positional embeddings in transformers EXPLAINED](https://www.youtube.com/watch?v=1biZfFLPRSY) (**AI Coffee Break**). *Why order matters and what the encoding adds.*
2. **See the sinusoids** — watch [Position Embeddings (Visual Guide)](https://www.youtube.com/watch?v=dichIcUZfOw) (**Hedu AI**). *The multi-frequency sine/cosine picture, visualized.*
3. **Get the math** — read ⭐ [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) (**Amirhossein Kazemnejad**). *The definitive derivation of the sinusoidal scheme.*
4. **Read the modern source** — [RoFormer: Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) (**Su et al., 2021**). *The relative-position encoding behind most current LLMs.*
5. **Make it concrete** — implement sinusoidal PE following [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). *Coding and plotting the encoding makes the frequencies click.*

## 🎓 Courses (free)
- [Stanford CS224N — Transformers (positional encoding)](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — why attention needs positions and how encodings supply them.
- [Dive into Deep Learning — Self-Attention and Positional Encoding](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html) — **Zhang et al.** — sinusoidal PE derived and implemented in code.

## 🎥 Videos
- [Positional embeddings in transformers EXPLAINED](https://www.youtube.com/watch?v=1biZfFLPRSY) — **AI Coffee Break with Letitia** — why position info is needed and how it's added.
- [Visual Guide to Transformers — Position Embeddings](https://www.youtube.com/watch?v=dichIcUZfOw) — **Hedu AI (Batool Haider)** — the sinusoidal frequencies, visualized.
- [Positional Encoding in Transformer Neural Networks Explained](https://www.youtube.com/watch?v=ZMxVe-HK174) — **CodeEmporium** — the sine/cosine formula step by step.
- [RoPE (Rotary Positional Embeddings) explained](https://www.youtube.com/watch?v=GQPOtyITy54) — **DeepLearning Hero** — the relative encoding behind modern LLMs.

## 📄 Key Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — introduces the sinusoidal positional encoding (§3.5).
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — **Su et al. (2021)** — RoPE, the de-facto encoding in current LLMs.
- [Train Short, Test Long: Attention with Linear Biases (ALiBi)](https://arxiv.org/abs/2108.12409) — **Press, Smith & Lewis (2021)** — bias-based positions that extrapolate to long contexts.

## 📰 Articles / Blogs (free, no paywall)
- [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) — **Amirhossein Kazemnejad** — the gold-standard derivation of the sinusoidal scheme.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — **Jay Alammar** — where positional encoding fits in the full model.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — **Harvard NLP** — sinusoidal PE implemented alongside the paper.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **§11.6 "Self-Attention and Positional Encoding"**](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html) — **Zhang et al.** — sinusoidal encoding derived and coded.
- [Speech and Language Processing, 3rd ed. — **Ch. 10 (Transformers, positional encoding)**](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — positional encoding in the transformer, free draft.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.03 Positional Encoding](../../../AI-ML-intuition/Module_1_Representation/1.03_Positional_Encoding.md)
- Prerequisites: [15 Attention Mechanism](15-Attention-Mechanism.md) · [16 Transformer Architecture](16-Transformer-Architecture.md)
- Field overview: [Deep Learning](../README.md)
- Related domain: [08. LLMs](../../08.%20LLMs/README.md) (RoPE/ALiBi for long-context models)
