---
id: "05-deep-learning/positional-encoding/references"
topic: "Positional Encoding — References"
parent: "05-deep-learning/positional-encoding"
type: references
updated: 2026-06-22
---

# Positional Encoding — references and further reading

> Companion link library for **[Positional Encoding](17-Positional-Encoding.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [Positional embeddings in transformers EXPLAINED](https://www.youtube.com/watch?v=1biZfFLPRSY) (**AI Coffee Break with Letitia**). *Why order matters and what the encoding adds.*
2. **See the sinusoids** — watch [Position Embeddings (Visual Guide)](https://www.youtube.com/watch?v=dichIcUZfOw) (**Hedu AI**). *The multi-frequency sine/cosine picture, visualized.*
3. **Get the math** — read [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) (**Amirhossein Kazemnejad**). *The definitive derivation of the sinusoidal scheme and the rotation property.*
4. **Read the modern source** — [RoFormer: Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) (**Su et al., 2021**). *The relative-position encoding behind most current LLMs.*
5. **Make it concrete** — implement sinusoidal PE following [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). *Coding and plotting the encoding makes the frequencies click.*

**Videos**:
- [Positional embeddings in transformers EXPLAINED](https://www.youtube.com/watch?v=1biZfFLPRSY) — **AI Coffee Break with Letitia** — why position info is needed and how it's added; the clearest short intro.
- [Visual Guide to Transformers — Position Embeddings](https://www.youtube.com/watch?v=dichIcUZfOw) — **Hedu AI (Batool Haider)** — the sinusoidal frequencies, visualized dimension by dimension.
- [Rotary Positional Embeddings: Combining Absolute and Relative](https://www.youtube.com/watch?v=o29P0Kpobz0) — **Efficient NLP** — the best concise explainer of RoPE's rotation idea and why it's relative.
- [Positional Encoding in Transformer Neural Networks Explained](https://www.youtube.com/watch?v=ZMxVe-HK174) — **CodeEmporium** — the sine/cosine formula derived step by step.
- [How positional encoding works in transformers](https://www.youtube.com/watch?v=T3OT8kqoqjc) — **BrainDrain** — a visual tour from sinusoidal through RoPE and ALiBi.

**Courses (free)**:
- [Stanford CS224N — Transformers (positional encoding)](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — why attention needs positions and how encodings supply them, in the canonical NLP course.
- [Dive into Deep Learning — §11.6 Self-Attention and Positional Encoding](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html) — **Zhang, Lipton, Li & Smola** — sinusoidal PE derived *and implemented in runnable code*.

**Articles / blogs (free, no paywall)**:
- [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) — **Amirhossein Kazemnejad** — the gold-standard derivation of the sinusoidal scheme and its relative-shift (rotation) property.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — **Jay Alammar** — where positional encoding fits in the full model, visually.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — **Harvard NLP (Rush et al.)** — sinusoidal PE implemented line-by-line alongside the paper.
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) — **EleutherAI** — the definitive practitioner write-up of RoPE, by the people who put it in GPT-NeoX.
- [You could have designed state-of-the-art positional encoding](https://huggingface.co/blog/designing-positional-encoding) — **Hugging Face (Christopher Fleetwood)** — derives RoPE *from the requirements*, the same way this page does.
- [Extending the RoPE](https://blog.eleuther.ai/yarn/) — **EleutherAI** — Position Interpolation, NTK-aware scaling, and YaRN explained together.

**Key papers**:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — introduces the sinusoidal positional encoding (§3.5).
- [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) — **Shaw, Uszkoreit & Vaswani (2018)** — the first relative-position scheme, injected into attention.
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](https://arxiv.org/abs/1910.10683) — **Raffel et al. (2020)** — the bucketed scalar relative-position bias.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — **Su et al. (2021)** — RoPE, the de-facto encoding in current LLMs.
- [Train Short, Test Long: Attention with Linear Biases (ALiBi)](https://arxiv.org/abs/2108.12409) — **Press, Smith & Lewis (2021)** — bias-based positions that extrapolate to long contexts.
- [The Impact of Positional Encoding on Length Generalization in Transformers (NoPE)](https://arxiv.org/abs/2305.19466) — **Kazemnejad et al. (2023)** — decoders can learn position with no encoding; a careful comparison of schemes.
- [Extending Context Window of LLMs via Position Interpolation](https://arxiv.org/abs/2306.15595) — **Chen et al. (2023)** — Position Interpolation for RoPE context extension.
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — **Peng et al. (2023)** — the NTK-aware / YaRN method for stretching RoPE to long context.

**Books (free chapters)**:
- [Dive into Deep Learning — §11.6 "Self-Attention and Positional Encoding"](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html) — **Zhang et al.** — sinusoidal encoding derived and coded.
- [Speech and Language Processing, 3rd ed. — Ch. 9 "Transformers"](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — positional encoding within the transformer, free online draft.

**In this platform**:
- Concept page (full explanation): [Positional Encoding](17-Positional-Encoding.md)
- Prerequisites (the *why* behind Q, K, V and the block): [Attention Mechanism](15-Attention-Mechanism.md) · [Transformer Architecture](16-Transformer-Architecture.md)
- Builds on this (RoPE caching, long context): [KV Cache](../../09.%20LLMs/concepts/05-KV-Cache.md)
- Concept depth (the *why*): [AI-ML-intuition 1.03 Positional Encoding](../../../AI-ML-intuition/Module_1_Representation/1.03_Positional_Encoding.md)
- Field overview: [Deep Learning](../README.md)
- Related domain: [09. LLMs](../../09.%20LLMs/concepts/README.md) (RoPE/ALiBi for long-context models)
