---
id: "05-deep-learning/transformer"
topic: "Transformer Architecture"
parent: "05-deep-learning"
level: intermediate
prereqs: ["attention", "backpropagation", "rnn-lstm-gru"]
interview_frequency: very-high
updated: 2026-06-19
---

# Transformer Architecture
> The architecture behind all modern LLMs. It drops recurrence entirely and relies on
> **self-attention** — every position attends to every other in parallel — stacked into encoder
> and/or decoder blocks with multi-head attention, position-wise feed-forward layers, residual
> connections, layer norm, and positional encodings. The result trains in parallel and scales.

**Why it matters:** the highest-frequency architecture question in ML interviews. Be ready to walk
through **scaled dot-product attention** (Q·Kᵀ/√dₖ → softmax → ·V), why we scale by √dₖ, what
**multi-head** buys you, the role of **positional encoding** (no recurrence ⇒ no inherent order),
the encoder vs decoder vs encoder–decoder split (BERT vs GPT vs T5), and why parallelism over
sequence length is the key advantage over RNNs.

**⭐ Start here — suggested path:**

1. **Build intuition** — read ⭐ [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (**Jay Alammar**). *The single best visual walk-through of the whole block.*
2. **See attention move** — watch [3Blue1Brown: Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) and [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M). *Watch how Q/K/V route information.*
3. **Get the math** — [Umar Jamil: Attention is all you need (with math)](https://www.youtube.com/watch?v=bCz4OMemCcA) + [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). *Every equation, then the line-by-line PyTorch implementation.*
4. **Read the source** — [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (**Vaswani et al., 2017**). *The paper; read it after the intuition so every component lands.*
5. **Build it from scratch** — [Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY). *Coding a decoder-only transformer makes attention, masking, and residual stream permanent.*

## 🎓 Courses (free)
- [Stanford CS224N — Self-Attention & Transformers](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the definitive university lecture deriving attention and the transformer block.
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — **Andrej Karpathy** — culminates in building a GPT-style transformer from scratch; the best hands-on path.

## 🎥 Videos
- [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — **3Blue1Brown** — the big-picture visual intro to the transformer and the residual stream.
- [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) — **3Blue1Brown** — the cleanest visualization of how Q/K/V attention actually routes information.
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **Andrej Karpathy** — implements a decoder-only transformer line by line.
- [Attention is all you need — model explanation (incl. math)](https://www.youtube.com/watch?v=bCz4OMemCcA) — **Umar Jamil** — full architecture with the math, training, and inference walk-through.
- [Transformer Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=zxQyTK8quyY) — **StatQuest (Josh Starmer)** — gentle, gated intro to attention and the encoder–decoder.

## 📄 Key Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — the transformer; non-negotiable reading.
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — **Bahdanau et al. (2014)** — the attention mechanism the transformer generalized.
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385) — **He et al. (2015)** — the residual connections that make deep transformer stacks trainable.

## 📰 Articles / Blogs (free, no paywall)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — **Jay Alammar** — the canonical visual explainer of the full architecture.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — **Harvard NLP** — the paper re-implemented in PyTorch, line by line alongside the text.
- [Transformers from scratch](https://peterbloem.nl/blog/transformers) — **Peter Bloem** — a rigorous, from-first-principles derivation of self-attention.
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — **Lilian Weng** — a thorough survey of attention variants leading up to the transformer.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 11 "Attention Mechanisms and Transformers"**](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) — **Zhang et al.** — attention, multi-head, positional encoding, and the full transformer with runnable code.
- [Speech and Language Processing, 3rd ed. — **Ch. 9 "RNNs and LSTMs" → Ch. 10 (Transformers)**](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — self-attention and transformers framed for language, free draft.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.15 The Transformer Block](../../../AI-ML-intuition/Module_4_Stabilization/4D_Nonlinearities/4.15_The_Transformer_Block.md) · [4.08 Multi-Head Attention](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.08_Multi-Head_Attention_Routing.md) · [1.03 Positional Encoding](../../../AI-ML-intuition/Module_1_Representation/1.03_Positional_Encoding.md)
- Prerequisite: [14 RNN / LSTM / GRU](14-RNN-LSTM-GRU.md) (what the transformer replaced)
- Field overview: [Deep Learning](../README.md)
- Related domains: [06. NLP](../../06.%20NLP/concepts/README.md) · [08. LLMs](../../08.%20LLMs/concepts/README.md) (where transformers are scaled and pretrained)
