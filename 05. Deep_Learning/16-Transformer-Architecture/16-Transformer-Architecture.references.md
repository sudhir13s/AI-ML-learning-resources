---
id: "05-deep-learning/transformer/references"
topic: "Transformer Architecture — References"
parent: "05-deep-learning/transformer"
type: references
updated: 2026-06-21
---

# Transformer Architecture — references and further reading

> Companion link library for **[Transformer Architecture](16-Transformer-Architecture.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build the picture** — read [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (**Jay Alammar**). *The single best visual walk-through of the whole block.*
2. **See the big idea** — watch [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) (**3Blue1Brown**). *The architecture and the residual stream, visually.*
3. **Get the math** — watch [Attention is all you need (with math)](https://www.youtube.com/watch?v=bCz4OMemCcA) (**Umar Jamil**) + read [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). *Every equation, then line-by-line PyTorch.*
4. **Read the source** — [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (**Vaswani et al., 2017**). *The paper; read it after the intuition so every component lands.*
5. **Build it from scratch** — watch [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) (**Andrej Karpathy**). *Coding a decoder-only transformer makes attention, masking, and the residual stream permanent.*

**Videos**:
- [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — **3Blue1Brown** — the big-picture visual intro to the transformer and the residual stream.
- [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) — **3Blue1Brown** — the cleanest visualization of how Q/K/V attention routes information.
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **Andrej Karpathy** — implements a decoder-only transformer line by line.
- [Attention is all you need — model explanation (incl. math)](https://www.youtube.com/watch?v=bCz4OMemCcA) — **Umar Jamil** — full architecture with the math, training, and inference walk-through.
- [Transformer Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=zxQyTK8quyY) — **StatQuest (Josh Starmer)** — gentle, gated intro to attention and the encoder–decoder.

**Interactive & visual**:
- [bbycroft.net/llm — LLM Visualizer (3D)](https://bbycroft.net/llm) — **Brendan Bycroft** — an animated 3D walk through an entire small GPT, block by block.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — **Georgia Tech (Polo Club)** — a live GPT-2 in the browser; see embeddings → attention → FFN → logits as you type.

**Courses (free)**:
- [Stanford CS224N — Self-Attention & Transformers](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the definitive university lecture deriving attention and the transformer block.
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — **Andrej Karpathy** — culminates in building a GPT-style transformer from scratch; the best hands-on path.

**Articles / blogs (free, no paywall)**:
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — **Jay Alammar** — the canonical visual explainer of the full architecture.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — **Harvard NLP** — the paper re-implemented in PyTorch, line by line alongside the text.
- [Transformers from scratch](https://peterbloem.nl/blog/transformers) — **Peter Bloem** — a rigorous, from-first-principles derivation of self-attention and the block.
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — **Lilian Weng (OpenAI)** — a thorough survey of attention variants leading up to the transformer.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — **Andrej Karpathy** — a clean, complete decoder-only transformer in ~300 lines; the canonical "read the whole thing" reference implementation.

**Key papers**:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — the transformer; non-negotiable reading.
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — **Bahdanau et al. (2014)** — the attention mechanism the transformer generalized.
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385) — **He et al. (2015)** — the residual connections that make deep transformer stacks trainable.
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — **Devlin et al. (2018)** — the encoder-only / masked-LM family.
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — **Radford et al. (2019)** — the decoder-only / causal-LM design (the 124M config this page derives) and weight tying.
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — **Brown et al. (2020)** — the decoder-only / causal-LM family at scale.
- [Exploring the Limits of Transfer Learning (T5)](https://arxiv.org/abs/1910.10683) — **Raffel et al. (2019)** — the encoder–decoder / text-to-text family.
- [RoFormer: Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) — **Su et al. (2021)** — relative positions via rotation; the modern default.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — **Kaplan et al. (2020)** — the original power-law scaling of loss in parameters, data, and compute.
- [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) — **Hoffmann et al. (2022)** — the scaling law correcting parameters-vs-data balance.
- [Llama: Open and Efficient Foundation Models](https://arxiv.org/abs/2302.13971) — **Touvron et al. (2023)** — the modern component stack (RoPE, RMSNorm, SwiGLU) in one model.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — **Xiong et al. (2020)** — why **pre-LN** trains deep transformers stably (no warmup), the modern default.
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) — **EleutherAI** — the parameter/memory accounting, incl. the ~12·d² per layer.

**Books (free chapters)**:
- [Dive into Deep Learning — Ch. 11 "Attention Mechanisms and Transformers"](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) — **Zhang et al.** — attention, multi-head, positional encoding, and the full transformer with runnable code.
- [Speech and Language Processing, 3rd ed. — Ch. 9–10 (RNNs → Transformers)](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — self-attention and transformers framed for language, free draft.

**In this platform**:
- Concept page (full explanation): [Transformer Architecture](16-Transformer-Architecture.md)
- Concept depth (the *why*): [AI-ML-intuition 4.15 The Transformer Block](../../../AI-ML-intuition/Module_4_Stabilization/4D_Nonlinearities/4.15_The_Transformer_Block.md) · [4.08 Multi-Head Attention](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.08_Multi-Head_Attention_Routing.md) · [1.03 Positional Encoding](../../../AI-ML-intuition/Module_1_Representation/1.03_Positional_Encoding.md)
- Core component: [15 Attention Mechanism](../15-Attention-Mechanism/15-Attention-Mechanism.md) — the operation at the heart of every block (the math is derived there)
- Block internals: [11 Normalization](../11-Normalization/11-Normalization.md) (LayerNorm / RMSNorm, pre- vs post-norm) · [18 Residual / Skip Connections](../18-Residual-Skip-Connections/18-Residual-Skip-Connections.md) (the residual stream) · [03 Activation Functions](../03-Activation-Functions/03-Activation-Functions.md) (ReLU → GELU → SwiGLU in the FFN) · [17 Positional Encoding](../17-Positional-Encoding/17-Positional-Encoding.md)
- Prerequisite: [14 RNN / LSTM / GRU](../14-RNN-LSTM-GRU/14-RNN-LSTM-GRU.md) (what the transformer replaced)
- At scale / inference: [KV Cache](../../09.%20LLMs/05-KV-Cache/05-KV-Cache.md) · [Efficient Attention (FlashAttention)](../../09.%20LLMs/06-Efficient-Attention-FlashAttention/06-Efficient-Attention-FlashAttention.md)
- Field overview: [Deep Learning](../README.md) · related: [06. NLP](../../06.%20NLP/README.md) · [09. LLMs](../../09.%20LLMs/README.md)
