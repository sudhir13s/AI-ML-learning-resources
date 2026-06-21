---
id: "05-deep-learning/attention-mechanism/references"
topic: "Attention Mechanism — References"
parent: "05-deep-learning/attention-mechanism"
type: references
updated: 2026-06-21
---

# Attention Mechanism — references and further reading

> Companion link library for **[Attention Mechanism](15-Attention-Mechanism.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build the picture** — watch [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) (**3Blue1Brown**). *The clearest visualization of how Q/K/V route information.*
2. **Get the gentle intuition** — watch [Attention for Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=PSs6nxngL6k) (**StatQuest**). *Why a model should weight some inputs more than others.*
3. **See the math, every variant** — read [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) (**Lilian Weng**). *Each attention form with its equations.*
4. **Read the source** — [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (**Vaswani et al., 2017**). *Scaled dot-product + multi-head; the transformer.*
5. **Make it permanent** — implement it following [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) (**Harvard NLP**). *Coding the softmax-weighted sum makes attention stop being magic.*

**Videos**:
- [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) — **3Blue1Brown** — the clearest visualization of how Q/K/V attention routes information.
- [Attention for Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=PSs6nxngL6k) — **StatQuest (Josh Starmer)** — gentle from-scratch intuition for attention weights.
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **Andrej Karpathy** — implements self-attention line by line inside a working GPT.
- [Attention is all you need (Transformer) — model, math, inference & training](https://www.youtube.com/watch?v=bCz4OMemCcA) — **Umar Jamil** — a deep, math-complete walk through the mechanism.
- [Attention Is All You Need](https://www.youtube.com/watch?v=iDulhoQ2pro) — **Yannic Kilcher** — the original paper explained section by section.

**Interactive & visual**:
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — **Georgia Tech (Polo Club)** — a live GPT-2 in the browser; watch attention weights light up as you type.
- [LLM Visualizer (3D)](https://bbycroft.net/llm) — **Brendan Bycroft** — the Q·Kᵀ→softmax→·V flow animated in a 3D model of a working transformer.

**Courses (free)**:
- [Stanford CS224N — Attention & Transformers](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — derives attention from the seq2seq bottleneck up to self-attention.
- [Dive into Deep Learning — Attention Mechanisms](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) — **Zhang et al.** — attention scoring functions and Q/K/V with runnable code.

**Articles / blogs (free, no paywall)**:
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — **Lilian Weng (OpenAI)** — the canonical survey of attention variants with equations.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — **Jay Alammar** — the visual walk-through of self-attention inside the transformer.
- [Transformers from scratch](https://peterbloem.nl/blog/transformers) — **Peter Bloem** — self-attention derived and built up carefully, with code.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — **Harvard NLP** — the Vaswani paper as runnable, annotated PyTorch.
- [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/) — **Distill (Olah & Carter)** — interactive piece bridging RNNs to attention.

**Key papers**:
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — **Bahdanau et al. (2014)** — introduces (additive) attention, fixing the seq2seq bottleneck.
- [Effective Approaches to Attention-based NMT](https://arxiv.org/abs/1508.04025) — **Luong, Pham & Manning (2015)** — dot-product (multiplicative) attention variants.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — scaled dot-product + multi-head attention; the transformer.
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — **Dao et al. (2022)** — IO-aware exact attention that never materializes the n×n matrix.
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — **Beltagy et al. (2020)** — sliding-window + global sparse attention for long sequences.
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) — **Choromanski et al. (2020)** — linear-time attention via kernel feature maps.

**Books (free chapters)**:
- [Dive into Deep Learning — Ch. 11 "Attention Mechanisms and Transformers"](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) — **Zhang et al.** — scoring functions, Q/K/V, and multi-head attention with code.
- [Speech and Language Processing, 3rd ed. — Ch. 9–10 (Attention & Transformers)](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — attention framed for language, free draft chapters.

**In this platform**:
- Concept page (full explanation): [Attention Mechanism](15-Attention-Mechanism.md)
- Concept depth (the *why*): [AI-ML-intuition 4.08 Multi-Head Attention Routing](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.08_Multi-Head_Attention_Routing.md) · [1.06 Scaled Dot-Product Similarity](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md)
- Prerequisite: [14 RNN / LSTM / GRU](14-RNN-LSTM-GRU.md) (the bottleneck attention fixed)
- Builds into: [16 Transformer Architecture](16-Transformer-Architecture.md) — attention stacked into a full model
- At inference / efficiency: [KV Cache](../../09.%20LLMs/concepts/05-KV-Cache.md) · [Efficient Attention (FlashAttention)](../../09.%20LLMs/concepts/06-Efficient-Attention-FlashAttention.md)
- Field overview: [Deep Learning](../README.md)
