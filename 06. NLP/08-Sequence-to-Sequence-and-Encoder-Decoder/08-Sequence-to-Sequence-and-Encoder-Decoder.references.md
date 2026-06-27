---
id: "06-nlp/seq2seq-encoder-decoder/references"
topic: "Sequence-to-Sequence & Encoder–Decoder — References"
parent: "06-nlp/seq2seq-encoder-decoder"
type: references
updated: 2026-06-27
---

# Sequence-to-Sequence & Encoder–Decoder — references and further reading

> Companion link library for **[Sequence-to-Sequence & Encoder–Decoder](08-Sequence-to-Sequence-and-Encoder-Decoder.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build the intuition** — watch [Seq2Seq Encoder–Decoder Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=L8HKweZIOmg) (**StatQuest**). *The encoder → context → decoder picture, gently, before any attention.*
2. **See why attention is needed** — read [Visualizing A Neural Machine Translation Model (seq2seq with attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) (**Jay Alammar**). *Watch the single-vector bottleneck appear and attention dissolve it.*
3. **Get the math** — read [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) (**Lena Voita, NLP Course for You**). *Encoder, decoder, attention scores, alignment, and training — rigorous and free.*
4. **Read the sources** — [Seq2Seq (Sutskever et al.)](https://arxiv.org/abs/1409.3215) + [Cho et al. GRU encoder–decoder](https://arxiv.org/abs/1406.1078) → [Bahdanau et al. attention](https://arxiv.org/abs/1409.0473). *The bottleneck, then its fix, in the original words.*
5. **Build it** — code the [PyTorch seq2seq-with-attention translation tutorial](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html). *A translating encoder–decoder with attention, end to end, and the alignment plots.*

**Videos**:
- [Seq2Seq Encoder–Decoder Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=L8HKweZIOmg) — **StatQuest (Josh Starmer)** — gentle, from-scratch intuition for the whole architecture.
- [Visualizing Attention, a Transformer's Heart](https://www.youtube.com/watch?v=eMlx5fFNoYc) — **3Blue1Brown** — the most visual explanation of query/key/value, the math attention reduces to.
- [Attention for Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=PSs6nxngL6k) — **StatQuest (Josh Starmer)** — attention added to seq2seq, step by step, with the alignment intuition.
- [PyTorch Seq2Seq with Attention for Machine Translation](https://www.youtube.com/watch?v=sQUqQddQtB4) — **Aladdin Persson** — implement encoder, decoder, and Bahdanau attention line by line.
- [Encoder–Decoder architecture: Overview](https://www.youtube.com/watch?v=zbdong_h-x4) — **Google Cloud Tech** — concise official overview tying RNN seq2seq to modern Transformer variants.

**Courses (free)**:
- [NLP Course for You — Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) — **Lena Voita (Yandex/Edinburgh)** — the clearest free chapter on encoder–decoder + attention, with interactive figures.
- [Stanford CS224N — NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the MT/seq2seq lecture introduces attention from first principles; full slides and videos.
- [Dive into Deep Learning — Ch. 10, "Modern Recurrent Neural Networks" (encoder–decoder, seq2seq, attention)](https://d2l.ai/chapter_recurrent-modern/index.html) — **Zhang, Lipton, Li & Smola** — runnable seq2seq + attention code alongside the theory.

**Articles / blogs (free, no paywall)**:
- [Visualizing A Neural Machine Translation Model (seq2seq with attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) — **Jay Alammar** — the definitive visual explainer of the bottleneck and attention.
- [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/) — **Olah & Carter (Distill)** — beautiful interactive intuition for attention as content-based addressing.
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — **Lilian Weng (OpenAI)** — a thorough survey of attention variants (additive/multiplicative, soft/hard, self/cross) starting from seq2seq.
- [Neural Machine Translation with attention (TensorFlow tutorial)](https://www.tensorflow.org/text/tutorials/nmt_with_attention) — **TensorFlow** — build a seq2seq + attention translator and plot the alignment matrices.
- [The Annotated Encoder–Decoder with Attention](https://bastings.github.io/annotated_encoder_decoder/) — **Joost Bastings** — Bahdanau's model implemented and annotated line by line in PyTorch.

**Key papers**:
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — **Sutskever, Vinyals & Le (2014)** — the original LSTM encoder–decoder; introduces the source-reversal trick.
- [Learning Phrase Representations using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078) — **Cho et al. (2014)** — the GRU encoder–decoder; the parallel seq2seq foundation, conditioning the decoder on the context each step.
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — **Bahdanau, Cho & Bengio (2015)** — introduces (additive) attention; removes the bottleneck and yields soft alignment.
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — **Luong, Pham & Manning (2015)** — multiplicative/dot-product attention; global vs local attention.
- [A Learning Algorithm for Continually Running Fully Recurrent Neural Networks](https://ieeexplore.ieee.org/document/6795228) — **Williams & Zipser (1989)** — the origin of **teacher forcing** (feed the gold previous token while training a recurrent net).
- [Scheduled Sampling for Sequence Prediction with RNNs](https://arxiv.org/abs/1506.03099) — **Bengio et al. (2015)** — the classic remedy for exposure bias (anneal from teacher forcing to own predictions).
- [Sequence Level Training with Recurrent Neural Networks](https://arxiv.org/abs/1511.06732) — **Ranzato et al. (2016)** — names exposure bias and trains at the sequence level (optimizing the metric directly).
- [Google's Neural Machine Translation System (GNMT)](https://arxiv.org/abs/1609.08144) — **Wu et al. (2016)** — production-scale seq2seq; the source of the **length-normalization** and **coverage** penalties for beam search.
- [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — **Shannon (1948)** — the source-coding bound behind the bottleneck's information-capacity argument ($S$ digits need $S\log_2 10$ bits).
- [Pointer Networks](https://arxiv.org/abs/1506.03134) — **Vinyals, Fortunato & Jaitly (2015)** — attention as a pointer that copies from the input.
- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368) — **See, Liu & Manning (2017)** — the copy/generate gate and coverage for abstractive summarization.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — the Transformer encoder–decoder; self-attention replaces recurrence, cross-attention replaces Bahdanau attention.
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](https://arxiv.org/abs/1910.10683) — **Raffel et al. (2020)** — every task as seq2seq through a Transformer encoder–decoder.
- [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — **Lewis et al. (2020)** — a denoising-autoencoder encoder–decoder, strong for summarization and generation.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 13 "Machine Translation"](https://web.stanford.edu/~jurafsky/slp3/13.pdf) — **Jurafsky & Martin** — encoder–decoder, attention, beam search, and BLEU in the standard text.
- [Dive into Deep Learning — Ch. 10–11 (encoder–decoder, seq2seq, Bahdanau attention)](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) — **Zhang et al.** — the math with runnable code and figures.

**Runnable code (this chapter)**:
- [Teaching notebook (executed, step-by-step)](code/08-Sequence-to-Sequence-and-Encoder-Decoder.ipynb) — the bottleneck and attention measured from scratch, one idea per cell, assert-before-print.
- [`seq2seq.py` — the seeded source of truth](code/seq2seq.py) — encoder, both decoders, training, free-running accuracy, alignment-matrix extractor, greedy/beam demo (device-agnostic).
- [`make_figures_08.py` — figure generator](code/make_figures_08.py) — regenerates every figure on the page from the *same* functions, so the prose and figures cannot drift.

**In this platform**:
- Concept page (full explanation): [Sequence-to-Sequence & Encoder–Decoder](08-Sequence-to-Sequence-and-Encoder-Decoder.md)
- Foundations (the mechanisms this builds on): [RNN / LSTM / GRU](../../05.%20Deep_Learning/concepts/14-RNN-LSTM-GRU.md) · [Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md) · [Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md)
- Prior NLP step: [N-gram Language Models and Smoothing](../04-N-gram-Language-Models-and-Smoothing/04-N-gram-Language-Models-and-Smoothing.md) — the statistical predecessor neural seq2seq replaced.
- Puts it to work: [Machine Translation](../12-Machine-Translation/12-Machine-Translation.md) · [Text Summarization](../13-Text-Summarization/13-Text-Summarization.md) · [Decoding Strategies](../17-Decoding-Strategies/17-Decoding-Strategies.md) · [NLP Evaluation Metrics](../18-NLP-Evaluation-Metrics/18-NLP-Evaluation-Metrics.md)
- Connects forward: [KV Cache](../../09.%20LLMs/05-KV-Cache/05-KV-Cache.md) — how encoder–decoder (and decoder-only) inference caches K/V, including the cross-attention cache.
- Concept depth (the *why* behind attention scores): [AI-ML-intuition 1.06 Scaled Dot-Product (attention scores)](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md)
