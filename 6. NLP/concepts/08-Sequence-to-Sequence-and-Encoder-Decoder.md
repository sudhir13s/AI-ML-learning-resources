---
id: "06-nlp/seq2seq-encoder-decoder"
topic: "Sequence-to-Sequence & Encoder–Decoder for MT"
parent: "06-nlp"
level: intermediate
prereqs: ["rnn-lstm", "attention", "word-embeddings"]
interview_frequency: high
updated: 2026-06-19
---

# Sequence-to-Sequence & Encoder–Decoder
> An architecture that maps one variable-length sequence to another: an **encoder** reads the input
> into a representation, a **decoder** generates the output token by token. The **attention**
> mechanism — letting the decoder look back at all encoder states — was born here and led to the
> Transformer.

**Why it matters:** seq2seq is the template behind translation, summarization, and every
encoder–decoder LLM (T5, BART). Be ready to explain the **information-bottleneck** of a single context
vector, how **attention** removes it (and the alignment it learns), **teacher forcing** vs free-running
decoding and exposure bias, and how this RNN-era idea generalizes directly to the Transformer's
cross-attention.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Seq2Seq Encoder–Decoder Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=L8HKweZIOmg) (**StatQuest**). *The encoder→context→decoder picture before any attention.*
2. **See why attention is needed** — read ⭐ [Visualizing NMT: seq2seq with attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) (**Jay Alammar**). *Watch the bottleneck appear and attention fix it.*
3. **Get the math** — read [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) (**Lena Voita, NLP Course for You**). *Encoder, decoder, attention scores, and training, rigorously and free.*
4. **Read the sources** — [Seq2Seq (Sutskever)](https://arxiv.org/abs/1409.3215) + [Cho GRU encoder–decoder](https://arxiv.org/abs/1406.1078) → [Bahdanau attention](https://arxiv.org/abs/1409.0473). *The bottleneck, then its fix.*
5. **Make it concrete** — code it with the [TensorFlow NMT-with-attention tutorial](https://www.tensorflow.org/tutorials/text/nmt_with_attention). *Build a translating encoder–decoder end to end.*

## 🎓 Courses (free)
- [NLP Course for You — Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) — **Lena Voita** — the clearest free chapter on encoder–decoder + attention.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the MT/seq2seq lecture introduces attention from first principles.

## 🎥 Videos
- [Seq2Seq Encoder–Decoder Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=L8HKweZIOmg) — **StatQuest (Josh Starmer)** — gentle, from-scratch intuition.
- [Encoder–decoder architecture: Overview](https://www.youtube.com/watch?v=zbdong_h-x4) — **Google Cloud Tech** — concise official overview tying RNN seq2seq to modern variants.
- [Seq2Seq Models](https://www.youtube.com/watch?v=MqugtGD605k) — **Weights & Biases** — a clean lecture-style walkthrough of the architecture.
- [PyTorch Seq2Seq with Attention for Machine Translation](https://www.youtube.com/watch?v=sQUqQddQtB4) — **Aladdin Persson** — implement the model line by line in PyTorch.

## 📄 Key Papers
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — **Sutskever, Vinyals & Le (2014)** — the original LSTM encoder–decoder.
- [Learning Phrase Representations using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078) — **Cho et al. (2014)** — GRU encoder–decoder; the parallel seq2seq foundation.
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — **Bahdanau, Cho & Bengio (2015)** — introduces attention; removes the bottleneck.

## 📰 Articles / Blogs (free, no paywall)
- [Visualizing A Neural Machine Translation Model (seq2seq with attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) — **Jay Alammar** — the definitive visual explainer.
- [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/) — **Olah & Carter (Distill)** — beautiful, interactive intuition for attention.
- [Neural Machine Translation with attention (TensorFlow tutorial)](https://www.tensorflow.org/tutorials/text/nmt_with_attention) — **TensorFlow** — build a seq2seq+attention translator.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 12 "Machine Translation"**](https://web.stanford.edu/~jurafsky/slp3/12.pdf) — **Jurafsky & Martin** — encoder–decoder and attention in the standard text.
- [Dive into Deep Learning — **Ch. 10 "Modern Recurrent Neural Networks" (encoder–decoder, seq2seq, attention)**](https://d2l.ai/chapter_recurrent-modern/index.html) — **Zhang et al.** — runnable seq2seq code.

## 🔗 In this platform
- Prior step: [04 N-gram Language Models](04-N-gram-Language-Models-and-Smoothing.md) — the statistical predecessor this replaces.
- Canonical homes: attention & transformers live in [5. Deep Learning](../../5.%20Deep_Learning/README.md); applied translation in [12 Machine Translation](12-Machine-Translation.md).
- Concept depth (the *why*): [AI-ML-intuition 1.06 Scaled Dot-Product (attention scores)](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md)
