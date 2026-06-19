---
id: "06-nlp/machine-translation"
topic: "Machine Translation"
parent: "06-nlp"
level: intermediate
prereqs: ["seq2seq-encoder-decoder", "attention", "tokenization"]
interview_frequency: medium
updated: 2026-06-19
---

# Machine Translation
> Automatically translating text between languages. The field that drove modern NLP: from
> statistical phrase tables to **neural MT** (encoder–decoder + attention) to the **Transformer**,
> which was *invented for translation* before it took over everything.

**Why it matters:** MT is where seq2seq, attention, beam search, subword tokenization, and BLEU all
come together — and it's the historical reason the Transformer exists. Be ready to contrast **SMT vs
NMT**, explain why attention solved the long-sentence bottleneck, how **BPE/SentencePiece** handle
rare words across languages, why decoding uses **beam search**, and the strengths/limits of **BLEU**
as the evaluation metric.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Seq2Seq Encoder–Decoder Neural Networks](https://www.youtube.com/watch?v=L8HKweZIOmg) (**StatQuest**). *The translation pipeline before attention.*
2. **See attention in MT** — read ⭐ [Visualizing NMT (seq2seq + attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) (**Jay Alammar**), then watch [Stanford CS224N Lec 10 — NMT & attention](https://www.youtube.com/watch?v=IxQtK2SjWWM). *Why attention fixed long-sentence translation.*
3. **Get the math** — read [SLP3 Ch. 12](https://web.stanford.edu/~jurafsky/slp3/12.pdf). *Encoder–decoder MT, beam search, and BLEU, rigorously.*
4. **Read the sources** — [Bahdanau attention](https://arxiv.org/abs/1409.0473) → [GNMT](https://arxiv.org/abs/1609.08144) → [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *RNN attention → production NMT → the Transformer.*
5. **Make it concrete** — code it with the [TensorFlow NMT-with-attention tutorial](https://www.tensorflow.org/tutorials/text/nmt_with_attention). *Train a real translator.*

## 🎓 Courses (free)
- [NLP Course for You — Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) — **Lena Voita** — the MT chapter (a translation researcher's own course).
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the NMT + attention lecture in depth.

## 🎥 Videos
- [Seq2Seq Encoder–Decoder Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=L8HKweZIOmg) — **StatQuest (Josh Starmer)** — gentle translation intuition.
- [Stanford CS224N Lec 10 — Neural Machine Translation & Models with Attention](https://www.youtube.com/watch?v=IxQtK2SjWWM) — **Stanford (Manning)** — the rigorous NMT lecture.
- [Encoder–decoder architecture: Overview](https://www.youtube.com/watch?v=zbdong_h-x4) — **Google Cloud Tech** — concise official framing.
- [Transformer Neural Networks — EXPLAINED! (Attention is all you need)](https://www.youtube.com/watch?v=TQQlZhbC5ps) — **CodeEmporium** — the architecture that replaced RNN-MT.

## 📄 Key Papers
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — **Bahdanau, Cho & Bengio (2015)** — attention for translation.
- [Google's Neural Machine Translation (GNMT)](https://arxiv.org/abs/1609.08144) — **Wu et al. (2016)** — production-scale NMT + WordPiece.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — the Transformer, introduced *for* translation.

## 📰 Articles / Blogs (free, no paywall)
- [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) — **Jay Alammar** — the definitive visual MT explainer.
- [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) — **Lena Voita** — free, rigorous MT chapter.
- [Neural Machine Translation with attention (TensorFlow tutorial)](https://www.tensorflow.org/tutorials/text/nmt_with_attention) — **TensorFlow** — build and train an NMT model.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 12 "Machine Translation"**](https://web.stanford.edu/~jurafsky/slp3/12.pdf) — **Jurafsky & Martin** — encoder–decoder MT, beam search, BLEU.
- [Dive into Deep Learning — **Ch. 10 (seq2seq, attention, NMT)**](https://d2l.ai/chapter_recurrent-modern/index.html) — **Zhang et al.** — runnable NMT code.

## 🔗 In this platform
- Prior step: [08 Sequence-to-Sequence & Encoder–Decoder](08-Sequence-to-Sequence-and-Encoder-Decoder.md) — the architecture MT runs on.
- Tokenization for MT: [02 Tokenization & Subword](02-Tokenization-and-Subword-Algorithms.md) · evaluation: [18 NLP Evaluation Metrics](18-NLP-Evaluation-Metrics.md) (BLEU).
- Concept depth (the *why*): [AI-ML-intuition 1.06 Scaled Dot-Product (attention)](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md)
