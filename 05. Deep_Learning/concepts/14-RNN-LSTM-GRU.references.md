---
id: "05-deep-learning/rnn-lstm-gru/references"
topic: "RNN / LSTM / GRU — References"
parent: "05-deep-learning/rnn-lstm-gru"
type: references
updated: 2026-06-21
---

# RNN / LSTM / GRU — references and further reading

> Companion link library for **[RNN / LSTM / GRU](14-RNN-LSTM-GRU.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [RNNs, Clearly Explained](https://www.youtube.com/watch?v=AsNTP8Kwu80) (**StatQuest**), then read [The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) (**Karpathy**). *What a hidden state does and what RNNs can generate.*
2. **See why LSTMs work** — read [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (**Chris Olah**), the canonical gate diagrams. *The clearest picture of the cell state and the three gates.*
3. **Get the math** — [LSTM, Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) (**StatQuest**) + [d2l Ch. 9–10](https://d2l.ai/chapter_recurrent-neural-networks/index.html). *BPTT, gradient flow through the cell state, and the gate equations.*
4. **Read the sources** — [LSTM (Hochreiter & Schmidhuber 1997)](https://deeplearning.cs.cmu.edu/F23/document/readings/LSTM.pdf) → [GRU (Cho et al. 2014)](https://arxiv.org/abs/1406.1078) → [empirical comparison](https://arxiv.org/abs/1412.3555).
5. **Make it concrete** — implement an RNN/LSTM with [d2l Ch. 10](https://d2l.ai/chapter_recurrent-modern/lstm.html).

**Videos**:
- [Recurrent Neural Networks (RNNs), Clearly Explained](https://www.youtube.com/watch?v=AsNTP8Kwu80) — **StatQuest (Josh Starmer)** — gentle, from-scratch intuition for the hidden state and unrolling.
- [Long Short-Term Memory (LSTM), Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) — **StatQuest (Josh Starmer)** — walks through every gate with numbers.
- [Illustrated Guide to LSTM's and GRU's](https://www.youtube.com/watch?v=8HyCNIVRbSU) — **The AI Hacker (Michael Phi)** — the clearest animated comparison of LSTM vs GRU gates.
- [Illustrated Guide to Recurrent Neural Networks](https://www.youtube.com/watch?v=LHXXI4-IEns) — **The AI Hacker (Michael Phi)** — visual intuition for how recurrence carries information through time.

**Interactive & visual**:
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — **Chris Olah** — the gold-standard illustrated walk-through of the cell state and gates (the diagram everyone copies).
- [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/) — **Distill** — an interactive piece bridging RNNs to attention (the motivation for what came next).

**Courses (free)**:
- [Stanford CS224N — Recurrent Networks & LSTMs](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the definitive lectures on RNNs, BPTT, and gated units (with vanishing-gradient analysis).
- [MIT 6.S191 — Recurrent Neural Networks](http://introtodeeplearning.com/) — **MIT (Amini et al.)** — concise, current lecture on sequence modeling and LSTMs.

**Articles / blogs (free, no paywall)**:
- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — **Andrej Karpathy** — what RNNs can learn to generate, with intuition and code.

**Key papers**:
- [Long Short-Term Memory](https://deeplearning.cs.cmu.edu/F23/document/readings/LSTM.pdf) — **Hochreiter & Schmidhuber (1997)** — introduces the LSTM cell and constant error carousel.
- [Learning Phrase Representations using RNN Encoder–Decoder (GRU)](https://arxiv.org/abs/1406.1078) — **Cho et al. (2014)** — introduces the GRU and the encoder–decoder framing.
- [Empirical Evaluation of Gated RNNs on Sequence Modeling](https://arxiv.org/abs/1412.3555) — **Chung et al. (2014)** — the LSTM-vs-GRU head-to-head.
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — **Sutskever et al. (2014)** — LSTM seq2seq; the precursor to attention/transformers.

**Books (free chapters)**:
- [Dive into Deep Learning — Ch. 9 (RNNs) + Ch. 10 (Modern RNNs: LSTM, GRU, seq2seq)](https://d2l.ai/chapter_recurrent-neural-networks/index.html) — **Zhang et al.** — recurrence, BPTT, and gated units with runnable code.
- [Deep Learning — Ch. 10 "Sequence Modeling: Recurrent and Recursive Nets"](https://www.deeplearningbook.org/contents/rnn.html) — **Goodfellow, Bengio & Courville** — the rigorous treatment of BPTT, gradient flow, and LSTMs.
- [Speech and Language Processing, 3rd ed. — Ch. 9 "RNNs and LSTMs"](https://web.stanford.edu/~jurafsky/slp3/9.pdf) — **Jurafsky & Martin** — RNNs/LSTMs framed for language, free draft chapter.

**In this platform**:
- Concept page (full explanation): [RNN / LSTM / GRU](14-RNN-LSTM-GRU.md)
- Concept depth (the *why*): [AI-ML-intuition 4.07 Gating Mechanisms (LSTM/GRU)](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.07_Gating_Mechanisms_LSTM_GRU.md)
- Prerequisite: [Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs.md)
- Related: [Vanishing / Exploding Gradients](06-Vanishing-Exploding-Gradients.md) (why RNNs forget) · [Transformer Architecture](16-Transformer-Architecture.md) (what replaced RNNs) · [06. NLP](../../06.%20NLP/concepts/README.md)
- Field overview: [Deep Learning](../README.md)
