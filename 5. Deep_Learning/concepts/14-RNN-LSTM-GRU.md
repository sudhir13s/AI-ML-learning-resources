---
id: "05-deep-learning/rnn-lstm-gru"
topic: "RNN / LSTM / GRU"
parent: "05-deep-learning"
level: intermediate
prereqs: ["feedforward-networks", "backpropagation"]
interview_frequency: high
updated: 2026-06-19
---

# RNN / LSTM / GRU
> Recurrent networks process sequences by maintaining a hidden state that carries information from
> one step to the next, sharing weights across time. Plain RNNs struggle to remember far-back
> context (vanishing/exploding gradients); **LSTMs** and **GRUs** add gated memory cells that learn
> what to keep, forget, and output — letting them model long-range dependencies.

**Why it matters:** a recurring sequence-modeling question — explain backpropagation-through-time,
*why* vanilla RNNs suffer vanishing/exploding gradients, how an **LSTM**'s forget/input/output gates
and cell state fix it, how a **GRU** simplifies that to update/reset gates, and the trade-offs vs
transformers (sequential compute, no parallel training, but cheap streaming inference).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: RNNs, Clearly Explained](https://www.youtube.com/watch?v=AsNTP8Kwu80), then ⭐ read [The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) (**Karpathy**). *See what a hidden state does and what RNNs can generate.*
2. **See why LSTMs work** — read [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (**Chris Olah**), the canonical gate diagrams. *The clearest picture of the cell state and the three gates.*
3. **Get the math** — [StatQuest: LSTM](https://www.youtube.com/watch?v=YCzL96nL7j0) + [d2l Ch. 9–10](https://d2l.ai/chapter_recurrent-neural-networks/index.html). *BPTT, gradient flow through the cell state, and the gate equations.*
4. **Read the sources** — [LSTM (Hochreiter & Schmidhuber, 1997)](https://deeplearning.cs.cmu.edu/F23/document/readings/LSTM.pdf) → [GRU (Cho et al., 2014)](https://arxiv.org/abs/1406.1078) → [empirical RNN comparison](https://arxiv.org/abs/1412.3555). *The originals, then the head-to-head.*
5. **Make it concrete** — implement an RNN/LSTM with [d2l Ch. 9–10](https://d2l.ai/chapter_recurrent-modern/lstm.html). *Coding BPTT and the gates makes the gradient story stick.*

## 🎓 Courses (free)
- [Stanford CS224N — Recurrent Networks & LSTMs](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the definitive lectures on RNNs, BPTT, and gated units (with vanishing-gradient analysis).
- [MIT 6.S191 — Recurrent Neural Networks](http://introtodeeplearning.com/) — **MIT (Amini et al.)** — concise, current lecture on sequence modeling and LSTMs.

## 🎥 Videos
- [Recurrent Neural Networks (RNNs), Clearly Explained](https://www.youtube.com/watch?v=AsNTP8Kwu80) — **StatQuest (Josh Starmer)** — gentle, from-scratch intuition for the hidden state and unrolling.
- [Long Short-Term Memory (LSTM), Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) — **StatQuest (Josh Starmer)** — walks through every gate with numbers.
- [Illustrated Guide to LSTM's and GRU's](https://www.youtube.com/watch?v=8HyCNIVRbSU) — **The AI Hacker (Michael Phi)** — the clearest animated comparison of LSTM vs GRU gates.
- [Illustrated Guide to Recurrent Neural Networks](https://www.youtube.com/watch?v=LHXXI4-IEns) — **The AI Hacker (Michael Phi)** — visual intuition for how recurrence carries information through time.

## 📄 Key Papers
- [Long Short-Term Memory](https://deeplearning.cs.cmu.edu/F23/document/readings/LSTM.pdf) — **Hochreiter & Schmidhuber (1997)** — introduces the LSTM cell and constant error carousel.
- [Learning Phrase Representations using RNN Encoder–Decoder (GRU)](https://arxiv.org/abs/1406.1078) — **Cho et al. (2014)** — introduces the GRU and the encoder–decoder framing.
- [Empirical Evaluation of Gated RNNs on Sequence Modeling](https://arxiv.org/abs/1412.3555) — **Chung et al. (2014)** — the LSTM-vs-GRU head-to-head.
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — **Sutskever et al. (2014)** — LSTM seq2seq; the precursor to attention/transformers.

## 📰 Articles / Blogs (free, no paywall)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — **Chris Olah** — the gold-standard explanation of the cell state and gates (everyone cites it).
- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — **Andrej Karpathy** — what RNNs can learn to generate, with intuition and code.
- [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/) — **Distill** — interactive piece bridging RNNs to attention (the motivation for what came next).

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 9 (RNNs)** + **Ch. 10 (Modern RNNs: LSTM, GRU, seq2seq)**](https://d2l.ai/chapter_recurrent-neural-networks/index.html) — **Zhang et al.** — recurrence, BPTT, and gated units with runnable code.
- [Deep Learning — **Ch. 10 "Sequence Modeling: Recurrent and Recursive Nets"**](https://www.deeplearningbook.org/contents/rnn.html) — **Goodfellow, Bengio & Courville** — the rigorous treatment of BPTT, gradient flow, and LSTMs.
- [Speech and Language Processing, 3rd ed. — **Ch. 9 "RNNs and LSTMs"**](https://web.stanford.edu/~jurafsky/slp3/9.pdf) — **Jurafsky & Martin** — RNNs/LSTMs framed for language, free draft chapter.

## 🔗 In this platform
- Prerequisite: [02 Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs.md)
- Next concept: [16 Transformer Architecture](16-Transformer-Architecture.md) (what replaced RNNs for most sequence tasks)
- Related domain: [6. NLP](../../6.%20NLP/README.md) (sequence-to-sequence, machine translation)
