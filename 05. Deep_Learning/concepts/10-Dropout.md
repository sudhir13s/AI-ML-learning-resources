---
id: "05-deep-learning/dropout"
topic: "Dropout"
parent: "05-deep-learning"
level: intermediate
prereqs: ["feedforward-networks", "regularization"]
interview_frequency: high
updated: 2026-06-19
---

# Dropout
> A stochastic regularizer: during training, randomly "drop" (zero out) a fraction of units each
> forward pass, forcing the network not to rely on any single neuron. It approximates training an
> ensemble of exponentially many sub-networks that share weights. At test time all units are kept and
> activations are scaled (or inverted-scaled during training) to match expectations.

**Why it matters:** a very common regularization question — explain why dropping units prevents
co-adaptation and acts like an ensemble, the train-vs-test difference (active dropout vs scaled
weights), why we use **inverted dropout** to avoid test-time scaling, and how dropout interacts with
BatchNorm (often you pick one). Be ready to reason about the dropout rate as a hyperparameter.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Dropout Regularization (C2W1L06)](https://www.youtube.com/watch?v=D8PJAL-MZv8) (**Andrew Ng**). *Why randomly killing units spreads out the learned weights.*
2. **See why it works** — watch ⭐ [Understanding Dropout (C2W1L07)](https://www.youtube.com/watch?v=ARq74QuavAo) (**Andrew Ng**). *The ensemble/no-co-adaptation argument, intuitively.*
3. **Get the math** — read [d2l: Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html). *Inverted dropout, expectations, and code.*
4. **Read the source** — [Dropout: A Simple Way to Prevent Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) (**Srivastava et al., 2014**). *The paper, with the ensemble interpretation.*
5. **Make it concrete** — toggle `nn.Dropout(p)` on a small net and watch the train/val gap close. *Seeing it regularize live makes the mechanism stick.*

## 🎓 Courses (free)
- [Stanford CS231n — Neural Networks Part 2 (Dropout)](https://cs231n.github.io/neural-networks-2/) — **Stanford (Karpathy / Li / Johnson)** — dropout as regularization with inverted-dropout implementation.
- [Dive into Deep Learning — Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html) — **Zhang et al.** — the method derived and implemented from scratch.

## 🎥 Videos
- [Dropout Regularization (C2W1L06)](https://www.youtube.com/watch?v=D8PJAL-MZv8) — **DeepLearningAI (Andrew Ng)** — the mechanics: random unit removal each pass.
- [Understanding Dropout (C2W1L07)](https://www.youtube.com/watch?v=ARq74QuavAo) — **DeepLearningAI (Andrew Ng)** — why it prevents co-adaptation and acts like an ensemble.
- [Regularization (C2W1L04)](https://www.youtube.com/watch?v=6g0t3Phly2M) — **DeepLearningAI (Andrew Ng)** — situates dropout alongside L1/L2.
- [Training a Neural Network explained](https://www.youtube.com/watch?v=sZAlS3_dnk0) — **deeplizard** — where dropout fits in the train/validation/test workflow.

## 📄 Key Papers
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) — **Srivastava et al. (2014)** — the canonical paper and ensemble interpretation.
- [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) — **Hinton et al. (2012)** — the original dropout proposal.
- [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142) — **Gal & Ghahramani (2016)** — dropout at test time as approximate Bayesian uncertainty.

## 📰 Articles / Blogs (free, no paywall)
- [CS231n — Dropout](https://cs231n.github.io/neural-networks-2/) — **Stanford CS231n** — inverted dropout and why test time stays clean.
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — **Andrej Karpathy** — where dropout fits in a real regularization strategy.
- [Cross-Entropy / training regularizers overview](https://www.pinecone.io/learn/cross-entropy-loss/) — **Pinecone** — concise context for how regularizers shape the objective.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **§5.6 "Dropout"**](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html) — **Zhang et al.** — inverted dropout, expectations, and code.
- [Deep Learning — **§7.12 "Dropout"**](https://www.deeplearningbook.org/contents/regularization.html) — **Goodfellow, Bengio & Courville** — the rigorous ensemble/bagging view of dropout.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.11 Dropout](../../../AI-ML-intuition/Module_2_Optimization/2.11_Dropout.md)
- Prerequisite: [09 Regularization](09-Regularization.md)
- Related concept: [11 Normalization](11-Normalization.md) (often chosen instead of, or alongside, dropout)
- Field overview: [Deep Learning](../README.md)
