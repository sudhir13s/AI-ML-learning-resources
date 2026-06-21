---
id: "05-deep-learning/regularization/references"
topic: "Regularization — References"
parent: "05-deep-learning/regularization"
type: references
updated: 2026-06-21
---

# Regularization — references and further reading

> Companion link library for **[Regularization](09-Regularization.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) (**StatQuest**). *Why penalizing weights reduces variance.*
2. **See it in nets** — watch [Why Regularization Reduces Overfitting](https://www.youtube.com/watch?v=NyG-7nRpsW8) (**Andrew Ng**), then toggle the regularization dropdown in [TensorFlow Playground](https://playground.tensorflow.org/). *Watch a jagged boundary turn smooth.*
3. **Get the math** — read [d2l: Weight Decay](https://d2l.ai/chapter_linear-regression/weight-decay.html). *L2 derived as a gradient modification, with code.*
4. **Connect to generalization** — read [CS231n: Regularization](https://cs231n.github.io/neural-networks-2/). *L1/L2/elastic-net/max-norm and when each helps.*
5. **Make it concrete** — add `weight_decay` and early stopping to a training loop and watch the train/val gap shrink.

**Videos**:
- [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) — **StatQuest (Josh Starmer)** — the clearest intuition for why an L2 penalty helps.
- [Why Regularization Reduces Overfitting (C2W1L05)](https://www.youtube.com/watch?v=NyG-7nRpsW8) — **DeepLearningAI (Andrew Ng)** — how the penalty drives the net toward simpler functions.
- [Regularization (C2W1L04)](https://www.youtube.com/watch?v=6g0t3Phly2M) — **DeepLearningAI (Andrew Ng)** — L2 and L1 penalties added to a neural-net cost.
- [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — **StatQuest (Josh Starmer)** — the trade-off that regularization manages.

**Interactive & visual**:
- [TensorFlow Playground](https://playground.tensorflow.org/) — **Google** — switch the Regularization dropdown (None / L1 / L2) and watch overfitting cured live as the decision boundary smooths.

**Courses (free)**:
- [Stanford CS231n — Neural Networks Part 2 (Regularization)](https://cs231n.github.io/neural-networks-2/) — **Stanford (Karpathy / Li / Johnson)** — L1/L2/max-norm and their effects on the loss surface.
- [Dive into Deep Learning — Weight Decay & Generalization](https://d2l.ai/chapter_linear-regression/weight-decay.html) — **Zhang et al.** — regularization derived from first principles, with code.

**Articles / blogs (free, no paywall)**:
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — **Andrej Karpathy** — practical regularization order (overfit first, then weight decay, dropout, augmentation, early stopping).
- [Setting the learning rate / training tricks](https://www.jeremyjordan.me/nn-learning-rate/) — **Jeremy Jordan** — early stopping and other implicit regularizers in context.

**Key papers**:
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — **Loshchilov & Hutter (2017)** — why L2 penalty ≠ weight decay under adaptive optimizers.
- [A Simple Weight Decay Can Improve Generalization](https://proceedings.neurips.cc/paper/1991/hash/8eefcfdf5990e441f0fb6f3fad709e21-Abstract.html) — **Krogh & Hertz (1991)** — the classic analysis of weight decay's effect.

**Books (free chapters)**:
- [Deep Learning — Ch. 7 "Regularization for Deep Learning"](https://www.deeplearningbook.org/contents/regularization.html) — **Goodfellow, Bengio & Courville** — the definitive chapter: L1/L2, early stopping, dropout, augmentation.
- [The Elements of Statistical Learning — §3.4 (Ridge & Lasso)](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the original diamond-vs-circle geometry; free PDF.
- [Dive into Deep Learning — §3.7 "Weight Decay" + §5.6 "Generalization"](https://d2l.ai/chapter_linear-regression/weight-decay.html) — **Zhang et al.** — regularization with runnable experiments.
- [Neural Networks and Deep Learning — Ch. 3 (overfitting & regularization)](http://neuralnetworksanddeeplearning.com/chap3.html) — **Michael Nielsen** — L2, early stopping, and why they help, from scratch.

**In this platform**:
- Concept page (full explanation): [Regularization](09-Regularization.md)
- Concept depth (the *why*): [AI-ML-intuition 2.10 Regularization (L1/L2)](../../../AI-ML-intuition/Module_2_Optimization/2.10_Regularization_L1_L2.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Prerequisite: [Loss Functions](04-Loss-Functions.md)
- Related: [Dropout](10-Dropout.md) (a stochastic regularizer) · [Optimizers](07-Optimizers.md) (AdamW and decoupled weight decay)
- Field overview: [Deep Learning](../README.md)
