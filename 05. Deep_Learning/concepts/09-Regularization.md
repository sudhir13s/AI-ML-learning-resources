---
id: "05-deep-learning/regularization"
topic: "Regularization (L1/L2 · weight decay · early stopping)"
parent: "05-deep-learning"
level: intermediate
prereqs: ["loss-functions", "optimizers", "bias-variance"]
interview_frequency: high
updated: 2026-06-19
---

# Regularization (L1/L2 · weight decay · early stopping)
> Techniques that combat overfitting by constraining the model so it generalizes rather than
> memorizes. **L2** (weight decay) penalizes large weights, shrinking them toward zero; **L1**
> encourages sparsity (some weights exactly zero); **early stopping** halts training when validation
> loss turns up. All trade a little training fit for much better test performance.

**Why it matters:** a core generalization question — derive how an L2 penalty modifies the gradient
update (and why that equals "weight decay"), contrast L1 (sparse) vs L2 (small-but-dense), explain
why early stopping behaves like an implicit regularizer, and connect it all to the **bias–variance**
trade-off interviewers anchor on.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) (**StatQuest**). *Why penalizing weights reduces variance.*
2. **See it in nets** — watch ⭐ [Why Regularization Reduces Overfitting (C2W1L05)](https://www.youtube.com/watch?v=NyG-7nRpsW8) (**Andrew Ng**). *How the penalty pushes a net toward simpler functions.*
3. **Get the math** — read [d2l: Weight Decay](https://d2l.ai/chapter_linear-regression/weight-decay.html). *L2 derived as a gradient modification, with code.*
4. **Connect to generalization** — read [CS231n: Regularization](https://cs231n.github.io/neural-networks-2/). *L1/L2/elastic-net/max-norm and when each helps.*
5. **Make it concrete** — add `weight_decay` and early stopping to a training loop and watch the train/val gap shrink. *Seeing the curves separate makes regularization click.*

## 🎓 Courses (free)
- [Stanford CS231n — Neural Networks Part 2 (Regularization)](https://cs231n.github.io/neural-networks-2/) — **Stanford (Karpathy / Li / Johnson)** — L1/L2/max-norm and their effects on the loss surface.
- [Dive into Deep Learning — Weight Decay & Generalization](https://d2l.ai/chapter_linear-regression/weight-decay.html) — **Zhang et al.** — regularization derived from first principles, with code.

## 🎥 Videos
- [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) — **StatQuest (Josh Starmer)** — the clearest intuition for why an L2 penalty helps.
- [Why Regularization Reduces Overfitting (C2W1L05)](https://www.youtube.com/watch?v=NyG-7nRpsW8) — **DeepLearningAI (Andrew Ng)** — how the penalty drives the net toward simpler functions.
- [Regularization (C2W1L04)](https://www.youtube.com/watch?v=6g0t3Phly2M) — **DeepLearningAI (Andrew Ng)** — L2 and L1 penalties added to a neural-net cost.
- [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — **StatQuest (Josh Starmer)** — the trade-off that regularization manages.

## 📄 Key Papers
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — **Loshchilov & Hutter (2017)** — why L2 penalty ≠ weight decay under adaptive optimizers.
- [A Simple Weight Decay Can Improve Generalization](https://proceedings.neurips.cc/paper/1991/hash/8eefcfdf5990e441f0fb6f3fad709e21-Abstract.html) — **Krogh & Hertz (1991)** — the classic analysis of weight decay's effect.

## 📰 Articles / Blogs (free, no paywall)
- [CS231n — Regularization](https://cs231n.github.io/neural-networks-2/) — **Stanford CS231n** — L1/L2/elastic-net/max-norm with practical guidance.
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — **Andrej Karpathy** — practical regularization order (weight decay, early stopping, augmentation) for real training.
- [Setting the learning rate / training tricks](https://www.jeremyjordan.me/nn-learning-rate/) — **Jeremy Jordan** — early stopping and other implicit regularizers in context.

## 📚 Books (free, with chapters)
- [Deep Learning — **Ch. 7 "Regularization for Deep Learning"**](https://www.deeplearningbook.org/contents/regularization.html) — **Goodfellow, Bengio & Courville** — the definitive chapter: L1/L2, early stopping, dropout, augmentation.
- [Dive into Deep Learning — **§3.7 "Weight Decay"** + **§5.6 "Generalization"**](https://d2l.ai/chapter_linear-regression/weight-decay.html) — **Zhang et al.** — regularization with runnable experiments.
- [Neural Networks and Deep Learning — **Ch. 3 (overfitting & regularization)**](http://neuralnetworksanddeeplearning.com/chap3.html) — **Michael Nielsen** — L2, early stopping, and why they help, from scratch.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.10 Regularization (L1/L2)](../../../AI-ML-intuition/Module_2_Optimization/2.10_Regularization_L1_L2.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Prerequisite: [04 Loss Functions](04-Loss-Functions.md)
- Related concept: [10 Dropout](10-Dropout.md) (a stochastic regularizer)
- Field overview: [Deep Learning](../README.md)
