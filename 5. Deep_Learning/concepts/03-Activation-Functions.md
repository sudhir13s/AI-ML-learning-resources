---
id: "05-deep-learning/activation-functions"
topic: "Activation Functions (ReLU · GELU · sigmoid · tanh · softmax)"
parent: "05-deep-learning"
level: beginner
prereqs: ["feedforward-networks", "calculus"]
interview_frequency: high
updated: 2026-06-19
---

# Activation Functions (ReLU · GELU · sigmoid · tanh · softmax)
> The nonlinearity applied after each layer's affine transform. Without it, stacking layers collapses
> to a single linear map — the activation is what lets a network learn curved decision boundaries.
> **ReLU** and **GELU** dominate hidden layers; **softmax** turns logits into a probability
> distribution at the output; **sigmoid/tanh** persist in gates and binary heads.

**Why it matters:** a staple interview topic — explain why a nonlinearity is *required*, compare
ReLU vs sigmoid/tanh on **vanishing gradients** and saturation, know the dead-ReLU problem and its
fixes (Leaky ReLU, GELU), derive **softmax** and why it pairs with cross-entropy, and reason about
which activation goes where (hidden vs output, classification vs regression).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [ReLU In Action](https://www.youtube.com/watch?v=68BZ5f7P94E) (**StatQuest**). *See how a piecewise-linear unit bends a function to fit.*
2. **Survey the zoo** — read ⭐ [Activation Functions in Neural Networks](https://www.jeremyjordan.me/neural-networks-activation-functions/) (**Jeremy Jordan**). *Sigmoid, tanh, ReLU and friends side by side, with their gradient behavior.*
3. **Get the math** — read [CS231n: Commonly used activation functions](https://cs231n.github.io/neural-networks-1/). *Saturation, the dead-ReLU problem, and how to choose.*
4. **Understand softmax** — read [Softmax from scratch](https://e2eml.school/softmax.html) (**Brandon Rohrer**). *Why softmax exponentiates-and-normalizes, and its derivative.*
5. **Read the modern source** — skim [GELU](https://arxiv.org/abs/1606.08415) and [Swish/SiLU](https://arxiv.org/abs/1710.05941). *The smooth activations behind transformers and modern CNNs.*

## 🎓 Courses (free)
- [Stanford CS231n — Neural Networks](https://cs231n.github.io/neural-networks-1/) — **Stanford (Karpathy / Li / Johnson)** — the canonical comparison of activation functions and their failure modes.
- [Dive into Deep Learning — Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) — **Zhang et al.** — activations introduced with plots, gradients, and runnable code.

## 🎥 Videos
- [Neural Networks Pt. 3: ReLU In Action](https://www.youtube.com/watch?v=68BZ5f7P94E) — **StatQuest (Josh Starmer)** — exactly how ReLU units combine to fit a curve.
- [Activation Functions in a Neural Network explained](https://www.youtube.com/watch?v=m0pIlLfpXWE) — **deeplizard** — clear visual tour of sigmoid, tanh, and ReLU and when to use each.
- [Activation Functions — EXPLAINED!](https://www.youtube.com/watch?v=s-V7gKrsels) — **CodeEmporium** — why nonlinearity is required and how the common choices differ.
- [What is an Activation Function in a Neural Network?](https://www.youtube.com/watch?v=Y9qdKsOHRjA) — **Learn With Jay** — from-scratch motivation and the main activation types.

## 📄 Key Papers
- [Gaussian Error Linear Units (GELU)](https://arxiv.org/abs/1606.08415) — **Hendrycks & Gimpel (2016)** — the smooth activation used in BERT/GPT-style transformers.
- [Searching for Activation Functions (Swish/SiLU)](https://arxiv.org/abs/1710.05941) — **Ramachandran, Zoph & Le (2017)** — a learned smooth activation that often beats ReLU.
- [Delving Deep into Rectifiers (PReLU)](https://arxiv.org/abs/1502.01852) — **He et al. (2015)** — Parametric ReLU plus the initialization tuned for rectifiers.

## 📰 Articles / Blogs (free, no paywall)
- [Neural Network Activation Functions](https://www.jeremyjordan.me/neural-networks-activation-functions/) — **Jeremy Jordan** — the clearest single-page survey with gradient intuition.
- [CS231n — Activation Functions](https://cs231n.github.io/neural-networks-1/) — **Stanford CS231n** — saturation, dead ReLUs, and practical guidance.
- [Softmax from scratch](https://e2eml.school/softmax.html) — **Brandon Rohrer** — softmax derived and visualized, with its Jacobian.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **§5.1 "Multilayer Perceptrons"** (Activation Functions)](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) — **Zhang et al.** — ReLU/sigmoid/tanh with plots, gradients, and code.
- [Deep Learning — **§6.3 "Hidden Units"**](https://www.deeplearningbook.org/contents/mlp.html) — **Goodfellow, Bengio & Courville** — rigorous discussion of ReLU and activation design.
- [Neural Networks and Deep Learning — **Ch. 1 (sigmoid neurons)**](http://neuralnetworksanddeeplearning.com/chap1.html) — **Michael Nielsen** — why a smooth activation enables gradient learning.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.14 Activation Functions & Softmax](../../../AI-ML-intuition/Module_4_Stabilization/4D_Nonlinearities/4.14_Activation_Functions_and_Softmax.md)
- Prerequisite: [01 Perceptron & MLP](01-Perceptron-and-MLP.md)
- Related concepts: [06 Vanishing / Exploding Gradients](06-Vanishing-Exploding-Gradients.md) (why saturating activations hurt) · [04 Loss Functions](04-Loss-Functions.md) (softmax + cross-entropy)
- Field overview: [Deep Learning](../README.md)
