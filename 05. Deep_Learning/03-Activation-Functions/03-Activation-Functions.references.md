---
id: "05-deep-learning/activation-functions/references"
topic: "Activation Functions — References"
parent: "05-deep-learning/activation-functions"
type: references
updated: 2026-06-22
---

# Activation Functions — references and further reading

> Companion link library for **[Activation Functions](03-Activation-Functions.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first within each group. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [ReLU In Action](https://www.youtube.com/watch?v=68BZ5f7P94E) (**StatQuest**). *See how a piecewise-linear unit bends a function to fit, then combines with others.*
2. **Survey the zoo** — read [Activation Functions in Neural Networks](https://www.jeremyjordan.me/neural-networks-activation-functions/) (**Jeremy Jordan**), then toggle the Activation dropdown in [TensorFlow Playground](https://playground.tensorflow.org/). *The functions side by side, then watch Linear vs ReLU carve a boundary live.*
3. **Get the math** — read [CS231n: Commonly used activation functions](https://cs231n.github.io/neural-networks-1/). *Saturation, the dead-ReLU problem, zero-centering, and how to choose.*
4. **Understand softmax** — read [Softmax from scratch](https://e2eml.school/softmax.html) (**Brandon Rohrer**). *Why softmax exponentiates-and-normalizes, and its Jacobian.*
5. **Read the modern sources** — skim [GELU](https://arxiv.org/abs/1606.08415), [Swish/SiLU](https://arxiv.org/abs/1710.05941), and [GLU Variants (SwiGLU)](https://arxiv.org/abs/2002.05202). *The smooth and gated activations behind transformers and modern LLMs.*

**Videos**:
- [Neural Networks Pt. 3: ReLU In Action](https://www.youtube.com/watch?v=68BZ5f7P94E) — **StatQuest (Josh Starmer)** — exactly how ReLU units combine to fit a curve.
- [Activation Functions in a Neural Network explained](https://www.youtube.com/watch?v=m0pIlLfpXWE) — **deeplizard** — clear visual tour of sigmoid, tanh, and ReLU and when to use each.
- [Activation Functions — EXPLAINED!](https://www.youtube.com/watch?v=s-V7gKrsels) — **CodeEmporium** — why nonlinearity is required and how the common choices differ.
- [What is an Activation Function in a Neural Network?](https://www.youtube.com/watch?v=Y9qdKsOHRjA) — **Learn With Jay** — from-scratch motivation and the main activation types.

**Interactive & visual**:
- [TensorFlow Playground](https://playground.tensorflow.org/) — **Google** — switch the Activation dropdown (ReLU / Tanh / Sigmoid / Linear) and watch the decision boundary's ability to curve change in real time; set Linear to *see* "depth is an illusion."

**Courses (free)**:
- [Stanford CS231n — Neural Networks](https://cs231n.github.io/neural-networks-1/) — **Stanford (Karpathy / Li / Johnson)** — the canonical comparison of activation functions and their failure modes.
- [Dive into Deep Learning — Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) — **Zhang et al.** — activations introduced with plots, gradients, and runnable code.

**Articles / blogs (free, no paywall)**:
- [Neural Network Activation Functions](https://www.jeremyjordan.me/neural-networks-activation-functions/) — **Jeremy Jordan** — the clearest single-page survey with gradient intuition.
- [Softmax from scratch](https://e2eml.school/softmax.html) — **Brandon Rohrer** — softmax derived and visualized, with its Jacobian.
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) — **Gregory Gundersen** — why and how to compute softmax/log-softmax stably (the max-subtraction).

**Key papers**:
- [Understanding the difficulty of training deep feedforward networks](https://proceedings.mlr.press/v9/glorot10a.html) — **Glorot & Bengio (2010)** — the saturation / vanishing-gradient analysis and Xavier initialization.
- [Rectified Linear Units Improve Restricted Boltzmann Machines (ReLU)](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) — **Nair & Hinton (2010)** — the introduction of ReLU.
- [Deep Sparse Rectifier Neural Networks](https://proceedings.mlr.press/v15/glorot11a.html) — **Glorot, Bordes & Bengio (2011)** — the case for ReLU in deep networks (sparsity, no saturation).
- [Rectifier Nonlinearities Improve Neural Network Acoustic Models (Leaky ReLU)](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf) — **Maas, Hannun & Ng (2013)** — the leaky-slope fix for dying units.
- [Delving Deep into Rectifiers (PReLU + He init)](https://arxiv.org/abs/1502.01852) — **He et al. (2015)** — Parametric ReLU and the initialization tuned for rectifiers.
- [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELU)](https://arxiv.org/abs/1511.07289) — **Clevert, Unterthiner & Hochreiter (2015)** — the saturating, mean-shifting negative branch.
- [Self-Normalizing Neural Networks (SELU)](https://arxiv.org/abs/1706.02515) — **Klambauer et al. (2017)** — the fixed-point scaling that keeps activations normalized through depth.
- [Gaussian Error Linear Units (GELU)](https://arxiv.org/abs/1606.08415) — **Hendrycks & Gimpel (2016)** — the smooth Gaussian-gated activation used in BERT/GPT.
- [Searching for Activation Functions (Swish)](https://arxiv.org/abs/1710.05941) — **Ramachandran, Zoph & Le (2017)** — a learned smooth activation that often beats ReLU.
- [Sigmoid-Weighted Linear Units (SiLU)](https://arxiv.org/abs/1702.03118) — **Elfwing, Uchibe & Doya (2017)** — the $x\sigma(x)$ unit, from a reinforcement-learning setting.
- [Language Modeling with Gated Convolutional Networks (GLU)](https://arxiv.org/abs/1612.08083) — **Dauphin et al. (2017)** — the original gated linear unit.
- [GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/abs/2002.05202) — **Shazeer (2020)** — the gated SiLU feed-forward block now standard in LLaMA/PaLM/Mistral.

**Books (free chapters)**:
- [Dive into Deep Learning — §5.1 "Multilayer Perceptrons" (Activation Functions)](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) — **Zhang et al.** — ReLU/sigmoid/tanh with plots, gradients, and code.
- [Deep Learning — §6.3 "Hidden Units"](https://www.deeplearningbook.org/contents/mlp.html) — **Goodfellow, Bengio & Courville** — rigorous discussion of ReLU and activation design.
- [Neural Networks and Deep Learning — Ch. 1 (sigmoid neurons)](http://neuralnetworksanddeeplearning.com/chap1.html) — **Michael Nielsen** — why a smooth activation enables gradient learning.

**In this platform**:
- Concept page (full explanation): [Activation Functions](03-Activation-Functions.md)
- Concept depth (the *why*): [AI-ML-intuition 4.14 Activation Functions & Softmax](../../../AI-ML-intuition/Module_4_Stabilization/4D_Nonlinearities/4.14_Activation_Functions_and_Softmax.md)
- Prerequisite: [Perceptron & MLP](../01-Perceptron-and-MLP/01-Perceptron-and-MLP.md)
- Related: [Vanishing / Exploding Gradients](../06-Vanishing-Exploding-Gradients/06-Vanishing-Exploding-Gradients.md) (why saturating activations hurt) · [Backpropagation & Computational Graphs](../02-Backpropagation-and-Computational-Graphs/02-Backpropagation-and-Computational-Graphs.md) (where the chained derivatives multiply) · [Loss Functions](../04-Loss-Functions/04-Loss-Functions.md) (softmax + cross-entropy) · [Transformer Architecture](../16-Transformer-Architecture/16-Transformer-Architecture.md) (where GELU/SwiGLU and softmax sit)
- Field overview: [Deep Learning](../README.md)
