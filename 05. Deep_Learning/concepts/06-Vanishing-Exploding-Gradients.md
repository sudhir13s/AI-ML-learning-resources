---
id: "05-deep-learning/vanishing-exploding-gradients"
topic: "Vanishing / Exploding Gradients & Gradient Clipping"
parent: "05-deep-learning"
level: intermediate
prereqs: ["backpropagation", "activation-functions"]
interview_frequency: high
updated: 2026-06-19
---

# Vanishing / Exploding Gradients & Gradient Clipping
> In deep (and recurrent) networks, gradients are products of many layer Jacobians during
> backprop. If those factors are consistently < 1 the gradient **vanishes** — early layers stop
> learning; if > 1 it **explodes** — training diverges. The standard fixes are good activations
> (ReLU), variance-preserving init, residual connections, normalization, and — for the explosion
> side — **gradient clipping**.

**Why it matters:** a core "why is my deep/RNN model not training?" question — explain how the chain
rule multiplies Jacobians and why repeated multiplication shrinks or blows up the signal, why
sigmoid/tanh saturation makes it worse, and the full toolkit of fixes (ReLU, He init, BatchNorm/
LayerNorm, residuals, LSTM gating, and gradient clipping by norm/value).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Vanishing/Exploding Gradients (C2W1L10)](https://www.youtube.com/watch?v=qhXZsFVxGKo) (**Andrew Ng**). *Why depth compounds the gradient signal up or down.*
2. **See it in code** — watch ⭐ [makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) (**Karpathy**). *Watch gradients die/blow up and the fixes applied live.*
3. **Get the math** — read [d2l: Numerical Stability and Initialization](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html). *The variance/Jacobian argument behind both failure modes.*
4. **Read the source** — [On the difficulty of training RNNs](https://arxiv.org/abs/1211.5063) (**Pascanu et al., 2013**). *The exploding-gradient analysis and the clipping fix.*
5. **Make it concrete** — add gradient clipping (`clip_grad_norm_`) to a deep/RNN model and watch divergence stop. *Seeing the norm get capped makes clipping click.*

## 🎓 Courses (free)
- [Stanford CS231n — Neural Networks Part 2](https://cs231n.github.io/neural-networks-2/) — **Stanford (Karpathy / Li / Johnson)** — how init and activations control gradient magnitude.
- [Dive into Deep Learning — Numerical Stability & Initialization](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) — **Zhang et al.** — vanishing/exploding gradients derived, with code.

## 🎥 Videos
- [Vanishing/Exploding Gradients (C2W1L10)](https://www.youtube.com/watch?v=qhXZsFVxGKo) — **DeepLearningAI (Andrew Ng)** — the clearest short explanation of why depth causes both failure modes.
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) — **Andrej Karpathy** — diagnoses and fixes gradient pathologies live in code.
- [Vanishing Gradient Problem, Quickly Explained](https://www.youtube.com/watch?v=8z3DFk4VxRo) — **Developers Hutt** — concise intuition for why gradients shrink through layers.
- [Vanishing Gradient Problem in RNNs Explained](https://www.youtube.com/watch?v=KFUSJBPFsYs) — **Super Data Science** — why recurrence makes the problem acute, and what gating does.

## 📄 Key Papers
- [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063) — **Pascanu, Mikolov & Bengio (2013)** — analyzes exploding gradients and introduces norm clipping.
- [Learning long-term dependencies with gradient descent is difficult](http://www.iro.umontreal.ca/~lisa/pointeurs/ieeetrnn94.pdf) — **Bengio, Simard & Frasconi (1994)** — the original vanishing-gradient analysis.
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) — **He et al. (2015)** — ReLU + tuned init as a gradient-flow fix.

## 📰 Articles / Blogs (free, no paywall)
- [CS231n — Gradient checks, sanity checks, babysitting learning](https://cs231n.github.io/neural-networks-3/) — **Stanford CS231n** — diagnosing dead/blown-up gradients in practice.
- [The Vanishing Gradient Problem](https://www.deeplearningbook.org/contents/rnn.html) — **Goodfellow et al. (§10.7)** — the canonical written treatment for recurrent nets.
- [Why ResNets work: residual connections and gradient flow](https://d2l.ai/chapter_convolutional-modern/resnet.html) — **Zhang et al.** — how skip connections create a gradient highway.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **§5.4 "Numerical Stability and Initialization"**](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) — **Zhang et al.** — both failure modes derived, with the init/activation fixes.
- [Deep Learning — **§8.2.5 "Cliffs and Exploding Gradients"** + **§10.7 "Long-Term Dependencies"**](https://www.deeplearningbook.org/contents/optimization.html) — **Goodfellow, Bengio & Courville** — the rigorous analysis and gradient clipping.
- [Neural Networks and Deep Learning — **Ch. 5 "Why are deep nets hard to train?"**](http://neuralnetworksanddeeplearning.com/chap5.html) — **Michael Nielsen** — the unstable-gradient problem explained from scratch.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.10 Gradient Clipping](../../../AI-ML-intuition/Module_4_Stabilization/4C_Training_Stability/4.10_Gradient_Clipping.md) · [4.12 Weight Initialization](../../../AI-ML-intuition/Module_4_Stabilization/4C_Training_Stability/4.12_Weight_Initialization_Xavier_He.md)
- Prerequisite: [02 Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs.md)
- Related concepts: [05 Weight Initialization](05-Weight-Initialization.md) · [11 Normalization](11-Normalization.md) · [18 Residual / Skip Connections](18-Residual-Skip-Connections.md)
- Field overview: [Deep Learning](../README.md)
