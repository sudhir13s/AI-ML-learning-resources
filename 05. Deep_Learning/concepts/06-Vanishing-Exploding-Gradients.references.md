---
id: "05-deep-learning/vanishing-exploding-gradients/references"
topic: "Vanishing / Exploding Gradients — References"
parent: "05-deep-learning/vanishing-exploding-gradients"
type: references
updated: 2026-06-21
---

# Vanishing / Exploding Gradients — references and further reading

> Companion link library for **[Vanishing / Exploding Gradients & Gradient Clipping](06-Vanishing-Exploding-Gradients.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Vanishing/Exploding Gradients (C2W1L10)](https://www.youtube.com/watch?v=qhXZsFVxGKo) (**Andrew Ng**). *Why depth compounds the gradient signal up or down.*
2. **See it in code** — watch [makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) (**Karpathy**). *Watch gradients die/blow up and the fixes applied live.*
3. **Get the math** — read [d2l: Numerical Stability and Initialization](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html). *The variance/Jacobian argument behind both failure modes.*
4. **Read the source** — [On the difficulty of training RNNs](https://arxiv.org/abs/1211.5063) (**Pascanu et al. 2013**). *The exploding-gradient analysis and the clipping fix.*
5. **Make it concrete** — add `clip_grad_norm_` to a deep/RNN model and watch divergence stop.

**Videos**:
- [Vanishing/Exploding Gradients (C2W1L10)](https://www.youtube.com/watch?v=qhXZsFVxGKo) — **DeepLearningAI (Andrew Ng)** — the clearest short explanation of why depth causes both failure modes.
- [Vanishing Gradient Problem, Quickly Explained](https://www.youtube.com/watch?v=8z3DFk4VxRo) — **Developers Hutt** — concise intuition for why gradients shrink through layers.
- [Vanishing Gradient Problem in RNNs Explained](https://www.youtube.com/watch?v=KFUSJBPFsYs) — **Super Data Science** — why recurrence makes the problem acute, and what gating does.

**Interactive & visual**:
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) — **Andrej Karpathy** — diagnoses and fixes gradient pathologies *live in code*, plotting activation/gradient histograms layer by layer.

**Courses (free)**:
- [Stanford CS231n — Neural Networks Part 2](https://cs231n.github.io/neural-networks-2/) — **Stanford (Karpathy / Li / Johnson)** — how init and activations control gradient magnitude.
- [Dive into Deep Learning — Numerical Stability & Initialization](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) — **Zhang et al.** — vanishing/exploding gradients derived, with code.

**Articles / blogs (free, no paywall)**:
- [CS231n — Gradient checks, sanity checks, babysitting learning](https://cs231n.github.io/neural-networks-3/) — **Stanford CS231n** — diagnosing dead/blown-up gradients in practice.
- [Why ResNets work: residual connections and gradient flow](https://d2l.ai/chapter_convolutional-modern/resnet.html) — **Zhang et al.** — how skip connections create a gradient highway.

**Key papers**:
- [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063) — **Pascanu, Mikolov & Bengio (2013)** — analyzes exploding gradients and introduces norm clipping.
- [Learning long-term dependencies with gradient descent is difficult](http://www.iro.umontreal.ca/~lisa/pointeurs/ieeetrnn94.pdf) — **Bengio, Simard & Frasconi (1994)** — the original vanishing-gradient analysis.
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) — **He et al. (2015)** — ReLU + He init as a gradient-flow fix.

**Books (free chapters)**:
- [Dive into Deep Learning — §5.4 "Numerical Stability and Initialization"](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) — **Zhang et al.** — both failure modes derived, with the init/activation fixes.
- [Deep Learning — §8.2.5 "Cliffs and Exploding Gradients" + §10.7 "Long-Term Dependencies"](https://www.deeplearningbook.org/contents/optimization.html) — **Goodfellow, Bengio & Courville** — the rigorous analysis and gradient clipping.
- [Neural Networks and Deep Learning — Ch. 5 "Why are deep nets hard to train?"](http://neuralnetworksanddeeplearning.com/chap5.html) — **Michael Nielsen** — the unstable-gradient problem explained from scratch.

**In this platform**:
- Concept page (full explanation): [Vanishing / Exploding Gradients](06-Vanishing-Exploding-Gradients.md)
- Concept depth (the *why*): [AI-ML-intuition 4.10 Gradient Clipping](../../../AI-ML-intuition/Module_4_Stabilization/4C_Training_Stability/4.10_Gradient_Clipping.md) · [4.12 Weight Initialization](../../../AI-ML-intuition/Module_4_Stabilization/4C_Training_Stability/4.12_Weight_Initialization_Xavier_He.md)
- Prerequisite: [Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs.md)
- Related: [Activation Functions](03-Activation-Functions.md) (saturation) · [Weight Initialization](05-Weight-Initialization.md) · [Normalization](11-Normalization.md) · [Residual / Skip Connections](18-Residual-Skip-Connections.md)
- Field overview: [Deep Learning](../README.md)
