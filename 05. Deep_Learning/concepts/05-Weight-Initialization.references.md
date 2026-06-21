---
id: "05-deep-learning/weight-initialization/references"
topic: "Weight Initialization — References"
parent: "05-deep-learning/weight-initialization"
type: references
updated: 2026-06-21
---

# Weight Initialization — references and further reading

> Companion link library for **[Weight Initialization](05-Weight-Initialization.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Weight Initialization (C2W1L11)](https://www.youtube.com/watch?v=s2coXdufOzE) (**Andrew Ng**). *Why variance scaling keeps signals alive through depth.*
2. **See it interactively** — read [Initializing neural networks](https://www.deeplearning.ai/ai-notes/initialization/index.html) (**DeepLearning.AI**). *Sliders show too-small/too-large init collapsing or exploding activations live.*
3. **Get the math** — read [How to initialize deep neural networks: Xavier and Kaiming](https://pouannes.github.io/blog/initialization/) (**Pierre Ouannes**). *Full variance derivation for both schemes.*
4. **Read the sources** — [Glorot & Bengio (Xavier)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) → [He et al. (Kaiming, for ReLU)](https://arxiv.org/abs/1502.01852). *The two papers everyone cites.*
5. **Make it concrete** — work through [d2l: Numerical Stability and Initialization](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html).

**Videos**:
- [Weight Initialization in a Deep Network (C2W1L11)](https://www.youtube.com/watch?v=s2coXdufOzE) — **DeepLearningAI (Andrew Ng)** — the clearest short derivation of variance-preserving init.
- [Weight Initialization explained — reducing the vanishing gradient problem](https://www.youtube.com/watch?v=8krd5qKVw-Q) — **deeplizard** — why zero/large init fails and how Xavier/He fix it.
- [Xavier & He Initialization](https://www.youtube.com/watch?v=LKWatKGRZLI) — **Six Sigma Pro SMART** — side-by-side walk-through of both schemes.
- [Why don't we initialize the weights of a neural network to zero?](https://www.youtube.com/watch?v=LBMVyXfZQy0) — **Bhavesh Bhatt** — the symmetry-breaking argument, concretely.

**Interactive & visual**:
- [Initializing neural networks](https://www.deeplearning.ai/ai-notes/initialization/index.html) — **DeepLearning.AI** — interactive sliders showing how init scale collapses or explodes signal flow through a network, with the variance math.

**Courses (free)**:
- [Stanford CS231n — Neural Networks Part 2 (Initialization)](https://cs231n.github.io/neural-networks-2/) — **Stanford (Karpathy / Li / Johnson)** — the canonical notes on calibrating initial variances.
- [Dive into Deep Learning — Numerical Stability & Initialization](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) — **Zhang et al.** — vanishing/exploding signals and the init fixes, with code.

**Articles / blogs (free, no paywall)**:
- [How to initialize deep neural networks: Xavier and Kaiming](https://pouannes.github.io/blog/initialization/) — **Pierre Ouannes** — the full variance derivation for both schemes.
- [CS231n — Weight Initialization](https://cs231n.github.io/neural-networks-2/) — **Stanford CS231n** — calibrated initialization and the pitfalls of naive choices.

**Key papers**:
- [Understanding the difficulty of training deep feedforward networks (Xavier/Glorot)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — **Glorot & Bengio (2010)** — derives the variance-preserving initialization.
- [Delving Deep into Rectifiers (He/Kaiming init + PReLU)](https://arxiv.org/abs/1502.01852) — **He et al. (2015)** — initialization tuned for ReLU networks (the factor-of-2 fix).

**Books (free chapters)**:
- [Dive into Deep Learning — §5.4 "Numerical Stability and Initialization"](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) — **Zhang et al.** — vanishing/exploding signals and Xavier init, with code.
- [Deep Learning — §8.4 "Parameter Initialization Strategies"](https://www.deeplearningbook.org/contents/optimization.html) — **Goodfellow, Bengio & Courville** — the rigorous treatment of init and its effect on optimization.
- [Neural Networks and Deep Learning — Ch. 3 (weight initialization)](http://neuralnetworksanddeeplearning.com/chap3.html) — **Michael Nielsen** — why scaling init by fan-in speeds early learning.

**In this platform**:
- Concept page (full explanation): [Weight Initialization](05-Weight-Initialization.md)
- Concept depth (the *why*): [AI-ML-intuition 4.12 Weight Initialization (Xavier/He)](../../../AI-ML-intuition/Module_4_Stabilization/4C_Training_Stability/4.12_Weight_Initialization_Xavier_He.md)
- Prerequisite: [Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs.md)
- Related: [Vanishing / Exploding Gradients](06-Vanishing-Exploding-Gradients.md) (the failure mode init prevents) · [Activation Functions](03-Activation-Functions.md) · [Normalization](11-Normalization.md)
- Field overview: [Deep Learning](../README.md)
