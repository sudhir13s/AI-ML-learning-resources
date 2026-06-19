---
id: "05-deep-learning/weight-initialization"
topic: "Weight Initialization (Xavier/Glorot · He)"
parent: "05-deep-learning"
level: intermediate
prereqs: ["feedforward-networks", "backpropagation", "linear-algebra"]
interview_frequency: medium
updated: 2026-06-19
---

# Weight Initialization (Xavier/Glorot · He)
> How you set the starting weights decides whether a deep network trains at all. Initialize too small
> and signals (and gradients) shrink to zero through the layers; too large and they explode.
> **Xavier/Glorot** and **He** initialization scale the random weights by the layer's fan-in/fan-out
> so the variance of activations and gradients stays roughly constant across depth.

**Why it matters:** a classic "why won't my deep net train?" question — explain why you can't
initialize all weights to zero (symmetry), how exploding/vanishing signals arise from variance
compounding through layers, derive the variance-preserving rule, and know that **Xavier** targets
tanh/sigmoid while **He** is tuned for ReLU (the extra factor of 2).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Weight Initialization (C2W1L11)](https://www.youtube.com/watch?v=s2coXdufOzE) (**Andrew Ng**). *Why variance scaling keeps signals alive through depth.*
2. **See it interactively** — read ⭐ [Initializing neural networks](https://www.deeplearning.ai/ai-notes/initialization/index.html) (**DeepLearning.AI**). *Sliders show too-small/too-large init collapsing or exploding activations live.*
3. **Get the math** — read [How to initialize deep neural networks: Xavier and Kaiming](https://pouannes.github.io/blog/initialization/) (**Pierre Ouannes**). *Full variance derivation for both schemes.*
4. **Read the sources** — [Glorot & Bengio (Xavier)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) → [He et al. (Kaiming, for ReLU)](https://arxiv.org/abs/1502.01852). *The two papers everyone cites.*
5. **Make it concrete** — work through [d2l: Numerical Stability and Initialization](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html). *Implementing the schemes shows the variance argument in code.*

## 🎓 Courses (free)
- [Stanford CS231n — Neural Networks Part 2 (Initialization)](https://cs231n.github.io/neural-networks-2/) — **Stanford (Karpathy / Li / Johnson)** — the canonical notes on calibrating initial variances.
- [Dive into Deep Learning — Numerical Stability & Initialization](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) — **Zhang et al.** — vanishing/exploding signals and the init fixes, with code.

## 🎥 Videos
- [Weight Initialization in a Deep Network (C2W1L11)](https://www.youtube.com/watch?v=s2coXdufOzE) — **DeepLearningAI (Andrew Ng)** — the clearest short derivation of variance-preserving init.
- [Weight Initialization explained — reducing the vanishing gradient problem](https://www.youtube.com/watch?v=8krd5qKVw-Q) — **deeplizard** — why zero/large init fails and how Xavier/He fix it.
- [Xavier & He Initialization](https://www.youtube.com/watch?v=LKWatKGRZLI) — **Six Sigma Pro SMART** — side-by-side walk-through of both schemes.
- [Why don't we initialize the weights of a neural network to zero?](https://www.youtube.com/watch?v=LBMVyXfZQy0) — **Bhavesh Bhatt** — the symmetry-breaking argument, concretely.
- [Weight Initialization for Deep Feedforward Neural Networks](https://www.youtube.com/watch?v=tYFO434Lpm0) — **AssemblyAI** — the practical fan-in/fan-out rules.

## 📄 Key Papers
- [Understanding the difficulty of training deep feedforward networks (Xavier/Glorot)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — **Glorot & Bengio (2010)** — derives the variance-preserving initialization.
- [Delving Deep into Rectifiers (He/Kaiming init + PReLU)](https://arxiv.org/abs/1502.01852) — **He et al. (2015)** — initialization tuned for ReLU networks (the factor-of-2 fix).

## 📰 Articles / Blogs (free, no paywall)
- [Initializing neural networks](https://www.deeplearning.ai/ai-notes/initialization/index.html) — **DeepLearning.AI** — interactive page showing the effect of init scale on signal flow.
- [How to initialize deep neural networks: Xavier and Kaiming](https://pouannes.github.io/blog/initialization/) — **Pierre Ouannes** — the full variance derivation for both schemes.
- [CS231n — Weight Initialization](https://cs231n.github.io/neural-networks-2/) — **Stanford CS231n** — calibrated initialization and the pitfalls of naive choices.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **§5.4 "Numerical Stability and Initialization"**](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) — **Zhang et al.** — vanishing/exploding signals and Xavier init, with code.
- [Deep Learning — **§8.4 "Parameter Initialization Strategies"**](https://www.deeplearningbook.org/contents/optimization.html) — **Goodfellow, Bengio & Courville** — the rigorous treatment of init and its effect on optimization.
- [Neural Networks and Deep Learning — **Ch. 3 (weight initialization)**](http://neuralnetworksanddeeplearning.com/chap3.html) — **Michael Nielsen** — why scaling init by fan-in speeds early learning.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.12 Weight Initialization (Xavier/He)](../../../AI-ML-intuition/Module_4_Stabilization/4C_Training_Stability/4.12_Weight_Initialization_Xavier_He.md)
- Prerequisite: [02 Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs.md)
- Related concept: [06 Vanishing / Exploding Gradients](06-Vanishing-Exploding-Gradients.md) (the failure mode init prevents)
- Field overview: [Deep Learning](../README.md)
