---
id: "05-deep-learning/optimizers/references"
topic: "Optimizers — References"
parent: "05-deep-learning/optimizers"
type: references
updated: 2026-06-21
---

# Optimizers — references and further reading

> Companion link library for **[Optimizers](07-Optimizers.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path:**

1. **See gradient descent move** — watch [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) (**3Blue1Brown**). *The visual foundation every optimizer builds on.*
2. **Tour the whole family** — watch [Optimization for Deep Learning](https://www.youtube.com/watch?v=NE88eqLngkg) (**DeepBean**). *Momentum, RMSprop, AdaGrad, Adam in one coherent picture.*
3. **Feel momentum** — read [Why Momentum Really Works](https://distill.pub/2017/momentum/) (**Distill**). *An interactive view of how velocity accelerates and stabilizes descent.*
4. **Get every update rule** — read [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) (**Sebastian Ruder**). *All the rules side by side, derived.*
5. **Read the sources** — [Adam](https://arxiv.org/abs/1412.6980) → [AdamW](https://arxiv.org/abs/1711.05101). *The two papers behind today's default optimizer.*

**Videos:**

- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — **3Blue1Brown** — the visual intuition for descent that every optimizer extends.
- [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — **DeepBean** — the cleanest single overview of the whole optimizer family.
- [Gradient Descent With Momentum (C2W2L06)](https://www.youtube.com/watch?v=k8fTYJPd3_I) — **DeepLearning.AI (Andrew Ng)** — momentum as an exponentially-weighted average of gradients.
- [Adam Optimization Algorithm (C2W2L08)](https://www.youtube.com/watch?v=JXQT_vxqwIs) — **DeepLearning.AI (Andrew Ng)** — momentum + RMSprop combined, with bias correction.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — **StatQuest (Josh Starmer)** — the mechanics of a gradient step, worked by hand.
- [All Optimizers In One Video — SGD, Momentum, Adagrad, RMSprop, Adam](https://www.youtube.com/watch?v=TudQZtgpoHk) — **Krish Naik** — every update rule contrasted end to end.

**Courses (free):**

- [Dive into Deep Learning — Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html) — **Zhang et al.** — SGD through Adam with runnable code and convergence intuition.
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — **Andrej Karpathy** — builds the training loop (loss → backprop → optimizer step) from scratch.

**Articles / blogs (free, no paywall):**

- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) — **Sebastian Ruder** — the canonical survey: SGD, momentum, AdaGrad, RMSprop, Adam.
- [Why Momentum Really Works](https://distill.pub/2017/momentum/) — **Distill (Gabriel Goh)** — interactive geometry of momentum and conditioning.
- [CS231n — Parameter Updates](https://cs231n.github.io/neural-networks-3/) — **Stanford CS231n** — a practical comparison of update rules and when each helps.

**Key papers:**

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — **Kingma & Ba (2014)** — first + second moment estimates with bias correction; the default optimizer.
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — **Loshchilov & Hutter (2017)** — why weight decay must be decoupled from the adaptive step.
- [On the Convergence of Adam and Beyond (AMSGrad)](https://arxiv.org/abs/1904.09237) — **Reddi, Kale & Kumar (2018)** — a known failure case of Adam and a fix.
- [On the difficulty of training RNNs](https://arxiv.org/abs/1211.5063) — **Pascanu, Mikolov & Bengio (2012)** — exploding gradients and the gradient-clipping fix.
- [Accurate, Large Minibatch SGD (linear scaling rule)](https://arxiv.org/abs/1706.02677) — **Goyal et al. (2017)** — scaling the learning rate with batch size, plus warmup.
- [Adafactor: Adaptive Learning Rates with Sublinear Memory](https://arxiv.org/abs/1804.04235) — **Shazeer & Stern (2018)** — factored second moments for memory-efficient training.
- [Symbolic Discovery of Optimization Algorithms (Lion)](https://arxiv.org/abs/2302.06675) — **Chen et al. (2023)** — a one-state, sign-based optimizer found by search.

**Books (free chapters):**

- [Deep Learning — §8.3 "Basic Algorithms" + §8.5 "Adaptive Learning Rates"](https://www.deeplearningbook.org/contents/optimization.html) — **Goodfellow, Bengio & Courville** — the rigorous treatment of momentum and adaptive methods.

**In this platform:**

- Concept page (full explanation): [Optimizers](07-Optimizers.md)
- Concept depth (the *why*): [AI-ML-intuition 2.05 Gradient Descent & SGD](../../../AI-ML-intuition/Module_2_Optimization/2.05_Gradient_Descent_and_SGD.md) · [2.06 SGD with Momentum](../../../AI-ML-intuition/Module_2_Optimization/2.06_SGD_with_Momentum.md) · [2.07 Adam](../../../AI-ML-intuition/Module_2_Optimization/2.07_Adam_Optimizer.md) · [2.08 AdamW](../../../AI-ML-intuition/Module_2_Optimization/2.08_AdamW_Decoupled_Weight_Decay.md)
- Prerequisite: [02 Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs.md)
- Next concept: [08 Learning-Rate Schedules & Warmup](08-Learning-Rate-Schedules-and-Warmup.md)
- Why it matters for LLMs: [LoRA & PEFT](../../09.%20LLMs/concepts/12-LoRA-and-PEFT.md) (optimizer-state memory)
- Field overview: [Deep Learning](../README.md)
