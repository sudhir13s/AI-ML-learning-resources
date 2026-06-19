---
id: "05-deep-learning/optimizers"
topic: "Optimizers (SGD · Momentum · Adam · AdamW · RMSprop)"
parent: "05-deep-learning"
level: intermediate
prereqs: ["backpropagation", "calculus"]
interview_frequency: very-high
updated: 2026-06-19
---

# Optimizers (SGD · Momentum · Adam · AdamW · RMSprop)
> The update rule that turns gradients into weight changes. **SGD** steps downhill; **Momentum** adds
> velocity to push through ravines; **RMSprop** scales each step by a running gradient magnitude;
> **Adam** combines momentum + per-parameter scaling; **AdamW** fixes Adam's weight-decay coupling.
> The optimizer largely determines how fast — and whether — a network converges.

**Why it matters:** the single most-asked training question — write the SGD/momentum/Adam update
rules, explain what the first and second moments (`m`, `v`) do in Adam and why we bias-correct them,
articulate why **AdamW** decouples weight decay from the adaptive step, and reason about the
SGD-vs-Adam generalization trade-off interviewers love to probe.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) (**DeepBean**). *One coherent picture of how the major optimizers differ.*
2. **See momentum visually** — read ⭐ [Why Momentum Really Works](https://distill.pub/2017/momentum/) (**Distill**). *An interactive view of how velocity accelerates and stabilizes descent.*
3. **Get the math** — read [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) (**Sebastian Ruder**). *Every update rule side by side, with derivations.*
4. **Read the sources** — [Adam](https://arxiv.org/abs/1412.6980) → [AdamW (decoupled weight decay)](https://arxiv.org/abs/1711.05101). *The two papers behind today's default optimizer.*
5. **Make it concrete** — implement SGD→Momentum→Adam from scratch following [d2l Ch. 12](https://d2l.ai/chapter_optimization/index.html). *Coding the moment updates makes Adam stop being a black box.*

## 🎓 Courses (free)
- [Stanford CS231n — Training Neural Networks II (Optimization)](https://www.youtube.com/watch?v=_JB0AO7QxSA) — **Stanford (Johnson / Li)** — the lecture deriving SGD, momentum, and adaptive methods.
- [Dive into Deep Learning — Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html) — **Zhang et al.** — SGD through Adam with runnable code and convergence intuition.

## 🎥 Videos
- [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — **DeepBean** — the cleanest single overview of the whole optimizer family.
- [Gradient Descent With Momentum (C2W2L06)](https://www.youtube.com/watch?v=k8fTYJPd3_I) — **DeepLearningAI (Andrew Ng)** — momentum as an exponentially-weighted average of gradients.
- [Adam Optimization Algorithm (C2W2L08)](https://www.youtube.com/watch?v=JXQT_vxqwIs) — **DeepLearningAI (Andrew Ng)** — momentum + RMSprop combined, with bias correction.
- [All Optimizers In One Video — SGD, Momentum, Adagrad, RMSprop, Adam](https://www.youtube.com/watch?v=TudQZtgpoHk) — **Krish Naik** — every update rule contrasted end to end.

## 📄 Key Papers
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — **Kingma & Ba (2014)** — first + second moment estimates with bias correction; the default optimizer.
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — **Loshchilov & Hutter (2017)** — why weight decay must be decoupled from the adaptive step.
- [On the Convergence of Adam and Beyond (AMSGrad)](https://arxiv.org/abs/1904.09237) — **Reddi, Kale & Kumar (2018)** — a known failure case of Adam and a fix.

## 📰 Articles / Blogs (free, no paywall)
- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) — **Sebastian Ruder** — the canonical survey: SGD, momentum, AdaGrad, RMSprop, Adam.
- [Why Momentum Really Works](https://distill.pub/2017/momentum/) — **Distill (Gabriel Goh)** — interactive geometry of momentum and conditioning.
- [CS231n — Parameter Updates](https://cs231n.github.io/neural-networks-3/) — **Stanford CS231n** — practical comparison of update rules and when each helps.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 12 "Optimization Algorithms"**](https://d2l.ai/chapter_optimization/index.html) — **Zhang et al.** — SGD, momentum, AdaGrad, RMSprop, Adam, AdamW with code.
- [Deep Learning — **§8.3 "Basic Algorithms"** + **§8.5 "Algorithms with Adaptive Learning Rates"**](https://www.deeplearningbook.org/contents/optimization.html) — **Goodfellow, Bengio & Courville** — the rigorous treatment of momentum and adaptive methods.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.05 Gradient Descent & SGD](../../../AI-ML-intuition/Module_2_Optimization/2.05_Gradient_Descent_and_SGD.md) · [2.06 SGD with Momentum](../../../AI-ML-intuition/Module_2_Optimization/2.06_SGD_with_Momentum.md) · [2.07 Adam](../../../AI-ML-intuition/Module_2_Optimization/2.07_Adam_Optimizer.md) · [2.08 AdamW](../../../AI-ML-intuition/Module_2_Optimization/2.08_AdamW_Decoupled_Weight_Decay.md)
- Prerequisite: [02 Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs.md)
- Next concept: [08 Learning-Rate Schedules & Warmup](08-Learning-Rate-Schedules-and-Warmup.md)
- Field overview: [Deep Learning](../README.md)
