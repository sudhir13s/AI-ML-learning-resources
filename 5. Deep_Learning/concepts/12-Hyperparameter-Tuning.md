---
id: "05-deep-learning/hyperparameter-tuning"
topic: "Hyperparameter Tuning"
parent: "05-deep-learning"
level: intermediate
prereqs: ["optimizers", "lr-schedules-warmup", "regularization"]
interview_frequency: medium
updated: 2026-06-19
---

# Hyperparameter Tuning
> Hyperparameters — learning rate, batch size, architecture width/depth, weight decay, dropout rate —
> are *not* learned by gradient descent; you search for them. Strategies range from **grid** and
> **random** search to **Bayesian optimization** and bandit methods like **Hyperband**, all guided by
> a held-out validation set. The learning rate is almost always the highest-leverage knob.

**Why it matters:** a practical-ML question that separates tinkerers from engineers — explain why
**random search beats grid search** in high dimensions, the role of the validation set (and why
tuning on test leaks), which hyperparameters matter most (LR first), how Bayesian optimization
models the objective to spend trials wisely, and the coarse-to-fine / log-scale search habit.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Tuning Process (C2W3L01)](https://www.youtube.com/watch?v=AXDByU3D1hA) (**Andrew Ng**). *Which knobs matter most and how to prioritize them.*
2. **See the search strategies** — watch ⭐ [Hyperparameter Tuning in Practice (C2W3L03)](https://www.youtube.com/watch?v=wKkcBPp3F1Y) (**Andrew Ng**). *Coarse-to-fine, log scales, and the panda-vs-caviar workflow.*
3. **Get the evidence** — read [Random Search for Hyper-Parameter Optimization](https://jmlr.org/papers/v13/bergstra12a.html) (**Bergstra & Bengio**). *The classic result: random beats grid when only a few dims matter.*
4. **Go Bayesian** — read [Exploring Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/) (**Distill**). *How a surrogate model picks the next trial.*
5. **Make it concrete** — run a random/Bayesian sweep (Optuna or Ray Tune) following [d2l Ch. 19](https://d2l.ai/chapter_hyperparameter-optimization/index.html). *Watching the search converge makes the trade-offs real.*

## 🎓 Courses (free)
- [Dive into Deep Learning — Hyperparameter Optimization](https://d2l.ai/chapter_hyperparameter-optimization/index.html) — **Zhang et al.** — random search, Bayesian opt, and Hyperband with runnable code.
- [Stanford CS231n — Babysitting the Learning Process](https://cs231n.github.io/neural-networks-3/) — **Stanford (Karpathy / Li / Johnson)** — hyperparameter search ranges and monitoring.

## 🎥 Videos
- [Tuning Process (C2W3L01)](https://www.youtube.com/watch?v=AXDByU3D1hA) — **DeepLearningAI (Andrew Ng)** — prioritizing which hyperparameters to tune first.
- [Hyperparameter Tuning in Practice (C2W3L03)](https://www.youtube.com/watch?v=wKkcBPp3F1Y) — **DeepLearningAI (Andrew Ng)** — coarse-to-fine search and log-scale sampling.
- [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — **DeepBean** — the optimizer hyperparameters you'll be tuning.
- [PyTorch LR Scheduler — Adjust the Learning Rate](https://www.youtube.com/watch?v=81NJgoR5RfY) — **Patrick Loeber** — tuning the single most important hyperparameter in code.

## 📄 Key Papers
- [Random Search for Hyper-Parameter Optimization](https://jmlr.org/papers/v13/bergstra12a.html) — **Bergstra & Bengio (2012)** — why random search dominates grid search in practice.
- [Practical Recommendations for Gradient-Based Training](https://arxiv.org/abs/1206.5533) — **Yoshua Bengio (2012)** — the canonical guide to setting LR, batch size, and regularizers.

## 📰 Articles / Blogs (free, no paywall)
- [CS231n — Hyperparameter optimization](https://cs231n.github.io/neural-networks-3/) — **Stanford CS231n** — search ranges, random vs grid, and validation discipline.
- [Exploring Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/) — **Distill** — interactive, intuitive treatment of surrogate-model search.
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — **Andrej Karpathy** — the disciplined tuning workflow that prevents wasted trials.

## 📚 Books (free, with chapters)
- [Deep Learning — **§11.4 "Selecting Hyperparameters"**](https://www.deeplearningbook.org/contents/guidelines.html) — **Goodfellow, Bengio & Courville** — manual vs automatic search and what each knob does.
- [Dive into Deep Learning — **Ch. 19 "Hyperparameter Optimization"**](https://d2l.ai/chapter_hyperparameter-optimization/index.html) — **Zhang et al.** — random search, Bayesian opt, and Hyperband with code.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.09 Learning Rate Schedules](../../../AI-ML-intuition/Module_2_Optimization/2.09_Learning_Rate_Schedules.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Prerequisites: [07 Optimizers](07-Optimizers.md) · [08 Learning-Rate Schedules & Warmup](08-Learning-Rate-Schedules-and-Warmup.md)
- Field overview: [Deep Learning](../README.md)
- Related domain: [11. Tools & Frameworks](../../11.%20Tools_and_Frameworks/README.md) (Optuna / Ray Tune in practice)
