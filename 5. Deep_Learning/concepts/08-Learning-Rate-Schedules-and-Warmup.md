---
id: "05-deep-learning/lr-schedules-warmup"
topic: "Learning-Rate Schedules & Warmup"
parent: "05-deep-learning"
level: intermediate
prereqs: ["optimizers", "backpropagation"]
interview_frequency: high
updated: 2026-06-19
---

# Learning-Rate Schedules & Warmup
> The learning rate is the single most important hyperparameter — and the best value *changes* during
> training. **Schedules** (step, exponential, cosine) decay it over time so you take big steps early
> and fine ones late; **warmup** ramps it up from near-zero over the first steps so a freshly
> initialized model (especially a transformer with adaptive optimizers) doesn't diverge.

**Why it matters:** a frequent practical-training question — explain why a fixed LR is suboptimal,
compare step/exponential/**cosine** decay, justify **warmup** (large early Adam updates on
high-variance gradients destabilize training), and connect schedules to the transformer recipe
(linear warmup → cosine/inverse-sqrt decay) that everyone copies.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Learning Rate Decay (C2W2L09)](https://www.youtube.com/watch?v=QzulmoOg2JE) (**Andrew Ng**). *Why shrinking the step late helps you settle into a minimum.*
2. **See the options** — read ⭐ [Setting the learning rate of your neural network](https://www.jeremyjordan.me/nn-learning-rate/) (**Jeremy Jordan**). *LR finder, schedules, warm restarts — all visualized.*
3. **Get the math** — read [d2l: Learning Rate Scheduling](https://d2l.ai/chapter_optimization/lr-scheduler.html). *Polynomial/cosine schedules with code and plots.*
4. **Read the sources** — [SGDR: Cosine annealing with warm restarts](https://arxiv.org/abs/1608.03983) + [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186). *The schedules behind modern recipes.*
5. **Make it concrete** — wire a `CosineAnnealingLR` + warmup into a training loop following [PyTorch LR Scheduler](https://www.youtube.com/watch?v=81NJgoR5RfY). *Plotting the LR curve makes the schedule tangible.*

## 🎓 Courses (free)
- [Dive into Deep Learning — Learning Rate Scheduling](https://d2l.ai/chapter_optimization/lr-scheduler.html) — **Zhang et al.** — the canonical chapter on factor/cosine schedules and warmup, with code.
- [Stanford CS231n — Training Neural Networks](https://cs231n.github.io/neural-networks-3/) — **Stanford (Karpathy / Li / Johnson)** — LR as the key hyperparameter and how to anneal it.

## 🎥 Videos
- [Learning Rate Decay (C2W2L09)](https://www.youtube.com/watch?v=QzulmoOg2JE) — **DeepLearningAI (Andrew Ng)** — the clearest motivation for decaying the LR over time.
- [PyTorch LR Scheduler — Adjust the Learning Rate for Better Results](https://www.youtube.com/watch?v=81NJgoR5RfY) — **Patrick Loeber** — hands-on with PyTorch's built-in schedulers.
- [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — **DeepBean** — situates schedules alongside the optimizers they modulate.
- [All Optimizers In One Video](https://www.youtube.com/watch?v=TudQZtgpoHk) — **Krish Naik** — covers adaptive LR behavior and decay in context.

## 📄 Key Papers
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) — **Loshchilov & Hutter (2016)** — cosine annealing with restarts, now ubiquitous.
- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) — **Leslie Smith (2015)** — the LR-range test and triangular cyclical schedules.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — defines the linear-warmup → inverse-sqrt-decay schedule used for transformers.

## 📰 Articles / Blogs (free, no paywall)
- [Setting the learning rate of your neural network](https://www.jeremyjordan.me/nn-learning-rate/) — **Jeremy Jordan** — schedules, the LR finder, and warm restarts, clearly visualized.
- [CS231n — Annealing the learning rate](https://cs231n.github.io/neural-networks-3/) — **Stanford CS231n** — step/exponential/1-over-t decay and how to pick.
- [How to initialize neural networks & why warmup helps](https://www.deeplearning.ai/ai-notes/initialization/index.html) — **DeepLearning.AI** — interactive view of why early steps need care (motivating warmup).

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **§12.11 "Learning Rate Scheduling"**](https://d2l.ai/chapter_optimization/lr-scheduler.html) — **Zhang et al.** — factor, multi-factor, and cosine schedules with warmup, with code.
- [Deep Learning — **§8.3 "Basic Algorithms"** (learning-rate selection)](https://www.deeplearningbook.org/contents/optimization.html) — **Goodfellow, Bengio & Courville** — why the LR dominates and how decay schedules are chosen.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.09 Learning Rate Schedules](../../../AI-ML-intuition/Module_2_Optimization/2.09_Learning_Rate_Schedules.md)
- Prerequisite: [07 Optimizers](07-Optimizers.md)
- Related concept: [12 Hyperparameter Tuning](12-Hyperparameter-Tuning.md) (the LR is the top hyperparameter to tune)
- Field overview: [Deep Learning](../README.md)
