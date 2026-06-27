---
id: "05-deep-learning/lr-schedules-warmup/references"
topic: "Learning-Rate Schedules & Warmup — References"
parent: "05-deep-learning/lr-schedules-warmup"
type: references
updated: 2026-06-22
---

# Learning-Rate Schedules & Warmup — references and further reading

> Companion link library for **[Learning-Rate Schedules & Warmup](08-Learning-Rate-Schedules-and-Warmup.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer, chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build the intuition** — watch [Learning Rate Decay (C2W2L09)](https://www.youtube.com/watch?v=QzulmoOg2JE) (**Andrew Ng**). *Why shrinking the step late helps you settle into a minimum — the clearest 5-minute motivation.*
2. **See every option visualized** — read [Setting the learning rate of your neural network](https://www.jeremyjordan.me/nn-learning-rate/) (**Jeremy Jordan**). *LR finder, decay schedules, warm restarts — all plotted.*
3. **Get the math** — read [d2l §12.11: Learning Rate Scheduling](https://d2l.ai/chapter_optimization/lr-scheduler.html) (**Zhang et al.**). *Factor/polynomial/cosine schedules and warmup, with runnable code.*
4. **Read the sources** — skim [SGDR (cosine annealing)](https://arxiv.org/abs/1608.03983) and the [transformer LR schedule](https://arxiv.org/abs/1706.03762) (§5.3). *The two schedules behind modern recipes.*
5. **Wire it up** — follow [PyTorch LR Scheduler](https://www.youtube.com/watch?v=81NJgoR5RfY) (**Patrick Loeber**) and plot the LR curve. *Makes the schedule tangible in code.*

**Videos**:
- [Learning Rate Decay (C2W2L09)](https://www.youtube.com/watch?v=QzulmoOg2JE) — **DeepLearning.AI (Andrew Ng)** — the clearest short motivation for decaying the LR over time.
- [PyTorch LR Scheduler — Adjust the Learning Rate for Better Results](https://www.youtube.com/watch?v=81NJgoR5RfY) — **Patrick Loeber** — hands-on with PyTorch's built-in schedulers, end to end.
- [The Learning Rate Finder (fast.ai)](https://www.youtube.com/watch?v=k1GIEkzQ8qc) — **Jeremy Howard (fast.ai)** — the LR-range test and one-cycle, from the person who popularized them.
- [Optimization for Deep Learning (Momentum, RMSProp, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — **DeepBean** — situates schedules alongside the adaptive optimizers they modulate.
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — **Andrej Karpathy** — implements warmup + cosine decay in a real GPT training loop (the recipe in practice).

**Courses (free)**:
- [Dive into Deep Learning — §12.11 Learning Rate Scheduling](https://d2l.ai/chapter_optimization/lr-scheduler.html) — **Zhang, Lipton, Li & Smola** — the canonical chapter on factor/polynomial/cosine schedules and warmup, with code.
- [Stanford CS231n — Training Neural Networks (annealing the LR)](https://cs231n.github.io/neural-networks-3/) — **Stanford (Karpathy / Li / Johnson)** — LR as the key hyperparameter and how to anneal it.

**Articles / blogs (free, no paywall)**:
- [Setting the learning rate of your neural network](https://www.jeremyjordan.me/nn-learning-rate/) — **Jeremy Jordan** — schedules, the LR finder, and warm restarts, clearly visualized.
- [How Do You Find a Good Learning Rate](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html) — **Sylvain Gugger (fast.ai / Hugging Face)** — the LR-range test explained and implemented, step by step.
- [The 1cycle policy](https://sgugger.github.io/the-1cycle-policy.html) — **Sylvain Gugger** — one-cycle and super-convergence, with the momentum-anti-correlation rationale.
- [PyTorch — How to adjust the learning rate (`torch.optim.lr_scheduler`)](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — **PyTorch team** — the authoritative reference for every built-in scheduler (`StepLR`, `CosineAnnealingLR`, `OneCycleLR`, `SequentialLR`, …).
- [Transformers — Optimizer schedules (`get_cosine_schedule_with_warmup`)](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules) — **Hugging Face** — the one-call warmup→decay schedules used to train LLMs in practice.
- [How to pick the best learning rate (and schedule)](https://www.deeplearning.ai/ai-notes/initialization/index.html) — **DeepLearning.AI** — interactive view of why early steps need care (motivating warmup).

**Key papers**:
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) — **Loshchilov & Hutter (2017)** — introduces **cosine annealing** and warm restarts; now ubiquitous.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)**, §5.3 — defines the **linear-warmup → inverse-sqrt (Noam)** schedule for transformers.
- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) — **Leslie Smith (2015)** — the **LR-range test** and triangular cyclical schedules.
- [Super-Convergence: Very Fast Training Using Large Learning Rates](https://arxiv.org/abs/1708.07120) — **Smith & Topin (2018)** — the **one-cycle** policy and super-convergence.
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) — **Goyal et al. (2017)** — the **linear scaling rule** + gradual warmup for large batches.
- [On the Variance of the Adaptive Learning Rate and Beyond (RAdam)](https://arxiv.org/abs/1908.03265) — **Liu et al. (2020)** — *why* warmup is needed: Adam's second-moment estimate has pathologically high variance early.
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — **Loshchilov & Hutter (2019)** — why weight decay is scaled by the (scheduled) learning rate, and the coupling that implies.

**Books (free chapters)**:
- [Dive into Deep Learning — §12.11 "Learning Rate Scheduling"](https://d2l.ai/chapter_optimization/lr-scheduler.html) — **Zhang et al.** — factor, multi-factor, and cosine schedules with warmup, with code.
- [Deep Learning — §8.3 "Basic Algorithms" (learning-rate selection)](https://www.deeplearningbook.org/contents/optimization.html) — **Goodfellow, Bengio & Courville** — why the LR dominates and how decay schedules are chosen.

**In this platform**:
- Concept page (full explanation): [Learning-Rate Schedules & Warmup](08-Learning-Rate-Schedules-and-Warmup.md)
- Prerequisite — the update rules schedules modulate: [Optimizers (SGD · Momentum · Adam · AdamW)](../07-Optimizers/07-Optimizers.md)
- Prerequisite — where the gradient comes from: [Backpropagation & Computational Graphs](../02-Backpropagation-and-Computational-Graphs/02-Backpropagation-and-Computational-Graphs.md)
- Why warmup matters for transformers: [Transformer Architecture](../16-Transformer-Architecture/16-Transformer-Architecture.md)
- Builds on this: [Hyperparameter Tuning](../12-Hyperparameter-Tuning/12-Hyperparameter-Tuning.md) (the LR is the top hyperparameter to tune) · [Normalization](../11-Normalization/11-Normalization.md) (interacts with early-training stability)
- The *why* (concept depth): [AI-ML-intuition 2.09 Learning Rate Schedules](../../../AI-ML-intuition/Module_2_Optimization/2.09_Learning_Rate_Schedules.md)
- Field overview: [Deep Learning](../README.md)
