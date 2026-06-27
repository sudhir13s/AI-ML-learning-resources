---
id: "05-deep-learning/regularization/references"
topic: "Regularization — References"
parent: "05-deep-learning/regularization"
type: references
updated: 2026-06-22
---

# Regularization — references and further reading

> Companion link library for **[Regularization](09-Regularization.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) (**StatQuest**). *Why penalizing weights reduces variance.*
2. **See it in nets** — watch [Why Regularization Reduces Overfitting](https://www.youtube.com/watch?v=NyG-7nRpsW8) (**Andrew Ng**), then toggle the regularization dropdown in [TensorFlow Playground](https://playground.tensorflow.org/). *Watch a jagged boundary turn smooth.*
3. **Get the math** — read [d2l: Weight Decay](https://d2l.ai/chapter_linear-regression/weight-decay.html) and [Deep Learning, Ch. 7](https://www.deeplearningbook.org/contents/regularization.html). *L2 derived as a gradient shrink + the MAP-prior and early-stopping≈L2 arguments.*
4. **Understand why it's needed** — skim [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530) (**Zhang et al.**). *Over-parameterized nets can fit random labels — capacity doesn't prevent memorization.*
5. **Make it concrete** — add `weight_decay`, `label_smoothing`, augmentation, and early stopping to a training loop and watch the train/val gap shrink.

**Videos**:
- [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) — **StatQuest (Josh Starmer)** — the clearest intuition for why an L2 penalty helps.
- [Why Regularization Reduces Overfitting (C2W1L05)](https://www.youtube.com/watch?v=NyG-7nRpsW8) — **DeepLearningAI (Andrew Ng)** — how the penalty drives the net toward simpler functions.
- [Regularization (C2W1L04)](https://www.youtube.com/watch?v=6g0t3Phly2M) — **DeepLearningAI (Andrew Ng)** — L2 and L1 penalties added to a neural-net cost.
- [Dropout (C2W1L06)](https://www.youtube.com/watch?v=ARq74QuavAo) — **DeepLearningAI (Andrew Ng)** — the stochastic regularizer and its ensemble/co-adaptation reading.
- [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — **StatQuest (Josh Starmer)** — the trade-off that regularization manages.

**Interactive & visual**:
- [TensorFlow Playground](https://playground.tensorflow.org/) — **Google** — switch the Regularization dropdown (None / L1 / L2) and watch overfitting cured live as the decision boundary smooths.
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — **Andrej Karpathy** — the practical "overfit first, then regularize" order, with the reasoning.

**Courses (free)**:
- [Stanford CS231n — Neural Networks Part 2 (Regularization)](https://cs231n.github.io/neural-networks-2/) — **Stanford (Karpathy / Li / Johnson)** — L1/L2/max-norm/dropout and their effects on the loss surface.
- [Dive into Deep Learning — Weight Decay & Generalization](https://d2l.ai/chapter_linear-regression/weight-decay.html) — **Zhang et al.** — regularization derived from first principles, with runnable code.

**Articles / blogs (free, no paywall)**:
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — **Andrej Karpathy** — practical regularization order (overfit first, then weight decay, dropout, augmentation, early stopping).
- [AdamW and Super-convergence / decoupled weight decay, explained](https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html) — **fast.ai (Sylvain Gugger & Jeremy Howard)** — the L2-vs-decoupled-decay distinction in plain language.
- [Setting the learning rate / training tricks](https://www.jeremyjordan.me/nn-learning-rate/) — **Jeremy Jordan** — early stopping and other implicit regularizers in context.

**Key papers**:
- [A Simple Weight Decay Can Improve Generalization](https://proceedings.neurips.cc/paper/1991/hash/8eefcfdf5990e441f0fb6f3fad709e21-Abstract.html) — **Krogh & Hertz (1992)** — the classic analysis of weight decay's effect.
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — **Loshchilov & Hutter (2019, ICLR)** — why L2 penalty ≠ weight decay under adaptive optimizers, and the fix.
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) — **Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov (2014, JMLR)** — the dropout paper and its ensemble interpretation.
- [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) — **Szegedy et al. (2016)** — §7 introduces label smoothing.
- [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) — **Müller, Kornblith & Hinton (2019)** — calibration benefit and the distillation caveat.
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) — **Zhang et al. (2018, ICLR)** — convex combinations of inputs+labels; the vicinal-risk framing.
- [CutMix: Regularization Strategy to Train Strong Classifiers](https://arxiv.org/abs/1905.04899) — **Yun et al. (2019)** — patch-mixing variant of mixup.
- [RandAugment: Practical automated data augmentation](https://arxiv.org/abs/1909.13719) — **Cubuk et al. (2020)** — augmentation search reduced to two scalars (N, M).
- [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382) — **Huang et al. (2016)** — "dropout for whole layers" in deep residual nets.
- [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530) — **Zhang et al. (2017, ICLR)** — over-parameterized nets fit random labels; the motivation for the whole page.
- [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836) — **Keskar et al. (2017, ICLR)** — large batch → sharp minima → worse generalization.
- [Averaging Weights Leads to Wider Optima and Better Generalization (SWA)](https://arxiv.org/abs/1803.05407) — **Izmailov et al. (2018)** — average late SGD iterates toward the flat-basin center.
- [Reconciling modern machine-learning practice and the bias–variance trade-off (double descent)](https://arxiv.org/abs/1812.11118) — **Belkin et al. (2019, PNAS)** — the interpolation peak and second descent.
- [Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/abs/1912.02292) — **Nakkiran et al. (2019)** — model-, epoch-, and sample-wise double descent in deep nets.
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) — **Power et al. (2022)** — weight decay drives a memorizing network to a generalizing one long after training accuracy saturates.

**Books (free chapters)**:
- [Deep Learning — Ch. 7 "Regularization for Deep Learning"](https://www.deeplearningbook.org/contents/regularization.html) — **Goodfellow, Bengio & Courville** — the definitive chapter: L1/L2, the MAP-prior view, early-stopping≈L2, dropout, augmentation, noise injection.
- [The Elements of Statistical Learning — §3.4 (Ridge & Lasso)](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the original diamond-vs-circle geometry; free PDF.
- [Dive into Deep Learning — §3.7 "Weight Decay" + §5.6 "Generalization"](https://d2l.ai/chapter_linear-regression/weight-decay.html) — **Zhang et al.** — regularization with runnable experiments.
- [Neural Networks and Deep Learning — Ch. 3 (overfitting & regularization)](http://neuralnetworksanddeeplearning.com/chap3.html) — **Michael Nielsen** — L2, early stopping, and why they help, from scratch.

**In this platform**:
- Concept page (full explanation): [Regularization](09-Regularization.md)
- Concept depth (the *why*): [AI-ML-intuition 2.10 Regularization (L1/L2)](../../../AI-ML-intuition/Module_2_Optimization/2.10_Regularization_L1_L2.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Prerequisite: [Loss Functions](../04-Loss-Functions/04-Loss-Functions.md)
- Related (deep-learning): [Dropout](../10-Dropout/10-Dropout.md) (the stochastic regularizer in full) · [Optimizers](../07-Optimizers/07-Optimizers.md) (AdamW and decoupled weight decay) · [Normalization](../11-Normalization/11-Normalization.md) (BatchNorm's regularizing side-effect) · [Residual / Skip Connections](../18-Residual-Skip-Connections/18-Residual-Skip-Connections.md) (the identity path stochastic depth rides on)
- Related (linear models): [Regularization for Linear Models — Ridge · Lasso · Elastic-Net](../../03.%20Supervised_Learning/concepts/03-Regularization-Linear-Models.md) (the diamond-vs-circle geometry + closed forms) · [Bias–Variance Tradeoff](../../03.%20Supervised_Learning/concepts/12-Bias-Variance-Tradeoff.md) (double descent)
- Field overview: [Deep Learning](../README.md)
