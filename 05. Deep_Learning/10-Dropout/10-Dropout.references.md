---
id: "05-deep-learning/dropout/references"
topic: "Dropout — References"
parent: "05-deep-learning/dropout"
type: references
updated: 2026-06-22
---

# Dropout — references and further reading

> Companion link library for **[Dropout](10-Dropout.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity. Every link verified (HTTP 200).

**Start here — suggested path**:
1. **Build intuition** — watch [Dropout Regularization (C2W1L06)](https://www.youtube.com/watch?v=D8PJAL-MZv8) (**Andrew Ng**). *Why randomly killing units spreads out the learned weights.*
2. **See why it works** — watch [Understanding Dropout (C2W1L07)](https://www.youtube.com/watch?v=ARq74QuavAo) (**Andrew Ng**). *The ensemble / no-co-adaptation argument, intuitively.*
3. **Get the math and run it** — read & run [d2l: Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html). *Inverted dropout, expectations, and from-scratch code.*
4. **Read the source** — [Dropout: A Simple Way to Prevent Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) (**Srivastava et al. 2014**). *The canonical paper, with the model-averaging derivation.*
5. **Go deeper on the theory** — [Dropout Training as Adaptive Regularization](https://arxiv.org/abs/1306.0543) (**Wager et al. 2013**). *Dropout as an adaptive $L_2$ penalty — the connection to weight decay.*
6. **Make it concrete** — toggle `nn.Dropout(p)` on a small over-fitting net and watch the train/val gap close (and sweep `p` to find the sweet spot).

**Videos**:
- [Dropout Regularization (C2W1L06)](https://www.youtube.com/watch?v=D8PJAL-MZv8) — **DeepLearningAI (Andrew Ng)** — the mechanics: random unit removal each pass.
- [Understanding Dropout (C2W1L07)](https://www.youtube.com/watch?v=ARq74QuavAo) — **DeepLearningAI (Andrew Ng)** — why it prevents co-adaptation and acts like an ensemble.
- [Regularization (C2W1L04)](https://www.youtube.com/watch?v=6g0t3Phly2M) — **DeepLearningAI (Andrew Ng)** — situates dropout alongside L1/L2.
- [Training a Neural Network explained](https://www.youtube.com/watch?v=sZAlS3_dnk0) — **deeplizard** — where dropout fits in the train/validation/test workflow.

**Interactive & visual**:
- [Dive into Deep Learning — Dropout (runnable)](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html) — **Zhang et al.** — implement inverted dropout from scratch and run it in-browser (Colab/notebook); watch the train/val gap respond to `p`.

**Courses (free)**:
- [Stanford CS231n — Neural Networks Part 2 (Dropout)](https://cs231n.github.io/neural-networks-2/) — **Stanford (Karpathy / Li / Johnson)** — dropout as regularization with the inverted-dropout implementation.
- [Dive into Deep Learning — Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html) — **Zhang et al.** — the method derived and implemented from scratch.

**Articles / blogs (free, no paywall)**:
- [CS231n — Dropout](https://cs231n.github.io/neural-networks-2/) — **Stanford CS231n** — inverted dropout and why test time stays clean.
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — **Andrej Karpathy** — where dropout fits in a real regularization strategy.
- [Dropout Explained](https://leimao.github.io/blog/Dropout-Explained/) — **Lei Mao** — a careful, derivation-first walkthrough of the inverted-dropout scaling and expectation.

**Key papers**:
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) — **Srivastava et al. (2014)** — the canonical paper: the $2^n$ ensemble, weight-scaling inference, the model-averaging interpretation.
- [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) — **Hinton et al. (2012)** — the original dropout proposal and the co-adaptation argument.
- [Dropout Training as Adaptive Regularization](https://arxiv.org/abs/1306.0543) — **Wager, Wang & Liang (2013)** — derives dropout ≈ an adaptive, variance-scaled $L_2$ penalty (the bridge to weight decay).
- [Regularization of Neural Networks using DropConnect](https://proceedings.mlr.press/v28/wan13.html) — **Wan et al. (2013)** — the generalization that drops *weights* instead of units ([PDF](http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf)).
- [Efficient Object Localization Using Convolutional Networks (SpatialDropout)](https://arxiv.org/abs/1411.4280) — **Tompson et al. (2015)** — why conv layers need channel-wise (spatial) dropout, not per-pixel.
- [DropBlock: A regularization method for convolutional networks](https://arxiv.org/abs/1810.12890) — **Ghiasi et al. (2018)** — drop contiguous regions of feature maps for deep CNNs.
- [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382) — **Huang et al. (2016)** — dropout lifted to whole residual blocks (DropPath); trains 1000+-layer ResNets.
- [A Theoretically Grounded Application of Dropout in RNNs (variational dropout)](https://arxiv.org/abs/1512.05287) — **Gal & Ghahramani (2016)** — one dropout mask per sequence; why per-step dropout breaks recurrence.
- [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142) — **Gal & Ghahramani (2016)** — dropout at test time as approximate Bayesian inference (MC-dropout for uncertainty).
- [Understanding the Disharmony between Dropout and Batch Normalization (variance shift)](https://arxiv.org/abs/1801.05134) — **Li et al. (2019)** — why stacking dropout and BatchNorm hurts, and the ordering fix.

**Books (free chapters)**:
- [Dive into Deep Learning — §5.6 "Dropout"](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html) — **Zhang et al.** — inverted dropout, expectations, and code.
- [Deep Learning — §7.12 "Dropout"](https://www.deeplearningbook.org/contents/regularization.html) — **Goodfellow, Bengio & Courville** — the rigorous ensemble/bagging and geometric-mean-averaging view.

**In this platform**:
- Concept page (full explanation): [Dropout](10-Dropout.md)
- Concept depth (the *why*): [AI-ML-intuition 2.11 Dropout](../../../AI-ML-intuition/Module_2_Optimization/2.11_Dropout.md)
- Prerequisite: [Regularization](../09-Regularization/09-Regularization.md) (L1/L2, early stopping — the family dropout belongs to)
- Related: [Normalization](../11-Normalization/11-Normalization.md) (BatchNorm — the variance-shift conflict; LayerNorm pairs with dropout) · [Residual / Skip Connections](../18-Residual-Skip-Connections/18-Residual-Skip-Connections.md) (stochastic depth / DropPath, dropout at the block level) · [Transformer Architecture](../16-Transformer-Architecture/16-Transformer-Architecture.md) (where attention / residual / FFN dropout live)
- Field overview: [Deep Learning](../README.md)
