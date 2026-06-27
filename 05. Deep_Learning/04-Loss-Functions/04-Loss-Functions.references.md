---
id: "05-deep-learning/loss-functions/references"
topic: "Loss Functions — References"
parent: "05-deep-learning/loss-functions"
type: references
updated: 2026-06-22
---

# Loss Functions — references and further reading

> Companion link library for **[Loss Functions](04-Loss-Functions.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity. Every link verified live.

**Start here — suggested path**:
1. **Build intuition** — watch [Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) (**StatQuest**). *Why we penalize confident-and-wrong predictions hard.*
2. **Connect to information theory** — watch [Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8) (**Aurélien Géron**). *Where the cross-entropy formula comes from, and the H(p)+KL decomposition.*
3. **Get the math** — read [Cross-entropy loss, explained](https://gombru.github.io/2018/05/23/cross_entropy_loss/) (**Raúl Gómez**). *Logits → softmax → cross-entropy, with the clean gradient.*
4. **See the MLE view** — read [d2l §4.1 (softmax + cross-entropy)](https://d2l.ai/chapter_linear-classification/index.html) (**Zhang et al.**). *Cross-entropy as negative log-likelihood, with runnable code.*
5. **Make it concrete** — implement MSE and cross-entropy by hand and check gradients. *Deriving $(\hat y - y)$ once makes the softmax+CE pairing permanent.*

**Videos**:
- [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8) — **Aurélien Géron** — the information-theory grounding: entropy, cross-entropy, KL, and how the loss falls out.
- [Intuitively Understanding the Cross Entropy Loss](https://www.youtube.com/watch?v=Pwgpl9mKars) — **Adian Liusie** — builds the loss up from likelihood, very clear.
- [Logistic Regression Cost Function](https://www.youtube.com/watch?v=SHEPb1JHw5o) — **DeepLearning.AI (Andrew Ng)** — binary cross-entropy derived as a Bernoulli likelihood.
- [Neural Networks Part 6: Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) — **StatQuest (Josh Starmer)** — the cleanest from-scratch intuition for cross-entropy.
- [Why we minimize the negative log-likelihood (MLE)](https://www.youtube.com/watch?v=tnBQzAU1lj0) — **ritvikmath** — the maximum-likelihood principle that generates every loss on this page.

**Interactive & visual**:
- [The Softmax function and its derivative](https://e2eml.school/softmax.html) — **Brandon Rohrer** — walks the cancellation that yields the clean $(\hat y - y)$ gradient, step by step.
- [Understanding Categorical Cross-Entropy Loss (with figures)](https://gombru.github.io/2018/05/23/cross_entropy_loss/) — **Raúl Gómez** — side-by-side diagrams of softmax/sigmoid + cross-entropy and the gradient.

**Courses (free)**:
- [Dive into Deep Learning — Linear & Softmax Regression](https://d2l.ai/chapter_linear-classification/index.html) — **Zhang et al.** — MSE and cross-entropy from first principles, with runnable code (§4.1 derives the softmax+CE gradient).
- [Stanford CS231n — Linear Classification (Softmax vs SVM loss)](https://cs231n.github.io/linear-classify/) — **Stanford (Karpathy / Li / Johnson)** — cross-entropy vs hinge, with worked gradients.
- [Stanford CS231n — Loss Functions and Optimization](https://cs231n.github.io/optimization-1/) — **Stanford** — how the loss surface drives the optimizer.

**Articles / blogs (free, no paywall)**:
- [Cross-Entropy, KL-Divergence, and Maximum Likelihood](https://leimao.github.io/blog/Cross-Entropy-KL-Divergence-MLE/) — **Lei Mao** — the full chain $\text{CE} = H(p) + \text{KL}$ and why minimizing CE = MLE, rigorously.
- [The exp-normalize / log-sum-exp trick](https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/) — **Tim Vieira** — exactly why you compute `log_softmax` in one stable step, never softmax-then-log.
- [Cross-Entropy Loss](https://www.pinecone.io/learn/cross-entropy-loss/) — **Pinecone** — concise, no-paywall walk-through with intuition and formulas.
- [`torch.nn.CrossEntropyLoss` reference](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — **PyTorch** — the fused logits→loss API, `weight`, `ignore_index`, and `reduction` documented.

**Key papers**:
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) — **Lin et al. (2017)** — the $(1-p_t)^\gamma$ class-imbalance reshaping of cross-entropy; a common interview follow-up.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — **Hinton, Vinyals & Dean (2015)** — soft targets at temperature $T$ and the KL-divergence distillation loss ([author PDF](https://www.cs.toronto.edu/~hinton/absps/distillation.pdf)).
- [Rethinking the Inception Architecture (label smoothing)](https://arxiv.org/abs/1512.00567) — **Szegedy et al. (2016)** — label smoothing to curb over-confidence (§7).
- [Deep Learning (Nature review)](https://www.nature.com/articles/nature14539) — **LeCun, Bengio & Hinton (2015)** — frames the loss/objective inside the training picture.

**Books (free chapters)**:
- [Deep Learning — Ch. 5.5 "Maximum Likelihood Estimation" + Ch. 6.2 (output units & loss)](https://www.deeplearningbook.org/contents/ml.html) — **Goodfellow, Bengio & Courville** — every loss derived as MLE, rigorously (the canonical treatment).
- [Pattern Recognition and Machine Learning](https://www.bishopbook.com/) — **Bishop** — §1.2.5 / §3.1.1 derive MSE as Gaussian MLE; §4.3.2 derives the softmax+CE gradient ([community code companion](https://github.com/gerdm/prml)).
- [Dive into Deep Learning — Ch. 4 "Linear Neural Networks for Classification"](https://d2l.ai/chapter_linear-classification/index.html) — **Zhang et al.** — the loss derived as negative log-likelihood, with code.
- [Neural Networks and Deep Learning — Ch. 3 (cross-entropy cost)](http://neuralnetworksanddeeplearning.com/chap3.html) — **Michael Nielsen** — why cross-entropy speeds learning vs the quadratic loss, with the saturation argument.

**Reference**:
- [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) — **Wikipedia** — the definition, the smooth-L1 connection, and the $\delta$ crossover, with the original Huber (1964) citation.

**In this platform**:
- Concept page (full explanation): [Loss Functions](04-Loss-Functions.md)
- Prerequisite: [Activation Functions](../03-Activation-Functions/03-Activation-Functions.md) (softmax — the partner that pairs with cross-entropy)
- The gradient in context: [Backpropagation & Computational Graphs](../02-Backpropagation-and-Computational-Graphs/02-Backpropagation-and-Computational-Graphs.md) (the $\hat y - y$ vector seeds the backward pass) · [Optimizers](../07-Optimizers/07-Optimizers.md) (what minimizes the loss surface)
- Related theory: [Cross-Entropy & KL Divergence](../../01.%20Foundations/concepts/23-Cross-Entropy-and-KL-Divergence.md) (the information-theory foundation) · [Regularization](../09-Regularization/09-Regularization.md) (penalized loss = MAP under a prior; label smoothing)
- Loss vs metric (evaluation side): [Classification Metrics](../../03.%20Supervised_Learning/concepts/14-Classification-Metrics.md) · [Regression Metrics](../../03.%20Supervised_Learning/concepts/15-Regression-Metrics.md)
- Concept depth (the *why*): [AI-ML-intuition 3.01 MSE / L2 Loss](../../../AI-ML-intuition/Module_3_Evaluation/3.01_Mean_Squared_Error_MSE_L2_Loss.md) · [3.03 Cross-Entropy / NLL](../../../AI-ML-intuition/Module_3_Evaluation/3.03_Categorical_Cross-Entropy_NLL.md) · [3.04 Maximum Likelihood](../../../AI-ML-intuition/Module_3_Evaluation/3.04_Maximum_Likelihood_Estimation.md)
- Field overview: [Deep Learning](../README.md)
