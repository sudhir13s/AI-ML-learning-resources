---
id: "03-supervised-learning/regularization-linear-models"
topic: "Regularization for Linear Models (Ridge · Lasso · Elastic-Net)"
parent: "03-supervised-learning"
level: intermediate
prereqs: ["linear-regression", "logistic-regression", "bias-variance", "gradient-descent"]
interview_frequency: very-high
updated: 2026-06-19
---

# Regularization for Linear Models — Ridge · Lasso · Elastic-Net
> Add a penalty on the size of the coefficients to the loss: **Ridge (L2)** shrinks weights smoothly
> toward zero, **Lasso (L1)** drives some weights *exactly* to zero (feature selection), and
> **Elastic-Net** blends both. The standard cure for overfitting and multicollinearity in linear models.

**Why it matters:** a top-tier interview topic because it forces you to connect bias–variance,
optimization geometry, and feature selection. Expect: *why* L1 produces sparsity but L2 doesn't (the
diamond vs circle constraint picture), how the penalty trades a little bias for a large drop in
variance, why you must standardize features first, and how `λ` (or `alpha`) is chosen by
cross-validation.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30). *See how shrinking the slope reduces variance and prevents overfitting on small data.*
2. **See the contrast** — watch [StatQuest: Lasso (L1) Regression](https://www.youtube.com/watch?v=NGf0voTMlcs) then [Ridge vs Lasso, Visualized](https://www.youtube.com/watch?v=Xm2C_gTAl8c). *Why L1 zeroes out coefficients (the diamond constraint) and L2 only shrinks them.*
3. **Get the math** — read [ISLR Ch. 6.2 "Shrinkage Methods"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + [ESL Ch. 3.4](https://hastie.su.domains/ElemStatLearn/). *The constrained-optimization view and the closed-form Ridge solution `w = (XᵀX + λI)⁻¹Xᵀy`.*
4. **Read the source** — skim the [Elastic-Net paper](https://hastie.su.domains/Papers/elasticnet.pdf). *Why combining L1 and L2 fixes Lasso's instability with correlated features.*
5. **Make it concrete** — implement with [scikit-learn Ridge / Lasso / ElasticNet](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification). *Sweep `alpha`, plot the coefficient paths, and watch features drop to zero under Lasso.*

## 🎓 Courses (free)
- [Machine Learning Specialization — Course 1](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — regularization for linear and logistic regression, from zero; free to audit.
- [CS229: Machine Learning — Lecture notes](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — regularization in the bias–variance and MAP-estimation framing, rigorously.
- [Google ML Crash Course — Regularization](https://developers.google.com/machine-learning/crash-course/overfitting/regularization) — **Google** — short, applied intro to L2 and the complexity/generalization trade.

## 🎥 Videos
- [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition for shrinkage and variance reduction.
- [Regularization Part 2: Lasso (L1) Regression](https://www.youtube.com/watch?v=NGf0voTMlcs) — **StatQuest (Josh Starmer)** — how L1 performs feature selection by zeroing coefficients.
- [Ridge vs Lasso Regression, Visualized!!!](https://www.youtube.com/watch?v=Xm2C_gTAl8c) — **StatQuest (Josh Starmer)** — the diamond-vs-circle picture: *why* L1 hits zero and L2 doesn't.
- [Regularization Part 3: Elastic-Net Regression](https://www.youtube.com/watch?v=1dKRdX9bfIo) — **StatQuest (Josh Starmer)** — combining L1 and L2 for correlated features.
- [Ridge vs Lasso (intuition)](https://www.youtube.com/watch?v=sO4ZirJh9ds) — **StatQuest (Josh Starmer)** — a compact recap of when to reach for each penalty.

## 📄 Key Papers
- [Regression Shrinkage and Selection via the Lasso](https://www.jstor.org/stable/2346178) — **Tibshirani (1996)** — the paper that introduced the Lasso (JSTOR page; see the open retrospective below for a free read).
- [Regression Shrinkage and Selection via the Lasso: a Retrospective](https://tibshirani.su.domains/ftp/lasso-retro.pdf) — **Tibshirani (2011)** — the author's free, readable recap of the method and its impact.
- [Regularization and Variable Selection via the Elastic Net](https://hastie.su.domains/Papers/elasticnet.pdf) — **Zou & Hastie (2005)** — the L1+L2 hybrid; author-hosted PDF, free.

## 📰 Articles / Blogs (free, no paywall)
- [MLU-Explain: Double Descent](https://mlu-explain.github.io/double-descent/) — **Amazon** — interactive view of model complexity, overfitting, and why regularization matters.
- [Linear models — Ridge, Lasso, Elastic-Net (scikit-learn user guide)](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification) — **scikit-learn** — the practical reference: penalties, solvers, and `alpha` selection.
- [Ordinary Least Squares (visual)](https://setosa.io/ev/ordinary-least-squares-regression/) — **Victor Powell / Setosa** — the baseline OLS picture that regularization modifies.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLR) — **Ch. 6.2 "Shrinkage Methods"**](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the best applied chapter on Ridge, Lasso, and tuning `λ`.
- [The Elements of Statistical Learning — **Ch. 3.4 "Shrinkage Methods"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous treatment with the constraint geometry.
- [Dive into Deep Learning — **Ch. 3.7 "Weight Decay"**](https://d2l.ai/chapter_linear-regression/weight-decay.html) — **Zhang et al.** — L2 regularization as weight decay, with runnable code; the bridge to deep learning.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.10 Regularization L1/L2](../../../AI-ML-intuition/Module_2_Optimization/2.10_Regularization_L1_L2.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Math prerequisites (the *why*): [01. Foundations](../../01.%20Foundations/README.md) — norms, convex optimization, MAP estimation.
- Prior / next concepts: [01 Linear Regression](01-Linear-Regression.md) · [02 Logistic Regression](02-Logistic-Regression.md) · Bias–Variance Tradeoff *(coming soon)*
- Related domain: [5. Deep Learning](../../05.%20Deep_Learning/README.md) — L2 regularization reappears as weight decay; dropout is its neural-net cousin.
