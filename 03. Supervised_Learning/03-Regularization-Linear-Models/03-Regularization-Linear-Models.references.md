---
id: "03-supervised-learning/regularization-linear-models/references"
topic: "Regularization for Linear Models — References"
parent: "03-supervised-learning/regularization-linear-models"
type: references
updated: 2026-06-22
---

# Regularization for Linear Models — references and further reading

> Companion link library for **[Regularization for Linear Models (Ridge · Lasso · Elastic-Net)](03-Regularization-Linear-Models.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) (**StatQuest**). *How shrinking the slope reduces variance and prevents overfitting on small data.*
2. **See the contrast** — watch [Lasso (L1) Regression](https://www.youtube.com/watch?v=NGf0voTMlcs) then [Ridge vs Lasso, Visualized](https://www.youtube.com/watch?v=Xm2C_gTAl8c) (**StatQuest**). *Why L1 zeroes coefficients (the diamond) and L2 only shrinks them.*
3. **Get the math** — read [ESL Ch. 3.4 "Shrinkage Methods"](https://hastie.su.domains/ElemStatLearn/). *The constraint geometry, the closed-form Ridge solution, and SVD shrinkage.*
4. **Read the source** — skim the [Lasso retrospective](https://tibshirani.su.domains/ftp/lasso-retro.pdf) (**Tibshirani**) and the [Elastic-Net paper](https://hastie.su.domains/Papers/elasticnet.pdf) (**Zou & Hastie**). *Why L1 selects, and why L1+L2 fixes its instability on correlated features.*
5. **Make it concrete** — implement with [scikit-learn Ridge / Lasso / ElasticNet](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification). *Sweep `alpha`, plot the coefficient paths, watch features drop to zero under Lasso.*

**Videos**:
- [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition for shrinkage and variance reduction.
- [Regularization Part 2: Lasso (L1) Regression](https://www.youtube.com/watch?v=NGf0voTMlcs) — **StatQuest (Josh Starmer)** — how L1 performs feature selection by zeroing coefficients.
- [Ridge vs Lasso Regression, Visualized!!!](https://www.youtube.com/watch?v=Xm2C_gTAl8c) — **StatQuest (Josh Starmer)** — the diamond-vs-circle picture: *why* L1 hits zero and L2 doesn't.
- [Regularization Part 3: Elastic-Net Regression](https://www.youtube.com/watch?v=1dKRdX9bfIo) — **StatQuest (Josh Starmer)** — combining L1 and L2 for correlated features.

**Interactive & visual**:
- [MLU-Explain: Double Descent](https://mlu-explain.github.io/double-descent/) — **Amazon** — interactive view of model complexity, overfitting, and why regularization matters.
- [Ordinary Least Squares Regression (visual)](https://setosa.io/ev/ordinary-least-squares-regression/) — **Victor Powell / Setosa** — the baseline OLS picture that regularization modifies, fully interactive.

**Courses (free)**:
- [Machine Learning Specialization — Course 1](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — regularization for linear and logistic regression, from zero; free to audit.
- [CS229: Machine Learning — Lecture notes](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — regularization in the bias–variance and MAP-estimation framing, rigorously.
- [Google ML Crash Course — Regularization](https://developers.google.com/machine-learning/crash-course/overfitting/regularization) — **Google** — short, applied intro to L2 and the complexity/generalization trade.

**Articles / blogs (free, no paywall)**:
- [Linear models — Ridge, Lasso, Elastic-Net (scikit-learn user guide)](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification) — **scikit-learn** — the practical reference: penalties, solvers, and `alpha` selection.
- [A Visual Explanation for Regularization of Linear Models](https://explained.ai/regularization/) — **Terence Parr & Jeremy Howard** — the clearest free deep-dive on the diamond/circle geometry, with interactive figures.
- [Lasso (statistics) — the soft-thresholding view](https://en.wikipedia.org/wiki/Lasso_(statistics)) — **Wikipedia** — the orthonormal-case soft-threshold derivation and the regularization path, with citations.

**Key papers**:
- [Ridge regression (history + the Hoerl–Kennard 1970 result)](https://en.wikipedia.org/wiki/Ridge_regression) — summarizing **Hoerl & Kennard (1970)**, *Technometrics* — the paper that introduced Ridge and the $\lambda I$ trick for multicollinearity (the original is paywalled; this is the free, well-cited summary).
- [Regression Shrinkage and Selection via the Lasso](https://www.jstor.org/stable/2346178) — **Tibshirani (1996)** — the paper that introduced the Lasso (JSTOR page; see the open retrospective for a free read).
- [Regression Shrinkage and Selection via the Lasso: a Retrospective](https://tibshirani.su.domains/ftp/lasso-retro.pdf) — **Tibshirani (2011)** — the author's free, readable recap of the method, the soft-threshold, and its impact.
- [Regularization and Variable Selection via the Elastic Net](https://hastie.su.domains/Papers/elasticnet.pdf) — **Zou & Hastie (2005)** — the L1+L2 hybrid and the grouping effect; author-hosted PDF, free.
- [Least Angle Regression (LARS)](https://hastie.su.domains/Papers/LARS/LeastAngle_2002.pdf) — **Efron, Hastie, Johnstone & Tibshirani (2004)** — the piecewise-linear Lasso path computed in one OLS-cost pass.

**Books (free, with chapters)**:
- [An Introduction to Statistical Learning (ISLR) — Ch. 6.2 "Shrinkage Methods"](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — the best applied chapter on Ridge, Lasso, and tuning $\lambda$ (free PDF on the site).
- [The Elements of Statistical Learning (ESL) — Ch. 3.4 "Shrinkage Methods"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous treatment with the constraint geometry, SVD shrinkage, and Bayesian view; free PDF on the book site.
- [Dive into Deep Learning — Ch. 3.7 "Weight Decay"](https://d2l.ai/chapter_linear-regression/weight-decay.html) — **Zhang et al.** — L2 regularization as weight decay, with runnable code; the bridge to deep learning.

**In this platform**:
- Concept page (full explanation): [Regularization for Linear Models](03-Regularization-Linear-Models.md)
- Concept depth (the *why*): [AI-ML-intuition 2.10 Regularization L1/L2](../../../AI-ML-intuition/Module_2_Optimization/2.10_Regularization_L1_L2.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Prior / related concepts: [Linear Regression](../01-Linear-Regression/01-Linear-Regression.md) (the OLS baseline this modifies) · [Logistic Regression](../02-Logistic-Regression/02-Logistic-Regression.md) (same penalties, classification) · [Bias–Variance Tradeoff](../12-Bias-Variance-Tradeoff/12-Bias-Variance-Tradeoff.md) (what the penalty trades) · [Cross-Validation](../13-Cross-Validation/13-Cross-Validation.md) (how $\lambda$ is chosen)
- Math prerequisites: [01. Foundations](../../01.%20Foundations/README.md) — norms, convex optimization, the SVD, and MAP estimation
- Related domain: [Regularization (deep nets)](../../05.%20Deep_Learning/09-Regularization/09-Regularization.md) — L2 reappears as weight decay; dropout is its neural-net cousin
