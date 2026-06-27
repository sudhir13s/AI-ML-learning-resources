---
id: "03-supervised-learning/linear-regression/references"
topic: "Linear Regression — References"
parent: "03-supervised-learning/linear-regression"
type: references
updated: 2026-06-22
---

# Linear Regression — references and further reading

> Companion link library for **[Linear Regression](01-Linear-Regression.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo) (**StatQuest**), then play with [MLU-Explain: Linear Regression](https://mlu-explain.github.io/linear-regression/). *See the line, residuals, and R² move before any algebra.*
2. **See why least squares works** — watch [Fitting a Line (Least Squares)](https://www.youtube.com/watch?v=PaFPbb66DxQ) (**StatQuest**). *Why "minimize the sum of squared residuals," concretely.*
3. **Get the math** — read [ISLR Ch. 3](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + [CS229 notes §1](https://cs229.stanford.edu/main_notes.pdf). *Derive the normal equation and the MLE-under-Gaussian-noise view.*
4. **See how it's optimized** — watch [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) (**StatQuest**). *Fitting by descending the MSE surface — the bridge to deep learning.*
5. **Make it concrete** — code both the closed form and gradient descent with [d2l Ch. 3](https://d2l.ai/chapter_linear-regression/index.html) or [scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html).

**Videos**:
- [Linear Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=nk2CQITm_eo) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition for fitting, R², and p-values.
- [The Main Ideas of Fitting a Line to Data (Least Squares)](https://www.youtube.com/watch?v=PaFPbb66DxQ) — **StatQuest (Josh Starmer)** — why we minimize *squared* residuals, step by step.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — **StatQuest (Josh Starmer)** — how you fit the line when you can't (or won't) invert XᵀX.
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — **3Blue1Brown** — the same loss-surface intuition, and the link to neural nets.
- [Linear regression (full lecture)](https://www.youtube.com/watch?v=4b4MUYve_U8) — **Andrew Ng / Stanford CS229** — the canonical lecture: model, cost, normal equations, and the probabilistic view derived on the board.

**Interactive & visual**:
- [MLU-Explain: Linear Regression](https://mlu-explain.github.io/linear-regression/) — **Amazon (Jared Wilber)** — fully interactive: drag points, watch the fit, residuals, and R² update live.
- [Ordinary Least Squares Regression (visual)](https://setosa.io/ev/ordinary-least-squares-regression/) — **Victor Powell / Setosa** — a beautiful visual explanation of OLS and correlation.

**Courses (free)**:
- [Machine Learning Specialization — Course 1 (Regression)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — the canonical from-zero treatment of linear regression + gradient descent.
- [CS229: Machine Learning — Lecture notes §1](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — the rigorous derivation (normal equations, probabilistic interpretation).
- [Google ML Crash Course — Linear Regression](https://developers.google.com/machine-learning/crash-course/linear-regression) — **Google** — short, applied, with loss/gradient interactives.

**Articles / blogs (free, no paywall)**:
- [Linear models (scikit-learn user guide)](https://scikit-learn.org/stable/modules/linear_model.html) — **scikit-learn** — the practical reference: OLS, Ridge, Lasso, and when each applies.
- [Ordinary least squares (Wikipedia)](https://en.wikipedia.org/wiki/Ordinary_least_squares) — concise, well-sourced reference for the normal equations, the hat matrix/projection geometry, and the Gauss–Markov theorem.

**Key papers**:
- [Gauss and the Invention of Least Squares](https://projecteuclid.org/journals/annals-of-statistics/volume-9/issue-3/Gauss-and-the-Invention-of-Least-Squares/10.1214/aos/1176345451.full) — **Stigler (1981)** — the method's surprisingly contested history (the Gauss–Legendre priority dispute).
- [Ridge regression](https://en.wikipedia.org/wiki/Ridge_regression) — overview citing **Hoerl & Kennard (1970)**, the original "Biased Estimation for Nonorthogonal Problems" that introduced Ridge and the $+\lambda I$ fix for multicollinearity.
- [Regression Shrinkage and Selection via the Lasso: a Retrospective](https://tibshirani.su.domains/ftp/lasso-retro.pdf) — **Tibshirani (2011)** — the author revisits where regularized linear regression came from.

**Books (free chapters)**:
- [An Introduction to Statistical Learning (ISLR) — Ch. 3 "Linear Regression"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the best applied chapter: simple → multiple regression, assumptions, diagnostics.
- [The Elements of Statistical Learning — Ch. 3 "Linear Methods for Regression"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous reference (subset selection, shrinkage, the geometry of least squares).
- [Pattern Recognition and Machine Learning — §3.1 "Linear Basis Function Models"](https://www.bishopbook.com/) — **Christopher Bishop** — the Bayesian / maximum-likelihood treatment (MSE = Gaussian MLE, Ridge = Gaussian prior); the author's official site links the free PDF.
- [Dive into Deep Learning — Ch. 3 "Linear Neural Networks for Regression"](https://d2l.ai/chapter_linear-regression/index.html) — **Zhang et al.** — frames linear regression as a one-layer net, with runnable code.

**In this platform**:
- Concept page (full explanation): [Linear Regression](01-Linear-Regression.md)
- Concept depth (the *why*): [AI-ML-intuition 3.01 MSE / L2 Loss](../../../AI-ML-intuition/Module_3_Evaluation/3.01_Mean_Squared_Error_MSE_L2_Loss.md) · [2.05 Gradient Descent & SGD](../../../AI-ML-intuition/Module_2_Optimization/2.05_Gradient_Descent_and_SGD.md)
- Related: [Logistic Regression](../02-Logistic-Regression/02-Logistic-Regression.md) (linear regression through a sigmoid) · [Loss Functions](../../05.%20Deep_Learning/04-Loss-Functions/04-Loss-Functions.md) (MSE = Gaussian MLE) · [Regularization (Linear Models)](../03-Regularization-Linear-Models/03-Regularization-Linear-Models.md) (Ridge/Lasso)
- Math prerequisites: [01. Foundations](../../01.%20Foundations/README.md) — linear algebra, gradient descent, maximum likelihood
