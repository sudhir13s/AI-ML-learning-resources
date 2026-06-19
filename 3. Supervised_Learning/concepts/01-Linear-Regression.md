---
id: "03-supervised-learning/linear-regression"
topic: "Linear Regression"
parent: "03-supervised-learning"
level: beginner
prereqs: ["linear-algebra", "gradient-descent", "calculus"]
interview_frequency: very-high
updated: 2026-06-19
---

# Linear Regression
> Fit a straight line (hyperplane) through data by minimizing squared error, so a continuous target
> becomes a weighted sum of features: `ŷ = w·x + b`. The simplest, most interpretable supervised
> model — and the foundation every other regression and the entire neural-net family builds on.

**Why it matters:** the canonical first ML interview question — derive the least-squares solution,
explain the closed-form normal equation vs gradient descent, state the assumptions (linearity,
independence, homoscedasticity, normal errors), and connect MSE to maximum-likelihood under Gaussian
noise. It's also the cleanest place to first see bias–variance and regularization.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo), then play with [MLU-Explain: Linear Regression](https://mlu-explain.github.io/linear-regression/). *See the line, the residuals, and R² move before any algebra.*
2. **See why least squares works** — watch [StatQuest: Fitting a Line (Least Squares)](https://www.youtube.com/watch?v=PaFPbb66DxQ). *Makes "minimize the sum of squared residuals" concrete and visual.*
3. **Get the math** — read [ISLR Ch. 3 "Linear Regression"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + [CS229 notes §1](https://cs229.stanford.edu/main_notes.pdf). *Derive the normal equation `w = (XᵀX)⁻¹Xᵀy` and the MLE-under-Gaussian-noise view of MSE.*
4. **See how it's optimized** — watch [StatQuest: Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8). *When `XᵀX` is too big to invert, you fit by descending the MSE surface — the bridge to deep learning.*
5. **Make it concrete** — implement it with [d2l Ch. 3 (linear regression from scratch + concise)](https://d2l.ai/chapter_linear-regression/index.html) or [scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html). *Code both the closed form and the gradient-descent version on a tiny dataset.*

## 🎓 Courses (free)
- [Machine Learning Specialization — Course 1 (Regression)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — the canonical from-zero treatment of linear regression + gradient descent; free to audit.
- [CS229: Machine Learning — Lecture notes §1](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — the rigorous derivation (normal equations, probabilistic interpretation) in a few pages.
- [Google ML Crash Course — Linear Regression](https://developers.google.com/machine-learning/crash-course/linear-regression) — **Google** — short, applied, with loss/gradient interactives.

## 🎥 Videos
- [Linear Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=nk2CQITm_eo) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition for fitting, R², and p-values.
- [The Main Ideas of Fitting a Line to Data (Least Squares)](https://www.youtube.com/watch?v=PaFPbb66DxQ) — **StatQuest (Josh Starmer)** — why we minimize *squared* residuals, drawn out step by step.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — **StatQuest (Josh Starmer)** — how you fit the line when you can't (or won't) invert `XᵀX`.
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — **3Blue1Brown** — the same loss-surface intuition, beautifully visualized (and the link to neural nets).

## 📄 Key Papers
- [Gauss and the Invention of Least Squares](https://projecteuclid.org/journals/annals-of-statistics/volume-9/issue-3/Gauss-and-the-Invention-of-Least-Squares/10.1214/aos/1176345451.full) — **Stigler (1981)** — the method's surprisingly contested history; free on Project Euclid.
- [Regression Shrinkage and Selection via the Lasso: a Retrospective](https://tibshirani.su.domains/ftp/lasso-retro.pdf) — **Tibshirani (2011)** — the author revisits where regularized linear regression came from; read after the basics.

## 📰 Articles / Blogs (free, no paywall)
- [MLU-Explain: Linear Regression](https://mlu-explain.github.io/linear-regression/) — **Amazon (Jared Wilber)** — fully interactive: drag points, watch the fit, residuals, and R² update live.
- [Ordinary Least Squares Regression (visual)](https://setosa.io/ev/ordinary-least-squares-regression/) — **Victor Powell / Setosa** — a beautiful visual explanation of OLS and correlation.
- [Linear models (scikit-learn user guide)](https://scikit-learn.org/stable/modules/linear_model.html) — **scikit-learn** — the practical reference: OLS, Ridge, Lasso, and when each applies.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLR) — **Ch. 3 "Linear Regression"**](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the best applied chapter: simple → multiple regression, assumptions, diagnostics.
- [The Elements of Statistical Learning — **Ch. 3 "Linear Methods for Regression"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous reference (subset selection, shrinkage, the geometry of least squares).
- [Dive into Deep Learning — **Ch. 3 "Linear Neural Networks for Regression"**](https://d2l.ai/chapter_linear-regression/index.html) — **Zhang et al.** — frames linear regression as a one-layer net, with runnable from-scratch + concise code.

## 🔗 In this platform
- Math prerequisites (the *why*): [1. Foundations](../../1.%20Foundations/README.md) — linear algebra, gradient descent, maximum likelihood.
- Next concepts: [02 Logistic Regression](02-Logistic-Regression.md) · Regularization for Linear Models *(coming soon)* · Regression Metrics *(coming soon)*
- Related domain: [5. Deep Learning](../../5.%20Deep_Learning/README.md) — linear regression is a single-neuron, no-activation network.
