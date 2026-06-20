---
id: "03-supervised-learning/logistic-regression"
topic: "Logistic Regression"
parent: "03-supervised-learning"
level: beginner
prereqs: ["linear-regression", "sigmoid", "maximum-likelihood", "gradient-descent"]
interview_frequency: very-high
updated: 2026-06-19
---

# Logistic Regression
> Turn a linear score `w·x + b` into a probability with the sigmoid `σ(z) = 1/(1+e⁻ᶻ)`, then fit by
> maximizing the likelihood of the labels (equivalently, minimizing cross-entropy / log-loss).
> The workhorse of binary classification and the output layer of every classification neural net.

**Why it matters:** the most-asked classification interview question — why use the sigmoid, why
log-loss instead of MSE (convexity + the right gradient), how it relates to linear regression and to
a single neuron, how the decision boundary is linear, and how regularization (L1/L2) controls it.
Interpreting coefficients as **log-odds** is a classic follow-up.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8), then play with [MLU-Explain: Logistic Regression](https://mlu-explain.github.io/logistic-regression/). *See the S-curve, the probability, and the decision boundary before the math.*
2. **See why it works** — watch [StatQuest: Logistic Regression Details Pt 1 (Coefficients)](https://www.youtube.com/watch?v=vN5cNN2-HWE). *Coefficients are log-odds — this is the interview follow-up that trips people up.*
3. **Get the math** — read [SLP3 Ch. 5 "Logistic Regression"](https://web.stanford.edu/~jurafsky/slp3/5.pdf) + [CS229 notes §1.2](https://cs229.stanford.edu/main_notes.pdf). *Derive cross-entropy from maximum likelihood and show its gradient is `(ŷ − y)·x` — strikingly like linear regression.*
4. **See how it's optimized** — watch [StatQuest: Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8). *Log-loss is convex, so gradient descent (or Newton/IRLS) reliably finds the optimum.*
5. **Make it concrete** — implement it with [d2l Ch. 4 (softmax/linear classification)](https://d2l.ai/chapter_linear-classification/index.html) or [scikit-learn `LogisticRegression`](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression). *Code the sigmoid, the log-loss, and its gradient on a 2-feature dataset.*

## 🎓 Courses (free)
- [Machine Learning Specialization — Course 1 (Classification)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — the canonical intro to logistic regression, sigmoid, and log-loss; free to audit.
- [CS229: Machine Learning — Lecture notes §1.2 (Classification)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — the rigorous MLE derivation and the GLM view in a few pages.
- [Google ML Crash Course — Logistic Regression](https://developers.google.com/machine-learning/crash-course/logistic-regression) — **Google** — short, applied, with log-loss and calibration interactives.

## 🎥 Videos
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition for the S-curve and odds.
- [Logistic Regression Details Pt 1: Coefficients](https://www.youtube.com/watch?v=vN5cNN2-HWE) — **StatQuest (Josh Starmer)** — coefficients as log-odds, the classic interview follow-up.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — **StatQuest (Josh Starmer)** — how the convex log-loss is actually minimized.
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — **3Blue1Brown** — the loss-surface picture, and why logistic regression is one neuron with a sigmoid.

## 📄 Key Papers
- [The Regression Analysis of Binary Sequences (Cox, 1958)](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1958-Cox.pdf) — **D. R. Cox** — the paper that introduced logistic regression; the foundational source, free PDF.
- [Maximum Likelihood Estimation of Logistic Regression Models (tutorial)](https://czep.net/stat/mlelr.pdf) — **Scott Czepiel** — a clean, free, self-contained derivation of the MLE + Newton-Raphson updates.

## 📰 Articles / Blogs (free, no paywall)
- [MLU-Explain: Logistic Regression](https://mlu-explain.github.io/logistic-regression/) — **Amazon (Jared Wilber)** — fully interactive: move the boundary, watch probabilities and log-loss respond.
- [Logistic Regression](https://www.jeremyjordan.me/logistic-regression/) — **Jeremy Jordan** — a clear, free walkthrough of the model, loss, and gradient.
- [Logistic regression (scikit-learn user guide)](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) — **scikit-learn** — the practical reference: solvers, regularization (L1/L2/elastic-net), multiclass.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLR) — **Ch. 4 "Classification"**](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — logistic regression, odds, decision boundaries, with applied examples.
- [Speech and Language Processing, 3rd ed. — **Ch. 5 "Logistic Regression"**](https://web.stanford.edu/~jurafsky/slp3/5.pdf) — **Jurafsky & Martin** — the cleanest cross-entropy + gradient derivation, from an NLP angle.
- [The Elements of Statistical Learning — **Ch. 4 "Linear Methods for Classification"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous treatment (logistic regression vs LDA, IRLS).

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.03 Categorical Cross-Entropy / NLL](../../../AI-ML-intuition/Module_3_Evaluation/3.03_Categorical_Cross-Entropy_NLL.md) · [3.05 Classification Metrics](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md)
- Math prerequisites (the *why*): [01. Foundations](../../01.%20Foundations/concepts/README.md) — sigmoid, maximum likelihood, cross-entropy, gradient descent.
- Prior / next concepts: [01 Linear Regression](01-Linear-Regression.md) · Classification Metrics *(coming soon)* · Regularization for Linear Models *(coming soon)*
- Related domain: [5. Deep Learning](../../05.%20Deep_Learning/concepts/README.md) — softmax/cross-entropy is multiclass logistic regression as a net's output layer.
