---
id: "03-supervised-learning/logistic-regression/references"
topic: "Logistic Regression — References"
parent: "03-supervised-learning/logistic-regression"
type: references
updated: 2026-06-22
---

# Logistic Regression — references and further reading

> Companion link library for **[Logistic Regression](02-Logistic-Regression.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) (**StatQuest**), then play with [MLU-Explain: Logistic Regression](https://mlu-explain.github.io/logistic-regression/). *See the S-curve, the probability, and the boundary before the math.*
2. **See the coefficients** — watch [Logistic Regression Details Pt 1: Coefficients](https://www.youtube.com/watch?v=vN5cNN2-HWE) (**StatQuest**). *Coefficients are log-odds — the interview follow-up that trips people up.*
3. **Get the math** — read [SLP3 Ch. 5](https://web.stanford.edu/~jurafsky/slp3/5.pdf) + [CS229 notes §1.2](https://cs229.stanford.edu/main_notes.pdf). *Derive cross-entropy from MLE and show its gradient is (p − y)·x.*
4. **See how it's optimized** — watch [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) (**StatQuest**). *Log-loss is convex, so gradient descent reliably finds the optimum.*
5. **Make it concrete** — code the sigmoid, log-loss, and its gradient, or use [scikit-learn `LogisticRegression`](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).

**Videos**:
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition for the S-curve and odds.
- [Logistic Regression Details Pt 1: Coefficients](https://www.youtube.com/watch?v=vN5cNN2-HWE) — **StatQuest (Josh Starmer)** — coefficients as log-odds, the classic interview follow-up.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — **StatQuest (Josh Starmer)** — how the convex log-loss is actually minimized.
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — **3Blue1Brown** — why logistic regression is one neuron with a sigmoid.

**Interactive & visual**:
- [MLU-Explain: Logistic Regression](https://mlu-explain.github.io/logistic-regression/) — **Amazon (Jared Wilber)** — fully interactive: move the boundary and watch probabilities and log-loss respond.

**Courses (free)**:
- [Machine Learning Specialization — Course 1 (Classification)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — the canonical intro to logistic regression, sigmoid, and log-loss; free to audit.
- [CS229: Machine Learning — Lecture notes §1.2 (Classification)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — the rigorous MLE derivation and the GLM view in a few pages.
- [Google ML Crash Course — Logistic Regression](https://developers.google.com/machine-learning/crash-course/logistic-regression) — **Google** — short, applied, with log-loss and calibration interactives.

**Articles / blogs (free, no paywall)**:
- [Logistic Regression](https://www.jeremyjordan.me/logistic-regression/) — **Jeremy Jordan** — a clear, free walkthrough of the model, loss, and gradient.
- [Logistic regression (scikit-learn user guide)](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) — **scikit-learn** — the practical reference: solvers, regularization (L1/L2/elastic-net), multiclass.

**Key papers**:
- [The Regression Analysis of Binary Sequences (Cox, 1958)](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1958-Cox.pdf) — **D. R. Cox** — the paper that introduced logistic regression; the foundational source.
- [Maximum Likelihood Estimation of Logistic Regression Models (tutorial)](https://czep.net/stat/mlelr.pdf) — **Scott Czepiel** — a clean, self-contained derivation of the MLE + Newton-Raphson updates.

**Books (free chapters)**:
- [An Introduction to Statistical Learning (ISLR) — Ch. 4 "Classification"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — logistic regression, odds, decision boundaries, with applied examples.
- [Speech and Language Processing, 3rd ed. — Ch. 5 "Logistic Regression"](https://web.stanford.edu/~jurafsky/slp3/5.pdf) — **Jurafsky & Martin** — the cleanest cross-entropy + gradient derivation.
- [The Elements of Statistical Learning — Ch. 4 "Linear Methods for Classification"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous treatment (logistic regression vs LDA, IRLS).

**In this platform**:
- Concept page (full explanation): [Logistic Regression](02-Logistic-Regression.md)
- Concept depth (the *why*): [AI-ML-intuition 3.03 Cross-Entropy / NLL](../../../AI-ML-intuition/Module_3_Evaluation/3.03_Categorical_Cross-Entropy_NLL.md) · [3.05 Classification Metrics](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md)
- Related: [Linear Regression](01-Linear-Regression.md) · [Classification Metrics](14-Classification-Metrics.md) (the probabilities you threshold) · [Loss Functions](../../05.%20Deep_Learning/concepts/04-Loss-Functions.md) (softmax+cross-entropy = multiclass logistic regression)
- Math prerequisites: [01. Foundations](../../01.%20Foundations/concepts/README.md) — sigmoid, maximum likelihood, cross-entropy, gradient descent
