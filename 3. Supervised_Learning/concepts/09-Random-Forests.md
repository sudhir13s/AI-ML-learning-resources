---
id: "03-supervised-learning/random-forests"
topic: "Random Forests"
parent: "03-supervised-learning"
level: intermediate
prereqs: ["decision-trees", "bagging", "bias-variance"]
interview_frequency: very-high
updated: 2026-06-19
---

# Random Forests
> Bag many decision trees, but at **each split** consider only a random subset of features. That extra
> randomness *decorrelates* the trees, so averaging them cuts variance far more than plain bagging.
> A robust, low-tuning, near-default model for tabular data — and a constant interview favorite.

**Why it matters:** the most-asked ensemble after gradient boosting. Expect: how it differs from
plain bagging (random **feature** subsampling at each split → decorrelated trees), why decorrelation is
the whole point (variance of an average drops faster when terms are uncorrelated), what `max_features`
and `n_estimators` do, how **out-of-bag** error gives free validation, and how feature importances are
computed (and their bias toward high-cardinality features).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: Random Forests Part 1](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ), then play with [MLU-Explain: Random Forest](https://mlu-explain.github.io/random-forest/). *See bootstrapped, feature-randomized trees vote.*
2. **See the details** — watch [StatQuest: Random Forests Part 2 (Missing Data & Clustering)](https://www.youtube.com/watch?v=sQ870aTKqiM). *Proximities, imputation, and what a forest does beyond prediction.*
3. **Get the math** — read [ISLR Ch. 8.2.2 "Random Forests"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + [ESL Ch. 15](https://hastie.su.domains/ElemStatLearn/). *Why feature subsampling decorrelates trees and the variance formula that makes it work.*
4. **Read the source** — skim [Breiman: Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf). *The original paper: the algorithm, OOB error, and a generalization-error bound; author-hosted PDF.*
5. **Make it concrete** — implement with [scikit-learn `RandomForestClassifier`](https://scikit-learn.org/stable/modules/ensemble.html#random-forests). *Tune `n_estimators` / `max_features`, read OOB score, and inspect feature importances.*

## 🎓 Courses (free)
- [Google ML Crash Course — Decision Forests](https://developers.google.com/machine-learning/decision-forests) — **Google** — a full free mini-course on trees → forests → boosting.
- [Kaggle Learn — Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) — **Kaggle** — builds and tunes a Random Forest in code, free.
- [CS229: Machine Learning — Lecture notes (Ensembles)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — bagging, random forests, and variance reduction, rigorously.

## 🎥 Videos
- [StatQuest: Random Forests Part 1 — Building, Using, Evaluating](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) — **StatQuest (Josh Starmer)** — the gentle, from-scratch build of a forest and OOB error.
- [StatQuest: Random Forests Part 2 — Missing Data & Clustering](https://www.youtube.com/watch?v=sQ870aTKqiM) — **StatQuest (Josh Starmer)** — proximities, imputation, and what else a forest gives you.
- [Random Forest Algorithm, Clearly Explained!](https://www.youtube.com/watch?v=v6VJ2RO66Ag) — **Normalized Nerd** — a clean visual walkthrough of bootstrap + feature randomness.
- [What is Random Forest?](https://www.youtube.com/watch?v=gkXX4h3qYm4) — **IBM Technology** — a crisp conceptual overview of why ensembling trees works.

## 📄 Key Papers
- [Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) — **Breiman (2001)** — the original paper: the algorithm, OOB error, importances, and a generalization bound; author-hosted PDF, free.
- [Bagging Predictors](https://www.stat.berkeley.edu/~breiman/bagging.pdf) — **Breiman (1996)** — the variance-reduction foundation forests build on; author-hosted PDF, free.

## 📰 Articles / Blogs (free, no paywall)
- [Ensemble methods — Forests of randomized trees (scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) — **scikit-learn** — the practical reference: parameters, OOB, importances, extra-trees.
- [MLU-Explain: Random Forest](https://mlu-explain.github.io/random-forest/) — **Amazon** — fully interactive: watch trees, votes, and the boundary form.
- [The Bias–Variance Tradeoff](https://scott.fortmann-roe.com/docs/BiasVariance.html) — **Scott Fortmann-Roe** — *why* decorrelated averaging reduces variance, clearly.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLR) — **Ch. 8.2 "Bagging, Random Forests, Boosting"**](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the applied chapter with labs.
- [The Elements of Statistical Learning — **Ch. 15 "Random Forests"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous treatment of decorrelation and variance.
- [Dive into Deep Learning — **Ensembles context**](https://d2l.ai/) — **Zhang et al.** — forests among ensemble methods with runnable code.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.08 Ensembles (Bagging/Boosting)](../../../AI-ML-intuition/Module_3_Evaluation/3.08_Ensembles_Bagging_Boosting.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Math prerequisites (the *why*): [1. Foundations](../../1.%20Foundations/README.md) — variance of an average, correlation, the bootstrap.
- Prior / next concepts: [08 Bagging](08-Bagging.md) — forests are bagging + feature randomness · [07 Decision Trees](07-Decision-Trees.md) — the base learner.
- Contrast: [10 Gradient Boosting](10-Gradient-Boosting-XGBoost.md) — boosting reduces **bias** sequentially; forests reduce **variance** in parallel.
