---
id: "03-supervised-learning/random-forests/references"
topic: "Random Forests — References"
parent: "03-supervised-learning/random-forests"
type: references
updated: 2026-06-22
---

# Random Forests — references and further reading

> Companion link library for **[Random Forests](09-Random-Forests.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Random Forests Part 1](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) (**StatQuest**), then play with [MLU-Explain: Random Forest](https://mlu-explain.github.io/random-forest/). *See bootstrapped, feature-randomized trees vote.*
2. **See the details** — watch [Random Forests Part 2 (Missing Data & Clustering)](https://www.youtube.com/watch?v=sQ870aTKqiM) (**StatQuest**). *Proximities, imputation, and what a forest does beyond prediction.*
3. **Get the math** — read [ISLR Ch. 8.2.2](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + [ESL Ch. 15](https://hastie.su.domains/ElemStatLearn/). *Why feature subsampling decorrelates trees and the variance formula behind it.*
4. **Read the source** — skim [Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) (**Breiman 2001**). *The algorithm, OOB error, and a generalization bound.*
5. **Make it concrete** — tune [scikit-learn `RandomForestClassifier`](https://scikit-learn.org/stable/modules/ensemble.html#random-forests); read OOB score and feature importances.

**Videos**:
- [Random Forests Part 1 — Building, Using, Evaluating](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) — **StatQuest (Josh Starmer)** — the gentle, from-scratch build of a forest and OOB error.
- [Random Forests Part 2 — Missing Data & Clustering](https://www.youtube.com/watch?v=sQ870aTKqiM) — **StatQuest (Josh Starmer)** — proximities, imputation, and what else a forest gives you.
- [Random Forest Algorithm, Clearly Explained!](https://www.youtube.com/watch?v=v6VJ2RO66Ag) — **Normalized Nerd** — a clean visual walkthrough of bootstrap + feature randomness.
- [What is Random Forest?](https://www.youtube.com/watch?v=gkXX4h3qYm4) — **IBM Technology** — a crisp conceptual overview of why ensembling trees works.

**Interactive & visual**:
- [MLU-Explain: Random Forest](https://mlu-explain.github.io/random-forest/) — **Amazon** — fully interactive: watch the trees, the votes, and the boundary form.

**Courses (free)**:
- [Google ML Crash Course — Decision Forests](https://developers.google.com/machine-learning/decision-forests) — **Google** — a full free mini-course on trees → forests → boosting.
- [Kaggle Learn — Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) — **Kaggle** — builds and tunes a Random Forest in code.
- [CS229: Machine Learning — Lecture notes (Ensembles)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — bagging, random forests, and variance reduction, rigorously.

**Articles / blogs (free, no paywall)**:
- [Ensemble methods — Forests of randomized trees (scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) — **scikit-learn** — the practical reference: parameters, OOB, importances, extra-trees.
- [Permutation feature importance (scikit-learn)](https://scikit-learn.org/stable/modules/permutation_importance.html) — **scikit-learn** — the fix for impurity importance's high-cardinality bias, with the worked caveats.
- [The Bias–Variance Tradeoff](https://scott.fortmann-roe.com/docs/BiasVariance.html) — **Scott Fortmann-Roe** — *why* decorrelated averaging reduces variance, clearly.

**Key papers**:
- [Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) — **Breiman (2001)** — the original paper: the algorithm (bootstrap + random feature subset per split), OOB error, importances, and a generalization bound.
- [Bagging Predictors](https://www.stat.berkeley.edu/~breiman/bagging.pdf) — **Breiman (1996)** — the variance-reduction foundation forests build on (bootstrap aggregating).
- [Extremely Randomized Trees](https://link.springer.com/article/10.1007/s10994-006-6226-1) — **Geurts, Ernst & Wehenkel (2006)** — push the decorrelation idea further by randomizing the split *thresholds* too (the `ExtraTrees` variant).

**Books (free chapters)**:
- [An Introduction to Statistical Learning (ISLR) — Ch. 8.2 "Bagging, Random Forests, Boosting"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the applied chapter with labs.
- [The Elements of Statistical Learning — Ch. 15 "Random Forests"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous treatment of decorrelation and variance.

**In this platform**:
- Concept page (full explanation): [Random Forests](09-Random-Forests.md)
- Concept depth (the *why*): [AI-ML-intuition 3.08 Ensembles (Bagging/Boosting)](../../../AI-ML-intuition/Module_3_Evaluation/3.08_Ensembles_Bagging_Boosting.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Related: [Bagging](08-Bagging.md) (forests are bagging + feature randomness) · [Decision Trees](07-Decision-Trees.md) (the base learner) · [Bias–Variance Tradeoff](12-Bias-Variance-Tradeoff.md) · [Gradient Boosting (XGBoost)](10-Gradient-Boosting-XGBoost.md) (boosting cuts bias; forests cut variance)
- Math prerequisites: [01. Foundations](../../01.%20Foundations/concepts/README.md) — variance of an average, correlation, the bootstrap
