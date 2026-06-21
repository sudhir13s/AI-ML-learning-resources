---
id: "03-supervised-learning/bagging/references"
topic: "Bagging — References"
parent: "03-supervised-learning/bagging"
type: references
updated: 2026-06-22
---

# Bagging — references and further reading

> Companion link library for **[Bagging (Bootstrap Aggregating)](08-Bagging.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Bootstrapping Main Ideas](https://www.youtube.com/watch?v=Xz0x-8-cgaQ) (**StatQuest**). *The bootstrap — resampling with replacement — that bagging is built on.*
2. **See the aggregation** — watch [Bootstrap Aggregation (Bagging)](https://www.youtube.com/watch?v=2Mg8QD0F1dQ) (**Udacity**). *How many bootstrap models combine into one lower-variance predictor.*
3. **Get the math** — read [ISLR Ch. 8.2.1](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + [ESL Ch. 8.7](https://hastie.su.domains/ElemStatLearn/). *Why averaging cuts variance ~1/B and out-of-bag error.*
4. **Read the source** — skim [Bagging Predictors](https://www.stat.berkeley.edu/~breiman/bagging.pdf) (**Breiman 1996**). *The paper that named and analyzed bagging.*
5. **Make it concrete** — bag deep trees with [scikit-learn `BaggingClassifier`](https://scikit-learn.org/stable/modules/ensemble.html); watch variance drop and compare to a single tree and a forest.

**Videos**:
- [Bootstrapping Main Ideas!!!](https://www.youtube.com/watch?v=Xz0x-8-cgaQ) — **StatQuest (Josh Starmer)** — the bootstrap resampling that bagging is built on, from scratch.
- [Bootstrap Aggregation (Bagging)](https://www.youtube.com/watch?v=2Mg8QD0F1dQ) — **Udacity** — how bootstrap models are aggregated into one lower-variance predictor.
- [Random Forests Part 1 — Building, Using, Evaluating](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) — **StatQuest (Josh Starmer)** — bagging in action (forests are bagged trees + feature subsampling).
- [Ensemble Learning: Bagging, Boosting & Stacking in 4 minutes](https://www.youtube.com/watch?v=eLt4a8-316E) — **AssemblyAI** — where bagging sits among the three ensemble families.

**Interactive & visual**:
- [MLU-Explain: Random Forest](https://mlu-explain.github.io/random-forest/) — **Amazon** — interactive: see bootstrap samples and aggregation reduce variance (bagging is the engine inside).

**Courses (free)**:
- [Google ML Crash Course — Decision Forests](https://developers.google.com/machine-learning/decision-forests) — **Google** — free mini-course: trees → bagging → forests → boosting.
- [CS229: Machine Learning — Lecture notes (Ensembles)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — bagging and the bias–variance decomposition, rigorously.
- [Kaggle Learn — Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) — **Kaggle** — hands-on trees and the over/underfitting that ensembles fix.

**Articles / blogs (free, no paywall)**:
- [Ensemble methods — Bagging (scikit-learn user guide)](https://scikit-learn.org/stable/modules/ensemble.html) — **scikit-learn** — the practical reference: `BaggingClassifier`, OOB scoring, base estimators.
- [The Bias–Variance Tradeoff](https://scott.fortmann-roe.com/docs/BiasVariance.html) — **Scott Fortmann-Roe** — the clearest free essay on *why* averaging high-variance models works.

**Key papers**:
- [Bagging Predictors](https://www.stat.berkeley.edu/~breiman/bagging.pdf) — **Breiman (1996)** — the original paper that introduced and analyzed bagging.
- [Bagging Predictors (Springer)](https://link.springer.com/article/10.1007/BF00058655) — **Breiman (1996)** — the same landmark paper, publisher page.

**Books (free chapters)**:
- [An Introduction to Statistical Learning (ISLR) — Ch. 8.2 "Bagging, Random Forests, Boosting"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the applied, intuitive treatment with labs.
- [The Elements of Statistical Learning — Ch. 8.7 "Bagging"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous variance-reduction analysis.

**In this platform**:
- Concept page (full explanation): [Bagging](08-Bagging.md)
- Concept depth (the *why*): [AI-ML-intuition 3.08 Ensembles (Bagging/Boosting)](../../../AI-ML-intuition/Module_3_Evaluation/3.08_Ensembles_Bagging_Boosting.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Related: [Decision Trees](07-Decision-Trees.md) (the unstable learner you bag) · [Random Forests](09-Random-Forests.md) (bagging + feature subsampling) · [Bias–Variance Tradeoff](12-Bias-Variance-Tradeoff.md) · [Gradient Boosting](10-Gradient-Boosting-XGBoost.md) (boosting cuts bias; bagging cuts variance)
- Math prerequisites: [01. Foundations](../../01.%20Foundations/concepts/README.md) — the bootstrap, variance of an average, independence
