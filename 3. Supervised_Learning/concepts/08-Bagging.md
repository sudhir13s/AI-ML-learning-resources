---
id: "03-supervised-learning/bagging"
topic: "Bagging (Bootstrap Aggregating)"
parent: "03-supervised-learning"
level: intermediate
prereqs: ["decision-trees", "bias-variance", "bootstrap"]
interview_frequency: medium
updated: 2026-06-19
---

# Bagging — Bootstrap Aggregating
> Train many copies of the same high-variance model on different **bootstrap samples** (sampled with
> replacement) of the data, then average their predictions (or vote). Averaging decorrelated,
> low-bias/high-variance learners cancels their errors — the variance-reduction half of ensembling, and
> the direct parent of Random Forests.

**Why it matters:** the cleanest way to demonstrate you understand the bias–variance tradeoff. Expect:
*why* bagging reduces **variance** but not bias (so you bag low-bias, high-variance models like deep
trees), what a bootstrap sample is and why ~37% of points are left out (**out-of-bag** estimation),
why the base learners must be *unstable* for bagging to help, and how it contrasts with boosting
(parallel/variance vs sequential/bias).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: Bootstrapping Main Ideas](https://www.youtube.com/watch?v=Xz0x-8-cgaQ). *Understand the bootstrap — resampling with replacement — that bagging is built on.*
2. **See the aggregation** — watch [Bagging (Bootstrap Aggregation)](https://www.youtube.com/watch?v=2Mg8QD0F1dQ). *How many bootstrap models are combined into one lower-variance predictor.*
3. **Get the math** — read [ISLR Ch. 8.2.1 "Bagging"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + [ESL Ch. 8.7](https://hastie.su.domains/ElemStatLearn/). *Why averaging cuts variance by roughly `1/B` for independent learners, and out-of-bag error.*
4. **Read the source** — skim [Breiman: Bagging Predictors](https://www.stat.berkeley.edu/~breiman/bagging.pdf). *The original paper that named and analyzed bagging; author-hosted PDF.*
5. **Make it concrete** — implement with [scikit-learn `BaggingClassifier`](https://scikit-learn.org/stable/modules/ensemble.html). *Bag deep decision trees, watch variance drop, and compare to a single tree and to a Random Forest.*

## 🎓 Courses (free)
- [Google ML Crash Course — Decision Forests](https://developers.google.com/machine-learning/decision-forests) — **Google** — free mini-course: trees → bagging → forests → boosting.
- [CS229: Machine Learning — Lecture notes (Ensembles)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — bagging and the bias–variance decomposition, rigorously.
- [Kaggle Learn — Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) — **Kaggle** — hands-on trees and the over/underfitting that ensembles fix, free.

## 🎥 Videos
- [Bootstrapping Main Ideas!!!](https://www.youtube.com/watch?v=Xz0x-8-cgaQ) — **StatQuest (Josh Starmer)** — the bootstrap resampling that bagging is built on, from scratch.
- [Bootstrap Aggregation (Bagging)](https://www.youtube.com/watch?v=2Mg8QD0F1dQ) — **Udacity** — how bootstrap models are aggregated into one lower-variance predictor.
- [StatQuest: Random Forests Part 1 — Building, Using, Evaluating](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) — **StatQuest (Josh Starmer)** — bagging in action (forests are bagged trees + feature subsampling).
- [Ensemble Learning: Bagging, Boosting & Stacking in 4 minutes](https://www.youtube.com/watch?v=eLt4a8-316E) — **AssemblyAI** — where bagging sits among the three ensemble families.

## 📄 Key Papers
- [Bagging Predictors](https://www.stat.berkeley.edu/~breiman/bagging.pdf) — **Breiman (1996)** — the original paper that introduced and analyzed bagging; author-hosted PDF, free.
- [Bagging Predictors (Springer)](https://link.springer.com/article/10.1007/BF00058655) — **Breiman (1996)** — the same landmark paper, publisher page.

## 📰 Articles / Blogs (free, no paywall)
- [Ensemble methods — Bagging (scikit-learn user guide)](https://scikit-learn.org/stable/modules/ensemble.html) — **scikit-learn** — the practical reference: `BaggingClassifier`, out-of-bag scoring, base estimators.
- [MLU-Explain: Random Forest](https://mlu-explain.github.io/random-forest/) — **Amazon** — interactive: see bootstrap samples and aggregation reduce variance.
- [The Bias–Variance Tradeoff](https://scott.fortmann-roe.com/docs/BiasVariance.html) — **Scott Fortmann-Roe** — the clearest free essay on *why* averaging high-variance models works.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLR) — **Ch. 8.2 "Bagging, Random Forests, Boosting"**](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the applied, intuitive treatment with labs.
- [The Elements of Statistical Learning — **Ch. 8.7 "Bagging"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous variance-reduction analysis.
- [Dive into Deep Learning — **Ensembles context**](https://d2l.ai/) — **Zhang et al.** — bagging situated among ensemble methods with runnable code.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.08 Ensembles (Bagging/Boosting)](../../../AI-ML-intuition/Module_3_Evaluation/3.08_Ensembles_Bagging_Boosting.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Math prerequisites (the *why*): [1. Foundations](../../1.%20Foundations/README.md) — the bootstrap, variance of an average, independence.
- Prior / next concepts: [07 Decision Trees](07-Decision-Trees.md) — the high-variance learner you bag · [09 Random Forests](09-Random-Forests.md) — bagging + feature subsampling.
- Contrast: [10 Gradient Boosting](10-Gradient-Boosting-XGBoost.md) — boosting reduces **bias** sequentially; bagging reduces **variance** in parallel.
