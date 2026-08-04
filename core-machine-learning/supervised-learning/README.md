---
id: "03-supervised-learning"
topic: "Supervised Learning"
level: intermediate
built_from: ["foundations", "data-preprocessing"]
updated: 2026-06-27
---

# Supervised Learning
> Learning from labeled data — regression, classification, trees, SVMs, and gradient boosting
> (still the king of tabular data).

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page and a curated
`.references.md` resource card (free, open courses · videos · papers · articles · books · cross-links).
> **✅ ready.** New to the field? Start with the field overview below, then work top to bottom.

### Linear models
1. ✅ [Linear Regression](regression/linear-regression/linear-regression.md)
2. ✅ [Logistic Regression](classification/logistic-regression/logistic-regression.md)
3. ✅ [Regularization for Linear Models (Ridge · Lasso · Elastic-Net)](regression/regularization-linear-models/regularization-linear-models.md)

### Instance-based & probabilistic
4. ✅ [k-Nearest Neighbors (k-NN)](classification/k-nearest-neighbors/k-nearest-neighbors.md)
5. ✅ [Naive Bayes](classification/naive-bayes/naive-bayes.md)

### Margin & tree models
6. ✅ [Support Vector Machines (SVM)](classification/support-vector-machines/support-vector-machines.md)
7. ✅ [Decision Trees](trees-and-ensembles/decision-trees/decision-trees.md)

### Ensembles
8. ✅ [Bagging](trees-and-ensembles/bagging/bagging.md)
9. ✅ [Random Forests](trees-and-ensembles/random-forests/random-forests.md)
10. ✅ [Gradient Boosting (XGBoost · LightGBM · CatBoost)](trees-and-ensembles/gradient-boosting-xgboost/gradient-boosting-xgboost.md)
11. ✅ [Stacking & Blending](trees-and-ensembles/stacking-and-blending/stacking-and-blending.md)

### Theory & evaluation
12. ✅ [Bias–Variance Tradeoff](../model-selection-and-evaluation/bias-variance-tradeoff/bias-variance-tradeoff.md)
13. ✅ [Cross-Validation](../model-selection-and-evaluation/cross-validation/cross-validation.md)
14. ✅ [Classification Metrics (precision · recall · F1 · ROC-AUC · PR-AUC)](classification/classification-metrics/classification-metrics.md)
15. ✅ [Regression Metrics (RMSE · MAE · R²)](regression/regression-metrics/regression-metrics.md)

### Related concepts (covered in another section)
> These topics are used across many areas, so they're kept in one place to avoid repetition.
- **Math & optimization** — Gradient Descent · Maximum Likelihood · Convexity · Linear Algebra → [01. Foundations](../../foundations/mathematical-foundations/README.md)
- **Neural networks** — MLPs · Backpropagation · Activation functions · Regularization (dropout/BN) → [05. Deep Learning](../../deep-learning/README.md)
- **Clustering & dimensionality reduction** — k-Means · PCA · t-SNE · GMMs → [04. Unsupervised Learning](../unsupervised-learning/README.md)
- **Feature engineering & data prep** — scaling · encoding · imputation · leakage → [02. Data Preprocessing](../../foundations/data-preparation/README.md)

## 🎓 Courses (free)
- [Machine Learning Specialization (Courses 1–2)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng** — regression → classification, the canonical intro.
- [Kaggle Learn: Intro + Intermediate ML](https://www.kaggle.com/learn) — **Kaggle** — trees → random forests → XGBoost, hands-on.

## 🎥 Videos
- [StatQuest ML playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) — **Josh Starmer** — linear/logistic regression, trees, SVMs, boosting — each clearly explained.

## 📄 Key Papers
- [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324) — **Breiman (2001)** — the bagging classic.
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) — **Chen & Guestrin (2016)** — the tabular-ML workhorse.

## 📚 Books (free)
- [An Introduction to Statistical Learning (ISLP)](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — free; the best applied-ML textbook, with Python labs.
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) — **Hastie et al.** — free; the rigorous reference.

## 🔗 In this platform
- Losses & metrics: [AI-ML-intuition Module 3](../../../AI-ML-intuition/Module_3_Evaluation/) · Practice: [problemsets](../../../AI-ML-problemsets/)
