---
id: "03-supervised-learning"
topic: "Supervised Learning"
level: intermediate
prereqs: ["foundations", "data-preprocessing"]
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
1. ✅ [Linear Regression](01-Linear-Regression/01-Linear-Regression.md)
2. ✅ [Logistic Regression](02-Logistic-Regression/02-Logistic-Regression.md)
3. ✅ [Regularization for Linear Models (Ridge · Lasso · Elastic-Net)](03-Regularization-Linear-Models/03-Regularization-Linear-Models.md)

### Instance-based & probabilistic
4. ✅ [k-Nearest Neighbors (k-NN)](04-k-Nearest-Neighbors/04-k-Nearest-Neighbors.md)
5. ✅ [Naive Bayes](05-Naive-Bayes/05-Naive-Bayes.md)

### Margin & tree models
6. ✅ [Support Vector Machines (SVM)](06-Support-Vector-Machines/06-Support-Vector-Machines.md)
7. ✅ [Decision Trees](07-Decision-Trees/07-Decision-Trees.md)

### Ensembles
8. ✅ [Bagging](08-Bagging/08-Bagging.md)
9. ✅ [Random Forests](09-Random-Forests/09-Random-Forests.md)
10. ✅ [Gradient Boosting (XGBoost · LightGBM · CatBoost)](10-Gradient-Boosting-XGBoost/10-Gradient-Boosting-XGBoost.md)
11. ✅ [Stacking & Blending](11-Stacking-and-Blending/11-Stacking-and-Blending.md)

### Theory & evaluation
12. ✅ [Bias–Variance Tradeoff](12-Bias-Variance-Tradeoff/12-Bias-Variance-Tradeoff.md)
13. ✅ [Cross-Validation](13-Cross-Validation/13-Cross-Validation.md)
14. ✅ [Classification Metrics (precision · recall · F1 · ROC-AUC · PR-AUC)](14-Classification-Metrics/14-Classification-Metrics.md)
15. ✅ [Regression Metrics (RMSE · MAE · R²)](15-Regression-Metrics/15-Regression-Metrics.md)

### Related concepts (covered in another section)
> These topics are used across many areas, so they're kept in one place to avoid repetition.
- **Math & optimization** — Gradient Descent · Maximum Likelihood · Convexity · Linear Algebra → [01. Foundations](../01.%20Foundations/README.md)
- **Neural networks** — MLPs · Backpropagation · Activation functions · Regularization (dropout/BN) → [05. Deep Learning](../05.%20Deep_Learning/README.md)
- **Clustering & dimensionality reduction** — k-Means · PCA · t-SNE · GMMs → [04. Unsupervised Learning](../04.%20Unsupervised_Learning/README.md)
- **Feature engineering & data prep** — scaling · encoding · imputation · leakage → [02. Data Preprocessing](../02.%20Data_Preprocessing/README.md)

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
- Losses & metrics: [AI-ML-intuition Module 3](../../AI-ML-intuition/Module_3_Evaluation/) · Practice: [problemsets](../../AI-ML-problemsets/)
