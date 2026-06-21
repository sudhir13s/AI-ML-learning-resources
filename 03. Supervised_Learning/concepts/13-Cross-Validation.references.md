---
id: "03-supervised-learning/cross-validation/references"
topic: "Cross-Validation — References"
parent: "03-supervised-learning/cross-validation"
type: references
updated: 2026-06-22
---

# Cross-Validation — references and further reading

> Companion link library for **[Cross-Validation](13-Cross-Validation.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) (**StatQuest**). *See k-fold rotate the validation set so every point gets used.*
2. **See it in context** — watch [K-fold Cross Validation (ISLR 5.2)](https://www.youtube.com/watch?v=AMfvd_hLssE) (**Stanford Online**). *LOOCV vs k-fold and the bias–variance of the estimate.*
3. **Get the math** — read [ISLR Ch. 5](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + [ESL Ch. 7.10](https://hastie.su.domains/ElemStatLearn/). *Validation-set, LOOCV, k-fold estimators and their bias–variance.*
4. **Avoid the trap** — read [scikit-learn: Cross-validation (and pitfalls)](https://scikit-learn.org/stable/modules/cross_validation.html). *Stratified / grouped / time-series splits, and why preprocessing must live inside the loop.*
5. **Make it concrete** — wrap a `Pipeline` in [`cross_val_score` / `GridSearchCV`](https://scikit-learn.org/stable/modules/cross_validation.html) so scaling is fit per-fold; then tune with nested CV.

**Videos**:
- [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition for k-fold.
- [Statistical Learning: K-fold Cross Validation (5.2)](https://www.youtube.com/watch?v=AMfvd_hLssE) — **Stanford Online (Hastie & Tibshirani)** — the ISLR authors on LOOCV vs k-fold.
- [K-Fold Cross Validation — Intro to Machine Learning](https://www.youtube.com/watch?v=TIgfjmp-4BA) — **Udacity** — a crisp visual walkthrough of the fold rotation.
- [What is Cross Validation? Leave-One-Out and K-Fold](https://www.youtube.com/watch?v=x1gz-M4VT14) — **Data Science with Yan** — when and why to use each CV variant.

**Interactive & visual**:
- [Visualizing cross-validation behavior in scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html) — **scikit-learn** — colour-coded plots of k-fold, stratified, grouped, and time-series splits side by side.

**Courses (free)**:
- [Machine Learning Specialization — Course 2 (Advice for applying ML)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — train/dev/test splits and model selection.
- [CS229: Machine Learning — Lecture notes (Model selection)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — cross-validation and the bias–variance of the estimator, rigorously.
- [Google ML Crash Course — Generalization & Validation](https://developers.google.com/machine-learning/crash-course/overfitting) — **Google** — short, applied intro to train/validation/test.

**Articles / blogs (free, no paywall)**:
- [Cross-validation: evaluating estimator performance (scikit-learn)](https://scikit-learn.org/stable/modules/cross_validation.html) — **scikit-learn** — the practical reference: k-fold, stratified, grouped, time-series, and pitfalls.
- [Common pitfalls in the interpretation of model evaluation (scikit-learn)](https://scikit-learn.org/stable/common_pitfalls.html) — **scikit-learn** — data leakage and why preprocessing belongs inside the CV loop.

**Key papers**:
- [A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection](https://ai.stanford.edu/~ronnyk/accEst.pdf) — **Kohavi (1995)** — the empirical study that established 10-fold CV as the default.
- [A Survey of Cross-Validation Procedures for Model Selection](https://projecteuclid.org/journals/statistics-surveys/volume-4/issue-none/A-survey-of-cross-validation-procedures-for-model-selection/10.1214/09-SS054.full) — **Arlot & Celisse (2010)** — the definitive modern survey of CV variants and their bias/variance.

**Books (free chapters)**:
- [An Introduction to Statistical Learning (ISLR) — Ch. 5 "Resampling Methods"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the best applied chapter on the validation set, LOOCV, k-fold, and the bootstrap.
- [The Elements of Statistical Learning — Ch. 7.10 "Cross-Validation"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous treatment, including the right vs wrong way to cross-validate.
- [Understanding Machine Learning — Ch. 11 "Model Selection and Validation"](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf) — **Shalev-Shwartz & Ben-David** — the learning-theory view of validation.

**In this platform**:
- Concept page (full explanation): [Cross-Validation](13-Cross-Validation.md)
- Concept depth (the *why*): [AI-ML-intuition 3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md) · [3.05 Classification Metrics](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md)
- Related: [Bias–Variance Tradeoff](12-Bias-Variance-Tradeoff.md) (CV measures where you sit on the curve) · [Classification Metrics](14-Classification-Metrics.md) (what you score per fold) · [Stacking & Blending](11-Stacking-and-Blending.md) (out-of-fold predictions power stacking)
- Math prerequisites: [01. Foundations](../../01.%20Foundations/concepts/README.md) — sampling, variance of an estimator, generalization
