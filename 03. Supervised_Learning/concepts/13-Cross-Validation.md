---
id: "03-supervised-learning/cross-validation"
topic: "Cross-Validation"
parent: "03-supervised-learning"
level: beginner
prereqs: ["supervised-learning-basics", "bias-variance", "overfitting"]
interview_frequency: very-high
updated: 2026-06-19
---

# Cross-Validation
> Estimate how a model generalizes by repeatedly splitting the data into train/validation folds,
> fitting on each train portion and scoring on the held-out portion, then averaging. **k-fold CV**
> uses every point for both training and validation, giving a lower-variance estimate than a single split.

**Why it matters:** the question behind every "how would you evaluate / tune this?" interview. Expect:
why a single train/test split is noisy, how k-fold reduces that variance and the bias–variance trade in
choosing `k` (LOOCV = low bias, high variance), why **stratified** CV matters for imbalanced classes,
when you need **grouped** or **time-series** CV to prevent leakage, and the cardinal sin of doing
preprocessing/feature-selection *outside* the CV loop (leakage that inflates your score).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw). *See k-fold rotate the validation set so every point gets used — the whole idea in one video.*
2. **See it in context** — watch [Stanford Online: K-fold Cross Validation (ISLR 5.2)](https://www.youtube.com/watch?v=AMfvd_hLssE). *The authors of ISLR explain LOOCV vs k-fold and the bias–variance of the estimate.*
3. **Get the math** — read [ISLR Ch. 5 "Resampling Methods"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + [ESL Ch. 7.10 "Cross-Validation"](https://hastie.su.domains/ElemStatLearn/). *The validation-set, LOOCV, and k-fold estimators and their bias–variance.*
4. **Avoid the trap** — read [scikit-learn: Cross-validation (and common pitfalls)](https://scikit-learn.org/stable/modules/cross_validation.html). *Stratified / grouped / time-series splits, and why preprocessing must live inside the CV loop.*
5. **Make it concrete** — implement with [`cross_val_score` / `GridSearchCV`](https://scikit-learn.org/stable/modules/cross_validation.html). *Wrap a `Pipeline` so scaling is fit per-fold, then tune a hyperparameter with nested CV.*

## 🎓 Courses (free)
- [Machine Learning Specialization — Course 2 (Advice for applying ML)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — train/dev/test splits and model selection; free to audit.
- [CS229: Machine Learning — Lecture notes (Model selection)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — cross-validation and the bias–variance of the estimator, rigorously.
- [Google ML Crash Course — Generalization & Validation](https://developers.google.com/machine-learning/crash-course/overfitting) — **Google** — short, applied intro to train/validation/test and overfitting.

## 🎥 Videos
- [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition for k-fold.
- [Statistical Learning: K-fold Cross Validation (5.2)](https://www.youtube.com/watch?v=AMfvd_hLssE) — **Stanford Online (Hastie & Tibshirani)** — the ISLR authors on LOOCV vs k-fold.
- [K-Fold Cross Validation — Intro to Machine Learning](https://www.youtube.com/watch?v=TIgfjmp-4BA) — **Udacity** — a crisp visual walkthrough of the fold rotation.
- [What is Cross Validation? Leave-One-Out and K-Fold](https://www.youtube.com/watch?v=x1gz-M4VT14) — **Data Science with Yan** — when and why to use each CV variant.

## 📄 Key Papers
- [A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection](https://ai.stanford.edu/~ronnyk/accEst.pdf) — **Kohavi (1995)** — the empirical study that established 10-fold CV as the default; author-hosted PDF, free.
- [A Survey of Cross-Validation Procedures for Model Selection](https://projecteuclid.org/journals/statistics-surveys/volume-4/issue-none/A-survey-of-cross-validation-procedures-for-model-selection/10.1214/09-SS054.full) — **Arlot & Celisse (2010)** — the definitive modern survey of CV variants and their bias/variance; open on Project Euclid.

## 📰 Articles / Blogs (free, no paywall)
- [Cross-validation: evaluating estimator performance (scikit-learn)](https://scikit-learn.org/stable/modules/cross_validation.html) — **scikit-learn** — the practical reference: k-fold, stratified, grouped, time-series, and pitfalls.
- [Common pitfalls in the interpretation of model evaluation (scikit-learn)](https://scikit-learn.org/stable/common_pitfalls.html) — **scikit-learn** — data leakage and why preprocessing belongs inside the CV loop.
- [The Bias–Variance Tradeoff](https://scott.fortmann-roe.com/docs/BiasVariance.html) — **Scott Fortmann-Roe** — what CV is actually estimating: generalization error.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLR) — **Ch. 5 "Resampling Methods"**](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the best applied chapter on the validation set, LOOCV, k-fold, and the bootstrap.
- [The Elements of Statistical Learning — **Ch. 7.10 "Cross-Validation"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous treatment, including the right vs wrong way to cross-validate.
- [Understanding Machine Learning — **Ch. 11 "Model Selection and Validation"**](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf) — **Shalev-Shwartz & Ben-David** — the learning-theory view of validation, free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md) · [3.05 Classification Metrics](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md)
- Math prerequisites (the *why*): [01. Foundations](../../01.%20Foundations/README.md) — sampling, variance of an estimator, generalization.
- Related concepts: [12 Bias–Variance Tradeoff](12-Bias-Variance-Tradeoff.md) — CV measures where you sit on the curve · [11 Stacking & Blending](11-Stacking-and-Blending.md) — out-of-fold CV predictions power stacking.
- Related domain: [2. Data Preprocessing](../../02.%20Data_Preprocessing/README.md) — leakage is the #1 way CV lies to you; fit transforms inside each fold.
