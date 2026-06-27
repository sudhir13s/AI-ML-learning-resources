---
id: "02-data-preprocessing/data-splits"
topic: "Train / Validation / Test Splits & Cross-Validation"
parent: "02-data-preprocessing"
level: beginner
prereqs: ["generalization", "bias-variance"]
interview_frequency: very-high
updated: 2026-06-20
---

# Train / Validation / Test Splits & Cross-Validation
> Partitioning data so you can tune honestly and estimate real-world performance — a held-out test set you
> touch once, a validation set (or k-fold CV) for model selection, and stratified/grouped/time-aware splits.

**Why it matters:** the foundation of trustworthy evaluation. Interviewers ask why you need three sets (not
two), what k-fold cross-validation buys you, when to use **stratified** (imbalanced classes), **group**
(repeated subjects), or **time-series** (no future leakage) splits, and the cardinal sin of touching the
test set during development. Get this wrong and every metric you report is optimistic.

**⭐ Start here — suggested path:**

1. **Why split at all** — watch [Training & Testing Data](https://www.youtube.com/watch?v=fwY9Qv96DJY). *The basic train/test idea and `train_test_split`.*
2. **Three sets** — watch [Train, Test & Validation sets](https://www.youtube.com/watch?v=600k_das5rc). *Why the validation set is separate from test.*
3. **Cross-validation** — watch [Cross Validation, Clearly Explained](https://www.youtube.com/watch?v=fSytzGwwBVw). *k-fold and why it beats a single split.*
4. **Variants & code** — read [sklearn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html). *StratifiedKFold, GroupKFold, TimeSeriesSplit, and when each applies.*
5. **Avoid optimistic bias** — read [sklearn: Nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html). *Why tuning + evaluating on the same folds inflates scores.*

## 🎓 Courses (free)
- [Google ML Crash Course — Dividing datasets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets) — **Google** — train/validation/test and why each exists.
- [Kaggle Learn — Intermediate ML (Cross-Validation)](https://www.kaggle.com/code/alexisbcook/cross-validation) — **Kaggle** — hands-on CV with a real pipeline.

## 🎥 Videos
- [Training & Testing Data](https://www.youtube.com/watch?v=fwY9Qv96DJY) — **codebasics** — the basic split with `train_test_split`, in code.
- [Train, Test & Validation sets](https://www.youtube.com/watch?v=600k_das5rc) — **S.M.D.S** — why validation is distinct from the held-out test set.
- [Cross Validation, Clearly Explained](https://www.youtube.com/watch?v=fSytzGwwBVw) — **StatQuest (Josh Starmer)** — k-fold CV intuition, the canonical explainer.
- [Cross Validation with Examples](https://www.youtube.com/watch?v=v6DtYYafrWQ) — **Gate Smashers** — LOOCV vs k-fold worked through step by step.

## 📄 Key Papers
- [A Study of Cross-Validation and Bootstrap for Accuracy Estimation](https://ai.stanford.edu/~ronnyk/accEst.pdf) — **Ron Kohavi (1995)** — the classic empirical comparison; why stratified 10-fold CV is the default.
- [No Unbiased Estimator of the Variance of K-Fold Cross-Validation](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf) — **Bengio & Grandvalet (2004)** — the subtle statistics of CV variance; free in JMLR.

## 📰 Articles / Blogs (free, no paywall)
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — **scikit-learn** — the authoritative guide to every splitter (Stratified/Group/TimeSeries).
- [Nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) — **scikit-learn** — how to tune + evaluate without optimistic bias.
- [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) — **scikit-learn** — splitting before any fitting, and how to avoid leakage across folds.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (Python) — **Ch. 5 "Resampling Methods"**](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — validation set, LOOCV, k-fold, bootstrap; free PDF.
- [Tidy Modeling with R — **Ch. 5 (spending data) & Ch. 10 (resampling)**](https://www.tmwr.org/) — **Kuhn & Silge** — principled data splitting + resampling (concepts transfer to Python); free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Next concepts: [11 Data Leakage](../11-Data-Leakage/11-Data-Leakage.md) · [12 Imbalanced Data](../12-Imbalanced-Data/12-Imbalanced-Data.md)
- Related domain: [03. Supervised Learning](../../03.%20Supervised_Learning/README.md)
