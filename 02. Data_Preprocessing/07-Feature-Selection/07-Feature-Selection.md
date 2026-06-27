---
id: "02-data-preprocessing/feature-selection"
topic: "Feature Selection"
parent: "02-data-preprocessing"
level: intermediate
prereqs: ["correlation", "mutual-information", "regularization"]
interview_frequency: high
updated: 2026-06-20
---

# Feature Selection
> Choosing the subset of features that actually helps — via filter (statistics), wrapper (search), or
> embedded (model-driven) methods — to fight the curse of dimensionality, overfitting, and noise.

**Why it matters:** more features isn't better — irrelevant/redundant ones add variance, slow training,
and hurt interpretability. Interviewers want the three families (filter vs wrapper vs embedded), concrete
examples (variance threshold, correlation/mutual-information, RFE, L1/Lasso, tree importances), and the
crucial gotcha: **feature selection must happen inside cross-validation**, on training folds only, or you
leak and over-report performance.

**⭐ Start here — suggested path:**

1. **The three families** — watch [Filter vs Wrapper vs Embedded](https://www.youtube.com/watch?v=Sc-TNxW3PiI). *The taxonomy every answer should start from.*
2. **Filter in practice** — watch [Drop features using correlation](https://www.youtube.com/watch?v=FndwYNcVe0U). *Removing redundant/constant features cheaply.*
3. **The right relevance metric** — watch [Mutual Information, Clearly Explained](https://www.youtube.com/watch?v=eJIp_mgVLwE). *Why MI catches nonlinear relevance that correlation misses.*
4. **Read the reference** — read [sklearn: Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html). *VarianceThreshold, SelectKBest, RFE, SelectFromModel, L1-based.*
5. **Trust importances carefully** — read [sklearn: Permutation importance](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html). *Why impurity importance misleads and permutation importance is safer.*

## 🎓 Courses (free)
- [Kaggle Learn — Feature Engineering (Mutual Information)](https://www.kaggle.com/learn/feature-engineering) — **Kaggle** — hands-on MI-based feature ranking on real data.
- [Google ML Crash Course — Numerical data](https://developers.google.com/machine-learning/crash-course/numerical-data) — **Google** — how regularization and feature relevance interact.

## 🎥 Videos
- [Filter vs Wrapper vs Embedded Methods](https://www.youtube.com/watch?v=Sc-TNxW3PiI) — **Learn with Whiteboard** — the clearest overview of the three selection families.
- [Feature Selection — Drop features using Pearson correlation](https://www.youtube.com/watch?v=FndwYNcVe0U) — **Krish Naik** — practical filter method removing redundant features.
- [Mutual Information, Clearly Explained](https://www.youtube.com/watch?v=eJIp_mgVLwE) — **StatQuest (Josh Starmer)** — the relevance metric behind SelectKBest(mutual_info).
- [Feature Selection — Variance Threshold (drop constant features)](https://www.youtube.com/watch?v=uMlU2JaiOd8) — **Krish Naik** — the cheapest first-pass filter, with code.

## 📄 Key Papers
- [An Introduction to Variable and Feature Selection](https://jmlr.org/papers/volume3/guyon03a/guyon03a.pdf) — **Guyon & Elisseeff (2003)** — the definitive survey of feature-selection methods; free in JMLR.
- [Conditional Likelihood Maximisation: A Unifying Framework for Information-Theoretic Feature Selection](https://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf) — **Brown et al. (2012)** — unifies mutual-information selection criteria; free in JMLR.

## 📰 Articles / Blogs (free, no paywall)
- [Feature selection — user guide](https://scikit-learn.org/stable/modules/feature_selection.html) — **scikit-learn** — the authoritative reference for every selector.
- [Permutation feature importance](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html) — **scikit-learn** — why impurity importance is biased and permutation importance is preferred.
- [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) — **scikit-learn** — selecting features on the whole dataset is leakage; do it inside CV.

## 📚 Books (free, with chapters)
- [Feature Engineering and Selection — **Ch. 10–11 "Feature Selection"**](http://www.feat.engineering/) — **Kuhn & Johnson** — filter/wrapper/embedded with rigorous resampling; free online.
- [An Introduction to Statistical Learning (Python) — **Ch. 6 "Linear Model Selection & Regularization"**](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — subset selection + Lasso; free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.05 PCA / SVD](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Next concepts: [06 Feature Engineering](../06-Feature-Engineering/06-Feature-Engineering.md) · [11 Data Leakage](../11-Data-Leakage/11-Data-Leakage.md)
- Related domain: [03. Supervised Learning](../../03.%20Supervised_Learning/README.md)
