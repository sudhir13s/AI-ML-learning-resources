---
id: "02-data-preprocessing/feature-scaling/references"
topic: "Feature Scaling & Normalization — References"
parent: "02-data-preprocessing/feature-scaling"
type: references
updated: 2026-07-03
---

# Feature Scaling & Normalization — references and further reading

> Companion link library for **[Feature Scaling & Normalization](feature-scaling-and-normalization.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Everything here is free / open, and every "Source / derivation" citation on the concept page appears below.

**Start here — suggested path**:
1. **Get the distinction** — watch [StatQuest: Normalization vs. Standardization, Clearly Explained](https://www.youtube.com/watch?v=1YpKUpitT98) (**StatQuest**). *The single most-asked framing — which one, when — cleared up in a few minutes.*
2. **Feel why it matters** — read [Importance of feature scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html) (**scikit-learn**). *Scaling materially changing PCA + a classifier's results, on the same Wine dataset this chapter uses.*
3. **See all the scalers at once** — read [Compare the effect of different scalers on data with outliers](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html) (**scikit-learn**). *Standard / min-max / robust / quantile side by side — the outlier behaviour made visual.*
4. **Learn it applied** — do [Google ML Crash Course — Numerical data: Normalization](https://developers.google.com/machine-learning/crash-course/numerical-data/normalization) (**Google**). *Z-score, min-max, log and clipping, with interactive widgets and when to use each.*
5. **Run it yourself** — open [this chapter's notebook](code/feature-scaling-and-normalization.ipynb). *Rebuild the three scalers, match them to scikit-learn, and measure the accuracy jump on real Wine.*

**Videos**:
- [StatQuest: Normalization vs. Standardization, Clearly Explained](https://www.youtube.com/watch?v=1YpKUpitT98) — **StatQuest (Josh Starmer)** — the canonical, intuition-first explanation of the two main scalers and when each applies.
- [Feature Scaling — why, and StandardScaler / MinMax / Robust](https://www.youtube.com/watch?v=nmBqnKSSKfM) — **Andrew Ng / DeepLearning.AI (Gradient Descent in Practice)** — why scaling makes gradient descent converge faster, from the Machine Learning Specialization, from first principles.
- [Feature Scaling — Standardization (Day 24)](https://www.youtube.com/watch?v=1Yw9sC0PNwY) — **CampusX** — z-score scaling in depth with code and the train-only fit rule.
- [Feature Scaling — Normalization: MinMax / MaxAbs / Robust](https://www.youtube.com/watch?v=eBrGyuA2MIg) — **CampusX** — all the non-standard scalers and when to use each, with sklearn.
- [Why & When Should We Perform Feature Normalization?](https://www.youtube.com/watch?v=s9e2A04lmXI) — **Krish Naik** — focuses on which algorithms actually need scaling (and which don't).
- [Data Preprocessing: Feature Scaling](https://www.youtube.com/watch?v=P3xPQyfMybg) — **DataMListic** — short, visual, standard-vs-normal comparison.

**Courses (free)**:
- [Google ML Crash Course — Numerical data (Normalization)](https://developers.google.com/machine-learning/crash-course/numerical-data/normalization) — **Google** — free, applied treatment of z-score, min-max, log, and clipping with widgets.
- [Machine Learning Specialization — Course 1 (free to audit)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng / DeepLearning.AI** — the "feature scaling to speed up gradient descent" lectures, taught from scratch.
- [Kaggle Learn — Feature Engineering](https://www.kaggle.com/learn/feature-engineering) — **Kaggle** — hands-on micro-course where scaling shows up inside real pipelines.

**Interactive & visual**:
- [Compare the effect of different scalers on data with outliers](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html) — **scikit-learn docs** — runnable side-by-side of Standard / MinMax / MaxAbs / Robust / Quantile scalers on a skewed real feature.
- [Importance of feature scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html) — **scikit-learn docs** — scaling changing PCA + a KNN classifier's results on the Wine dataset — the same experiment as this chapter.
- [Preprocessing data — user guide](https://scikit-learn.org/stable/modules/preprocessing.html) — **scikit-learn docs** — the authoritative reference for `StandardScaler` / `MinMaxScaler` / `RobustScaler` / `Normalizer`, with the exact formulas.

**Articles / blogs (free, no paywall)**:
- [About Feature Scaling and Normalization](https://sebastianraschka.com/Articles/2014_about_feature_scaling.html) — **Sebastian Raschka** — the clearest long-form essay: the math of z-score vs. min-max, worked examples, and which algorithms are affected.
- [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) & [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) — **scikit-learn docs** — the exact API, defaults (median/IQR for robust, population std for standard), and formulas used in this chapter.
- [Compose transformers and estimators with Pipeline](https://scikit-learn.org/stable/modules/compose.html) — **scikit-learn docs** — how to fit the scaler on training folds only, structurally preventing leakage.

**Key sources / derivations** (cited on the concept page):
- [An Introduction to Statistical Learning — Ch. 5 (Resampling) & Ch. 6 (Linear Model Selection & Regularization)](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — free PDF; the fit-preprocessing-on-training-data-only protocol, and why ridge/lasso require standardized predictors (Ch. 6.2).
- [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) — **LeCun, Bottou, Orr & Müller (1998)** — §4.3: why zero-mean, unit-variance (and decorrelated) inputs speed up gradient learning — the classic case for input normalization.
- [Deep Learning — Ch. 4.3 (Numerical Computation / conditioning) & Ch. 8 (Optimization, incl. input & batch normalization)](https://www.deeplearningbook.org/) — **Goodfellow, Bengio & Courville** — the modern account of why an ill-conditioned (unscaled) loss surface slows gradient descent, and how normalization fixes it.
- [Robust Statistics](https://onlinelibrary.wiley.com/doi/book/10.1002/0471725250) — **Peter J. Huber (1981)** — the foundational text on median/IQR-style robust estimators of location and scale — the theory under RobustScaler.
- [Robust Statistics: The Approach Based on Influence Functions](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118186435) — **Hampel, Ronchetti, Rousseeuw & Stahel (1986)** — the influence-function view of why the median and IQR resist outliers.
- [A Practical Guide to Support Vector Classification](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) — **Hsu, Chang & Lin (libsvm)** — opens by insisting on feature scaling for SVMs, with the failure mode spelled out.

**Books (free, with chapters)**:
- [Feature Engineering and Selection — Ch. 6 "Engineering Numeric Predictors"](http://www.feat.engineering/) — **Kuhn & Johnson** — centering, scaling, and transforms in the modeling workflow; free online.
- [Python Data Science Handbook — §5.4 "Feature Engineering"](https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html) — **Jake VanderPlas** — scaling in the broader preprocessing context, with runnable code.
- [Dive into Deep Learning — Ch. 5.4 "Numerical Stability and Initialization"](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) — **Zhang et al.** — why input/activation scale controls the conditioning of gradient-based learning.

**In this platform**:
- Concept page (full explanation): [Feature Scaling & Normalization](feature-scaling-and-normalization.md)
- The correctness rule, in full — why fitting preprocessing on all data leaks: [11 Data Leakage](../data-leakage/data-leakage.md)
- The split that scaling is fit inside: [10 Train/Validation/Test Splits](../train-validation-test-splits/train-validation-test-splits.md)
- Handle the outliers that make robust scaling necessary: [05 Outlier Detection & Treatment](../outlier-detection-and-treatment/outlier-detection-and-treatment.md)
- Where scaling lives in a real pipeline (`ColumnTransformer` + `Pipeline`): [13 Data Pipelines](../data-pipelines/data-pipelines.md)
- Why the condition number governs gradient-descent convergence: [13 Gradient Descent Theory](../../mathematical-foundations/gradient-descent-theory/gradient-descent-theory.md) · [04 How Models Learn](../../ai-ml-orientation/how-models-learn/how-models-learn.md)
- The scale-sensitive models this chapter measured: [03 Supervised Learning](../../../core-machine-learning/supervised-learning/README.md) (KNN, SVM) · [04 Unsupervised Learning](../../../core-machine-learning/unsupervised-learning/README.md) (k-means, PCA)
- Normalization taken inside the network (batch/layer norm): [05 Deep Learning](../../../deep-learning/README.md)
- Next preprocessing step — building new features: [06 Feature Engineering](../feature-engineering/feature-engineering.md)
