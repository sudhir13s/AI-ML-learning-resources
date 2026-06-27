---
id: "02-data-preprocessing/feature-scaling"
topic: "Feature Scaling & Normalization"
parent: "02-data-preprocessing"
level: beginner
prereqs: ["mean-variance", "distance-metrics"]
interview_frequency: very-high
updated: 2026-06-20
---

# Feature Scaling & Normalization
> Bringing features onto a comparable scale — standardization (mean 0, std 1), min-max to [0,1],
> or robust scaling — so distance- and gradient-based models aren't dominated by large-magnitude columns.

**Why it matters:** the classic "when and why do you scale?" question. You should be able to say which
models need it (KNN, SVM, k-means, PCA, neural nets, anything gradient-based) vs which don't (tree
ensembles), the difference between standardization and normalization, why you **fit the scaler on train
only**, and when RobustScaler beats StandardScaler (heavy outliers).

**⭐ Start here — suggested path:**

1. **Get the distinction** — watch [Normalization vs Standardization](https://www.youtube.com/watch?v=mnKm3YP56PY). *The single most-asked framing; clears it up fast.*
2. **Standardization in depth** — watch [Feature Scaling — Standardization](https://www.youtube.com/watch?v=1Yw9sC0PNwY). *Z-score scaling, why mean/std come from train only.*
3. **The other scalers** — watch [Normalization — MinMax/MaxAbs/Robust](https://www.youtube.com/watch?v=eBrGyuA2MIg). *When [0,1] vs robust-to-outliers scaling is the right call.*
4. **See it visually** — read [sklearn: Compare the effect of scalers](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html). *Side-by-side plots make the trade-offs obvious.*
5. **Know when it matters** — read [sklearn: Importance of feature scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html). *Shows scaling changing PCA/model results concretely.*

## 🎓 Courses (free)
- [Google ML Crash Course — Numerical data](https://developers.google.com/machine-learning/crash-course/numerical-data) — **Google** — free, applied treatment of normalization, scaling, and bucketing.
- [Kaggle Learn — Feature Engineering](https://www.kaggle.com/learn/feature-engineering) — **Kaggle** — hands-on micro-course where scaling shows up in real pipelines.

## 🎥 Videos
- [Standardization vs Normalization — Feature Scaling](https://www.youtube.com/watch?v=mnKm3YP56PY) — **Krish Naik** — the clearest "which one, when" explanation.
- [Feature Scaling — Standardization (Day 24)](https://www.youtube.com/watch?v=1Yw9sC0PNwY) — **CampusX** — z-score scaling with code and the train-only fit rule.
- [Feature Scaling — Normalization: MinMax/MaxAbs/Robust](https://www.youtube.com/watch?v=eBrGyuA2MIg) — **CampusX** — all the non-standard scalers and when to use each.
- [Normalization vs Standardization — Explained](https://www.youtube.com/watch?v=87C5hkTY8RI) — **DataMListic** — short, visual, intuition-first comparison.
- [Why & When Should We Perform Feature Normalization?](https://www.youtube.com/watch?v=s9e2A04lmXI) — **Krish Naik** — focuses on which algorithms actually need scaling.

## 📄 Key Papers
- [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) — **LeCun et al. (1998)** — §4.3 explains why input normalization (zero-mean, decorrelated) speeds up gradient learning.

## 📰 Articles / Blogs (free, no paywall)
- [Compare the effect of different scalers](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html) — **scikit-learn** — visual side-by-side of Standard/MinMax/Robust/Quantile scalers.
- [Importance of feature scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html) — **scikit-learn** — shows scaling materially changing PCA + classifier results.
- [Preprocessing data — user guide](https://scikit-learn.org/stable/modules/preprocessing.html) — **scikit-learn** — the authoritative reference for StandardScaler/MinMaxScaler/RobustScaler.

## 📚 Books (free, with chapters)
- [Feature Engineering and Selection — **Ch. 6 "Engineering Numeric Predictors"**](http://www.feat.engineering/) — **Kuhn & Johnson** — centering, scaling, and transforms; free online (open the book and jump to Ch. 6).
- [Python Data Science Handbook — **§5 "Feature Engineering"**](https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html) — **Jake VanderPlas** — scaling in the broader preprocessing context.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 0.03 Expectation, Variance, Covariance](../../../AI-ML-intuition/Module_0_Foundations/0.03_Expectation_Variance_Covariance.md) · [1.07–1.08 Distances: Euclidean vs Cosine](../../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md)
- Next concepts: [13 Data Pipelines](../13-Data-Pipelines/13-Data-Pipelines.md) · [06 Feature Engineering](../06-Feature-Engineering/06-Feature-Engineering.md)
- Related domain: [01. Foundations](../../01.%20Foundations/README.md)
