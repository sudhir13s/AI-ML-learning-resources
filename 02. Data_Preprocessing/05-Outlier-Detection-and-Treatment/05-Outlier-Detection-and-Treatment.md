---
id: "02-data-preprocessing/outliers"
topic: "Outlier Detection & Treatment"
parent: "02-data-preprocessing"
level: beginner
prereqs: ["mean-std", "quartiles-iqr", "gaussian"]
interview_frequency: high
updated: 2026-06-20
---

# Outlier Detection & Treatment
> Finding the points that don't fit — via Z-score, IQR, or percentiles — then deciding whether to keep,
> cap (winsorize), transform, or drop them, depending on whether they're errors or genuine extremes.

**Why it matters:** outliers wreck mean-based statistics, distort scaling, and skew linear models — but
blindly deleting them throws away signal (fraud, anomalies, the very thing you may want to detect). Expect
to compare Z-score vs IQR vs percentile rules, explain why median/IQR are robust while mean/std aren't, and
justify capping vs removal vs a log transform — and to do detection **using training-set statistics only**.

**⭐ Start here — suggested path:**

1. **What is an outlier** — watch [What are Outliers](https://www.youtube.com/watch?v=Lln1PKgGr_M). *Defines outliers and why they matter before any method.*
2. **Z-score method** — watch [Outlier removal using Z-score](https://www.youtube.com/watch?v=OnPE-Z8jtqM). *The ±3σ rule and its Gaussian assumption.*
3. **IQR method** — watch [Outlier removal using IQR](https://www.youtube.com/watch?v=Ccv1-W5ilak). *The robust 1.5×IQR fences that don't assume normality.*
4. **Cap instead of drop** — watch [Percentile method & Winsorization](https://www.youtube.com/watch?v=bcXA4CqRXvM). *When to clip extremes rather than delete rows.*
5. **Reference detectors** — read [sklearn: Novelty & outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html). *Isolation Forest, LOF, Elliptic Envelope for multivariate outliers.*

## 🎓 Courses (free)
- [Kaggle Learn — Data Cleaning](https://www.kaggle.com/learn/data-cleaning) — **Kaggle** — practical lessons touching outliers and inconsistent values.
- [Google ML Crash Course — Numerical data](https://developers.google.com/machine-learning/crash-course/numerical-data) — **Google** — handling outliers via clipping, log scaling, and bucketing.

## 🎥 Videos
- [What are Outliers in Machine Learning](https://www.youtube.com/watch?v=Lln1PKgGr_M) — **CampusX** — definition, causes, and why outliers distort models.
- [Outlier Detection & Removal — Z-score Method](https://www.youtube.com/watch?v=OnPE-Z8jtqM) — **CampusX** — ±3σ rule with code and its normality assumption.
- [Outlier Detection & Removal — IQR Method](https://www.youtube.com/watch?v=Ccv1-W5ilak) — **CampusX** — robust 1.5×IQR fences, no Gaussian assumption.
- [Outlier Detection — Percentile Method & Winsorization](https://www.youtube.com/watch?v=bcXA4CqRXvM) — **CampusX** — capping extremes instead of deleting rows.
- [Outlier detection & removal using IQR (with code)](https://www.youtube.com/watch?v=A3gClkblXK8) — **codebasics** — clean pandas walkthrough on a real dataset.

## 📄 Key Papers
- [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) — **Liu, Ting & Zhou (2008)** — isolates anomalies via random partitioning; the standard scalable outlier detector.
- [Deep Learning for Anomaly Detection: A Review](https://arxiv.org/abs/2007.02500) — **Pang et al. (2020)** — survey of modern outlier/anomaly methods; free on arXiv.

## 📰 Articles / Blogs (free, no paywall)
- [Novelty and Outlier Detection — user guide](https://scikit-learn.org/stable/modules/outlier_detection.html) — **scikit-learn** — Isolation Forest, Local Outlier Factor, and Elliptic Envelope explained.
- [NIST/SEMATECH e-Handbook — Detection of Outliers](https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm) — **NIST** — Grubbs', Tietjen-Moore, and boxplot rules, free and rigorous.
- [Forecasting: Principles and Practice — Missing values & outliers](https://otexts.com/fpp3/missing-outliers.html) — **Hyndman & Athanasopoulos** — outlier handling in a time-series context.

## 📚 Books (free, with chapters)
- [Feature Engineering and Selection — **Ch. 6 "Engineering Numeric Predictors" (transformations)**](http://www.feat.engineering/) — **Kuhn & Johnson** — transforms that tame skew and extremes; free online.
- [Python Data Science Handbook — **Ch. 3 (Pandas: detecting/filtering)**](https://jakevdp.github.io/PythonDataScienceHandbook/) — **Jake VanderPlas** — boolean masking and aggregation to find/remove outliers.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 0.02 Distributions & the Gaussian](../../../AI-ML-intuition/Module_0_Foundations/0.02_Distributions_and_the_Gaussian.md) · [1.10 Mahalanobis Distance](../../../AI-ML-intuition/Module_1_Representation/1.10_Mahalanobis_Distance.md)
- Next concepts: [02 Feature Scaling & Normalization](../02-Feature-Scaling-and-Normalization/02-Feature-Scaling-and-Normalization.md) · [06 Feature Engineering](../06-Feature-Engineering/06-Feature-Engineering.md)
- Related domain: [04. Unsupervised Learning](../../04.%20Unsupervised_Learning/README.md)
