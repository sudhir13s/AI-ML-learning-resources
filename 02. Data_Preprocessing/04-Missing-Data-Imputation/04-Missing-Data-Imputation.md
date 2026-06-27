---
id: "02-data-preprocessing/missing-data"
topic: "Missing Data Imputation"
parent: "02-data-preprocessing"
level: beginner
prereqs: ["pandas", "mean-median-mode", "knn"]
interview_frequency: high
updated: 2026-06-20
---

# Missing Data Imputation
> Filling in (or flagging) missing values — mean/median/mode, KNN, or model-based MICE — while
> respecting *why* the data is missing (MCAR/MAR/MNAR) and never leaking statistics from test into train.

**Why it matters:** real datasets are full of holes, and how you fill them changes results. Interviewers
probe the missingness mechanisms (MCAR vs MAR vs MNAR), why median beats mean under skew, when a
**missing-indicator** column is worth adding, why KNN/MICE can beat simple imputation, and the cardinal
rule: **fit the imputer on training data only** (compute fill values from train, apply to test).

**⭐ Start here — suggested path:**

1. **The decisions** — watch [Complete Case Analysis](https://www.youtube.com/watch?v=aUnNWZorGmk). *When dropping rows is OK vs dangerous, and the MCAR assumption behind it.*
2. **Simple imputation** — watch [Simple Imputer for numerical data](https://www.youtube.com/watch?v=mCL2xLBDw8M). *Mean/median + the train-only fit rule.*
3. **Multivariate imputation** — watch [KNN Imputer](https://www.youtube.com/watch?v=-fK-xEev2I8). *Using similar rows to fill values.*
4. **Model-based** — watch [MICE / Iterative Imputer](https://www.youtube.com/watch?v=a38ehxv3kyk). *Chained-equations imputation, the gold standard for MAR data.*
5. **Reference the API** — read [sklearn: Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html). *SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator.*

## 🎓 Courses (free)
- [Kaggle Learn — Data Cleaning (Handling Missing Values)](https://www.kaggle.com/learn/data-cleaning) — **Kaggle** — short, hands-on lesson dedicated to missing data.
- [Google ML Crash Course — Numerical data](https://developers.google.com/machine-learning/crash-course/numerical-data) — **Google** — covers handling missing/anomalous numeric values.

## 🎥 Videos
- [Handling Missing Data Part 1 — Complete Case Analysis](https://www.youtube.com/watch?v=aUnNWZorGmk) — **CampusX** — when dropping rows is safe and when it biases results.
- [Handling Missing Data — Numerical, Simple Imputer](https://www.youtube.com/watch?v=mCL2xLBDw8M) — **CampusX** — mean/median imputation done leakage-safe.
- [KNN Imputer — Multivariate Imputation](https://www.youtube.com/watch?v=-fK-xEev2I8) — **CampusX** — filling values from nearest-neighbor rows.
- [MICE Algorithm — Iterative Imputer](https://www.youtube.com/watch?v=a38ehxv3kyk) — **CampusX** — chained-equations / model-based imputation explained.
- [Handling Missing Categorical Data — Most-Frequent & Missing-Category](https://www.youtube.com/watch?v=l_Wip8bEDFQ) — **CampusX** — mode imputation and the "Missing" category trick.

## 📄 Key Papers
- [mice: Multivariate Imputation by Chained Equations in R](https://www.jstatsoft.org/article/view/v045i03) — **van Buuren & Groothuis-Oudshoorn (2011)** — the standard reference for MICE; free in JSS.
- [Missing-data imputation using machine learning: A review](https://arxiv.org/abs/2106.04619) — **survey** — modern overview of imputation methods and when each wins; free on arXiv.

## 📰 Articles / Blogs (free, no paywall)
- [Imputation of missing values — user guide](https://scikit-learn.org/stable/modules/impute.html) — **scikit-learn** — authoritative reference for SimpleImputer/KNNImputer/IterativeImputer.
- [Flexible Imputation of Missing Data (online book)](https://stefvanbuuren.name/fimd/) — **Stef van Buuren** — the definitive open text on missing-data theory + MICE.
- [pandas — Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html) — **pandas docs** — detection (`isna`), dropping, and filling at the dataframe level.

## 📚 Books (free, with chapters)
- [Flexible Imputation of Missing Data — **Ch. 1–4 (mechanisms, simple & multiple imputation)**](https://stefvanbuuren.name/fimd/) — **Stef van Buuren** — MCAR/MAR/MNAR and MICE, free online.
- [Feature Engineering and Selection — **Ch. 8 "Handling Missing Data"**](http://www.feat.engineering/) — **Kuhn & Johnson** — applied imputation in a modeling workflow; free online.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 0.01 Probability & Bayes](../../../AI-ML-intuition/Module_0_Foundations/0.01_Probability_and_Bayes_Theorem.md) · [0.03 Expectation, Variance, Covariance](../../../AI-ML-intuition/Module_0_Foundations/0.03_Expectation_Variance_Covariance.md)
- Next concepts: [05 Outlier Detection & Treatment](../05-Outlier-Detection-and-Treatment/05-Outlier-Detection-and-Treatment.md) · [13 Data Pipelines](../13-Data-Pipelines/13-Data-Pipelines.md)
- Related domain: [03. Supervised Learning](../../03.%20Supervised_Learning/README.md)
