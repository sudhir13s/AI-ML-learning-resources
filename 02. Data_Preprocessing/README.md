---
id: "02-data-preprocessing"
topic: "Data Handling & Feature Engineering"
level: beginner
prereqs: ["python", "statistics"]
updated: 2026-06-27
---

# Data Handling, Preprocessing & Feature Engineering
> 80% of real ML work. Cleaning, scaling, encoding, splitting, and engineering features that
> make models work — before any model is trained.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) — pick a concept to open its
page: a short guided learning path plus the best **free, open** courses, videos, papers, articles,
and books for that topic.
> **✅ ready.** New here? Start with the field overview below, then work top to bottom.

### Understand the data first
1. ✅ [Exploratory Data Analysis (EDA)](01-Exploratory-Data-Analysis/01-Exploratory-Data-Analysis.md)

### Cleaning & transforming features
2. ✅ [Feature Scaling & Normalization (Standard · MinMax · Robust)](02-Feature-Scaling-and-Normalization/02-Feature-Scaling-and-Normalization.md)
3. ✅ [Encoding Categorical Variables (one-hot · ordinal · target)](03-Encoding-Categorical-Variables/03-Encoding-Categorical-Variables.md)
4. ✅ [Missing Data Imputation (mean/median · KNN · MICE)](04-Missing-Data-Imputation/04-Missing-Data-Imputation.md)
5. ✅ [Outlier Detection & Treatment (Z-score · IQR · winsorize)](05-Outlier-Detection-and-Treatment/05-Outlier-Detection-and-Treatment.md)

### Engineering & selecting features
6. ✅ [Feature Engineering (construction · transforms · binning)](06-Feature-Engineering/06-Feature-Engineering.md)
7. ✅ [Feature Selection (filter · wrapper · embedded)](07-Feature-Selection/07-Feature-Selection.md)
8. ✅ [Handling Date/Time & Cyclical Features](08-Date-Time-and-Cyclical-Features/08-Date-Time-and-Cyclical-Features.md)
9. ✅ [Text & Image Preprocessing (overview)](09-Text-and-Image-Preprocessing-Overview/09-Text-and-Image-Preprocessing-Overview.md)

### Splitting, leakage & class balance
10. ✅ [Train / Validation / Test Splits & Cross-Validation](10-Train-Validation-Test-Splits/10-Train-Validation-Test-Splits.md)
11. ✅ [Data Leakage (train/test contamination · target leakage)](11-Data-Leakage/11-Data-Leakage.md)
12. ✅ [Imbalanced Data (resampling · SMOTE · class weights)](12-Imbalanced-Data/12-Imbalanced-Data.md)

### Putting it together
13. ✅ [Data Pipelines (sklearn Pipeline · ColumnTransformer)](13-Data-Pipelines/13-Data-Pipelines.md)

### Related concepts (covered in another section)
> These topics are used across many areas, so they're kept in one place to avoid repetition.
- **PCA / SVD math & dimensionality reduction** → [01. Foundations](../01.%20Foundations/README.md)
- **Clustering · t-SNE · UMAP** → [04. Unsupervised Learning](../04.%20Unsupervised_Learning/README.md)
- **Tokenization · text normalization · subword algorithms** → [06. NLP](../06.%20NLP/README.md)
- **Image augmentation & vision-specific preprocessing** → [07. Computer Vision](../07.%20Computer%20Vision/README.md)
- **Feature stores & serving-time feature pipelines** → [14. Deployment & MLOps](../14.%20Deployment_and_MLOps/README.md)
- **Bias–variance & generalization** → [03. Supervised Learning](../03.%20Supervised_Learning/README.md)

## 🎓 Courses (free)
- [Kaggle Learn: Data Cleaning + Feature Engineering](https://www.kaggle.com/learn) — **Kaggle** — short, hands-on, free micro-courses with real datasets.
- [Data Analysis with Python](https://www.freecodecamp.org/learn/data-analysis-with-python/) — **freeCodeCamp** — Pandas/NumPy end to end.

## 🎥 Videos
- [Pandas tutorials](https://www.youtube.com/playlist?list=PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS) — **Corey Schafer** — the clearest Pandas walkthroughs.
- [Feature Engineering](https://www.youtube.com/watch?v=6WDFfaYtN6s) — **StatQuest / Krish Naik** — encoding, scaling, leakage explained.

## 📰 Articles
- [scikit-learn: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — **scikit-learn docs** — the authoritative reference + recipes.
- [Data leakage, explained](https://machinelearningmastery.com/data-leakage-machine-learning/) — **Machine Learning Mastery** — the #1 silent bug in applied ML.

## 📚 Books (free)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) — **Jake VanderPlas** — free; NumPy/Pandas/sklearn bible.
- [Feature Engineering and Selection](http://www.feat.engineering/) — **Kuhn & Johnson** — free online.

## 🔗 In this platform
- Why scaling/encoding matters mathematically: [AI-ML-intuition Module 1](../../AI-ML-intuition/Module_1_Representation/)
