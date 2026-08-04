---
id: "02-data-preprocessing"
topic: "Data Handling & Feature Engineering"
level: beginner
built_from: ["python", "statistics"]
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
1. ✅ [Exploratory Data Analysis (EDA)](exploratory-data-analysis/exploratory-data-analysis.md)

### Cleaning & transforming features
2. ✅ [Feature Scaling & Normalization (Standard · MinMax · Robust)](feature-scaling-and-normalization/feature-scaling-and-normalization.md)
3. ✅ [Encoding Categorical Variables (one-hot · ordinal · target)](encoding-categorical-variables/encoding-categorical-variables.md)
4. ✅ [Missing Data Imputation (mean/median · KNN · MICE)](missing-data-imputation/missing-data-imputation.md)
5. ✅ [Outlier Detection & Treatment (Z-score · IQR · winsorize)](outlier-detection-and-treatment/outlier-detection-and-treatment.md)

### Engineering & selecting features
6. ✅ [Feature Engineering (construction · transforms · binning)](feature-engineering/feature-engineering.md)
7. ✅ [Feature Selection (filter · wrapper · embedded)](feature-selection/feature-selection.md)
8. ✅ [Handling Date/Time & Cyclical Features](date-time-and-cyclical-features/date-time-and-cyclical-features.md)
9. ✅ [Text & Image Preprocessing (overview)](text-and-image-preprocessing-overview/text-and-image-preprocessing-overview.md)

### Splitting, leakage & class balance
10. ✅ [Train / Validation / Test Splits & Cross-Validation](train-validation-test-splits/train-validation-test-splits.md)
11. ✅ [Data Leakage (train/test contamination · target leakage)](data-leakage/data-leakage.md)
12. ✅ [Imbalanced Data (resampling · SMOTE · class weights)](imbalanced-data/imbalanced-data.md)

### Putting it together
13. ✅ [Data Pipelines (sklearn Pipeline · ColumnTransformer)](data-pipelines/data-pipelines.md)

### Related concepts (covered in another section)
> These topics are used across many areas, so they're kept in one place to avoid repetition.
- **PCA / SVD math & dimensionality reduction** → [01. Foundations](../mathematical-foundations/README.md)
- **Clustering · t-SNE · UMAP** → [04. Unsupervised Learning](../../core-machine-learning/unsupervised-learning/README.md)
- **Tokenization · text normalization · subword algorithms** → [06. NLP](../../modalities-and-generative-models/natural-language-processing/README.md)
- **Image augmentation & vision-specific preprocessing** → [07. Computer Vision](../../modalities-and-generative-models/computer-vision/README.md)
- **Feature stores & serving-time feature pipelines** → [14. Deployment & MLOps](../../deployment-and-mlops/README.md)
- **Bias–variance & generalization** → [03. Supervised Learning](../../core-machine-learning/supervised-learning/README.md)

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
- Why scaling/encoding matters mathematically: [AI-ML-intuition Module 1](../../../AI-ML-intuition/representation/)
