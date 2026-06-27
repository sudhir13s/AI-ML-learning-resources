---
id: "02-data-preprocessing/data-pipelines"
topic: "Data Pipelines (sklearn Pipeline · ColumnTransformer)"
parent: "02-data-preprocessing"
level: intermediate
prereqs: ["scaling", "encoding", "imputation", "cross-validation"]
interview_frequency: high
updated: 2026-06-20
---

# Data Pipelines (sklearn Pipeline · ColumnTransformer)
> Chaining every preprocessing step + the model into one fit/transform object — so the exact same
> transforms (fit on train) apply to validation, test, and production, and nothing leaks across the split.

**Why it matters:** pipelines are how you make preprocessing **correct and reproducible**. Fitting a scaler
or imputer on the whole dataset is the #1 cause of leakage; a `Pipeline` + `ColumnTransformer` fits all
transforms on training folds only, applies them consistently everywhere, and serializes cleanly for
deployment. Interviewers love this because it ties together scaling, encoding, imputation, CV, and leakage
into one production-grade pattern.

**⭐ Start here — suggested path:**

1. **Why pipelines** — watch [Pipelines A-Z](https://www.youtube.com/watch?v=xOccYkgRV4Q). *What problem they solve and how fit/predict flow.*
2. **Different columns, different transforms** — watch [ColumnTransformer](https://www.youtube.com/watch?v=5TVj6iEBR4I). *Numeric → scale, categorical → encode, in one object.*
3. **Build one end to end** — watch [ML Pipeline step by step](https://www.youtube.com/watch?v=T9ETsSD1I0w). *Preprocessing + model + cross-validation together.*
4. **Read the reference** — read [sklearn: Pipelines & composite estimators](https://scikit-learn.org/stable/modules/compose.html). *Pipeline, ColumnTransformer, FeatureUnion, get_feature_names_out.*
5. **Lock in leakage-safety** — read [sklearn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html). *Why pipelines + CV are the structural fix for leakage.*

## 🎓 Courses (free)
- [Kaggle Learn — Intermediate ML (Pipelines)](https://www.kaggle.com/code/alexisbcook/pipelines) — **Kaggle** — the best free hands-on pipelines lesson.
- [Kaggle Learn — Feature Engineering](https://www.kaggle.com/learn/feature-engineering) — **Kaggle** — where pipelines tie preprocessing + features together.

## 🎥 Videos
- [Machine Learning Pipelines A-Z (Day 29)](https://www.youtube.com/watch?v=xOccYkgRV4Q) — **CampusX** — the clearest end-to-end pipeline walkthrough.
- [Column Transformer in Sklearn](https://www.youtube.com/watch?v=5TVj6iEBR4I) — **CampusX** — applying different transforms to numeric vs categorical columns.
- [Building an ML Pipeline — Step by Step](https://www.youtube.com/watch?v=T9ETsSD1I0w) — **Ryan & Matt Data Science** — preprocessing + model + tuning in one object.
- [Understanding Pipeline in scikit-learn](https://www.youtube.com/watch?v=jzKSAeJpC6s) — **Dr. Data Science** — fit/transform semantics and why pipelines prevent leakage.

## 📄 Key Papers
- [Scikit-learn: Machine Learning in Python](https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf) — **Pedregosa et al. (2011)** — the JMLR paper introducing the fit/transform/Pipeline API design; free.
- [API design for machine learning software: experiences from the scikit-learn project](https://arxiv.org/abs/1309.0238) — **Buitinck et al. (2013)** — the design philosophy behind estimators, transformers, and pipelines; free on arXiv.

## 📰 Articles / Blogs (free, no paywall)
- [Pipelines and composite estimators — user guide](https://scikit-learn.org/stable/modules/compose.html) — **scikit-learn** — the authoritative reference for Pipeline + ColumnTransformer.
- [ColumnTransformer for heterogeneous data](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html) — **scikit-learn** — a complete mixed numeric/categorical example.
- [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) — **scikit-learn** — why pipelines + CV are the leakage-safe pattern.

## 📚 Books (free, with chapters)
- [Python Data Science Handbook — **§5.4 "Feature Engineering" (Pipelines)**](https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html) — **Jake VanderPlas** — building processing pipelines with sklearn.
- [Tidy Modeling with R — **Ch. 8 "Feature Engineering with recipes"**](https://www.tmwr.org/) — **Kuhn & Silge** — the recipes/workflow pattern (the R analogue of Pipeline); concepts transfer; free.

## 🔗 In this platform
- Builds on: [02 Feature Scaling](../02-Feature-Scaling-and-Normalization/02-Feature-Scaling-and-Normalization.md) · [03 Encoding Categoricals](../03-Encoding-Categorical-Variables/03-Encoding-Categorical-Variables.md) · [04 Missing Data Imputation](../04-Missing-Data-Imputation/04-Missing-Data-Imputation.md)
- Prevents: [11 Data Leakage](../11-Data-Leakage/11-Data-Leakage.md) — pipelines are the structural fix.
- Related domain: [12. Deployment & MLOps](../../14.%20Deployment_and_MLOps/README.md) — where pipelines become serving-time feature transforms.
